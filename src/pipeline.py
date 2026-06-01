"""Пайплайн: YouTube URL → TimeBasedReport + реальные кадры/клипы.

Режимы:
  offline=True  — полностью фейковый (без сети, для тестов UI)
  offline=False, dummy=True  — реальное скачивание + реальные кадры, мок-анализ
  offline=False, dummy=False — полный боевой пайплайн
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import random
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable, Optional

import sys

import imageio_ffmpeg
from PIL import Image, ImageDraw

from .pz1_keyframes import extract_keyframes
from .pz3_ocr import ocr_keyframes
from .pz4_audio import transcribe_and_classify as _run_pz4
from .pz5_yolo import yolo_keyframes
from .pz6_clip import clip_keyframes
from .pz7_nsfw import nsfw_keyframes
from .pz8_action import xclip_action
from .presets import get_preset, DEFAULT_PRESET
from .logs import get_logger, stage
from .gpu_lock import GpuSlot

from .schemas import (
    Detection,
    DetectionType,
    SourceInfo,
    Subclass,
    TimeBasedReport,
    TopSourceInfo,
)

ProgressCb = Callable[[float, str], Awaitable[None]]

log = get_logger(__name__)

_warm_lock = threading.Lock()
_warmed = False


def _warm_imports(use_ocr: bool = True) -> None:
    """Прогрев тяжёлых ленивых импортов ОДИН раз, однопоточно.

    Каналы clip/nsfw/action/ocr иначе одновременно впервые тянут одни и те же
    тяжёлые подмодули (`transformers.AutoModel`, `torch._dynamo`, `easyocr`) из
    параллельных потоков. Ни ленивый загрузчик transformers, ни питоновский
    import-lock на это не рассчитаны → случайный поток ловит
    `cannot import name 'AutoModel'` или `deadlock detected ... torch._dynamo`.
    Резолвим их заранее в один поток — дальше потоки берут из кэша модулей.
    """
    global _warmed
    with _warm_lock:
        if _warmed:
            return
        try:
            import torch  # noqa: F401
            import torch._dynamo  # noqa: F401  (главный источник дедлока)
            from transformers import (  # noqa: F401
                AutoModel, AutoProcessor,
                AutoModelForImageClassification, AutoImageProcessor,
                XCLIPModel, XCLIPProcessor, CLIPModel, CLIPProcessor,
            )
            if use_ocr:
                import easyocr  # noqa: F401  (тоже тянет torch-машинерию)
        except Exception as exc:
            log.warning("Прогрев импортов не удался: %s", exc)
        _warmed = True

_SUBCLASS_COLORS = {
    "alcohol":   (245, 158, 11),
    "drugs":     (239, 68, 68),
    "smoking":   (107, 114, 128),
    "extremism": (220, 38, 38),
    "ludomania": (139, 92, 246),
    "lgbt":      (236, 72, 153),
    "violence":  (185, 28, 28),
    "vandalism": (217, 119, 6),
    "nsfw":          (190, 24, 93),
    "selfharm":      (127, 29, 29),
    "animal_cruelty": (146, 64, 14),
}


@dataclass
class DetectionMedia:
    detection: Detection
    frame: Optional[Image.Image]
    clip_path: Optional[Path]


def _fmt_hms(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _fmt_clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


async def _noop(progress: float, message: str) -> None:
    return None


def _placeholder_frame(subclass: str, time_interval: str, confidence: float) -> Image.Image:
    """Цветной placeholder для offline-режима."""
    w, h = 640, 360
    color = _SUBCLASS_COLORS.get(subclass, (100, 100, 100))
    bg = tuple(max(0, c - 60) for c in color)
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    for i in range(40):
        a = 1 - i / 40
        c = tuple(int(color[j] * a + bg[j] * (1 - a)) for j in range(3))
        draw.rectangle([0, i, w, i + 1], fill=c)
    label_map = {
        "alcohol": "ALCOHOL", "drugs": "DRUGS", "smoking": "SMOKING",
        "extremism": "EXTREMISM", "ludomania": "GAMBLING",
        "lgbt": "LGBT", "violence": "VIOLENCE", "vandalism": "VANDALISM",
        "nsfw": "NSFW", "selfharm": "SELF-HARM", "animal_cruelty": "ANIMAL CRUELTY",
    }
    draw.text((w // 2, h // 2 - 40), label_map.get(subclass, subclass.upper()),
              fill=(255, 255, 255), anchor="mm")
    draw.text((w // 2, h // 2 + 10), time_interval, fill=(220, 220, 220), anchor="mm")
    draw.text((w // 2, h // 2 + 40), f"conf: {confidence:.2f}", fill=(180, 180, 180), anchor="mm")
    draw.rectangle([0, 0, w - 1, h - 1], outline=color, width=4)
    return img


def _annotate_frame(img: Image.Image, subclass: str, time_interval: str, confidence: float) -> Image.Image:
    """Добавляет цветную подпись поверх реального кадра."""
    draw = ImageDraw.Draw(img)
    color = _SUBCLASS_COLORS.get(subclass, (255, 255, 255))
    draw.rectangle([0, 0, img.width, 34], fill=(0, 0, 0))
    draw.rectangle([0, 0, 6, 34], fill=color)
    draw.text((14, 10), f"{subclass.upper()}  {time_interval}  conf: {confidence:.2f}",
              fill=color)
    return img


class VideoAnalyzer:
    """
    offline=True  → цветные placeholder-кадры, скачивания нет
    offline=False → всегда скачивает и извлекает реальные кадры;
                    dummy=True → мок-детекции, dummy=False → реальный анализ
    """

    def __init__(
        self,
        dummy: bool = True,
        offline: bool = False,
        preset: str = DEFAULT_PRESET,
        whisper_model: Optional[str] = None,
        use_ocr: bool = True,
        use_clip: bool = True,
        use_nsfw: bool = True,
        use_action: bool = True,
        use_llm: bool = True,
        use_llm_vision: bool = False,
        stage_timeout: float = 600.0,
        clips_dir: Optional[Path] = None,
    ):
        self.dummy = dummy
        self.offline = offline
        self.preset = get_preset(preset)
        # явный whisper_model переопределяет пресет (обратная совместимость)
        self.whisper_model = whisper_model
        self.use_ocr = use_ocr
        self.use_clip = use_clip
        self.use_nsfw = use_nsfw
        self.use_action = use_action
        self.use_llm = use_llm
        self.use_llm_vision = use_llm_vision
        self.stage_timeout = stage_timeout
        self.clips_dir = clips_dir or Path("outputs/clips")

        # Пул потоков: каждый ML-этап получает свой поток
        # CPU-bound библиотеки (PyTorch, OpenCV, EasyOCR) снимают GIL → реальный параллелизм
        n = max(4, (os.cpu_count() or 2) + 2)
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=n, thread_name_prefix="cv_stage"
        )

    async def run(
        self,
        url: str,
        quality: str = "720p",
        on_progress: Optional[ProgressCb] = None,
    ) -> tuple[TimeBasedReport, list[DetectionMedia]]:
        progress = on_progress or _noop
        t0 = time.time()
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        log.info("═" * 60)
        log.info("Анализ: %s | пресет=%s | каналы: clip=%s ocr=%s nsfw=%s action=%s llm=%s",
                 url, self.preset.name, self.use_clip, self.use_ocr,
                 self.use_nsfw, self.use_action, self.use_llm)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # ── Download ──────────────────────────────────────────────────
            await progress(0.05, "Скачиваю видео с YouTube...")
            with stage(log, "download", "скачивание/чтение видео"):
                video_path, duration_sec, fps, frame_count = await self._download(
                    url, tmp_path, quality
                )
            log.info("Видео: %.0fс, %.1f fps, %d кадров", duration_sec, fps, frame_count)

            # ── Analysis (параллельно) ────────────────────────────────────
            # Межпроцессный слот: на 8-ГБ GPU одновременно идёт ТОЛЬКО ОДИН
            # анализ. Второй (из UI/API/benchmark) ждёт в очереди, а не лезет
            # в ту же VRAM и не валит оба по таймауту. Скачивание держим вне лока.
            await progress(0.20, "Извлекаю ключевые кадры...")
            loop = asyncio.get_event_loop()
            slot = None
            if not self.dummy and not self.offline:
                slot = GpuSlot(on_wait=lambda: log.info(
                    "⏳ Очередь: ждём освобождения GPU (идёт другой анализ)..."))
                await progress(0.18, "Очередь на GPU (если занят другим анализом)...")
                await loop.run_in_executor(None, slot.__enter__)
            try:
                log.info("▶ Параллельный анализ: аудио-ветка + видео-ветка")
                audio_task = asyncio.create_task(
                    self._analyze_audio(video_path, fps, duration_sec, progress)
                )
                video_task = asyncio.create_task(
                    self._analyze_video(video_path, fps, duration_sec, progress)
                )
                audio_dets, video_dets = await asyncio.gather(audio_task, video_task)
            finally:
                if slot is not None:
                    await loop.run_in_executor(None, slot.__exit__, None, None, None)
            log.info("Сырых детекций: аудио=%d, видео=%d", len(audio_dets), len(video_dets))

            # ── Post-processing ───────────────────────────────────────────
            await progress(0.85, "Свожу результаты воедино...")
            with stage(log, "fusion", "фьюзия + temporal NMS"):
                detections = self._postprocess(audio_dets + video_dets, fps=fps)
            log.info("После фьюзии: %d событий", len(detections))

            # ── Извлечение медиа ──────────────────────────────────────────
            await progress(0.92, "Извлекаю кадры и клипы...")
            with stage(log, "media", "кадры + нарезка клипов"):
                media = await self._extract_media(video_path, detections, fps, duration_sec)

            # ── LLM vision-проверка пограничных видео-детекций (опц.) ──────
            if self.use_llm and self.use_llm_vision and not self.offline:
                await progress(0.95, "LLM-проверка кадров...")
                with stage(log, "llm-vision", "проверка пограничных кадров"):
                    media = await self._llm_verify(media)
                detections = [m.detection for m in media]

            # ── LLM-резюме отчёта (опц.) ───────────────────────────────────
            llm_summary = None
            if self.use_llm and not self.offline:
                await progress(0.98, "LLM-резюме...")
                with stage(log, "llm-summary", "резюме на русском"):
                    llm_summary = await self._llm_summary(detections)

            await progress(1.0, "Готово.")

        # Освобождаем VRAM после прогона (важно на 8 ГБ — модели кэшируются,
        # но промежуточные тензоры/фрагментацию чистим между анализами).
        from .device import empty_cache
        empty_cache()

        processing_time = time.time() - t0
        now = datetime.now().isoformat()

        # Источник для идентификации отчёта: YouTube-ссылка как есть, либо
        # абсолютный путь к локальному файлу + папка, где он лежит.
        source = url
        source_dir = None
        local = url[7:] if url.startswith("file://") else url
        try:
            p = Path(local)
            if p.exists() and p.is_file():
                source = str(p.resolve())
                source_dir = str(p.resolve().parent)
        except (OSError, ValueError):
            pass

        report = TimeBasedReport(
            source=source,
            source_dir=source_dir,
            preset=self.preset.name,
            llm_summary=llm_summary,
            source_info=TopSourceInfo(
                frameCount=frame_count,
                fps=fps,
                video_duration_formatted=_fmt_hms(duration_sec),
                analysis_timestamp=now,
            ),
            detections=detections,
            sourceInfo=SourceInfo(
                frameCount=0, fps=0, video_path="",
                video_duration_seconds=round(duration_sec, 2),
                processing_time_seconds=round(processing_time, 2),
                processing_efficiency=round(processing_time / max(duration_sec, 1e-6), 2),
                video_duration_formatted=_fmt_hms(duration_sec),
                processing_time_formatted=_fmt_hms(processing_time),
                analysis_timestamp=now,
            ),
        )

        log.info("ИТОГ: %d событий | обработка %.1fс | efficiency %.2f",
                 len(detections), processing_time,
                 processing_time / max(duration_sec, 1e-6))
        self._save_report(report, url)
        return report, media

    def _save_report(self, report: TimeBasedReport, url: str) -> None:
        """Сохраняет каждый выходной JSON-отчёт в outputs/reports/ (для истории)."""
        try:
            reports_dir = Path("outputs") / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # короткий идентификатор источника (yt id или имя файла)
            tag = "".join(c for c in Path(url).name[-20:] if c.isalnum()) or "video"
            path = reports_dir / f"{ts}_{report.preset}_{tag}.json"
            path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            log.info("Отчёт сохранён: %s", path)
        except Exception as exc:
            log.warning("Не удалось сохранить отчёт: %s", exc)

    # ── Download ──────────────────────────────────────────────────────────────

    @staticmethod
    def _probe(path: Path):
        """Читает fps/кадры/длительность локального видеофайла через OpenCV."""
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0.0
        cap.release()
        return duration, fps, frame_count

    async def _download(self, url: str, tmp: Path, quality: str):
        if self.offline:
            await asyncio.sleep(0.3)
            return tmp / "video.mp4", 180.0, 30.0, 5400

        # Локальный файл (загружен через UI или передан путём) — без yt-dlp
        local = url[7:] if url.startswith("file://") else url
        if local and Path(local).exists() and Path(local).is_file():
            p = Path(local)
            duration, fps, frame_count = self._probe(p)
            return p, duration, fps, frame_count

        out = tmp / "video.mp4"
        h = quality.rstrip("p")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        fmt = f"bestvideo[height<={h}]+bestaudio/best[height<={h}]/best"
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "yt_dlp",
            "--ffmpeg-location", ffmpeg_exe,
            "-f", fmt, "--merge-output-format", "mp4", "-o", str(out), url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"yt-dlp: {stderr.decode(errors='ignore')[-500:]}")

        import cv2
        cap = cv2.VideoCapture(str(out))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0.0
        cap.release()
        return out, duration, fps, frame_count

    # ── Video analysis ────────────────────────────────────────────────────────

    async def _analyze_video(self, video_path: Path, fps: float, duration: float, progress: ProgressCb):
        if self.offline:
            await progress(0.45, "Видео (мок)...")
            await asyncio.sleep(0.5)
            return self._mock_video_detections(fps, duration)

        loop = asyncio.get_event_loop()

        # keyframes (должны быть первыми, остальные каналы зависят от них)
        preset = self.preset
        with stage(log, "keyframes", "извлечение ключевых кадров"):
            keyframes = await loop.run_in_executor(
                self._pool,
                lambda: extract_keyframes(
                    video_path,
                    stride_sec=preset.keyframe_stride_sec,
                    sharp_window=preset.sharp_window,
                ),
            )
        log.info("Ключевых кадров: %d", len(keyframes))
        await progress(0.30, f"Кадры готовы: {len(keyframes)} шт.")

        # ── Параллельный запуск PZ3 + PZ5 + PZ6 ─────────────────────────────
        stage_map: dict[str, object] = {
            "yolo": lambda: yolo_keyframes(keyframes, fps, preset),
        }
        if self.use_ocr:
            stage_map["ocr"] = lambda: ocr_keyframes(keyframes, fps, preset)
        if self.use_clip:
            stage_map["clip"] = lambda: clip_keyframes(keyframes, fps, preset)
        if self.use_nsfw:
            stage_map["nsfw"] = lambda: nsfw_keyframes(keyframes, fps)
        if self.use_action:
            stage_map["action"] = lambda: xclip_action(keyframes, fps)

        await progress(0.33, f"Параллельный анализ: {', '.join(stage_map.keys())}...")

        # Прогрев тяжёлых ленивых импортов до параллельных потоков (см. выше)
        await loop.run_in_executor(self._pool, lambda: _warm_imports(self.use_ocr))

        _names = {"yolo": "объекты (YOLO+World)", "ocr": "текст в кадре (OCR)",
                  "clip": "сцена (SigLIP)", "nsfw": "NSFW (Falconsai)",
                  "action": "действия (X-CLIP)"}

        async def _run_stage(sid: str, fn) -> list:
            ts = time.time()
            log.info("▶ [%s] %s — старт", sid, _names.get(sid, sid))
            try:
                dets = await asyncio.wait_for(
                    loop.run_in_executor(self._pool, fn),
                    timeout=self.stage_timeout,
                )
                log.info("✔ [%s] — %d детекций за %.1fс", sid, len(dets), time.time() - ts)
                return dets
            except asyncio.TimeoutError:
                log.warning("✖ [%s] — таймаут (%.0fс), пропускаю", sid, self.stage_timeout)
                return []
            except Exception as exc:
                log.warning("✖ [%s] — ошибка: %s", sid, exc)
                return []

        results = await asyncio.gather(
            *[_run_stage(sid, fn) for sid, fn in stage_map.items()]
        )

        all_dets: list = []
        for dets in results:
            all_dets.extend(dets)

        await progress(0.80, f"Видео-анализ: {len(all_dets)} событий")
        return all_dets

    # ── Audio analysis ────────────────────────────────────────────────────────

    async def _analyze_audio(self, video_path: Path, fps: float, duration: float, progress: ProgressCb):
        if self.offline or not video_path.exists():
            await progress(0.6, "Аудио (мок): Whisper + классификация...")
            await asyncio.sleep(0.5)
            return self._mock_audio_detections(fps, duration)

        # Аудио: Whisper → транскрипт → словарь + (опц.) LLM-классификация.
        # Ошибка аудио-ветки (нет звуковой дорожки, битый ffmpeg) НЕ должна
        # валить весь анализ — видео-каналы продолжают работать.
        loop = asyncio.get_event_loop()
        _ts = time.time()
        log.info("▶ [audio] речь: Whisper + классификация — старт")

        def _safe_audio() -> list:
            try:
                return _run_pz4(
                    video_path, fps,
                    preset=self.preset,
                    model_size=self.whisper_model,
                    use_llm=self.use_llm,
                )
            except Exception as exc:
                log.warning("✖ [audio] — ошибка (вероятно нет аудиодорожки): %s", exc)
                return []

        detections = await loop.run_in_executor(None, _safe_audio)
        log.info("✔ [audio] — %d детекций за %.1fс", len(detections), time.time() - _ts)
        await progress(0.65, f"Аудио: {len(detections)} событий")
        return detections

    # ── Post-processing (PZ8) ────────────────────────────────────────────────

    def _cross_modal_boost(self, detections: list[Detection], fps: float) -> None:
        """Кросс-модальное согласие: если один субкласс независимо найден и в
        аудио, и в видео в пересекающихся интервалах — поднимаем уверенность
        обеим детекциям. Согласие двух модальностей — сильный сигнал, что это
        не ложняк. Мутирует confidence на месте.
        """
        tol = int(2.0 * fps)  # допускаем расхождение тайм-кодов до 2 сек
        for d in detections:
            for o in detections:
                if o is d or o.subclass != d.subclass or o.type == d.type:
                    continue
                # пересечение интервалов с допуском
                if d.startFrame - tol <= o.endFrame and o.startFrame - tol <= d.endFrame:
                    d.confidence = round(min(0.99, d.confidence + 0.12), 3)
                    break

    def _postprocess(self, detections: list[Detection], fps: float = 30.0) -> list[Detection]:
        """Кросс-модальная фьюзия + temporal NMS по каждому субклассу."""
        if not detections:
            return []
        self._cross_modal_boost(detections, fps)
        gap_frames = int(2.0 * fps)
        by_class: dict[str, list[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.subclass.value, []).append(d)

        result: list[Detection] = []
        for cls, dets in by_class.items():
            dets = sorted(dets, key=lambda d: d.startFrame)
            merged = [dets[0]]
            for d in dets[1:]:
                last = merged[-1]
                if d.startFrame - last.endFrame <= gap_frames:
                    new_ef = max(last.endFrame, d.endFrame)
                    new_conf = round(max(last.confidence, d.confidence), 3)
                    t1 = _fmt_clock(new_ef / fps)
                    merged[-1] = Detection(
                        startFrame=last.startFrame, endFrame=new_ef,
                        start_time=last.start_time, end_time=t1,
                        time_interval=f"{last.start_time} - {t1}",
                        subclass=last.subclass,
                        confidence=new_conf,
                        type=last.type,
                    )
                else:
                    merged.append(d)
            result.extend(merged)

        return sorted(result, key=lambda d: d.startFrame)

    # ── Media extraction ──────────────────────────────────────────────────────

    async def _extract_media(
        self,
        video_path: Path,
        detections: list[Detection],
        fps: float,
        duration_sec: float,
    ) -> list[DetectionMedia]:
        results: list[DetectionMedia] = []
        for i, det in enumerate(detections):
            if self.offline:
                frame = _placeholder_frame(det.subclass.value, det.time_interval, det.confidence)
                results.append(DetectionMedia(detection=det, frame=frame, clip_path=None))
            else:
                frame = self._grab_frame(video_path, det, fps)
                clip = self._cut_clip(video_path, det, fps, duration_sec, i)
                results.append(DetectionMedia(detection=det, frame=frame, clip_path=clip))
        return results

    def _grab_frame(self, video_path: Path, det: Detection, fps: float) -> Optional[Image.Image]:
        try:
            import cv2
            mid = (det.startFrame + det.endFrame) // 2
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return _annotate_frame(img, det.subclass.value, det.time_interval, det.confidence)
        except Exception:
            return None

    def _cut_clip(
        self,
        video_path: Path,
        det: Detection,
        fps: float,
        duration_sec: float,
        idx: int,
    ) -> Optional[Path]:
        try:
            start = max(0.0, det.startFrame / fps - 0.5)
            end = min(duration_sec, det.endFrame / fps + 0.5)
            out = self.clips_dir / f"clip_{idx:03d}_{det.subclass.value}.mp4"
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [ffmpeg, "-y", "-i", str(video_path),
                 "-ss", str(start), "-to", str(end),
                 "-c:v", "libx264", "-c:a", "aac", "-preset", "fast", str(out)],
                capture_output=True, check=True,
            )
            return out
        except Exception:
            return None

    # ── LLM-слой (опционально, поверх ядра) ────────────────────────────────────

    async def _llm_verify(self, media: list[DetectionMedia]) -> list[DetectionMedia]:
        """Vision-проверка пограничных видео-детекций. Отбрасывает не подтверждённые,
        поднимает уверенность подтверждённым. Аудио-детекции не трогаем.
        """
        from . import llm as _llm
        model = self.preset.llm_vision_model
        if not _llm.is_available(model):
            return media

        loop = asyncio.get_event_loop()
        kept: list[DetectionMedia] = []
        for m in media:
            d = m.detection
            borderline = (d.type == DetectionType.video
                          and m.frame is not None
                          and 0.45 <= d.confidence < 0.70)
            if not borderline:
                kept.append(m)
                continue
            conf = await loop.run_in_executor(
                self._pool,
                lambda fr=m.frame, sub=d.subclass.value:
                    _llm.verify_frame(fr, sub, model),
            )
            if conf is None:                # LLM не ответил — оставляем как есть
                kept.append(m)
            elif conf <= 0.0:               # не подтверждено — отбрасываем
                continue
            else:                           # подтверждено — берём максимум
                d.confidence = round(max(d.confidence, conf), 3)
                kept.append(m)
        return kept

    async def _llm_summary(self, detections: list[Detection]) -> Optional[str]:
        from . import llm as _llm
        model = self.preset.llm_text_model
        if not detections or not _llm.is_available(model):
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool, lambda: _llm.summarize(detections, model)
        )

    # ── Mock helpers ──────────────────────────────────────────────────────────

    def _mock_video_detections(self, fps: float, duration: float) -> list[Detection]:
        random.seed(42)
        out = []
        safe_end = max(10.0, duration - 5)
        for _ in range(8):
            t0 = random.uniform(3, safe_end)
            dur = random.uniform(1, min(4, duration - t0))
            sf, ef = int(t0 * fps), int((t0 + dur) * fps)
            out.append(Detection(
                startFrame=sf, endFrame=ef,
                start_time=_fmt_clock(t0), end_time=_fmt_clock(t0 + dur),
                time_interval=f"{_fmt_clock(t0)} - {_fmt_clock(t0 + dur)}",
                subclass=random.choice([Subclass.alcohol, Subclass.smoking,
                                        Subclass.extremism, Subclass.lgbt]),
                confidence=round(random.uniform(0.55, 0.98), 3),
                type=DetectionType.video,
            ))
        return out

    def _mock_audio_detections(self, fps: float, duration: float) -> list[Detection]:
        random.seed(7)
        out = []
        safe_end = max(10.0, duration - 5)
        for _ in range(6):
            t0 = random.uniform(3, safe_end)
            dur = random.uniform(3, min(8, duration - t0))
            sf, ef = int(t0 * fps), int((t0 + dur) * fps)
            out.append(Detection(
                startFrame=sf, endFrame=ef,
                start_time=_fmt_clock(t0), end_time=_fmt_clock(t0 + dur),
                time_interval=f"{_fmt_clock(t0)} - {_fmt_clock(t0 + dur)}",
                subclass=random.choice([Subclass.drugs, Subclass.alcohol,
                                        Subclass.ludomania, Subclass.violence]),
                confidence=0.98,
                type=DetectionType.audio,
            ))
        return out
