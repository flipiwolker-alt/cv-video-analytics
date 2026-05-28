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

from .schemas import (
    Detection,
    DetectionType,
    SourceInfo,
    Subclass,
    TimeBasedReport,
    TopSourceInfo,
)

ProgressCb = Callable[[float, str], Awaitable[None]]

_SUBCLASS_COLORS = {
    "alcohol":   (245, 158, 11),
    "drugs":     (239, 68, 68),
    "smoking":   (107, 114, 128),
    "extremism": (220, 38, 38),
    "ludomania": (139, 92, 246),
    "lgbt":      (236, 72, 153),
    "violence":  (185, 28, 28),
    "vandalism": (217, 119, 6),
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
        whisper_model: str = "tiny",
        use_ocr: bool = True,
        use_clip: bool = True,
        stage_timeout: float = 180.0,
        clips_dir: Optional[Path] = None,
    ):
        self.dummy = dummy
        self.offline = offline
        self.whisper_model = whisper_model
        self.use_ocr = use_ocr
        self.use_clip = use_clip
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

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # ── Download ──────────────────────────────────────────────────
            await progress(0.05, "Скачиваю видео с YouTube...")
            video_path, duration_sec, fps, frame_count = await self._download(
                url, tmp_path, quality
            )

            # ── Analysis (параллельно) ────────────────────────────────────
            await progress(0.20, "Извлекаю ключевые кадры...")
            audio_task = asyncio.create_task(
                self._analyze_audio(video_path, fps, duration_sec, progress)
            )
            video_task = asyncio.create_task(
                self._analyze_video(video_path, fps, duration_sec, progress)
            )
            audio_dets, video_dets = await asyncio.gather(audio_task, video_task)

            # ── Post-processing ───────────────────────────────────────────
            await progress(0.85, "Свожу результаты воедино...")
            detections = self._postprocess(audio_dets + video_dets, fps=fps)

            # ── Извлечение медиа ──────────────────────────────────────────
            await progress(0.92, "Извлекаю кадры и клипы...")
            media = await self._extract_media(video_path, detections, fps, duration_sec)

            await progress(1.0, "Готово.")

        processing_time = time.time() - t0
        now = datetime.now().isoformat()

        report = TimeBasedReport(
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
        return report, media

    # ── Download ──────────────────────────────────────────────────────────────

    async def _download(self, url: str, tmp: Path, quality: str):
        if self.offline:
            await asyncio.sleep(0.3)
            return tmp / "video.mp4", 180.0, 30.0, 5400

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

        # PZ1 — keyframes (должен быть первым, остальные зависят от него)
        keyframes = await loop.run_in_executor(
            self._pool, lambda: extract_keyframes(video_path, stride_sec=1.5)
        )
        await progress(0.30, f"Кадры готовы: {len(keyframes)} шт.")

        # ── Параллельный запуск PZ3 + PZ5 + PZ6 ─────────────────────────────
        stage_map: dict[str, object] = {
            "yolo": lambda: yolo_keyframes(keyframes, fps),
        }
        if self.use_ocr:
            stage_map["ocr"] = lambda: ocr_keyframes(keyframes, fps)
        if self.use_clip:
            stage_map["clip"] = lambda: clip_keyframes(keyframes, fps)

        await progress(0.33, f"Параллельный анализ: {', '.join(stage_map.keys())}...")

        async def _run_stage(sid: str, fn) -> list:
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(self._pool, fn),
                    timeout=self.stage_timeout,
                )
            except (asyncio.TimeoutError, Exception):
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

        # PZ4: реальный Whisper
        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(
            None,
            lambda: _run_pz4(video_path, fps, self.whisper_model),
        )
        await progress(0.65, f"Аудио: {len(detections)} событий")
        return detections

    # ── Post-processing (PZ8) ────────────────────────────────────────────────

    def _postprocess(self, detections: list[Detection], fps: float = 30.0) -> list[Detection]:
        """Temporal NMS: объединяем близкие детекции одного класса."""
        if not detections:
            return []
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
