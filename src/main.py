"""FastAPI-сервер: два режима работы.

Асинхронный (background task):
  POST /analyze          → task_id
  GET  /status/{task_id} → прогресс
  GET  /result/{task_id} → JSON-отчёт

Синхронный (ждёт результат):
  POST /analyze/sync     → JSON-отчёт (блокирующий, до ~5 мин)
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .pipeline import VideoAnalyzer
from .schemas import AnalyzeRequest, TaskInfo, TaskStatus, TimeBasedReport

app = FastAPI(
    title="CV Video Analytics API",
    description="Детектирование нежелательного контента в YouTube-видео",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TASKS: dict[str, TaskInfo] = {}


def _make_analyzer(preset: str = "balanced", use_llm: bool = True) -> VideoAnalyzer:
    return VideoAnalyzer(dummy=False, offline=False, preset=preset, use_llm=use_llm)


# ── Health-check ──────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "tasks_total": len(TASKS)}


# ── Асинхронный режим ─────────────────────────────────────────────────────────

@app.post("/analyze", response_model=TaskInfo, summary="Запустить анализ (асинхронно)")
async def analyze_async(req: AnalyzeRequest, bg: BackgroundTasks):
    """
    Запускает анализ в фоне. Возвращает task_id немедленно.
    Следите за прогрессом через GET /status/{task_id}.
    Результат забирайте через GET /result/{task_id}.
    """
    task_id = uuid.uuid4().hex[:12]
    info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.queued,
        created_at=datetime.now().isoformat(),
    )
    TASKS[task_id] = info
    bg.add_task(_run_task, task_id, str(req.url), req.quality, req.preset, req.use_llm)
    return info


@app.get("/status/{task_id}", response_model=TaskInfo, summary="Статус задачи")
def get_status(task_id: str):
    info = TASKS.get(task_id)
    if not info:
        raise HTTPException(404, detail=f"Задача {task_id!r} не найдена")
    return info


@app.get("/result/{task_id}", summary="Получить JSON-отчёт")
def get_result(task_id: str):
    info = TASKS.get(task_id)
    if not info:
        raise HTTPException(404, detail=f"Задача {task_id!r} не найдена")
    if info.status != TaskStatus.done:
        raise HTTPException(409, detail=f"Задача ещё не завершена (статус: {info.status})")
    return info.report


# ── Синхронный режим ──────────────────────────────────────────────────────────

class SyncRequest(BaseModel):
    url: str
    quality: str = "720p"
    preset: str = "balanced"          # fast | balanced | accurate
    use_clip: bool = True
    use_ocr: bool = True              # текст в кадре (баннеры/субтитры/логотипы); медленно на CPU
    use_nsfw: bool = True             # профильный детектор порнографии/наготы
    use_action: bool = True           # X-CLIP: действия по видео-окну (драка/порча); тяжело на CPU
    use_llm: bool = True              # LLM-классификация речи + резюме (Ollama)
    use_llm_vision: bool = False      # vision-проверка кадров (медленно на CPU)
    whisper_model: str | None = None  # переопределяет размер из пресета


@app.post("/analyze/sync", summary="Анализ и сразу результат (синхронно, до 5 мин)")
async def analyze_sync(req: SyncRequest):
    """
    Принимает ссылку на YouTube, запускает полный анализ, возвращает JSON-отчёт.

    Блокирует запрос до завершения (обычно 1-5 минут).
    Подходит для скриптов и Postman. Пример:

        curl -X POST http://localhost:8000/analyze/sync \\
             -H "Content-Type: application/json" \\
             -d '{"url": "https://www.youtube.com/watch?v=XXXX"}'
    """
    analyzer = VideoAnalyzer(
        dummy=False,
        offline=False,
        preset=req.preset,
        whisper_model=req.whisper_model,
        use_clip=req.use_clip,
        use_ocr=req.use_ocr,
        use_nsfw=req.use_nsfw,
        use_action=req.use_action,
        use_llm=req.use_llm,
        use_llm_vision=req.use_llm_vision,
    )
    try:
        report, _media = await analyzer.run(req.url, quality=req.quality)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))
    return report.model_dump(mode="json")


# ── Загрузка файла (multipart) — синхронно ────────────────────────────────────

@app.post("/analyze/upload", summary="Загрузить видеофайл и сразу получить результат")
async def analyze_upload(
    file: UploadFile = File(..., description="Видеофайл (mp4/mkv/mov/webm)"),
    preset: str = Form("balanced"),       # fast | balanced | accurate
    use_clip: bool = Form(True),
    use_ocr: bool = Form(True),
    use_nsfw: bool = Form(True),
    use_action: bool = Form(True),
    use_llm: bool = Form(True),
    use_llm_vision: bool = Form(False),
    whisper_model: str | None = Form(None),
):
    """
    Принимает видеофайл напрямую (multipart/form-data), анализирует локально
    (без интернета/yt-dlp) и возвращает JSON-отчёт. Удобно для Postman:
    form-data → ключ `file` тип File + при желании текстовые поля (preset и пр.).

    Блокирует запрос до завершения (обычно 1-5 минут).
    """
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with tmp:
            shutil.copyfileobj(file.file, tmp)
        analyzer = VideoAnalyzer(
            dummy=False, offline=False, preset=preset, whisper_model=whisper_model,
            use_clip=use_clip, use_ocr=use_ocr, use_nsfw=use_nsfw,
            use_action=use_action, use_llm=use_llm, use_llm_vision=use_llm_vision,
        )
        report, _media = await analyzer.run(tmp.name, quality="720p")
        return report.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))
    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ── Background worker ─────────────────────────────────────────────────────────

async def _run_task(task_id: str, url: str, quality: str,
                    preset: str = "balanced", use_llm: bool = True) -> None:
    info = TASKS[task_id]

    async def on_progress(p: float, msg: str) -> None:
        info.progress = round(p, 3)
        info.message = msg
        if p < 0.22:
            info.status = TaskStatus.downloading
        elif p < 0.32:
            info.status = TaskStatus.extracting
        elif p < 0.82:
            info.status = TaskStatus.video
        elif p < 0.93:
            info.status = TaskStatus.aggregating

    try:
        analyzer = _make_analyzer(preset=preset, use_llm=use_llm)
        report, _media = await analyzer.run(url, quality=quality, on_progress=on_progress)
        info.report = report
        info.status = TaskStatus.done
        info.progress = 1.0
        info.message = "Готово"
    except Exception as exc:
        info.status = TaskStatus.failed
        info.error = str(exc)
    finally:
        info.finished_at = datetime.now().isoformat()
