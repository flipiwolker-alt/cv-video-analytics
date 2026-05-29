"""Пресеты качества — единая «ручка» точность ⇄ скорость.

Всё железо на CPU, поэтому размеры моделей напрямую определяют и точность,
и время анализа. Пресет выбирается одним параметром в API/пайплайне и
раскатывается на ВСЕ сферы детекции сразу (YOLO, CLIP, Whisper, OCR).

    fast      — быстрый прогон, умеренная точность   (~1-3 мин на 3-мин ролик)
    balanced  — хорошая точность, разумная скорость   (~3-7 мин)
    accurate  — максимум качества, медленно            (~5-15 мин)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    name: str

    # ── YOLO (объекты) ──────────────────────────────────────────────────────
    yolo_model: str            # классический COCO-детектор
    yolo_world_model: str | None  # open-vocabulary (None = выключен)
    yolo_conf: float
    yolo_imgsz: int            # разрешение инференса (больше → лучше мелкие объекты)

    # ── Подготовка кадров ───────────────────────────────────────────────────
    sharp_window: int          # ±кадров для выбора самого резкого (0 = выкл)

    # ── CLIP (сцена) ────────────────────────────────────────────────────────
    clip_model: str
    clip_max_frames: int

    # ── Whisper (речь) ──────────────────────────────────────────────────────
    whisper_model: str
    whisper_beam: int

    # ── OCR (текст в кадре) ─────────────────────────────────────────────────
    ocr_max_frames: int

    # ── Keyframes (плотность выборки кадров) ────────────────────────────────
    keyframe_stride_sec: float

    # ── LLM (слой поверх ядра) ──────────────────────────────────────────────
    llm_text_model: str        # текстовая модель Ollama для классификации речи
    llm_vision_model: str      # vision-модель Ollama для проверки кадров


# Каталог пресетов. Все модели скачиваются автоматически при первом запуске.
PRESETS: dict[str, Preset] = {
    "fast": Preset(
        name="fast",
        yolo_model="yolo11s.pt",
        yolo_world_model="yolov8s-worldv2.pt",   # open-vocab в ultralytics — на базе v8
        yolo_conf=0.35,
        yolo_imgsz=640,
        sharp_window=0,                          # fast: без выбора резкого (только дедуп)
        clip_model="google/siglip2-base-patch16-224",
        clip_max_frames=80,
        whisper_model="small",
        whisper_beam=5,
        ocr_max_frames=30,
        keyframe_stride_sec=2.0,
        llm_text_model="qwen2.5:3b",
        llm_vision_model="qwen2.5vl:7b",
    ),
    "balanced": Preset(
        name="balanced",
        yolo_model="yolo11m.pt",
        yolo_world_model="yolov8m-worldv2.pt",
        yolo_conf=0.30,
        yolo_imgsz=960,
        sharp_window=1,                          # balanced: выбор из 3 кадров
        clip_model="google/siglip2-base-patch16-384",
        clip_max_frames=120,
        whisper_model="large-v3-turbo",
        whisper_beam=5,
        ocr_max_frames=50,
        keyframe_stride_sec=1.5,
        llm_text_model="qwen2.5:7b",
        llm_vision_model="qwen2.5vl:7b",
    ),
    "accurate": Preset(
        name="accurate",
        yolo_model="yolo11x.pt",
        yolo_world_model="yolov8x-worldv2.pt",
        yolo_conf=0.25,
        yolo_imgsz=1280,
        sharp_window=2,                          # accurate: выбор из 5 кадров
        clip_model="google/siglip2-so400m-patch14-384",
        clip_max_frames=200,
        whisper_model="large-v3",
        whisper_beam=8,
        ocr_max_frames=80,
        keyframe_stride_sec=1.0,
        llm_text_model="qwen2.5:7b",
        llm_vision_model="qwen2.5vl:7b",
    ),
}

DEFAULT_PRESET = "balanced"


def get_preset(name: str | None) -> Preset:
    """Возвращает пресет по имени; неизвестное имя → DEFAULT_PRESET."""
    if not name:
        return PRESETS[DEFAULT_PRESET]
    return PRESETS.get(name.lower().strip(), PRESETS[DEFAULT_PRESET])
