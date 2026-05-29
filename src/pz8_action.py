"""PZ8: Распознавание действий по ВИДЕО-окну (а не одному кадру).

Одиночный кадр плохо отличает «драку» от «объятий» или «порчу» от «ремонта» —
нужна временная динамика. Используем zero-shot видео-модель **X-CLIP**
(microsoft/xclip-base-patch32, ~400 MB, MIT): она кодирует ПОСЛЕДОВАТЕЛЬНОСТЬ
из 8 кадров и сравнивает с текстовыми описаниями действий.

Ловит «процессные» нарушения: насилие (драка/избиение) и вандализм
(порча/поджог/граффити). Скачивается автоматически при первом запуске.

Тяжёлая на CPU — число окон ограничено (max_windows).
"""
from __future__ import annotations

import threading

import torch

from .pz4_audio import _fmt_clock
from .schemas import Detection, DetectionType, Subclass
from .pz1_keyframes import Keyframe
from .device import DEVICE, gpu_guard

_MODEL_ID = "microsoft/xclip-base-patch32"
_NUM_FRAMES = 8           # X-CLIP base ожидает 8 кадров на «видео»
_THRESHOLD = 0.35         # мин. доля вероятности у action-промпта (softmax по всем)

# Промпты действий → субкласс. Несколько формулировок на класс.
_ACTION_PROMPTS: list[tuple[str, str]] = [
    ("people physically fighting and hitting each other", "violence"),
    ("a brutal assault, beating someone up", "violence"),
    ("a person kicking and punching another person", "violence"),
    ("someone smashing and destroying property", "vandalism"),
    ("a person spray painting graffiti on a wall", "vandalism"),
    ("setting fire to property, arson", "vandalism"),
]
# Нейтральные действия — забирают вероятность у обычных сцен.
_NEUTRAL_PROMPTS: list[str] = [
    "people calmly talking or standing together",
    "a normal everyday activity",
    "people walking on the street",
    "a person sitting, working or dancing",
]

_ALL_TEXTS = [p for p, _ in _ACTION_PROMPTS] + _NEUTRAL_PROMPTS
_IDX_TO_SUB = {i: sub for i, (_, sub) in enumerate(_ACTION_PROMPTS)}  # action-индексы

_bundle: dict | None = None
_lock = threading.Lock()


def _init() -> dict:
    global _bundle
    with _lock:
        if _bundle is not None:
            return _bundle
        from transformers import XCLIPModel, XCLIPProcessor
        model = XCLIPModel.from_pretrained(_MODEL_ID)
        processor = XCLIPProcessor.from_pretrained(_MODEL_ID)
        model.eval().to(DEVICE)
        _bundle = {"model": model, "processor": processor}
        return _bundle


def _build_windows(keyframes: list[Keyframe], max_windows: int) -> list[list[Keyframe]]:
    """Окна по 8 последовательных keyframe-ов с 50% перекрытием, с ограничением числа."""
    n = len(keyframes)
    if n < _NUM_FRAMES:
        return []
    stride = max(1, _NUM_FRAMES // 2)
    starts = list(range(0, n - _NUM_FRAMES + 1, stride))
    if len(starts) > max_windows:                      # прорежаем равномерно
        step = len(starts) / max_windows
        starts = [starts[int(i * step)] for i in range(max_windows)]
    return [keyframes[s:s + _NUM_FRAMES] for s in starts]


def xclip_action(
    keyframes: list[Keyframe],
    fps: float,
    threshold: float = _THRESHOLD,
    max_windows: int = 24,
) -> list[Detection]:
    """Zero-shot распознавание действий по видео-окнам. Детекции violence/vandalism."""
    windows = _build_windows(keyframes, max_windows)
    if not windows:
        return []

    try:
        b = _init()
    except Exception:
        return []   # модель недоступна — не валим пайплайн
    model, processor = b["model"], b["processor"]

    detections: list[Detection] = []
    for win in windows:
        videos = [[kf.image.convert("RGB") for kf in win]]   # одно «видео» = 8 кадров
        try:
            inputs = processor(text=_ALL_TEXTS, videos=videos,
                               return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad(), gpu_guard():
                probs = model(**inputs).logits_per_video.softmax(dim=-1)[0]
        except Exception:
            continue

        top_idx = int(probs.argmax())
        if top_idx not in _IDX_TO_SUB:        # победил нейтральный промпт
            continue
        score = float(probs[top_idx])
        if score < threshold:
            continue

        sub = _IDX_TO_SUB[top_idx]
        conf = round(min(0.95, 0.5 + (score - threshold) / (1 - threshold) * 0.45), 3)
        t0 = win[0].time_sec
        t1 = win[-1].time_sec + 1.0
        detections.append(Detection(
            startFrame=int(t0 * fps),
            endFrame=int(t1 * fps),
            start_time=_fmt_clock(t0),
            end_time=_fmt_clock(t1),
            time_interval=f"{_fmt_clock(t0)} - {_fmt_clock(t1)}",
            subclass=Subclass(sub),
            confidence=conf,
            type=DetectionType.video,
        ))
    return detections
