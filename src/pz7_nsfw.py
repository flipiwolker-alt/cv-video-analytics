"""PZ7: Детекция NSFW (порнография / нагота) профильной моделью.

Zero-shot (CLIP/SigLIP) для наготы слабоват, поэтому используем специализированный
бесплатный классификатор Falconsai/nsfw_image_detection (ViT, ~340 MB, Apache-2.0).
Он на порядок точнее по этой теме. Дополняет SigLIP-промпты и YOLO-World.

Модель скачивается автоматически при первом запуске.
"""
from __future__ import annotations

import threading

import torch

from .pz4_audio import _fmt_clock
from .schemas import Detection, DetectionType, Subclass
from .pz1_keyframes import Keyframe
from .device import DEVICE, gpu_guard

_MODEL_ID = "Falconsai/nsfw_image_detection"
_NSFW_THRESHOLD = 0.70   # порог вероятности nsfw для срабатывания

_bundle: dict | None = None
_lock = threading.Lock()


def _init() -> dict:
    global _bundle
    with _lock:
        if _bundle is not None:
            return _bundle
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        model = AutoModelForImageClassification.from_pretrained(_MODEL_ID)
        processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
        model.eval().to(DEVICE)
        # индекс метки "nsfw" из конфига (устойчиво к порядку меток)
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        nsfw_idx = next((i for i, lbl in id2label.items() if "nsfw" in lbl), 1)
        _bundle = {"model": model, "processor": processor, "nsfw_idx": nsfw_idx}
        return _bundle


def nsfw_keyframes(
    keyframes: list[Keyframe],
    fps: float,
    threshold: float = _NSFW_THRESHOLD,
    batch_size: int = 16,
    max_frames: int = 200,
) -> list[Detection]:
    """Классифицирует keyframe-ы на NSFW, возвращает детекции subclass=nsfw."""
    if len(keyframes) > max_frames:
        step = max(1, len(keyframes) // max_frames)
        keyframes = keyframes[::step][:max_frames]

    try:
        b = _init()
    except Exception:
        return []   # модель недоступна — не валим пайплайн
    model, processor, nsfw_idx = b["model"], b["processor"], b["nsfw_idx"]

    detections: list[Detection] = []
    for i in range(0, len(keyframes), batch_size):
        batch = keyframes[i:i + batch_size]
        images = [kf.image.convert("RGB") for kf in batch]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad(), gpu_guard():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, nsfw_idx]

        for kf, p in zip(batch, probs):
            score = float(p)
            if score < threshold:
                continue
            conf = round(min(0.98, score), 3)
            t0 = max(0.0, kf.time_sec - 0.5)
            t1 = kf.time_sec + 1.5
            detections.append(Detection(
                startFrame=int(t0 * fps),
                endFrame=int(t1 * fps),
                start_time=_fmt_clock(t0),
                end_time=_fmt_clock(t1),
                time_interval=f"{_fmt_clock(t0)} - {_fmt_clock(t1)}",
                subclass=Subclass.nsfw,
                confidence=conf,
                type=DetectionType.video,
            ))
    return detections
