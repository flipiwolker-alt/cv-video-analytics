"""PZ3: OCR на keyframe-ах → keyword-классификатор.

Ищет текстовые надписи в кадре: субтитры, баннеры, логотипы, рекламные оверлеи.
Результат → тот же keyword-классификатор что у PZ4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .pz4_audio import _classify_segment, _fmt_clock
from .schemas import Detection, DetectionType, Subclass
from .pz1_keyframes import Keyframe

# Модель инициализируется один раз, потом переиспользуется
_ocr_model = None
_ocr_lock = __import__("threading").Lock()


def _get_ocr():
    global _ocr_model
    with _ocr_lock:
        if _ocr_model is None:
            import easyocr
            from .config import MODELS_DIR
            model_dir = str(MODELS_DIR / "easyocr")
            globals()["_ocr_model"] = easyocr.Reader(
                ["ru", "en"], gpu=False, verbose=False,
                model_storage_directory=model_dir,
            )
    return _ocr_model


def ocr_keyframes(
    keyframes: list[Keyframe],
    fps: float,
    conf_threshold: float = 0.5,
    min_text_len: int = 4,
    max_frames: int = 40,
) -> list[Detection]:
    """
    max_frames — сколько кадров максимум отдать OCR (равномерная подвыборка).
    EasyOCR на CPU: ~2-4 сек/кадр, поэтому 40 кадров ≈ 2 мин максимум.
    """
    reader = _get_ocr()
    detections: list[Detection] = []

    # Равномерная подвыборка если кадров слишком много
    if len(keyframes) > max_frames:
        step = len(keyframes) // max_frames
        keyframes = keyframes[::step][:max_frames]

    for kf in keyframes:
        img_np = np.array(kf.image)
        results = reader.readtext(img_np, detail=1, paragraph=False)

        full_text = " ".join(
            text
            for (_, text, conf) in results
            if conf >= conf_threshold and len(text.strip()) >= min_text_len
        )
        if not full_text.strip():
            continue

        hits = _classify_segment(full_text)
        for subclass, confidence in hits:
            # интервал — ±1 секунда от кадра
            t0 = max(0.0, kf.time_sec - 0.5)
            t1 = kf.time_sec + 1.0
            detections.append(Detection(
                startFrame=int(t0 * fps),
                endFrame=int(t1 * fps),
                start_time=_fmt_clock(t0),
                end_time=_fmt_clock(t1),
                time_interval=f"{_fmt_clock(t0)} - {_fmt_clock(t1)}",
                subclass=Subclass(subclass),
                confidence=round(confidence * 0.9, 3),  # OCR чуть ненадёжнее Whisper
                type=DetectionType.video,
            ))

    return detections
