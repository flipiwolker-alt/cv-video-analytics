"""PZ3: OCR на keyframe-ах → keyword-классификатор.

Ищет текстовые надписи в кадре: субтитры, баннеры, логотипы, рекламные оверлеи.
Результат → тот же keyword-классификатор что у PZ4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .pz4_audio import _classify_segment, _fmt_clock
from .schemas import Detection, DetectionType, Subclass
from .pz1_keyframes import Keyframe
from .presets import Preset, get_preset
from .device import ON_CUDA, gpu_guard

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
                ["ru", "en"], gpu=ON_CUDA, verbose=False,  # easyocr.gpu = только CUDA
                model_storage_directory=model_dir,
            )
    return _ocr_model


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _prep_for_ocr(img_np: np.ndarray, target_long: int = 1280) -> np.ndarray:
    """Подготовка кадра под OCR: апскейл мелкого кадра + CLAHE-контраст надписей.

    Применяется ТОЛЬКО к OCR-ветке (текст не «натуральная сцена» — распределение
    для CLIP/SigLIP здесь не важно, а контраст и размер реально поднимают recall).
    """
    h, w = img_np.shape[:2]
    longest = max(h, w)
    if longest < target_long:
        scale = min(2.0, target_long / max(longest, 1))
        img_np = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def ocr_keyframes(
    keyframes: list[Keyframe],
    fps: float,
    preset: Preset | str | None = None,
    conf_threshold: float = 0.5,
    min_text_len: int = 4,
    max_frames: int | None = None,
    rotations: bool = False,
) -> list[Detection]:
    """
    max_frames — сколько кадров максимум отдать OCR (равномерная подвыборка).
    По умолчанию берётся из пресета. EasyOCR на CPU ~2-4 сек/кадр.
    """
    p = preset if isinstance(preset, Preset) else get_preset(preset)
    if max_frames is None:
        max_frames = p.ocr_max_frames
    reader = _get_ocr()
    detections: list[Detection] = []

    # Равномерная подвыборка если кадров слишком много
    if len(keyframes) > max_frames:
        step = len(keyframes) // max_frames
        keyframes = keyframes[::step][:max_frames]

    # Тюнинг детектора CRAFT под recall: ниже пороги → ловим бледный/мелкий текст.
    # Наклонённые строки EasyOCR выпрямляет сам (rectify повёрнутого quad).
    _det = dict(text_threshold=0.6, low_text=0.3, link_threshold=0.4,
                mag_ratio=1.0, canvas_size=2560)

    for kf in keyframes:
        img_np = _prep_for_ocr(np.array(kf.image))   # CPU-препроцесс
        with gpu_guard():                             # GPU-распознавание сериализуем
            results = reader.readtext(img_np, detail=1, paragraph=False, **_det)
            # Текста нет совсем — возможно, он повёрнут на 90°/180°: пробуем повороты
            if rotations and not results:
                results = reader.readtext(img_np, detail=1, paragraph=False,
                                          rotation_info=[90, 180, 270], **_det)

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
