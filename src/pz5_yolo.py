"""PZ5: YOLO детекция объектов.

YOLOv8n (nano) обнаруживает объекты из COCO — маппим на наши 8 субклассов.
Модель скачивается автоматически (~6 MB для nano).
"""
from __future__ import annotations

import numpy as np

from .pz4_audio import _fmt_clock
from .schemas import Detection, DetectionType, Subclass
from .pz1_keyframes import Keyframe

# COCO class_id → наш субкласс + базовый confidence
# Полный список COCO: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
_COCO_TO_SUBCLASS: dict[int, tuple[str, float]] = {
    # Алкоголь
    39: ("alcohol", 0.75),   # bottle
    40: ("alcohol", 0.85),   # wine glass
    41: ("alcohol", 0.50),   # cup (слабый сигнал)
    # Насилие
    43: ("violence", 0.70),  # knife
    76: ("violence", 0.90),  # scissors (низкий приоритет, но COCO-friendly)
    # Курение — в стандартном COCO нет, используем YOLOWorld-подход
    # (будет добавлено через PZ6/custom)
}

_yolo_model = None
_yolo_lock = __import__("threading").Lock()


def _get_yolo(model_name: str = "yolov8n.pt"):
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            from ultralytics import YOLO
            from .config import MODELS_DIR
            local = MODELS_DIR / "ultralytics" / model_name
            globals()["_yolo_model"] = YOLO(str(local) if local.exists() else model_name)
            if not local.exists():
                import shutil, pathlib
                downloaded = pathlib.Path(_yolo_model.ckpt_path)
                if downloaded.exists() and downloaded != local:
                    local.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(downloaded, local)
    return _yolo_model


def yolo_keyframes(
    keyframes: list[Keyframe],
    fps: float,
    conf_threshold: float = 0.45,
    model_name: str = "yolov8n.pt",
) -> list[Detection]:
    """Детектирует объекты на каждом keyframe, возвращает Detection-ы."""
    model = _get_yolo(model_name)
    detections: list[Detection] = []

    # Батч: по 16 кадров за раз
    batch_size = 16
    for i in range(0, len(keyframes), batch_size):
        batch = keyframes[i:i + batch_size]
        images = [np.array(kf.image) for kf in batch]

        results = model(images, conf=conf_threshold, verbose=False)

        for kf, result in zip(batch, results):
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id not in _COCO_TO_SUBCLASS:
                    continue

                subclass_str, base_conf = _COCO_TO_SUBCLASS[cls_id]
                # финальный confidence = среднее между YOLO и базовым
                final_conf = round((conf + base_conf) / 2, 3)

                t0 = max(0.0, kf.time_sec - 0.5)
                t1 = kf.time_sec + 1.0
                detections.append(Detection(
                    startFrame=int(t0 * fps),
                    endFrame=int(t1 * fps),
                    start_time=_fmt_clock(t0),
                    end_time=_fmt_clock(t1),
                    time_interval=f"{_fmt_clock(t0)} - {_fmt_clock(t1)}",
                    subclass=Subclass(subclass_str),
                    confidence=final_conf,
                    type=DetectionType.video,
                ))

    return detections
