"""PZ5: Детекция объектов — COCO-детектор + YOLO-World (open-vocabulary).

Два детектора работают вместе:
  1. Классический YOLOv8 (s/m/x по пресету) — высокая точность на частых
     COCO-объектах (бутылка, бокал, нож).
  2. YOLO-World — open-vocabulary детекция по текстовым промптам. Ловит то,
     чего НЕТ в COCO: сигареты, оружие, шприцы, граффити, карты/казино,
     экстремистскую символику.

Результаты обоих сливаются; дедуп близких детекций делает пайплайн (PZ8).
Модели скачиваются автоматически при первом запуске.
"""
from __future__ import annotations

import threading

import numpy as np

from .pz4_audio import _fmt_clock
from .schemas import Detection, DetectionType, Subclass
from .pz1_keyframes import Keyframe
from .presets import Preset, get_preset
from .device import DEVICE, gpu_guard

# ── COCO class_id → (субкласс, базовый приор уверенности) ─────────────────────
_COCO_TO_SUBCLASS: dict[int, tuple[str, float]] = {
    39: ("alcohol", 0.70),   # bottle
    40: ("alcohol", 0.85),   # wine glass
    41: ("alcohol", 0.45),   # cup (слабый сигнал)
    43: ("violence", 0.70),  # knife
}

# ── Open-vocabulary словарь YOLO-World: промпт → (субкласс, приор) ────────────
# Промпты — короткие именные группы (так YOLO-World работает точнее всего).
_WORLD_VOCAB: list[tuple[str, str, float]] = [
    # smoking
    ("cigarette",                 "smoking", 0.80),
    ("lit cigarette",             "smoking", 0.80),
    ("person smoking",            "smoking", 0.70),
    ("electronic cigarette vape", "smoking", 0.70),
    ("hookah",                    "smoking", 0.75),
    ("cigar",                     "smoking", 0.70),
    # alcohol (дополняет COCO)
    ("beer bottle",               "alcohol", 0.80),
    ("beer can",                  "alcohol", 0.75),
    ("whiskey bottle",            "alcohol", 0.80),
    ("vodka bottle",              "alcohol", 0.80),
    # drugs
    ("syringe",                   "drugs",   0.75),
    ("hypodermic needle",         "drugs",   0.75),
    ("pills on table",            "drugs",   0.55),
    ("lines of white powder",     "drugs",   0.80),
    ("smoking bong",              "drugs",   0.75),
    # violence / weapons
    ("handgun",                   "violence", 0.85),
    ("pistol",                    "violence", 0.85),
    ("rifle",                     "violence", 0.85),
    ("assault rifle",             "violence", 0.85),
    ("knife as weapon",           "violence", 0.65),
    ("blood on person",           "violence", 0.60),
    # ludomania
    ("playing cards",             "ludomania", 0.55),
    ("casino chips",              "ludomania", 0.80),
    ("roulette wheel",            "ludomania", 0.85),
    ("slot machine",              "ludomania", 0.85),
    ("poker table",               "ludomania", 0.70),
    # vandalism
    ("graffiti on wall",          "vandalism", 0.70),
    ("spray paint can",           "vandalism", 0.55),
    ("broken window",             "vandalism", 0.55),
    # extremism
    ("nazi swastika symbol",      "extremism", 0.80),
    ("extremist flag",            "extremism", 0.65),
    # nsfw
    ("naked person",              "nsfw",      0.70),
    ("nude body",                 "nsfw",      0.70),
    ("person in lingerie",        "nsfw",      0.55),
    # selfharm
    ("hanging noose rope",        "selfharm",  0.70),
    ("cutting wrist with blade",  "selfharm",  0.70),
    ("blood on wrist",            "selfharm",  0.55),
    # animal cruelty
    ("person hitting an animal",  "animal_cruelty", 0.65),
    ("injured animal",            "animal_cruelty", 0.55),
]

_WORLD_PROMPTS = [p for p, _, _ in _WORLD_VOCAB]
_WORLD_IDX_TO_SUBCLASS: dict[int, tuple[str, float]] = {
    i: (sub, prior) for i, (_, sub, prior) in enumerate(_WORLD_VOCAB)
}

_models: dict[str, object] = {}
_world_models: dict[str, object] = {}
_lock = threading.Lock()


def _resolve_weights(model_name: str) -> str:
    """Ищет локальные веса (корень проекта, затем MODELS_DIR/ultralytics).
    Если найдены — возвращает путь (ultralytics НЕ качает). Иначе — голое имя
    (ultralytics скачает один раз; в offline это уже не понадобится)."""
    from .config import ULTRALYTICS_WEIGHTS_DIRS
    for d in ULTRALYTICS_WEIGHTS_DIRS:
        p = d / model_name
        if p.exists():
            return str(p)
    return model_name


def _get_yolo(model_name: str):
    with _lock:
        if model_name not in _models:
            from ultralytics import YOLO
            _models[model_name] = YOLO(_resolve_weights(model_name))
    return _models[model_name]


def _get_yolo_world(model_name: str):
    with _lock:
        if model_name not in _world_models:
            from ultralytics import YOLOWorld
            m = YOLOWorld(_resolve_weights(model_name))
            m.set_classes(_WORLD_PROMPTS)
            _world_models[model_name] = m
    return _world_models[model_name]


def _calib(yolo_conf: float, prior: float) -> float:
    """Калибровка: взвешенное среднее score детектора и приора класса."""
    return round(min(0.98, 0.6 * yolo_conf + 0.4 * prior), 3)


def _make_det(kf: Keyframe, fps: float, subclass: str, conf: float) -> Detection:
    t0 = max(0.0, kf.time_sec - 0.5)
    t1 = kf.time_sec + 1.0
    return Detection(
        startFrame=int(t0 * fps),
        endFrame=int(t1 * fps),
        start_time=_fmt_clock(t0),
        end_time=_fmt_clock(t1),
        time_interval=f"{_fmt_clock(t0)} - {_fmt_clock(t1)}",
        subclass=Subclass(subclass),
        confidence=conf,
        type=DetectionType.video,
    )


def yolo_keyframes(
    keyframes: list[Keyframe],
    fps: float,
    preset: Preset | str | None = None,
    conf_threshold: float | None = None,
    model_name: str | None = None,
) -> list[Detection]:
    """Детектирует объекты на keyframe-ах двумя детекторами (COCO + open-vocab)."""
    p = preset if isinstance(preset, Preset) else get_preset(preset)
    coco_model_name = model_name or p.yolo_model
    conf = conf_threshold if conf_threshold is not None else p.yolo_conf
    imgsz = p.yolo_imgsz

    detections: list[Detection] = []
    batch_size = 16

    # ── 1. Классический COCO-детектор ─────────────────────────────────────────
    coco = _get_yolo(coco_model_name)
    for i in range(0, len(keyframes), batch_size):
        batch = keyframes[i:i + batch_size]
        images = [np.array(kf.image) for kf in batch]
        with gpu_guard():
            results = coco(images, conf=conf, imgsz=imgsz, device=DEVICE, verbose=False)
        for kf, result in zip(batch, results):
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in _COCO_TO_SUBCLASS:
                    continue
                sub, prior = _COCO_TO_SUBCLASS[cls_id]
                detections.append(_make_det(kf, fps, sub, _calib(float(box.conf[0]), prior)))

    # ── 2. YOLO-World (open-vocabulary) ───────────────────────────────────────
    if p.yolo_world_model:
        try:
            world = _get_yolo_world(p.yolo_world_model)
            for i in range(0, len(keyframes), batch_size):
                batch = keyframes[i:i + batch_size]
                images = [np.array(kf.image) for kf in batch]
                with gpu_guard():
                    results = world(images, conf=conf, imgsz=imgsz, device=DEVICE, verbose=False)
                for kf, result in zip(batch, results):
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id not in _WORLD_IDX_TO_SUBCLASS:
                            continue
                        sub, prior = _WORLD_IDX_TO_SUBCLASS[cls_id]
                        detections.append(_make_det(kf, fps, sub, _calib(float(box.conf[0]), prior)))
        except Exception:
            # YOLO-World недоступен (старый ultralytics / нет весов) — не валим пайплайн
            pass

    return detections
