"""PZ6: Zero-shot классификация сцены.

Бэкенд выбирается по имени модели в пресете:
  • SigLIP 2 (google/siglip2-*)  — по умолчанию, точнее и меньше ложняков;
  • CLIP (openai/clip-*)         — fallback / совместимость.

Точность:
  • Ансамбль промптов: скор класса = максимум сходства по его формулировкам.
  • Калиброванная уверенность через нейтральный baseline: вероятность =
    сигмоида от разницы «сходство с классом − сходство с нейтральной сценой».
    Относительная схема устойчива к тому, что абсолютные сходства «плавают».
  • Строгий порог вероятности — режет пограничный мусор (0.5x).

Совместимо с transformers 4.x/5.x.
"""
from __future__ import annotations

import math
import threading

import torch

from .pz4_audio import _fmt_clock
from .schemas import Detection, DetectionType, Subclass
from .pz1_keyframes import Keyframe
from .presets import Preset, get_preset
from .device import DEVICE, gpu_guard

_PROMPTS: dict[str, list[str]] = {
    "alcohol": [
        "a person drinking alcohol",
        "beer bottle or wine glass on table",
        "people drinking at a bar or party",
        "alcoholic beverages whiskey vodka rum",
    ],
    "drugs": [
        "a person using illegal drugs",
        "drug paraphernalia syringe needle",
        "someone snorting powder cocaine",
        "illegal substance pills drugs",
    ],
    "smoking": [
        "a person smoking a cigarette",
        "someone vaping or smoking tobacco",
        "cigarette smoke in mouth",
        "person holding a lit cigarette",
    ],
    "extremism": [
        "extremist propaganda symbols",
        "nazi swastika symbol hate",
        "terrorist propaganda violent rally",
        "extremist ideology symbols flags",
    ],
    "ludomania": [
        "casino slot machines gambling hall",
        "people playing poker roulette cards",
        "sports betting gambling screen",
        "casino chips cards gambling table",
    ],
    "lgbt": [
        "LGBT propaganda aimed at children",
        "same sex couple targeting minors propaganda",
        "rainbow pride flag activism",
    ],
    "violence": [
        "violent fight people hitting each other",
        "person being attacked or beaten up",
        "graphic violence blood injury",
        "people fighting brawl street",
        "person pointing a gun weapon",
    ],
    "vandalism": [
        "graffiti spray paint on walls",
        "property destruction broken windows",
        "vandalism arson fire building",
        "smashing breaking public property",
    ],
    "profanity": [
        "screen showing profanity or obscene text",
        "subtitle or caption with swear words",
        "offensive obscene writing on screen",
    ],
    "nsfw": [
        "explicit sexual content nudity",
        "naked person exposed body",
        "pornographic adult scene",
        "person in underwear suggestive pose",
    ],
    "selfharm": [
        "person cutting their wrist self harm",
        "suicide attempt hanging from a rope noose",
        "scars and cuts on arm from self harm",
        "person standing on the edge of a high building",
    ],
    "animal_cruelty": [
        "a person beating or hitting an animal",
        "injured abused animal in distress",
        "cruelty and violence towards an animal",
        "a dead or mistreated animal",
    ],
}

# Нейтральные промпты — baseline «обычной» сцены.
_NEUTRAL_PROMPTS: list[str] = [
    "a normal everyday scene",
    "people walking on a street",
    "a person talking to camera",
    "indoor scene people sitting",
    "outdoor landscape nature",
    "a regular conversation between people",
    "ordinary household objects",
    "a music performance on stage",
    "a person sitting at a desk",
]

# Масштаб сигмоиды (разница сходств мала → большой масштаб).
_LOGIT_SCALE = 50.0
# Строгий порог вероятности для срабатывания (поднят, чтобы убрать пограничный мусор).
_PROB_THRESHOLD = 0.72
# Минимальный разрыв с нейтральным baseline (абсолютный, в единицах cosine).
_MIN_GAP = 0.015
# Абсолютный пол сходства по бэкендам (у SigLIP cosine ниже, чем у CLIP).
_SIM_FLOOR = {"clip": 0.20, "siglip": 0.05}

_models: dict[str, dict] = {}
_lock = threading.Lock()


def _to_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(output, attr, None)
        if val is not None and isinstance(val, torch.Tensor):
            return val[:, 0] if val.dim() == 3 else val
    raise TypeError(f"Не удалось извлечь тензор из {type(output).__name__}")


def _backend(model_id: str) -> str:
    return "siglip" if "siglip" in model_id.lower() else "clip"


def _init(model_id: str) -> dict:
    """Thread-safe загрузка модели и предвычисление текстовых эмбеддингов."""
    with _lock:
        if model_id in _models:
            return _models[model_id]

        backend = _backend(model_id)
        if backend == "siglip":
            from transformers import AutoModel, AutoProcessor
            model = AutoModel.from_pretrained(model_id)
            processor = AutoProcessor.from_pretrained(model_id)
            text_kwargs = dict(padding="max_length", max_length=64, truncation=True)
        else:
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained(model_id)
            processor = CLIPProcessor.from_pretrained(model_id)
            text_kwargs = dict(padding=True)
        model.eval().to(DEVICE)

        def _encode_text(texts: list[str]) -> torch.Tensor:
            tok = processor(text=texts, return_tensors="pt", **text_kwargs).to(DEVICE)
            with torch.no_grad(), gpu_guard():
                feats = _to_tensor(model.get_text_features(**tok))
            return feats / feats.norm(dim=-1, keepdim=True)

        all_texts = [t for prompts in _PROMPTS.values() for t in prompts]
        feats = _encode_text(all_texts)
        text_features: dict[str, torch.Tensor] = {}
        idx = 0
        for cls, prompts in _PROMPTS.items():
            n = len(prompts)
            text_features[cls] = feats[idx:idx + n]
            idx += n

        bundle = {
            "backend": backend,
            "model": model,
            "processor": processor,
            "text_features": text_features,
            "neutral_features": _encode_text(_NEUTRAL_PROMPTS),
            "floor": _SIM_FLOOR[backend],
        }
        _models[model_id] = bundle
        return bundle


def clip_keyframes(
    keyframes: list[Keyframe],
    fps: float,
    preset: Preset | str | None = None,
    batch_size: int = 8,
    max_frames: int | None = None,
) -> list[Detection]:
    """Zero-shot классификация сцены по keyframe-ам (SigLIP 2 / CLIP)."""
    p = preset if isinstance(preset, Preset) else get_preset(preset)
    model_id = p.clip_model
    cap = max_frames if max_frames is not None else p.clip_max_frames

    if len(keyframes) > cap:
        step = max(1, len(keyframes) // cap)
        keyframes = keyframes[::step][:cap]

    bundle = _init(model_id)
    model = bundle["model"]
    processor = bundle["processor"]
    text_features = bundle["text_features"]
    neutral_features = bundle["neutral_features"]
    floor = bundle["floor"]

    detections: list[Detection] = []

    for i in range(0, len(keyframes), batch_size):
        batch = keyframes[i:i + batch_size]
        images = [kf.image for kf in batch]

        pix = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad(), gpu_guard():
            img_feats = _to_tensor(model.get_image_features(**pix))
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        neutral_scores = (img_feats @ neutral_features.T).max(dim=-1).values  # (batch,)

        for (kf, img_feat), neutral_score in zip(zip(batch, img_feats), neutral_scores):
            ns = float(neutral_score)
            for subclass, feats in text_features.items():
                max_sim = float((img_feat @ feats.T).max())
                gap = max_sim - ns
                if max_sim < floor or gap < _MIN_GAP:
                    continue
                prob = 1.0 / (1.0 + math.exp(-_LOGIT_SCALE * gap))
                if prob < _PROB_THRESHOLD:
                    continue
                conf = round(min(0.98, 0.5 + (prob - _PROB_THRESHOLD) /
                                 (1.0 - _PROB_THRESHOLD) * 0.48), 3)

                t0 = max(0.0, kf.time_sec - 0.5)
                t1 = kf.time_sec + 1.5
                detections.append(Detection(
                    startFrame=int(t0 * fps),
                    endFrame=int(t1 * fps),
                    start_time=_fmt_clock(t0),
                    end_time=_fmt_clock(t1),
                    time_interval=f"{_fmt_clock(t0)} - {_fmt_clock(t1)}",
                    subclass=Subclass(subclass),
                    confidence=conf,
                    type=DetectionType.video,
                ))

    return detections
