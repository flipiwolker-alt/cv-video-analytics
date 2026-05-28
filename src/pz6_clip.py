"""PZ6: Zero-shot классификация сцены через CLIP.

Совместим с transformers 4.x и 5.x (обе версии меняют возврат get_text_features).
Модель: openai/clip-vit-base-patch32 (~150 MB, скачивается один раз).
"""
from __future__ import annotations

import threading
from pathlib import Path

import torch

from .pz4_audio import _fmt_clock
from .schemas import Detection, DetectionType, Subclass
from .pz1_keyframes import Keyframe

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
    ],
    "violence": [
        "violent fight people hitting each other",
        "person being attacked or beaten up",
        "graphic violence blood injury",
        "people fighting brawl street",
    ],
    "vandalism": [
        "graffiti spray paint on walls",
        "property destruction broken windows",
        "vandalism arson fire building",
        "smashing breaking public property",
    ],
    "profanity": [
        "person shouting and swearing aggressively",
        "angry person using offensive language",
        "screen showing profanity or obscene text",
        "subtitle or caption with swear words",
    ],
}

# Нейтральные промпты — baseline для отсечки ложных срабатываний.
# Кадр помечается только если его сходство с классом ПРЕВЫШАЕТ нейтральный
# baseline на величину _MARGIN, а не просто перекрывает абсолютный порог.
_NEUTRAL_PROMPTS: list[str] = [
    "a normal everyday scene",
    "people walking on a street",
    "a person talking to camera",
    "indoor scene people sitting",
    "outdoor landscape nature",
]

# Минимальный абсолютный порог — ниже него даже «победитель» не считается
_SIM_THRESHOLD = 0.22
# Минимальное превышение над нейтральным baseline
_NEUTRAL_MARGIN = 0.03

_clip_model = None
_clip_processor = None
_text_features: dict[str, torch.Tensor] = {}
_neutral_features: torch.Tensor | None = None
_clip_lock = threading.Lock()


def _to_tensor(output) -> torch.Tensor:
    """Извлекает тензор из вывода CLIP — совместимо с transformers 4.x и 5.x."""
    if isinstance(output, torch.Tensor):
        return output
    # transformers 5.x: CLIPTextModelOutput / CLIPVisionModelOutput
    for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(output, attr, None)
        if val is not None and isinstance(val, torch.Tensor):
            # pooler_output и last_hidden_state — берём CLS-токен если 3D
            return val[:, 0] if val.dim() == 3 else val
    raise TypeError(f"Не удалось извлечь тензор из {type(output).__name__}")


def _init_clip():
    """Thread-safe инициализация CLIP. Вызывается один раз."""
    global _clip_model, _clip_processor, _text_features, _neutral_features

    if _clip_model is not None:
        return

    with _clip_lock:
        if _clip_model is not None:  # double-checked locking
            return

        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(model_id)
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model.eval()

        # Предвычисляем эмбеддинги всех текстовых промптов один раз
        all_texts = [t for prompts in _PROMPTS.values() for t in prompts]
        tok = _clip_processor(text=all_texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            raw = _clip_model.get_text_features(**tok)
            feats = _to_tensor(raw)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        idx = 0
        for cls, prompts in _PROMPTS.items():
            n = len(prompts)
            _text_features[cls] = feats[idx:idx + n]
            idx += n

        # Предвычисляем нейтральный baseline — mean-pooled эмбеддинг нейтральных сцен
        ntok = _clip_processor(text=_NEUTRAL_PROMPTS, return_tensors="pt", padding=True)
        with torch.no_grad():
            nraw = _clip_model.get_text_features(**ntok)
            nfeats = _to_tensor(nraw)
            nfeats = nfeats / nfeats.norm(dim=-1, keepdim=True)
        _neutral_features = nfeats  # shape (n_neutral, D)


def clip_keyframes(
    keyframes: list[Keyframe],
    fps: float,
    batch_size: int = 8,
    threshold: float = _SIM_THRESHOLD,
    max_frames: int = 80,
) -> list[Detection]:
    """Прогоняет CLIP zero-shot по keyframe-ам, возвращает детекции."""
    if len(keyframes) > max_frames:
        step = max(1, len(keyframes) // max_frames)
        keyframes = keyframes[::step][:max_frames]

    _init_clip()
    detections: list[Detection] = []

    for i in range(0, len(keyframes), batch_size):
        batch = keyframes[i:i + batch_size]
        images = [kf.image for kf in batch]

        pix = _clip_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            raw_img = _clip_model.get_image_features(**pix)
            img_feats = _to_tensor(raw_img)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        # Нейтральный baseline: максимальное сходство с нейтральными промптами
        neutral_sims = img_feats @ _neutral_features.T   # (batch, n_neutral)
        neutral_scores = neutral_sims.max(dim=-1).values  # (batch,)

        for (kf, img_feat), neutral_score in zip(zip(batch, img_feats), neutral_scores):
            ns = float(neutral_score)
            for subclass, text_feats in _text_features.items():
                sims = img_feat @ text_feats.T      # (n_prompts,)
                max_sim = float(sims.max())

                # Пропускаем если не превышает абсолютный порог ИЛИ нейтральный baseline
                if max_sim < threshold:
                    continue
                if max_sim < ns + _NEUTRAL_MARGIN:
                    continue

                # Масштабируем similarity → confidence [0.50, 0.98]
                conf = 0.50 + (max_sim - threshold) / max(0.40 - threshold, 0.01) * 0.48
                conf = round(min(conf, 0.98), 3)

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
