"""Единый выбор устройства (GPU/CPU) и сериализация доступа к GPU.

Поддерживает три бэкенда:
  • CUDA  — NVIDIA (Windows/Linux, арендованный сервер, Kaggle);
  • MPS   — Apple Silicon (M1–M4), маковский GPU через Metal;
  • CPU   — всё остальное.

На GPU с ограниченной/единой памятью (8 ГБ VRAM или unified memory Mac) все
модели сразу не помещаются, а стадии пайплайна идут параллельно в потоках.
Поэтому тяжёлые GPU-операции сериализуются ОДНИМ локом: GPU настолько быстрее
CPU, что даже последовательное выполнение кратно выигрывает. На CPU gpu_guard()
— no-op (стадии по-прежнему параллельны).
"""
from __future__ import annotations

import contextlib
import os
import threading

# 8 ГБ VRAM — память делят 6 моделей. expandable_segments снижает фрагментацию
# и риск OOM при параллельных стадиях. Ставим ДО импорта torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
    _CUDA = bool(torch.cuda.is_available())
    try:
        _MPS = (not _CUDA) and bool(torch.backends.mps.is_available())
    except Exception:
        _MPS = False
except Exception:
    _CUDA = _MPS = False

if _CUDA:
    DEVICE = "cuda"
elif _MPS:
    DEVICE = "mps"
    # Неподдержанные MPS-операции уводим на CPU, чтобы ничего не падало на Маке
    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
else:
    DEVICE = "cpu"

ON_CUDA = _CUDA                              # для easyocr.gpu и float16
ON_GPU = _CUDA or _MPS                       # сериализуем доступ к любой GPU-памяти
WHISPER_DEVICE = "cuda" if _CUDA else "cpu"  # faster-whisper (CTranslate2) не умеет mps

_GPU_LOCK = threading.Lock()


def gpu_guard():
    """Контекст-менеджер: на GPU сериализует доступ, на CPU — пустышка."""
    return _GPU_LOCK if ON_GPU else contextlib.nullcontext()


def empty_cache() -> None:
    """Освобождает неиспользуемую VRAM (вызывать между тяжёлыми прогонами)."""
    if _CUDA:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def describe() -> str:
    """Человекочитаемая строка про выбранное устройство (для баннера при старте)."""
    if _CUDA:
        try:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"GPU (CUDA): {name}, {total:.0f} ГБ VRAM"
        except Exception:
            return "GPU (CUDA)"
    if _MPS:
        return "GPU (Apple MPS)"
    return "CPU (CUDA недоступна — будет медленно)"


def banner() -> str:
    """Баннер устройства + режима моделей для печати при запуске UI/API."""
    try:
        from .config import OFFLINE, MODELS_DIR
        mode = "offline (из кэша, без сети)" if OFFLINE else "online (докачка при отсутствии)"
        return (f"[cv-video-analytics] Вычисления: {describe()}\n"
                f"[cv-video-analytics] Модели: {mode} — {MODELS_DIR}")
    except Exception:
        return f"[cv-video-analytics] Вычисления: {describe()}"
