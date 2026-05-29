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
import threading

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
