"""Разовая загрузка ВСЕХ моделей в локальный кэш (E:\\cv_models по умолчанию).

Запусти ОДИН раз после установки зависимостей:

    python scripts/download_models.py            # модели всех пресетов
    python scripts/download_models.py --preset balanced   # только один пресет

После успеха ставится маркер <cache>/.models_ready, и обычные запуски
(run_ui.py / run_api.py) переходят в offline-режим: HuggingFace больше не
ходит в сеть на проверку ревизий — анализ стартует мгновенно из кэша.

Повторный запуск ничего не качает заново — только то, чего не хватает.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Качаем — значит сеть нужна: принудительно online на время скачивания.
os.environ["CV_MODELS_OFFLINE"] = "0"

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODELS_DIR, MODELS_READY_FLAG  # noqa: E402  (ставит кэш-пути)
from src.presets import PRESETS                        # noqa: E402

# ── Что качаем ────────────────────────────────────────────────────────────────
# HuggingFace-репозитории сцены/NSFW/действий + CLIP-fallback.
_HF_SCENE = {
    "fast":     ["google/siglip2-base-patch16-224"],
    "balanced": ["google/siglip2-base-patch16-384"],
    "accurate": ["google/siglip2-so400m-patch14-384"],
}
_HF_ALWAYS = [
    "Falconsai/nsfw_image_detection",   # NSFW (pz7)
    "microsoft/xclip-base-patch32",     # действия (pz8)
    "openai/clip-vit-base-patch32",     # fallback сцены
]
# Whisper-размеры по пресетам (faster-whisper сам резолвит repo_id).
_WHISPER = {"fast": "small", "balanced": "large-v3-turbo", "accurate": "large-v3"}


def _hf(repo_id: str) -> None:
    from huggingface_hub import snapshot_download
    print(f"  HF: {repo_id} ...", flush=True)
    snapshot_download(repo_id)


def _yolo(name: str) -> None:
    from ultralytics import YOLO, YOLOWorld
    print(f"  YOLO: {name} ...", flush=True)
    dst = MODELS_DIR / "ultralytics" / name
    (MODELS_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
    cls = YOLOWorld if "world" in name.lower() else YOLO
    m = cls(name)  # качает .pt в CWD при отсутствии
    # переносим веса в общий кэш, если ultralytics положил их в корень
    src = Path(name)
    if src.exists() and not dst.exists():
        try:
            dst.write_bytes(src.read_bytes())
        except Exception:
            pass


def _whisper(size: str) -> None:
    from faster_whisper import WhisperModel
    print(f"  Whisper: {size} ...", flush=True)
    WhisperModel(size, device="cpu", compute_type="int8")  # триггерит скачивание


def _easyocr() -> None:
    import easyocr
    print("  EasyOCR: ru+en ...", flush=True)
    easyocr.Reader(["ru", "en"], gpu=False, verbose=False,
                   model_storage_directory=str(MODELS_DIR / "easyocr"),
                   download_enabled=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS) + ["all"], default="all")
    args = ap.parse_args()
    names = list(PRESETS) if args.preset == "all" else [args.preset]

    print(f"Кэш моделей: {MODELS_DIR}")
    print(f"Качаю для пресетов: {', '.join(names)}\n")

    print("[1/5] Сцена (SigLIP/CLIP):")
    for n in names:
        for repo in _HF_SCENE[n]:
            _hf(repo)
    print("[2/5] NSFW + действия + CLIP-fallback:")
    for repo in _HF_ALWAYS:
        _hf(repo)
    print("[3/5] Whisper:")
    for n in names:
        _whisper(_WHISPER[n])
    print("[4/5] YOLO (COCO + open-vocab):")
    for n in names:
        p = PRESETS[n]
        _yolo(p.yolo_model)
        if p.yolo_world_model:
            _yolo(p.yolo_world_model)
    print("[5/5] EasyOCR:")
    _easyocr()

    MODELS_READY_FLAG.write_text("ready\n", encoding="utf-8")
    print(f"\nГотово. Маркер: {MODELS_READY_FLAG}")
    print("Теперь run_ui.py / run_api.py работают offline (из кэша, без сети).")


if __name__ == "__main__":
    main()
