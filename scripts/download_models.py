"""Скачивает все модели всех пресетов заранее.

Запуск:
    python scripts/download_models.py            # все пресеты
    python scripts/download_models.py --preset balanced  # только balanced
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Гарантируем что src.config устанавливает переменные окружения до любых импортов ML
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODELS_DIR  # noqa: E402 — намеренно после sys.path

print(f"📁 Папка моделей: {MODELS_DIR}\n")

parser = argparse.ArgumentParser()
parser.add_argument("--preset", choices=["fast", "balanced", "accurate", "all"], default="all")
args = parser.parse_args()

PRESETS = {
    "fast": {
        "yolo":        ["yolo11s.pt"],
        "yolo_world":  ["yolov8s-worldv2.pt"],
        "hf":          ["google/siglip2-base-patch16-224"],
        "whisper":     ["small"],
    },
    "balanced": {
        "yolo":        ["yolo11m.pt"],
        "yolo_world":  ["yolov8m-worldv2.pt"],
        "hf":          ["google/siglip2-base-patch16-384"],
        "whisper":     ["large-v3-turbo"],
    },
    "accurate": {
        "yolo":        ["yolo11x.pt"],
        "yolo_world":  ["yolov8x-worldv2.pt"],
        "hf":          ["google/siglip2-so400m-patch14-384"],
        "whisper":     ["large-v3"],
    },
}

# Общие модели для всех пресетов
COMMON_HF = [
    "Falconsai/nsfw_image_detection",   # ~340 MB
    "microsoft/xclip-base-patch32",     # ~400 MB
]

selected = list(PRESETS.keys()) if args.preset == "all" else [args.preset]

yolo_models      = []
yolo_world_models = []
hf_models        = list(COMMON_HF)
whisper_models   = []

for p in selected:
    cfg = PRESETS[p]
    yolo_models      += cfg["yolo"]
    yolo_world_models += cfg["yolo_world"]
    hf_models        += cfg["hf"]
    whisper_models   += cfg["whisper"]

# Дедупликация
yolo_models       = list(dict.fromkeys(yolo_models))
yolo_world_models = list(dict.fromkeys(yolo_world_models))
hf_models         = list(dict.fromkeys(hf_models))
whisper_models    = list(dict.fromkeys(whisper_models))

# ── 1. YOLO (Ultralytics) ─────────────────────────────────────────────────────
print("=" * 60)
print("1/4  YOLO-модели (Ultralytics)")
print("=" * 60)
from ultralytics import YOLO, YOLOWorld  # noqa: E402

for name in yolo_models:
    dest = MODELS_DIR / "ultralytics" / name
    if dest.exists():
        print(f"  ✅  {name} — уже есть")
        continue
    print(f"  ⬇️   {name} ...")
    m = YOLO(name)
    # ultralytics сохраняет в cwd, перекладываем в MODELS_DIR
    local_pt = Path(name)
    if local_pt.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        local_pt.rename(dest)
    print(f"  ✅  {name} → {dest}")

for name in yolo_world_models:
    dest = MODELS_DIR / "ultralytics" / name
    if dest.exists():
        print(f"  ✅  {name} — уже есть")
        continue
    print(f"  ⬇️   {name} ...")
    m = YOLOWorld(name)
    local_pt = Path(name)
    if local_pt.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        local_pt.rename(dest)
    print(f"  ✅  {name} → {dest}")

# ── 2. HuggingFace (SigLIP2, NSFW, X-CLIP) ───────────────────────────────────
print()
print("=" * 60)
print("2/4  HuggingFace-модели")
print("=" * 60)
from huggingface_hub import snapshot_download  # noqa: E402

for model_id in hf_models:
    cache_dir = MODELS_DIR / "huggingface" / "hub"
    # Проверяем наличие по имени папки (huggingface кэш: models--org--name)
    slug = "models--" + model_id.replace("/", "--")
    if (cache_dir / slug).exists():
        print(f"  ✅  {model_id} — уже есть")
        continue
    print(f"  ⬇️   {model_id} ...")
    snapshot_download(repo_id=model_id, cache_dir=str(cache_dir))
    print(f"  ✅  {model_id}")

# ── 3. Whisper (faster-whisper / CTranslate2) ────────────────────────────────
print()
print("=" * 60)
print("3/4  Whisper-модели (faster-whisper)")
print("=" * 60)
from faster_whisper import WhisperModel  # noqa: E402
import os

whisper_cache = str(MODELS_DIR / "whisper")
os.makedirs(whisper_cache, exist_ok=True)

for size in whisper_models:
    # faster-whisper кэширует по имени модели
    model_dir = Path(whisper_cache) / f"models--Systran--faster-whisper-{size}"
    if model_dir.exists():
        print(f"  ✅  whisper/{size} — уже есть")
        continue
    print(f"  ⬇️   whisper/{size} ...")
    WhisperModel(size, device="cpu", download_root=whisper_cache)
    print(f"  ✅  whisper/{size}")

# ── 4. EasyOCR ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("4/4  EasyOCR (ru + en)")
print("=" * 60)
try:
    import easyocr  # noqa: E402
    ocr_dir = str(MODELS_DIR / "easyocr")
    if Path(ocr_dir, "craft_mlt_25k.pth").exists():
        print("  ✅  EasyOCR — уже есть")
    else:
        print("  ⬇️   EasyOCR ru+en ...")
        easyocr.Reader(["ru", "en"], gpu=False, model_storage_directory=ocr_dir, verbose=False)
        print("  ✅  EasyOCR")
except Exception as e:
    print(f"  ⚠️   EasyOCR пропущен (OCR не обязателен): {e}")

print()
print("=" * 60)
print("🎉  Все модели скачаны! Можно запускать python run_ui.py")
print("=" * 60)
