"""Скачивает ВСЕ модели всех пресетов заранее — параллельно где возможно.

После этого скрипта анализ запускается без какого-либо скачивания.

Запуск:
    python scripts/download_models.py            # все пресеты (по умолчанию)
    python scripts/download_models.py --preset balanced
"""
from __future__ import annotations
import argparse
import sys
import os
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODELS_DIR  # noqa: E402

print(f"📁 Папка моделей: {MODELS_DIR}\n")

parser = argparse.ArgumentParser()
parser.add_argument("--preset", choices=["fast", "balanced", "accurate", "all"], default="all")
args = parser.parse_args()

PRESETS = {
    "fast": {
        "yolo":       ["yolo11s.pt"],
        "yolo_world": ["yolov8s-worldv2.pt"],
        "hf":         ["google/siglip2-base-patch16-224"],
        "whisper":    ["small"],
    },
    "balanced": {
        "yolo":       ["yolo11m.pt"],
        "yolo_world": ["yolov8m-worldv2.pt"],
        "hf":         ["google/siglip2-base-patch16-384"],
        "whisper":    ["large-v3-turbo"],
    },
    "accurate": {
        "yolo":       ["yolo11x.pt"],
        "yolo_world": ["yolov8x-worldv2.pt"],
        "hf":         ["google/siglip2-so400m-patch14-384"],
        "whisper":    ["large-v3"],
    },
}

# Общие модели для всех пресетов
COMMON_HF = [
    "Falconsai/nsfw_image_detection",   # ~340 MB
    "microsoft/xclip-base-patch32",     # ~400 MB
]

selected = list(PRESETS.keys()) if args.preset == "all" else [args.preset]

yolo_models       = []
yolo_world_models = []
hf_models         = list(COMMON_HF)
whisper_models    = []

for p in selected:
    cfg = PRESETS[p]
    yolo_models       += cfg["yolo"]
    yolo_world_models += cfg["yolo_world"]
    hf_models         += cfg["hf"]
    whisper_models    += cfg["whisper"]

# Дедупликация
yolo_models       = list(dict.fromkeys(yolo_models))
yolo_world_models = list(dict.fromkeys(yolo_world_models))
hf_models         = list(dict.fromkeys(hf_models))
whisper_models    = list(dict.fromkeys(whisper_models))

_print_lock = threading.Lock()

def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


# ── 1. YOLO (Ultralytics) ─────────────────────────────────────────────────────
def download_yolo(name: str, world: bool = False):
    dest = MODELS_DIR / "ultralytics" / name
    if dest.exists():
        log(f"  ✅  {name} — уже есть")
        return
    log(f"  ⬇️   {name} ...")
    if world:
        from ultralytics import YOLOWorld
        YOLOWorld(name)
    else:
        from ultralytics import YOLO
        YOLO(name)
    local_pt = Path(name)
    if local_pt.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        local_pt.rename(dest)
    log(f"  ✅  {name} → {dest}")


# ── 2. HuggingFace ────────────────────────────────────────────────────────────
def download_hf(model_id: str):
    from huggingface_hub import snapshot_download
    cache_dir = MODELS_DIR / "huggingface" / "hub"
    slug = "models--" + model_id.replace("/", "--")
    if (cache_dir / slug).exists():
        log(f"  ✅  {model_id} — уже есть")
        return
    log(f"  ⬇️   {model_id} ...")
    snapshot_download(repo_id=model_id, cache_dir=str(cache_dir))
    log(f"  ✅  {model_id}")


# ── 3. Whisper ────────────────────────────────────────────────────────────────
def download_whisper(size: str):
    whisper_cache = str(MODELS_DIR / "whisper")
    os.makedirs(whisper_cache, exist_ok=True)
    model_dir = Path(whisper_cache) / f"models--Systran--faster-whisper-{size}"
    if model_dir.exists():
        log(f"  ✅  whisper/{size} — уже есть")
        return
    log(f"  ⬇️   whisper/{size} ...")
    from faster_whisper import WhisperModel
    WhisperModel(size, device="cpu", download_root=whisper_cache)
    log(f"  ✅  whisper/{size}")


# ── 4. OpenAI CLIP (ViT-B/32) ────────────────────────────────────────────────
def download_openai_clip():
    import torch
    clip_cache = Path.home() / ".cache" / "clip"
    clip_file = clip_cache / "ViT-B-32.pt"
    # Проверяем что файл есть и не повреждён
    if clip_file.exists() and clip_file.stat().st_size > 300_000_000:
        log("  ✅  OpenAI CLIP ViT-B/32 — уже есть")
        return
    log("  ⬇️   OpenAI CLIP ViT-B/32 (~338 МБ) ...")
    clip_cache.mkdir(parents=True, exist_ok=True)
    url = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
    import urllib.request
    def progress(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size)
        if pct % 10 == 0:
            mb = count * block_size / 1024 / 1024
            log(f"    CLIP: {pct}% ({mb:.0f}/{total_size//1024//1024} МБ)")
    urllib.request.urlretrieve(url, str(clip_file), reporthook=progress)
    log(f"  ✅  OpenAI CLIP ViT-B/32 → {clip_file}")


# ── 5. EasyOCR ────────────────────────────────────────────────────────────────
def download_easyocr():
    try:
        import easyocr
        ocr_dir = str(MODELS_DIR / "easyocr")
        if Path(ocr_dir, "craft_mlt_25k.pth").exists():
            log("  ✅  EasyOCR — уже есть")
            return
        log("  ⬇️   EasyOCR ru+en ...")
        easyocr.Reader(["ru", "en"], gpu=False, model_storage_directory=ocr_dir, verbose=False)
        log("  ✅  EasyOCR")
    except Exception as e:
        log(f"  ⚠️   EasyOCR пропущен: {e}")


# ── Запуск параллельно ────────────────────────────────────────────────────────
print("=" * 60)
print("Скачиваем все модели параллельно...")
print("=" * 60)

tasks = []

# YOLO последовательно (ультралитики не любит параллельность)
for name in yolo_models:
    tasks.append(("YOLO", lambda n=name: download_yolo(n, world=False)))
for name in yolo_world_models:
    tasks.append(("YOLO-World", lambda n=name: download_yolo(n, world=True)))

# HuggingFace, Whisper, CLIP, EasyOCR — параллельно
hf_tasks  = [("HF", lambda m=m: download_hf(m)) for m in hf_models]
wh_tasks  = [("Whisper", lambda s=s: download_whisper(s)) for s in whisper_models]
misc_tasks = [
    ("CLIP",    download_openai_clip),
    ("EasyOCR", download_easyocr),
]

# YOLO качаем первыми (sequential)
print("\n[1/3] YOLO-модели:")
for label, fn in tasks:
    fn()

# Остальное — параллельно
print("\n[2/3] HuggingFace + Whisper + CLIP + EasyOCR (параллельно):")
parallel = hf_tasks + wh_tasks + misc_tasks
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(fn): label for label, fn in parallel}
    for f in as_completed(futures):
        try:
            f.result()
        except Exception as e:
            log(f"  ❌  {futures[f]}: {e}")

print()
print("=" * 60)
print("🎉  Все модели скачаны! Можно запускать python run_ui.py")
print("=" * 60)
