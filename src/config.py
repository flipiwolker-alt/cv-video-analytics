import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Папка для хранения всех моделей ──────────────────────────────────────────
# Приоритет: 1) переменная окружения CV_MODELS_DIR
#             2) E:\cv_models / C:\Users\User\cv_models  (Windows)
#             3) ~/.cache/cv_models  (Mac/Linux/Kaggle — кроссплатформенно)
#             4) <корень проекта>/models  (последний фолбэк)

def _can_write(p: Path) -> bool:
    """Проверяет реальную возможность записи: пытается создать тестовую подпапку."""
    test = p / ".write_test"
    try:
        test.mkdir(parents=True, exist_ok=True)
        test.rmdir()
        return True
    except (PermissionError, OSError):
        return False


def _resolve_models_dir() -> Path:
    candidates = []
    if "CV_MODELS_DIR" in os.environ:
        candidates.append(Path(os.environ["CV_MODELS_DIR"]))
    if os.name == "nt":  # Windows only
        candidates += [Path(r"E:\cv_models"), Path(r"C:\Users\User\cv_models")]
    candidates += [
        Path.home() / ".cache" / "cv_models",                      # Mac/Linux/Kaggle
        ROOT / "models",                                           # фолбэк в проекте
    ]

    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            pass
        if p.exists() and _can_write(p):
            return p

    raise RuntimeError("Не удалось найти папку для моделей с правами записи.")


MODELS_DIR = _resolve_models_dir()

# Выставляем кэш-переменные ДО импорта любых ML-библиотек
os.environ["HF_HOME"]               = str(MODELS_DIR / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR / "huggingface" / "hub")
os.environ["TRANSFORMERS_CACHE"]    = str(MODELS_DIR / "huggingface" / "hub")
os.environ["YOLO_CONFIG_DIR"]       = str(MODELS_DIR / "ultralytics")
os.environ["EASYOCR_MODULE_PATH"]   = str(MODELS_DIR / "easyocr")

# Создаём подпапки — ошибки игнорируем, библиотеки сами создадут при необходимости
for _sub in ["huggingface/hub", "ultralytics", "easyocr", "whisper"]:
    try:
        (MODELS_DIR / _sub).mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        pass

# ── Offline-режим: модели качаются ОДИН раз, потом грузятся из кэша ────────────
# Скрипт scripts/download_models.py скачивает всё и ставит маркер .models_ready.
# Если маркер есть — HuggingFace/transformers больше НЕ ходят в хаб на проверку
# ревизий: анализ стартует мгновенно из локального кэша, без сети.
# Принудительно: CV_MODELS_OFFLINE=1 (всегда offline) / =0 (всегда online).
MODELS_READY_FLAG = MODELS_DIR / ".models_ready"

def _offline_mode() -> bool:
    forced = os.environ.get("CV_MODELS_OFFLINE")
    if forced is not None:
        return forced.strip() not in ("0", "false", "no", "")
    return MODELS_READY_FLAG.exists()

OFFLINE = _offline_mode()
if OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # ultralytics не лезет в сеть, если веса найдены локально (см. pz5_yolo)

# Пути к локальным весам YOLO (корень проекта) — чтобы ultralytics не качал заново
ULTRALYTICS_WEIGHTS_DIRS = [ROOT, MODELS_DIR / "ultralytics"]

FRAMES_DIR          = ROOT / "outputs" / "frames"
OCR_RESULTS_DIR     = ROOT / "outputs" / "ocr_results"
DETECTIONS_DIR      = ROOT / "outputs" / "detections"
CLASSIFICATIONS_DIR = ROOT / "outputs" / "classifications"
FINAL_REPORT        = ROOT / "outputs" / "final_report.json"

FRAME_STEP       = 30
CONF_YOLO        = 0.4
OCR_DEDUP_THR    = 85
DET_MERGE_WINDOW = 5
WHISPER_MODEL    = "base"

YOLO_MODEL       = "yolov8n.pt"
RESNET_MODEL     = "resnet50"

QWEN_BASE_URL    = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL       = "qwen-vl-max"
GEMINI_MODEL     = "gemini-1.5-flash"
