from pathlib import Path

ROOT = Path(__file__).parent.parent

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
