"""Запуск Gradio-интерфейса.

Локально:       python run_ui.py
Публичный URL:  python run_ui.py --share
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.device import banner
print(banner())

from src.ui import build_ui, CUSTOM_CSS
import gradio as gr

parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true",
                    help="Создать публичный URL через Gradio (72 часа)")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--host", default="0.0.0.0")
args = parser.parse_args()

demo = build_ui()
demo.launch(
    server_name=args.host,
    server_port=args.port,
    share=args.share,
    show_error=True,
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.slate,
    ),
)
