"""Запуск FastAPI-сервера.

Локально:           python run_api.py
С перезагрузкой:    python run_api.py --reload
На другом порту:    python run_api.py --port 8080

После запуска:
  Документация:     http://localhost:8000/docs
  Синхронный анализ (пример):
    curl -X POST http://localhost:8000/analyze/sync \\
         -H "Content-Type: application/json" \\
         -d '{"url": "https://www.youtube.com/watch?v=XXXX"}'
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

from src.device import banner
print(banner())

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--reload", action="store_true")
args = parser.parse_args()

uvicorn.run(
    "src.main:app",
    host=args.host,
    port=args.port,
    reload=args.reload,
)
