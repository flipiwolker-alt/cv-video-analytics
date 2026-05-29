#!/usr/bin/env bash
# ============================================================
#  Установка и запуск на macOS (Apple Silicon, M1–M4)
#  Всё считается локально на GPU Мака (Metal/MPS) — без Kaggle.
#
#  Использование:
#     cd cv-video-analytics
#     bash scripts/setup_mac.sh
#  Затем открой http://localhost:7860
# ============================================================
set -e
cd "$(dirname "$0")/.."

echo "==> 1/6  Homebrew + системные пакеты (ffmpeg, ollama, python 3.12)"
if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
brew install ffmpeg ollama python@3.12 || true

echo "==> 2/6  Запуск Ollama и загрузка модели qwen2.5:7b (~4.7 ГБ, один раз)"
brew services start ollama 2>/dev/null || (ollama serve >/dev/null 2>&1 &)
sleep 4
ollama pull qwen2.5:7b

echo "==> 3/6  Виртуальное окружение Python 3.12"
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "==> 4/6  Python-зависимости (torch с MPS ставится автоматически на Apple Silicon)"
pip install -r requirements.txt

echo "==> 5/6  Проверка GPU (MPS)"
python -c "import torch; print('  MPS доступен:', torch.backends.mps.is_available())"

echo "==> 6/6  Запуск веб-интерфейса"
echo "    Открой в браузере:  http://localhost:7860"
echo "    (модели CV скачаются при первом анализе, ~3–4 ГБ, один раз)"
python run_ui.py
