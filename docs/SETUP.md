# Локальный запуск

## Требования

- Python 3.10+
- ffmpeg (для аудио): `winget install ffmpeg` или скачать с https://ffmpeg.org
- CUDA (опционально, для GPU)

## Установка

```bash
git clone https://github.com/ВАШ_ЮЗЕР/cv-video-analytics.git
cd cv-video-analytics
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Запуск ноутбуков локально

```bash
pip install jupyter
jupyter notebook notebooks/
```

## API ключи

Создать файл `.env` в корне проекта:

```
GEMINI_API_KEY=ваш_ключ_gemini
QWEN_API_KEY=ваш_ключ_qwen
```

`.env` добавлен в `.gitignore` — ключи не попадут в GitHub.

## Docker (для ПЗ 7 и курсовой)

```bash
cd docker
docker-compose up -d
# Сервис: http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```
