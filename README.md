# CV Video Analytics

Учебный проект по компьютерному зрению и анализу видео. Состоит из 8 практических заданий и итоговой курсовой работы. Ноутбуки выполняются в Google Colab, FastAPI-сервер развёрнут в Docker на облачной ВМ Timeweb.

---

## Подробное описание каждого ПЗ

### ПЗ 1 — Обработка изображений (OpenCV)
**Цель:** освоить базовые операции с изображениями.
**Что происходит:**
- Загрузка одного изображения через OpenCV (`cv2.imread`)
- Применяются преобразования: resize, перевод в grayscale (`cv2.cvtColor`), регулировка яркости/контраста (`cv2.convertScaleAbs`), Гауссово размытие (`cv2.GaussianBlur`), поворот (`cv2.getRotationMatrix2D` + `cv2.warpAffine`)
- Каждое преобразование визуализируется через `matplotlib`

**Вход:** одно изображение
**Выход:** серия преобразованных версий
📓 [PZ_1_OpenCV_Processing.ipynb](notebooks/PZ_1_OpenCV_Processing.ipynb)

---

### ПЗ 2 — Нарезка видео на кадры
**Цель:** подготовить датасет кадров для всех остальных ПЗ.
**Что происходит:**
- Видео загружается на Google Drive, монтируется в Colab (`drive.mount`)
- Через `cv2.VideoCapture` читается покадрово
- Сохраняется каждый N-й кадр (по умолчанию `FRAME_STEP = 30`, т.е. 1 кадр в секунду при 30fps) в формате JPG
- Имена файлов: `frame_000000.jpg`, `frame_000030.jpg` и т.д.

**Вход:** видеофайл на Drive
**Выход:** папка с кадрами `/MyDrive/cv-frames/кадры/`
📓 [PZ_2_Video_Download.ipynb](notebooks/PZ_2_Video_Download.ipynb)

---

### ПЗ 3 — OCR (распознавание текста)
**Цель:** извлечь весь текст с кадров (субтитры, вывески, надписи).
**Что происходит:**
- Используется EasyOCR с поддержкой `['ru', 'en']`
- Каждый кадр прогоняется через `reader.readtext()` — возвращает bounding boxes + распознанный текст + confidence
- Результаты собираются в DataFrame: `frame_num`, `text`, `confidence`, `bbox`
- Сохраняется в CSV

**Вход:** кадры из ПЗ 2
**Выход:** `outputs/ocr_results/ocr.csv`
📓 [PZ_3_OCR_Recognition.ipynb](notebooks/PZ_3_OCR_Recognition.ipynb)

---

### ПЗ 4 — Транскрипция аудио (Whisper)
**Цель:** получить текстовую расшифровку речи из видео.
**Что происходит:**
- Из видео извлекается аудиодорожка через `moviepy` (`VideoFileClip(...).audio.write_audiofile(...)`)
- Загружается модель Whisper `base` (можно заменить на `small`/`medium` для лучшего качества)
- `model.transcribe(audio_path)` возвращает текст + сегменты с тайм-кодами
- Поддерживается мультиязычность (автодетекция языка)

**Вход:** видеофайл
**Выход:** `outputs/transcript.txt` + сегменты с тайм-кодами
📓 [PZ_4_Whisper_Audio.ipynb](notebooks/PZ_4_Whisper_Audio.ipynb)

---

### ПЗ 5 — Детекция объектов (YOLOv8)
**Цель:** найти и классифицировать все объекты на каждом кадре.
**Что происходит:**
- Загружается предобученная YOLOv8n (`yolov8n.pt`) — лёгкая версия для быстрого инференса
- Каждый кадр прогоняется через `model.predict(frame, conf=0.4)`
- Для каждой детекции извлекается: класс (из 80 COCO-классов), confidence, bounding box (x1, y1, x2, y2)
- Опционально рисуются bbox'ы и сохраняется аннотированный кадр
- Все детекции пишутся в CSV: `frame_num`, `class`, `conf`, `x1`, `y1`, `x2`, `y2`

**Вход:** кадры из ПЗ 2
**Выход:** `outputs/detections/detections.csv` + аннотированные кадры
📓 [PZ_5_YOLO_Detection.ipynb](notebooks/PZ_5_YOLO_Detection.ipynb)

---

### ПЗ 6 — Классификация кадров (ResNet50)
**Цель:** для каждого кадра выдать топ-5 классов из ImageNet.
**Что происходит:**
- Загружается ResNet50 с весами ImageNet (`torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`)
- Кадры предобрабатываются: resize 224x224, нормализация по ImageNet mean/std
- `softmax(model(x))` → топ-5 индексов → имена классов
- Визуализация: кадр + подпись с топ-5 классами и вероятностями

**Вход:** кадры из ПЗ 2
**Выход:** `outputs/classifications/classifications.csv` (frame, top5_classes, top5_probs)
📓 [PZ_6_ResNet_Classification.ipynb](notebooks/PZ_6_ResNet_Classification.ipynb)

---

### ПЗ 7 — Описание кадров через LLM (OpenRouter)
**Цель:** получить семантическое описание сцены на каждом кадре через мультимодальную LLM.
**Что происходит:**
- Кадры из Drive отправляются POST-запросом на собственный FastAPI-сервер (`http://92.51.37.40:8000/describe`)
- Сервер кодирует изображение в base64 и шлёт в OpenRouter API
- Используется бесплатная модель `nvidia/nemotron-nano-12b-v2-vl:free` (vision-language)
- Промпт: `"List all objects visible in this image, one line, comma separated"`
- Ответ — список объектов на английском
- Реализован retry-механизм (3 попытки с задержкой 10с) на случай rate-limit
- Результаты собираются в JSON

**Зачем свой сервер:** OpenRouter блокирует запросы из РФ. Сервер с европейским IP служит прокси.

**Вход:** кадры из ПЗ 2
**Выход:** `outputs/llm_descriptions.json` + визуализация кадров с подписями
📓 [PZ_7_LLM__API.ipynb](notebooks/PZ_7_LLM__API.ipynb)

---

### ПЗ 8 — Постобработка результатов
**Цель:** превратить сырые данные из ПЗ 3, 5, 7 в осмысленный отчёт.
**Что происходит:**

**OCR (из ПЗ 3):**
- Дедупликация почти-одинаковых строк через `rapidfuzz.fuzz.ratio` (порог 85%)
- Удаляет повторы вроде "Hello world" и "Hello wrld" — оставляет одну версию

**YOLO (из ПЗ 5):**
- Группировка детекций по классу
- Склейка соседних кадров одного класса в "событие" (если разрыв ≤ 5 кадров)
- Получаем: `class`, `start_frame`, `end_frame`, `avg_conf` — вместо тысяч точечных детекций ~десяток событий

**LLM (из ПЗ 7):**
- Парсинг описаний, подсчёт частоты упоминаний объектов
- Топ-10 самых частых объектов в видео

**Итог:** сводный JSON-отчёт + графики (распределения, временные шкалы)
📓 [PZ_8_PostProcessing.ipynb](notebooks/PZ_8_PostProcessing.ipynb)

---

### Курсовая — Полный пайплайн
Объединяет ПЗ 2 → 3 → 4 → 5 → 6 → 7 → 8 в единый сценарий. Запускается одной кнопкой, на выходе — итоговый отчёт по видео.
📓 [Coursework_Full_Pipeline.ipynb](notebooks/Coursework_Full_Pipeline.ipynb)

---

## Деплой и запуск сервера

### Архитектура

```
Google Colab (ноутбуки)
     │
     │ HTTP POST с кадрами
     ▼
Timeweb Cloud VM (92.51.37.40)
     │
     ▼
Docker контейнер (cv-api)
     │ FastAPI :8000
     │
     ▼
OpenRouter API (Европа)
     │
     ▼
nvidia/nemotron-nano-12b-v2-vl:free
```

### Сервер
- **Хостинг:** Timeweb Cloud
- **IP:** `92.51.37.40`
- **ОС:** Ubuntu (Linux)
- **Доступ:** `ssh root@92.51.37.40`

### Стек контейнера
- **Контейнеризация:** Docker + Docker Compose
- **Базовый образ:** `python:3.10-slim`
- **Системные зависимости:** `ffmpeg`, `libgl1`, `libglib2.0-0`
- **Порт:** 8000 (проброшен наружу)
- **Restart-policy:** `unless-stopped` (контейнер автоматически поднимается после рестарта VM)

### Переменные окружения (`.env`)

# Здесь мы храним ключи доступа к внешним сервисам, которые контейнер подгружает при запуске

```bash
OPENROUTER_API_KEY=sk-or-v1-...
GEMINI_API_KEY=...           # опционально
QWEN_API_KEY=...             # опционально
```

### Команды деплоя

# Здесь мы клонируем репозиторий, создаём файл с ключами и поднимаем контейнер в фоновом режиме

**Первичный запуск:**
```bash
git clone https://github.com/flipiwolker-alt/cv-video-analytics.git
cd cv-video-analytics
nano .env  # добавить ключи
docker compose -f docker/docker-compose.yml --env-file .env up -d
```

# Здесь мы подтягиваем последние изменения из репозитория, пересобираем образ и перезапускаем контейнер

**Обновление после изменений в коде:**
```bash
cd ~/cv-video-analytics
git pull
docker compose -f docker/docker-compose.yml --env-file .env build
docker compose -f docker/docker-compose.yml --env-file .env up -d
```

# Здесь мы выводим последние 50 строк логов контейнера и продолжаем следить за ними в реальном времени

**Просмотр логов:**
```bash
docker logs docker-cv-api-1 --tail 50 -f
```

# Здесь мы проверяем, что контейнер запущен и порт 8000 проброшен наружу

**Проверка статуса:**
```bash
docker compose -f docker/docker-compose.yml ps
```

---

## API endpoints

### `GET /`
Health-check. Проверяет, что сервер жив и API-ключ загружен.

# Здесь мы выполняем простой GET-запрос на корневой адрес и получаем подтверждение работоспособности сервиса

**Ответ:**
```json
{ "status": "ok", "api_key_loaded": true }
```

---

### `POST /describe`
Описание одного изображения. Принимает файл, возвращает список объектов.

# Здесь мы отправляем на сервер изображение в виде multipart-формы и получаем перечень объектов, найденных на нём

**Запрос:**
```bash
curl -X POST http://92.51.37.40:8000/describe \
  -F "file=@frame.jpg"
```

**Ответ:**
```json
{
  "objects": "person, car, building, traffic light, road",
  "filename": "frame.jpg"
}
```

**Что делает:**
1. Принимает multipart-файл
2. Открывает через PIL, конвертирует в RGB, кодирует в base64
3. Шлёт в OpenRouter с промптом "list objects"
4. Retry × 3 с задержкой 10с при ошибках
5. Возвращает текст объектов

---

### `POST /describe-video`
Описание видео по ссылке. Скачивает видео, нарезает на кадры, описывает каждый.

# Здесь мы передаём серверу ссылку на видео и интервал между кадрами, а в ответ получаем описание содержимого видео с разбивкой по тайм-кодам

**Запрос:**
```bash
curl -X POST http://92.51.37.40:8000/describe-video \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=...", "interval_sec": 5}'
```

**Параметры:**
- `url` (обяз.) — ссылка на видео (YouTube, прямая ссылка на mp4 и др. поддерживается через `yt-dlp`)
- `interval_sec` (опц., по умолчанию 5) — интервал между анализируемыми кадрами в секундах

**Ответ:**
```json
{
  "url": "...",
  "frames_analyzed": 12,
  "results": [
    {"time_sec": 0,  "objects": "person, microphone, stage"},
    {"time_sec": 5,  "objects": "car, road, sky"},
    {"time_sec": 10, "objects": "..."}
  ]
}
```

**Что делает:**
1. `yt-dlp` скачивает видео во временную папку
2. `cv2.VideoCapture` определяет FPS, читает покадрово
3. Каждый `interval_sec * fps`-й кадр кодируется и шлётся в OpenRouter
4. Результаты с тайм-кодами агрегируются в JSON
5. Временные файлы удаляются после обработки

**Время ответа:** зависит от длины видео и `interval_sec`. Для 1-минутного видео с шагом 5с ≈ 12 кадров × 5-10с = 1-2 минуты.

---

## Структура репозитория

```
cv-video-analytics/
├── notebooks/                     ← Jupyter ноутбуки для каждого ПЗ
│   ├── PZ_1_OpenCV_Processing.ipynb
│   ├── PZ_2_Video_Download.ipynb
│   ├── PZ_3_OCR_Recognition.ipynb
│   ├── PZ_4_Whisper_Audio.ipynb
│   ├── PZ_5_YOLO_Detection.ipynb
│   ├── PZ_6_ResNet_Classification.ipynb
│   ├── PZ_7_LLM__API.ipynb
│   ├── PZ_8_PostProcessing.ipynb
│   └── Coursework_Full_Pipeline.ipynb
├── src/
│   ├── api.py                     ← FastAPI приложение (endpoints)
│   ├── config.py                  ← пути и параметры
│   └── utils.py                   ← extract_frames, deduplicate_texts, merge_detections
├── docker/
│   ├── Dockerfile                 ← python:3.10-slim + ffmpeg
│   ├── docker-compose.yml         ← сервис cv-api на :8000
│   └── requirements-docker.txt    ← зависимости для контейнера
├── docs/
│   ├── SETUP.md                   ← локальная установка
│   └── COLAB_GUIDE.md             ← работа в Colab
├── requirements.txt               ← зависимости для локальной разработки
└── README.md
```
