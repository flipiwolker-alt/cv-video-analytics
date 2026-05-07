# cv-video-analytics

Учебный проект: CV/Video Analytics — 8 практических заданий + курсовая.

---

## Содержание ПЗ

| № | Название | Notebook |
|---|----------|----------|
| 1 | Обработка изображений (OpenCV) | [PZ_1_OpenCV_Processing.ipynb](notebooks/PZ_1_OpenCV_Processing.ipynb) |
| 2 | Нарезка видео на кадры | [PZ_2_Video_Download.ipynb](notebooks/PZ_2_Video_Download.ipynb) |
| 3 | OCR из потока кадров (субтитры) | [PZ_3_OCR_Recognition.ipynb](notebooks/PZ_3_OCR_Recognition.ipynb) |
| 4 | Извлечение звука + Whisper | [PZ_4_Whisper_Audio.ipynb](notebooks/PZ_4_Whisper_Audio.ipynb) |
| 5 | Детекция объектов YOLO | [PZ_5_YOLO_Detection.ipynb](notebooks/PZ_5_YOLO_Detection.ipynb) |
| 6 | Классификация ResNet | [PZ_6_ResNet_Classification.ipynb](notebooks/PZ_6_ResNet_Classification.ipynb) |
| 7 | Распознавание через LLM (Qwen/Gemini) | [PZ_7_LLM_Qwen_API.ipynb](notebooks/PZ_7_LLM_Qwen_API.ipynb) |
| 8 | Постобработка результатов | [PZ_8_PostProcessing.ipynb](notebooks/PZ_8_PostProcessing.ipynb) |
| — | Курсовая: полный пайплайн | [Coursework_Full_Pipeline.ipynb](notebooks/Coursework_Full_Pipeline.ipynb) |

---

## Этапы с нуля: Git + Colab

### Этап 0 — Подготовка аккаунтов и инструментов

```
0.1  Создать аккаунт на GitHub (https://github.com)
0.2  Установить Git: https://git-scm.com/download/win
0.3  Настроить Git глобально:
       git config --global user.name  "Ваше Имя"
       git config --global user.email "you@example.com"
0.4  Убедиться, что есть аккаунт Google (нужен для Colab)
```

---

### Этап 1 — Создание репозитория на GitHub

```
1.1  Зайти на github.com → New repository
1.2  Имя: cv-video-analytics
1.3  Visibility: Public (или Private — по желанию)
1.4  Поставить галочку "Add a README file" → Create repository
1.5  Скопировать URL репозитория (пример):
       https://github.com/ВАШ_ЮЗЕР/cv-video-analytics.git
```

---

### Этап 2 — Локальная инициализация и первый push

```bash
# Перейти в папку проекта
cd C:\Users\User\PycharmProjects\cv-video-analytics

# Инициализировать git
git init

# Привязать удалённый репозиторий
git remote add origin https://github.com/ВАШ_ЮЗЕР/cv-video-analytics.git

# Добавить все файлы
git add .

# Первый коммит
git commit -m "init: project structure + PZ 1-8 notebooks"

# Переименовать ветку в main (если нужно)
git branch -M main

# Отправить на GitHub
git push -u origin main
```

---

### Этап 3 — Синхронизация: рабочий цикл

```bash
# Перед началом работы — получить изменения с GitHub
git pull origin main

# После изменений — зафиксировать и отправить
git add .
git commit -m "pz1: добавил обработку контраста"
git push origin main
```

**Рекомендуемые коммиты:**
- `pz1: описание изменения`
- `pz2: описание изменения`
- `fix: исправил баг в utils.py`
- `docs: обновил README`

---

### Этап 4 — Перенос ПЗ в Google Colab

```
4.1  Открыть https://colab.research.google.com
4.2  File → Open notebook → GitHub
4.3  Вставить URL репозитория:
       https://github.com/ВАШ_ЮЗЕР/cv-video-analytics
4.4  Выбрать нужный notebook из папки notebooks/
4.5  Нажать "Open" — ноутбук откроется в Colab

Альтернатива (ручная загрузка):
  File → Upload notebook → выбрать .ipynb файл
```

---

### Этап 5 — Runtime в Colab (GPU)

```
5.1  Runtime → Change runtime type
5.2  Hardware accelerator → GPU (T4 или лучше)
5.3  Нажать Save
5.4  Запустить первую ячейку с !pip install ...
```

---

### Этап 6 — Сохранение изменений из Colab обратно в Git

```
Способ 1 — через Colab:
  File → Save a copy in GitHub
  → выбрать репозиторий и ветку → Save

Способ 2 — скачать и запушить локально:
  File → Download → Download .ipynb
  → скопировать в папку notebooks/
  → git add . && git commit -m "pz1: правки из Colab" && git push
```

---

### Этап 7 — Структура папок проекта

```
cv-video-analytics/
├── notebooks/          ← Colab ноутбуки (основная работа)
├── src/                ← общие вспомогательные модули
├── data/               ← тестовые данные (мелкие файлы)
├── outputs/            ← результаты (в .gitignore для больших файлов)
├── docker/             ← для ПЗ 7 и курсовой (деплой на ВМ)
└── docs/               ← документация
```

---

### Этап 8 — Деплой на ВМ (Timeweb, ПЗ 7 + курсовая)

```
8.1  Зарегистрировать ВМ на https://timeweb.cloud
8.2  Выбрать Ubuntu 22.04, минимум 4GB RAM
8.3  Подключиться по SSH:
       ssh root@IP_АДРЕС
8.4  Установить Docker:
       curl -fsSL https://get.docker.com | sh
8.5  Клонировать репозиторий:
       git clone https://github.com/ВАШ_ЮЗЕР/cv-video-analytics.git
8.6  Запустить через Docker Compose:
       cd cv-video-analytics/docker
       docker-compose up -d
8.7  Сервис будет доступен на http://IP_АДРЕС:8000
```

---

## Быстрый старт (локально)

```bash
pip install -r requirements.txt
```

Для каждого ПЗ открыть соответствующий ноутбук в Jupyter или Colab.

---

## Итоговый сценарий (курсовая)

```
Загрузить видеофайл
    └─► Нарезать на кадры (ПЗ 2)
            ├─► OCR субтитров (ПЗ 3) ──► Дедупликация текста (ПЗ 8)
            ├─► YOLO детекция (ПЗ 5) ──► Склейка детекций (ПЗ 8)
            └─► ResNet/LLM (ПЗ 6/7)
    + Извлечь аудио (ПЗ 4) ──► Whisper транскрипция
    └─► Итоговый отчёт JSON/HTML
```
