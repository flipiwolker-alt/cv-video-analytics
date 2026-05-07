# Руководство по работе в Google Colab

## Открыть ноутбук из GitHub

1. Перейти на https://colab.research.google.com
2. **File → Open notebook → GitHub**
3. Вставить URL: `https://github.com/ВАШ_ЮЗЕР/cv-video-analytics`
4. Выбрать ноутбук из списка `notebooks/`

## Включить GPU

`Runtime → Change runtime type → Hardware accelerator → GPU (T4) → Save`

## Сохранить изменения обратно в GitHub

`File → Save a copy in GitHub → выбрать репо и ветку → Save`

## Порядок запуска ПЗ

```
PZ_1  → самостоятельный (изображения)
PZ_2  → нарезает видео → создаёт outputs/frames/
PZ_3  → читает outputs/frames/ → outputs/ocr_results/
PZ_4  → читает видео → outputs/audio.wav + транскрипт
PZ_5  → читает outputs/frames/ → outputs/detections/
PZ_6  → читает outputs/frames/ → outputs/classifications/
PZ_7  → читает outputs/frames/ → вызывает LLM API (нужен API ключ)
PZ_8  → читает outputs из ПЗ 3 и ПЗ 5 → постобработка
Курсовая → запускает всё в одном ноутбуке
```

## Монтирование Google Drive (для хранения файлов между сессиями)

```python
from google.colab import drive
drive.mount('/content/drive')
# Далее используйте /content/drive/MyDrive/cv-video-analytics/
```
