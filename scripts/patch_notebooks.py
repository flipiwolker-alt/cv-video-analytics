"""Однократно приводит все ноутбуки к виду, исполнимому в Colab «с нуля».

Запуск: python3 scripts/patch_notebooks.py
Идемпотентно: повторный запуск ничего не ломает (cells сравниваются по содержимому).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

# ---------- общие сниппеты ----------

SETUP_HEADER = """# === Общий setup (Colab + локальный fallback) ===
import os, sys, subprocess, warnings, shutil
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=UserWarning)

IN_COLAB = 'google.colab' in sys.modules
USE_DRIVE = IN_COLAB and os.environ.get('CV_USE_DRIVE', '1') == '1'

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        BASE = '/content/drive/MyDrive/cv-frames'
    except Exception as e:
        print(f'[setup] drive недоступен ({e}) — пишем в /content/cv-frames')
        BASE = '/content/cv-frames'
elif IN_COLAB:
    BASE = '/content/cv-frames'
else:
    BASE = str((os.path.expanduser('~/cv-frames')))

BASE_DRIVE  = BASE
VIDEO_DIR   = f'{BASE}/видио'
FRAMES_DIR  = f'{BASE}/кадры'
RESULTS_DIR = f'{BASE}/результаты'
for d in (VIDEO_DIR, FRAMES_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)
print(f'[setup] BASE={BASE}')
"""

# стабильный публичный CC-ролик (~5 MB, ~ minute)
DEFAULT_VIDEO_URL = (
    "https://download.blender.org/peach/bigbuckbunny_movies/"
    "BigBuckBunny_320x180.mp4"
)

DOWNLOAD_VIDEO_SNIPPET = f"""# === Скачивание видео (yt-dlp работает и с прямыми URL, и с YouTube) ===
import os, subprocess
VIDEO_URL = globals().get('VIDEO_URL', '{DEFAULT_VIDEO_URL}')
video_path = f'{{VIDEO_DIR}}/video.mp4'

if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
    print(f'[video] уже скачано: {{video_path}} ({{os.path.getsize(video_path)/1e6:.1f}} MB)')
else:
    try:
        r = subprocess.run(
            ['yt-dlp', '-f', 'mp4/best[ext=mp4]/best',
             '-o', video_path, '--no-playlist', VIDEO_URL],
            capture_output=True, text=True, timeout=180
        )
        if r.returncode != 0 or not os.path.exists(video_path):
            raise RuntimeError(r.stderr[-500:] if r.stderr else 'yt-dlp вернул код != 0')
        print(f'[video] скачано: {{os.path.getsize(video_path)/1e6:.1f}} MB')
    except Exception as e:
        print(f'[video] yt-dlp упал ({{e}}), пробуем прямой URL по умолчанию')
        import urllib.request
        urllib.request.urlretrieve('{DEFAULT_VIDEO_URL}', video_path)
        print(f'[video] скачано fallback: {{os.path.getsize(video_path)/1e6:.1f}} MB')
"""

GPU_FLAG = "GPU_OK = False\ntry:\n    import torch; GPU_OK = torch.cuda.is_available()\nexcept Exception:\n    pass"


# ---------- утилиты ----------

def load(name):
    return json.loads((NB_DIR / name).read_text(encoding='utf-8'))


def save(name, nb):
    (NB_DIR / name).write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + '\n',
        encoding='utf-8',
    )


def set_cell(nb, idx, new_src):
    """Заменить source ячейки на новый текст (str → list[str] построчно)."""
    lines = new_src.splitlines(keepends=True)
    nb['cells'][idx]['source'] = lines
    # очистить старый output
    if nb['cells'][idx]['cell_type'] == 'code':
        nb['cells'][idx]['outputs'] = []
        nb['cells'][idx]['execution_count'] = None


def insert_cell(nb, idx, src, cell_type='code'):
    cell = {
        'cell_type': cell_type,
        'metadata': {},
        'source': src.splitlines(keepends=True),
    }
    if cell_type == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    nb['cells'].insert(idx, cell)


def already_patched(nb):
    return any(
        'CV-PATCHED' in ''.join(c.get('source', []))
        for c in nb['cells']
    )


def mark_patched(nb):
    insert_cell(nb, 0, '<!-- CV-PATCHED -->', cell_type='markdown')


# ---------- патчи ----------

def patch_pz1():
    nb = load('PZ_1_OpenCV_Processing.ipynb')
    if already_patched(nb):
        return
    set_cell(nb, 2, "!pip install -q opencv-python-headless easyocr pandas matplotlib 'numpy<2'\n")
    set_cell(nb, 3, """# Загрузка изображения: в Colab — через диалог, локально/при отмене — sample
import os, sys, urllib.request
file_path = None
if 'google.colab' in sys.modules:
    try:
        from google.colab import files
        uploaded = files.upload()
        if uploaded:
            file_path = list(uploaded.keys())[0]
    except Exception as e:
        print(f'[upload] пропуск: {e}')
if not file_path:
    file_path = '/tmp/sample.jpg'
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(
            'https://ultralytics.com/images/bus.jpg', file_path
        )
    print(f'используем sample: {file_path}')
""")
    # cell 4 — заменить gpu=True
    src4 = ''.join(nb['cells'][4]['source']).replace(
        'reader = easyocr.Reader([\'ru\', \'en\'], gpu=True)',
        "import torch\n"
        "reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())"
    )
    src4 = src4.replace(
        'img = cv2.imread(file_path)',
        "img = cv2.imread(file_path)\n"
        "if img is None:\n"
        "    raise SystemExit(f'не удалось прочитать {file_path}')"
    )
    set_cell(nb, 4, src4)
    mark_patched(nb)
    save('PZ_1_OpenCV_Processing.ipynb', nb)


def patch_pz2():
    nb = load('PZ_2_Video_Download.ipynb')
    if already_patched(nb):
        return
    set_cell(nb, 1, "!pip install -q opencv-python-headless yt-dlp tqdm 'numpy<2'\n")
    set_cell(nb, 2, SETUP_HEADER)
    set_cell(nb, 3,
        f"# Пример: публичный Big Buck Bunny. Замените на свой URL при желании.\n"
        f"VIDEO_URL = '{DEFAULT_VIDEO_URL}'\n\n" + DOWNLOAD_VIDEO_SNIPPET)
    # cell 4 — alt upload, оставим как есть (комментарии)
    # cell 5 — open video — добавим проверку
    src5 = ''.join(nb['cells'][5]['source'])
    if 'video_path' in src5 and 'raise' not in src5:
        set_cell(nb, 5,
            "import cv2\n"
            "cap = cv2.VideoCapture(video_path)\n"
            "if not cap.isOpened():\n"
            "    raise SystemExit('видео не открылось, проверьте файл/URL')\n"
            "fps   = cap.get(cv2.CAP_PROP_FPS) or 25\n"
            "total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
            "print(f'fps: {fps:.1f}, кадров: {total}, длина: {total/fps:.1f} сек')\n")
    mark_patched(nb)
    save('PZ_2_Video_Download.ipynb', nb)


def patch_pz3():
    nb = load('PZ_3_OCR_Recognition.ipynb')
    if already_patched(nb):
        return
    set_cell(nb, 2, "!pip install -q easyocr opencv-python-headless pandas tqdm 'numpy<2'\n")
    set_cell(nb, 3, SETUP_HEADER + """
frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg'))
if not frames:
    raise SystemExit(f'нет кадров в {FRAMES_DIR}. Сначала запустите ПЗ 2.')
print(f'кадров: {len(frames)}')
""")
    # cell 4 — gpu fix
    src4 = ''.join(nb['cells'][4]['source']).replace(
        "reader = easyocr.Reader(['ru', 'en'], gpu=True)",
        "import torch\n"
        "reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())"
    )
    set_cell(nb, 4, src4)
    mark_patched(nb)
    save('PZ_3_OCR_Recognition.ipynb', nb)


def patch_pz4():
    nb = load('PZ_4_Whisper_Audio.ipynb')
    if already_patched(nb):
        return
    set_cell(nb, 2,
        "!pip install -q openai-whisper 'moviepy==1.0.3' 'numpy<2'\n"
        "# ffmpeg в Colab уже установлен; если запускаете локально — поставьте через свой пакетный менеджер\n")
    set_cell(nb, 3, SETUP_HEADER + """
videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mkv'))]
if not videos:
    raise SystemExit(f'нет видео в {VIDEO_DIR}. Сначала запустите ПЗ 2.')
video_path = f'{VIDEO_DIR}/{videos[0]}'
print(f'видео: {video_path}')
""")
    mark_patched(nb)
    save('PZ_4_Whisper_Audio.ipynb', nb)


def patch_pz5():
    nb = load('PZ_5_YOLO_Detection.ipynb')
    if already_patched(nb):
        return
    set_cell(nb, 2, "!pip install -q ultralytics opencv-python-headless pandas tqdm 'numpy<2'\n")
    set_cell(nb, 3, SETUP_HEADER + """
frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg'))
if not frames:
    raise SystemExit(f'нет кадров в {FRAMES_DIR}. Сначала запустите ПЗ 2.')
print(f'кадров: {len(frames)}')
""" + "\n" + GPU_FLAG + "\nprint(f'GPU доступен: {GPU_OK}')\n")
    mark_patched(nb)
    save('PZ_5_YOLO_Detection.ipynb', nb)


def patch_pz6():
    nb = load('PZ_6_ResNet_Classification.ipynb')
    if already_patched(nb):
        return
    set_cell(nb, 2, "!pip install -q torch torchvision Pillow tqdm pandas 'numpy<2'\n")
    set_cell(nb, 3, SETUP_HEADER + """
frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg'))
if not frames:
    raise SystemExit(f'нет кадров в {FRAMES_DIR}. Сначала запустите ПЗ 2.')
print(f'кадров: {len(frames)}')
""")
    set_cell(nb, 4, """import json, urllib.request
import torch
from torchvision import models, transforms

URLS = [
    'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json',
    'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt',
]
labels_path = '/tmp/imagenet_labels.json'
LABELS = None
for url in URLS:
    try:
        urllib.request.urlretrieve(url, labels_path)
        text = open(labels_path, encoding='utf-8').read()
        LABELS = json.loads(text) if text.strip().startswith('[') else [l.strip() for l in text.splitlines() if l.strip()]
        if LABELS and len(LABELS) >= 1000:
            break
    except Exception as e:
        print(f'[labels] fail {url}: {e}')
if not LABELS or len(LABELS) < 1000:
    LABELS = [f'class_{i}' for i in range(1000)]
    print('[labels] fallback на class_0..999')
else:
    print(f'[labels] загружено {len(LABELS)} классов')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
print(f'resnet50 готов, устройство: {device}')
""")
    mark_patched(nb)
    save('PZ_6_ResNet_Classification.ipynb', nb)


def patch_pz7():
    nb = load('PZ_7_LLM__API.ipynb')
    if already_patched(nb):
        return
    set_cell(nb, 2, "!pip install -q requests pillow tqdm 'numpy<2'\n")
    set_cell(nb, 3, SETUP_HEADER + """
frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg'))
if not frames:
    raise SystemExit(f'нет кадров в {FRAMES_DIR}. Сначала запустите ПЗ 2.')
print(f'кадров всего: {len(frames)}')
""")
    set_cell(nb, 4, """# Два режима работы ПЗ 7:
#   1) прямой вызов OpenRouter, если задан OPENROUTER_API_KEY
#      (export OPENROUTER_API_KEY=... или getpass-ввод ниже);
#   2) если ключа нет — graceful skip с mock-JSON, чтобы ПЗ 8 не сломался.
import os, requests

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '').strip()
if not OPENROUTER_API_KEY:
    try:
        from getpass import getpass
        OPENROUTER_API_KEY = getpass(
            'OpenRouter API key (Enter — пропустить, сделать mock-результат): '
        ).strip()
    except Exception:
        pass

USE_LLM = bool(OPENROUTER_API_KEY)
print('режим:', 'OpenRouter API' if USE_LLM else 'mock (без ключа)')
""")
    set_cell(nb, 5, """import json, time, base64
from tqdm.notebook import tqdm

OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
MODEL = 'nvidia/nemotron-nano-12b-v2-vl:free'
PROMPT = 'List all objects visible in this image, one line, comma separated. Only the list, no explanations.'

def describe_frame_openrouter(image_path):
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        OPENROUTER_URL,
        headers={'Authorization': f'Bearer {OPENROUTER_API_KEY}'},
        json={'model': MODEL, 'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
            {'type': 'text', 'text': PROMPT},
        ]}]},
        timeout=60,
    )
    resp.raise_for_status()
    return {'objects': resp.json()['choices'][0]['message']['content'].strip()}

STEP  = max(1, len(frames) // 20)
BATCH = frames[::STEP][:20]
PAUSE = 3
print(f'обрабатываем {len(BATCH)} кадров (PAUSE={PAUSE}s)')
""")
    set_cell(nb, 6, """results = []
for i, fname in enumerate(tqdm(BATCH, desc='openrouter' if USE_LLM else 'mock')):
    if USE_LLM:
        try:
            res = describe_frame_openrouter(f'{FRAMES_DIR}/{fname}')
        except Exception as e:
            res = {'objects': 'ошибка', 'error': str(e)}
    else:
        res = {'objects': 'mock: object1, object2, object3'}
    res['frame'] = fname
    results.append(res)
    if USE_LLM and i < len(BATCH) - 1:
        time.sleep(PAUSE)

with open(f'{RESULTS_DIR}/llm_detections.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'сохранено: {RESULTS_DIR}/llm_detections.json ({len(results)} записей)')
""")
    mark_patched(nb)
    save('PZ_7_LLM__API.ipynb', nb)


def patch_pz8():
    nb = load('PZ_8_PostProcessing.ipynb')
    if already_patched(nb):
        return
    set_cell(nb, 2, "!pip install -q rapidfuzz pandas matplotlib 'numpy<2'\n")
    set_cell(nb, 3, SETUP_HEADER + """
print('файлы в результатах:')
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f'  {f}')
""")
    # cell 5 — OCR dedup: обернуть в проверку файла
    set_cell(nb, 5, """import pandas as pd
from rapidfuzz import fuzz, process

ocr_path = f'{RESULTS_DIR}/ocr_results.csv'
if not os.path.exists(ocr_path):
    print(f'[skip] {ocr_path} не найден — пропускаем дедупликацию')
    df_ocr, unique = pd.DataFrame(columns=['text']), []
else:
    df_ocr = pd.read_csv(ocr_path)
    print(f'строк до дедупликации: {len(df_ocr)}')

    def deduplicate(texts, threshold=85):
        out = []
        for t in texts:
            if not out:
                out.append(t); continue
            best = process.extractOne(t, out, scorer=fuzz.ratio)
            if best is None or best[1] < threshold:
                out.append(t)
        return out

    unique = deduplicate(df_ocr['text'].dropna().tolist())
    print(f'строк после дедупликации: {len(unique)}')
    pd.DataFrame({'text': unique}).to_csv(
        f'{RESULTS_DIR}/ocr_dedup.csv', index=False, encoding='utf-8-sig')
""")
    # cell 7 — YOLO merge: проверка файла
    set_cell(nb, 7, """yolo_path = f'{RESULTS_DIR}/yolo_detections.csv'
if not os.path.exists(yolo_path):
    print(f'[skip] {yolo_path} не найден')
    df_det, df_merged = pd.DataFrame(columns=['class','frame_num','conf']), pd.DataFrame(columns=['class'])
else:
    df_det = pd.read_csv(yolo_path)
    WINDOW = 5

    def merge_detections(group):
        group = group.sort_values('frame_num').reset_index(drop=True)
        events, start, prev = [], group.iloc[0], group.iloc[0]['frame_num']
        for _, row in group.iloc[1:].iterrows():
            if row['frame_num'] - prev > WINDOW:
                events.append({'class': start['class'], 'start_frame': start['frame_num'],
                               'end_frame': prev, 'avg_conf': round(group['conf'].mean(), 3)})
                start = row
            prev = row['frame_num']
        events.append({'class': start['class'], 'start_frame': start['frame_num'],
                       'end_frame': prev, 'avg_conf': round(group['conf'].mean(), 3)})
        return pd.DataFrame(events)

    df_merged = (df_det.groupby('class', group_keys=False)
                       .apply(merge_detections)
                       .sort_values('start_frame').reset_index(drop=True))
    df_merged.to_csv(f'{RESULTS_DIR}/yolo_merged.csv', index=False)
    print(f'детекций: {len(df_det)} -> событий: {len(df_merged)}')
""")
    # cell 9 — LLM
    set_cell(nb, 9, """import json
from collections import Counter

llm_path = f'{RESULTS_DIR}/llm_detections.json'
llm_data, counter = [], Counter()
if not os.path.exists(llm_path):
    print(f'[skip] {llm_path} не найден')
else:
    with open(llm_path, encoding='utf-8') as f:
        llm_data = json.load(f)
    all_objects = []
    for item in llm_data:
        objects = item.get('objects', '')
        if objects and objects != 'ошибка':
            for obj in objects.split(','):
                obj = obj.strip().lower()
                if obj:
                    all_objects.append(obj)
    counter = Counter(all_objects)
    print(f'обработано кадров: {len(llm_data)}')
    for obj, cnt in counter.most_common(15):
        print(f'  {obj}: {cnt}')
""")
    mark_patched(nb)
    save('PZ_8_PostProcessing.ipynb', nb)


def patch_coursework():
    nb = load('Coursework_Full_Pipeline.ipynb')
    if already_patched(nb):
        return
    set_cell(nb, 2,
        "!pip install -q yt-dlp easyocr ultralytics openai-whisper "
        "'moviepy==1.0.3' rapidfuzz torch torchvision tqdm requests pandas matplotlib 'numpy<2'\n")
    set_cell(nb, 4, f"""# Параметры пайплайна
VIDEO_URL = '{DEFAULT_VIDEO_URL}'  # публичный CC-ролик; замените при необходимости

FRAME_STEP = 30
CONF       = 0.4
WINDOW     = 5
PAUSE      = 3

# OpenRouter: ключ через env или пустая строка (тогда LLM-блок будет mock)
import os
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '').strip()
""")
    # cell 6 — папки с idem-check
    set_cell(nb, 6, SETUP_HEADER + """
import re, json as _json
video_title = 'video'
try:
    r = subprocess.run(['yt-dlp', '--print', 'title', '--no-playlist', VIDEO_URL],
                       capture_output=True, text=True, timeout=30)
    if r.returncode == 0 and r.stdout.strip():
        video_title = re.sub(r'[^\\w\\s\\-]', '', r.stdout.strip()).strip().replace(' ', '_')[:80] or 'video'
except Exception:
    pass
raw_title = video_title

VIDEO_DIR   = f'{BASE}/{video_title}'
FRAMES_DIR  = f'{VIDEO_DIR}/кадры'
RESULTS_DIR = VIDEO_DIR
REPORT_PATH = f'{VIDEO_DIR}/final_report.json'
os.makedirs(FRAMES_DIR, exist_ok=True)
print(f'папка: {VIDEO_DIR}')

if os.path.exists(REPORT_PATH):
    with open(REPORT_PATH, encoding='utf-8') as f:
        print(_json.dumps(_json.load(f), ensure_ascii=False, indent=2))
    print('\\n[idem] отчёт уже есть; перезапишите, удалив final_report.json')
""")
    # cell 8 — video download (используем общий сниппет)
    set_cell(nb, 8, DOWNLOAD_VIDEO_SNIPPET)

    # cell 12 — easyocr GPU
    src12 = ''.join(nb['cells'][12]['source']).replace(
        "reader = easyocr.Reader(['ru', 'en'], gpu=True)",
        "import torch\n"
        "reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())"
    )
    set_cell(nb, 12, src12)

    # cell 21 — ResNet labels fallback
    set_cell(nb, 21, """import json as _json
import torch, urllib.request
from torchvision import models, transforms
from PIL import Image

URLS = [
    'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json',
    'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt',
]
LABELS = None
for url in URLS:
    try:
        urllib.request.urlretrieve(url, '/tmp/imagenet_labels.json')
        text = open('/tmp/imagenet_labels.json', encoding='utf-8').read()
        LABELS = _json.loads(text) if text.strip().startswith('[') else [l.strip() for l in text.splitlines() if l.strip()]
        if LABELS and len(LABELS) >= 1000:
            break
    except Exception as e:
        print(f'[labels] {url} fail: {e}')
if not LABELS or len(LABELS) < 1000:
    LABELS = [f'class_{i}' for i in range(1000)]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
print(f'resnet50 на {device}, классов: {len(LABELS)}')
""")

    # cell 24 — API check → теперь решение режима LLM
    set_cell(nb, 24, """# Режим LLM: прямой OpenRouter, если есть OPENROUTER_API_KEY; иначе mock
import base64, requests, time
USE_LLM = bool(OPENROUTER_API_KEY)
print('режим:', 'OpenRouter API' if USE_LLM else 'mock (без ключа)')
""")

    # cell 25 — describe_frame через прямой OpenRouter
    set_cell(nb, 25, """OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
MODEL = 'nvidia/nemotron-nano-12b-v2-vl:free'
PROMPT_LLM = 'List all objects visible in this image, one line, comma separated. Only the list, no explanations.'

def describe_frame(image_path):
    if not USE_LLM:
        return {'objects': 'mock: object1, object2, object3'}
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        OPENROUTER_URL,
        headers={'Authorization': f'Bearer {OPENROUTER_API_KEY}'},
        json={'model': MODEL, 'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
            {'type': 'text', 'text': PROMPT_LLM},
        ]}]}, timeout=60)
    resp.raise_for_status()
    return {'objects': resp.json()['choices'][0]['message']['content'].strip()}

STEP  = max(1, len(frames) // 20)
BATCH = frames[::STEP][:20]
print(f'обрабатываем {len(BATCH)} кадров')
""")

    # cell 26 — LLM loop (обёрнут)
    set_cell(nb, 26, """results = []
for i, fname in enumerate(tqdm(BATCH, desc='llm')):
    try:
        res = describe_frame(f'{FRAMES_DIR}/{fname}')
    except Exception as e:
        res = {'objects': 'ошибка', 'error': str(e)}
    res['frame'] = fname
    results.append(res)
    if USE_LLM and i < len(BATCH) - 1:
        time.sleep(PAUSE)

with open(f'{RESULTS_DIR}/llm_detections.json', 'w', encoding='utf-8') as f:
    _json.dump(results, f, ensure_ascii=False, indent=2)
print(f'обработано: {len(results)} кадров')
""")
    mark_patched(nb)
    save('Coursework_Full_Pipeline.ipynb', nb)


# ---------- запуск всех патчей ----------

PATCHES = [patch_pz1, patch_pz2, patch_pz3, patch_pz4,
           patch_pz5, patch_pz6, patch_pz7, patch_pz8, patch_coursework]

if __name__ == '__main__':
    for fn in PATCHES:
        try:
            fn()
            print(f'✓ {fn.__name__}')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'✗ {fn.__name__}: {e}')
