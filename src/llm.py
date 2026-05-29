"""LLM-слой поверх алгоритмического ядра — локальная Ollama, бесплатно и офлайн.

Принципы:
  • Ядро детекции работает БЕЗ LLM. Этот модуль — необязательный слой поверх.
  • Если Ollama не запущена / модель не скачана — все функции тихо возвращают
    пустой результат (graceful degrade), пайплайн продолжает работать.
  • Связь по локальному HTTP API Ollama (http://localhost:11434).

Применение в пайплайне (выбрано как самое ценное):
  1. classify_transcript() — контекстная классификация речи. Намного точнее
     keyword-regex для контекстных категорий (экстремизм, ЛГБТ-пропаганда,
     угрозы насилием), где важен смысл, а не отдельные слова.
  2. verify_frame()        — vision-проверка погранично-уверенных кадров
     YOLO/CLIP → срезает ложные срабатывания.
  3. summarize()           — краткое человекочитаемое резюме отчёта.

Установка Ollama (один раз):
    1. Скачать с https://ollama.com/download (Windows-инсталлятор)
    2. ollama pull qwen2.5:7b      # текстовая классификация
    3. ollama pull llava:7b        # vision-проверка (опционально)
    Сервис поднимается сам на localhost:11434.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
from typing import Optional

import httpx
from PIL import Image

from .subclass_info import SUBCLASS_DICT

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
if not OLLAMA_HOST.startswith(("http://", "https://")):
    OLLAMA_HOST = "http://" + OLLAMA_HOST   # допускаем OLLAMA_HOST без схемы
_VALID_SUBCLASSES = set(SUBCLASS_DICT.keys())

# Кэш доступности — чтобы не дёргать сеть на каждый сегмент
_avail_cache: dict[str, tuple[float, bool]] = {}
_avail_lock = threading.Lock()
_AVAIL_TTL = 30.0  # сек


# ── Доступность ──────────────────────────────────────────────────────────────

def is_available(model: Optional[str] = None, timeout: float = 2.0) -> bool:
    """Запущена ли Ollama (и доступна ли конкретная модель, если указана).

    Результат кэшируется на _AVAIL_TTL секунд.
    """
    key = model or "__any__"
    now = time.time()
    with _avail_lock:
        cached = _avail_cache.get(key)
        if cached and now - cached[0] < _AVAIL_TTL:
            return cached[1]

    ok = False
    try:
        resp = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        names = {m.get("name", "") for m in tags}
        # имена в Ollama вида "qwen2.5:7b"; latest может быть без тега
        if model is None:
            ok = True
        else:
            ok = any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
    except Exception:
        ok = False

    with _avail_lock:
        _avail_cache[key] = (now, ok)
    return ok


# ── Низкоуровневые вызовы ──────────────────────────────────────────────────────

def _chat(model: str, messages: list[dict], *, json_mode: bool = False,
          images: Optional[list[str]] = None, timeout: float = 120.0) -> Optional[str]:
    """Один вызов /api/chat. Возвращает content или None при любой ошибке."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192},
    }
    if json_mode:
        payload["format"] = "json"
    if images:
        # картинки крепятся к последнему user-сообщению
        payload["messages"][-1]["images"] = images
    try:
        resp = httpx.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    except Exception:
        return None


def _extract_json(text: str):
    """Достаёт первый JSON-объект/массив из ответа модели (на случай мусора вокруг)."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


# ── 1. Классификация транскрипта ───────────────────────────────────────────────

_CATEGORY_DESC = "\n".join(
    f"  - {k}: {v.ru_name} — {v.description}" for k, v in SUBCLASS_DICT.items()
)

_CLASSIFY_SYSTEM = (
    "Ты — модератор контента. Анализируешь фрагмент расшифровки речи из видео "
    "и определяешь, к каким категориям нежелательного контента он относится. "
    "Учитывай СМЫСЛ и КОНТЕКСТ, а не только отдельные слова. "
    "Отвечай строго в JSON.\n\n"
    "Категории:\n" + _CATEGORY_DESC + "\n\n"
    "Формат ответа — JSON-объект вида:\n"
    '{"detections": [{"category": "<ключ>", "confidence": <0..1>, "quote": "<фраза-улика>"}]}\n'
    "Если нарушений нет — верни {\"detections\": []}. "
    "category должен быть точно одним из ключей выше (на английском)."
)


def classify_transcript_segment(text: str, model: str, *,
                                 min_conf: float = 0.5) -> list[tuple[str, float, str]]:
    """LLM-классификация одного сегмента речи.

    Возвращает список (subclass, confidence, quote). Пустой список — нет нарушений
    либо LLM недоступна.
    """
    text = (text or "").strip()
    if len(text) < 3 or not is_available(model):
        return []

    content = _chat(
        model,
        [
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {"role": "user", "content": f"Фрагмент: «{text}»"},
        ],
        json_mode=True,
        timeout=90.0,
    )
    data = _extract_json(content)
    if not isinstance(data, dict):
        return []

    out: list[tuple[str, float, str]] = []
    for item in data.get("detections", []):
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category", "")).strip().lower()
        if cat not in _VALID_SUBCLASSES:
            continue
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(conf, 1.0))
        if conf < min_conf:
            continue
        quote = str(item.get("quote", ""))[:200]
        out.append((cat, round(conf, 3), quote))
    return out


# ── 2. Vision-проверка кадра ────────────────────────────────────────────────────

def verify_frame(image: Image.Image, subclass: str, model: str,
                 timeout: float = 90.0) -> Optional[float]:
    """Подтверждает ли кадр наличие признаков subclass.

    Возвращает confidence [0..1], или None если LLM недоступна / ошибка.
    """
    if subclass not in _VALID_SUBCLASSES or not is_available(model):
        return None
    entry = SUBCLASS_DICT[subclass]

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        f"Категория: «{entry.ru_name}» — {entry.description}\n"
        f"Есть ли на этом изображении явные визуальные признаки этой категории? "
        f'Ответь строго JSON: {{"present": true|false, "confidence": <0..1>}}.'
    )
    content = _chat(
        model,
        [{"role": "user", "content": prompt}],
        json_mode=True,
        images=[b64],
        timeout=timeout,
    )
    data = _extract_json(content)
    if not isinstance(data, dict):
        return None
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    present = bool(data.get("present", False))
    return round(max(0.0, min(conf, 1.0)), 3) if present else 0.0


# ── 3. Резюме отчёта ────────────────────────────────────────────────────────────

def summarize(detections: list, model: str, timeout: float = 90.0) -> Optional[str]:
    """Краткое человекочитаемое резюме по найденным нарушениям. None если LLM нет."""
    if not detections or not is_available(model):
        return None

    lines = []
    for d in detections[:60]:
        sub = d.subclass.value if hasattr(d.subclass, "value") else str(d.subclass)
        lines.append(f"- {sub} @ {d.time_interval} (conf {d.confidence})")
    listing = "\n".join(lines)

    content = _chat(
        model,
        [
            {"role": "system", "content":
                "Ты — аналитик контента. По списку детекций напиши краткое (3-5 предложений) "
                "резюме на русском: какие категории нарушений преобладают, насколько серьёзно, "
                "на какие тайм-коды обратить внимание. Без воды."},
            {"role": "user", "content": f"Детекции:\n{listing}"},
        ],
        timeout=timeout,
    )
    return content.strip() if content else None
