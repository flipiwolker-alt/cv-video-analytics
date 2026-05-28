"""Gradio UI — CV Video Analytics.

Прогресс стримится через yield (Gradio generator) — никакого JS-оверлея.
Дерево этапов и фразы обновляются каждые 2 сек из Python.
"""
from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

import gradio as gr

from .pipeline import VideoAnalyzer
from .subclass_info import SEVERITY_ORDER, SUBCLASS_DICT
from .schemas import TimeBasedReport
from .pipeline import DetectionMedia


# ── Фразы по этапам ──────────────────────────────────────────────────────────

_PHRASES: dict[str, list[str]] = {
    "dl": [
        "Подключаюсь к YouTube...", "Проверяю доступность видео...",
        "Выбираю оптимальный поток...", "Скачиваю видеопоток...",
        "Скачиваю аудиодорожку...", "Объединяю треки...",
        "Читаю метаданные...", "Определяю частоту кадров...",
        "Считаю число кадров...", "Готовлю файл к анализу...",
    ],
    "kf": [
        "Открываю видеофайл...", "Ищу смены сцен...",
        "Анализирую первые кадры...", "Обнаружена смена сцены...",
        "Декодирую видеопоток...", "Нормализую изображения...",
        "Готовлю кадры для нейросетей...", "Отбираю информативные кадры...",
        "Индексирую кадры по таймлайну...",
    ],
    "au": [
        "Извлекаю аудиодорожку...", "Конвертирую в WAV 16kHz...",
        "Инициализирую Whisper...", "Распознаю речь на русском...",
        "Транскрибирую первую минуту...", "Обрабатываю середину ролика...",
        "Ищу ключевые слова...", "Проверяю нецензурную лексику...",
        "Объединяю соседние фрагменты...", "Аудио-анализ завершён!",
    ],
    "yo": [
        "Загружаю YOLOv8n...", "Инициализирую детектор...",
        "Обнаруживаю объекты в кадрах...", "Ищу бутылки и алкоголь...",
        "Детектирую сигареты и вейпы...", "Проверяю наличие оружия...",
        "Фильтрую по порогу уверенности...", "Сопоставляю с субклассами...",
        "YOLO завершил работу!",
    ],
    "cl": [
        "Загружаю CLIP ViT-B/32...", "Кодирую текстовые промпты...",
        "Вычисляю эмбеддинги кадров...", "Анализирую контекст сцен...",
        "Ищу нарушения в кадре...", "Сравниваю с neutral baseline...",
        "Фильтрую ложные срабатывания...", "CLIP завершил классификацию!",
    ],
    "oc": [
        "Загружаю EasyOCR (ru+en)...", "Инициализирую движок OCR...",
        "Распознаю текст в кадрах...", "Ищу надписи на баннерах...",
        "Читаю субтитры и оверлеи...", "Классифицирую найденный текст...",
        "OCR завершён!",
    ],
    "po": [
        "Собираю результаты этапов...", "Запускаю Temporal NMS...",
        "Объединяю близкие события...", "Удаляю дубликаты...",
        "Вычисляю финальный confidence...", "Сортирую по таймлайну...",
        "Формирую отчёт...",
    ],
    "me": [
        "Извлекаю кадры событий...", "Аннотирую кадры...",
        "Вырезаю видеофрагменты...", "Перекодирую для браузера...",
        "Сохраняю превью...", "Формирую галерею...", "Почти готово...",
    ],
}

_STAGE_LIST = [
    ("dl",  0.00, 0.22, "⬇", "Загрузка видео"),
    ("kf",  0.22, 0.32, "🎞", "Ключевые кадры"),
    ("par", 0.32, 0.82, "⚡", "Параллельный анализ"),
    ("po",  0.82, 0.93, "🔧", "Финальная обработка"),
    ("me",  0.93, 1.00, "🎬", "Подготовка медиа"),
]


def _get_stage(frac: float) -> str:
    for sid, s, e, *_ in _STAGE_LIST:
        if frac < e:
            return sid
    return "me"


def _rand_phrase(frac: float) -> str:
    stage = _get_stage(frac)
    key = "au" if stage == "par" else stage
    return random.choice(_PHRASES.get(key, ["Обрабатываю..."]))


# ── Утилиты ──────────────────────────────────────────────────────────────────

def _make_analyzer(whisper_model: str, use_ocr: bool, use_clip: bool) -> VideoAnalyzer:
    return VideoAnalyzer(
        dummy=False, offline=False,
        whisper_model=whisper_model,
        use_ocr=use_ocr,
        use_clip=use_clip,
        stage_timeout=240.0,
    )


def _estimate_eta(use_ocr: bool, use_clip: bool, whisper_model: str) -> int:
    t = 25
    t += {"tiny": 35, "base": 55, "small": 100, "medium": 200}.get(whisper_model, 55)
    t += 16
    if use_clip: t += 18
    if use_ocr:  t += 90
    t += 15
    return t


# ── Дерево прогресса (чистый HTML, Python обновляет) ─────────────────────────

def _progress_html(
    frac: float,
    phrase: str,
    t_start: float,
    eta_total: int,
    use_clip: bool = True,
    use_ocr: bool = False,
) -> str:
    pct = max(1, min(99, int(frac * 100)))

    elapsed = time.time() - t_start
    rem = max(0, int(eta_total - elapsed))
    eta_str = f"~{rem} сек" if rem < 60 else f"~{rem // 60} мин {rem % 60:02d} сек"

    rows = ""
    for i, (sid, s0, s1, icon, label) in enumerate(_STAGE_LIST):
        is_last = i == len(_STAGE_LIST) - 1

        if frac < s0:
            dc, db, tc, bw, bc, tick = "#2a2a4a", "#0d0d14", "#475569", 0, "#1e1e3a", ""
            vl_c = "#1e1e3a"
        elif frac >= s1:
            dc, db, tc, bw, bc, tick = "#22c55e", "#22c55e", "#4ade80", 100, "#22c55e", " ✓"
            vl_c = "#22c55e"
        else:
            sf = (frac - s0) / (s1 - s0)
            dc, db, tc, bw, bc, tick = "#6366f1", "#6366f1", "#e2e8f0", int(sf * 100), "#6366f1", ""
            vl_c = "#6366f1"

        vl_html = (
            "" if is_last
            else f'<div style="width:2px;flex:1;background:{vl_c};min-height:10px"></div>'
        )

        if sid == "par":
            subs = [("🎤", "Речь (Whisper)"), ("📦", "Объекты (YOLO)")]
            if use_clip: subs.append(("🧠", "Сцены (CLIP)"))
            if use_ocr:  subs.append(("📝", "Текст (OCR)"))

            sub_rows = ""
            for sicon, slabel in subs:
                if frac < s0:
                    sc, sb, stc, sw, sbc, st = "#2a2a4a", "#0d0d14", "#475569", 0,  "#1e1e3a", ""
                elif frac >= s1:
                    sc, sb, stc, sw, sbc, st = "#22c55e", "#22c55e", "#4ade80", 100, "#22c55e", " ✓"
                else:
                    sc, sb, stc, sw, sbc, st = "#6366f1", "#6366f1", "#c7d2fe", 40,  "#6366f1", ""

                sub_rows += f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
  <div style="width:8px;height:8px;border-radius:50%;border:2px solid {sc};background:{sb};flex-shrink:0"></div>
  <div style="flex:1">
    <div style="font-size:.79rem;font-weight:600;color:{stc};margin-bottom:2px">{sicon} {slabel}{st}</div>
    <div style="height:3px;background:#1a1f2e;border-radius:2px;overflow:hidden">
      <div style="height:100%;width:{sw}%;background:{sbc};border-radius:2px"></div>
    </div>
  </div>
</div>"""

            badge = (
                '<div style="display:inline-flex;align-items:center;gap:6px;'
                'background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.25);'
                'color:#818cf8;border-radius:6px;padding:3px 10px;'
                'font-size:.7rem;font-weight:700;margin-bottom:8px">⚡ параллельные процессы</div>'
            )
            rows += f"""
<div style="display:flex;align-items:stretch;gap:0">
  <div style="display:flex;flex-direction:column;align-items:center;width:20px;flex-shrink:0">
    <div style="width:14px;height:14px;border-radius:50%;border:2px solid {dc};background:{db};margin-top:3px;flex-shrink:0"></div>
    {vl_html}
  </div>
  <div style="flex:1;padding-left:10px;padding-bottom:10px">
    <div style="font-size:.86rem;font-weight:600;color:{tc};margin-bottom:8px">{icon} {label}{tick}</div>
    <div style="margin-left:4px">{badge}{sub_rows}</div>
    <div style="height:4px;background:#1a1f2e;border-radius:2px;overflow:hidden;margin-top:4px">
      <div style="height:100%;width:{bw}%;background:{bc};border-radius:2px"></div>
    </div>
  </div>
</div>"""
        else:
            rows += f"""
<div style="display:flex;align-items:stretch;gap:0">
  <div style="display:flex;flex-direction:column;align-items:center;width:20px;flex-shrink:0">
    <div style="width:14px;height:14px;border-radius:50%;border:2px solid {dc};background:{db};margin-top:3px;flex-shrink:0"></div>
    {vl_html}
  </div>
  <div style="flex:1;padding-left:10px;padding-bottom:10px">
    <div style="font-size:.86rem;font-weight:600;color:{tc};margin-bottom:5px">{icon} {label}{tick}</div>
    <div style="height:4px;background:#1a1f2e;border-radius:2px;overflow:hidden">
      <div style="height:100%;width:{bw}%;background:{bc};border-radius:2px"></div>
    </div>
  </div>
</div>"""

    return f"""
<div style="background:linear-gradient(160deg,#12121e 0%,#0f1520 100%);
    border:1px solid #1e2a3a;border-radius:20px;padding:28px 32px;
    font-family:sans-serif;color:#cdd6f4;
    box-shadow:0 8px 40px rgba(0,0,0,.7);max-width:600px;margin:16px auto">

  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
    <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0">Анализирую видео...</div>
    <div style="font-size:.82rem;font-weight:600;color:#4ade80;
        background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.25);
        border-radius:8px;padding:4px 12px">{eta_str}</div>
  </div>

  <div style="height:12px;background:#1a1f2e;border-radius:6px;overflow:hidden;margin-bottom:6px;
      box-shadow:inset 0 1px 3px rgba(0,0,0,.5)">
    <div style="height:100%;width:{pct}%;border-radius:6px;
        background:linear-gradient(90deg,#22c55e,#4ade80,#86efac);
        box-shadow:0 0 14px rgba(34,197,94,.5)"></div>
  </div>
  <div style="text-align:right;font-size:.75rem;color:#4b5563;margin-bottom:18px">{pct}%</div>

  <div style="text-align:center;font-size:.84rem;color:#6366f1;
      font-style:italic;margin-bottom:22px;min-height:20px">{phrase}</div>

  {rows}
</div>
"""


# ── Заглушка (до запуска анализа) ────────────────────────────────────────────

_PLACEHOLDER_HTML = """
<div style="text-align:center;padding:56px 20px;font-family:sans-serif">
  <div style="font-size:3.5rem;margin-bottom:16px">🎬</div>
  <div style="font-size:1.05rem;font-weight:600;color:#6b7280;margin-bottom:8px">
    Введите ссылку на YouTube и нажмите «Запустить анализ»
  </div>
  <div style="font-size:.82rem;color:#374151">
    Поддерживаются видео до 60 минут · 9 категорий нарушений
  </div>
</div>
"""


def _error_html(msg: str) -> str:
    return f"""
<div style="text-align:center;padding:40px 20px;font-family:sans-serif;
    background:#1c0e0e;border:1px solid #7f1d1d;border-radius:16px;
    color:#fca5a5;margin:16px 0">
  <div style="font-size:2.5rem;margin-bottom:12px">❌</div>
  <div style="font-size:1rem;font-weight:600;color:#f87171;margin-bottom:8px">Ошибка анализа</div>
  <div style="font-size:.83rem">{msg}</div>
</div>
"""


# ── HTML helpers (результаты) ─────────────────────────────────────────────────

def _summary_html(report: TimeBasedReport) -> str:
    counts: dict[str, int] = {}
    for d in report.detections:
        counts[d.subclass.value] = counts.get(d.subclass.value, 0) + 1

    if not counts:
        return (
            "<div style='text-align:center;padding:40px;color:#6b7280;font-family:sans-serif'>"
            "<div style='font-size:3rem'>✅</div>"
            "<div style='font-size:1.1rem;margin-top:10px;color:#4ade80'>"
            "Нежелательный контент не обнаружен</div></div>"
        )

    sorted_keys = sorted(
        counts,
        key=lambda k: SEVERITY_ORDER.get(
            SUBCLASS_DICT[k].severity if k in SUBCLASS_DICT else "low", 99
        ),
    )

    cards = "".join(
        f'<div style="border-radius:12px;padding:16px 18px;flex:1 1 210px;'
        f'border-left:5px solid {e.color};background:#161622;color:#cdd6f4;'
        f'font-family:sans-serif;box-shadow:0 2px 16px rgba(0,0,0,.5)">'
        f'<div style="font-size:2rem">{e.icon}</div>'
        f'<div style="font-size:1.05rem;font-weight:700;margin:4px 0">{e.ru_name}</div>'
        f'<div style="font-size:.78rem;color:#a6adc8;margin-bottom:6px">{e.description}</div>'
        f'<span style="display:inline-block;border-radius:6px;padding:2px 8px;'
        f'font-size:.7rem;font-weight:700;color:#fff;'
        f'background:{"#dc2626" if e.severity=="critical" else "#b91c1c" if e.severity=="high" else "#d97706" if e.severity=="medium" else "#6b7280"}">'
        f'{e.severity.upper()}</span>'
        f'&nbsp;<span style="font-size:1.4rem;font-weight:800;color:{e.color}">{counts[k]}</span>'
        f'<span style="font-size:.8rem;color:#a6adc8"> эпизодов</span></div>'
        for k in sorted_keys
        if (e := SUBCLASS_DICT.get(k))
    )

    total = sum(counts.values())
    meta = (
        f"<div style='margin-bottom:14px;padding:12px 16px;background:#161622;"
        f"border-radius:10px;font-family:sans-serif;color:#cdd6f4;"
        f"display:flex;gap:24px;flex-wrap:wrap;font-size:.88rem'>"
        f"<span>🎬 <b>Длительность:</b> {report.sourceInfo.video_duration_formatted}</span>"
        f"<span>⏱ <b>Обработано за:</b> {report.sourceInfo.processing_time_formatted}</span>"
        f"<span>⚡ <b>{report.sourceInfo.processing_efficiency:.2f}× от реального времени</b></span>"
        f"<span>🔎 <b>Событий:</b> {total}</span>"
        f"</div>"
    )
    return meta + '<div style="display:flex;flex-wrap:wrap;gap:12px;padding:8px 0">' + cards + "</div>"


def _timeline_html(report: TimeBasedReport) -> str:
    if not report.detections:
        return ""
    dur = report.sourceInfo.video_duration_seconds or 1
    max_ef = max((d.endFrame for d in report.detections), default=1)
    fps_approx = max_ef / dur if dur else 30.0

    bars = "".join(
        f'<div title="{(e := SUBCLASS_DICT.get(d.subclass.value)) and e.ru_name or d.subclass.value}: '
        f'{d.time_interval} (conf {d.confidence:.2f})" '
        f'style="position:absolute;left:{d.startFrame / (dur * fps_approx) * 100:.2f}%;'
        f'width:{max(0.5, (d.endFrame - d.startFrame) / (dur * fps_approx) * 100):.2f}%;'
        f'height:100%;background:{e.color if e else "#888"};opacity:.85;border-radius:3px"></div>'
        for d in report.detections
        if True
    )

    rows = "".join(
        f"<tr style='border-bottom:1px solid #1e1e3a'>"
        f"<td style='padding:6px 8px'>{(e := SUBCLASS_DICT.get(d.subclass.value)) and e.icon or ''}</td>"
        f"<td style='padding:6px 8px;color:{e.color if e else '#fff'};font-weight:600'>"
        f"{e.ru_name if e else d.subclass.value}</td>"
        f"<td style='padding:6px 8px;color:#a6adc8'>{d.time_interval}</td>"
        f"<td style='padding:6px 8px'>{'🎥 видео' if d.type.value == 'video' else '🎙 аудио'}</td>"
        f"<td style='padding:6px 8px'>"
        f"<span style='border:1px solid {e.color if e else '#6b7280'};color:{e.color if e else '#fff'};"
        f"border-radius:5px;padding:1px 7px;font-size:.78rem'>{d.confidence:.2f}</span></td></tr>"
        for d in report.detections
    )

    return (
        "<h4 style='color:#cdd6f4;font-family:sans-serif;margin-bottom:4px;font-size:.95rem'>"
        "Таймлайн событий</h4>"
        f"<div style='position:relative;height:22px;background:#161622;border-radius:5px;"
        f"overflow:hidden;margin:6px 0'>{bars}</div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:.83rem;"
        f"font-family:sans-serif;color:#cdd6f4;margin-top:8px'>"
        f"<thead style='background:#161622'><tr>"
        f"<th style='padding:6px 8px;text-align:left;color:#4b5563'></th>"
        f"<th style='padding:6px 8px;text-align:left;color:#4b5563'>Категория</th>"
        f"<th style='padding:6px 8px;text-align:left;color:#4b5563'>Интервал</th>"
        f"<th style='padding:6px 8px;text-align:left;color:#4b5563'>Источник</th>"
        f"<th style='padding:6px 8px;text-align:left;color:#4b5563'>Уверенность</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _tech_html(use_ocr: bool, use_clip: bool, whisper_model: str) -> str:
    wd = {
        "tiny": "Tiny — быстрый", "base": "Base — сбалансированный",
        "small": "Small — точный", "medium": "Medium — максимальный",
    }.get(whisper_model, whisper_model)
    chips = [
        ("#22c55e", f"🎤 Речь: Whisper {wd}"),
        ("#f59e0b", "📦 Объекты: YOLOv8n — бутылки, оружие"),
    ]
    if use_clip: chips.append(("#8b5cf6", "🧠 Сцены: CLIP zero-shot"))
    if use_ocr:  chips.append(("#06b6d4", "📝 Текст: EasyOCR — субтитры и надписи"))
    chips.append(("#6366f1", "⚡ Слияние: Temporal NMS — убирает дубли"))
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:8px">'
        + "".join(
            f'<div style="background:#161622;border:1px solid #2a2a3e;border-radius:20px;'
            f'padding:5px 12px;font-size:.78rem;color:#a6adc8;'
            f'display:flex;align-items:center;gap:6px">'
            f'<div style="width:8px;height:8px;border-radius:50%;background:{c};flex-shrink:0"></div>{l}</div>'
            for c, l in chips
        )
        + "</div>"
    )


# ── Основная функция анализа (генератор!) ─────────────────────────────────────
# Шаг 1: сразу показывает дерево прогресса
# Шаг 2: обновляет дерево каждые 2 сек + на каждый коллбэк из pipeline
# Шаг 3: в конце показывает результаты, прячет прогресс

async def analyze_url(
    url: str, quality: str, whisper_model: str, use_ocr: bool, use_clip: bool
):
    # Outputs:
    # 0  progress_out    gr.HTML
    # 1  summary_out     gr.HTML
    # 2  timeline_out    gr.HTML
    # 3  gallery_row     gr.Row
    # 4  gallery_out     gr.Gallery
    # 5  clip_state      gr.State
    # 6  tech_out        gr.HTML  (внутри аккордеона)
    # 7  json_accordion  gr.Accordion
    # 8  json_out        gr.JSON

    def _emit(progress_val, **kw):
        """Возвращает кортеж из 9 значений; незаполненные — gr.update()."""
        return (
            progress_val,
            kw.get("summary",    gr.update()),
            kw.get("timeline",   gr.update()),
            kw.get("gallery_row",gr.update()),
            kw.get("gallery",    gr.update()),
            kw.get("clips",      gr.update()),
            kw.get("tech",       gr.update()),
            kw.get("json_acc",   gr.update()),
            kw.get("json",       gr.update()),
        )

    # ── Валидация ────────────────────────────────────────────────────────────
    if not url or not url.strip():
        yield _emit(_error_html("Введите ссылку на YouTube"))
        return
    if "youtu" not in url:
        yield _emit(_error_html("Нужна ссылка на YouTube (youtube.com или youtu.be)"))
        return

    # ── Инициализация ────────────────────────────────────────────────────────
    eta_total = _estimate_eta(use_ocr, use_clip, whisper_model)
    t_start = time.time()
    queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(frac: float, msg: str) -> None:
        await queue.put((frac, msg))

    # Первый yield: сразу показываем дерево, прячем результаты
    yield _emit(
        _progress_html(0.02, "Начинаю анализ...", t_start, eta_total, use_clip, use_ocr),
        summary=gr.update(visible=False, value=""),
        timeline=gr.update(visible=False, value=""),
        gallery_row=gr.update(visible=False),
        gallery=gr.update(value=[]),
        clips=[],
        json_acc=gr.update(visible=False),
        json=gr.update(value={}),
    )

    # ── Запуск анализа в фоне ────────────────────────────────────────────────
    analyzer = _make_analyzer(whisper_model, use_ocr, use_clip)
    task = asyncio.create_task(
        analyzer.run(url, quality=quality, on_progress=on_progress)
    )

    frac = 0.02
    phrase = "Начинаю анализ..."

    # ── Стриминг прогресса ───────────────────────────────────────────────────
    # Ждём коллбэк из pipeline (timeout=2 сек → меняем фразу и снова ждём)
    while not task.done():
        try:
            new_frac, new_msg = await asyncio.wait_for(queue.get(), timeout=2.0)
            frac, phrase = new_frac, new_msg
        except asyncio.TimeoutError:
            # Pipeline молчит → вращаем фразы
            phrase = _rand_phrase(frac)

        yield _emit(
            _progress_html(frac, phrase, t_start, eta_total, use_clip, use_ocr)
        )

    # ── Получаем результат ───────────────────────────────────────────────────
    try:
        report, media_list = await task
    except Exception as exc:
        yield _emit(_error_html(str(exc)[:300]))
        return

    # ── Строим галерею ───────────────────────────────────────────────────────
    gallery_items = []
    for m in media_list:
        if m.frame:
            entry = SUBCLASS_DICT.get(m.detection.subclass.value)
            icon  = entry.icon if entry else ""
            name  = entry.ru_name if entry else m.detection.subclass.value
            label = f"{icon} {name} · {m.detection.time_interval} · {m.detection.confidence:.2f}"
            gallery_items.append((m.frame, label))

    clip_paths = [str(m.clip_path) if m.clip_path else None for m in media_list]

    # ── Финальный yield: прячем прогресс, показываем результаты ─────────────
    yield _emit(
        gr.update(value="", visible=False),          # прячем дерево
        summary=gr.update(value=_summary_html(report),   visible=True),
        timeline=gr.update(value=_timeline_html(report), visible=True),
        gallery_row=gr.update(visible=True),
        gallery=gallery_items,
        clips=clip_paths,
        tech=_tech_html(use_ocr, use_clip, whisper_model),
        json_acc=gr.update(visible=True),
        json=report.model_dump(mode="json"),
    )


# ── Обработчик клика по кадру в галерее ──────────────────────────────────────

def _on_gallery_select(media_state: list, evt: gr.SelectData):
    idx = evt.index
    if idx < len(media_state):
        p = media_state[idx]
        if p and Path(p).exists():
            return gr.update(value=p, visible=True)
    return gr.update(value=None, visible=False)


# ── CSS ───────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
body, .gradio-container { background: #0d0d14 !important; }

#cv-analyze-btn button {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    border: none !important; color: #fff !important;
    font-size: 1.05rem !important; font-weight: 700 !important;
    border-radius: 10px !important; padding: 13px 32px !important;
    cursor: pointer !important; transition: opacity .2s !important;
    box-shadow: 0 4px 20px rgba(34,197,94,.3) !important;
}
#cv-analyze-btn button:hover { opacity:.88 !important; }
"""


# ── Построение интерфейса ─────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CV Video Analytics") as demo:

        # ── Заголовок ────────────────────────────────────────────────────────
        gr.HTML("""
<div style="text-align:center;padding:28px 0 8px;font-family:sans-serif">
  <div style="font-size:2rem;font-weight:800;color:#e2e8f0;letter-spacing:-.5px">
    CV Video Analytics
  </div>
  <div style="color:#4b5563;font-size:.88rem;margin-top:6px">
    Нейросетевой анализ видео · 9 категорий нарушений
  </div>
</div>""")

        # ── Ввод ─────────────────────────────────────────────────────────────
        with gr.Row():
            url_input = gr.Textbox(
                label="Ссылка на YouTube",
                placeholder="https://www.youtube.com/watch?v=...",
                scale=5,
            )
            quality_dd = gr.Dropdown(
                choices=[
                    ("360p — быстро", "360p"),
                    ("480p — стандартное", "480p"),
                    ("720p — рекомендуется", "720p"),
                    ("1080p — высокое", "1080p"),
                ],
                value="720p", label="Качество", scale=2,
            )

        with gr.Row():
            whisper_dd = gr.Dropdown(
                choices=[
                    ("Tiny — очень быстро", "tiny"),
                    ("Base — скорость и точность", "base"),
                    ("Small — высокая точность", "small"),
                    ("Medium — максимум", "medium"),
                ],
                value="tiny", label="Распознавание речи (Whisper)",
                info="Больше модель → точнее речь, но дольше", scale=3,
            )
            ocr_cb = gr.Checkbox(
                value=False, label="Текст в кадре (OCR)",
                info="Медленно — 2-4 сек/кадр. Субтитры и баннеры.", scale=2,
            )
            clip_cb = gr.Checkbox(
                value=True, label="Анализ сцен (CLIP)",
                info="Понимает контекст всего кадра. Умеренно.", scale=2,
            )

        # ── Кнопки ───────────────────────────────────────────────────────────
        with gr.Row():
            analyze_btn = gr.Button(
                "▶ Запустить анализ", variant="primary",
                elem_id="cv-analyze-btn", scale=3,
            )
            cancel_btn = gr.Button("⬛ Остановить", variant="stop", scale=1)

        gr.HTML("<hr style='border-color:#1e1e3a;margin:14px 0'>")

        # ── Блок прогресса (показывается во время анализа) ───────────────────
        # Изначально — заглушка с подсказкой, затем — дерево этапов
        progress_out = gr.HTML(value=_PLACEHOLDER_HTML)

        # ── Результаты (скрыты до завершения анализа) ────────────────────────
        summary_out  = gr.HTML(visible=False)
        timeline_out = gr.HTML(visible=False)

        with gr.Row(visible=False) as gallery_row:
            with gr.Column(scale=2):
                gallery_out = gr.Gallery(
                    label="Обнаруженные моменты (нажмите кадр — откроется вырезка)",
                    columns=3, height=420, object_fit="cover",
                )
            with gr.Column(scale=1):
                clip_player = gr.Video(label="Видеовырезка", visible=False, height=320)
                gr.HTML(
                    "<div style='color:#374151;font-size:.78rem;text-align:center;margin-top:4px'>"
                    "Нажмите на кадр слева</div>"
                )

        with gr.Accordion("Что анализируем и как", open=False):
            tech_out = gr.HTML(
                "<div style='color:#4b5563;font-size:.85rem;font-family:sans-serif'>"
                "Запустите анализ — здесь появятся активные модули.</div>"
            )

        with gr.Accordion("Полный JSON-отчёт", open=False, visible=False) as json_accordion:
            json_out = gr.JSON()

        clip_state = gr.State([])

        # ── События ──────────────────────────────────────────────────────────
        run_event = analyze_btn.click(
            fn=analyze_url,
            inputs=[url_input, quality_dd, whisper_dd, ocr_cb, clip_cb],
            outputs=[
                progress_out,   # 0
                summary_out,    # 1
                timeline_out,   # 2
                gallery_row,    # 3
                gallery_out,    # 4
                clip_state,     # 5
                tech_out,       # 6
                json_accordion, # 7
                json_out,       # 8
            ],
        )

        cancel_btn.click(fn=None, cancels=[run_event])

        gallery_out.select(
            fn=_on_gallery_select,
            inputs=[clip_state],
            outputs=[clip_player],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.green,
            neutral_hue=gr.themes.colors.slate,
        ),
    )
