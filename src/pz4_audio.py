"""PZ4: Реальный аудио-анализ.

Этапы:
  1. ffmpeg → WAV (моно, 16 kHz)
  2. faster-whisper → транскрипт с тайм-кодами сегментов
  3. Keyword-классификатор → список Detection

Модель скачивается автоматически (~145 MB для base, ~39 MB для tiny).
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import imageio_ffmpeg

from .schemas import Detection, DetectionType, Subclass

# ── Словарь ключевых слов по субклассам ──────────────────────────────────────
# Русские + английские термины

_KEYWORDS: dict[str, list[str]] = {
    "alcohol": [
        # ru
        "пиво", "вино", "водка", "виски", "коньяк", "алкоголь", "спирт",
        "выпить", "выпьем", "пьяный", "бухать", "бухло", "напиться",
        "бухаем", "шампанское", "ром", "джин", "текила", "похмелье",
        "перебрал", "рюмка", "напился", "выпивать", "пьём", "напитки",
        # en
        "beer", "wine", "vodka", "whiskey", "whisky", "alcohol", "drunk",
        "drinking", "booze", "cocktail", "liquor", "champagne", "rum", "gin",
        "tequila", "hangover", "shots", "bartender", "pub", "cheers",
    ],
    "drugs": [
        # ru
        "наркотик", "героин", "кокаин", "марихуана", "трава", "ширяться",
        "ширнуться", "дурь", "кайф", "обкуриться", "колоться", "амфетамин",
        "экстази", "закинуть", "крек", "мефедрон", "спайс", "закинулся",
        "нарк", "наркоман", "передоз",
        # en
        "drug", "drugs", "cocaine", "heroin", "marijuana", "weed", "meth",
        "amphetamine", "ecstasy", "overdose", "narcotic", "dope", "crack",
        "high", "stoned", "junkie",
    ],
    "smoking": [
        # ru
        "курить", "сигарета", "папироса", "вейп", "кальян", "затянуться",
        "покурить", "курево", "смолить", "табак", "сигара", "никотин",
        # en
        "smoke", "smoking", "cigarette", "cigar", "vape", "tobacco",
        "nicotine", "hookah", "inhale", "puff",
    ],
    "extremism": [
        # ru
        "фашизм", "нацизм", "экстремизм", "джихад", "террор", "расизм",
        "скинхед", "ненависть", "геноцид", "убивать неверных",
        "национал-социализм", "белая раса",
        # en
        "fascism", "nazism", "extremism", "jihad", "terror", "racism",
        "genocide", "white supremacy", "neo-nazi",
    ],
    "ludomania": [
        # ru
        "казино", "ставка", "рулетка", "покер", "букмекер", "джекпот",
        "слот", "ставки", "тотализатор", "лудоман", "азартные", "пари",
        "выиграл в казино", "проиграл всё",
        # en
        "casino", "bet", "betting", "roulette", "poker", "jackpot",
        "slot machine", "gambling", "bookmaker", "wager", "odds",
    ],
    "lgbt": [
        # ru (в контексте пропаганды несовершеннолетним)
        "гей-парад", "лгбт пропаганда", "однополый брак детям",
        # en
        "gay propaganda", "lgbt propaganda",
    ],
    "violence": [
        # ru
        "убить", "убийство", "насилие", "избить", "зарезать", "застрелить",
        "труп", "расправа", "пытка", "истязание", "бандит", "мочить",
        # en
        "kill", "murder", "violence", "shoot", "stab", "assault",
        "brutal", "torture", "massacre", "beat up",
    ],
    "vandalism": [
        # ru
        "поджечь", "граффити", "вандализм", "порча имущества",
        "разгромить", "разбить витрину",
        # en
        "vandalism", "graffiti", "arson", "smash", "destroy property",
        "defacement",
    ],
    "profanity": [
        # ru — нецензурная лексика (транслитерация и вариации для regex)
        "блять", "блядь", "бляд", "бля", "сука", "суки", "сучка",
        "ёбаный", "ёбаная", "ёб", "ёбут", "ёбал", "ёбала",
        "пиздец", "пизда", "пизды", "пиздёж", "пиздит", "пиздёт",
        "хуй", "хуйня", "хуёво", "хуёвый", "нахуй", "похуй", "нихуя",
        "ёбтвоюмать", "еблан", "ебло",
        "мудак", "мудаки", "мудила",
        "долбоёб", "долбоёбы",
        "залупа", "залупон",
        "пиздабол", "пиздаболы",
        "ёбнутый", "ёбнутая",
        "охуел", "охуела", "охуели", "охуеть",
        "нахуй", "идинахуй",
        "выёбываться", "выёбывается",
        "заебал", "заебала", "заебали", "заебись",
        "ёбаный рот", "мать твою",
        # en
        "fuck", "fucking", "fucked", "fucker", "motherfucker",
        "shit", "bullshit", "shitty",
        "bitch", "bitches",
        "asshole", "ass hole",
        "cunt", "bastard",
        "damn", "goddamn",
        "dickhead", "dick",
        "prick", "twat",
        "whore", "slut",
    ],
}

# Предкомпиляция паттернов для скорости
_PATTERNS: dict[str, re.Pattern] = {
    cls: re.compile(
        r"\b(" + "|".join(re.escape(kw) for kw in kws) + r")\b",
        re.IGNORECASE | re.UNICODE,
    )
    for cls, kws in _KEYWORDS.items()
}

# Минимальный confidence для разного числа совпадений
_CONF_BY_HITS = {1: 0.55, 2: 0.72, 3: 0.85}


def _classify_segment(text: str) -> list[tuple[str, float]]:
    """Возвращает список (subclass, confidence) для одного текстового сегмента."""
    results = []
    for cls, pattern in _PATTERNS.items():
        hits = pattern.findall(text)
        if hits:
            n = len(hits)
            conf = _CONF_BY_HITS.get(min(n, 3), 0.92)
            results.append((cls, conf))
    return results


def _fmt_clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _extract_audio_wav(video_path: Path, out_wav: Path) -> None:
    """Извлекает аудио из видео в WAV моно 16kHz."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(video_path),
            "-vn",                    # без видео
            "-ar", "16000",           # 16 kHz — оптимально для Whisper
            "-ac", "1",               # моно
            "-f", "wav",
            str(out_wav),
        ],
        check=True,
        capture_output=True,
    )


def transcribe_and_classify(
    video_path: Path,
    fps: float,
    model_size: str = "base",
    language: Optional[str] = None,
) -> list[Detection]:
    """
    Главная функция PZ4.
    model_size: "tiny" (~39 MB), "base" (~145 MB), "small" (~461 MB)
    language: None = автоопределение, "ru" = только русский, "en" = только английский
    """
    from faster_whisper import WhisperModel

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.wav"
        _extract_audio_wav(video_path, audio_path)

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,           # пропускаем тишину
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,     # нам достаточно segment-level timestamps
        )

        detections: list[Detection] = []
        for seg in segments_gen:
            if not seg.text.strip():
                continue
            hits = _classify_segment(seg.text)
            for subclass, conf in hits:
                sf = int(seg.start * fps)
                ef = int(seg.end * fps)
                detections.append(Detection(
                    startFrame=sf,
                    endFrame=ef,
                    start_time=_fmt_clock(seg.start),
                    end_time=_fmt_clock(seg.end),
                    time_interval=f"{_fmt_clock(seg.start)} - {_fmt_clock(seg.end)}",
                    subclass=Subclass(subclass),
                    confidence=conf,
                    type=DetectionType.audio,
                ))

    return _merge_nearby(detections)


def _merge_nearby(
    detections: list[Detection],
    gap_sec: float = 2.0,
    fps: float = 30.0,
) -> list[Detection]:
    """Объединяет детекции одного класса с зазором < gap_sec."""
    if not detections:
        return []

    by_class: dict[str, list[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.subclass.value, []).append(d)

    result = []
    gap_frames = int(gap_sec * fps)

    for cls, dets in by_class.items():
        dets = sorted(dets, key=lambda d: d.startFrame)
        merged = [dets[0]]
        for d in dets[1:]:
            last = merged[-1]
            if d.startFrame - last.endFrame <= gap_frames:
                # расширяем предыдущую детекцию
                new_ef = max(last.endFrame, d.endFrame)
                new_conf = max(last.confidence, d.confidence)
                merged[-1] = Detection(
                    startFrame=last.startFrame,
                    endFrame=new_ef,
                    start_time=last.start_time,
                    end_time=_fmt_clock(new_ef / fps),
                    time_interval=f"{last.start_time} - {_fmt_clock(new_ef / fps)}",
                    subclass=last.subclass,
                    confidence=new_conf,
                    type=DetectionType.audio,
                )
            else:
                merged.append(d)
        result.extend(merged)

    return sorted(result, key=lambda d: d.startFrame)
