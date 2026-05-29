"""PZ4: Аудио-анализ.

Этапы:
  1. ffmpeg → WAV (моно, 16 kHz)
  2. faster-whisper → транскрипт с тайм-кодами (размер модели по пресету)
  3. Классификация сегментов:
       • keyword-матчер (быстрый, морфология-устойчивый) — всегда
       • LLM-классификатор (контекстный, Ollama) — если доступен

Модель Whisper скачивается автоматически.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import imageio_ffmpeg
from rapidfuzz import fuzz

from .schemas import Detection, DetectionType, Subclass
from .presets import Preset, get_preset
from .device import WHISPER_DEVICE, gpu_guard

# ── Словарь ключевых слов по субклассам (ru + en) ─────────────────────────────
_KEYWORDS: dict[str, list[str]] = {
    "alcohol": [
        "пиво", "вино", "водка", "виски", "коньяк", "алкоголь", "спирт",
        "выпить", "выпьем", "пьяный", "бухать", "бухло", "напиться",
        "бухаем", "шампанское", "ром", "джин", "текила", "похмелье",
        "перебрал", "рюмка", "напился", "выпивать", "пьём", "напитки",
        "beer", "wine", "vodka", "whiskey", "whisky", "alcohol", "drunk",
        "drinking", "booze", "cocktail", "liquor", "champagne", "rum", "gin",
        "tequila", "hangover", "shots", "bartender", "pub", "cheers",
    ],
    "drugs": [
        "наркотик", "героин", "кокаин", "марихуана", "ширяться",
        "ширнуться", "обкуриться", "колоться", "амфетамин",
        "экстази", "мефедрон", "спайс", "закинулся",
        "наркоман", "передоз",
        "drug", "drugs", "cocaine", "heroin", "marijuana", "weed", "meth",
        "amphetamine", "ecstasy", "overdose", "narcotic", "dope", "crack",
        "stoned", "junkie",
    ],
    "smoking": [
        "курить", "сигарета", "папироса", "вейп", "кальян", "затянуться",
        "покурить", "курево", "смолить", "табак", "сигара", "никотин",
        "smoke", "smoking", "cigarette", "cigar", "vape", "tobacco",
        "nicotine", "hookah", "inhale",
    ],
    "extremism": [
        "фашизм", "нацизм", "экстремизм", "джихад", "террор", "расизм",
        "скинхед", "геноцид", "национал-социализм",
        "fascism", "nazism", "extremism", "jihad", "terror", "racism",
        "genocide", "supremacy",
    ],
    "ludomania": [
        "казино", "ставка", "рулетка", "покер", "букмекер", "джекпот",
        "слот", "тотализатор", "лудоман", "азартные",
        "casino", "bet", "betting", "roulette", "poker", "jackpot",
        "gambling", "bookmaker", "wager",
    ],
    "lgbt": [
        "гей-парад", "лгбт пропаганда", "однополый брак детям",
        "gay propaganda", "lgbt propaganda",
    ],
    "violence": [
        "убить", "убийство", "насилие", "избить", "зарезать", "застрелить",
        "расправа", "пытка", "истязание",
        "kill", "murder", "violence", "shoot", "stab", "assault",
        "brutal", "torture", "massacre",
    ],
    "vandalism": [
        "поджечь", "граффити", "вандализм", "разгромить",
        "vandalism", "graffiti", "arson", "defacement",
    ],
    "profanity": [
        "блять", "блядь", "бляд", "сука", "сучка",
        "ёбаный", "ёбаная", "ёбут", "ёбал", "ёбала",
        "пиздец", "пизда", "пизды", "пиздёж", "пиздит",
        "хуй", "хуйня", "хуёво", "хуёвый", "нахуй", "похуй", "нихуя",
        "еблан", "ебло", "мудак", "мудила",
        "долбоёб", "залупа", "пиздабол",
        "ёбнутый", "охуел", "охуела", "охуеть",
        "выёбываться", "заебал", "заебись",
        "fuck", "fucking", "fucked", "fucker", "motherfucker",
        "shit", "bullshit", "bitch", "asshole",
        "cunt", "bastard", "dickhead", "prick", "twat",
        "whore", "slut",
    ],
    "nsfw": [
        "порно", "порнуха", "порнография", "голый", "голая", "обнажённый",
        "обнажённая", "эротика", "интим", "нагота", "стриптиз",
        "porn", "porno", "nude", "naked", "explicit", "nsfw", "xxx",
        "striptease", "erotic", "pornography",
    ],
    "selfharm": [
        "суицид", "самоубийство", "покончить с собой", "повеситься",
        "вскрыть вены", "порезать себя", "селфхарм", "свести счёты с жизнью",
        "наложить на себя руки", "хочу умереть", "убить себя",
        "suicide", "kill myself", "self harm", "self-harm", "selfharm",
        "cut myself", "cutting myself", "hang myself", "end my life",
        "want to die",
    ],
    "animal_cruelty": [
        "живодёр", "мучить животное", "издеваться над животным",
        "жестокость к животным", "убить собаку", "бить кошку",
        "замучить животное", "пинать собаку",
        "animal cruelty", "animal abuse", "beat the dog", "kill the cat",
        "torture animal", "hurt the animal", "abuse animals",
    ],
}


def _norm(s: str) -> str:
    return s.lower().replace("ё", "е")


# Leet-замены: цифры/символы → буквы (для деобфускации «b!tch», «з0внись»).
_LEET = {
    "0": "о", "1": "i", "3": "е", "4": "a", "5": "s", "7": "t",
    "@": "a", "$": "s", "!": "i", "|": "i", "*": "", "€": "e",
}
_LEET_TABLE = str.maketrans(_LEET)
_MASK_RE = re.compile(r"[.\-_+`~^]")
_REPEAT_RE = re.compile(r"(.)\1{2,}")


def _deobfuscate(s: str) -> str:
    """Снимает простую маскировку: leet (b!tch→bitch), маскеры (з*ебись→зебись),
    схлопывает 3+ повтора буквы (сууука→суука). Для запиканного/звёздочного мата.
    """
    s = _norm(s).translate(_LEET_TABLE)
    s = _MASK_RE.sub("", s)
    return _REPEAT_RE.sub(r"\1\1", s)


def _build_pattern(kws: list[str]) -> re.Pattern:
    """Морфология-устойчивый паттерн.

    Длинные русские слова матчатся по СТЕМУ (стем + до 4 любых букв-суффиксов),
    чтобы ловить падежи/склонения: «сигарета» → «сигарету», «сигарет».
    Короткие слова — строго целиком (\\bслово\\b), чтобы «ром» не матчил «роман».
    """
    alts: list[str] = []
    for kw in kws:
        n = _norm(kw)
        is_cyr = bool(re.search(r"[а-я]", n))
        if len(n) >= 6 and is_cyr and " " not in n and "-" not in n:
            stem = re.escape(n[:-2])
            alts.append(rf"\b{stem}[а-я]{{0,4}}")
        else:
            alts.append(rf"\b{re.escape(n)}\b")
    return re.compile("|".join(alts), re.IGNORECASE | re.UNICODE)


_PATTERNS: dict[str, re.Pattern] = {cls: _build_pattern(kws) for cls, kws in _KEYWORDS.items()}

_CONF_BY_HITS = {1: 0.55, 2: 0.72, 3: 0.85}

# Деобфусцированные одиночные стемы мата для fuzzy-матча (ловит «з*ебись», опечатки).
_PROFANITY_FUZZY = sorted({
    _deobfuscate(k) for k in _KEYWORDS["profanity"]
    if " " not in k and len(_deobfuscate(k)) >= 5
})
_TOKEN_RE = re.compile(r"[a-zа-я]{5,}")


def _fuzzy_profanity(deobf_text: str) -> bool:
    """Нечёткое совпадение токена с матерным стемом (порог 90, близкая длина)."""
    for tok in _TOKEN_RE.findall(deobf_text):
        for kw in _PROFANITY_FUZZY:
            if abs(len(tok) - len(kw)) <= 2 and fuzz.ratio(tok, kw) >= 90:
                return True
    return False


def _classify_segment(text: str) -> list[tuple[str, float]]:
    """keyword-классификация сегмента (+ деобфускация и fuzzy для мата)."""
    t = _norm(text)
    deobf = _deobfuscate(text)
    per_class: dict[str, int] = {}
    for cls, pattern in _PATTERNS.items():
        hits = len(pattern.findall(t))
        if deobf != t:
            hits += len(pattern.findall(deobf))
        if hits:
            per_class[cls] = hits
    # fuzzy-добор для нецензурщины (звёздочки, leet, лёгкие опечатки)
    if "profanity" not in per_class and _fuzzy_profanity(deobf):
        per_class["profanity"] = 1
    return [(cls, _CONF_BY_HITS.get(min(n, 3), 0.92)) for cls, n in per_class.items()]


def _fmt_clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _extract_audio_wav(video_path: Path, out_wav: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(video_path), "-vn",
         "-ar", "16000", "-ac", "1", "-f", "wav", str(out_wav)],
        check=True, capture_output=True,
    )


def transcribe_and_classify(
    video_path: Path,
    fps: float,
    preset: Preset | str | None = None,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    use_llm: bool = False,
) -> list[Detection]:
    """PZ4. Транскрибирует речь и классифицирует сегменты.

    use_llm=True → добавляет контекстную LLM-классификацию (Ollama) поверх
    keyword-матчера; уверенность по субклассу берётся как максимум из двух.
    """
    from faster_whisper import WhisperModel

    p = preset if isinstance(preset, Preset) else get_preset(preset)
    size = model_size or p.whisper_model

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.wav"
        _extract_audio_wav(video_path, audio_path)

        # CUDA → float16, CPU/Mac → int8; доступ к GPU сериализуем
        compute = "float16" if WHISPER_DEVICE == "cuda" else "int8"
        with gpu_guard():
            model = WhisperModel(size, device=WHISPER_DEVICE, compute_type=compute)
            segments_gen, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=p.whisper_beam,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                word_timestamps=False,
            )
            segments = [s for s in segments_gen if s.text.strip()]

    # LLM доступен? (проверяем один раз)
    llm_on = False
    if use_llm:
        from . import llm as _llm
        llm_on = _llm.is_available(p.llm_text_model)

    detections: list[Detection] = []
    for seg in segments:
        # уверенность по субклассу: max(keyword, llm)
        per_class: dict[str, float] = {}
        for sub, conf in _classify_segment(seg.text):
            per_class[sub] = max(per_class.get(sub, 0.0), conf)
        if llm_on:
            from . import llm as _llm
            for sub, conf, _quote in _llm.classify_transcript_segment(seg.text, p.llm_text_model):
                per_class[sub] = max(per_class.get(sub, 0.0), conf)

        sf, ef = int(seg.start * fps), int(seg.end * fps)
        for sub, conf in per_class.items():
            detections.append(Detection(
                startFrame=sf, endFrame=ef,
                start_time=_fmt_clock(seg.start), end_time=_fmt_clock(seg.end),
                time_interval=f"{_fmt_clock(seg.start)} - {_fmt_clock(seg.end)}",
                subclass=Subclass(sub),
                confidence=round(conf, 3),
                type=DetectionType.audio,
            ))

    return _merge_nearby(detections, fps=fps)


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
