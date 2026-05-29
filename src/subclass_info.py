"""Словарь субклассов нежелательного контента."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubclassEntry:
    icon: str
    ru_name: str
    description: str
    severity: str         # low / medium / high / critical
    color: str            # hex для UI


SUBCLASS_DICT: dict[str, SubclassEntry] = {
    "alcohol": SubclassEntry(
        icon="🍺",
        ru_name="Алкоголь",
        description="Демонстрация или пропаганда употребления алкогольных напитков, распитие спиртного в кадре.",
        severity="medium",
        color="#f59e0b",
    ),
    "drugs": SubclassEntry(
        icon="💊",
        ru_name="Наркотики",
        description="Демонстрация, употребление или пропаганда запрещённых психоактивных веществ.",
        severity="critical",
        color="#ef4444",
    ),
    "smoking": SubclassEntry(
        icon="🚬",
        ru_name="Курение",
        description="Курение сигарет, кальяна, вейпа или иных табачных изделий в кадре.",
        severity="low",
        color="#6b7280",
    ),
    "extremism": SubclassEntry(
        icon="⚠️",
        ru_name="Экстремизм",
        description="Символика, лозунги или материалы экстремистского характера, призывы к насилию по идеологическим мотивам.",
        severity="critical",
        color="#dc2626",
    ),
    "ludomania": SubclassEntry(
        icon="🎰",
        ru_name="Азартные игры",
        description="Реклама или демонстрация азартных игр, казино, ставок, букмекерских контор.",
        severity="medium",
        color="#8b5cf6",
    ),
    "lgbt": SubclassEntry(
        icon="🏳️‍🌈",
        ru_name="ЛГБТ-пропаганда",
        description="Материалы, пропагандирующие нетрадиционные сексуальные отношения среди несовершеннолетних (согласно законодательству РФ).",
        severity="high",
        color="#ec4899",
    ),
    "violence": SubclassEntry(
        icon="👊",
        ru_name="Насилие",
        description="Сцены физического насилия, жестокости, агрессии по отношению к людям или животным.",
        severity="high",
        color="#b91c1c",
    ),
    "vandalism": SubclassEntry(
        icon="🔨",
        ru_name="Вандализм",
        description="Умышленное повреждение или уничтожение имущества, граффити, порча общественных объектов.",
        severity="medium",
        color="#d97706",
    ),
    "profanity": SubclassEntry(
        icon="🤬",
        ru_name="Нецензурная лексика",
        description="Использование ненормативной лексики, мата, оскорблений в аудио или текстовых надписях.",
        severity="medium",
        color="#f97316",
    ),
    "nsfw": SubclassEntry(
        icon="🔞",
        ru_name="Порнография / нагота",
        description="Откровенный сексуальный контент, порнография, обнажённое тело, эротические сцены.",
        severity="high",
        color="#be185d",
    ),
    "selfharm": SubclassEntry(
        icon="🩸",
        ru_name="Суицид / селфхарм",
        description="Демонстрация, описание или пропаганда самоубийства и членовредительства (порезы, повешение, прыжки).",
        severity="critical",
        color="#7f1d1d",
    ),
    "animal_cruelty": SubclassEntry(
        icon="🐾",
        ru_name="Жестокость к животным",
        description="Сцены жестокого обращения, насилия, истязания или убийства животных.",
        severity="high",
        color="#92400e",
    ),
}

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
