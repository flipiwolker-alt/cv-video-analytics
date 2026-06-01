"""Единый логгер сервиса: видно, какая стадия пайплайна сейчас идёт.

Пишет в консоль И в файл outputs/logs/service.log (с ротацией).
Формат: время | уровень | стадия | сообщение.

Использование:
    from .logs import get_logger
    log = get_logger(__name__)
    log.info("...")

    from .logs import stage
    with stage(log, "yolo", "детекция объектов"):
        ...   # на входе и выходе печатается старт/финиш + время
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_LOG_DIR = _ROOT / "outputs" / "logs"
_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger("cv")
    root.setLevel(logging.INFO)
    root.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        fileh = RotatingFileHandler(
            _LOG_DIR / "service.log", maxBytes=2_000_000, backupCount=3,
            encoding="utf-8",
        )
        fileh.setFormatter(fmt)
        root.addHandler(fileh)
    except (PermissionError, OSError):
        pass  # без файла — только консоль


def get_logger(name: str = "cv") -> logging.Logger:
    """Логгер под пространством имён 'cv.*' (короткое имя из __name__)."""
    _configure()
    short = name.split(".")[-1] if name else "cv"
    return logging.getLogger(f"cv.{short}")


@contextmanager
def stage(log: logging.Logger, name: str, what: str = ""):
    """Логирует старт/финиш стадии и затраченное время."""
    label = f"[{name}] {what}".rstrip()
    log.info("▶ %s — старт", label)
    t0 = time.time()
    try:
        yield
    except Exception as exc:
        log.exception("✖ %s — ошибка через %.1fс: %s", label, time.time() - t0, exc)
        raise
    else:
        log.info("✔ %s — готово за %.1fс", label, time.time() - t0)
