"""Межпроцессный лок на анализ: на одной 8-ГБ видеокарте одновременно должен
идти ТОЛЬКО ОДИН полный анализ.

Зачем: UI, API и benchmark — это РАЗНЫЕ процессы. Внутрипроцессный лок их не
координирует, а два анализа сразу не помещаются в 8 ГБ VRAM → оба упираются в
память и отваливаются по таймауту (а процесс может зависнуть, держа GPU).

Решение: файловый лок в кэше моделей. Второй анализ ждёт освобождения и встаёт
в очередь, а не конкурирует. Лок самолечится: если державший его процесс умер
(зависший/убитый UI), запись считается протухшей и перехватывается — по PID
через psutil.

Использование:
    from .gpu_lock import GpuSlot
    with GpuSlot(on_wait=lambda: log.info("жду освобождения GPU")):
        ...  # тяжёлый GPU-анализ
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

from .config import MODELS_DIR

_LOCKFILE = MODELS_DIR / ".gpu_analysis.lock"


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        # без psutil считаем живым (консервативно — лучше подождать, чем украсть)
        return True


def _read_owner() -> Optional[int]:
    try:
        return int(_LOCKFILE.read_text(encoding="utf-8").split(":", 1)[0])
    except (OSError, ValueError):
        return None


def _try_create() -> bool:
    """Атомарно создать lock-файл. True — захватили, False — занят."""
    try:
        fd = os.open(str(_LOCKFILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    except OSError:
        return False
    try:
        os.write(fd, f"{os.getpid()}:{time.time():.0f}".encode())
    finally:
        os.close(fd)
    return True


class GpuSlot:
    """Контекст-менеджер: захватывает межпроцессный слот анализа.

    on_wait вызывается один раз, если пришлось ждать (для лога/прогресса).
    poll — период опроса, сек.
    """

    def __init__(self, on_wait: Optional[Callable[[], None]] = None, poll: float = 1.0):
        self._on_wait = on_wait
        self._poll = poll
        self._held = False

    def __enter__(self) -> "GpuSlot":
        warned = False
        while True:
            if _try_create():
                self._held = True
                return self
            # занят — проверяем, не протух ли (державший процесс умер)
            owner = _read_owner()
            if owner is None or not _pid_alive(owner):
                # протухший лок — забираем
                try:
                    _LOCKFILE.unlink()
                except OSError:
                    pass
                continue
            if not warned:
                warned = True
                if self._on_wait:
                    try:
                        self._on_wait()
                    except Exception:
                        pass
            time.sleep(self._poll)

    def __exit__(self, *exc) -> None:
        if self._held and _read_owner() == os.getpid():
            try:
                _LOCKFILE.unlink()
            except OSError:
                pass
        self._held = False
