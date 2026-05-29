"""PZ1: Извлечение информативных keyframe-ов из видео.

Стратегия:
  1. ContentDetector (scenedetect) — кадры в точках смены сцены
  2. Равномерный stride — кадры каждые stride_sec секунд между сценами
  3. ВЫБОР КАЧЕСТВА (не порча пикселей — это сдвигает распределение и вредит CLIP):
       • из окрестности ±sharp_window берём САМЫЙ РЕЗКИЙ кадр (по дисперсии
         Лапласиана) — выкидываем смазанные движением;
       • near-duplicate дедуп по dHash — не тратим бюджет на одинаковые кадры.
  Итого детекторы получают резкие и разнообразные кадры тем же числом.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class Keyframe:
    frame_idx: int
    time_sec: float
    image: Image.Image


def _sharpness(gray: np.ndarray) -> float:
    """Дисперсия Лапласиана — мера резкости (выше = чётче, меньше смаза)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _dhash(gray: np.ndarray, size: int = 8) -> int:
    """Перцептивный hash (difference hash) для дедупа похожих кадров."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def extract_keyframes(
    video_path: Path,
    stride_sec: float = 1.0,
    scene_threshold: float = 27.0,
    sharp_window: int = 1,
    dedup_hamming: int = 5,
) -> list[Keyframe]:
    """
    stride_sec      — брать кадр каждые N секунд (между сменами сцен)
    scene_threshold — чувствительность ContentDetector (ниже = чувствительнее)
    sharp_window    — ±кадров вокруг цели для выбора самого резкого (0 = выкл)
    dedup_hamming   — порог похожести dHash для отсева near-duplicate (0 = выкл)
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        _use_scenedetect = True
    except ImportError:
        _use_scenedetect = False

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    scene_cut_frames: set[int] = set()
    if _use_scenedetect:
        try:
            video = open_video(str(video_path))
            manager = SceneManager()
            manager.add_detector(ContentDetector(threshold=scene_threshold))
            manager.detect_scenes(video, show_progress=False)
            for start, _ in manager.get_scene_list():
                scene_cut_frames.add(start.get_frames())
        except Exception:
            pass

    stride_frames = max(1, int(stride_sec * fps))
    targets: set[int] = set(range(0, total_frames, stride_frames))
    targets |= scene_cut_frames
    target_list = sorted(f for f in targets if f < total_frames)

    keyframes: list[Keyframe] = []
    cap = cv2.VideoCapture(str(video_path))
    last_hash: int | None = None

    for target in target_list:
        # ── Выбор самого резкого кадра из окрестности ────────────────────────
        best_frame = None
        best_idx = target
        best_sharp = -1.0
        for off in range(-sharp_window, sharp_window + 1):
            idx = target + off
            if idx < 0 or idx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = _sharpness(gray) if sharp_window > 0 else 1.0
            if sharp > best_sharp:
                best_sharp, best_idx, best_frame = sharp, idx, frame
            if sharp_window == 0:
                break  # окно выключено — берём ровно целевой кадр

        if best_frame is None:
            continue

        # ── Дедуп near-duplicate по dHash ────────────────────────────────────
        if dedup_hamming > 0:
            gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
            h = _dhash(gray)
            if last_hash is not None and _hamming(h, last_hash) <= dedup_hamming:
                continue
            last_hash = h

        img = Image.fromarray(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
        keyframes.append(Keyframe(frame_idx=best_idx, time_sec=best_idx / fps, image=img))

    cap.release()
    return keyframes
