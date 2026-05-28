"""PZ1: Извлечение информативных keyframe-ов из видео.

Стратегия:
  1. ContentDetector (scenedetect) — кадры в точках смены сцены
  2. Равномерный stride — кадры каждые stride_sec секунд между сценами
  Итого: ~100-400 кадров на 8-минутное видео (vs 14400 при анализе всех).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image


@dataclass
class Keyframe:
    frame_idx: int
    time_sec: float
    image: Image.Image


def extract_keyframes(
    video_path: Path,
    stride_sec: float = 1.0,
    scene_threshold: float = 27.0,
) -> list[Keyframe]:
    """
    stride_sec     — брать кадр каждые N секунд (между сменами сцен)
    scene_threshold — чувствительность ContentDetector (ниже = чувствительнее)
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
            scenes = manager.get_scene_list()
            for start, _ in scenes:
                scene_cut_frames.add(start.get_frames())
        except Exception:
            pass

    # Равномерный stride поверх
    stride_frames = max(1, int(stride_sec * fps))
    selected: set[int] = set(range(0, total_frames, stride_frames))
    selected |= scene_cut_frames
    selected = sorted(f for f in selected if f < total_frames)

    # Считываем кадры
    keyframes: list[Keyframe] = []
    cap = cv2.VideoCapture(str(video_path))
    for frame_idx in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        keyframes.append(Keyframe(
            frame_idx=frame_idx,
            time_sec=frame_idx / fps,
            image=img,
        ))
    cap.release()

    return keyframes
