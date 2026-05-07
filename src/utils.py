import os
import cv2
from pathlib import Path
from rapidfuzz import fuzz, process


def extract_frames(video_path: str, output_dir: str, step: int = 30) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    saved = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(path, frame)
            saved.append(path)
        idx += 1
    cap.release()
    return saved


def deduplicate_texts(texts: list[str], threshold: int = 85) -> list[str]:
    unique: list[str] = []
    for t in texts:
        if not unique:
            unique.append(t)
            continue
        best = process.extractOne(t, unique, scorer=fuzz.ratio)
        if best is None or best[1] < threshold:
            unique.append(t)
    return unique


def merge_detections(df, window: int = 5):
    """Merge YOLO detections into temporal events per class."""
    import pandas as pd

    def _merge(group):
        group = group.sort_values("frame_num").reset_index(drop=True)
        events = []
        start = group.iloc[0]
        prev = start["frame_num"]
        for _, row in group.iloc[1:].iterrows():
            if row["frame_num"] - prev > window:
                events.append({
                    "class": start["class"],
                    "start_frame": start["frame_num"],
                    "end_frame": prev,
                    "avg_conf": round(group["conf"].mean(), 3),
                })
                start = row
            prev = row["frame_num"]
        events.append({
            "class": start["class"],
            "start_frame": start["frame_num"],
            "end_frame": prev,
            "avg_conf": round(group["conf"].mean(), 3),
        })
        return pd.DataFrame(events)

    return df.groupby("class", group_keys=False).apply(_merge).reset_index(drop=True)
