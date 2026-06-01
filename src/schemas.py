"""Pydantic-схемы входа и выхода API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class Subclass(str, Enum):
    alcohol = "alcohol"
    drugs = "drugs"
    smoking = "smoking"
    extremism = "extremism"
    ludomania = "ludomania"
    lgbt = "lgbt"
    violence = "violence"
    vandalism = "vandalism"
    profanity = "profanity"
    nsfw = "nsfw"
    selfharm = "selfharm"
    animal_cruelty = "animal_cruelty"


class DetectionType(str, Enum):
    video = "video"
    audio = "audio"


class Detection(BaseModel):
    startFrame: int
    endFrame: int
    start_time: str
    end_time: str
    time_interval: str
    subclass: Subclass
    confidence: float = Field(ge=0.0, le=1.0)
    type: DetectionType


class SourceInfo(BaseModel):
    frameCount: int = 0
    fps: float = 0.0
    video_path: str = ""
    video_duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    processing_efficiency: float = 0.0
    video_duration_formatted: str = "0:00:00"
    processing_time_formatted: str = "0:00:00"
    analysis_timestamp: str = ""


class TopSourceInfo(BaseModel):
    frameCount: int
    fps: float
    video_duration_formatted: str
    analysis_timestamp: str


class TimeBasedReport(BaseModel):
    report_type: Literal["TIME_BASED_REPORT"] = "TIME_BASED_REPORT"
    source: Optional[str] = None          # ссылка на YouTube или путь к локальному файлу
    source_dir: Optional[str] = None      # папка, где лежит ролик (для локальных файлов)
    source_info: TopSourceInfo
    detections: list[Detection]
    sourceInfo: SourceInfo
    preset: Optional[str] = None
    llm_summary: Optional[str] = None


class AnalyzeRequest(BaseModel):
    url: HttpUrl
    quality: str = "720p"
    preset: str = "balanced"
    use_llm: bool = True
    keyframe_stride_sec: float = 1.0


class TaskStatus(str, Enum):
    queued = "queued"
    downloading = "downloading"
    extracting = "extracting"
    audio = "audio"
    video = "video"
    aggregating = "aggregating"
    done = "done"
    failed = "failed"


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    created_at: str
    finished_at: Optional[str] = None
    report: Optional[TimeBasedReport] = None
