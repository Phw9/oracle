from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0


@dataclass(frozen=True)
class FaceQuality:
    ready: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)
    eye_count: int = 0
    eyebrow_score: float = 0.0


@dataclass(frozen=True)
class CaptureDecision:
    state: str
    elapsed_seconds: float
    face: FaceBox | None
    quality: FaceQuality | None
    should_capture: bool
    message: str


@dataclass(frozen=True)
class CaptureArtifact:
    image_path: Path
    face: FaceBox
    captured_at: datetime
    quality: FaceQuality


@dataclass(frozen=True)
class SequentialPairCaptureArtifact:
    left: CaptureArtifact
    right: CaptureArtifact


@dataclass(frozen=True)
class BirthProfile:
    name: str
    birth_datetime: datetime
    timezone: str = "Asia/Seoul"
    gender: str = ""
    birth_time_known: bool = True


@dataclass(frozen=True)
class ReportArtifact:
    markdown: str
    output_path: Path | None
