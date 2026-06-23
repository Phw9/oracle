from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from oracle_report.config import CaptureConfig
from oracle_report.models import FaceBox
from oracle_report.vision.detection import HaarFaceDetector, _import_cv2
from oracle_report.vision.quality import OpenCvFaceQualityAnalyzer


def open_camera(config: CaptureConfig) -> tuple[Any, Any]:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(config.camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    capture.set(cv2.CAP_PROP_FPS, config.camera_fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open camera index {config.camera_index}")
    result = (cv2, capture)
    return result


def build_default_face_detector(config: CaptureConfig) -> HaarFaceDetector:
    result = HaarFaceDetector(
        min_size_px=config.face_min_size_px,
        detection_scale=config.face_detection_scale,
        detection_interval=config.face_detection_interval,
    )
    return result


def build_default_quality_analyzer(
    config: CaptureConfig,
) -> OpenCvFaceQualityAnalyzer:
    result = OpenCvFaceQualityAnalyzer(
        eye_min_count=config.eye_min_count,
        eyebrow_min_edge_density=config.eyebrow_min_edge_density,
    )
    return result


def draw_overlay(
    cv2: Any,
    frame: np.ndarray,
    message: str,
    faces: Sequence[FaceBox],
    warning: bool,
) -> None:
    color = (0, 180, 0)
    if warning:
        color = (0, 180, 255)
    for face in faces:
        cv2.rectangle(
            frame,
            (face.x, face.y),
            (face.x + face.width, face.y + face.height),
            color,
            2,
        )
    cv2.putText(
        frame,
        message,
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
