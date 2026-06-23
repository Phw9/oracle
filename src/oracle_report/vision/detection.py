from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from oracle_report.models import FaceBox


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        ...


_MIN_SCALED_FACE_SIZE_PX = 24


class HaarFaceDetector:
    def __init__(
        self,
        min_size_px: int = 96,
        detection_scale: float = 0.5,
        detection_interval: int = 2,
    ) -> None:
        self._cv2 = _import_cv2()
        cascade_path = self._resolve_cascade_path()
        self._cascade = self._cv2.CascadeClassifier(str(cascade_path))
        self._min_size_px = min_size_px
        self._detection_scale = detection_scale
        self._detection_interval = detection_interval
        self._frame_index = 0
        self._cached_faces: list[FaceBox] = []
        if self._cascade.empty():
            raise RuntimeError(f"failed to load face cascade: {cascade_path}")
        if detection_scale <= 0.0 or detection_scale > 1.0:
            raise ValueError("detection_scale must be > 0.0 and <= 1.0.")
        if detection_interval <= 0:
            raise ValueError("detection_interval must be greater than 0.")

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        should_run_detection = self._frame_index % self._detection_interval == 0
        self._frame_index = self._frame_index + 1
        if should_run_detection:
            self._cached_faces = self._detect_now(frame)
        result = list(self._cached_faces)
        return result

    def _detect_now(self, frame: np.ndarray) -> list[FaceBox]:
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        detection_gray = self._resize_for_detection(gray)
        scaled_min_size = max(
            _MIN_SCALED_FACE_SIZE_PX,
            int(round(self._min_size_px * self._detection_scale)),
        )
        faces = self._cascade.detectMultiScale(
            detection_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(scaled_min_size, scaled_min_size),
        )
        result = [
            _restore_face_box(x, y, w, h, self._detection_scale)
            for (x, y, w, h) in faces
        ]
        result.sort(key=lambda face: face.width * face.height, reverse=True)
        return result

    def _resize_for_detection(self, gray: np.ndarray) -> np.ndarray:
        result = gray
        if self._detection_scale < 1.0:
            result = self._cv2.resize(
                gray,
                (0, 0),
                fx=self._detection_scale,
                fy=self._detection_scale,
                interpolation=self._cv2.INTER_AREA,
            )
        return result

    def _resolve_cascade_path(self) -> Path:
        data_dir = Path(self._cv2.data.haarcascades)
        result = data_dir / "haarcascade_frontalface_default.xml"
        return result


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for camera capture. Install python3-opencv on "
            "Raspberry Pi or pip install -e '.[camera]'.",
        ) from exc
    result = cv2
    return result


def _restore_face_box(
    x: int,
    y: int,
    width: int,
    height: int,
    detection_scale: float,
) -> FaceBox:
    result = FaceBox(
        int(round(x / detection_scale)),
        int(round(y / detection_scale)),
        int(round(width / detection_scale)),
        int(round(height / detection_scale)),
        1.0,
    )
    return result
