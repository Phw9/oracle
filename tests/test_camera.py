from __future__ import annotations

from pathlib import Path

import numpy as np

from oracle_report.config import CaptureConfig
from oracle_report.models import FaceBox
from oracle_report.vision import camera
from oracle_report.vision.camera import (
    _camera_candidate_indices,
    _configure_capture,
    build_capture_processors,
    draw_overlay,
    mirror_face_boxes,
    mirror_landmark_points,
)
from oracle_report.vision.framing import build_capture_guide


class FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 38


class FakeCapture:
    def __init__(self, backend_name: str, read_ok: bool = True) -> None:
        self._backend_name = backend_name
        self.set_calls: list[tuple[int, int]] = []
        self._opened = True
        self._read_ok = read_ok
        self.released = False

    def getBackendName(self) -> str:
        result = self._backend_name
        return result

    def set(self, property_id: int, value: int) -> bool:
        self.set_calls.append((property_id, value))
        result = True
        return result

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        if not self._read_ok:
            frame = None
        result = (self._read_ok, frame)
        return result

    def release(self) -> None:
        self.released = True


class FakeCv2Open:
    CAP_V4L2 = 200
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self, open_indices: set[int], readable_indices: set[int] | None = None) -> None:
        self.open_indices = open_indices
        self.readable_indices = open_indices if readable_indices is None else readable_indices
        self.calls: list[int] = []

    def VideoCapture(self, camera_index: int, backend=None):
        self.calls.append(camera_index)
        capture = FakeCapture("V4L2", read_ok=camera_index in self.readable_indices)
        capture._opened = camera_index in self.open_indices
        return capture


class FakeCv2WindowsOpen:
    CAP_DSHOW = 700
    CAP_MSMF = 1400
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self, open_backend: int) -> None:
        self.open_backend = open_backend
        self.calls: list[tuple[int, int | None]] = []

    def VideoCapture(self, camera_index: int, backend=None):
        self.calls.append((camera_index, backend))
        capture = FakeCapture("DSHOW")
        capture._opened = backend == self.open_backend
        return capture


class FakeDrawCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self) -> None:
        self.rectangle_calls: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        self.line_calls: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        self.circle_calls: list[tuple[tuple[int, int], int]] = []
        self.text_calls: list[str] = []

    def rectangle(
        self,
        frame,
        start: tuple[int, int],
        end: tuple[int, int],
        color,
        thickness: int,
    ) -> None:
        self.rectangle_calls.append((start, end, thickness))

    def line(
        self,
        frame,
        start: tuple[int, int],
        end: tuple[int, int],
        color,
        thickness: int,
    ) -> None:
        self.line_calls.append((start, end, thickness))

    def circle(self, frame, point: tuple[int, int], radius: int, color, thickness: int) -> None:
        self.circle_calls.append((point, radius))

    def putText(self, frame, message: str, *args) -> None:
        self.text_calls.append(message)


def test_configure_capture_skips_property_writes_for_gstreamer() -> None:
    capture = FakeCapture("GStreamer")

    _configure_capture(FakeCv2, capture, _capture_config())

    assert capture.set_calls == []


def test_configure_capture_applies_property_writes_for_v4l2() -> None:
    capture = FakeCapture("V4L2")

    _configure_capture(FakeCv2, capture, _capture_config())

    assert capture.set_calls == [
        (FakeCv2.CAP_PROP_FRAME_WIDTH, 640),
        (FakeCv2.CAP_PROP_FRAME_HEIGHT, 480),
        (FakeCv2.CAP_PROP_FPS, 15),
        (FakeCv2.CAP_PROP_BUFFERSIZE, 1),
    ]


def test_draw_overlay_shows_only_head_guide() -> None:
    cv2 = FakeDrawCv2()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    guide = build_capture_guide(frame.shape[1], frame.shape[0])

    draw_overlay(cv2, frame, "ready", [], False)

    assert len(cv2.rectangle_calls) == 2
    assert cv2.rectangle_calls[0][0] == (guide.head_box.x, guide.head_box.y)
    assert cv2.rectangle_calls[0][1] == (
        guide.head_box.x + guide.head_box.width,
        guide.head_box.y + guide.head_box.height,
    )
    assert len(cv2.line_calls) == 0


def test_draw_overlay_can_hide_head_guide_for_web_preview() -> None:
    cv2 = FakeDrawCv2()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    draw_overlay(cv2, frame, "ready", [], False, (), False)

    assert len(cv2.rectangle_calls) == 1
    assert cv2.rectangle_calls[0][0] == (0, 0)


def test_mirror_overlay_geometry_flips_boxes_and_landmarks() -> None:
    faces = mirror_face_boxes(640, [FaceBox(250, 20, 80, 100)])
    landmarks = mirror_landmark_points(640, [(250, 10), (300, 20)])

    assert faces == (FaceBox(310, 20, 80, 100),)
    assert landmarks == ((389, 10), (339, 20))


def test_camera_candidate_indices_prefer_configured_index() -> None:
    config = _capture_config()

    result = _camera_candidate_indices(config)

    assert result[:4] == (0, 1, 2, 3)


def test_camera_candidate_indices_can_disable_auto_detect() -> None:
    config = CaptureConfig(
        camera_index=2,
        frame_width=640,
        frame_height=480,
        camera_fps=15,
        min_face_seconds=2.0,
        face_min_size_px=96,
        face_detection_scale=0.5,
        face_detection_interval=2,
        output_dir=Path("runs"),
        show_preview=False,
        eye_min_count=2,
        eyebrow_min_edge_density=0.018,
        camera_auto_detect=False,
    )

    result = _camera_candidate_indices(config)

    assert result == (2,)


def test_open_camera_falls_back_to_next_device(monkeypatch) -> None:
    fake_cv2 = FakeCv2Open({1})

    monkeypatch.setattr(camera.os, "name", "posix")
    monkeypatch.setattr(camera, "_import_cv2", lambda: fake_cv2)

    cv2, capture = camera.open_camera(_capture_config())

    assert cv2 is fake_cv2
    assert fake_cv2.calls[:2] == [0, 0]
    assert 1 in fake_cv2.calls
    assert capture.isOpened() is True


def test_open_camera_skips_open_device_when_frame_read_fails(monkeypatch) -> None:
    fake_cv2 = FakeCv2Open({0, 1}, readable_indices={1})

    monkeypatch.setattr(camera.os, "name", "posix")
    monkeypatch.setattr(camera, "_import_cv2", lambda: fake_cv2)

    cv2, capture = camera.open_camera(_capture_config())

    assert cv2 is fake_cv2
    assert fake_cv2.calls[:3] == [0, 0, 1]
    assert capture.isOpened() is True


def test_open_camera_prefers_windows_directshow_backend(monkeypatch) -> None:
    fake_cv2 = FakeCv2WindowsOpen(FakeCv2WindowsOpen.CAP_DSHOW)

    monkeypatch.setattr(camera.os, "name", "nt")
    monkeypatch.setattr(camera, "_import_cv2", lambda: fake_cv2)

    cv2, capture = camera.open_camera(_capture_config())

    assert cv2 is fake_cv2
    assert fake_cv2.calls[0] == (0, FakeCv2WindowsOpen.CAP_DSHOW)
    assert capture.isOpened() is True


def test_open_camera_reports_permission_hint_for_inaccessible_video_devices(monkeypatch) -> None:
    fake_cv2 = FakeCv2Open(set())

    monkeypatch.setattr(camera.os, "name", "posix")
    monkeypatch.setattr(camera, "_import_cv2", lambda: fake_cv2)
    monkeypatch.setattr(camera, "_discover_video_device_paths", lambda: ["/dev/video0"])
    monkeypatch.setattr(camera.os, "access", lambda path, mode: False)

    try:
        camera.open_camera(_capture_config())
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected open_camera to fail")

    assert "attempted indices: 0, 1, 2, 3, 4, 5" in message
    assert "/dev/video0" in message
    assert "video group membership" in message


def test_build_capture_processors_can_use_opencv_mode(monkeypatch) -> None:
    monkeypatch.setenv("ORACLE_FACE_ANALYSIS_MODE", "1")

    detector, analyzer = build_capture_processors(_capture_config())

    assert detector.__class__.__name__ == "HaarFaceDetector"
    assert analyzer.__class__.__name__ == "OpenCvFaceQualityAnalyzer"


def test_build_capture_processors_falls_back_when_mediapipe_solutions_missing(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ORACLE_FACE_ANALYSIS_MODE", "2")

    class MissingSolutionsDetector:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("installed mediapipe package is missing solutions.face_mesh")

    monkeypatch.setattr(
        camera,
        "_build_mediapipe_capture_processors",
        lambda config: (_raise_missing_solutions(), None),
    )

    detector, analyzer = build_capture_processors(_capture_config())

    assert detector.__class__.__name__ == "HaarFaceDetector"
    assert analyzer.__class__.__name__ == "OpenCvFaceQualityAnalyzer"


def _raise_missing_solutions():
    raise RuntimeError("installed mediapipe package is missing solutions.face_mesh")


def _capture_config() -> CaptureConfig:
    result = CaptureConfig(
        camera_index=0,
        frame_width=640,
        frame_height=480,
        camera_fps=15,
        min_face_seconds=2.0,
        face_min_size_px=96,
        face_detection_scale=0.5,
        face_detection_interval=2,
        output_dir=Path("runs"),
        show_preview=False,
        eye_min_count=2,
        eyebrow_min_edge_density=0.018,
    )
    return result
