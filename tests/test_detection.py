from __future__ import annotations

from oracle_report.vision.detection import _restore_face_box, resolve_haar_cascade_path


class FakeCv2WithoutData:
    pass


def test_restore_face_box_maps_downscaled_detection_to_original_frame() -> None:
    face = _restore_face_box(20, 30, 50, 60, 0.5)

    assert face.x == 40
    assert face.y == 60
    assert face.width == 100
    assert face.height == 120


def test_resolve_haar_cascade_path_without_cv2_data(monkeypatch, tmp_path) -> None:
    cascade_dir = tmp_path / "haarcascades"
    cascade_dir.mkdir()
    cascade_path = cascade_dir / "haarcascade_frontalface_default.xml"
    cascade_path.write_text("<opencv_storage></opencv_storage>", encoding="utf-8")
    monkeypatch.setenv("ORACLE_HAAR_CASCADE_DIR", str(cascade_dir))

    resolved_path = resolve_haar_cascade_path(
        FakeCv2WithoutData(),
        "haarcascade_frontalface_default.xml",
    )

    assert resolved_path == cascade_path
