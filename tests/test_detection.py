from __future__ import annotations

from oracle_report.vision.detection import _restore_face_box


def test_restore_face_box_maps_downscaled_detection_to_original_frame() -> None:
    face = _restore_face_box(20, 30, 50, 60, 0.5)

    assert face.x == 40
    assert face.y == 60
    assert face.width == 100
    assert face.height == 120
