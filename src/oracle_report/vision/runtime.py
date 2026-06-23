from __future__ import annotations

from pathlib import Path

from oracle_report.config import CaptureConfig
from oracle_report.models import CaptureArtifact, CaptureDecision
from oracle_report.vision.capture import FaceCaptureHarness, save_capture_artifact
from oracle_report.vision.camera import (
    build_default_face_detector,
    build_default_quality_analyzer,
    draw_overlay,
    open_camera,
)


def run_capture(config: CaptureConfig, output_dir: Path | None = None) -> CaptureArtifact:
    destination = output_dir or config.output_dir
    cv2, capture = open_camera(config)
    detector = build_default_face_detector(config)
    analyzer = build_default_quality_analyzer(config)
    harness = FaceCaptureHarness(
        detector=detector,
        quality_analyzer=analyzer,
        min_face_seconds=config.min_face_seconds,
        face_min_size_px=config.face_min_size_px,
    )
    artifact: CaptureArtifact | None = None
    latest_decision = CaptureDecision(
        state="searching",
        elapsed_seconds=0.0,
        face=None,
        quality=None,
        should_capture=False,
        message="정면 얼굴을 카메라 중앙에 맞춰 주세요.",
    )

    try:
        while artifact is None:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("failed to read camera frame")
            latest_decision = harness.observe(frame)
            faces = [] if latest_decision.face is None else [latest_decision.face]
            draw_overlay(
                cv2,
                frame,
                latest_decision.message,
                faces,
                latest_decision.state == "warning",
            )
            if config.show_preview:
                cv2.imshow("oracle-report", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt("capture cancelled")
            if latest_decision.should_capture:
                artifact = save_capture_artifact(frame, latest_decision, destination)
    finally:
        capture.release()
        if config.show_preview:
            cv2.destroyAllWindows()

    result = artifact
    return result
