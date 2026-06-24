from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oracle_report.models import FaceQuality
from oracle_report.prompt_templates import render_prompt_template


@dataclass(frozen=True)
class FaceReadingInput:
    image_path: Path | None
    quality: FaceQuality | None


def build_face_prompt(face_input: FaceReadingInput) -> str:
    quality_text = format_face_quality(face_input.quality)
    result = render_prompt_template(
        "face_prompt",
        {
            "quality_text": quality_text,
        },
    )
    return result


def format_face_quality(quality: FaceQuality | None) -> str:
    result = "- 품질 정보 없음"
    if quality is not None:
        warnings = ", ".join(quality.warnings) if quality.warnings else "경고 없음"
        result = (
            f"- 눈 개수: {quality.eye_count}, "
            f"눈썹 점수: {quality.eyebrow_score:.3f}, "
            f"정면 점수: {quality.frontality_score:.2f}, "
            f"가림 추정 점수: {quality.occlusion_score:.2f}, "
            f"경고: {warnings}"
        )
    return result
