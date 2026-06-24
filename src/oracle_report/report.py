from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oracle_report.config import LlmConfig
from oracle_report.llm import LlamaCppChatClient
from oracle_report.models import BirthProfile, ReportArtifact
from oracle_report.physiognomy import (
    FaceReadingInput,
    build_face_prompt,
    format_face_quality,
)
from oracle_report.prompt_templates import render_prompt_template
from oracle_report.saju.engine import SajuReading, build_saju_reading, format_saju_reading


@dataclass(frozen=True)
class ReportRequest:
    birth_profile: BirthProfile
    face_input: FaceReadingInput
    output_path: Path | None


def generate_report(request: ReportRequest, llm_config: LlmConfig) -> ReportArtifact:
    saju_reading = build_saju_reading(request.birth_profile.birth_datetime)
    prompt = build_report_prompt(
        birth_profile=request.birth_profile,
        saju_reading=saju_reading,
        face_input=request.face_input,
    )
    markdown = LlamaCppChatClient(llm_config).generate(
        prompt=prompt,
        image_path=request.face_input.image_path,
    )
    output_path = request.output_path
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    result = ReportArtifact(markdown=markdown, output_path=output_path)
    return result


def build_report_prompt(
    birth_profile: BirthProfile,
    saju_reading: SajuReading,
    face_input: FaceReadingInput,
) -> str:
    saju_text = format_saju_reading(saju_reading)
    face_prompt = build_face_prompt(face_input)
    result = render_prompt_template(
        "report",
        {
            "name": birth_profile.name,
            "birth_datetime": birth_profile.birth_datetime.isoformat(sep=" "),
            "timezone": birth_profile.timezone,
            "saju_text": saju_text,
            "face_prompt": face_prompt,
        },
    )
    return result


def build_face_analysis_prompt(
    birth_profile: BirthProfile,
    face_input: FaceReadingInput,
) -> str:
    quality_text = format_face_quality(face_input.quality)
    result = render_prompt_template(
        "face_analysis",
        {
            "name": birth_profile.name,
            "gender": _gender_text(birth_profile),
            "birth_datetime": birth_profile.birth_datetime.isoformat(sep=" "),
            "birth_time_text": _birth_time_text(birth_profile),
            "quality_text": quality_text,
        },
    )
    return result


def build_personal_final_prompt(
    birth_profile: BirthProfile,
    saju_text: str,
    face_analysis: str,
    recommendation_text: str,
) -> str:
    result = render_prompt_template(
        "personal_final",
        {
            "name": birth_profile.name,
            "gender": _gender_text(birth_profile),
            "birth_datetime": birth_profile.birth_datetime.isoformat(sep=" "),
            "birth_time_text": _birth_time_text(birth_profile),
            "timezone": birth_profile.timezone,
            "saju_text": saju_text,
            "face_analysis": face_analysis,
            "recommendation_text": recommendation_text,
        },
    )
    return result


def build_compatibility_final_prompt(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_saju_text: str,
    right_saju_text: str,
    face_analysis: str,
) -> str:
    result = render_prompt_template(
        "compatibility_final",
        {
            "left_name": left_profile.name,
            "left_gender": _gender_text(left_profile),
            "left_birth_datetime": left_profile.birth_datetime.isoformat(sep=" "),
            "left_birth_time_text": _birth_time_text(left_profile),
            "right_name": right_profile.name,
            "right_gender": _gender_text(right_profile),
            "right_birth_datetime": right_profile.birth_datetime.isoformat(sep=" "),
            "right_birth_time_text": _birth_time_text(right_profile),
            "mode": mode,
            "left_saju_text": left_saju_text,
            "right_saju_text": right_saju_text,
            "face_analysis": face_analysis,
        },
    )
    return result


def _gender_text(profile: BirthProfile) -> str:
    result = profile.gender
    if result == "":
        result = "미입력"
    return result


def _birth_time_text(profile: BirthProfile) -> str:
    result = "입력됨"
    if not profile.birth_time_known:
        result = "미입력: 정오 기준으로 사주를 보조 계산함"
    return result
