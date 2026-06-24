from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from oracle_report.models import BirthProfile
from oracle_report.physiognomy import FaceReadingInput
from oracle_report.report import build_face_analysis_prompt, build_report_prompt
from oracle_report.saju.engine import build_saju_reading


def test_report_prompt_contains_required_sections_and_safety_rules() -> None:
    profile = BirthProfile(name="홍길동", birth_datetime=datetime(1995, 3, 15, 14, 30))
    reading = build_saju_reading(profile.birth_datetime)
    prompt = build_report_prompt(profile, reading, FaceReadingInput(None, None))

    assert "사주 명식" in prompt
    assert "관상 보조 해석" in prompt
    assert "신원, 나이, 성별, 민족, 건강" in prompt
    assert "# 홍길동 종합 운세 리포트" in prompt


def test_prompt_template_can_be_overridden_from_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps({"face_analysis": "CUSTOM ${name} ${quality_text}"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORACLE_PROMPTS_PATH", str(prompt_path))
    profile = BirthProfile(name="tester", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_face_analysis_prompt(profile, FaceReadingInput(None, None))

    assert prompt == "CUSTOM tester - 품질 정보 없음"
