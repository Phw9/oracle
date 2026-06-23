from __future__ import annotations

from datetime import datetime

from oracle_report.models import BirthProfile
from oracle_report.physiognomy import FaceReadingInput
from oracle_report.report import build_report_prompt
from oracle_report.saju.engine import build_saju_reading


def test_report_prompt_contains_required_sections_and_safety_rules() -> None:
    profile = BirthProfile(name="홍길동", birth_datetime=datetime(1995, 3, 15, 14, 30))
    reading = build_saju_reading(profile.birth_datetime)
    prompt = build_report_prompt(profile, reading, FaceReadingInput(None, None))

    assert "사주 명식" in prompt
    assert "관상 보조 풀이" in prompt
    assert "신원, 나이, 성별, 민족, 건강" in prompt
    assert "# 홍길동 님 종합 운세 리포트" in prompt
