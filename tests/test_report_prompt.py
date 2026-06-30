from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from oracle_report import prompt_templates
from oracle_report.models import BirthProfile
from oracle_report.report import (
    build_couple_saju_reading_prompt,
    build_saju_reading_prompt,
)


_PROMPT_TEMPLATE_NAMES = (
    "saju_reading",
    "saju_reading_couple",
)


def test_runtime_prompts_define_explicit_cache_prefixes() -> None:
    prompt_path = Path("configs/prompts.json")
    root = json.loads(prompt_path.read_text(encoding="utf-8"))

    for prompt_name in _PROMPT_TEMPLATE_NAMES:
        prompt_config = root[prompt_name]

        assert isinstance(prompt_config, dict)
        assert isinstance(prompt_config["id_slot"], int)
        assert isinstance(prompt_config["prefix"], list)
        assert isinstance(prompt_config["body"], list)
        assert prompt_config["prefix"]
        assert prompt_config["body"]
        assert "${" not in "\n".join(prompt_config["prefix"])
        assert "${" in "\n".join(prompt_config["body"])


def test_saju_reading_prompt_omits_face_and_recommendation_schema() -> None:
    profile = BirthProfile(name="홍길동", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_saju_reading_prompt(
        profile,
        "사주 입력",
    )

    assert "\"saju_blocks\"" in prompt
    assert "\"face_blocks\"" not in prompt
    assert "\"recommendation_title\"" not in prompt
    assert "사주 입력" in prompt
    assert "얼굴 관찰 메모" not in prompt
    assert "추천받고 싶은 얼굴" not in prompt
    assert prompt.name == "saju_reading"
    assert prompt.slot_id == 1
    assert prompt.prefix.strip() != ""
    assert "JSON 객체 1개만 출력하세요" in prompt.prefix
    assert "body는 절대 대충 요약해서 짧게 끝내지 마세요" in prompt.prefix
    assert "최소 6~8개 이상의 긴 문장" in prompt.prefix
    assert "summary와 body는 각각 정확히" not in prompt.prefix
    assert "summary와 body의 문장 수는 서로 같아야" not in prompt.prefix
    assert "자동 줄바꿈 기준" not in prompt.prefix
    assert "5~6줄" not in prompt.prefix
    assert "180~220자" not in prompt.prefix
    assert "줄바꿈 이스케이프" not in prompt.prefix
    assert "줄바꿈은 \\n으로 표현" not in prompt.prefix
    assert "사용자의 이름" in prompt.prefix
    assert "호칭(님은, 님이)" in prompt.prefix
    assert "사주 전문 용어" in prompt.prefix
    assert "사주 근거" in prompt.prefix
    assert "구체적 미래 점술" in prompt.prefix
    assert "실천 꿀팁" in prompt.prefix
    assert "사주 입력" not in prompt.prefix
    assert "사주 입력" in prompt.body
    assert "[블록별 도출해야 할 족집게 타겟 정보" in prompt.prefix
    assert "총평 및 인생 조언" in prompt.prefix


def test_saju_reading_prompt_avoids_input_name_honorifics() -> None:
    profile = BirthProfile(name="홍길동", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_saju_reading_prompt(
        profile,
        "일간은 임수입니다.",
    )

    assert "- 이름: 홍길동" in prompt.body
    assert "임수님" not in prompt.body
    assert "사용자의 이름" in prompt.prefix
    assert "호칭(님은, 님이)" in prompt.prefix
    assert "사주 전문 용어" in prompt.prefix
    assert "임수 일간은" not in prompt.prefix


def test_couple_saju_reading_prompt_uses_pair_saju_only() -> None:
    left = BirthProfile(name="left", birth_datetime=datetime(1995, 3, 15, 14, 30))
    right = BirthProfile(name="right", birth_datetime=datetime(1997, 5, 20, 9, 0))

    prompt = build_couple_saju_reading_prompt(
        left,
        right,
        "연인",
        "LEFT SAJU INPUT",
        "RIGHT SAJU INPUT",
    )

    assert "\"saju_blocks\"" in prompt
    assert "\"pair_blocks\"" not in prompt
    assert "LEFT SAJU INPUT" in prompt
    assert "RIGHT SAJU INPUT" in prompt
    assert "JSON 객체 1개만 출력하세요" in prompt.prefix
    assert "body는 절대 대충 요약해서 짧게 끝내지 마세요" in prompt.prefix
    assert "최소 6~8개 이상의 긴 문장" in prompt.prefix
    assert "summary와 body는 각각 정확히" not in prompt.prefix
    assert "summary와 body의 문장 수는 서로 같아야" not in prompt.prefix
    assert "자동 줄바꿈 기준" not in prompt.prefix
    assert "5~6줄" not in prompt.prefix
    assert "180~220자" not in prompt.prefix
    assert "줄바꿈 이스케이프" not in prompt.prefix
    assert "줄바꿈은 \\n으로 표현" not in prompt.prefix
    assert "left_name과 right_name 뒤에 띄어쓰기 없이 '님'" in prompt.prefix
    assert "사주 전문 용어" in prompt.prefix
    assert "궁합 모드" in prompt.prefix
    assert "구체적 궁합 예측" in prompt.prefix
    assert "실천 꿀팁" in prompt.prefix


def test_report_body_length_guidance_matches_runtime_prompt() -> None:
    profile = BirthProfile(name="tester", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_saju_reading_prompt(profile, "사주 입력")
    debug_prompt = prompt_templates.render_debug_prompt_template(
        "personal_final",
        {
            "name": "tester",
            "gender": "male",
            "birth_datetime": "1995-03-15 미시(未時)",
            "birth_time_text": "미시(未時)",
            "timezone": "Asia/Seoul",
            "saju_text": "사주 입력",
            "face_analysis": "얼굴 관찰",
            "recommendation_text": "추천 정보",
        },
    )

    assert "최소 6~8개 이상의 긴 문장" in prompt.prefix
    assert "body는 정확히 5개의 완성된 문장" in debug_prompt
    assert "summary와 body는 각각 정확히" not in prompt.prefix
    assert "summary와 body는 각각 정확히" not in debug_prompt
    assert "정확히 6개의 완성된 문장" not in prompt.prefix


def test_prompt_template_can_be_overridden_from_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps({"saju_reading": "CUSTOM ${name} ${saju_text}"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORACLE_PROMPTS_PATH", str(prompt_path))
    profile = BirthProfile(name="tester", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_saju_reading_prompt(profile, "SAJU")

    assert prompt == "CUSTOM tester SAJU"
