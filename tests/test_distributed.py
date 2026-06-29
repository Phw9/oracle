from __future__ import annotations

import pytest

from oracle_report.prompt_templates import render_distributed_prompt_template
from oracle_report.workflow import DistributedTaskScheduler


def test_distributed_prompt_split_metadata() -> None:
    values = {
        "name": "홍길동",
        "gender": "남성",
        "birth_datetime": "1999-10-20T10:25:00",
        "birth_time_text": "오시(午時)",
        "quality_text": "좋음",
    }

    rendered = render_distributed_prompt_template(
        name="personal_face_analysis",
        values=values,
        is_metadata=True,
    )

    assert "face_subtitle" in rendered.body
    assert '"face_blocks":' not in rendered.prefix
    assert "personal_face_analysis_split" in rendered.name


def test_distributed_prompt_split_category() -> None:
    values = {
        "name": "홍길동",
        "gender": "남성",
        "birth_datetime": "1999-10-20T10:25:00",
        "birth_time_text": "오시(午時)",
        "quality_text": "좋음",
    }

    rendered = render_distributed_prompt_template(
        name="personal_face_analysis",
        values=values,
        target_category="눈과 눈썹",
    )

    assert "눈과 눈썹" in rendered.body
    assert "category" in rendered.body
    assert "body" in rendered.body
    assert '"face_blocks":' not in rendered.prefix
    assert '"face_subtitle":' not in rendered.prefix
    assert "[분석 대상 카테고리]" in rendered.body


def test_distributed_prompt_rejects_saju_split() -> None:
    values = {
        "name": "홍길동",
        "gender": "남성",
        "birth_datetime": "1999-10-20T10:25:00",
        "birth_time_text": "오시(午時)",
        "timezone": "KST",
        "saju_text": "사주 테스트 텍스트",
    }

    with pytest.raises(ValueError, match="unsupported distributed prompt template"):
        render_distributed_prompt_template(
            name="saju_reading",
            values=values,
            is_metadata=True,
        )


def test_distributed_task_scheduler_round_robin() -> None:
    slaves = ["http://192.168.0.10:8501", "http://192.168.0.11:8501"]
    scheduler = DistributedTaskScheduler(slaves)

    assert scheduler.select_slave("task1") == "http://192.168.0.10:8501"
    assert scheduler.select_slave("task2") == "http://192.168.0.11:8501"
    assert scheduler.select_slave("task3") == "http://192.168.0.10:8501"
