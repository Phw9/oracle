from __future__ import annotations

import json
from oracle_report.prompt_templates import render_distributed_prompt_template
from oracle_report.workflow import DistributedTaskScheduler

def test_distributed_prompt_split_metadata() -> None:
    values = {
        "name": "홍길동",
        "gender": "남성",
        "birth_datetime": "1999-10-20T10:25:00",
        "birth_time_text": "오시(午時)",
        "saju_text": "사주 테스트 텍스트",
        "timezone": "KST",
        "quality_text": "좋음"
    }
    
    rendered = render_distributed_prompt_template(
        name="saju_reading",
        values=values,
        is_metadata=True
    )
    
    assert "essence" in rendered.prefix
    assert '"saju_blocks":' not in rendered.prefix
    assert "saju_reading_split" in rendered.name

    rendered_face = render_distributed_prompt_template(
        name="personal_face_analysis",
        values=values,
        is_metadata=True
    )
    assert "face_subtitle" in rendered_face.prefix
    assert '"face_blocks":' not in rendered_face.prefix


def test_distributed_prompt_split_category() -> None:
    values = {
        "name": "홍길동",
        "gender": "남성",
        "birth_datetime": "1999-10-20T10:25:00",
        "birth_time_text": "오시(午時)",
        "saju_text": "사주 테스트 텍스트",
        "timezone": "KST",
        "quality_text": "좋음"
    }
    
    rendered = render_distributed_prompt_template(
        name="saju_reading",
        values=values,
        target_category="재물운과 적성"
    )
    
    assert "재물운과 적성" in rendered.prefix
    assert "category" in rendered.prefix
    assert "body" in rendered.prefix
    assert '"saju_blocks":' not in rendered.prefix
    assert '"essence":' not in rendered.prefix
    assert "[분석 대상 카테고리]" in rendered.body


def test_distributed_task_scheduler_round_robin() -> None:
    slaves = ["http://192.168.0.10:8501", "http://192.168.0.11:8501"]
    scheduler = DistributedTaskScheduler(slaves)
    
    assert scheduler.select_slave("task1") == "http://192.168.0.10:8501"
    assert scheduler.select_slave("task2") == "http://192.168.0.11:8501"
    assert scheduler.select_slave("task3") == "http://192.168.0.10:8501"
