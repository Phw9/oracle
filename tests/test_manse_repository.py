from __future__ import annotations

from datetime import datetime
from pathlib import Path

from oracle_report.models import BirthProfile
from oracle_report.saju.repository import (
    MANSE_GENDERS,
    MANSE_TIME_BRANCH_LABELS,
    ManseRepository,
    normalize_gender,
    representative_time_from_time_branch,
    time_branch_display_from_index,
    time_branch_index_from_datetime,
)


def test_manse_lookup_calculates_runtime_pillars_without_db() -> None:
    repository = ManseRepository()
    profile = BirthProfile(
        name="홍길동",
        birth_datetime=datetime(1999, 10, 20, 10, 25),
        gender="남성",
    )

    result = repository.lookup(profile)

    assert result.reading.chart.year.label == "기묘"
    assert result.reading.chart.month.label == "갑술"
    assert result.reading.chart.day.label == "을사"
    assert result.reading.chart.hour.label == "신사"
    assert "만세력/사주명식" in result.formatted_text
    assert result.gender in MANSE_GENDERS
    assert result.time_branch_label in MANSE_TIME_BRANCH_LABELS


def test_manse_lookup_ignores_missing_legacy_db_path(tmp_path: Path) -> None:
    repository = ManseRepository(tmp_path / "missing.sqlite")
    profile = BirthProfile(
        name="미래",
        birth_datetime=datetime(2300, 1, 1, 0, 0),
        gender="여성",
    )

    result = repository.lookup(profile)

    assert result.daeun_direction in {"순행", "역행"}
    assert "사주명식" in result.formatted_text


def test_time_branch_boundaries_use_two_hour_shichen() -> None:
    assert time_branch_index_from_datetime(datetime(2024, 3, 10, 23, 0)) == 0
    assert time_branch_index_from_datetime(datetime(2024, 3, 10, 0, 59)) == 0
    assert time_branch_index_from_datetime(datetime(2024, 3, 10, 1, 0)) == 1
    assert time_branch_index_from_datetime(datetime(2024, 3, 10, 21, 0)) == 11
    assert time_branch_index_from_datetime(datetime(2024, 3, 10, 22, 59)) == 11


def test_time_branch_display_and_parsing_use_shichen_names() -> None:
    assert time_branch_display_from_index(6) == "오시(午時)"
    assert representative_time_from_time_branch("오시") == "12:00"
    assert representative_time_from_time_branch("午時") == "12:00"
    assert representative_time_from_time_branch("오시(午時)") == "12:00"


def test_normalizes_supported_gender_aliases() -> None:
    assert normalize_gender("남자") == "남성"
    assert normalize_gender("female") == "여성"
