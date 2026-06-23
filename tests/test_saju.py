from __future__ import annotations

from datetime import datetime

from oracle_report.saju.calendar import build_saju_chart, julian_day_number
from oracle_report.saju.engine import build_saju_reading


def test_julian_day_number_known_reference() -> None:
    result = julian_day_number(datetime(1949, 10, 1).date())
    assert result == 2433191


def test_day_pillar_known_reference_from_sexagenary_table() -> None:
    chart = build_saju_chart(datetime(1949, 10, 1, 12, 0))
    assert chart.day.label == "갑자"


def test_1984_after_ipchun_is_jiazi_year() -> None:
    chart = build_saju_chart(datetime(1984, 2, 4, 10, 0))
    assert chart.year.label == "갑자"


def test_hour_pillar_for_jia_day_zi_hour() -> None:
    chart = build_saju_chart(datetime(1949, 10, 1, 23, 30))
    assert chart.hour.label == "갑자"


def test_saju_reading_counts_eight_visible_elements() -> None:
    reading = build_saju_reading(datetime(1995, 3, 15, 14, 30))
    result = sum(reading.element_counts.values())
    assert result == 8
