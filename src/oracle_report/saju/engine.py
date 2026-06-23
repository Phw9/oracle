from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from oracle_report.saju.calendar import (
    BRANCH_ANIMALS,
    BRANCH_ELEMENTS,
    EARTHLY_BRANCHES,
    HEAVENLY_STEMS,
    STEM_ELEMENTS,
    SajuChart,
    build_saju_chart,
)
from oracle_report.saju.rules import BALANCE_RULES, DAY_MASTER_RULES, DOMINANT_RULES


ELEMENTS = ("목", "화", "토", "금", "수")


@dataclass(frozen=True)
class SajuReading:
    chart: SajuChart
    element_counts: dict[str, int]
    day_master: str
    summary_lines: tuple[str, ...]
    interpretation: str


def build_saju_reading(birth_datetime: datetime) -> SajuReading:
    chart = build_saju_chart(birth_datetime)
    element_counts = _count_elements(chart)
    day_master = chart.day.stem
    strongest = _strongest_element(element_counts)
    weakest = _weakest_element(element_counts)
    summary_lines = (
        f"일간은 {day_master}{STEM_ELEMENTS[chart.day.stem_index]}({chart.day.polarity})입니다.",
        f"가장 강한 오행은 {strongest}, 보완하면 좋은 오행은 {weakest}입니다.",
        f"연지는 {BRANCH_ANIMALS[chart.year.branch_index]}의 흐름으로 사회적 첫인상을 참고합니다.",
    )
    interpretation = _build_interpretation(
        chart=chart,
        element_counts=element_counts,
        strongest=strongest,
        weakest=weakest,
    )
    result = SajuReading(
        chart=chart,
        element_counts=element_counts,
        day_master=day_master,
        summary_lines=summary_lines,
        interpretation=interpretation,
    )
    return result


def format_saju_reading(reading: SajuReading) -> str:
    chart = reading.chart
    counts = ", ".join(
        f"{element} {reading.element_counts[element]}" for element in ELEMENTS
    )
    result = "\n".join(
        (
            "[사주 명식]",
            f"- 년주: {chart.year.label}",
            f"- 월주: {chart.month.label}",
            f"- 일주: {chart.day.label}",
            f"- 시주: {chart.hour.label}",
            "",
            "[오행 분포]",
            f"- {counts}",
            "",
            "[룰 기반 해석]",
            reading.interpretation,
        ),
    )
    return result


def _count_elements(chart: SajuChart) -> dict[str, int]:
    counts = {element: 0 for element in ELEMENTS}
    pillars = (chart.year, chart.month, chart.day, chart.hour)
    for pillar in pillars:
        counts[STEM_ELEMENTS[pillar.stem_index]] += 1
        counts[BRANCH_ELEMENTS[pillar.branch_index]] += 1
    result = counts
    return result


def _strongest_element(counts: dict[str, int]) -> str:
    result = max(ELEMENTS, key=lambda element: (counts[element], -ELEMENTS.index(element)))
    return result


def _weakest_element(counts: dict[str, int]) -> str:
    result = min(ELEMENTS, key=lambda element: (counts[element], ELEMENTS.index(element)))
    return result


def _build_interpretation(
    chart: SajuChart,
    element_counts: dict[str, int],
    strongest: str,
    weakest: str,
) -> str:
    day_master_rule = DAY_MASTER_RULES[chart.day.stem]
    dominant_rule = DOMINANT_RULES[strongest]
    balance_rule = BALANCE_RULES[weakest]
    month_branch = EARTHLY_BRANCHES[chart.month.branch_index]
    month_stem = HEAVENLY_STEMS[chart.month.stem_index]
    count_text = ", ".join(f"{key}:{value}" for key, value in element_counts.items())
    result = "\n".join(
        (
            day_master_rule,
            f"월주는 {month_stem}{month_branch}라서 계절감과 사회적 리듬을 함께 봅니다.",
            f"오행 카운트는 {count_text}입니다.",
            dominant_rule,
            balance_rule,
        ),
    )
    return result
