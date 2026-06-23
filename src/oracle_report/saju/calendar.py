from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


HEAVENLY_STEMS = ("갑", "을", "병", "정", "무", "기", "경", "신", "임", "계")
EARTHLY_BRANCHES = ("자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해")
STEM_ELEMENTS = ("목", "목", "화", "화", "토", "토", "금", "금", "수", "수")
STEM_POLARITIES = ("양", "음", "양", "음", "양", "음", "양", "음", "양", "음")
BRANCH_ELEMENTS = ("수", "토", "목", "목", "토", "화", "화", "토", "금", "금", "토", "수")
BRANCH_ANIMALS = ("쥐", "소", "호랑이", "토끼", "용", "뱀", "말", "양", "원숭이", "닭", "개", "돼지")


@dataclass(frozen=True)
class StemBranch:
    stem_index: int
    branch_index: int

    @property
    def stem(self) -> str:
        result = HEAVENLY_STEMS[self.stem_index]
        return result

    @property
    def branch(self) -> str:
        result = EARTHLY_BRANCHES[self.branch_index]
        return result

    @property
    def element(self) -> str:
        result = STEM_ELEMENTS[self.stem_index]
        return result

    @property
    def polarity(self) -> str:
        result = STEM_POLARITIES[self.stem_index]
        return result

    @property
    def label(self) -> str:
        result = f"{self.stem}{self.branch}"
        return result


@dataclass(frozen=True)
class SajuChart:
    birth_datetime: datetime
    year: StemBranch
    month: StemBranch
    day: StemBranch
    hour: StemBranch


def build_saju_chart(birth_datetime: datetime) -> SajuChart:
    adjusted_year = _solar_term_year(birth_datetime.date())
    year = _year_pillar(adjusted_year)
    month = _month_pillar(birth_datetime.date(), year.stem_index)
    day = _day_pillar(birth_datetime.date())
    hour = _hour_pillar(birth_datetime, day.stem_index)
    result = SajuChart(
        birth_datetime=birth_datetime,
        year=year,
        month=month,
        day=day,
        hour=hour,
    )
    return result


def julian_day_number(day: date) -> int:
    result = day.toordinal() + 1721425
    return result


def _year_pillar(year: int) -> StemBranch:
    cycle_index = (year - 1984) % 60
    result = _sexagenary_from_cycle_index(cycle_index)
    return result


def _month_pillar(day: date, year_stem_index: int) -> StemBranch:
    branch_index = _month_branch_index(day)
    tiger_month_stem = ((year_stem_index % 5) * 2 + 2) % 10
    month_offset = (branch_index - 2) % 12
    stem_index = (tiger_month_stem + month_offset) % 10
    result = StemBranch(stem_index=stem_index, branch_index=branch_index)
    return result


def _day_pillar(day: date) -> StemBranch:
    jdn = julian_day_number(day)
    stem_index = (jdn - 1) % 10
    branch_index = (jdn + 1) % 12
    result = StemBranch(stem_index=stem_index, branch_index=branch_index)
    return result


def _hour_pillar(birth_datetime: datetime, day_stem_index: int) -> StemBranch:
    branch_index = ((birth_datetime.hour + 1) // 2) % 12
    zi_hour_stem = (day_stem_index % 5) * 2
    stem_index = (zi_hour_stem + branch_index) % 10
    result = StemBranch(stem_index=stem_index, branch_index=branch_index)
    return result


def _sexagenary_from_cycle_index(cycle_index: int) -> StemBranch:
    result = StemBranch(stem_index=cycle_index % 10, branch_index=cycle_index % 12)
    return result


def _solar_term_year(day: date) -> int:
    result = day.year
    if (day.month, day.day) < (2, 4):
        result = day.year - 1
    return result


def _month_branch_index(day: date) -> int:
    month_day = (day.month, day.day)
    result = 0
    if (1, 6) <= month_day < (2, 4):
        result = 1
    elif (2, 4) <= month_day < (3, 6):
        result = 2
    elif (3, 6) <= month_day < (4, 5):
        result = 3
    elif (4, 5) <= month_day < (5, 6):
        result = 4
    elif (5, 6) <= month_day < (6, 6):
        result = 5
    elif (6, 6) <= month_day < (7, 7):
        result = 6
    elif (7, 7) <= month_day < (8, 8):
        result = 7
    elif (8, 8) <= month_day < (9, 8):
        result = 8
    elif (9, 8) <= month_day < (10, 8):
        result = 9
    elif (10, 8) <= month_day < (11, 7):
        result = 10
    elif (11, 7) <= month_day < (12, 7):
        result = 11
    return result
