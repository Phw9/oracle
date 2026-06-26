from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from oracle_report.saju.solar_terms_data import solar_term_correction_minutes


MS_PER_DAY = 86_400_000
MS_PER_MINUTE = 60_000
KST_OFFSET_MINUTES = 540
KST_STANDARD_MERIDIAN = 135
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
LICHUN_INDEX = 2
NAIVE_EPOCH = datetime(1970, 1, 1)
UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
JEOL_TO_MONTH = (
    (2, 1),
    (4, 2),
    (6, 3),
    (8, 4),
    (10, 5),
    (12, 6),
    (14, 7),
    (16, 8),
    (18, 9),
    (20, 10),
    (22, 11),
    (0, 12),
)
SOLAR_TERM_NAMES = (
    "소한",
    "대한",
    "입춘",
    "우수",
    "경칩",
    "춘분",
    "청명",
    "곡우",
    "입하",
    "소만",
    "망종",
    "하지",
    "소서",
    "대서",
    "입추",
    "처서",
    "백로",
    "추분",
    "한로",
    "상강",
    "입동",
    "소설",
    "대설",
    "동지",
)
_SOLAR_TERM_CACHE: dict[tuple[int, int], int] = {}


def instant_utc_ms_from_kst(birth_datetime: datetime) -> int:
    wall_datetime = birth_datetime.replace(tzinfo=None)
    wall_ms = _milliseconds_since_epoch(wall_datetime)
    result = wall_ms - KST_OFFSET_MINUTES * MS_PER_MINUTE
    return result


def apparent_ms_from_kst(birth_datetime: datetime) -> int:
    instant_ms = instant_utc_ms_from_kst(birth_datetime)
    result = instant_ms + KST_STANDARD_MERIDIAN * 4 * MS_PER_MINUTE
    return result


def apparent_datetime_from_kst(birth_datetime: datetime) -> datetime:
    apparent_ms = apparent_ms_from_kst(birth_datetime)
    result = NAIVE_EPOCH + timedelta(milliseconds=apparent_ms)
    return result


def utc_datetime_from_ms(ms: int) -> datetime:
    result = UTC_EPOCH + timedelta(milliseconds=ms)
    return result


def solar_term_instant_ms(year: int, index: int) -> int:
    key = (year, index)
    result = _SOLAR_TERM_CACHE.get(key)
    if result is None:
        target = _solar_term_longitude(index)
        month_index = index // 2
        guess_ms = _utc_ms(year, month_index + 1, 15, 0, 0)
        meeus_minute = round(_solve_solar_longitude_instant(target, guess_ms) / MS_PER_MINUTE)
        correction = solar_term_correction_minutes(year, index)
        result = (meeus_minute + correction) * MS_PER_MINUTE
        _SOLAR_TERM_CACHE[key] = result
    return result


def saju_year_for_datetime(birth_datetime: datetime) -> int:
    instant_ms = instant_utc_ms_from_kst(birth_datetime)
    lichun_ms = solar_term_instant_ms(birth_datetime.year, LICHUN_INDEX)
    result = birth_datetime.year
    if instant_ms < lichun_ms:
        result = birth_datetime.year - 1
    return result


def saju_month_number_for_datetime(birth_datetime: datetime) -> int:
    instant_ms = instant_utc_ms_from_kst(birth_datetime)
    result = saju_month_number_for_instant_ms(instant_ms)
    return result


def saju_month_number_for_instant_ms(instant_ms: int) -> int:
    year = utc_datetime_from_ms(instant_ms).year
    best_boundary = -math.inf
    result = 12
    for solar_year in (year - 1, year, year + 1):
        for term_index, month_number in JEOL_TO_MONTH:
            boundary = solar_term_instant_ms(solar_year, term_index)
            if boundary <= instant_ms and boundary > best_boundary:
                best_boundary = boundary
                result = month_number
    return result


def solar_term_boundary_minutes_for_kst_date(day: datetime) -> tuple[int, ...]:
    start_ms = instant_utc_ms_from_kst(day.replace(hour=0, minute=0, second=0, microsecond=0))
    end_ms = start_ms + MS_PER_DAY
    boundaries: list[int] = []
    for year in (day.year - 1, day.year, day.year + 1):
        for index in range(len(SOLAR_TERM_NAMES)):
            term_ms = solar_term_instant_ms(year, index)
            if start_ms < term_ms < end_ms:
                boundaries.append((term_ms - start_ms) // MS_PER_MINUTE)
    result = tuple(sorted(set(int(item) for item in boundaries)))
    return result


def _solar_term_longitude(index: int) -> float:
    result = (285.0 + 15.0 * index) % 360.0
    return result


def _utc_ms(year: int, month: int, day: int, hour: int, minute: int) -> int:
    value = datetime(year, month, day, hour, minute)
    result = _milliseconds_since_epoch(value)
    return result


def _milliseconds_since_epoch(value: datetime) -> int:
    delta = value - NAIVE_EPOCH
    result = (
        delta.days * MS_PER_DAY
        + delta.seconds * 1000
        + delta.microseconds // 1000
    )
    return result


def _julian_day_from_ms(ms: int | float) -> float:
    result = ms / MS_PER_DAY + 2440587.5
    return result


def _normalize_degrees(degrees: float) -> float:
    result = degrees % 360.0
    if result < 0.0:
        result += 360.0
    return result


def _apparent_solar_longitude(ms: int | float) -> float:
    jd = _julian_day_from_ms(ms)
    elements = _solar_elements(jd)
    true_longitude = elements["L0"] + elements["C"]
    omega = 125.04 - 1934.136 * elements["T"]
    apparent = true_longitude - 0.00569 - 0.00478 * math.sin(omega * DEG2RAD)
    result = _normalize_degrees(apparent)
    return result


def _solar_elements(jd: float) -> dict[str, float]:
    t = (jd - 2451545.0) / 36525.0
    l0 = _normalize_degrees(280.46646 + 36000.76983 * t + 0.0003032 * t * t)
    mean_anomaly = 357.52911 + 35999.05029 * t - 0.0001537 * t * t
    mean_anomaly_rad = mean_anomaly * DEG2RAD
    center = (
        (1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(mean_anomaly_rad)
        + (0.019993 - 0.000101 * t) * math.sin(2.0 * mean_anomaly_rad)
        + 0.000289 * math.sin(3.0 * mean_anomaly_rad)
    )
    result = {"L0": l0, "C": center, "T": t}
    return result


def _solve_solar_longitude_instant(target_longitude: float, guess_ms: int) -> float:
    result = float(guess_ms)
    degrees_per_day = 360.0 / 365.2422
    for _ in range(8):
        current = _apparent_solar_longitude(result)
        diff = ((current - target_longitude + 540.0) % 360.0) - 180.0
        if abs(diff) >= 1e-7:
            result -= (diff / degrees_per_day) * MS_PER_DAY
    return result
