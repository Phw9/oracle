from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from oracle_report.models import BirthProfile
from oracle_report.saju.engine import SajuReading, build_saju_reading, format_saju_reading


@dataclass(frozen=True)
class SajuLookupResult:
    reading: SajuReading
    formatted_text: str
    cache_hit: bool


class SajuRepository:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def lookup(self, profile: BirthProfile) -> SajuLookupResult:
        self._ensure_schema()
        cache_key = _cache_key(profile)
        cached_text = self._read_cached_text(cache_key)
        reading = build_saju_reading(profile.birth_datetime)
        formatted_text = format_saju_reading(reading)
        cache_hit = cached_text is not None
        if not cache_hit:
            self._write_cached_text(cache_key, profile, formatted_text)
        result = SajuLookupResult(
            reading=reading,
            formatted_text=formatted_text,
            cache_hit=cache_hit,
        )
        return result

    def _ensure_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS saju_cache (
                    cache_key TEXT PRIMARY KEY,
                    birth_datetime TEXT NOT NULL,
                    gender TEXT NOT NULL,
                    birth_time_known INTEGER NOT NULL,
                    formatted_text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """,
            )
            connection.commit()

    def _read_cached_text(self, cache_key: str) -> str | None:
        result: str | None = None
        with sqlite3.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT formatted_text FROM saju_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is not None:
                result = str(row[0])
        return result

    def _write_cached_text(
        self,
        cache_key: str,
        profile: BirthProfile,
        formatted_text: str,
    ) -> None:
        with sqlite3.connect(self._db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO saju_cache (
                    cache_key,
                    birth_datetime,
                    gender,
                    birth_time_known,
                    formatted_text
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    profile.birth_datetime.isoformat(sep=" "),
                    profile.gender,
                    int(profile.birth_time_known),
                    formatted_text,
                ),
            )
            connection.commit()


def _cache_key(profile: BirthProfile) -> str:
    result = "|".join(
        (
            profile.birth_datetime.isoformat(),
            profile.gender,
            "time-known" if profile.birth_time_known else "time-unknown",
        ),
    )
    return result
