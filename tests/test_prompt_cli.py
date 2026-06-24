from __future__ import annotations

from pathlib import Path

from oracle_report.cli import main
from oracle_report.saju.repository import build_manse_database


def test_prompt_command_prints_face_analysis_prompt(capsys) -> None:
    result = main(
        [
            "prompt",
            "face-analysis",
            "--name",
            "tester",
            "--birth-date",
            "1995-03-15",
            "--birth-time",
            "14:30",
            "--gender",
            "male",
        ],
    )

    output = capsys.readouterr().out

    assert result == 0
    assert "1995-03-15 14:30:00" in output


def test_prompt_command_prints_saju_reading(capsys, tmp_path: Path) -> None:
    manse_db_path = _build_test_manse_db(tmp_path)

    result = main(
        [
            "prompt",
            "saju-reading",
            "--name",
            "tester",
            "--birth-date",
            "1995-03-15",
            "--birth-time",
            "14:30",
            "--gender",
            "male",
            "--manse-db",
            str(manse_db_path),
        ],
    )

    output = capsys.readouterr().out

    assert result == 0
    assert "1995-03-15" in output


def test_prompt_command_prints_personal_final_prompt(
    capsys,
    tmp_path: Path,
) -> None:
    manse_db_path = _build_test_manse_db(tmp_path)

    result = main(
        [
            "prompt",
            "personal-final",
            "--name",
            "tester",
            "--birth-date",
            "1995-03-15",
            "--birth-time",
            "14:30",
            "--gender",
            "male",
            "--target-gender",
            "female",
            "--manse-db",
            str(manse_db_path),
            "--face-db",
            str(tmp_path / "faces.sqlite"),
            "--face-analysis",
            "face analysis fixture",
        ],
    )

    output = capsys.readouterr().out

    assert result == 0
    assert "1995-03-15" in output
    assert "face analysis fixture" in output


def _build_test_manse_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "manse.sqlite"
    build_manse_database(db_path, start_year=1995, end_year=1995)
    result = db_path
    return result
