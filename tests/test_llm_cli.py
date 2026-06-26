from __future__ import annotations

import json
from pathlib import Path

from oracle_report.cli import main


class FakeLlamaClient:
    prompt: str = ""
    image_path: Path | None = None

    def __init__(self, config) -> None:
        del config

    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        FakeLlamaClient.prompt = prompt
        FakeLlamaClient.image_path = image_path
        result = "LLM RESULT ONLY"
        return result


def test_llm_command_runs_saju_reading_prompt_from_config(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            {
                "saju_reading": (
                    "CUSTOM ${name}\n"
                    "${birth_datetime}\n"
                    "${birth_time_text}\n"
                    "${saju_text}"
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manse_db_path = _build_test_manse_db(tmp_path)
    monkeypatch.setenv("ORACLE_PROMPTS_PATH", str(prompt_path))
    monkeypatch.setattr("oracle_report.cli.LlamaCppChatClient", FakeLlamaClient)

    result = main(
        [
            "llm",
            "saju-reading",
            "--name",
            "tester",
            "--birth-date",
            "1995-03-15",
            "--birth-time",
            "모름",
            "--gender",
            "male",
            "--manse-db",
            str(manse_db_path),
        ],
    )

    output = capsys.readouterr().out

    assert result == 0
    assert output == "LLM RESULT ONLY\n"
    assert "CUSTOM tester" in FakeLlamaClient.prompt
    assert "1995-03-15 시간 미상 (오시(午時) 보조 기준)" in FakeLlamaClient.prompt
    assert "오시(午時) 보조 기준" in FakeLlamaClient.prompt
    assert "[만세력/사주명식]" in FakeLlamaClient.prompt
    assert FakeLlamaClient.image_path is None


def test_llm_personal_final_uses_debug_prompt_config(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    debug_prompt_path = tmp_path / "prompts_debug.json"
    prompt_path.write_text(
        json.dumps({"saju_reading": "REGULAR ${name}"}, ensure_ascii=False),
        encoding="utf-8",
    )
    debug_prompt_path.write_text(
        json.dumps(
            {"personal_final": "DEBUG ${name} ${saju_text} ${face_analysis}"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manse_db_path = _build_test_manse_db(tmp_path)
    monkeypatch.setenv("ORACLE_PROMPTS_PATH", str(prompt_path))
    monkeypatch.setenv("ORACLE_DEBUG_PROMPTS_PATH", str(debug_prompt_path))
    monkeypatch.setattr("oracle_report.cli.LlamaCppChatClient", FakeLlamaClient)

    result = main(
        [
            "llm",
            "personal-final",
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
            "--face-analysis",
            "FACE DEBUG INPUT",
            "--recommendation-text",
            "RECOMMEND DEBUG INPUT",
        ],
    )

    output = capsys.readouterr().out

    assert result == 0
    assert output == "LLM RESULT ONLY\n"
    assert "DEBUG tester" in FakeLlamaClient.prompt
    assert "FACE DEBUG INPUT" in FakeLlamaClient.prompt
    assert "REGULAR" not in FakeLlamaClient.prompt
    assert FakeLlamaClient.image_path is None


def _build_test_manse_db(tmp_path: Path) -> Path:
    result = tmp_path / "unused-manse.sqlite"
    return result
