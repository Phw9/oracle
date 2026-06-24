from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from string import Template
from typing import Any


_DEFAULT_PROMPTS_PATH = Path("configs/prompts.json")
_PROMPTS_PATH_ENV_NAME = "ORACLE_PROMPTS_PATH"


def render_prompt_template(name: str, values: Mapping[str, object]) -> str:
    templates = _load_prompt_templates(_prompt_templates_path())
    template_text = _template_text(templates, name)
    string_values = {key: str(value) for key, value in values.items()}
    result = Template(template_text).substitute(string_values).strip()
    return result


def _prompt_templates_path() -> Path:
    configured_path = os.getenv(_PROMPTS_PATH_ENV_NAME, "")
    result = _DEFAULT_PROMPTS_PATH
    if configured_path.strip() != "":
        result = Path(configured_path)
    return result


def _load_prompt_templates(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as prompt_file:
        root = json.load(prompt_file)
    if not isinstance(root, dict):
        raise ValueError(f"prompt template file must contain a JSON object: {path}")
    result = root
    return result


def _template_text(templates: Mapping[str, Any], name: str) -> str:
    raw_value = templates.get(name)
    result = ""
    if isinstance(raw_value, str):
        result = raw_value
    elif isinstance(raw_value, list) and all(isinstance(item, str) for item in raw_value):
        result = "\n".join(raw_value)
    else:
        raise ValueError(
            f"prompt template '{name}' must be a string or a list of strings.",
        )
    return result
