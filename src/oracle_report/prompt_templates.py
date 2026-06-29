from __future__ import annotations

import json
import os
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from string import Template
from typing import Any


_DEFAULT_PROMPTS_PATH = Path("configs/prompts.json")
_DEFAULT_DEBUG_PROMPTS_PATH = Path("configs/prompts_debug.json")
_PROMPTS_PATH_ENV_NAME = "ORACLE_PROMPTS_PATH"
_DEBUG_PROMPTS_PATH_ENV_NAME = "ORACLE_DEBUG_PROMPTS_PATH"
REPORT_BLOCK_SENTENCE_COUNT = 6
_REPORT_BLOCK_SENTENCE_COUNT_TOKEN = "{{report_block_sentence_count}}"
_DEFAULT_PROMPT_SLOTS = {
    "personal_face_analysis": 0,
    "saju_reading": 1,
    "face_analysis_copule": 2,
    "saju_reading_couple": 3,
    "compatibility_face_analysis": 4,
}


@dataclass(frozen=True)
class RenderedPrompt:
    name: str
    prefix: str
    body: str
    slot_id: int | None

    @property
    def text(self) -> str:
        result = self.body
        if self.prefix.strip() != "":
            result = f"{self.prefix.strip()}\n\n{self.body.strip()}".strip()
        return result

    def __str__(self) -> str:
        result = self.text
        return result

    def __contains__(self, item: str) -> bool:
        result = item in self.text
        return result

    def __eq__(self, other: object) -> bool:
        result = False
        if isinstance(other, str):
            result = self.text == other
        elif isinstance(other, RenderedPrompt):
            result = (
                self.name == other.name
                and self.prefix == other.prefix
                and self.body == other.body
                and self.slot_id == other.slot_id
            )
        return result


@dataclass(frozen=True)
class PromptTemplateInfo:
    name: str
    prefix: str
    body_template: str
    slot_id: int | None


def render_prompt_template(name: str, values: Mapping[str, object]) -> RenderedPrompt:
    templates = _load_prompt_templates(_prompt_templates_path())
    template_parts = _template_parts(templates, name)
    string_values = {key: str(value) for key, value in values.items()}
    prefix = Template(template_parts.prefix).substitute(string_values).strip()
    body = Template(template_parts.body_template).substitute(string_values).strip()
    result = RenderedPrompt(
        name=name,
        prefix=prefix,
        body=body,
        slot_id=template_parts.slot_id,
    )
    return result


def render_debug_prompt_template(name: str, values: Mapping[str, object]) -> str:
    templates = _load_prompt_templates(_debug_prompt_templates_path())
    template_text = _template_text(templates, name)
    string_values = {key: str(value) for key, value in values.items()}
    result = Template(template_text).substitute(string_values).strip()
    return result


def list_prompt_template_info() -> tuple[PromptTemplateInfo, ...]:
    templates = _load_prompt_templates(_prompt_templates_path())
    result = tuple(_template_parts(templates, name) for name in templates)
    return result


def _prompt_templates_path() -> Path:
    result = _configured_prompt_path(_PROMPTS_PATH_ENV_NAME, _DEFAULT_PROMPTS_PATH)
    return result


def _debug_prompt_templates_path() -> Path:
    result = _configured_prompt_path(
        _DEBUG_PROMPTS_PATH_ENV_NAME,
        _DEFAULT_DEBUG_PROMPTS_PATH,
    )
    return result


def _configured_prompt_path(env_name: str, default_path: Path) -> Path:
    configured_path = os.getenv(env_name, "")
    result = _DEFAULT_PROMPTS_PATH
    if configured_path.strip() != "":
        result = Path(configured_path)
    else:
        result = default_path
    return result


def _load_prompt_templates(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as prompt_file:
        root = json.load(prompt_file)
    if not isinstance(root, dict):
        raise ValueError(f"prompt template file must contain a JSON object: {path}")
    result = root
    return result


def _template_text(templates: Mapping[str, Any], name: str) -> str:
    template_parts = _template_parts(templates, name)
    result = template_parts.prefix
    if template_parts.prefix.strip() != "":
        result = f"{template_parts.prefix.strip()}\n\n{template_parts.body_template.strip()}"
    elif template_parts.body_template.strip() != "":
        result = template_parts.body_template
    return result


def _template_parts(templates: Mapping[str, Any], name: str) -> PromptTemplateInfo:
    raw_value = templates.get(name)
    if isinstance(raw_value, dict):
        result = _dict_template_parts(name, raw_value)
    elif isinstance(raw_value, (str, list)):
        result = _legacy_template_parts(name, raw_value)
    else:
        raise ValueError(
            "prompt template "
            f"'{name}' must be a string, a list of strings, or an object.",
        )
    return result


def _dict_template_parts(name: str, raw_value: Mapping[str, Any]) -> PromptTemplateInfo:
    prefix = _template_fragment_text(raw_value.get("prefix", ""))
    body = _template_fragment_text(raw_value.get("body", raw_value.get("input", "")))
    slot_id = _template_slot_id(name, raw_value.get("id_slot", raw_value.get("slot_id")))
    result = PromptTemplateInfo(
        name=name,
        prefix=prefix,
        body_template=body,
        slot_id=slot_id,
    )
    return result


def _legacy_template_parts(name: str, raw_value: object) -> PromptTemplateInfo:
    template_text = _template_fragment_text(raw_value)
    lines = template_text.splitlines()
    first_dynamic_index = _first_dynamic_line_index(lines)
    prefix_lines = lines
    body_lines: list[str] = []
    if first_dynamic_index is not None:
        prefix_lines = lines[:first_dynamic_index]
        body_lines = lines[first_dynamic_index:]
    result = PromptTemplateInfo(
        name=name,
        prefix="\n".join(prefix_lines).strip(),
        body_template="\n".join(body_lines).strip(),
        slot_id=_template_slot_id(name, None),
    )
    return result


def _template_fragment_text(raw_value: object) -> str:
    result = ""
    if isinstance(raw_value, str):
        result = raw_value
    elif isinstance(raw_value, list) and all(isinstance(item, str) for item in raw_value):
        result = "\n".join(raw_value)
    else:
        raise ValueError("prompt template fragment must be a string or list of strings.")
    result = _apply_template_constants(result)
    return result


def _apply_template_constants(template_text: str) -> str:
    result = template_text.replace(
        _REPORT_BLOCK_SENTENCE_COUNT_TOKEN,
        str(REPORT_BLOCK_SENTENCE_COUNT),
    )
    return result


def _first_dynamic_line_index(lines: list[str]) -> int | None:
    result = None
    for index, line in enumerate(lines):
        if "${" in line:
            result = index
            break
    return result


def _template_slot_id(name: str, configured_value: object) -> int | None:
    result = _DEFAULT_PROMPT_SLOTS.get(name)
    if configured_value is not None:
        result = int(configured_value)
    return result
