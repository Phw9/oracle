from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, TypeVar

from oracle_report import prompt_templates
from oracle_report.config import CaptureConfig, LlmConfig
from oracle_report.llm import LlamaCppChatClient
from oracle_report.models import (
    BirthProfile,
    CaptureArtifact,
    FaceBox,
    SequentialPairCaptureArtifact,
)
from oracle_report.recommender import (
    FaceRecommendation,
    recommend_faces,
)
from oracle_report.report import (
    build_couple_saju_reading_prompt,
    build_saju_reading_prompt,
)
from oracle_report.report_html import (
    render_compatibility_report_html,
    render_personal_report_html,
)
from oracle_report.saju.repository import (
    ManseLookupResult,
    ManseRepository,
    UNKNOWN_BIRTH_TIME_REPRESENTATIVE,
    representative_time_from_time_branch,
)
from oracle_report.vision.runtime import run_capture


COMPATIBILITY_MODES = ("연인", "친구", "직장동료")
_UNKNOWN_BIRTH_TIME_VALUES = frozenset(("", "모름", "미상", "unknown", "none"))
_SAJU_DAY_MASTER_HONORIFICS = (
    "갑목님",
    "을목님",
    "병화님",
    "정화님",
    "무토님",
    "기토님",
    "경금님",
    "신금님",
    "임수님",
    "계수님",
)
_T = TypeVar("_T")


class TextGenerator(Protocol):
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        ...


@dataclass(frozen=True)
class PersonalWorkflowInput:
    name: str
    birth_date: str
    birth_time: str
    gender: str
    target_gender: str
    skip_face: bool = False


@dataclass(frozen=True)
class CompatibilityWorkflowInput:
    left_name: str
    left_birth_date: str
    left_birth_time: str
    left_gender: str
    right_name: str
    right_birth_date: str
    right_birth_time: str
    right_gender: str
    mode: str


@dataclass(frozen=True)
class PersonalWorkflowResult:
    markdown: str
    report_html: str
    report_fragment_html: str
    output_path: Path
    capture_path: Path | None
    recommendations: tuple[FaceRecommendation, ...]
    face_analysis: str
    manse_status: str
    timing_log_path: Path | None = None


@dataclass(frozen=True)
class CompatibilityWorkflowResult:
    markdown: str
    report_html: str
    report_fragment_html: str
    output_path: Path
    left_capture_path: Path
    right_capture_path: Path
    face_analysis: str
    left_manse_status: str
    right_manse_status: str
    timing_log_path: Path | None = None


@dataclass(frozen=True)
class _GeneratedText:
    text: str
    error: str


@dataclass(frozen=True)
class _WorkflowTimingEntry:
    label: str
    elapsed_seconds: float
    started_at: datetime
    finished_at: datetime


@dataclass(frozen=True)
class _TimedCallResult(Generic[_T]):
    value: _T
    timing: _WorkflowTimingEntry


@dataclass
class _WorkflowTimingRecorder:
    workflow_name: str
    started_at: datetime = field(default_factory=datetime.now)
    started_counter: float = field(default_factory=time.perf_counter)
    entries: list[_WorkflowTimingEntry] = field(default_factory=list)

    def run(
        self,
        label: str,
        function: Callable[..., _T],
        *args: object,
        **kwargs: object,
    ) -> _T:
        timed_result = _timed_call(label, function, *args, **kwargs)
        self.add(timed_result.timing)
        result = timed_result.value
        return result

    def add(self, timing: _WorkflowTimingEntry) -> None:
        self.entries.append(timing)
        print(_format_timing_line(timing))

    def finish_total(self) -> None:
        finished_at = datetime.now()
        timing = _WorkflowTimingEntry(
            label=self.workflow_name,
            elapsed_seconds=time.perf_counter() - self.started_counter,
            started_at=self.started_at,
            finished_at=finished_at,
        )
        self.add(timing)

    def write_log(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _format_timing_log(self.workflow_name, self.entries),
            encoding="utf-8",
        )
        print(f"[timing] log saved: {path}")
        result = path
        return result


def run_personal_workflow(
    workflow_input: PersonalWorkflowInput,
    capture_config: CaptureConfig,
    report_llm_config: LlmConfig | None = None,
    manse_db_path: Path | None = None,
    recommendation_db_path: Path | None = None,
    report_client: TextGenerator | None = None,
    capture_runner=run_capture,
) -> PersonalWorkflowResult:
    del manse_db_path
    if report_llm_config is None:
        raise ValueError("report_llm_config is required.")
    if recommendation_db_path is None:
        recommendation_db_path = Path("data/face_recommendations.sqlite")
    profile = _build_birth_profile(
        workflow_input.name,
        workflow_input.birth_date,
        workflow_input.birth_time,
        workflow_input.gender,
    )
    output_dir = _new_session_dir(capture_config.output_dir, "personal")
    timing_recorder = _WorkflowTimingRecorder("personal_workflow")
    repository = ManseRepository()
    active_report_client = report_client or LlamaCppChatClient(report_llm_config)

    capture_artifact = None
    if not workflow_input.skip_face:
        with ThreadPoolExecutor(max_workers=2) as executor:
            manse_future = executor.submit(
                _timed_call,
                "manse_lookup",
                repository.lookup,
                profile,
            )
            capture_future = executor.submit(
                _timed_call,
                "capture",
                capture_runner,
                capture_config,
                output_dir,
            )
            capture_timed = capture_future.result()
            timing_recorder.add(capture_timed.timing)
            capture_artifact = capture_timed.value
            manse_timed = manse_future.result()
            timing_recorder.add(manse_timed.timing)
            manse_lookup = manse_timed.value
    else:
        manse_timed = _timed_call(
            "manse_lookup",
            repository.lookup,
            profile,
        )
        timing_recorder.add(manse_timed.timing)
        manse_lookup = manse_timed.value

    if not workflow_input.skip_face and capture_artifact is not None:
        face_analysis = timing_recorder.run(
            "face_analysis",
            _build_single_face_analysis,
            profile,
            capture_artifact,
        )
        face_analysis_text = face_analysis.text
    else:
        face_analysis_text = ""

    saju_analysis = timing_recorder.run(
        "saju_analysis",
        _build_saju_analysis,
        active_report_client,
        profile,
        manse_lookup,
    )
    saju_analysis_text = saju_analysis.text
    recommendations: tuple[FaceRecommendation, ...] = ()
    if not workflow_input.skip_face:
        recommendations = timing_recorder.run(
            "recommend_faces",
            recommend_faces,
            recommendation_db_path,
            workflow_input.target_gender,
            manse_lookup.reading,
        )
    markdown = timing_recorder.run(
        "assemble_report",
        _build_personal_report_json,
        manse_lookup,
        face_analysis_text,
        saju_analysis_text,
        recommendations,
        workflow_input.skip_face,
    )
    report_html = timing_recorder.run(
        "render_report_html",
        render_personal_report_html,
        profile,
        manse_lookup,
        face_analysis_text,
        recommendations,
        markdown,
        True,
        workflow_input.skip_face,
    )
    report_fragment_html = timing_recorder.run(
        "render_report_fragment_html",
        render_personal_report_html,
        profile,
        manse_lookup,
        face_analysis_text,
        recommendations,
        markdown,
        False,
        workflow_input.skip_face,
    )
    output_path = output_dir / "personal_report.html"
    timing_recorder.run(
        "save_report",
        output_path.write_text,
        report_html,
        encoding="utf-8",
    )
    (output_dir / "personal_report.md").write_text(markdown, encoding="utf-8")
    timing_recorder.finish_total()
    timing_log_path = timing_recorder.write_log(output_dir / "timings.log")
    result = PersonalWorkflowResult(
        markdown=markdown,
        report_html=report_html,
        report_fragment_html=report_fragment_html,
        output_path=output_path,
        capture_path=capture_artifact.image_path if capture_artifact is not None else None,
        recommendations=recommendations,
        face_analysis=face_analysis_text,
        manse_status="조회 완료",
        timing_log_path=timing_log_path,
    )
    return result


def run_compatibility_workflow(
    workflow_input: CompatibilityWorkflowInput,
    capture_config: CaptureConfig,
    report_llm_config: LlmConfig | None = None,
    manse_db_path: Path | None = None,
    report_client: TextGenerator | None = None,
    capture_runner=run_capture,
    inter_capture_delay_seconds: float = 3.0,
) -> CompatibilityWorkflowResult:
    del manse_db_path
    if report_llm_config is None:
        raise ValueError("report_llm_config is required.")
    mode = _validate_mode(workflow_input.mode)
    left_profile = _build_birth_profile(
        workflow_input.left_name,
        workflow_input.left_birth_date,
        workflow_input.left_birth_time,
        workflow_input.left_gender,
    )
    right_profile = _build_birth_profile(
        workflow_input.right_name,
        workflow_input.right_birth_date,
        workflow_input.right_birth_time,
        workflow_input.right_gender,
    )
    output_dir = _new_session_dir(capture_config.output_dir, "compatibility")
    timing_recorder = _WorkflowTimingRecorder("compatibility_workflow")
    repository = ManseRepository()
    active_report_client = report_client or LlamaCppChatClient(report_llm_config)

    with ThreadPoolExecutor(max_workers=2) as executor:
        manse_future = executor.submit(
            _timed_call,
            "manse_lookup_pair",
            _lookup_pair_manse,
            repository,
            left_profile,
            right_profile,
        )
        capture_future = executor.submit(
            _timed_call,
            "capture_pair",
            _run_sequential_pair_capture,
            capture_runner,
            capture_config,
            output_dir,
            inter_capture_delay_seconds,
        )
        capture_timed = capture_future.result()
        timing_recorder.add(capture_timed.timing)
        capture_artifact = capture_timed.value
        manse_timed = manse_future.result()
        timing_recorder.add(manse_timed.timing)
        left_manse, right_manse = manse_timed.value

    face_analysis = timing_recorder.run(
        "face_analysis_pair",
        _build_pair_face_analysis,
        left_profile,
        right_profile,
        capture_artifact,
        mode,
    )
    saju_analysis = timing_recorder.run(
        "saju_analysis_pair",
        _build_compatibility_saju_analysis,
        active_report_client,
        left_profile,
        right_profile,
        mode,
        left_manse,
        right_manse,
    )
    markdown = timing_recorder.run(
        "final_report",
        _build_compatibility_report_json,
        face_analysis.text,
        saju_analysis.text,
    )
    report_html = timing_recorder.run(
        "render_report_html",
        render_compatibility_report_html,
        left_profile,
        right_profile,
        mode,
        left_manse,
        right_manse,
        face_analysis.text,
        markdown,
    )
    report_fragment_html = timing_recorder.run(
        "render_report_fragment_html",
        render_compatibility_report_html,
        left_profile,
        right_profile,
        mode,
        left_manse,
        right_manse,
        face_analysis.text,
        markdown,
        False,
    )
    output_path = output_dir / "compatibility_report.html"
    timing_recorder.run(
        "save_report",
        output_path.write_text,
        report_html,
        encoding="utf-8",
    )
    timing_recorder.finish_total()
    timing_log_path = timing_recorder.write_log(output_dir / "timings.log")
    result = CompatibilityWorkflowResult(
        markdown=markdown,
        report_html=report_html,
        report_fragment_html=report_fragment_html,
        output_path=output_path,
        left_capture_path=capture_artifact.left.image_path,
        right_capture_path=capture_artifact.right.image_path,
        face_analysis=face_analysis.text,
        left_manse_status="조회 완료",
        right_manse_status="조회 완료",
        timing_log_path=timing_log_path,
    )
    return result


def _timed_call(
    label: str,
    function: Callable[..., _T],
    *args: object,
    **kwargs: object,
) -> _TimedCallResult[_T]:
    started_at = datetime.now()
    started_counter = time.perf_counter()
    value = function(*args, **kwargs)
    finished_at = datetime.now()
    timing = _WorkflowTimingEntry(
        label=label,
        elapsed_seconds=time.perf_counter() - started_counter,
        started_at=started_at,
        finished_at=finished_at,
    )
    result = _TimedCallResult(value=value, timing=timing)
    return result


def _format_timing_line(timing: _WorkflowTimingEntry) -> str:
    result = f"[timing] {timing.label}: {timing.elapsed_seconds:.3f}s"
    return result


def _format_timing_log(
    workflow_name: str,
    entries: list[_WorkflowTimingEntry],
) -> str:
    lines = [
        "# Oracle workflow timing log",
        f"workflow={workflow_name}",
        "",
    ]
    for entry in entries:
        lines.append(
            "\t".join(
                (
                    entry.started_at.isoformat(timespec="milliseconds"),
                    entry.finished_at.isoformat(timespec="milliseconds"),
                    entry.label,
                    f"{entry.elapsed_seconds:.3f}s",
                ),
            ),
        )
    result = "\n".join(lines) + "\n"
    return result


def _build_single_face_analysis(
    profile: BirthProfile,
    artifact: CaptureArtifact,
) -> _GeneratedText:
    result = _build_single_rule_based_face_analysis(profile, artifact)
    return result


def _build_single_rule_based_face_analysis(
    profile: BirthProfile,
    artifact: CaptureArtifact,
) -> _GeneratedText:
    from oracle_report.vision.physiognomy_text_variations import build_personal_face_payload

    matches = _quality_rule_matches(artifact.quality)
    text = ""
    error = ""
    if matches:
        payload = build_personal_face_payload(
            matches,
            _single_face_seed(profile, artifact, matches),
        )
        text = json.dumps(payload, ensure_ascii=False)
    else:
        text = artifact.quality.face_payload_json.strip()
    if text == "":
        text = artifact.face_analysis.strip()
        error = "rule-based face payload is unavailable; using capture face analysis memo"
    result = _GeneratedText(text=text, error=error)
    return result


def _build_pair_face_analysis(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    artifact: SequentialPairCaptureArtifact,
    mode: str,
) -> _GeneratedText:
    result = _build_pair_rule_based_face_analysis(
        left_profile,
        right_profile,
        artifact,
        mode,
    )
    return result

def _build_pair_rule_based_face_analysis(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    artifact: SequentialPairCaptureArtifact,
    mode: str,
) -> _GeneratedText:
    from oracle_report.vision.physiognomy_text_variations import build_pair_face_payload

    left_matches = _quality_rule_matches(artifact.left.quality)
    right_matches = _quality_rule_matches(artifact.right.quality)
    payload = build_pair_face_payload(
        left_matches,
        right_matches,
        left_profile.name,
        right_profile.name,
        _pair_face_seed(
            left_profile,
            right_profile,
            artifact,
            left_matches,
            right_matches,
        ),
        mode=mode,
    )
    text = json.dumps(payload, ensure_ascii=False)
    result = _GeneratedText(text=text, error="")
    return result


def _quality_rule_matches(quality) -> tuple[Any, ...]:
    from oracle_report.vision.physiognomy_rule_repository import PhysiognomyRuleMatch

    raw_text = getattr(quality, "landmark_matches_json", "").strip()
    rows: list[Any] = []
    if raw_text != "":
        try:
            loaded = json.loads(raw_text)
        except json.JSONDecodeError:
            loaded = []
        if isinstance(loaded, list):
            rows = loaded
    result = tuple(
        PhysiognomyRuleMatch(
            rule_id=str(row.get("rule_id", "")),
            metric=str(row.get("metric", "")),
            title=str(row.get("title", "")),
            basis=str(row.get("basis", "")),
            tag=str(row.get("tag", "")),
            observation=str(row.get("observation", "")),
            interpretation=str(row.get("interpretation", "")),
            value=float(row.get("value", 0.0)),
        )
        for row in rows
        if isinstance(row, dict)
    )
    return result


def _pair_face_seed(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    artifact: SequentialPairCaptureArtifact,
    left_matches: tuple[Any, ...],
    right_matches: tuple[Any, ...],
) -> str:
    left_tags = ",".join(getattr(match, "tag", "") for match in left_matches[:6])
    right_tags = ",".join(getattr(match, "tag", "") for match in right_matches[:6])
    result = (
        f"{left_profile.name}:{right_profile.name}:"
        f"{artifact.left.captured_at.isoformat()}:{artifact.right.captured_at.isoformat()}:"
        f"{left_tags}:{right_tags}"
    )
    return result


def _single_face_seed(
    profile: BirthProfile,
    artifact: CaptureArtifact,
    matches: tuple[Any, ...],
) -> str:
    tags = ",".join(getattr(match, "tag", "") for match in matches[:8])
    result = f"{profile.name}:{artifact.captured_at.isoformat()}:{tags}"
    return result


def _build_saju_analysis(
    client: TextGenerator,
    profile: BirthProfile,
    manse_lookup: ManseLookupResult,
) -> _GeneratedText:
    prompt = build_saju_reading_prompt(profile, manse_lookup.formatted_text)
    result = _safe_generate(
        client,
        prompt,
        None,
        "사주정보를 생성하지 못했습니다.",
        debug_label="saju_analysis",
    )
    result = _replace_day_master_honorifics(result, profile.name, "saju_analysis")
    return result


def _build_compatibility_saju_analysis(
    client: TextGenerator,
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
) -> _GeneratedText:
    prompt = build_couple_saju_reading_prompt(
        left_profile,
        right_profile,
        mode,
        left_manse.formatted_text,
        right_manse.formatted_text,
    )
    result = _safe_generate(
        client,
        prompt,
        None,
        "궁합 사주정보를 생성하지 못했습니다.",
        debug_label="saju_analysis_couple",
    )
    return result



def _build_personal_report_json(
    manse_lookup: ManseLookupResult,
    face_analysis: str,
    saju_analysis: str,
    recommendations: tuple[FaceRecommendation, ...],
    skip_face: bool = False,
) -> str:
    face_payload, face_error = ({}, "")
    saju_payload, saju_error = _load_json_payload_or_error(
        saju_analysis,
        label="saju_analysis",
    )
    if not skip_face:
        face_payload, face_error = _load_json_payload_or_error(
            face_analysis,
            label="face_analysis",
        )
    if saju_error:
        print(
            "[UI FALLBACK:saju_analysis] invalid LLM output; "
            f"renderer will fill missing saju fields. reason={saju_error}",
        )
    if face_error:
        print(
            "[UI FALLBACK:face_analysis] invalid LLM output; "
            f"renderer will fill missing face fields. reason={face_error}",
        )
    payload = _merge_personal_payloads(
        manse_lookup,
        face_payload,
        saju_payload,
        recommendations,
        skip_face,
    )
    payload = _normalize_payload_text(payload)
    result = json.dumps(payload, ensure_ascii=False)
    return result


def _build_compatibility_report_json(
    face_analysis: str,
    saju_analysis: str,
) -> str:
    face_payload, face_error = _load_json_payload_or_error(
        face_analysis,
        label="pair_face_analysis",
    )
    saju_payload, saju_error = _load_json_payload_or_error(
        saju_analysis,
        label="saju_analysis_couple",
    )
    if face_error:
        print(
            "[UI FALLBACK:pair_face_analysis] invalid face output; "
            f"renderer will fill missing pair fields. reason={face_error}",
        )
    if saju_error:
        print(
            "[UI FALLBACK:saju_analysis_couple] invalid LLM output; "
            f"renderer will fill missing saju fields. reason={saju_error}",
        )
    payload = _merge_compatibility_payloads(face_payload, saju_payload)
    payload = _normalize_payload_text(payload)
    result = json.dumps(payload, ensure_ascii=False)
    return result


def _merge_compatibility_payloads(
    face_payload: dict[str, Any],
    saju_payload: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in (
        "essence",
        "saju_subtitle",
        "saju_blocks",
        "synthesis_title",
        "synthesis_body",
        "action_title",
        "action_body",
        "tags",
        "disclaimer",
    ):
        value = saju_payload.get(key)
        if value:
            payload[key] = value
    for key in ("pair_subtitle", "pair_blocks"):
        value = face_payload.get(key)
        if value:
            payload[key] = value
    for key in ("essence", "synthesis_title", "synthesis_body", "action_title", "action_body"):
        if key not in payload:
            value = face_payload.get(key)
            if value:
                payload[key] = value
    payload["convergence"] = _combined_pair_convergence(face_payload, saju_payload)
    result = payload
    return result


def _combined_pair_convergence(
    face_payload: dict[str, Any],
    saju_payload: dict[str, Any],
) -> list[dict[str, str]]:
    existing = face_payload.get("convergence") or saju_payload.get("convergence")
    result = []
    if isinstance(existing, list) and existing:
        result = existing
    else:
        for index in range(3):
            result.append(
                {
                    "face": _block_summary(
                        face_payload,
                        "pair_blocks",
                        index,
                        "두 사람의 얼굴 관찰에서 보이는 관계 분위기",
                    ),
                    "saju": _block_summary(
                        saju_payload,
                        "saju_blocks",
                        index,
                        "두 사람의 사주 흐름에서 보이는 상호 보완점",
                    ),
                },
            )
    return result


def _merge_personal_payloads(
    manse_lookup: ManseLookupResult,
    face_payload: dict[str, Any],
    saju_payload: dict[str, Any],
    recommendations: tuple[FaceRecommendation, ...],
    skip_face: bool,
) -> dict[str, Any]:
    reading = manse_lookup.reading
    strongest = _dominant_element(reading.element_counts, strongest=True)
    weakest = _dominant_element(reading.element_counts, strongest=False)
    payload: dict[str, Any] = {}
    for key in (
        "essence",
        "element_note",
        "saju_subtitle",
        "saju_blocks",
        "tags",
        "disclaimer",
    ):
        value = saju_payload.get(key)
        if value:
            payload[key] = value
    if not skip_face:
        for key in ("face_subtitle", "face_blocks"):
            value = face_payload.get(key)
            if value:
                payload[key] = value
        payload["convergence"] = _combined_convergence(face_payload, saju_payload)
        payload["recommendation_title"] = f"{weakest} 기운을 보완해 줄 얼굴"
        payload["recommendation_lead"] = _recommendation_lead(recommendations)
    payload["synthesis_title"] = _synthesis_title(skip_face)
    payload["synthesis_body"] = _synthesis_body(
        face_payload,
        saju_payload,
        strongest,
        weakest,
        skip_face,
    )
    payload["synthesis_summary"] = (
        "결론은 단정이 아니라 참고입니다. 강점은 살리고 부족한 리듬은 생활에서 "
        "보완하세요."
    )
    result = payload
    return result


def _combined_convergence(
    face_payload: dict[str, Any],
    saju_payload: dict[str, Any],
) -> list[dict[str, str]]:
    result = []
    for index in range(3):
        result.append(
            {
                "face": _block_summary(
                    face_payload,
                    "face_blocks",
                    index,
                    "얼굴 관찰에서 보이는 표현 리듬",
                ),
                "saju": _block_summary(
                    saju_payload,
                    "saju_blocks",
                    index,
                    "사주 데이터에서 보이는 생활 리듬",
                ),
            },
        )
    return result


def _block_summary(
    payload: dict[str, Any],
    key: str,
    index: int,
    default: str,
) -> str:
    blocks = payload.get(key)
    result = default
    if isinstance(blocks, list) and index < len(blocks):
        block = blocks[index]
        if isinstance(block, dict):
            result = _text_from_payload(
                block,
                "summary",
                _text_from_payload(block, "title", default),
            )
    return result


def _text_from_payload(payload: dict[str, Any], key: str, default: str) -> str:
    value = payload.get(key)
    result = default
    if isinstance(value, str) and value.strip():
        result = value.strip()
    return result


def _normalize_payload_text(value: Any) -> Any:
    result = value
    if isinstance(value, str):
        result = _normalize_inline_text(value)
    elif isinstance(value, list):
        result = [_normalize_payload_text(item) for item in value]
    elif isinstance(value, dict):
        result = {key: _normalize_payload_text(item) for key, item in value.items()}
    return result


def _normalize_inline_text(text: str) -> str:
    normalized_text = text.replace("\\r\\n", " ")
    normalized_text = normalized_text.replace("\\n", " ")
    normalized_text = normalized_text.replace("\\r", " ")
    normalized_text = normalized_text.replace("\r\n", " ")
    normalized_text = normalized_text.replace("\n", " ")
    normalized_text = normalized_text.replace("\r", " ")
    result = " ".join(normalized_text.split())
    return result


def _recommendation_lead(recommendations: tuple[FaceRecommendation, ...]) -> str:
    result = (
        "사주에서 보완이 필요한 리듬을 기준으로, 얼굴 추천 후보를 참고용으로 "
        "정리했어요."
    )
    if recommendations:
        result = recommendations[0].reason
    return result


def _synthesis_title(skip_face: bool) -> str:
    result = "사주와 얼굴 관찰이 만나는 지점"
    if skip_face:
        result = "사주 흐름을 정리하면"
    return result


def _synthesis_body(
    face_payload: dict[str, Any],
    saju_payload: dict[str, Any],
    strongest: str,
    weakest: str,
    skip_face: bool,
) -> str:
    saju_line = _text_from_payload(
        saju_payload,
        "essence",
        f"{strongest} 기운을 살리고 {weakest} 기운을 보완하는 흐름이 보여요.",
    )
    face_line = _text_from_payload(
        face_payload,
        "face_summary",
        "얼굴 관찰은 표현 방식과 대화 분위기를 보조적으로 보여줘요.",
    )
    result = saju_line
    if not skip_face:
        result = f"{saju_line} {face_line}"
    return result


def _dominant_element(counts: dict[str, int], strongest: bool) -> str:
    elements = tuple(counts.keys())
    selector = max
    score = lambda element: (counts[element], -elements.index(element))
    if not strongest:
        selector = min
        score = lambda element: (counts[element], elements.index(element))
    result = selector(elements, key=score)
    return result


def _lookup_pair_manse(
    repository: ManseRepository,
    left_profile: BirthProfile,
    right_profile: BirthProfile,
) -> tuple[ManseLookupResult, ManseLookupResult]:
    left_result = repository.lookup(left_profile)
    right_result = repository.lookup(right_profile)
    result = (left_result, right_result)
    return result


def _run_sequential_pair_capture(
    capture_runner,
    capture_config: CaptureConfig,
    output_dir: Path,
    inter_capture_delay_seconds: float,
) -> SequentialPairCaptureArtifact:
    left_dir = output_dir / "person_1"
    right_dir = output_dir / "person_2"
    left_config = _pair_capture_config(capture_config, "left")
    right_config = _pair_capture_config(capture_config, "right")
    left_artifact = capture_runner(left_config, left_dir)
    if inter_capture_delay_seconds > 0.0:
        time.sleep(inter_capture_delay_seconds)
    right_artifact = capture_runner(right_config, right_dir)
    result = SequentialPairCaptureArtifact(
        left=left_artifact,
        right=right_artifact,
    )
    return result


def _pair_capture_config(capture_config: CaptureConfig, side: str) -> CaptureConfig:
    result = capture_config
    if capture_config.mock_capture_enabled:
        metrics_json = ""
        if side == "left":
            metrics_json = capture_config.mock_pair_left_landmark_metrics_json
        elif side == "right":
            metrics_json = capture_config.mock_pair_right_landmark_metrics_json
        if metrics_json.strip():
            result = replace(capture_config, mock_landmark_metrics_json=metrics_json)
    return result


def _safe_generate(
    client: TextGenerator,
    prompt: str,
    image_path: Path | None,
    fallback: str,
    debug_label: str = "llm",
) -> _GeneratedText:
    text = fallback
    error = ""
    try:
        text = client.generate(prompt, image_path=image_path)
        text = _normalize_generated_output_text(text, debug_label)
        print(f"\n[LLM RAW:{debug_label}:BEGIN]\n{text}\n[LLM RAW:{debug_label}:END]\n")
    except Exception as exc:
        error = str(exc)
        text = f"{fallback}\n\n오류: {error}"
        print(f"\n[LLM RAW:{debug_label}:ERROR] {error}\n")
    result = _GeneratedText(text=text, error=error)
    return result


def _normalize_generated_output_text(text: str, label: str = "llm_json") -> str:
    payload, error = _load_json_payload_or_error(text, label=label)
    result = text
    if not error and payload:
        result = json.dumps(_normalize_payload_text(payload), ensure_ascii=False)
    return result


def _replace_day_master_honorifics(
    generated: _GeneratedText,
    name: str,
    label: str,
) -> _GeneratedText:
    cleaned_name = name.strip()
    result = generated
    if generated.error != "" or cleaned_name == "":
        return result
    payload, error = _load_json_payload_or_error(generated.text, label=label)
    if error:
        return result
    fixed_payload = _replace_day_master_honorifics_in_value(
        payload,
        f"{cleaned_name}님",
    )
    if fixed_payload != payload:
        print(f"[LLM JSON REPAIR:{label}] replaced day-master honorifics with input name.")
        result = _GeneratedText(
            text=json.dumps(_normalize_payload_text(fixed_payload), ensure_ascii=False),
            error="",
        )
    return result


def _replace_day_master_honorifics_in_value(value: Any, replacement: str) -> Any:
    result = value
    if isinstance(value, str):
        result = value
        for honorific in _SAJU_DAY_MASTER_HONORIFICS:
            result = result.replace(honorific, replacement)
    elif isinstance(value, list):
        result = [
            _replace_day_master_honorifics_in_value(item, replacement)
            for item in value
        ]
    elif isinstance(value, dict):
        result = {
            key: _replace_day_master_honorifics_in_value(item, replacement)
            for key, item in value.items()
        }
    return result


def _load_json_payload_or_error(
    text: str,
    label: str = "llm_json",
) -> tuple[dict[str, Any], str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_markdown_fence_text(cleaned)
    start = cleaned.find("{")
    if start >= 0:
        cleaned = cleaned[start:]
    payload: dict[str, Any] = {}
    error = ""
    try:
        loaded, repair_steps = _load_json_with_repairs(cleaned)
        if repair_steps:
            print(
                f"[LLM JSON REPAIR:{label}] applied repairs: "
                + ", ".join(repair_steps),
            )
        if isinstance(loaded, dict):
            payload = loaded
        else:
            error = "LLM JSON must be an object"
    except json.JSONDecodeError as exc:
        error = (
            "LLM JSON parse failed: "
            f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
        )
    result = (payload, error)
    return result


def _load_json_with_repairs(text: str) -> tuple[Any, tuple[str, ...]]:
    candidates = [text]
    candidate_steps = [tuple()]
    normalized_quotes = _normalize_json_quotes(text)
    if normalized_quotes != text:
        candidates.append(normalized_quotes)
        candidate_steps.append(("normalize_quotes",))
    repaired = normalized_quotes
    repaired_steps = list(candidate_steps[-1])
    for _ in range(3):
        next_repaired, step_names = _repair_json_text(repaired)
        if next_repaired == repaired:
            break
        repaired = next_repaired
        repaired_steps.extend(step_names)
        candidates.append(repaired)
        candidate_steps.append(tuple(repaired_steps))
    last_error: json.JSONDecodeError | None = None
    decoder = json.JSONDecoder()
    for candidate, steps in zip(candidates, candidate_steps):
        try:
            loaded, _ = decoder.raw_decode(candidate)
            return loaded, steps
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is None:
        raise json.JSONDecodeError("empty JSON", text, 0)
    raise last_error


def _normalize_json_quotes(text: str) -> str:
    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "’": "'",
        "‘": "'",
    }
    result = text
    for old_text, new_text in replacements.items():
        result = result.replace(old_text, new_text)
    return result


def _repair_json_text(text: str) -> tuple[str, tuple[str, ...]]:
    result = text
    applied_steps: list[str] = []
    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", result)
    if without_trailing_commas != result:
        applied_steps.append("remove_trailing_commas")
        result = without_trailing_commas
    with_missing_commas = re.sub(
        (
            r'("(?:[^"\\]|\\.)*"|\btrue\b|\bfalse\b|\bnull\b|'
            r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[}\]])'
            r'(\s*)(?=(?:"|[{\[]|\btrue\b|\bfalse\b|\bnull\b|-?\d))'
        ),
        _insert_missing_comma,
        result,
    )
    if with_missing_commas != result:
        applied_steps.append("insert_missing_commas")
        result = with_missing_commas
    return result, tuple(applied_steps)


def _insert_missing_comma(match: re.Match[str]) -> str:
    token = match.group(1)
    whitespace = match.group(2)
    if "," in whitespace:
        return match.group(0)
    return f"{token},{whitespace}"


def _strip_markdown_fence_text(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    result = "\n".join(lines)
    return result


def _build_birth_profile(
    name: str,
    birth_date: str,
    birth_time: str,
    gender: str,
) -> BirthProfile:
    cleaned_gender = gender.strip()
    if cleaned_gender == "":
        raise ValueError("성별은 남성 또는 여성으로 입력해야 합니다.")
    time_text, birth_time_known = _normalize_birth_time(birth_time)
    birth_datetime = datetime.strptime(
        f"{birth_date.strip()} {time_text}",
        "%Y-%m-%d %H:%M",
    )
    result = BirthProfile(
        name=name.strip(),
        birth_datetime=birth_datetime,
        gender=cleaned_gender,
        birth_time_known=birth_time_known,
    )
    return result


def _normalize_birth_time(birth_time: str) -> tuple[str, bool]:
    cleaned_time = birth_time.strip()
    birth_time_known = cleaned_time.lower() not in _UNKNOWN_BIRTH_TIME_VALUES
    time_text = cleaned_time
    if not birth_time_known:
        time_text = UNKNOWN_BIRTH_TIME_REPRESENTATIVE
    else:
        time_branch_time = representative_time_from_time_branch(cleaned_time)
        if time_branch_time is not None:
            time_text = time_branch_time
    result = (time_text, birth_time_known)
    return result


def _validate_mode(mode: str) -> str:
    cleaned = mode.strip()
    if cleaned not in COMPATIBILITY_MODES:
        raise ValueError("궁합 모드는 연인, 친구, 직장동료 중 하나여야 합니다.")
    result = cleaned
    return result



def _new_session_dir(base_dir: Path, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result = base_dir / f"{prefix}_{stamp}"
    result.mkdir(parents=True, exist_ok=True)
    return result
