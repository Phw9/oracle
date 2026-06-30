#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oracle_report.config import (
    _MOCK_LANDMARK_METRICS_JSON,
    _MOCK_PAIR_LEFT_LANDMARK_METRICS_JSON,
    _MOCK_PAIR_RIGHT_LANDMARK_METRICS_JSON,
)
from oracle_report.vision.landmarks import (
    LandmarkMetrics,
    _evaluate_physio_rules,
    _face_payload_seed,
    _format_prompt_metric_snapshot,
    _format_prompt_observation_context,
    _physiognomy_rule_repository,
    build_rule_based_face_analysis,
)
from oracle_report.vision.physiognomy_text_variations import (
    build_pair_face_payload,
    build_personal_face_payload,
)


BASE_METRICS = LandmarkMetrics(
    frontality_score=0.94,
    occlusion_score=0.96,
    eye_count=2,
    eyebrow_score=0.08,
    face_aspect_ratio=1.32,
    eye_width_ratio=0.182,
    eye_height_ratio=0.051,
    eye_aspect_ratio=0.279,
    eye_spacing_ratio=0.286,
    eye_tail_tilt=0.012,
    nose_length_ratio=0.241,
    mouth_width_ratio=0.362,
    mouth_height_ratio=0.043,
    lower_face_ratio=0.334,
    nose_length_width_ratio=1.418,
    mouth_corner_delta=0.004,
    upper_zone_ratio=0.332,
    middle_zone_ratio=0.338,
    lower_zone_ratio=0.330,
    third_balance_error=0.008,
    brow_eye_span_ratio=1.080,
    brow_eye_gap_ratio=0.081,
    nose_width_ratio=0.190,
    philtrum_chin_ratio=0.262,
    chin_length_ratio=0.214,
    jaw_width_ratio=0.662,
    mouth_balance_delta=0.006,
)


CASE_OVERRIDES: dict[str, dict[str, float | int]] = {
    "mock_personal": json.loads(_MOCK_LANDMARK_METRICS_JSON),
    "mock_pair_left": json.loads(_MOCK_PAIR_LEFT_LANDMARK_METRICS_JSON),
    "mock_pair_right": json.loads(_MOCK_PAIR_RIGHT_LANDMARK_METRICS_JSON),
    "balanced": {},
    "large_eyes_small_nose": {
        "eye_width_ratio": 0.205,
        "eye_aspect_ratio": 0.335,
        "eye_spacing_ratio": 0.302,
        "nose_length_ratio": 0.205,
        "nose_width_ratio": 0.158,
        "nose_length_width_ratio": 1.240,
        "brow_eye_gap_ratio": 0.096,
    },
    "narrow_eyes_reserved_mouth": {
        "eye_width_ratio": 0.145,
        "eye_aspect_ratio": 0.205,
        "eye_spacing_ratio": 0.244,
        "mouth_width_ratio": 0.305,
        "mouth_height_ratio": 0.031,
        "brow_eye_span_ratio": 0.930,
    },
    "wide_mouth_asymmetry": {
        "mouth_width_ratio": 0.438,
        "mouth_height_ratio": 0.056,
        "mouth_balance_delta": 0.054,
        "mouth_corner_delta": 0.052,
        "eye_tail_tilt": 0.035,
    },
    "long_face_strong_jaw": {
        "face_aspect_ratio": 1.475,
        "upper_zone_ratio": 0.318,
        "middle_zone_ratio": 0.330,
        "lower_zone_ratio": 0.352,
        "third_balance_error": 0.034,
        "chin_length_ratio": 0.255,
        "jaw_width_ratio": 0.742,
        "philtrum_chin_ratio": 0.226,
    },
    "wide_face_soft_jaw": {
        "face_aspect_ratio": 1.165,
        "upper_zone_ratio": 0.340,
        "middle_zone_ratio": 0.337,
        "lower_zone_ratio": 0.323,
        "third_balance_error": 0.017,
        "jaw_width_ratio": 0.585,
        "chin_length_ratio": 0.174,
        "philtrum_chin_ratio": 0.318,
    },
    "long_nose_focused_eyes": {
        "eye_width_ratio": 0.154,
        "eye_aspect_ratio": 0.220,
        "eye_spacing_ratio": 0.238,
        "nose_length_ratio": 0.292,
        "nose_width_ratio": 0.176,
        "nose_length_width_ratio": 1.690,
        "brow_eye_gap_ratio": 0.066,
    },
    "open_eyes_expressive_mouth": {
        "eye_width_ratio": 0.210,
        "eye_aspect_ratio": 0.350,
        "eye_spacing_ratio": 0.318,
        "eye_tail_tilt": 0.046,
        "mouth_width_ratio": 0.430,
        "mouth_height_ratio": 0.060,
        "brow_eye_span_ratio": 1.180,
    },
    "unbalanced_thirds": {
        "upper_zone_ratio": 0.285,
        "middle_zone_ratio": 0.370,
        "lower_zone_ratio": 0.345,
        "third_balance_error": 0.060,
        "face_aspect_ratio": 1.385,
        "chin_length_ratio": 0.238,
        "jaw_width_ratio": 0.688,
    },
}


def main() -> None:
    args = _parse_args()
    selected_cases = _selected_cases(args.case)
    if args.output_dir:
        _write_case_reports(args.output_dir, selected_cases, args)
        return
    if args.output:
        _write_comparison_report(args.output, selected_cases, args)
        return
    if args.mode in ("personal", "all"):
        for case_name in selected_cases:
            _print_personal_case(case_name, _metrics_for_case(case_name), args)
    if args.mode in ("pair", "all"):
        left_case = args.left_case or "mock_pair_left"
        right_case = args.right_case or "mock_pair_right"
        _print_pair_case(left_case, right_case, args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview rule-based physiognomy text with varied landmark metric presets. "
            "This does not touch runtime mock capture settings."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("personal", "pair", "all"),
        default="personal",
        help="Which report payload to preview.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASE_OVERRIDES),
        help=(
            "Personal case to preview. Repeat to preview multiple cases. "
            "mock_personal is the same default used by ORACLE_MOCK_CAPTURE_ENABLED=1."
        ),
    )
    parser.add_argument(
        "--left-case",
        choices=tuple(CASE_OVERRIDES),
        help="Left face preset for pair mode. Defaults to the runtime mock_pair_left preset.",
    )
    parser.add_argument(
        "--right-case",
        choices=tuple(CASE_OVERRIDES),
        help="Right face preset for pair mode. Defaults to the runtime mock_pair_right preset.",
    )
    parser.add_argument(
        "--pair-mode",
        choices=("연인", "친구", "직장동료"),
        default="친구",
        help="Relationship mode used for pair text variations.",
    )
    parser.add_argument("--left-name", default="첫번째")
    parser.add_argument("--right-name", default="두번째")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON payloads instead of a readable preview.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a combined comparison report to a file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write one report file per case. Personal files use the case name.",
    )
    parser.add_argument(
        "--output-format",
        choices=("markdown", "json"),
        default="markdown",
        help="Format used with --output.",
    )
    parser.add_argument(
        "--show-analysis",
        action="store_true",
        help="Include the markdown-style rule analysis text.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available preset case names and exit.",
    )
    args = parser.parse_args()
    if args.list_cases:
        for case_name in CASE_OVERRIDES:
            print(case_name)
        raise SystemExit(0)
    return args


def _selected_cases(case_names: list[str] | None) -> list[str]:
    result = list(CASE_OVERRIDES) if not case_names else case_names
    return result


def _metrics_for_case(case_name: str) -> LandmarkMetrics:
    values = dict(asdict(BASE_METRICS))
    values.update(CASE_OVERRIDES[case_name])
    result = replace(BASE_METRICS, **values)
    return result


def _matches_for_metrics(metrics: LandmarkMetrics):
    repository = _physiognomy_rule_repository()
    result = _evaluate_physio_rules(metrics, repository)
    return result


def _write_comparison_report(
    output_path: Path,
    selected_cases: list[str],
    args: argparse.Namespace,
) -> None:
    if args.output_format == "json":
        text = _build_json_comparison_report(selected_cases, args)
    else:
        text = _build_markdown_comparison_report(selected_cases, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Wrote comparison report: {output_path}")


def _write_case_reports(
    output_dir: Path,
    selected_cases: list[str],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".json" if args.output_format == "json" else ".md"
    written_paths: list[Path] = []
    if args.mode in ("personal", "all"):
        for case_name in selected_cases:
            output_path = output_dir / f"{case_name}{suffix}"
            if args.output_format == "json":
                text = json.dumps(
                    _personal_case_data(case_name, include_analysis=args.show_analysis),
                    ensure_ascii=False,
                    indent=2,
                ) + "\n"
            else:
                text = _single_personal_case_markdown(case_name, args.show_analysis)
            output_path.write_text(text, encoding="utf-8")
            written_paths.append(output_path)
    if args.mode in ("pair", "all"):
        left_case = args.left_case or "mock_pair_left"
        right_case = args.right_case or "mock_pair_right"
        file_name = f"pair_{left_case}_{right_case}_{args.pair_mode}{suffix}"
        output_path = output_dir / file_name
        if args.output_format == "json":
            text = json.dumps(
                _pair_case_data(left_case, right_case, args),
                ensure_ascii=False,
                indent=2,
            ) + "\n"
        else:
            text = _single_pair_case_markdown(left_case, right_case, args)
        output_path.write_text(text, encoding="utf-8")
        written_paths.append(output_path)
    print(f"Wrote {len(written_paths)} report files to: {output_dir}")


def _build_json_comparison_report(
    selected_cases: list[str],
    args: argparse.Namespace,
) -> str:
    report: dict[str, Any] = {
        "mode": args.mode,
        "personal_cases": [],
        "pair_case": None,
    }
    if args.mode in ("personal", "all"):
        report["personal_cases"] = [
            _personal_case_data(case_name, include_analysis=args.show_analysis)
            for case_name in selected_cases
        ]
    if args.mode in ("pair", "all"):
        left_case = args.left_case or "mock_pair_left"
        right_case = args.right_case or "mock_pair_right"
        report["pair_case"] = _pair_case_data(left_case, right_case, args)
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def _build_markdown_comparison_report(
    selected_cases: list[str],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# 관상 룰베이스 케이스 비교",
        "",
        f"- mode: `{args.mode}`",
        f"- pair_mode: `{args.pair_mode}`",
        "",
    ]
    if args.mode in ("personal", "all"):
        lines.extend(["## 개인 리포트 케이스", ""])
        for case_name in selected_cases:
            lines.extend(_personal_case_markdown(case_name, args.show_analysis))
            lines.append("")
    if args.mode in ("pair", "all"):
        left_case = args.left_case or "mock_pair_left"
        right_case = args.right_case or "mock_pair_right"
        lines.extend(_pair_case_markdown(left_case, right_case, args))
    return "\n".join(lines).rstrip() + "\n"


def _personal_case_data(
    case_name: str,
    *,
    include_analysis: bool = False,
) -> dict[str, Any]:
    metrics = _metrics_for_case(case_name)
    matches = _matches_for_metrics(metrics)
    payload = build_personal_face_payload(matches, _face_payload_seed(metrics))
    result: dict[str, Any] = {
        "case": case_name,
        "metrics": asdict(metrics),
        "metric_snapshot": _format_prompt_metric_snapshot(metrics),
        "matched_observations": _format_prompt_observation_context(matches),
        "payload": payload,
    }
    if include_analysis:
        result["analysis"] = build_rule_based_face_analysis(metrics, matches)
    return result


def _pair_case_data(
    left_case: str,
    right_case: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    left_metrics = _metrics_for_case(left_case)
    right_metrics = _metrics_for_case(right_case)
    left_matches = _matches_for_metrics(left_metrics)
    right_matches = _matches_for_metrics(right_metrics)
    seed = f"{left_case}:{right_case}:{args.pair_mode}"
    payload = build_pair_face_payload(
        left_matches,
        right_matches,
        args.left_name,
        args.right_name,
        seed,
        mode=args.pair_mode,
    )
    return {
        "left_case": left_case,
        "right_case": right_case,
        "pair_mode": args.pair_mode,
        "left_name": args.left_name,
        "right_name": args.right_name,
        "left_metrics": asdict(left_metrics),
        "right_metrics": asdict(right_metrics),
        "left_matched_observations": _format_prompt_observation_context(left_matches),
        "right_matched_observations": _format_prompt_observation_context(right_matches),
        "payload": payload,
    }


def _personal_case_markdown(
    case_name: str,
    include_analysis: bool,
) -> list[str]:
    data = _personal_case_data(case_name, include_analysis=include_analysis)
    payload = data["payload"]
    lines = [
        f"### {case_name}",
        "",
        "#### Metric Snapshot",
        "```text",
        data["metric_snapshot"],
        "```",
        "",
        "#### Matched Observations",
        "```text",
        data["matched_observations"],
        "```",
        "",
        f"#### Subtitle",
        str(payload.get("face_subtitle", "")),
        "",
        "#### Blocks",
    ]
    lines.extend(_payload_blocks_markdown(payload.get("face_blocks", [])))
    lines.extend(
        [
            "",
            "#### Summary",
            str(payload.get("face_summary", "")),
        ],
    )
    if include_analysis:
        lines.extend(
            [
                "",
                "#### Markdown Analysis",
                "```text",
                str(data.get("analysis", "")),
                "```",
            ],
        )
    return lines


def _single_personal_case_markdown(
    case_name: str,
    include_analysis: bool,
) -> str:
    lines = [
        f"# {case_name}",
        "",
        "개인 관상 룰베이스 미리보기",
        "",
    ]
    lines.extend(_personal_case_markdown(case_name, include_analysis))
    return "\n".join(lines).rstrip() + "\n"


def _pair_case_markdown(
    left_case: str,
    right_case: str,
    args: argparse.Namespace,
) -> list[str]:
    data = _pair_case_data(left_case, right_case, args)
    payload = data["payload"]
    lines = [
        "## 궁합 리포트 케이스",
        "",
        f"### {left_case} + {right_case} ({args.pair_mode})",
        "",
        f"#### {args.left_name} Matched Observations",
        "```text",
        data["left_matched_observations"],
        "```",
        "",
        f"#### {args.right_name} Matched Observations",
        "```text",
        data["right_matched_observations"],
        "```",
        "",
        "#### Blocks",
    ]
    lines.extend(_payload_blocks_markdown(payload.get("pair_blocks", [])))
    lines.extend(
        [
            "",
            "#### Synthesis",
            f"**{payload.get('synthesis_title', '')}**",
            "",
            str(payload.get("synthesis_body", "")),
            "",
            "#### Action",
            f"**{payload.get('action_title', '')}**",
            "",
            str(payload.get("action_body", "")),
        ],
    )
    return lines


def _single_pair_case_markdown(
    left_case: str,
    right_case: str,
    args: argparse.Namespace,
) -> str:
    lines = [
        f"# pair_{left_case}_{right_case}_{args.pair_mode}",
        "",
        "궁합 관상 룰베이스 미리보기",
        "",
    ]
    lines.extend(_pair_case_markdown(left_case, right_case, args))
    return "\n".join(lines).rstrip() + "\n"


def _payload_blocks_markdown(blocks: Any) -> list[str]:
    lines: list[str] = []
    if not isinstance(blocks, list):
        return lines
    for index, block in enumerate(blocks, start=1):
        if not isinstance(block, dict):
            continue
        lines.extend(
            [
                "",
                f"{index}. **{block.get('category', '')} / {block.get('title', '')}**",
                "",
                f"   - summary: {block.get('summary', '')}",
                f"   - body: {block.get('body', '')}",
            ],
        )
    return lines


def _print_personal_case(
    case_name: str,
    metrics: LandmarkMetrics,
    args: argparse.Namespace,
) -> None:
    matches = _matches_for_metrics(metrics)
    payload = build_personal_face_payload(matches, _face_payload_seed(metrics))
    if args.json:
        print(
            json.dumps(
                {
                    "case": case_name,
                    "metrics": asdict(metrics),
                    "payload": payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return
    print(f"\n## {case_name}")
    _print_metric_snapshot(metrics)
    print("\n### matched observations")
    print(_format_prompt_observation_context(matches))
    print("\n### face payload")
    _print_payload(payload)
    if args.show_analysis:
        print("\n### markdown analysis")
        print(build_rule_based_face_analysis(metrics, matches))


def _print_pair_case(
    left_case: str,
    right_case: str,
    args: argparse.Namespace,
) -> None:
    left_metrics = _metrics_for_case(left_case)
    right_metrics = _metrics_for_case(right_case)
    left_matches = _matches_for_metrics(left_metrics)
    right_matches = _matches_for_metrics(right_metrics)
    seed = f"{left_case}:{right_case}:{args.pair_mode}"
    payload = build_pair_face_payload(
        left_matches,
        right_matches,
        args.left_name,
        args.right_name,
        seed,
        mode=args.pair_mode,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "left_case": left_case,
                    "right_case": right_case,
                    "pair_mode": args.pair_mode,
                    "left_metrics": asdict(left_metrics),
                    "right_metrics": asdict(right_metrics),
                    "payload": payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return
    print(f"\n## pair: {left_case} + {right_case} ({args.pair_mode})")
    print(f"\n### {args.left_name} matched observations")
    print(_format_prompt_observation_context(left_matches))
    print(f"\n### {args.right_name} matched observations")
    print(_format_prompt_observation_context(right_matches))
    print("\n### pair payload")
    _print_payload(payload)


def _print_metric_snapshot(metrics: LandmarkMetrics) -> None:
    lines = _format_prompt_metric_snapshot(metrics).splitlines()
    for line in lines:
        print(line)


def _print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
