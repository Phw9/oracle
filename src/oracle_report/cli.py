from __future__ import annotations

import argparse
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from oracle_report.config import (
    load_capture_config,
    load_face_llm_config,
    load_llm_config,
    load_report_llm_config,
)
from oracle_report.llm import LlamaCppChatClient
from oracle_report.models import BirthProfile
from oracle_report.physiognomy import FaceReadingInput
from oracle_report.recommender import format_recommendations, recommend_faces
from oracle_report.report import (
    ReportRequest,
    build_face_analysis_prompt,
    build_personal_final_prompt,
    generate_report,
)
from oracle_report.saju.engine import SajuReading
from oracle_report.saju.repository import ManseLookupResult, ManseRepository
from oracle_report.vision.runtime import run_capture


_DEFAULT_FACE_ANALYSIS_TEXT = "관상 분석 결과를 여기에 넣습니다."
_DEFAULT_FACE_DB_PATH = "data/face_recommendations.sqlite"
_DEFAULT_MANSE_DB_PATH = "data/manse.sqlite"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = 0
    try:
        if args.command == "capture":
            result = _run_capture_command(args)
        elif args.command == "report":
            result = _run_report_command(args)
        elif args.command == "run":
            result = _run_full_command(args)
        elif args.command == "serve":
            result = _run_serve_command(args)
        elif args.command == "prompt":
            result = _run_prompt_command(args)
        elif args.command == "prompt-run":
            result = _run_prompt_result_command(args)
        else:
            parser.print_help()
            result = 2
    except KeyboardInterrupt:
        print("cancelled")
        result = 130
    except Exception as exc:
        print(f"error: {exc}")
        result = 1
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oracle-report")
    subparsers = parser.add_subparsers(dest="command")

    capture = subparsers.add_parser("capture", help="capture a face image")
    _add_capture_args(capture)

    report = subparsers.add_parser("report", help="generate a report from inputs")
    _add_report_args(report)

    run = subparsers.add_parser("run", help="capture then generate a report")
    _add_capture_args(run)
    _add_report_args(run)

    serve = subparsers.add_parser("serve", help="run the lightweight Flask UI")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8501)
    serve.add_argument("--debug", action="store_true")

    prompt = subparsers.add_parser("prompt", help="print workflow prompt inputs")
    _add_prompt_args(prompt)

    prompt_run = subparsers.add_parser("prompt-run", help="run one workflow prompt")
    _add_prompt_args(prompt_run)

    result = parser
    return result


def _add_capture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--face-analysis-mode", type=int, choices=(1, 2))


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", required=True)
    parser.add_argument("--birth-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--birth-time", required=True, help="HH:MM")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)


def _add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        choices=("face-analysis", "saju-reading", "personal-final"),
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--birth-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--birth-time", default="", help="HH:MM")
    parser.add_argument("--gender", required=True)
    parser.add_argument("--target-gender", default="")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--manse-db", type=Path)
    parser.add_argument("--face-db", type=Path)
    parser.add_argument("--face-analysis", default="")
    parser.add_argument("--face-analysis-file", type=Path)
    parser.add_argument("--recommendation-text", default="")
    parser.add_argument("--recommendation-file", type=Path)


def _run_capture_command(args: argparse.Namespace) -> int:
    config = _override_capture_config(args)
    artifact = run_capture(config, output_dir=args.output_dir)
    print(artifact.image_path)
    result = 0
    return result


def _run_report_command(args: argparse.Namespace) -> int:
    request = _build_report_request(args, image_path=args.image)
    artifact = generate_report(request, load_llm_config())
    _print_report_result(artifact.markdown, artifact.output_path)
    result = 0
    return result


def _run_full_command(args: argparse.Namespace) -> int:
    config = _override_capture_config(args)
    capture_artifact = run_capture(config, output_dir=args.output_dir)
    request = _build_report_request(args, image_path=capture_artifact.image_path)
    face_input = FaceReadingInput(
        image_path=capture_artifact.image_path,
        quality=capture_artifact.quality,
    )
    request = ReportRequest(
        birth_profile=request.birth_profile,
        face_input=face_input,
        output_path=request.output_path,
    )
    artifact = generate_report(request, load_llm_config())
    _print_report_result(artifact.markdown, artifact.output_path)
    result = 0
    return result


def _run_serve_command(args: argparse.Namespace) -> int:
    from oracle_report.web import create_app

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    result = 0
    return result


def _run_prompt_command(args: argparse.Namespace) -> int:
    prompt_text = _build_prompt_text(args)
    print(prompt_text)
    result = 0
    return result


def _run_prompt_result_command(args: argparse.Namespace) -> int:
    prompt_text = _build_prompt_text(args)
    output_text = prompt_text
    if args.target == "face-analysis":
        output_text = LlamaCppChatClient(load_face_llm_config()).generate(
            prompt_text,
            image_path=args.image,
        )
    elif args.target == "personal-final":
        output_text = LlamaCppChatClient(load_report_llm_config()).generate(
            prompt_text,
            image_path=None,
        )
    print(output_text)
    result = 0
    return result


def _override_capture_config(args: argparse.Namespace):
    config = load_capture_config()
    camera_index = config.camera_index if args.camera_index is None else args.camera_index
    output_dir = config.output_dir if args.output_dir is None else args.output_dir
    show_preview = config.show_preview and not args.no_preview
    face_analysis_mode = (
        config.face_analysis_mode
        if args.face_analysis_mode is None
        else args.face_analysis_mode
    )
    result = replace(
        config,
        camera_index=camera_index,
        output_dir=output_dir,
        show_preview=show_preview,
        face_analysis_mode=face_analysis_mode,
    )
    return result


def _build_prompt_text(args: argparse.Namespace) -> str:
    profile = _build_prompt_birth_profile(args)
    result = ""
    if args.target == "face-analysis":
        face_input = FaceReadingInput(image_path=args.image, quality=None)
        result = build_face_analysis_prompt(profile, face_input)
    elif args.target == "saju-reading":
        result = _lookup_manse(args, profile).formatted_text
    elif args.target == "personal-final":
        manse_lookup = _lookup_manse(args, profile)
        face_analysis = _read_text_option(
            args.face_analysis,
            args.face_analysis_file,
            _DEFAULT_FACE_ANALYSIS_TEXT,
        )
        recommendation_text = _build_recommendation_text(args, manse_lookup.reading)
        result = build_personal_final_prompt(
            profile,
            manse_lookup.formatted_text,
            face_analysis,
            recommendation_text,
        )
    return result


def _build_prompt_birth_profile(args: argparse.Namespace) -> BirthProfile:
    birth_time = args.birth_time.strip()
    birth_time_known = birth_time != ""
    parse_time = birth_time
    if not birth_time_known:
        parse_time = "12:00"
    birth_datetime = _parse_birth_datetime(args.birth_date, parse_time)
    result = BirthProfile(
        name=args.name.strip(),
        birth_datetime=birth_datetime,
        gender=args.gender.strip(),
        birth_time_known=birth_time_known,
    )
    return result


def _lookup_manse(
    args: argparse.Namespace,
    profile: BirthProfile,
) -> ManseLookupResult:
    db_path = _configured_path(
        args.manse_db,
        "ORACLE_MANSE_DB_PATH",
        _DEFAULT_MANSE_DB_PATH,
    )
    result = ManseRepository(db_path).lookup(profile)
    return result


def _build_recommendation_text(
    args: argparse.Namespace,
    reading: SajuReading,
) -> str:
    result = _read_text_option(
        args.recommendation_text,
        args.recommendation_file,
        "",
    )
    if result == "":
        db_path = _configured_path(
            args.face_db,
            "ORACLE_FACE_DB_PATH",
            _DEFAULT_FACE_DB_PATH,
        )
        recommendations = recommend_faces(
            db_path,
            args.target_gender,
            reading,
        )
        result = format_recommendations(recommendations)
    return result


def _configured_path(
    argument_path: Path | None,
    env_name: str,
    default_path: str,
) -> Path:
    result = argument_path
    if result is None:
        result = Path(os.getenv(env_name, default_path))
    return result


def _read_text_option(
    inline_text: str,
    file_path: Path | None,
    default_text: str,
) -> str:
    result = default_text
    if inline_text.strip() != "":
        result = inline_text
    if file_path is not None:
        result = file_path.read_text(encoding="utf-8")
    return result


def _build_report_request(
    args: argparse.Namespace,
    image_path: Path | None,
) -> ReportRequest:
    birth_datetime = _parse_birth_datetime(args.birth_date, args.birth_time)
    birth_profile = BirthProfile(name=args.name, birth_datetime=birth_datetime)
    face_input = FaceReadingInput(image_path=image_path, quality=None)
    result = ReportRequest(
        birth_profile=birth_profile,
        face_input=face_input,
        output_path=args.output,
    )
    return result


def _parse_birth_datetime(birth_date: str, birth_time: str) -> datetime:
    result = datetime.strptime(f"{birth_date} {birth_time}", "%Y-%m-%d %H:%M")
    return result


def _print_report_result(markdown: str, output_path: Path | None) -> None:
    if output_path is None:
        print(markdown)
    else:
        print(output_path)
