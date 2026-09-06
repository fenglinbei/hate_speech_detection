#!/usr/bin/env python3
"""Build and operate the offline WP3 S2.1 development-review lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_lex.terminology_candidate_review import (  # noqa: E402
    CandidateReviewError,
    build_development_frame,
    build_development_report,
    finalize_development_gold,
    validate_development_frame,
)
from data.training_artifacts import (  # noqa: E402
    TrainingArtifactError,
    resolve_locator_ref,
)


DEFAULT_STAGE_ROOT = Path("exps/causal_context/stage1_p0")
DEFAULT_GENERATOR_ROOT = DEFAULT_STAGE_ROOT / "wp3_candidate_generators_v2"


def _in_workspace(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _resolve_target(
    *,
    workspace_root: Path,
    explicit: Path | None,
    locator: Path,
    expected_kind: str,
) -> Path:
    if explicit is not None:
        return _in_workspace(workspace_root, explicit).resolve()
    _, target = resolve_locator_ref(
        _in_workspace(workspace_root, locator), expected_kind
    )
    return target


def _common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    command.add_argument(
        "--generator-config",
        type=Path,
        default=Path("config/stage1/wp3_candidate_generators_v1.json"),
    )


def _frame_source(command: argparse.ArgumentParser) -> None:
    source = command.add_mutually_exclusive_group()
    source.add_argument("--frame-dir", type=Path)
    source.add_argument(
        "--frame-ref",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "refs/development_frame_ref.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build-dev-frame", help="Build the immutable 424-case S2.1 frame."
    )
    _common(build)
    build.add_argument(
        "--data-ref", type=Path, default=DEFAULT_STAGE_ROOT / "refs/data_ref.json"
    )
    build.add_argument(
        "--train-partition-ref",
        type=Path,
        default=DEFAULT_STAGE_ROOT / "refs/train_partition_ref.json",
    )
    build.add_argument(
        "--a1-package",
        type=Path,
        default=(
            DEFAULT_STAGE_ROOT
            / "review_packages/termspan-review-4ed4230bd227ab2744be"
        ),
    )
    build.add_argument(
        "--a1-source-map",
        type=Path,
        default=(
            DEFAULT_STAGE_ROOT
            / "review_packages/.private/termspan-review-4ed4230bd227ab2744be.source_map.json"
        ),
    )
    build.add_argument(
        "--dual-model-package",
        type=Path,
        default=(
            DEFAULT_STAGE_ROOT
            / "review_packages/dmspan-review-30583732ec1adb2f"
        ),
    )
    build.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "development_frames",
    )
    build.add_argument(
        "--write-ref",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "refs/development_frame_ref.json",
    )

    validate = commands.add_parser(
        "validate-dev-frame", help="Independently validate a development frame."
    )
    _common(validate)
    _frame_source(validate)

    serve = commands.add_parser(
        "serve-dev-review", help="Serve the two-phase loopback review UI."
    )
    _common(serve)
    _frame_source(serve)
    serve.add_argument("--reviewer-id", required=True)
    serve.add_argument(
        "--session-file",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "review_working/session.json",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument(
        "--public-origin",
        help="Exact trusted HTTPS reverse-proxy origin, for example https://hsd.example.",
    )
    serve.add_argument("--check", action="store_true")

    finalize = commands.add_parser(
        "finalize-dev-gold", help="Finalize a complete two-phase review."
    )
    _common(finalize)
    _frame_source(finalize)
    finalize.add_argument("--reviewer-id", required=True)
    finalize.add_argument(
        "--session-file",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "review_working/session.json",
    )
    finalize.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "development_gold",
    )
    finalize.add_argument(
        "--write-ref",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "refs/development_gold_ref.json",
    )

    report = commands.add_parser(
        "report-dev", help="Build and independently validate the S2.1 report."
    )
    _common(report)
    _frame_source(report)
    gold_source = report.add_mutually_exclusive_group()
    gold_source.add_argument("--gold-dir", type=Path)
    gold_source.add_argument(
        "--gold-ref",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "refs/development_gold_ref.json",
    )
    report.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "development_reports",
    )
    report.add_argument(
        "--write-ref",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "refs/development_report_ref.json",
    )
    return parser


def _safe_frame_result(result: dict[str, Any]) -> dict[str, Any]:
    manifest = result["manifest"]
    return {
        "frame_id": result["frame_id"],
        "target": result["target"],
        "payload_manifest_sha256": result["payload_manifest_sha256"],
        "case_count": result["case_count"],
        "selection_counts": manifest["historical_sources"]["selection_counts"],
        "fit_data_sha256": manifest["fit_data_sha256"],
        "fit_ids_sha256": manifest["fit_ids_sha256"],
        "development_only": True,
        "sealed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.workspace_root.resolve()
    generator_config = _in_workspace(root, args.generator_config)
    try:
        if args.command == "build-dev-frame":
            result = build_development_frame(
                workspace_root=root,
                data_ref=_in_workspace(root, args.data_ref),
                train_partition_ref=_in_workspace(root, args.train_partition_ref),
                generator_config_path=generator_config,
                a1_package=_in_workspace(root, args.a1_package),
                a1_source_map=_in_workspace(root, args.a1_source_map),
                dual_model_package=_in_workspace(root, args.dual_model_package),
                output_root=_in_workspace(root, args.output_root),
                write_ref=_in_workspace(root, args.write_ref),
            )
            output = _safe_frame_result(result)
        else:
            frame = _resolve_target(
                workspace_root=root,
                explicit=args.frame_dir,
                locator=args.frame_ref,
                expected_kind="wp3-s21-development-frame",
            )
            if args.command == "validate-dev-frame":
                output = _safe_frame_result(
                    validate_development_frame(
                        frame,
                        workspace_root=root,
                        generator_config_path=generator_config,
                    )
                )
            elif args.command == "serve-dev-review":
                from tools.wp3_candidate_review_ui.server import run_server

                return run_server(
                    workspace_root=root,
                    frame_dir=frame,
                    generator_config_path=generator_config,
                    session_path=_in_workspace(root, args.session_file),
                    reviewer_id=args.reviewer_id,
                    host=args.host,
                    port=args.port,
                    check=args.check,
                    public_origin=args.public_origin,
                )
            elif args.command == "finalize-dev-gold":
                result = finalize_development_gold(
                    frame_dir=frame,
                    session_path=_in_workspace(root, args.session_file),
                    workspace_root=root,
                    generator_config_path=generator_config,
                    output_root=_in_workspace(root, args.output_root),
                    reviewer_id=args.reviewer_id,
                    write_ref=_in_workspace(root, args.write_ref),
                )
                output = {
                    "gold_id": result["gold_id"],
                    "target": result["target"],
                    "payload_manifest_sha256": result[
                        "payload_manifest_sha256"
                    ],
                    "development_only": True,
                    "sealed": False,
                }
            else:
                gold = _resolve_target(
                    workspace_root=root,
                    explicit=args.gold_dir,
                    locator=args.gold_ref,
                    expected_kind="wp3-s21-development-gold",
                )
                result = build_development_report(
                    frame_dir=frame,
                    gold_dir=gold,
                    output_root=_in_workspace(root, args.output_root),
                    write_ref=_in_workspace(root, args.write_ref),
                )
                output = {
                    "report_id": result["report_id"],
                    "target": result["target"],
                    "payload_manifest_sha256": result[
                        "payload_manifest_sha256"
                    ],
                    "scope": result["report"]["scope"],
                    "sealed": False,
                }
    except (CandidateReviewError, TrainingArtifactError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
