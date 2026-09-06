#!/usr/bin/env python3
"""Operate the side-by-side WP3 S2.1b development lifecycle."""

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

from build_lex.terminology_candidate_development_v2 import (  # noqa: E402
    CandidateDevelopmentV2Error,
    RAW_GOLD_ARTIFACT_KIND,
    build_locked_raw_gold,
    load_s21b_protocol,
    validate_locked_raw_gold,
)
from data.training_artifacts import (  # noqa: E402
    TrainingArtifactError,
    resolve_locator_ref,
)


DEFAULT_STAGE_ROOT = Path("exps/causal_context/stage1_p0")
DEFAULT_GENERATOR_ROOT = DEFAULT_STAGE_ROOT / "wp3_candidate_generators_v2"
DEFAULT_FRAME_REF = DEFAULT_GENERATOR_ROOT / "refs/development_frame_ref.json"
DEFAULT_RAW_GOLD_REF = DEFAULT_GENERATOR_ROOT / "refs/s21b_locked_raw_gold_ref.json"
DEFAULT_PROTOCOL = Path("config/stage1/wp3_candidate_generators_s21b_v1.json")
DEFAULT_SCHEMA = Path("schemas/wp3_s21_locked_raw_gold_v1.schema.json")
DEFAULT_LEGACY_CONFIG = Path("config/stage1/wp3_candidate_generators_v1.json")


def _in_workspace(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _resolve(
    *, root: Path, explicit: Path | None, ref: Path, expected_kind: str
) -> Path:
    if explicit is not None:
        return _in_workspace(root, explicit).resolve()
    _, target = resolve_locator_ref(_in_workspace(root, ref), expected_kind)
    return target


def _common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    command.add_argument(
        "--legacy-generator-config", type=Path, default=DEFAULT_LEGACY_CONFIG
    )
    command.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)


def _frame(command: argparse.ArgumentParser) -> None:
    source = command.add_mutually_exclusive_group()
    source.add_argument("--frame-dir", type=Path)
    source.add_argument("--frame-ref", type=Path, default=DEFAULT_FRAME_REF)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser(
        "freeze-locked-raw-gold",
        help="Freeze 424 locked raw annotations without completing legacy Phase B.",
    )
    _common(freeze)
    _frame(freeze)
    freeze.add_argument("--reviewer-id", required=True)
    freeze.add_argument(
        "--session-file",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "review_working/session.json",
    )
    freeze.add_argument("--annotation-schema", type=Path, default=DEFAULT_SCHEMA)
    freeze.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_GENERATOR_ROOT / "s21b_locked_raw_gold",
    )
    freeze.add_argument("--write-ref", type=Path, default=DEFAULT_RAW_GOLD_REF)

    validate = commands.add_parser(
        "validate-locked-raw-gold",
        help="Independently validate a locked raw-gold artifact.",
    )
    _common(validate)
    _frame(validate)
    source = validate.add_mutually_exclusive_group()
    source.add_argument("--raw-gold-dir", type=Path)
    source.add_argument("--raw-gold-ref", type=Path, default=DEFAULT_RAW_GOLD_REF)
    validate.add_argument("--require-current-sources", action="store_true")
    validate.add_argument("--require-current-validator", action="store_true")

    show = commands.add_parser(
        "show-protocol", help="Validate and summarize the accepted S2.1b protocol."
    )
    _common(show)
    return parser


def _safe_result(result: dict[str, Any]) -> dict[str, Any]:
    manifest = result["manifest"]
    return {
        "raw_gold_id": result["raw_gold_id"],
        "target": result["target"],
        "payload_manifest_sha256": result["payload_manifest_sha256"],
        "case_count": manifest["case_count"],
        "mention_count": manifest["mention_count"],
        "empty_record_count": manifest["empty_record_count"],
        "raw_amendment_count": manifest["raw_amendment_count"],
        "post_reveal_raw_amendment_count": manifest[
            "post_reveal_raw_amendment_count"
        ],
        "legacy_diagnostic_decision_count_at_freeze": manifest[
            "legacy_diagnostic_decision_count_at_freeze"
        ],
        "legacy_phase_b_status": manifest["legacy_phase_b_status"],
        "scope": manifest["scope"],
        "sealed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.workspace_root.resolve()
    protocol_path = _in_workspace(root, args.protocol)
    legacy_config = _in_workspace(root, args.legacy_generator_config)
    try:
        if args.command == "show-protocol":
            protocol = load_s21b_protocol(protocol_path)
            output = {
                "protocol_revision": protocol["protocol_revision"],
                "legacy_frame_dependency": protocol["legacy_frame_dependency"],
                "legacy_phase_b": protocol["legacy_phase_b"],
                "locked_raw_gold": protocol["locked_raw_gold"],
                "current_generator_pilot": protocol["current_generator_pilot"],
                "phase_b_v2": protocol["phase_b_v2"],
                "execution": protocol["execution"],
                "s22": protocol["s22"],
            }
        else:
            frame_dir = _resolve(
                root=root,
                explicit=args.frame_dir,
                ref=args.frame_ref,
                expected_kind="wp3-s21-development-frame",
            )
            if args.command == "freeze-locked-raw-gold":
                result = build_locked_raw_gold(
                    frame_dir=frame_dir,
                    session_path=_in_workspace(root, args.session_file),
                    workspace_root=root,
                    legacy_generator_config_path=legacy_config,
                    protocol_path=protocol_path,
                    annotation_schema_path=_in_workspace(
                        root, args.annotation_schema
                    ),
                    output_root=_in_workspace(root, args.output_root),
                    reviewer_id=args.reviewer_id,
                    write_ref=_in_workspace(root, args.write_ref),
                )
                output = _safe_result(result)
            else:
                raw_gold_dir = _resolve(
                    root=root,
                    explicit=args.raw_gold_dir,
                    ref=args.raw_gold_ref,
                    expected_kind=RAW_GOLD_ARTIFACT_KIND,
                )
                result = validate_locked_raw_gold(
                    raw_gold_dir,
                    frame_dir=frame_dir,
                    workspace_root=root,
                    legacy_generator_config_path=legacy_config,
                    protocol_path=protocol_path,
                    require_current_sources=args.require_current_sources,
                    require_current_validator=args.require_current_validator,
                )
                output = _safe_result(result)
    except (
        CandidateDevelopmentV2Error,
        TrainingArtifactError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
