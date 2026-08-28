#!/usr/bin/env python3
"""Build and validate Stage 1 training evidence/schedule artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.training_artifacts import TrainingArtifactError
from data.training_evidence import build_training_evidence, validate_training_evidence
from data.training_schedule import (
    build_training_schedule,
    validate_training_schedule,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate content-addressed Stage 1 training schedules."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    evidence = commands.add_parser(
        "build-evidence", help="Freeze train queries and canonical L/D evidence sets."
    )
    evidence.add_argument("--context-ref", required=True, type=Path)
    evidence.add_argument("--train-partition-ref", required=True, type=Path)
    evidence.add_argument("--base-model-ref", required=True, type=Path)
    evidence.add_argument("--write-ref", required=True, type=Path)
    evidence.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    evidence.add_argument("--target-root", type=Path)

    validate_evidence = commands.add_parser(
        "validate-evidence", help="Read-only validation of frozen training evidence."
    )
    validate_evidence.add_argument("--training-evidence-ref", required=True, type=Path)
    validate_evidence.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)

    build = commands.add_parser(
        "build-schedule", help="Build the complete formal slot×epoch schedule."
    )
    build.add_argument("--training-plan-ref", required=True, type=Path)
    build.add_argument(
        "--training-evidence-ref",
        type=Path,
        help="Optional explicit assertion; the authoritative dependency is resolved from the plan.",
    )
    build.add_argument(
        "--train-partition-ref",
        type=Path,
        help="Optional explicit assertion; the authoritative dependency is resolved from the plan.",
    )
    build.add_argument(
        "--tokenizer-revision",
        help="Optional assertion; by default the revision is inherited from frozen evidence.",
    )
    build.add_argument("--artifact-root", type=Path)
    build.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    build.add_argument("--write-ref", required=True, type=Path)

    validate = commands.add_parser(
        "validate-schedule", help="Read-only validation of a frozen schedule."
    )
    validate.add_argument("--schedule-ref", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-evidence":
            result = build_training_evidence(
                context_ref=args.context_ref,
                train_partition_ref=args.train_partition_ref,
                base_model_ref=args.base_model_ref,
                write_ref=args.write_ref,
                workspace_root=args.workspace_root,
                target_root=args.target_root,
            )
        elif args.command == "validate-evidence":
            result = validate_training_evidence(
                args.training_evidence_ref,
                workspace_root=args.workspace_root,
            )
        elif args.command == "build-schedule":
            result = build_training_schedule(
                training_plan_ref=args.training_plan_ref,
                training_evidence_ref=args.training_evidence_ref,
                train_partition_ref=args.train_partition_ref,
                tokenizer=None,
                tokenizer_revision=args.tokenizer_revision,
                artifact_root=args.artifact_root,
                workspace_root=args.workspace_root,
                write_ref=args.write_ref,
            )
        elif args.command == "validate-schedule":
            result = validate_training_schedule(
                schedule_ref=args.schedule_ref, tokenizer=None
            )
        else:  # pragma: no cover - argparse prevents this branch
            raise TrainingArtifactError(f"unsupported command: {args.command}")
    except (TrainingArtifactError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
