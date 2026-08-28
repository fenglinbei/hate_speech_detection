#!/usr/bin/env python3
"""Freeze or validate an immutable Stage 1 training plan."""

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
from data.training_plan import freeze_training_plan, validate_training_plan


def _model_binding(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "model binding must be MODEL_KEY=MODEL_REF"
        )
    model_key, model_ref = value.split("=", 1)
    if not model_key or not model_ref:
        raise argparse.ArgumentTypeError(
            "model binding must be MODEL_KEY=MODEL_REF"
        )
    return model_key, Path(model_ref)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze", help="resolve a mutable source recipe into an immutable plan"
    )
    freeze.add_argument("--source-spec", required=True, type=Path)
    freeze.add_argument(
        "--scope",
        required=True,
        choices=("formal", "pilot", "engineering-smoke"),
    )
    freeze.add_argument("--context-ref", required=True, type=Path)
    freeze.add_argument(
        "--training-evidence-ref",
        type=Path,
        help="required for pilot/formal; forbidden for engineering-smoke",
    )
    freeze.add_argument(
        "--train-partition-ref",
        type=Path,
        help="required for pilot/formal; forbidden for engineering-smoke",
    )
    freeze.add_argument(
        "--base-model-ref",
        type=Path,
        help="required for pilot/formal; forbidden for engineering-smoke",
    )
    freeze.add_argument("--environment-ref", required=True, type=Path)
    freeze.add_argument(
        "--bind-non-training-model",
        action="append",
        default=[],
        type=_model_binding,
        metavar="MODEL_KEY=MODEL_REF",
        help=(
            "exact legacy-smoke-only binding; required once per engineering "
            "non-training slot and forbidden for pilot/formal"
        ),
    )
    freeze.add_argument("--write-ref", required=True, type=Path)
    freeze.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    freeze.add_argument("--target-root", type=Path, default=None)
    freeze.add_argument(
        "--decision-register",
        type=Path,
        default=REPOSITORY_ROOT / "config/stage1/decision_register.json",
    )
    freeze.add_argument(
        "--train-code",
        type=Path,
        default=REPOSITORY_ROOT / "src/finetune/train.py",
    )
    freeze.add_argument(
        "--runtime-code",
        type=Path,
        default=REPOSITORY_ROOT / "src/finetune/stage1_runtime.py",
    )
    validate = subparsers.add_parser(
        "validate", help="read-only validation of an immutable training plan"
    )
    validate.add_argument("--training-plan-ref", required=True, type=Path)
    validate.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_training_plan(
                source_spec_path=args.source_spec,
                scope=args.scope,
                context_ref=args.context_ref,
                training_evidence_ref=args.training_evidence_ref,
                train_partition_ref=args.train_partition_ref,
                base_model_ref=args.base_model_ref,
                environment_ref=args.environment_ref,
                write_ref=args.write_ref,
                workspace_root=args.workspace_root,
                target_root=args.target_root,
                decision_register_path=args.decision_register,
                train_code_path=args.train_code,
                runtime_code_path=args.runtime_code,
                non_training_model_bindings=args.bind_non_training_model,
            )
        else:
            result = validate_training_plan(
                args.training_plan_ref,
                workspace_root=args.workspace_root,
            )
    except (TrainingArtifactError, OSError, ValueError) as exc:
        print(f"[stage1-training-plan] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
