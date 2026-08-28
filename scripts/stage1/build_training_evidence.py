#!/usr/bin/env python3
"""Build or validate the immutable Stage 1 train-only evidence target."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-evidence", help="freeze train query and selected L/D evidence sets"
    )
    build.add_argument("--context-ref", required=True, type=Path)
    build.add_argument("--train-partition-ref", required=True, type=Path)
    build.add_argument("--base-model-ref", required=True, type=Path)
    build.add_argument("--write-ref", required=True, type=Path)
    build.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    build.add_argument("--target-root", type=Path, default=None)
    validate = subparsers.add_parser(
        "validate-evidence", help="read-only validation of a training evidence target"
    )
    validate.add_argument("--training-evidence-ref", required=True, type=Path)
    validate.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        else:
            result = validate_training_evidence(
                args.training_evidence_ref,
                workspace_root=args.workspace_root,
            )
    except (TrainingArtifactError, OSError, ValueError) as exc:
        print(f"[stage1-training-evidence] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
