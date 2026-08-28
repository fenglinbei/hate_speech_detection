#!/usr/bin/env python3
"""Build or validate the immutable Stage 1 fit/calibration partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.train_partition import (
    DEFAULT_CONFIG,
    TrainPartitionError,
    build_train_partition,
    validate_train_partition,
)
from data.training_artifacts import TrainingArtifactError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build", help="freeze the complete content-clustered train partition"
    )
    build.add_argument("--data-ref", required=True, type=Path)
    build.add_argument("--write-ref", required=True, type=Path)
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    build.add_argument("--target-root", type=Path, default=None)

    validate = subparsers.add_parser(
        "validate", help="deep-validate and replay a frozen train partition"
    )
    validate.add_argument("--train-partition-ref", required=True, type=Path)
    validate.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_train_partition(
                data_ref=args.data_ref,
                write_ref=args.write_ref,
                config_path=args.config,
                workspace_root=args.workspace_root,
                target_root=args.target_root,
            )
        else:
            result = validate_train_partition(
                args.train_partition_ref,
                workspace_root=args.workspace_root,
            )
    except (TrainPartitionError, TrainingArtifactError, OSError, ValueError) as exc:
        print(f"[stage1-train-partition] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
