#!/usr/bin/env python3
"""Seal or validate an immutable Stage-1 data dual-blind review artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from review.data_review_artifact import (
    DataReviewArtifactError,
    seal_data_review,
    validate_data_review_ref,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal", help="Validate and seal an existing review target.")
    seal.add_argument("--audit-ref", required=True, type=Path)
    seal.add_argument("--review-target", required=True, type=Path)
    seal.add_argument("--output-root", required=True, type=Path)
    seal.add_argument("--write-ref", required=True, type=Path)
    seal.add_argument("--workspace-root", default=REPOSITORY_ROOT, type=Path)
    validate = commands.add_parser("validate", help="Resolve and replay a sealed review ref.")
    validate.add_argument("--review-ref", required=True, type=Path)
    validate.add_argument("--workspace-root", default=REPOSITORY_ROOT, type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "seal":
            result = seal_data_review(
                audit_ref=args.audit_ref,
                review_target=args.review_target,
                output_root=args.output_root,
                write_ref=args.write_ref,
                workspace_root=args.workspace_root,
            )
        else:
            result = validate_data_review_ref(
                args.review_ref, workspace_root=args.workspace_root
            )
    except (DataReviewArtifactError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
