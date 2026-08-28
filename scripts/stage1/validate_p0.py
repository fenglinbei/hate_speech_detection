#!/usr/bin/env python3
"""Emit the deterministic read-only Stage 1 P0 blocking report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.stage1_p0_validation import (  # noqa: E402
    MODES,
    Stage1P0ValidationError,
    validate_stage1_p0,
    write_report_sidecar,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--write-report",
        type=Path,
        help="Optional mutable sidecar path; immutable targets and refs are rejected.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = validate_stage1_p0(
            workspace_root=args.workspace_root,
            mode=args.mode,
        )
        if args.write_report is not None:
            write_report_sidecar(
                report,
                args.write_report,
                workspace_root=args.workspace_root,
            )
    except Stage1P0ValidationError as exc:
        print(f"stage1-p0 validation error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
