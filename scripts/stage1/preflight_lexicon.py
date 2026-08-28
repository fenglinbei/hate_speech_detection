#!/usr/bin/env python3
"""Run the read-only Stage-1 WP3 formal lexicon preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.stage1_preflight import preflight_formal_lexicon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only fail-closed preflight for a formal Stage-1 WP3 lexicon build. "
            "It performs no HTTP requests and writes no files."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-ref", required=True, type=Path)
    parser.add_argument("--train-partition-ref", required=True, type=Path)
    parser.add_argument("--env-file", default=REPOSITORY_ROOT / ".env", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = preflight_formal_lexicon(
        config_path=args.config,
        data_ref=args.data_ref,
        train_partition_ref=args.train_partition_ref,
        env_file=args.env_file,
        repository_root=REPOSITORY_ROOT,
    )
    print(json.dumps(result.to_public_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
