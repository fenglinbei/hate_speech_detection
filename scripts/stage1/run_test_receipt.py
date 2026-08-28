#!/usr/bin/env python3
"""Run, publish, and validate the immutable Stage 1 P0 test receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for _path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from data.test_receipt import (  # noqa: E402
    TestReceiptError,
    build_test_receipt,
    check_json_documents,
    check_json_schemas,
    check_python_sources,
    validate_test_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute the frozen manifest and publish on all-pass.")
    run_parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    run_parser.add_argument("--target-root", type=Path)
    run_parser.add_argument("--ref-path", type=Path)
    run_parser.add_argument("--jobs", type=int, default=1)

    validate_parser = subparsers.add_parser("validate", help="Deep-validate one published receipt.")
    validate_parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    validate_parser.add_argument("--ref", type=Path, required=True)

    for name, help_text in (
        ("check-python", "Compile every frozen Python source with py_compile."),
        ("check-json", "Parse every frozen Stage 1 config/schema JSON file."),
        ("check-schemas", "Compile every repository JSON Schema."),
    ):
        check_parser = subparsers.add_parser(name, help=help_text)
        check_parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def _emit(value: object) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "run":
            locator, receipt = build_test_receipt(
                workspace_root=args.workspace_root,
                target_root=args.target_root,
                ref_path=args.ref_path,
                jobs=args.jobs,
            )
            _emit(
                {
                    "artifact_kind": locator["artifact_kind"],
                    "test_receipt_id": receipt["test_receipt_id"],
                    "all_passed": receipt["all_passed"],
                    "command_count": receipt["command_count"],
                    "payload_manifest_sha256": locator["payload_manifest_sha256"],
                }
            )
        elif args.command == "validate":
            _emit(
                validate_test_receipt(
                    args.ref,
                    workspace_root=args.workspace_root,
                )
            )
        elif args.command == "check-python":
            _emit(check_python_sources(args.workspace_root))
        elif args.command == "check-json":
            _emit(check_json_documents(args.workspace_root))
        elif args.command == "check-schemas":
            _emit(check_json_schemas(args.workspace_root))
        else:  # pragma: no cover - argparse enforces the command set.
            raise TestReceiptError("unsupported command")
    except (TestReceiptError, OSError, ValueError) as exc:
        print(f"stage1-test-receipt error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
