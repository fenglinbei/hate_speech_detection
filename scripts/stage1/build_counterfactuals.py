#!/usr/bin/env python3
"""Propose, finalize, and validate immutable Stage-1 counterfactuals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.counterfactual_lifecycle import (  # noqa: E402
    CounterfactualLifecycleError,
    finalize_counterfactual_artifact,
    propose_counterfactual_artifact,
    validate_cf_ref,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the two-stage, model-blind Stage-1 CF lifecycle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    propose = subparsers.add_parser("propose-cf")
    propose.add_argument("--config", type=Path)
    propose.add_argument("--foil-policy", type=Path)
    propose.add_argument("--review-rubric", type=Path)
    propose.add_argument("--context-ref", type=Path)
    propose.add_argument("--test-context-ref", type=Path)
    propose.add_argument("--frozen-cf-ref", type=Path)
    propose.add_argument("--write-ref", required=True, type=Path)
    propose.add_argument("--split", choices=("dev", "test"), default="dev")
    propose.add_argument("--target-root", type=Path)
    propose.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    fixture = propose.add_argument_group(
        "engineering-only fixture (mutually exclusive with --context-ref)"
    )
    fixture.add_argument("--engineering-context-records", type=Path)
    fixture.add_argument("--engineering-train-records", type=Path)

    finalize = subparsers.add_parser("finalize-cf")
    finalize.add_argument("--proposal-ref", required=True, type=Path)
    finalize.add_argument(
        "--blind-review-ref",
        required=True,
        type=Path,
        help=(
            "Validated cf-blind-review locator whose proposal and frozen "
            "automatic/human partitions must exactly match the completed review."
        ),
    )
    finalize.add_argument("--review-file", required=True, type=Path)
    finalize.add_argument("--reviewer-declaration", required=True, type=Path)
    finalize.add_argument("--write-review-ref", required=True, type=Path)
    finalize.add_argument("--write-ref", required=True, type=Path)
    finalize.add_argument("--target-root", type=Path)
    finalize.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    finalize.add_argument(
        "--sealed",
        action="store_true",
        help="Required assertion for test finalize; forbidden for dev.",
    )

    validate = subparsers.add_parser("validate-cf")
    validate.add_argument("--cf-ref", required=True, type=Path)
    validate.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def _load_records(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise CounterfactualLifecycleError(
                        f"blank engineering JSONL row at {path}:{line_number}"
                    )
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise CounterfactualLifecycleError(
                        f"engineering JSONL row is not an object at {path}:{line_number}"
                    )
                rows.append(row)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise CounterfactualLifecycleError(
            f"engineering fixture must be a JSON array of objects: {path}"
        )
    return value


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "propose-cf":
            context_ref = args.test_context_ref or args.context_ref
            if args.test_context_ref is not None and args.context_ref is not None:
                raise CounterfactualLifecycleError(
                    "provide only one of --context-ref/--test-context-ref"
                )
            if args.split == "dev":
                if args.config is None or args.review_rubric is None:
                    raise CounterfactualLifecycleError(
                        "dev propose-cf requires --config and --review-rubric"
                    )
                foil_policy = args.foil_policy or (
                    REPOSITORY_ROOT / "config/stage1/cf_foil_policy.json"
                )
            else:
                foil_policy = args.foil_policy
            result = propose_counterfactual_artifact(
                config=args.config,
                foil_policy=foil_policy,
                review_rubric=args.review_rubric,
                context_ref=context_ref,
                frozen_cf_ref=args.frozen_cf_ref,
                engineering_context_records=_load_records(
                    args.engineering_context_records
                ),
                engineering_train_records=_load_records(
                    args.engineering_train_records
                ),
                write_ref=args.write_ref,
                split=args.split,
                target_root=args.target_root,
                workspace_root=args.workspace_root,
            )
        elif args.command == "finalize-cf":
            result = finalize_counterfactual_artifact(
                proposal_ref=args.proposal_ref,
                blind_review_ref=args.blind_review_ref,
                review_file=args.review_file,
                reviewer_declaration=args.reviewer_declaration,
                write_review_ref=args.write_review_ref,
                write_ref=args.write_ref,
                target_root=args.target_root,
                workspace_root=args.workspace_root,
                sealed=args.sealed,
            )
        else:
            result = validate_cf_ref(
                args.cf_ref, workspace_root=args.workspace_root
            )
    except (CounterfactualLifecycleError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
