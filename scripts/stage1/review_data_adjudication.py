#!/usr/bin/env python3
"""Safely inspect, complete, or validate Stage 1 human data adjudication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from review.human_adjudication import (  # noqa: E402
    FROZEN_REVIEWER_ID,
    HumanAdjudicationError,
    HumanAdjudicationIncomplete,
    load_human_review_workspace,
    review_interactively,
    validate_human_review_workspace,
    workspace_status,
)


def _add_workspace_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--audit-ref", required=True, type=Path)
    command.add_argument("--packet-file", required=True, type=Path)
    command.add_argument("--adjudication-file", required=True, type=Path)
    review_source = command.add_mutually_exclusive_group()
    review_source.add_argument(
        "--review-ref",
        type=Path,
        help="Validated data-blind-review locator required for dual-model queues.",
    )
    review_source.add_argument(
        "--review-target",
        type=Path,
        help="Legacy/raw review directory; retained only for compatibility tests.",
    )
    command.add_argument("--workspace-root", default=REPOSITORY_ROOT, type=Path)
    command.add_argument("--reviewer-id", default=FROZEN_REVIEWER_ID)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("status", "Validate the frame and report resumable completion counts."),
        ("review", "Open the model-output-free interactive terminal reviewer."),
        ("validate", "Require every row to be complete and rubric-valid."),
    ):
        command = commands.add_parser(name, help=help_text)
        _add_workspace_arguments(command)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        workspace = load_human_review_workspace(
            audit_ref=args.audit_ref,
            packet_file=args.packet_file,
            adjudication_file=args.adjudication_file,
            review_target=args.review_target,
            review_ref=args.review_ref,
            workspace_root=args.workspace_root,
            reviewer_id=args.reviewer_id,
        )
        if args.command == "status":
            result = workspace_status(workspace)
        elif args.command == "validate":
            result = validate_human_review_workspace(workspace)
        else:
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                raise HumanAdjudicationError(
                    "interactive review requires a real terminal; piped decisions are forbidden"
                )
            result = review_interactively(workspace)
    except HumanAdjudicationIncomplete as exc:
        print(f"incomplete: {exc}", file=sys.stderr)
        return 3
    except (HumanAdjudicationError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
