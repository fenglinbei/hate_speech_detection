#!/usr/bin/env python3
"""Run or merge Stage 1 independent dual-model blind review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from review.blind_review import (
    BlindReviewError,
    export_human_adjudication_workspace,
    merge_human_adjudication,
    preflight_data_review,
    retry_failed_data_review,
    run_data_review,
)
from review.cf_blind_review import (
    export_cf_human_review,
    merge_cf_human_review,
    preflight_cf_blind_review,
    run_cf_blind_review,
)
from review.d14_contract import D14_LIVE_EXECUTION_MODE


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("data", help="Run two independent reviewers on a data audit.")
    run.add_argument("--audit-ref", required=True, type=Path)
    run.add_argument("--policy", required=True, type=Path)
    run.add_argument("--env-file", default=REPOSITORY_ROOT / ".env", type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument(
        "--issue-kind",
        action="append",
        dest="issue_kinds",
        help="Explicit audit issue kind to send; repeatable. Omit for the complete audit.",
    )
    preflight = commands.add_parser("preflight", help="Review one item per provider without writing artifacts.")
    preflight.add_argument("--audit-ref", required=True, type=Path)
    preflight.add_argument("--policy", required=True, type=Path)
    preflight.add_argument("--env-file", default=REPOSITORY_ROOT / ".env", type=Path)
    preflight.add_argument(
        "--issue-kind",
        action="append",
        dest="issue_kinds",
        help="Explicit audit issue kind eligible for the preflight; repeatable.",
    )
    preflight.add_argument(
        "--reviewer-id",
        action="append",
        dest="reviewer_ids",
        help="Reviewer eligible for this preflight; repeatable.",
    )
    merge = commands.add_parser("merge-human", help="Merge the frozen human queue with auto consensus.")
    merge.add_argument("--audit-ref", required=True, type=Path)
    merge.add_argument("--review-ref", required=True, type=Path)
    merge.add_argument("--human-completed", required=True, type=Path)
    merge.add_argument("--output", required=True, type=Path)
    merge.add_argument("--workspace-root", default=REPOSITORY_ROOT, type=Path)
    export_human = commands.add_parser(
        "export-human", help="Export the frozen human queue to an editable blind workspace."
    )
    export_human.add_argument("--review-ref", required=True, type=Path)
    export_human.add_argument("--output", required=True, type=Path)
    export_human.add_argument("--packet-output", required=True, type=Path)
    export_human.add_argument("--workspace-root", default=REPOSITORY_ROOT, type=Path)
    retry = commands.add_parser(
        "retry-failed-data",
        help="Retry only failed reviewer/item calls from an immutable parent run.",
    )
    retry.add_argument("--audit-ref", required=True, type=Path)
    retry.add_argument("--policy", required=True, type=Path)
    retry.add_argument("--env-file", default=REPOSITORY_ROOT / ".env", type=Path)
    retry.add_argument("--parent-review-target", required=True, type=Path)
    retry.add_argument("--output-dir", required=True, type=Path)
    cf = commands.add_parser(
        "cf", help="Run the live frozen GLM/DeepSeek review over a CF proposal."
    )
    cf.add_argument("--proposal-ref", required=True, type=Path)
    cf.add_argument("--policy", required=True, type=Path)
    cf.add_argument("--env-file", default=REPOSITORY_ROOT / ".env", type=Path)
    cf.add_argument("--output-dir", required=True, type=Path)
    cf.add_argument("--write-ref", required=True, type=Path)
    cf.add_argument("--workspace-root", default=REPOSITORY_ROOT, type=Path)
    cf_preflight = commands.add_parser(
        "preflight-cf",
        help="Call each live frozen CF reviewer once without writing an artifact.",
    )
    cf_preflight.add_argument("--proposal-ref", required=True, type=Path)
    cf_preflight.add_argument("--policy", required=True, type=Path)
    cf_preflight.add_argument("--env-file", default=REPOSITORY_ROOT / ".env", type=Path)
    cf_preflight.add_argument("--workspace-root", default=REPOSITORY_ROOT, type=Path)
    cf_export = commands.add_parser(
        "export-cf-human",
        help="Export a CF human queue without exposing model votes.",
    )
    cf_export.add_argument("--review-ref", required=True, type=Path)
    cf_export.add_argument("--output", required=True, type=Path)
    cf_export.add_argument("--packet-output", required=True, type=Path)
    cf_export.add_argument("--workspace-root", default=REPOSITORY_ROOT, type=Path)
    cf_merge = commands.add_parser(
        "merge-cf-human",
        help="Merge completed CF human rows with the frozen automatic rows.",
    )
    cf_merge.add_argument("--review-ref", required=True, type=Path)
    cf_merge.add_argument("--human-completed", required=True, type=Path)
    cf_merge.add_argument("--output", required=True, type=Path)
    cf_merge.add_argument("--declaration-output", type=Path)
    cf_merge.add_argument("--workspace-root", default=REPOSITORY_ROOT, type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "data":
            result = run_data_review(
                audit_ref=args.audit_ref,
                policy_path=args.policy,
                env_file=args.env_file,
                output_dir=args.output_dir,
                issue_kinds=args.issue_kinds,
            )
        elif args.command == "preflight":
            result = preflight_data_review(
                audit_ref=args.audit_ref,
                policy_path=args.policy,
                env_file=args.env_file,
                issue_kinds=args.issue_kinds,
                reviewer_ids=args.reviewer_ids,
            )
        elif args.command == "retry-failed-data":
            result = retry_failed_data_review(
                audit_ref=args.audit_ref,
                policy_path=args.policy,
                env_file=args.env_file,
                parent_review_target=args.parent_review_target,
                output_dir=args.output_dir,
            )
        elif args.command == "cf":
            result = run_cf_blind_review(
                proposal_ref=args.proposal_ref,
                policy_path=args.policy,
                env_file=args.env_file,
                output_dir=args.output_dir,
                write_ref=args.write_ref,
                workspace_root=args.workspace_root,
                execution_mode=D14_LIVE_EXECUTION_MODE,
            )
        elif args.command == "preflight-cf":
            result = preflight_cf_blind_review(
                proposal_ref=args.proposal_ref,
                policy_path=args.policy,
                env_file=args.env_file,
                workspace_root=args.workspace_root,
                execution_mode=D14_LIVE_EXECUTION_MODE,
            )
        elif args.command == "export-cf-human":
            result = export_cf_human_review(
                blind_review_ref=args.review_ref,
                output=args.output,
                packet_output=args.packet_output,
                workspace_root=args.workspace_root,
            )
        elif args.command == "merge-cf-human":
            result = merge_cf_human_review(
                blind_review_ref=args.review_ref,
                human_completed=args.human_completed,
                output=args.output,
                declaration_output=args.declaration_output,
                workspace_root=args.workspace_root,
            )
        elif args.command == "merge-human":
            result = merge_human_adjudication(
                audit_ref=args.audit_ref,
                review_ref=args.review_ref,
                workspace_root=args.workspace_root,
                human_completed=args.human_completed,
                output=args.output,
            )
        else:
            result = export_human_adjudication_workspace(
                review_ref=args.review_ref,
                workspace_root=args.workspace_root,
                output=args.output,
                packet_output=args.packet_output,
            )
    except (BlindReviewError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
