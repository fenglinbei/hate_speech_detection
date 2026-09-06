#!/usr/bin/env python3
"""Validate and summarize the WP3 S2.1b current-run provider plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_lex.terminology_candidate_run_plan import (  # noqa: E402
    CandidateRunPlanError,
    build_logical_slot_grid,
    load_bound_public_tasks,
    load_current_run_plan,
)
from data.training_artifacts import TrainingArtifactError  # noqa: E402


DEFAULT_PLAN = Path("config/stage1/wp3_candidate_current_run_plan_v2.json")
DEFAULT_SCHEMA = Path("schemas/wp3_candidate_current_run_plan_v1.schema.json")
DEFAULT_GENERATOR_ROOT = Path(
    "exps/causal_context/stage1_p0/wp3_candidate_generators_v2"
)
DEFAULT_FRAME_REF = DEFAULT_GENERATOR_ROOT / "refs/development_frame_ref.json"


def _in_workspace(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--frame-ref", type=Path, default=DEFAULT_FRAME_REF)
    parser.add_argument(
        "--no-current-source-check",
        action="store_true",
        help="Skip current file/ref checks; schema and contract checks still run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.workspace_root.resolve()
    try:
        plan = load_current_run_plan(
            _in_workspace(root, args.plan),
            workspace_root=root,
            schema_path=_in_workspace(root, args.schema),
            require_current_sources=not args.no_current_source_check,
        )
        tasks = load_bound_public_tasks(
            plan, _in_workspace(root, args.frame_ref)
        )
        slots = build_logical_slot_grid(plan, tasks)
    except (CandidateRunPlanError, TrainingArtifactError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = {
        "plan_id": plan["plan_id"],
        "plan_revision": plan["plan_revision"],
        "supersedes": plan["supersedes"],
        "models": {
            provider_id: provider["requested_model"]
            for provider_id, provider in plan["providers"].items()
        },
        "request_profile": plan["request_profile"],
        "g1": plan["g1"],
        "g2": plan["g2"],
        "g3": plan["g3"],
        "total_logical_model_slot_count": plan[
            "total_logical_model_slot_count"
        ],
        "bound_public_task_count": len(tasks),
        "bound_logical_slot_count": len(slots),
        "execution": plan["execution"],
        "preserved_raw_gold": plan["preserved_raw_gold"],
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
