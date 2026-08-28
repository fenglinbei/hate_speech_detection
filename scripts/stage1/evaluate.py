#!/usr/bin/env python3
"""Stage-1 strict metric and paired-statistics CLI.

The ``free-fixture`` command is deliberately engineering-only: formal analyses
must consume an immutable generation-run ref once that lifecycle is finalized.
It nevertheless executes the exact same parser, assignments, sufficient stats,
and reducers used by the formal path.
"""

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

from metrics.stage1_metrics import (  # noqa: E402
    aggregate_query_metrics,
    evaluate_query,
    flip_table,
)
from metrics.stage1_statistics import (  # noqa: E402
    factorial_effects,
    fixed_seed_paired_bootstrap,
)
from metrics.stage1_artifacts import (  # noqa: E402
    Stage1ArtifactError,
    build_factorial_analysis_artifact,
    build_free_evaluation_artifact,
    validate_analysis_ref,
    validate_evaluation_ref_report,
    validate_margin_ref,
)
from metrics.stage1_margin_lifecycle import build_margin_artifact  # noqa: E402


class EvaluationCliError(ValueError):
    pass


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationCliError(f"cannot read JSON {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise EvaluationCliError(f"blank JSONL line at {path}:{line_number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise EvaluationCliError(f"row must be an object at {path}:{line_number}")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationCliError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path.write_text(rendered + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _free_fixture(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.input_jsonl)
    if not rows:
        raise EvaluationCliError("free fixture input cannot be empty")
    seen: set[tuple[str, str]] = set()
    evaluated: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        required = {"id", "condition", "raw_output", "gold"}
        missing = sorted(required - set(row))
        if missing:
            raise EvaluationCliError(f"input ordinal {ordinal} missing keys: {missing}")
        key = (str(row["condition"]), str(row["id"]))
        if key in seen:
            raise EvaluationCliError(f"duplicate condition/id: {key}")
        seen.add(key)
        evaluated.append(
            evaluate_query(
                query_id=str(row["id"]),
                condition=str(row["condition"]),
                raw_output=str(row["raw_output"]),
                gold=row["gold"],
                runner_status=str(row.get("runner_status", "ok")),
                content_sha256=row.get("content_sha256"),
                gold_sha256=row.get("gold_sha256"),
                prompt_sha256=row.get("prompt_sha256"),
                context_record_sha256=row.get("context_record_sha256"),
                soft_threshold=args.soft_threshold,
            )
        )
    conditions = sorted({str(row["condition"]) for row in evaluated})
    summaries = {
        condition: aggregate_query_metrics(
            [row for row in evaluated if row["condition"] == condition]
        )
        for condition in conditions
    }
    _write_jsonl(args.output_dir / "per_query_metrics.jsonl", evaluated)
    _write_json(args.output_dir / "summary.json", summaries)
    meta = {
        "schema_version": "stage1-engineering-evaluation/v1",
        "scientific_eligible": False,
        "reason": "direct-fixture-input-not-generation-run-ref",
        "conditions": conditions,
        "query_condition_row_count": len(evaluated),
        "soft_threshold": args.soft_threshold,
    }
    _write_json(args.output_dir / "evaluation.meta.json", meta)
    return meta


def _flip(args: argparse.Namespace) -> dict[str, Any]:
    left = _read_jsonl(args.a_jsonl)
    right = _read_jsonl(args.b_jsonl)
    result = flip_table(left, right, args.endpoint)
    _write_json(args.output, result)
    return result


def _factorial_fixture(args: argparse.Namespace) -> dict[str, Any]:
    raw = _read_json(args.cells_json)
    if not isinstance(raw, dict) or not raw:
        raise EvaluationCliError("cells JSON must map seed to condition arrays")
    seed_cells: dict[int, dict[str, list[float]]] = {}
    for seed_text, cells in raw.items():
        if not isinstance(cells, dict):
            raise EvaluationCliError(f"seed {seed_text} cells must be an object")
        seed_cells[int(seed_text)] = {
            str(condition): [float(value) for value in values]
            for condition, values in cells.items()
        }
    result = fixed_seed_paired_bootstrap(
        seed_cells=seed_cells,
        metric=lambda values: sum(float(value) for value in values) / len(values),
        effect=factorial_effects,
        conditions=("C0", "CL", "CD", "CLD"),
        family=args.family,
        bootstrap_seed=args.seed,
        replicates=args.replicates,
    )
    result["scientific_eligible"] = False
    result["reason"] = "numeric-fixture-not-frozen-analysis-refs"
    _write_json(args.output, result)
    return result


def _free(args: argparse.Namespace) -> dict[str, Any]:
    return build_free_evaluation_artifact(
        generation_run_ref=args.generation_run_ref,
        evaluation_profile=args.evaluation_profile,
        write_ref=args.write_ref,
        split=args.split,
        sealed=args.sealed,
        target_root=args.target_root,
        workspace_root=args.workspace_root,
    )


def _factorial(args: argparse.Namespace) -> dict[str, Any]:
    return build_factorial_analysis_artifact(
        model_registry_ref=args.model_registry_ref,
        run_ref_map=args.run_ref_map,
        analysis_profile=args.analysis_profile,
        decision_register=args.decision_register,
        write_ref=args.write_ref,
        split=args.split,
        mode=args.mode,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        sealed=args.sealed,
        target_root=args.target_root,
        workspace_root=args.workspace_root,
    )


def _margin(args: argparse.Namespace) -> dict[str, Any]:
    conditions = [value.strip() for value in args.conditions.split(",")]
    if any(not value for value in conditions):
        raise EvaluationCliError("--conditions must be a non-empty comma-separated list")
    return build_margin_artifact(
        model_registry_ref=args.model_registry_ref,
        model_key=args.model_key,
        context_ref=args.context_ref,
        control_ref=args.control_ref,
        cf_ref=args.cf_ref,
        scorer_profile=args.scorer_profile,
        conditions=conditions,
        split=args.split,
        sealed=args.sealed,
        write_ref=args.write_ref,
        target_root=args.target_root,
        workspace_root=args.workspace_root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1 strict evaluation utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    free_artifact = sub.add_parser(
        "free", help="Build an immutable evaluation from one generation-run ref"
    )
    free_artifact.add_argument("--generation-run-ref", required=True, type=Path)
    free_artifact.add_argument("--evaluation-profile", required=True, type=Path)
    free_artifact.add_argument("--write-ref", required=True, type=Path)
    free_artifact.add_argument("--split", required=True, choices=("dev", "test"))
    free_artifact.add_argument("--sealed", action="store_true", default=None)
    free_artifact.add_argument("--target-root", type=Path)
    free_artifact.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)

    margin_artifact = sub.add_parser(
        "margin", help="Build an immutable local-HF teacher-forced margin run"
    )
    margin_artifact.add_argument("--model-registry-ref", required=True, type=Path)
    margin_artifact.add_argument("--model-key", required=True)
    margin_artifact.add_argument("--context-ref", required=True, type=Path)
    margin_artifact.add_argument(
        "--control-ref",
        required=True,
        type=Path,
        help="Use control for dev or test-control for sealed test; kinds cannot be mixed.",
    )
    margin_artifact.add_argument("--cf-ref", required=True, type=Path)
    margin_artifact.add_argument("--scorer-profile", required=True, type=Path)
    margin_artifact.add_argument(
        "--conditions", default="C0,CL,CD,CLD,PL,PD",
        help="Assertion against the frozen scorer profile; never an override",
    )
    margin_artifact.add_argument("--write-ref", required=True, type=Path)
    margin_artifact.add_argument("--split", required=True, choices=("dev", "test"))
    margin_artifact.add_argument("--sealed", action="store_true", default=None)
    margin_artifact.add_argument("--target-root", type=Path)
    margin_artifact.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)

    factorial_artifact = sub.add_parser(
        "factorial", help="Build immutable paired factorial/bootstrap analysis artifacts"
    )
    factorial_artifact.add_argument("--model-registry-ref", required=True, type=Path)
    factorial_artifact.add_argument("--run-ref-map", required=True, type=Path)
    factorial_artifact.add_argument("--analysis-profile", required=True, type=Path)
    factorial_artifact.add_argument(
        "--decision-register",
        type=Path,
        default=REPOSITORY_ROOT / "config/stage1/decision_register.json",
    )
    factorial_artifact.add_argument("--write-ref", required=True, type=Path)
    factorial_artifact.add_argument("--split", required=True, choices=("dev", "test"))
    factorial_artifact.add_argument(
        "--mode", choices=("engineering-smoke", "pilot", "confirmatory")
    )
    factorial_artifact.add_argument("--n-bootstrap", type=int)
    factorial_artifact.add_argument("--seed", type=int)
    factorial_artifact.add_argument("--sealed", action="store_true", default=None)
    factorial_artifact.add_argument("--target-root", type=Path)
    factorial_artifact.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)

    validate_free = sub.add_parser("validate-free", help="Read-only evaluation replay validator")
    validate_free.add_argument("--evaluation-ref", required=True, type=Path)
    validate_free.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)

    validate_factorial = sub.add_parser(
        "validate-factorial", help="Read-only factorial-analysis replay validator"
    )
    validate_factorial.add_argument("--analysis-ref", required=True, type=Path)
    validate_factorial.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)

    validate_margin = sub.add_parser(
        "validate-margin", help="Read-only expected-margin target replay validator"
    )
    validate_margin.add_argument("--margin-ref", required=True, type=Path)
    validate_margin.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)

    free = sub.add_parser("free-fixture", help="Run strict evaluator on engineering JSONL")
    free.add_argument("--input-jsonl", required=True, type=Path)
    free.add_argument("--output-dir", required=True, type=Path)
    free.add_argument("--soft-threshold", type=float, default=0.5)

    flip = sub.add_parser("flip", help="Compute an endpoint-specific paired flip table")
    flip.add_argument("--a-jsonl", required=True, type=Path)
    flip.add_argument("--b-jsonl", required=True, type=Path)
    flip.add_argument("--endpoint", required=True)
    flip.add_argument("--output", required=True, type=Path)

    factorial = sub.add_parser("factorial-fixture", help="Bootstrap numeric factorial fixtures")
    factorial.add_argument("--cells-json", required=True, type=Path)
    factorial.add_argument("--family", default="engineering-smoke")
    factorial.add_argument("--seed", type=int, default=42)
    factorial.add_argument("--replicates", type=int, default=200)
    factorial.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "free":
            result = _free(args)
        elif args.command == "margin":
            result = _margin(args)
        elif args.command == "factorial":
            result = _factorial(args)
        elif args.command == "validate-free":
            result = validate_evaluation_ref_report(
                args.evaluation_ref, workspace_root=args.workspace_root
            )
        elif args.command == "validate-factorial":
            result = validate_analysis_ref(
                args.analysis_ref, workspace_root=args.workspace_root
            )
        elif args.command == "validate-margin":
            _, result, _ = validate_margin_ref(
                args.margin_ref, workspace_root=args.workspace_root
            )
        elif args.command == "free-fixture":
            result = _free_fixture(args)
        elif args.command == "flip":
            result = _flip(args)
        elif args.command == "factorial-fixture":
            result = _factorial_fixture(args)
        else:  # pragma: no cover
            raise EvaluationCliError(f"unsupported command: {args.command}")
    except (EvaluationCliError, Stage1ArtifactError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
