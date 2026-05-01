#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare before/after runner metrics on fixed lexicon-hit buckets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
BOOTSTRAP_DIR = REPO_ROOT / "scripts" / "paired_bootstrap"
for path in (SRC_DIR, BOOTSTRAP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paired_bootstrap_llm import (  # noqa: E402
    InstanceCounts,
    SystemData,
    compute_instance_counts,
    load_system,
    observed_metrics,
    sum_counts,
)


BUCKET_ORDER = ["Exact-hit", "Semantic-only-hit", "Both-hit", "No-hit"]
METRIC_COLUMNS = [
    ("f1_target", "Tar-F1"),
    ("f1_hate", "Hate-F1"),
    ("f1_hard", "Hard-F1"),
    ("f1_soft", "Soft-F1"),
    ("f1_avg", "Avg-F1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two runner outputs on fixed lexicon-hit bucket IDs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bucket-json",
        type=Path,
        default=Path("output/analyse/lexicon_bucket/ours_prompt_al1280_bucket_metrics.json"),
        help="Audit JSON produced by lexicon_hit_bucket_report.py.",
    )
    parser.add_argument(
        "--before-runner-output",
        type=Path,
        default=Path("output/runner/ablation/wo_lex.json"),
        help="Runner output before introducing lexicon.",
    )
    parser.add_argument(
        "--after-runner-output",
        type=Path,
        default=Path("output/runner/method_comparison/ours_prompt_al1280.json"),
        help="Runner output after introducing lexicon.",
    )
    parser.add_argument("--before-name", default="w/o lexicon")
    parser.add_argument("--after-name", default="Ours")
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("output/analyse/lexicon_bucket/wo_lex_vs_ours_prompt_al1280_bucket_metrics.md"),
        help="Markdown comparison report output path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Audit JSON output path. Defaults to --out-md with .json suffix.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Soft-match threshold for metric recomputation.",
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for validating recomputed all-sample metrics.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sort_id(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def load_bucket_ids(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    payload = read_json(path)
    raw_bucket_ids = payload.get("bucket_ids")
    if not isinstance(raw_bucket_ids, dict):
        raise ValueError(f"{path} must contain a top-level 'bucket_ids' object.")

    buckets: dict[str, list[str]] = {}
    for bucket in BUCKET_ORDER:
        ids = raw_bucket_ids.get(bucket)
        if not isinstance(ids, list):
            raise ValueError(f"{path} bucket_ids.{bucket} must be a list.")
        buckets[bucket] = [str(instance_id) for instance_id in ids]

    all_ids = [instance_id for bucket in BUCKET_ORDER for instance_id in buckets[bucket]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"{path} assigns at least one id to multiple buckets.")

    metadata = {
        "selected_tau": payload.get("selected_tau"),
        "semantic_top_k": payload.get("semantic_top_k"),
        "threshold_grid": payload.get("threshold_grid"),
        "source_bucket_json": rel_path(path),
        "bucket_sizes": {bucket: len(ids) for bucket, ids in buckets.items()},
    }
    return buckets, metadata


def validate_system_ids(system: SystemData, bucket_ids: dict[str, list[str]], label: str) -> dict[str, Any]:
    bucket_id_set = {instance_id for ids in bucket_ids.values() for instance_id in ids}
    system_id_set = set(system.ids)
    missing_in_system = sorted(bucket_id_set - system_id_set, key=sort_id)
    extra_in_system = sorted(system_id_set - bucket_id_set, key=sort_id)
    if missing_in_system or extra_in_system:
        raise ValueError(
            f"{label} ids do not match bucket ids: "
            f"missing_in_system={missing_in_system[:10]}, "
            f"extra_in_system={extra_in_system[:10]}"
        )
    return {
        "label": label,
        "runner_output": rel_path(system.path),
        "total": len(system.ids),
        "ids_match_buckets": True,
    }


def counts_to_dict(counts: Any) -> dict[str, int]:
    return {"tp": int(counts.tp), "fp": int(counts.fp), "fn": int(counts.fn)}


def metric_counts_for_instances(instances: Sequence[InstanceCounts]) -> dict[str, dict[str, int]]:
    return {
        "hard": counts_to_dict(sum_counts(item.hard for item in instances)),
        "soft": counts_to_dict(sum_counts(item.soft for item in instances)),
        "target": counts_to_dict(sum_counts(item.target for item in instances)),
        "hate": counts_to_dict(sum_counts(item.hate for item in instances)),
    }


def nested_get(payload: dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def validate_metrics_against_payload(
    system: SystemData,
    all_metrics: dict[str, float],
    tolerance: float,
) -> dict[str, Any]:
    payload = read_json(system.path)
    metric_payload = payload.get("metric", {})
    metric_paths = {
        "f1_hard": ("f1_hard",),
        "f1_soft": ("f1_soft",),
        "f1_avg": ("f1_avg",),
        "f1_target": ("field_metrics", "targeted_group", "f1"),
        "f1_hate": ("field_metrics", "hateful", "f1"),
    }
    diffs: dict[str, dict[str, float]] = {}
    for metric_name, path in metric_paths.items():
        expected = nested_get(metric_payload, path)
        if expected is None:
            continue
        observed = all_metrics[metric_name]
        diff = abs(float(expected) - observed)
        diffs[metric_name] = {
            "payload": float(expected),
            "recomputed": observed,
            "abs_diff": diff,
        }
        if diff > tolerance:
            raise ValueError(
                f"Metric validation failed for {system.path} {metric_name}: "
                f"payload={expected}, recomputed={observed:.8f}, diff={diff:.8f}."
            )
    return {"metric_validation_ok": True, "tolerance": tolerance, "diffs": diffs}


def compute_counts_by_id(system: SystemData, similarity_threshold: float) -> dict[str, InstanceCounts]:
    return {
        instance_id: compute_instance_counts(system.results_by_id[instance_id], similarity_threshold)
        for instance_id in system.ids
    }


def compute_bucket_metrics(
    buckets: dict[str, list[str]],
    counts_by_id: dict[str, InstanceCounts],
) -> dict[str, dict[str, Any]]:
    metrics_by_bucket: dict[str, dict[str, Any]] = {}
    for bucket in BUCKET_ORDER:
        instances = [counts_by_id[instance_id] for instance_id in buckets[bucket]]
        metrics = observed_metrics(instances)
        metrics_by_bucket[bucket] = {
            **{key: float(value) for key, value in metrics.items()},
            "counts": metric_counts_for_instances(instances),
        }
    return metrics_by_bucket


def build_delta_metrics(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    deltas: dict[str, dict[str, float]] = {}
    for bucket in BUCKET_ORDER:
        deltas[bucket] = {
            metric_key: after[bucket][metric_key] - before[bucket][metric_key]
            for metric_key, _label in METRIC_COLUMNS
        }
    return deltas


def format_float(value: float) -> str:
    return f"{value:.4f}"


def format_delta(value: float) -> str:
    return f"{value:+.4f}"


def render_markdown(
    args: argparse.Namespace,
    out_json: Path,
    bucket_metadata: dict[str, Any],
    buckets: dict[str, list[str]],
    before_metrics: dict[str, dict[str, Any]],
    after_metrics: dict[str, dict[str, Any]],
    delta_metrics: dict[str, dict[str, float]],
    validations: dict[str, Any],
) -> str:
    header = ["Bucket", "N"]
    for _metric_key, metric_label in METRIC_COLUMNS:
        header.extend(
            [
                f"{metric_label} Before",
                f"{metric_label} After",
                f"{metric_label} Delta",
            ]
        )

    align = ["---", "---:"]
    for _ in METRIC_COLUMNS:
        align.extend(["---:", "---:", "---:"])

    lines = [
        "# Lexicon Bucket Before/After Metrics",
        "",
        "## Settings",
        "",
        f"- Bucket source: `{bucket_metadata['source_bucket_json']}`",
        f"- Before: `{rel_path(resolve_path(args.before_runner_output))}` (`{args.before_name}`)",
        f"- After: `{rel_path(resolve_path(args.after_runner_output))}` (`{args.after_name}`)",
        f"- Semantic threshold from bucket source: `{float(bucket_metadata['selected_tau']):.4f}`",
        f"- Semantic top-k from bucket source: `{bucket_metadata['semantic_top_k']}`",
        f"- Audit JSON: `{rel_path(out_json)}`",
        "",
        "## Metrics",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(align) + " |",
    ]

    for bucket in BUCKET_ORDER:
        row = [bucket, str(len(buckets[bucket]))]
        for metric_key, _metric_label in METRIC_COLUMNS:
            row.extend(
                [
                    format_float(before_metrics[bucket][metric_key]),
                    format_float(after_metrics[bucket][metric_key]),
                    format_delta(delta_metrics[bucket][metric_key]),
                ]
            )
        lines.append("| " + " | ".join(row) + " |")

    delta_header = ["Bucket", "N", *[f"{metric_label} Delta" for _metric_key, metric_label in METRIC_COLUMNS]]
    delta_align = ["---", "---:", *["---:" for _ in METRIC_COLUMNS]]
    lines.extend(
        [
            "",
            "## Delta Only",
            "",
            "| " + " | ".join(delta_header) + " |",
            "| " + " | ".join(delta_align) + " |",
        ]
    )
    for bucket in BUCKET_ORDER:
        row = [
            bucket,
            str(len(buckets[bucket])),
            *[
                format_delta(delta_metrics[bucket][metric_key])
                for metric_key, _metric_label in METRIC_COLUMNS
            ],
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Bucket IDs are fixed from `{bucket_metadata['source_bucket_json']}`.",
            f"- Before IDs: `{validations['before_ids']['total']}` samples, matched bucket IDs.",
            f"- After IDs: `{validations['after_ids']['total']}` samples, matched bucket IDs.",
            f"- Metric validation tolerance: `{args.metric_tolerance}`.",
        ]
    )
    return "\n".join(lines)


def build_audit_json(
    args: argparse.Namespace,
    out_md: Path,
    bucket_metadata: dict[str, Any],
    buckets: dict[str, list[str]],
    before_metrics: dict[str, dict[str, Any]],
    after_metrics: dict[str, dict[str, Any]],
    delta_metrics: dict[str, dict[str, float]],
    validations: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bucket_source": bucket_metadata,
        "before": {
            "name": args.before_name,
            "runner_output": rel_path(resolve_path(args.before_runner_output)),
            "metrics": before_metrics,
        },
        "after": {
            "name": args.after_name,
            "runner_output": rel_path(resolve_path(args.after_runner_output)),
            "metrics": after_metrics,
        },
        "delta": delta_metrics,
        "bucket_order": BUCKET_ORDER,
        "bucket_sizes": {bucket: len(buckets[bucket]) for bucket in BUCKET_ORDER},
        "bucket_ids": buckets,
        "validations": validations,
        "markdown_report": rel_path(out_md),
    }


def main() -> int:
    args = parse_args()
    bucket_json = resolve_path(args.bucket_json)
    before_path = resolve_path(args.before_runner_output)
    after_path = resolve_path(args.after_runner_output)
    out_md = resolve_path(args.out_md)
    out_json = resolve_path(args.out_json) if args.out_json else out_md.with_suffix(".json")

    buckets, bucket_metadata = load_bucket_ids(bucket_json)
    before_system = load_system(before_path)
    after_system = load_system(after_path)

    before_id_validation = validate_system_ids(before_system, buckets, args.before_name)
    after_id_validation = validate_system_ids(after_system, buckets, args.after_name)

    before_counts = compute_counts_by_id(before_system, args.similarity_threshold)
    after_counts = compute_counts_by_id(after_system, args.similarity_threshold)

    before_all_metrics = observed_metrics([before_counts[instance_id] for instance_id in before_system.ids])
    after_all_metrics = observed_metrics([after_counts[instance_id] for instance_id in after_system.ids])
    before_metric_validation = validate_metrics_against_payload(
        before_system, before_all_metrics, args.metric_tolerance
    )
    after_metric_validation = validate_metrics_against_payload(
        after_system, after_all_metrics, args.metric_tolerance
    )

    before_metrics = compute_bucket_metrics(buckets, before_counts)
    after_metrics = compute_bucket_metrics(buckets, after_counts)
    delta_metrics = build_delta_metrics(before_metrics, after_metrics)

    validations = {
        "before_ids": before_id_validation,
        "after_ids": after_id_validation,
        "before_metric_validation": before_metric_validation,
        "after_metric_validation": after_metric_validation,
    }
    audit_payload = build_audit_json(
        args=args,
        out_md=out_md,
        bucket_metadata=bucket_metadata,
        buckets=buckets,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        delta_metrics=delta_metrics,
        validations=validations,
    )
    write_json(out_json, audit_payload)
    write_text(
        out_md,
        render_markdown(
            args=args,
            out_json=out_json,
            bucket_metadata=bucket_metadata,
            buckets=buckets,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            delta_metrics=delta_metrics,
            validations=validations,
        ),
    )

    print(f"Wrote Markdown report: {out_md}")
    print(f"Wrote audit JSON: {out_json}")
    print("Bucket sizes:", json.dumps(audit_payload["bucket_sizes"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
