#!/usr/bin/env python3
"""Read coverage progress receipts and SQLite row counts without reading scores."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RUN = Path(__file__).resolve().parents[1] / "runs" / "coverage-01"
COHORT_BLOCKS = {"regression": 128, "validation": 384, "boundary": 64}
PASS_SUFFIXES = ("r0", "r1", "padding", "prefix", "members", "replica")
DEV_BLOCKS = 10288
DEVICES = (0, 1, 2, 3)
BUSY_TIMEOUT_MS = 250


def read_receipt(path):
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            return None, "malformed: expected JSON object"
        return value, None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as error:
        return None, f"unavailable: {type(error).__name__}: {error}"


def checkpoint_count(path, *, busy_timeout_ms=BUSY_TIMEOUT_MS):
    """COUNT(*) is the only table read; payload and score columns stay unopened."""
    connection = None
    try:
        if not path.is_file():
            return {"count": None, "error": "missing"}
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True,
                                     timeout=busy_timeout_ms / 1000)
        connection.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        connection.execute("PRAGMA query_only=ON")
        count = connection.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        if type(count) is not int or count < 0:
            return {"count": None, "error": "invalid row count"}
        return {"count": count, "error": None}
    except (OSError, sqlite3.Error) as error:
        return {"count": None, "error": f"unavailable: {type(error).__name__}: {error}"}
    finally:
        if connection is not None:
            connection.close()


def pass_status(directory, expected_blocks, plan_id):
    manifest, manifest_error = read_receipt(directory / "manifest.json")
    sealed = None
    if manifest is not None:
        identity = manifest.get("identity")
        if (not isinstance(identity, dict) or not plan_id or identity.get("plan_id") != plan_id
                or identity.get("pass_name") != directory.name):
            manifest_error = "manifest or run plan identity unavailable/mismatched"
        elif manifest.get("status") == "complete" and manifest.get("blocks") == expected_blocks:
            sealed = True
        elif manifest.get("status") == "complete":
            manifest_error = "complete manifest has unexpected block count"
        else:
            manifest_error = "manifest is not complete"
    shards = []
    for device in DEVICES:
        path = directory / "shards" / str(device) / directory.name / "checkpoint.sqlite3"
        observed = checkpoint_count(path)
        if observed["count"] is not None and observed["count"] > expected_blocks // len(DEVICES):
            observed = {"count": None, "error": "checkpoint count exceeds expected shard capacity"}
        shards.append({"device": device, "path": str(path), **observed})
    known = [row["count"] for row in shards if row["count"] is not None]
    committed = sum(known) if len(known) == len(DEVICES) else None
    if sealed is True and committed is not None and committed != expected_blocks:
        manifest_error = "sealed manifest and committed checkpoint counts disagree"
        sealed = None
    return {"pass": directory.name, "expected_blocks": expected_blocks, "committed_blocks": committed,
            "known_committed_subtotal": sum(known) if known else None, "readable_shards": len(known),
            "expected_shards": len(DEVICES), "sealed_manifest": sealed,
            "manifest_error": manifest_error, "shards": shards}


def snapshot(run=DEFAULT_RUN):
    run = Path(run).resolve()
    terminal, terminal_error = read_receipt(run / "run_manifest.json")
    plan_id = None
    state = "unknown"
    if terminal is not None:
        registered_states = {"running", "interrupted", "raw_complete", "complete", "preflight_failed", "failed"}
        if (terminal.get("schema_version") == "general-model-coverage-run/v1"
                and isinstance(terminal.get("plan_id"), str)
                and terminal["plan_id"].startswith("gmlcoverage-")
                and terminal.get("status") in registered_states):
            plan_id, state = terminal["plan_id"], terminal["status"]
        else:
            terminal_error = "malformed coverage run receipt"
    passes = [pass_status(run / "preflight" / f"{cohort}-b1-{suffix}", expected, plan_id)
              for cohort, expected in COHORT_BLOCKS.items() for suffix in PASS_SUFFIXES]
    gate, gate_error = read_receipt(run / "preflight/preflight_report.json")
    gate_state = "unknown"
    if gate is not None:
        if (not plan_id or gate.get("plan_id") != plan_id
                or gate.get("schema_version") != "general-model-coverage-preflight/v1"
                or type(gate.get("passed")) is not bool or type(gate.get("complete")) is not bool):
            gate_error = "malformed or mismatched preflight receipt"
        elif gate["complete"] and gate["passed"]:
            gate_state = "passed"
        elif gate.get("failure"):
            gate_state = "failed"
        else:
            gate_state = "incomplete"
    full_dev = pass_status(run / "dev-b1", DEV_BLOCKS, plan_id)
    analysis_manifest, analysis_error = read_receipt(run / "analysis/manifest.json")
    analysis_state = "unknown"
    if terminal is not None and plan_id:
        if terminal.get("analysis_published") is True:
            if (analysis_manifest is not None and analysis_manifest.get("plan_id") == plan_id
                    and analysis_manifest.get("schema_version") == "general-model-coverage-analysis/v1"
                    and analysis_manifest.get("gold_join_after_raw_sealed") is True
                    and analysis_manifest.get("test_content_read") is False
                    and (run / "analysis/analysis.json").is_file()):
                analysis_state, analysis_error = "published (receipt)", None
            else:
                analysis_error = analysis_error or "published analysis metadata is inconsistent/incomplete"
        elif terminal.get("analysis_published") is False:
            analysis_state = "pending-or-running" if state == "raw_complete" else "not-published"
    return {"schema_version": "coverage-read-only-status/v1", "run": str(run),
            "observed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_state": state, "run_receipt_error": terminal_error, "plan_id": plan_id,
            "preflight_state": gate_state, "preflight_receipt_error": gate_error,
            "sealed_preflight_passes": sum(row["sealed_manifest"] is True for row in passes),
            "expected_preflight_passes": 18,
            "unknown_preflight_pass_manifests": sum(row["sealed_manifest"] is None for row in passes),
            "preflight_passes": passes, "full_dev": full_dev,
            "analysis_state": analysis_state, "analysis_receipt_error": analysis_error,
            "process_liveness_checked": False, "score_payload_read": False, "query_gold_read": False,
            "note": "Persisted metadata/count snapshot only; no process liveness, score integrity, or scientific audit claimed."}


def count_label(row):
    if row["committed_blocks"] is not None:
        return f"{row['committed_blocks']}/{row['expected_blocks']}"
    known = row["known_committed_subtotal"]
    return (f"unknown/{row['expected_blocks']} (known={known if known is not None else 'unknown'}, "
            f"readable shards={row['readable_shards']}/{row['expected_shards']})")


def render_text(status):
    lines = [f"Run: {status['run']}", f"Observed: {status['observed_at_utc']}",
             f"Run state (receipt): {status['run_state']}",
             f"Preflight: {status['preflight_state']}; sealed manifests "
             f"{status['sealed_preflight_passes']}/{status['expected_preflight_passes']}; "
             f"unknown/missing {status['unknown_preflight_pass_manifests']}"]
    for row in status["preflight_passes"]:
        seal = "sealed" if row["sealed_manifest"] is True else "unknown"
        lines.append(f"  {row['pass']:<25} {count_label(row)} | {seal}")
    lines.append(f"Full dev: {count_label(status['full_dev'])}; "
                 f"manifest={'sealed' if status['full_dev']['sealed_manifest'] is True else 'unknown'}")
    lines.append(f"Analysis: {status['analysis_state']}")
    for name in ("run_receipt_error", "preflight_receipt_error", "analysis_receipt_error"):
        if status[name]:
            lines.append(f"{name}: {status[name]}")
    lines.append(status["note"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--json", action="store_true", help="Include per-shard missing/locked error details as JSON")
    args = parser.parse_args(argv)
    status = snapshot(args.run)
    print(json.dumps(status, ensure_ascii=True, indent=2) if args.json else render_text(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
