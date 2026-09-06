"""Checkpointed model-parallel preflight and full-dev phases, with no gate bypass."""

from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
from pathlib import Path

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_numeric as base
from diagnostics import general_model_numeric_v2 as scoring
from diagnostics.general_model_coverage import POLICY
from diagnostics.general_model_coverage_execution import _reference_comparison
from diagnostics.general_model_numeric_v3 import validate_geometry
from diagnostics.general_model_package import ROOT, PackageError, read_json, read_jsonl, write_json


SCHEMA = "general-model-coverage-replication-run/v1"
PREFLIGHT_SCHEMA = "general-model-coverage-replication-preflight/v1"
COHORTS = ("regression", "validation", "boundary")
PASSES = (("r0", {"reference": True}, "reference"), ("r1", {}, "repeat"),
          ("padding", {"padding_extra": 64}, "padding"), ("prefix", {"prefix": True}, "prefix"),
          ("members", {"permuted": True}, "members"), ("replica", {"replica_shift": 1}, "replica"))


def selected(plan, contexts, cohort):
    ids = set(plan["cohorts"][cohort])
    rows = [row for row in contexts if row["query_id"] in ids]
    if len(rows) != len(ids) * 16:
        raise PackageError("replication engineering frame differs")
    return rows


def runtime_path(output, shift):
    return output / ("runtime-replica.json" if shift else "runtime-baseline.json")


def save_runtime(plan, output, runner, shift):
    from diagnostics.general_model_numeric_sharded import validate_runtime_identity
    validate_runtime_identity(plan, runner.identity, replica_shift=shift)
    path = runtime_path(output, shift)
    if path.exists():
        if read_json(path) != runner.identity:
            raise PackageError("resumed sharded runtime differs from original identity")
    else:
        base.atomic_json(path, runner.identity)


def read_pass(plan, output, contexts, options, runtime):
    from diagnostics.general_model_numeric_sharded import validate_runtime_identity
    validate_runtime_identity(plan, runtime, replica_shift=options.get("replica_shift", 0))
    receipt = read_json(output / "manifest.json")
    profile = scoring.scoring_profile(**{k: v for k, v in options.items() if k != "reference"})
    expected = {"plan_id": plan["plan_id"], "runtime": runtime, "batch_size": 1,
                "reference": options.get("reference", False), "pass_name": output.name,
                "records": [row["record_id"] for row in contexts], "scoring_profile": profile}
    if (receipt.get("identity") != expected or receipt.get("status") != "complete"
            or receipt.get("schema_version") != "general-model-ld-numeric-pass/v2"
            or receipt.get("scores_sha256") != sha256_file(output / "scores.jsonl")
            or receipt.get("blocks") != len(contexts)
            or receipt.get("candidates") != sum(len(plan["catalog"][r["task"]]) for r in contexts)
            or any(receipt.get(k) is not False for k in ("query_gold_loaded", "test_content_read", "mixed_execution_modes"))):
        raise PackageError("replication raw pass identity or seal differs")
    rows = read_jsonl(output / "scores.jsonl")
    proof = validate_geometry(rows, contexts, plan, **options)
    expected_cohort = next((name for name in ("regression", "validation") if output.name.startswith(name)), "dev")
    devices = sorted(set(runtime["device_map"].values()))
    uuids = [row["uuid"] for row in runtime["hardware"]]
    for row in rows:
        if (row.get("runtime_sha256") != canonical_json_sha256(runtime) or row.get("pass_name") != output.name
                or row.get("cohort") != expected_cohort):
            raise PackageError("replication raw block runtime/pass attribution differs")
        for candidate in row["candidates"]:
            peaks = candidate.get("peak_memory_by_device")
            if (candidate.get("physical_gpu_indices") != devices
                    or candidate.get("physical_gpu_uuids") != uuids
                    or candidate.get("model_device_map_sha256") != runtime["device_map_sha256"]
                    or type(candidate.get("replica_shift")) is not int
                    or candidate["replica_shift"] != options.get("replica_shift", 0)
                    or candidate.get("fp32_operator_dispatch_checked") is not True
                    or type(candidate.get("fp32_operator_count")) is not int
                    or candidate["fp32_operator_count"] <= 0
                    or not isinstance(peaks, dict) or set(peaks) != {str(device) for device in devices}):
                raise PackageError("replication candidate multi-device/FP32 evidence differs")
            for peak in peaks.values():
                if (not isinstance(peak, dict) or set(peak) != {"allocated_bytes", "reserved_bytes"}
                        or any(type(value) is not int or value < 0 for value in peak.values())
                        or peak["allocated_bytes"] > peak["reserved_bytes"]):
                    raise PackageError("replication candidate device memory evidence is invalid")
    return rows, receipt, proof


def _evidence(directory):
    return {file.relative_to(directory).as_posix(): sha256_file(file)
            for file in sorted(directory.rglob("*")) if file.is_file()
            and (file.name in {"manifest.json", "scores.jsonl"}
                 or file.name.endswith(("-differences.json", "-geometry-proof.json")))}


def _preflight_order(plan):
    # Match the two model-load loops, not cohort-major presentation order.
    return [(cohort, suffix, options, label) for shift in (0, 1) for cohort in COHORTS
            if plan["cohorts"][cohort] for suffix, options, label in PASSES
            if options.get("replica_shift", 0) == shift]


def _verify_remapping(baseline, replica):
    first = {row["physical_gpu_index"]: row["uuid"] for row in baseline["hardware"]}
    second = {row["physical_gpu_index"]: row["uuid"] for row in replica["hardware"]}
    if (set(baseline["device_map"]) != set(replica["device_map"])
            or any(first[index] != second[index] for index in first.keys() & second.keys())
            or any(first[device] == second[replica["device_map"][module]]
                   for module, device in baseline["device_map"].items())):
        raise PackageError("replication placement challenge did not change every module's physical GPU")


def verify_preflight(plan, contexts, output):
    directory = output / "preflight"
    report = read_json(directory / "preflight_report.json")
    if (report.get("schema_version") != PREFLIGHT_SCHEMA or report.get("plan_id") != plan["plan_id"]
            or report.get("numeric_policy") != POLICY or report.get("files") != _evidence(directory)
            or report.get("query_gold_loaded") is not False or report.get("test_content_read") is not False
            or report.get("scientific_effect_checked") is not False):
        raise PackageError("replication preflight identity or evidence changed")
    order = _preflight_order(plan)
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or len(checks) > len(order):
        raise PackageError("replication preflight check inventory differs")
    executed = order[:len(checks)]
    if set(checks) != {f"{cohort}-{label}" for cohort, _, _, label in executed}:
        raise PackageError("replication preflight checks are not an execution prefix")
    expected_files = set()
    for cohort, suffix, _, label in executed:
        expected_files.update((f"{cohort}-b1-{suffix}/manifest.json", f"{cohort}-b1-{suffix}/scores.jsonl",
                               f"{cohort}-{label}-differences.json", f"{cohort}-{label}-geometry-proof.json"))
    if set(report["files"]) != expected_files:
        raise PackageError("replication preflight contains missing or unregistered pass evidence")
    shifts = {str(options.get("replica_shift", 0)) for _, _, options, _ in executed}
    if not isinstance(report.get("runtime_sha256"), dict) or set(report["runtime_sha256"]) != shifts:
        raise PackageError("replication preflight runtime inventory differs")
    runtimes = {}
    for shift in shifts:
        path = runtime_path(output, int(shift))
        if sha256_file(path) != report["runtime_sha256"][shift]:
            raise PackageError("replication preflight runtime seal changed")
        runtimes[int(shift)] = read_json(path)
    if 1 in runtimes:
        _verify_remapping(runtimes[0], runtimes[1])
    comparisons = {}
    for cohort in COHORTS:
        frame = selected(plan, contexts, cohort)
        if not frame:
            continue
        baseline = None
        for suffix, options, label in PASSES:
            key = f"{cohort}-{label}"
            if key not in report["checks"]:
                continue
            shift = options.get("replica_shift", 0)
            runtime = runtimes[shift]
            path = directory / f"{cohort}-b1-{suffix}"
            rows, _, proof = read_pass(plan, path, frame, options, runtime)
            if suffix == "r0":
                baseline = rows
                comparison = _reference_comparison(rows)
            else:
                if baseline is None:
                    raise PackageError("preflight challenge has no baseline")
                comparison = base.compare_passes(baseline, rows)
            if (read_json(directory / f"{key}-differences.json") != comparison
                    or read_json(directory / f"{key}-geometry-proof.json") != proof):
                raise PackageError("replication preflight proof does not recompute")
            limit = POLICY["reference_abs_tolerance"] if label == "reference" else (
                POLICY["repeat_abs_tolerance"] if label == "repeat" else POLICY["epsilon"])
            comparisons[key] = {"max_abs_error": comparison["max_abs_error"], "limit": limit,
                                "passed": comparison["max_abs_error"] <= limit}
    expected_keys = {f"{cohort}-{label}" for cohort in COHORTS if plan["cohorts"][cohort] for _, _, label in PASSES}
    if set(report["checks"]) - expected_keys or comparisons != report["checks"]:
        raise PackageError("replication preflight check inventory differs")
    complete = set(comparisons) == expected_keys
    passed = complete and all(row["passed"] for row in comparisons.values())
    failures = [key for key, row in comparisons.items() if not row["passed"]]
    if (report.get("complete") is not complete or report.get("passed") is not passed
            or (passed and "failure" in report)
            or (not passed and (len(failures) != 1 or report.get("failure") != failures[0]
                                or failures[0] != f"{executed[-1][0]}-{executed[-1][3]}"))):
        raise PackageError("replication preflight pass/failure claim differs")
    return report


def run_preflight(plan, contexts, output, runner_factory):
    from diagnostics.general_model_numeric_sharded import score_batch, score_prefix_block
    directory = output / "preflight"
    directory.mkdir(exist_ok=True)
    report_path = directory / "preflight_report.json"
    if report_path.exists():
        return verify_preflight(plan, contexts, output)
    report = {"schema_version": PREFLIGHT_SCHEMA, "plan_id": plan["plan_id"], "numeric_policy": POLICY,
              "checks": {}, "complete": False, "passed": False, "query_gold_loaded": False,
              "test_content_read": False, "scientific_effect_checked": False, "runtime_sha256": {},
              "placement_challenge": "same-layer-partition-different-physical-GPUs",
              "inherited_E8_is_not_new_model_calibration": True}
    baseline_rows = {}
    # Reuse a model load across cohorts; the physical remapping uses a second load.
    for shift in (0, 1):
        runner = runner_factory(plan, replica_shift=shift)
        try:
            save_runtime(plan, output, runner, shift)
            report["runtime_sha256"][str(shift)] = sha256_file(runtime_path(output, shift))
            for cohort in COHORTS:
                frame = selected(plan, contexts, cohort)
                if not frame:
                    continue
                for suffix, options, label in PASSES:
                    if options.get("replica_shift", 0) != shift:
                        continue
                    path = directory / f"{cohort}-b1-{suffix}"
                    scoring.score_pass(runner, frame, plan, path, batch_size=1,
                                       scorer=score_batch, prefix_scorer=score_prefix_block, **options)
                    rows, _, proof = read_pass(plan, path, frame, options, runner.identity)
                    if suffix == "r0":
                        baseline_rows[cohort] = rows
                        comparison = _reference_comparison(rows)
                    else:
                        comparison = base.compare_passes(baseline_rows[cohort], rows)
                    limit = POLICY["reference_abs_tolerance"] if label == "reference" else (
                        POLICY["repeat_abs_tolerance"] if label == "repeat" else POLICY["epsilon"])
                    key = f"{cohort}-{label}"
                    check = {"max_abs_error": comparison["max_abs_error"], "limit": limit,
                             "passed": comparison["max_abs_error"] <= limit}
                    report["checks"][key] = check
                    base.atomic_json(directory / f"{key}-differences.json", comparison)
                    base.atomic_json(directory / f"{key}-geometry-proof.json", proof)
                    base.progress("replication-preflight-check", model=plan["model"]["key"], check=key, **check)
                    if not check["passed"]:
                        report["failure"] = key
                        break
                if "failure" in report:
                    break
        finally:
            runner.close()
        if "failure" in report:
            break
    expected = sum(bool(plan["cohorts"][name]) for name in COHORTS) * len(PASSES)
    report["complete"] = len(report["checks"]) == expected
    report["passed"] = report["complete"] and all(check["passed"] for check in report["checks"].values())
    report["files"] = _evidence(directory)
    base.atomic_json(report_path, report)
    return verify_preflight(plan, contexts, output)


def analyze(plan, contexts, output, result):
    if result["status"] not in {"raw_complete", "complete"}:
        raise PackageError("replication analysis requires complete sealed raw")
    report = verify_preflight(plan, contexts, output)
    if not report["passed"] or sha256_file(output / "preflight/preflight_report.json") != result["preflight_report_sha256"]:
        raise PackageError("replication analysis lacks the verified passed preflight")
    raw_path = output / "dev-b1"
    if sha256_file(raw_path / "manifest.json") != result["raw_manifest_sha256"]:
        raise PackageError("replication full raw seal changed")
    rows, _, geometry = read_pass(plan, raw_path, contexts, {}, read_json(runtime_path(output, 0)))
    destination = output / "analysis"
    if destination.exists():
        receipt = read_json(destination / "manifest.json")
        if (receipt.get("schema_version") != "general-model-coverage-replication-analysis/v1"
                or receipt.get("model_key") != plan["model"]["key"]
                or receipt.get("production_geometry") != geometry
                or receipt.get("plan_id") != plan["plan_id"] or receipt.get("raw_manifest_sha256") != result["raw_manifest_sha256"]
                or receipt.get("analysis_sha256") != sha256_file(destination / "analysis.json")
                or receipt.get("gold_join_after_raw_sealed") is not True or receipt.get("test_content_read") is not False):
            raise PackageError("replication existing analysis is not bound to sealed raw")
        return receipt
    if result["status"] == "complete":
        raise PackageError("completed replication lost its analysis")
    package = Path(plan["package_path"])
    if sha256_file(package / "manifest.json") != plan["package_manifest_sha256"]:
        raise PackageError("replication query gold manifest changed")
    entry = next(row for row in read_json(package / "manifest.json")["files"] if row["path"] == "queries.dev.jsonl")
    if sha256_file(package / "queries.dev.jsonl") != entry["sha256"]:
        raise PackageError("replication query gold source changed")
    # This is the first query-gold deserialization in the new model's pipeline.
    gold = {str(row["id"]): row["projection"] for row in read_jsonl(package / "queries.dev.jsonl")}
    from diagnostics.general_model_numeric_coverage_analysis import analyze_blocks
    bootstrap = plan["config"]["analysis"]["bootstrap"]
    analysis = analyze_blocks(rows, frame=plan["frame"], gold_by_query=gold, epsilon=POLICY["epsilon"],
                              bootstrap_replicates=bootstrap["repetitions"], bootstrap_seed=bootstrap["seed"])
    staging = Path(tempfile.mkdtemp(prefix=".analyzing-", dir=output))
    try:
        write_json(staging / "analysis.json", analysis)
        receipt = {"schema_version": "general-model-coverage-replication-analysis/v1", "plan_id": plan["plan_id"],
                   "model_key": plan["model"]["key"], "raw_manifest_sha256": result["raw_manifest_sha256"],
                   "analysis_sha256": sha256_file(staging / "analysis.json"), "gold_join_after_raw_sealed": True,
                   "test_content_read": False, "production_geometry": geometry}
        write_json(staging / "manifest.json", receipt)
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return receipt


def execute(plan, contexts, output, *, phase, runner_factory=None):
    from diagnostics.general_model_numeric_sharded import ShardedRunner, score_batch, score_prefix_block
    if phase not in {"preflight", "dev", "analyze"}:
        raise PackageError("unknown replication execution phase")
    runner_factory = runner_factory or ShardedRunner
    output = Path(output).resolve()
    allowed = (ROOT / plan["config"]["output_root"] / "runs").resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise PackageError("replication output must be inside its own runs namespace")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".writer.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = output / "run_manifest.json"
    runner = None
    started = False
    result = {"schema_version": SCHEMA, "plan_id": plan["plan_id"], "model_key": plan["model"]["key"],
              "status": "running", "execution": "model-parallel-fp32", "full_dev_started": False,
              "analysis_published": False, "query_gold_loaded_during_scoring": False,
              "test_content_read": False, "automatic_profile_search": False}
    try:
        if manifest.exists():
            previous = read_json(manifest)
            if (previous.get("plan_id") != plan["plan_id"] or previous.get("schema_version") != SCHEMA
                    or previous.get("model_key") != plan["model"]["key"]
                    or previous.get("execution") != "model-parallel-fp32"
                    or any(previous.get(k) is not False for k in
                           ("query_gold_loaded_during_scoring", "test_content_read", "automatic_profile_search"))):
                raise PackageError("replication run belongs to another plan/boundary")
            result = previous
            if (type(result.get("full_dev_started")) is not bool
                    or result.get("analysis_published") is not (result.get("status") == "complete")
                    or (result.get("status") in {"raw_complete", "complete"}
                        and (result["full_dev_started"] is not True or result.get("raw_blocks") != len(contexts)
                             or not result.get("raw_manifest_sha256") or not result.get("preflight_report_sha256")))
                    or (result.get("raw_manifest_sha256") and result.get("status") not in {"raw_complete", "complete"})):
                raise PackageError("replication run state disagrees with its sealed artifacts")
            if result["status"] in {"failed", "preflight_failed"}:
                raise PackageError("failed replication is sealed; no automatic retry")
            if result["status"] == "complete":
                receipt = analyze(plan, contexts, output, result)
                if result["analysis_manifest_sha256"] != sha256_file(output / "analysis/manifest.json"):
                    raise PackageError("replication analysis manifest changed")
                return result
            if result["status"] not in {"running", "interrupted", "preflight_passed", "raw_complete"}:
                raise PackageError("unknown replication run status")
        elif any((output / name).exists() for name in
                 ("preflight", "dev-b1", "analysis", "runtime-baseline.json", "runtime-replica.json")):
            raise PackageError("refusing unbound replication artifacts")
        if phase == "preflight" and result.get("preflight_report_sha256"):
            report = verify_preflight(plan, contexts, output)
            if not report["passed"] or sha256_file(output / "preflight/preflight_report.json") != result["preflight_report_sha256"]:
                raise PackageError("replication preflight binding changed")
            return result
        if phase in {"dev", "analyze"}:
            report = verify_preflight(plan, contexts, output)
            if not report["passed"] or sha256_file(output / "preflight/preflight_report.json") != result.get("preflight_report_sha256"):
                raise PackageError("full dev requires this model's sealed successful preflight")
        if phase == "analyze" and result["status"] != "raw_complete":
            raise PackageError("analysis cannot initiate GPU scoring")
        started = True
        result["active_phase"] = phase
        if result["status"] != "raw_complete":
            result["status"] = "running"
        if "error_type" in result:
            result.setdefault("prior_interruptions", []).append({key: result.pop(key) for key in ("error_type", "error") if key in result})
        base.atomic_json(manifest, result)
        if phase == "preflight":
            report = run_preflight(plan, contexts, output, runner_factory)
            result["preflight_report_sha256"] = sha256_file(output / "preflight/preflight_report.json")
            result["status"] = "preflight_passed" if report["passed"] else "preflight_failed"
            if not report["passed"]:
                result["failure"] = report["failure"]
        else:
            if result["status"] != "raw_complete":
                runner = runner_factory(plan, replica_shift=0)
                save_runtime(plan, output, runner, 0)
                result["full_dev_started"] = True
                base.atomic_json(manifest, result)
                scoring.score_pass(runner, contexts, plan, output / "dev-b1", batch_size=1,
                                   scorer=score_batch, prefix_scorer=score_prefix_block)
                read_pass(plan, output / "dev-b1", contexts, {}, runner.identity)
                result.update(status="raw_complete", raw_blocks=len(contexts),
                              raw_manifest_sha256=sha256_file(output / "dev-b1/manifest.json"))
                base.atomic_json(manifest, result)
                runner.close()
                runner = None
            analyze(plan, contexts, output, result)
            result.update(status="complete", analysis_published=True,
                          analysis_manifest_sha256=sha256_file(output / "analysis/manifest.json"))
        base.atomic_json(manifest, result)
        return result
    except BaseException as error:
        if started:
            result.update(status="raw_complete" if result.get("raw_manifest_sha256") else (
                          "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"),
                          error_type=type(error).__name__, error=str(error), analysis_published=False)
            base.atomic_json(manifest, result)
        raise
    finally:
        if runner is not None:
            runner.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
