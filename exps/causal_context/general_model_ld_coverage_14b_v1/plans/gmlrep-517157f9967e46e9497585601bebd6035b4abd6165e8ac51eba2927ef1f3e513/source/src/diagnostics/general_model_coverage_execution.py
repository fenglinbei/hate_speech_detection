"""Fail-closed execution of the separately registered merged-lexicon experiment."""

from __future__ import annotations

import copy
import fcntl
import math
import os
import shutil
import tempfile
from pathlib import Path

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_numeric as base
from diagnostics import general_model_numeric_v2 as scoring
from diagnostics.general_model_numeric_analysis import candidate_scores
from diagnostics.general_model_numeric_pool import PersistentNumericPool, validate_sealed_pass
from diagnostics.general_model_numeric_v3 import validate_geometry
from diagnostics.general_model_package import ROOT, PackageError, read_json, read_jsonl, write_json


COHORTS = ("regression", "validation", "boundary")
CHALLENGES = ("padding", "prefix", "members", "replica")
NOT_APPLICABLE = ["batch-four", "tail-size-two", "within-batch-row-position"]
POLICY = {"E8": 0.00067138671875, "epsilon": 0.0013427734375,
          "repeat_abs_tolerance": 1e-4, "reference_abs_tolerance": 1e-4,
          "padding_extra": 64, "replica_shift": 1}
RAW_PATH = "dev-b1"
CALIBRATION_MODE = "inherited-fixed-tolerance-new-input-revalidation"


def _definitions():
    return (("r0", {"reference": True}, "baseline"), ("r1", {}, "repeat"),
            ("padding", {"padding_extra": 64}, "padding"), ("prefix", {"prefix": True}, "prefix"),
            ("members", {"permuted": True}, "members"), ("replica", {"replica_shift": 1}, "replica"))


def _devices(plan):
    return plan["config"]["execution"]["device_indices"]


def validate_execution_plan(plan: dict, contexts: list[dict]) -> None:
    if plan["numeric_policy"] != POLICY:
        raise PackageError("coverage execution cannot recalibrate or loosen the numerical policy")
    config = plan["config"]
    if (config["tasks"] != ["hate", "group"] or len(config["conditions"]) != 8
            or len(set(config["conditions"])) != 8 or _devices(plan) != [0, 1, 2, 3]):
        raise PackageError("coverage execution model/task/condition/device matrix differs")
    qids = {row["query_id"] for row in plan["frame"]}
    expected = {f"{q}:{task}:{condition}" for q in qids for task in config["tasks"]
                for condition in config["conditions"]}
    if (len(qids) != 643 or len(plan["frame"]) != 643 or len(contexts) != 10288
            or len(plan["blocks"]) != 10288 or {row["record_id"] for row in contexts} != expected
            or sum(len(plan["catalog"][row["task"]]) for row in contexts) != 174896):
        raise PackageError("coverage full-development frame is incomplete")
    for actual, descriptor in zip(contexts, plan["blocks"], strict=True):
        if any(actual.get(key) != value for key, value in descriptor.items()):
            raise PackageError("coverage execution inputs differ from the frozen descriptors")
    if set(plan["cohorts"]) != set(COHORTS):
        raise PackageError("coverage preflight cohort names differ")
    seen = set()
    for name in COHORTS:
        ids = plan["cohorts"][name]
        count_ok = len(ids) == {"regression": 8, "validation": 24}.get(name, len(ids))
        if (not count_ok or (name == "boundary" and not 0 <= len(ids) <= 4)
                or len(ids) != len(set(ids)) or seen & set(ids) or not set(ids) <= qids):
            raise PackageError("coverage preflight cohorts overlap or have invalid coverage")
        seen.update(ids)
    runtime = config["runtime"]
    required = {"dtype": "float32", "padding_policy": "dynamic", "use_cache": False,
                "attention_implementation": "eager", "max_sequence_tokens": 8192,
                "enable_thinking": False, "seed": 42, "padding_side": "right"}
    if any(runtime.get(key) != value for key, value in required.items()):
        raise PackageError("coverage numerical runtime changed the registered profile")
    parent = plan["runtime_parent_plan"]
    if (parent["blocks"] != plan["blocks"] or parent["catalog"] != plan["catalog"]
            or parent["package_path"] != plan["package_path"]
            or parent["generation_runtime_identity"] != plan["generation_runtime_identity"]):
        raise PackageError("coverage model initialization is bound to other inputs")


def validate_runtime(plan: dict, identity: dict) -> None:
    runtime = plan["config"]["runtime"]
    if (identity.get("numeric_runtime") != runtime or identity.get("device_indices") != _devices(plan)
            or identity.get("execution") != "data-parallel-identical-fp32"):
        raise PackageError("coverage runtime differs from the registered identical FP32 replicas")
    lengths = {}
    for row in plan["blocks"]:
        key = (row["query_id"], row["task"])
        length = row["prompt_tokens"] + max(c["answer_tokens"] for c in plan["catalog"][row["task"]]) + 1
        lengths[key] = max(length, lengths.get(key, 0))
    wanted = {
        "source_generation_runtime": plan["generation_runtime_identity"], "numeric_runtime": runtime,
        "transformer_dtype": "torch.float32", "lm_head_dtype": "torch.float32",
        "tf32_matmul": False, "tf32_cudnn": False, "use_cache": False,
        "projection": "answer-and-eos-prediction-positions-only", "logprob_arithmetic": "float32",
        "aggregation": "float64", "cpu_threads": runtime["cpu_threads"],
        "bf16_reduced_precision_reduction": runtime["bf16_reduced_precision_reduction"],
        "fixed_length_table": [{"query_id": q, "task": t, "tokens": n} for (q, t), n in sorted(lengths.items())],
        "global_padding_length": max(lengths.values()),
    }
    replicas = identity.get("replicas", [])
    if ([row.get("physical_gpu_index") for row in replicas] != _devices(plan)
            or len({row.get("hardware", {}).get("uuid") for row in replicas}) != len(replicas)):
        raise PackageError("coverage runtime does not prove distinct assigned GPU replicas")
    for row in replicas:
        actual = row.get("actual_numeric_identity", {})
        if (actual != wanted or row.get("actual_numeric_identity_sha256") != canonical_json_sha256(actual)
                or row.get("hardware", {}).get("physical_gpu_index") != row["physical_gpu_index"]
                or not str(row.get("hardware", {}).get("uuid", "")).startswith("GPU-")):
            raise PackageError("coverage actual runtime or new input geometry differs")
        sources = row.get("source_sha256", {})
        if not sources or any(plan["code_sha256"].get(name) != digest for name, digest in sources.items()):
            raise PackageError("coverage worker source attestation differs from the frozen plan")


def _selected(plan, contexts, cohort):
    ids = set(plan["cohorts"][cohort])
    selected = [row for row in contexts if row["query_id"] in ids]
    expected = {f"{qid}:{task}:{condition}" for qid in ids for task in plan["config"]["tasks"]
                for condition in plan["config"]["conditions"]}
    if len(selected) != len(expected) or {row["record_id"] for row in selected} != expected:
        raise PackageError("coverage preflight cohort matrix is incomplete")
    return selected


def _reference_comparison(rows):
    reference = copy.deepcopy(rows)
    for row in reference:
        for candidate in row["candidates"]:
            candidate.update(candidate["reference_scores"])
            candidate["scores"] = candidate_scores(candidate["token_logprobs"], candidate["eos_logprob"])
    return base.compare_passes(reference, rows)


def _evidence_files(directory):
    return {path.relative_to(directory).as_posix(): sha256_file(path)
            for path in sorted(directory.rglob("*")) if path.is_file()
            and (path.name in {"scores.jsonl", "manifest.json", "pool_binding.json"}
                 or path.name.endswith(("-differences.json", "-proof.json")))}


def preflight(runner, plan: dict, contexts: list[dict], output: Path) -> dict:
    validate_runtime(plan, runner.identity)
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "general-model-coverage-preflight/v1", "plan_id": plan["plan_id"],
              "runtime_identity": runner.identity, "passed": False, "complete": False, "cohorts": {},
              "numeric_policy": POLICY, "E8": POLICY["E8"], "epsilon": POLICY["epsilon"],
              "calibration_mode": CALIBRATION_MODE, "error_families": list(CHALLENGES),
              "not_applicable": NOT_APPLICABLE, "query_gold_loaded": False, "test_content_read": False,
              "scientific_effect_checked": False, "formal_test_authorized": False}
    for cohort in COHORTS:
        selected = _selected(plan, contexts, cohort)
        if not selected:
            report["cohorts"][cohort] = {"blocks": 0, "skipped": True, "reason": "no-new-boundary-query"}
            continue
        baseline = None
        info = {"blocks": len(selected), "challenges": {}, "baseline_passed": False,
                "complete": False, "passed": False}
        report["cohorts"][cohort] = info
        for suffix, options, label in _definitions():
            rows, _ = scoring.score_pass(runner, selected, plan, output / f"{cohort}-b1-{suffix}",
                                          batch_size=1, **options)
            write_json(output / f"{cohort}-{label}-geometry-proof.json", validate_geometry(rows, selected, plan, **options))
            if suffix == "r0":
                baseline = rows
                comparison = _reference_comparison(rows)
                name = "reference"
                info["reference_max_abs_error"] = comparison["max_abs_error"]
                limit = POLICY["reference_abs_tolerance"]
            else:
                comparison = base.compare_passes(baseline, rows)
                name = "repeat" if suffix == "r1" else suffix
                limit = POLICY["repeat_abs_tolerance"] if suffix == "r1" else POLICY["epsilon"]
                if suffix == "r1":
                    info["baseline_repeat_max_abs_error"] = comparison["max_abs_error"]
                    info["baseline_passed"] = comparison["max_abs_error"] <= limit
                else:
                    info["challenges"][name] = {"max_abs_error": comparison["max_abs_error"],
                        "largest_error": comparison["largest_error"], "passed": comparison["max_abs_error"] <= limit}
            write_json(output / f"{cohort}-{name}-differences.json", comparison)
            if suffix == "replica":
                write_json(output / f"{cohort}-replica-producer-proof.json", scoring.replica_proof(baseline, rows))
            if comparison["max_abs_error"] > limit:
                report["failure"] = f"{cohort}-{name}-gate"
                break
        info["complete"] = len(info["challenges"]) == len(CHALLENGES)
        info["passed"] = info["complete"] and info["baseline_passed"] and all(
            check["passed"] for check in info["challenges"].values())
        if not info["passed"]:
            break
    report["complete"] = set(report["cohorts"]) == set(COHORTS) and all(
        info.get("complete", info.get("skipped", False)) for info in report["cohorts"].values())
    report["passed"] = report["complete"] and all(
        info.get("passed", info.get("skipped", False)) for info in report["cohorts"].values())
    report["observed_max_abs_error"] = max((check["max_abs_error"] for info in report["cohorts"].values()
        for check in info.get("challenges", {}).values()), default=0.0)
    report["files"] = _evidence_files(output)
    base.atomic_json(output / "preflight_report.json", report)
    base.progress("coverage-preflight-finished", passed=report["passed"], failure=report.get("failure"),
                  epsilon=POLICY["epsilon"], observed_max_abs_error=report["observed_max_abs_error"])
    return report


def _verify_pass(plan, path, contexts, runtime, options):
    receipt = validate_sealed_pass(path, plan)
    expected = {"plan_id": plan["plan_id"], "runtime": runtime, "batch_size": 1,
                "reference": options.get("reference", False), "pass_name": path.name,
                "records": [row["record_id"] for row in contexts],
                "scoring_profile": scoring.scoring_profile(**{k: v for k, v in options.items() if k != "reference"})}
    if (any(receipt.get("identity", {}).get(key) != value for key, value in expected.items())
            or receipt.get("execution") != "data-parallel-identical-fp32"
            or any(receipt.get(key) is not False for key in
                   ("query_gold_loaded", "test_content_read", "mixed_execution_modes"))
            or receipt.get("blocks") != len(contexts)
            or receipt.get("candidates") != sum(len(plan["catalog"][r["task"]]) for r in contexts)):
        raise PackageError("coverage sealed pass identity, counts, or data boundary differs")
    rows = read_jsonl(path / "scores.jsonl")
    proof = validate_geometry(rows, contexts, plan, **options)
    return rows, receipt, proof


def verified_preflight(plan: dict, output: Path, run: dict) -> dict:
    directory = output / "preflight"
    path = directory / "preflight_report.json"
    if sha256_file(path) != run["preflight_report_sha256"]:
        raise PackageError("coverage sealed preflight report changed")
    report = read_json(path)
    expected = {"schema_version": "general-model-coverage-preflight/v1", "plan_id": plan["plan_id"],
                "numeric_policy": POLICY, "E8": POLICY["E8"], "epsilon": POLICY["epsilon"],
                "calibration_mode": CALIBRATION_MODE, "error_families": list(CHALLENGES),
                "not_applicable": NOT_APPLICABLE, "query_gold_loaded": False, "test_content_read": False,
                "scientific_effect_checked": False, "formal_test_authorized": False}
    if (any(report.get(key) != value for key, value in expected.items())
            or report.get("runtime_identity") != read_json(output / "runtime_identity.json")
            or report.get("files") != _evidence_files(directory)):
        raise PackageError("coverage preflight identity, boundary, or sealed evidence inventory differs")
    validate_runtime(plan, report["runtime_identity"])
    if set(report.get("cohorts", {})) != set(COHORTS[:len(report.get("cohorts", {}))]):
        raise PackageError("coverage preflight cohorts are not a consecutive execution prefix")
    observed = []
    failure = None
    verified_cohorts = {}
    pass_names = set()
    for cohort in COHORTS:
        if cohort not in report["cohorts"]:
            break
        contexts = _selected(plan, plan["blocks"], cohort)
        info = report["cohorts"][cohort]
        if not contexts:
            if info != {"blocks": 0, "skipped": True, "reason": "no-new-boundary-query"}:
                raise PackageError("coverage empty boundary cohort was misrepresented as tested")
            verified_cohorts[cohort] = info
            continue
        reconstructed = {"blocks": len(contexts), "challenges": {}, "baseline_passed": False,
                         "complete": False, "passed": False}
        baseline = None
        for suffix, options, label in _definitions():
            pass_name = f"{cohort}-b1-{suffix}"
            pass_names.add(pass_name)
            rows, _, proof = _verify_pass(plan, directory / pass_name, contexts,
                                           report["runtime_identity"], options)
            if proof != read_json(directory / f"{cohort}-{label}-geometry-proof.json"):
                raise PackageError("coverage preflight geometry proof differs")
            if suffix == "r0":
                baseline = rows
                comparison = _reference_comparison(rows)
                name, limit = "reference", POLICY["reference_abs_tolerance"]
                reconstructed["reference_max_abs_error"] = comparison["max_abs_error"]
            else:
                comparison = base.compare_passes(baseline, rows)
                name = "repeat" if suffix == "r1" else suffix
                limit = POLICY["repeat_abs_tolerance"] if suffix == "r1" else POLICY["epsilon"]
                if suffix == "r1":
                    reconstructed["baseline_repeat_max_abs_error"] = comparison["max_abs_error"]
                    reconstructed["baseline_passed"] = comparison["max_abs_error"] <= limit
                else:
                    reconstructed["challenges"][name] = {"max_abs_error": comparison["max_abs_error"],
                        "largest_error": comparison["largest_error"], "passed": comparison["max_abs_error"] <= limit}
                    observed.append(comparison["max_abs_error"])
            if (not math.isfinite(comparison["max_abs_error"]) or comparison["max_abs_error"] < 0
                    or comparison != read_json(directory / f"{cohort}-{name}-differences.json")):
                raise PackageError("coverage preflight recomputation disagrees with report evidence")
            if suffix == "replica" and scoring.replica_proof(baseline, rows) != read_json(
                    directory / f"{cohort}-replica-producer-proof.json"):
                raise PackageError("coverage physical replica proof differs")
            if comparison["max_abs_error"] > limit:
                failure = f"{cohort}-{name}-gate"
                break
        reconstructed["complete"] = len(reconstructed["challenges"]) == len(CHALLENGES)
        reconstructed["passed"] = reconstructed["complete"] and reconstructed["baseline_passed"] and all(
            check["passed"] for check in reconstructed["challenges"].values())
        if info != reconstructed:
            raise PackageError("coverage preflight cohort summary disagrees with raw evidence")
        verified_cohorts[cohort] = reconstructed
        if failure:
            break
    complete = set(verified_cohorts) == set(COHORTS) and all(
        info.get("complete", info.get("skipped", False)) for info in verified_cohorts.values())
    passed = complete and all(info.get("passed", info.get("skipped", False)) for info in verified_cohorts.values())
    if (verified_cohorts != report["cohorts"] or report.get("complete") is not complete
            or report.get("passed") is not passed or report.get("failure") != failure
            or {path.name for path in directory.iterdir() if path.is_dir()} != pass_names):
        raise PackageError("coverage terminal preflight flags or executed pass inventory disagree")
    if report.get("observed_max_abs_error") != max(observed, default=0.0):
        raise PackageError("coverage observed preflight maximum differs")
    return report


def analyze_run(plan: dict, output: Path, *, blocks=None) -> dict:
    run = read_json(output / "run_manifest.json")
    if run.get("plan_id") != plan["plan_id"] or run.get("status") not in {"raw_complete", "complete"}:
        raise PackageError("coverage analysis requires the sealed complete raw run")
    report = verified_preflight(plan, output, run)
    if not report["passed"]:
        raise PackageError("coverage analysis requires this experiment's passed preflight")
    raw = output / RAW_PATH
    if sha256_file(raw / "manifest.json") != run["raw_manifest_sha256"]:
        raise PackageError("coverage raw manifest changed after sealing")
    stored, _, geometry = _verify_pass(plan, raw, plan["blocks"], report["runtime_identity"], {})
    if blocks is not None and blocks != stored:
        raise PackageError("coverage analysis input differs from sealed raw")
    directory = output / "analysis"
    if directory.exists():
        existing = read_json(directory / "manifest.json")
        expected = {"schema_version": "general-model-coverage-analysis/v1", "plan_id": plan["plan_id"],
                    "raw_manifest_sha256": run["raw_manifest_sha256"], "gold_join_after_raw_sealed": True,
                    "test_content_read": False, "production_geometry": geometry}
        if (any(existing.get(key) != value for key, value in expected.items())
                or sha256_file(directory / "analysis.json") != existing.get("analysis_sha256")
                or (run["status"] == "complete" and sha256_file(directory / "manifest.json") != run.get("analysis_manifest_sha256"))):
            raise PackageError("coverage existing analysis identity differs")
        return existing
    if run["status"] == "complete":
        raise PackageError("coverage complete run lost its sealed analysis")
    # Query gold is first deserialized only after all raw and preflight evidence was verified.
    package = Path(plan["package_path"])
    manifest = read_json(package / "manifest.json")
    gold_entry = next(row for row in manifest["files"] if row["path"] == "queries.dev.jsonl")
    if (sha256_file(package / "manifest.json") != plan["package_manifest_sha256"]
            or sha256_file(package / "queries.dev.jsonl") != gold_entry["sha256"]):
        raise PackageError("coverage gold source changed after scoring")
    gold = {str(row["id"]): row["projection"] for row in read_jsonl(package / "queries.dev.jsonl")}
    from diagnostics.general_model_numeric_coverage_analysis import analyze_blocks

    bootstrap = plan["config"]["analysis"]["bootstrap"]
    analysis = analyze_blocks(stored, frame=plan["frame"], gold_by_query=gold, epsilon=POLICY["epsilon"],
                              bootstrap_replicates=bootstrap["repetitions"], bootstrap_seed=bootstrap["seed"])
    staging = Path(tempfile.mkdtemp(prefix=".analyzing-", dir=output))
    try:
        write_json(staging / "analysis.json", analysis)
        receipt = {"schema_version": "general-model-coverage-analysis/v1", "plan_id": plan["plan_id"],
                   "raw_manifest_sha256": run["raw_manifest_sha256"], "gold_join_after_raw_sealed": True,
                   "test_content_read": False, "production_geometry": geometry,
                   "analysis_sha256": sha256_file(staging / "analysis.json")}
        write_json(staging / "manifest.json", receipt)
        os.rename(staging, directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return receipt


def run_pipeline(plan: dict, contexts: list[dict], output: Path, *, root: Path = ROOT) -> dict:
    validate_execution_plan(plan, contexts)
    output = output.resolve()
    registered_runs = (root / plan["config"]["output_root"] / "runs").resolve()
    if output == registered_runs or not output.is_relative_to(registered_runs):
        raise PackageError("coverage run must use a new directory under the registered runs namespace")
    if output.is_relative_to(Path(plan["package_path"]).resolve()):
        raise PackageError("coverage run may not overwrite its source package")
    if "historical_run" in plan and output.is_relative_to(Path(plan["historical_run"]).resolve()):
        raise PackageError("coverage run may not overwrite its historical reference")
    output.mkdir(parents=True, exist_ok=True)
    binding = {"plan_id": plan["plan_id"], "physical_device_indices": _devices(plan),
               "execution": "data-parallel-identical-fp32", "phase": "merged-lexicon-coverage",
               "raw_path": RAW_PATH, "production_batch_size": 1, "numeric_policy": POLICY}
    binding_path = output / "binding.json"
    lock = (output / ".writer.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runner = None
    execution_started = False
    manifest_path = output / "run_manifest.json"
    result = {**binding, "schema_version": "general-model-coverage-run/v1", "status": "running",
              "test_content_read": False, "query_gold_loaded_during_scoring": False,
              "automatic_profile_search": False, "full_dev_started": False, "analysis_published": False}
    try:
        if binding_path.exists():
            if read_json(binding_path) != binding:
                raise PackageError("coverage run directory belongs to another execution")
        else:
            if manifest_path.exists() or any((output / name).exists() for name in ("preflight", RAW_PATH, "analysis", "runtime_identity.json")):
                raise PackageError("refusing unbound existing coverage artifacts")
            base.atomic_json(binding_path, binding)
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if (any(previous.get(key) != value for key, value in binding.items())
                    or any(previous.get(key) is not False for key in
                           ("test_content_read", "query_gold_loaded_during_scoring", "automatic_profile_search"))):
                raise PackageError("coverage run manifest binding or boundary differs")
            if previous.get("status") in {"complete", "preflight_failed"}:
                report = verified_preflight(plan, output, previous)
                if (previous["status"] == "complete") != report["passed"]:
                    raise PackageError("coverage terminal execution disagrees with preflight")
                if previous["status"] == "complete":
                    if previous.get("full_dev_started") is not True or previous.get("analysis_published") is not True:
                        raise PackageError("coverage completed run flags differ")
                    analyze_run(plan, output)
                elif (previous.get("failure") != report.get("failure")
                      or previous.get("full_dev_started") is not False or previous.get("analysis_published") is not False
                      or (output / RAW_PATH).exists() or (output / "analysis").exists()):
                    raise PackageError("coverage failed preflight has downstream artifacts or inconsistent flags")
                return previous
            if previous.get("status") == "failed":
                raise PackageError("failed coverage run is sealed; automatic retry or profile search is forbidden")
            if previous.get("status") == "raw_complete":
                result = previous
                analyze_run(plan, output)
                result.update(status="complete", analysis_published=True,
                              analysis_manifest_sha256=sha256_file(output / "analysis/manifest.json"))
                base.atomic_json(manifest_path, result)
                return result
            if previous.get("status") not in {"running", "interrupted"}:
                raise PackageError("unknown coverage execution status")
            result = {**previous, "status": "running", "analysis_published": False}
        if result.get("preflight_report_sha256"):
            verified_preflight(plan, output, result)
        execution_started = True
        base.atomic_json(manifest_path, result)
        runner = PersistentNumericPool(plan["runtime_parent_plan"], plan["config"]["runtime"], _devices(plan), root)
        validate_runtime(plan, runner.identity)
        runtime_path = output / "runtime_identity.json"
        if runtime_path.exists() and read_json(runtime_path) != runner.identity:
            raise PackageError("coverage resumed runtime differs from its original identity")
        if not runtime_path.exists():
            write_json(runtime_path, runner.identity)
        report_path = output / "preflight/preflight_report.json"
        if report_path.exists():
            digest = sha256_file(report_path)
            if result.get("preflight_report_sha256", digest) != digest:
                raise PackageError("coverage resumed preflight report changed")
            result["preflight_report_sha256"] = digest
            report = verified_preflight(plan, output, result)
        else:
            report = preflight(runner, plan, contexts, output / "preflight")
            result["preflight_report_sha256"] = sha256_file(report_path)
            report = verified_preflight(plan, output, result)
        if not report["passed"]:
            result.update(status="preflight_failed", failure=report["failure"])
        else:
            result["full_dev_started"] = True
            base.atomic_json(manifest_path, result)
            base.progress("coverage-preflight-passed-starting-dev", blocks=len(contexts), candidates=174896, batch_size=1)
            blocks, receipt = scoring.score_pass(runner, contexts, plan, output / RAW_PATH, batch_size=1)
            result.update(status="raw_complete", raw_blocks=receipt["blocks"],
                          raw_manifest_sha256=sha256_file(output / RAW_PATH / "manifest.json"))
            base.atomic_json(manifest_path, result)
            runner.close()
            runner = None
            analyze_run(plan, output, blocks=blocks)
            result.update(status="complete", analysis_published=True,
                          analysis_manifest_sha256=sha256_file(output / "analysis/manifest.json"))
        base.atomic_json(manifest_path, result)
        return result
    except BaseException as error:
        if execution_started:
            status = "raw_complete" if result.get("raw_manifest_sha256") else (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
            result.update(status=status, error_type=type(error).__name__, error=str(error), analysis_published=False)
            base.atomic_json(manifest_path, result)
        raise
    finally:
        if runner is not None:
            runner.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
