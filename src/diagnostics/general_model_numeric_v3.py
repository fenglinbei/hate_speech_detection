"""Registered whole-run batch-one fallback with inherited numerical tolerance."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_numeric as base
from diagnostics import general_model_numeric_v2 as parent
from diagnostics.general_model_numeric_analysis import analyze_blocks, candidate_scores
from diagnostics.general_model_package import ROOT, PackageError, read_json, read_jsonl, write_json


DEFAULT_CONFIG = ROOT / "config/stage1/general_model_ld_numeric_v3.json"
CODE_FILES = (
    "src/diagnostics/general_model_numeric_v3.py",
    "scripts/stage1/general_model_numeric_v3.py",
    "src/tests/test_general_model_numeric_v3.py",
)
CHALLENGES = ("padding", "prefix", "members", "replica")
NOT_APPLICABLE = ["batch-four", "tail-size-two", "within-batch-row-position"]
CALIBRATION_MODE = "inherited-source-failure-no-recalibration"
COHORT_STATUS = "previously-exposed-preregistered-fallback-revalidation"


def validate_config(config: dict, original: dict) -> None:
    contract = {
        "schema_version": "general-model-ld-numeric-fallback-config/v1",
        "production_batch_size": 1, "raw_path": "dev-b1", "device_indices": [0, 1, 2, 3],
        "auto_expand_on_pass": True, "stop_on_any_gate_failure": True,
        "automatic_profile_search": False, "test_access": False, "query_gold_in_scoring": False,
    }
    if any(type(config.get(key)) is not type(value) or config.get(key) != value
           for key, value in contract.items()):
        raise PackageError("registered batch-one fallback contract differs")
    validation = {
        "repeat_abs_tolerance": 1e-4, "reference_abs_tolerance": 1e-4,
        "inherit_epsilon": True, "epsilon_recalibration": False,
        "padding_extra": 64, "replica_shift": 1, "challenges": list(CHALLENGES),
    }
    if config.get("validation") != validation:
        raise PackageError("fallback validation cannot change the inherited numerical policy")
    allowed = set(contract) | {"validation", "parent_plan_ref", "source_failed_run", "protocol_path", "output_root"}
    if set(config) != allowed:
        raise PackageError("fallback config must not override scientific, input or runtime fields")
    for key in ("parent_plan_ref", "source_failed_run", "protocol_path", "output_root"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise PackageError(f"missing fallback path: {key}")
    fallback = original["execution"].get("fallback", {})
    if (fallback.get("batch_size") != 1 or fallback.get("new_run_required") is not True
            or fallback.get("automatic") is not False
            or fallback.get("mixed_batch_artifacts_allowed") is not False):
        raise PackageError("parent does not preregister a separate unmixed batch-one fallback")
    if (original["runtime"].get("dtype") != "float32"
            or original["runtime"].get("padding_policy") != "dynamic"
            or original["execution"].get("device_indices") != config["device_indices"]):
        raise PackageError("fallback must reuse the frozen parent numerical runtime")


def _inherited_sources(plan: dict) -> dict:
    result = _inherited_sources(plan["parent_plan"]) if "parent_plan" in plan else {}
    for name, digest in plan["code_sha256"].items():
        if name in result and result[name] != digest:
            raise PackageError("parent source identities disagree")
        result[name] = digest
    return result


def source_failure_binding(original: dict, directory: Path) -> dict:
    run = read_json(directory / "run_manifest.json")
    if (run.get("plan_id") != original["plan_id"] or run.get("status") != "preflight_failed"
            or run.get("failure") != "validation-batch-compatibility-gate"
            or run.get("full_dev_started") is not False or run.get("analysis_published") is not False
            or run.get("test_content_read") is not False
            or run.get("query_gold_loaded_during_scoring") is not False
            or (directory / "dev-b4").exists() or (directory / "analysis").exists()):
        raise PackageError("fallback source is not the sealed pre-development batch-four failure")
    report = parent._verified_preflight(original, directory, run)
    runtime = read_json(directory / "runtime_identity.json")
    regression = report.get("cohorts", {}).get("regression", {})
    validation = report.get("cohorts", {}).get("validation", {})
    policy = original["config"]["validation"]
    if (report.get("schema_version") != "general-model-ld-numeric-calibration/v2"
            or report.get("passed") is not False or report.get("complete") is not False
            or report.get("failure") != run["failure"] or report.get("runtime_identity") != runtime
            or report.get("error_families") != list(parent.CHALLENGES)
            or any(report.get(key) is not False for key in
                   ("query_gold_loaded", "test_content_read", "scientific_effect_checked", "formal_test_authorized"))
            or regression.get("complete") is not True or regression.get("passed") is not True
            or regression.get("baseline_passed") is not True
            or set(regression.get("challenges", {})) != set(parent.CHALLENGES)):
        raise PackageError("source regression or failed gate identity differs")
    for cohort in (regression, validation):
        for key, limit in (("baseline_repeat_max_abs_error", policy["repeat_abs_tolerance"]),
                           ("reference_max_abs_error", policy["reference_abs_tolerance"])):
            value = cohort.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= limit:
                raise PackageError("source baseline is not a successful registered arithmetic/repeat check")
    errors = [row["max_abs_error"] for row in regression["challenges"].values()]
    if (not all(math.isfinite(value) and 0 <= value <= policy["calibration_max_abs_error"] for value in errors)
            or not all(row.get("passed") is True for row in regression["challenges"].values())):
        raise PackageError("source regression calibration failed")
    e8 = max(errors)
    epsilon = min(policy["epsilon_ceiling"], max(policy["epsilon_floor"], policy["epsilon_multiplier"] * e8))
    failed = validation.get("challenges", {}).get("batch", {})
    if (report.get("E8") != e8 or regression.get("max_abs_error") != e8 or report.get("epsilon") != epsilon
            or validation.get("baseline_passed") is not True
            or failed.get("passed") is not False or not math.isfinite(failed.get("max_abs_error", math.nan))
            or failed["max_abs_error"] <= epsilon):
        raise PackageError("source epsilon formula or batch-four failure differs")
    names = {"run_manifest.json", "runtime_identity.json", "preflight/preflight_report.json"}
    names.update("preflight/" + name for name in report["files"])
    return {"E8": e8, "epsilon": epsilon, "calibration_sha256": canonical_json_sha256(regression),
            "runtime_identity_sha256": canonical_json_sha256(runtime),
            "failure": run["failure"], "hashes": {name: sha256_file(directory / name) for name in sorted(names)}}


def build_plan(config_path: Path = DEFAULT_CONFIG, *, root: Path = ROOT) -> dict:
    config = read_json(config_path)
    reference = parent._parent_ref(Path(config["parent_plan_ref"]), root).resolve()
    original, _ = parent.load_plan(reference, root=root)
    validate_config(config, original["config"])
    failed = parent._parent_ref(Path(config["source_failed_run"]), root).resolve()
    plan = {
        "schema_version": "general-model-ld-numeric-plan/v3", "fallback_config": config,
        "fallback_config_sha256": canonical_json_sha256(config),
        "parent_plan_ref": str(reference), "parent_ref_sha256": sha256_file(reference),
        "parent_plan_id": original["plan_id"], "parent_plan_sha256": canonical_json_sha256(original),
        "source_failed_run": str(failed), "source_failure": source_failure_binding(original, failed),
        "code_sha256": {name: sha256_file(root / name) for name in CODE_FILES},
        "inherited_code_sha256": _inherited_sources(original),
        "protocol_sha256": sha256_file(root / config["protocol_path"]),
        "scientific_scope_sha256": canonical_json_sha256(original["config"]),
        "query_gold_loaded": False, "test_content_read": False,
    }
    plan["plan_id"] = "gmlnum3-" + canonical_json_sha256(plan)
    output = root / config["output_root"]
    plans = output / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    destination = plans / plan["plan_id"]
    if destination.exists():
        if read_json(destination / "plan.json") != plan:
            raise PackageError("existing fallback plan differs")
    else:
        staging = Path(tempfile.mkdtemp(prefix=".building-", dir=plans))
        try:
            write_json(staging / "plan.json", plan)
            write_json(staging / "fallback_config.json", config)
            (staging / "protocol.md").write_bytes((root / config["protocol_path"]).read_bytes())
            (staging / "source_preflight_report.json").write_bytes((failed / "preflight/preflight_report.json").read_bytes())
            for name in {**plan["inherited_code_sha256"], **plan["code_sha256"]}:
                target = staging / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((root / name).read_bytes())
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    ref = {"plan_id": plan["plan_id"], "target_path": str(destination), "plan_sha256": sha256_file(destination / "plan.json")}
    base.atomic_json(output / "plan_ref.json", ref)
    base.progress("registered-batch-one-fallback-plan-frozen", **ref)
    return ref


def load_plan(path: Path, *, root: Path = ROOT) -> tuple[dict, list[dict]]:
    if path.is_dir():
        directory = path
    else:
        ref = read_json(path)
        directory = Path(ref["target_path"])
        if sha256_file(directory / "plan.json") != ref["plan_sha256"]:
            raise PackageError("fallback plan reference differs")
    plan = read_json(directory / "plan.json")
    expected = "gmlnum3-" + canonical_json_sha256({k: v for k, v in plan.items() if k != "plan_id"})
    if plan.get("schema_version") != "general-model-ld-numeric-plan/v3" or plan.get("plan_id") != expected:
        raise PackageError("fallback plan identity differs")
    reference = Path(plan["parent_plan_ref"])
    if sha256_file(reference) != plan["parent_ref_sha256"]:
        raise PackageError("fallback parent reference changed")
    original, contexts = parent.load_plan(reference, root=root)
    if original["plan_id"] != plan["parent_plan_id"] or canonical_json_sha256(original) != plan["parent_plan_sha256"]:
        raise PackageError("fallback parent plan changed")
    validate_config(plan["fallback_config"], original["config"])
    if (canonical_json_sha256(original["config"]) != plan["scientific_scope_sha256"]
            or plan["inherited_code_sha256"] != _inherited_sources(original)
            or set(plan["code_sha256"]) != set(CODE_FILES)
            or plan.get("query_gold_loaded") is not False or plan.get("test_content_read") is not False):
        raise PackageError("fallback scientific or source inheritance differs")
    sources = {**plan["inherited_code_sha256"], **plan["code_sha256"]}
    for name, digest in sources.items():
        if sha256_file(root / name) != digest or sha256_file(directory / "source" / name) != digest:
            raise PackageError(f"fallback source snapshot differs: {name}")
    config = plan["fallback_config"]
    if (sha256_file(directory / "protocol.md") != plan["protocol_sha256"]
            or read_json(directory / "fallback_config.json") != config
            or canonical_json_sha256(config) != plan["fallback_config_sha256"]
            or parent._parent_ref(Path(config["parent_plan_ref"]), root).resolve() != reference.resolve()
            or parent._parent_ref(Path(config["source_failed_run"]), root).resolve() != Path(plan["source_failed_run"]).resolve()):
        raise PackageError("fallback protocol or input snapshot differs")
    failed = source_failure_binding(original, Path(plan["source_failed_run"]))
    if (failed != plan["source_failure"] or sha256_file(directory / "source_preflight_report.json")
            != failed["hashes"]["preflight/preflight_report.json"]):
        raise PackageError("sealed source failure or inherited calibration changed")
    return {**original, **plan, "config": original["config"], "parent_plan": original, "code_sha256": sources}, contexts


def validate_geometry(rows: list[dict], contexts: list[dict], plan: dict, *, reference=False,
                      prefix=False, padding_extra=0, permuted=False, replica_shift=0) -> dict:
    profile = parent.scoring_profile(prefix=prefix, padding_extra=padding_extra, permuted=permuted, replica_shift=replica_shift)
    if len(rows) != len(contexts):
        raise PackageError("fallback geometry block coverage differs")
    count = 0
    eos = plan["eos_token_id"]
    if type(eos) is not int:
        raise PackageError("fallback requires the inherited unambiguous EOS identity")
    for row, context in zip(rows, contexts, strict=True):
        base.validate_block(row, context, plan["catalog"])
        if row.get("execution_batch_size") != 1 or row.get("scoring_profile") != profile or row.get("plan_id") != plan["plan_id"]:
            raise PackageError("fallback block execution identity is not the registered batch-one profile")
        catalog = plan["catalog"][context["task"]]
        canonical_ids = [candidate["candidate_id"] for candidate in catalog]
        members = [f"{context['record_id']}:{cid}" for cid in canonical_ids]
        prefixes = {tuple(candidate["answer_token_ids"][:i]) for candidate in catalog
                    for i in range(candidate["answer_tokens"] + 1)}
        for index, (candidate, wanted) in enumerate(zip(row["candidates"], catalog, strict=True)):
            expected = {
                "batch_size": 1, "effective_batch_size": 1, "prefix_reference": prefix,
                "batch_member_ordinal": index if prefix else 0,
                "batch_members": members if prefix else [members[index]],
                "prompt_tokens": context["prompt_tokens"],
                "prompt_token_ids_sha256": context["prompt_token_ids_sha256"],
                "answer_token_ids": wanted["answer_token_ids"],
                "answer_token_ids_sha256": wanted["answer_token_ids_sha256"],
                "answer_tokens": wanted["answer_tokens"],
                "eos_token_id": eos,
                "sequence_tokens": context["prompt_tokens"] + wanted["answer_tokens"] + 1,
                "causal_shift": 1, "use_cache": False, "padding_side": "right",
                "model_logits_dtype": "torch.float32", "logprob_arithmetic_dtype": "torch.float32",
                "finite_target_logits_checked": True, "token_boundary_checked": True,
                "reference_checked": reference,
                "scoring_implementation": "uncached-prefix-only" if prefix else "full-sequence-selected-projection",
            }
            if prefix:
                expected.update(padded_sequence_tokens=None, prefix_padding=False,
                                prefix_unique_forward_count=len(prefixes), batch_ordinal=0)
            else:
                expected.update(padded_sequence_tokens=expected["sequence_tokens"] + padding_extra,
                                padding_challenge_extra=padding_extra)
            if any(key not in candidate or candidate[key] != value for key, value in expected.items()):
                raise PackageError("candidate metadata does not prove the registered batch-one forward geometry")
            if (type(candidate.get("batch_ordinal")) is not int or candidate["batch_ordinal"] < 0
                    or type(candidate.get("eos_token_id")) is not int or candidate["eos_token_id"] < 0
                    or candidate["eos_token_id"] in candidate["answer_token_ids"]):
                raise PackageError("candidate execution ordinal or EOS identity is invalid")
            if reference and (candidate.get("reference_arithmetic_dtype") != "cpu.torch.float64"
                              or "reference_scores" not in candidate):
                raise PackageError("baseline lacks the registered same-logits arithmetic reference")
            count += 1
        if not prefix:
            actual = sorted(row["candidates"], key=lambda candidate: candidate["batch_ordinal"])
            ordinals = [candidate["batch_ordinal"] for candidate in actual]
            order = list(reversed(canonical_ids if context["task"] == "hate" else canonical_ids[1:] + canonical_ids[:1])) if permuted else canonical_ids
            if (ordinals != list(range(ordinals[0], ordinals[0] + len(ordinals)))
                    or [candidate["candidate_id"] for candidate in actual] != order):
                raise PackageError("candidate batch ordinals do not prove the registered execution order")
    return {"passed": True, "blocks": len(rows), "candidates": count, "scoring_profile": profile,
            "true_batch_one": True, "prefix_is_reference_only": prefix,
            "execution_order_verified": not prefix, "within_batch_row_position_claimed": False}


def preflight(runner, plan: dict, contexts: list[dict], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    policy = plan["fallback_config"]["validation"]
    source = plan["source_failure"]
    if canonical_json_sha256(runner.identity) != source["runtime_identity_sha256"]:
        raise PackageError("fallback runtime differs from the actual source calibration runtime")
    report = {
        "schema_version": "general-model-ld-numeric-calibration/v3", "plan_id": plan["plan_id"],
        "runtime_identity": runner.identity, "passed": False, "cohorts": {},
        "E8": source["E8"], "epsilon": source["epsilon"], "calibration_mode": CALIBRATION_MODE,
        "source_calibration_sha256": source["calibration_sha256"], "source_failed_run": plan["source_failed_run"],
        "validation_cohort_status": COHORT_STATUS, "error_families": list(CHALLENGES),
        "not_applicable": NOT_APPLICABLE, "query_gold_loaded": False, "test_content_read": False,
        "scientific_effect_checked": False, "formal_test_authorized": False,
    }
    definitions = (
        ("padding", {"padding_extra": policy["padding_extra"]}),
        ("prefix", {"prefix": True}),
        ("members", {"permuted": True}),
        ("replica", {"replica_shift": policy["replica_shift"]}),
    )
    for cohort in ("regression", "validation"):
        ids = set(plan["cohorts"][cohort])
        selected = [row for row in contexts if row["query_id"] in ids]
        expected = {f"{qid}:{task}:{condition}" for qid in ids for task in plan["config"]["tasks"]
                    for condition in plan["config"]["conditions"]}
        if (len(ids) != plan["config"]["validation"][f"{cohort}_query_count"]
                or len(plan["cohorts"][cohort]) != len(ids)
                or set(plan["cohorts"]["regression"]) & set(plan["cohorts"]["validation"])
                or len(selected) != len(expected) or {row["record_id"] for row in selected} != expected):
            raise PackageError("fallback cohort matrix is incomplete or overlapping")
        baseline, _ = parent.score_pass(runner, selected, plan, output / f"{cohort}-b1-r0", batch_size=1, reference=True)
        proof = validate_geometry(baseline, selected, plan, reference=True)
        write_json(output / f"{cohort}-baseline-geometry-proof.json", proof)
        repeated, _ = parent.score_pass(runner, selected, plan, output / f"{cohort}-b1-r1", batch_size=1)
        write_json(output / f"{cohort}-repeat-geometry-proof.json", validate_geometry(repeated, selected, plan))
        repeat = base.compare_passes(baseline, repeated)
        write_json(output / f"{cohort}-repeat-differences.json", repeat)
        reference_rows = copy.deepcopy(baseline)
        for row in reference_rows:
            for candidate in row["candidates"]:
                candidate.update(candidate["reference_scores"])
                candidate["scores"] = candidate_scores(candidate["token_logprobs"], candidate["eos_logprob"])
        arithmetic = base.compare_passes(reference_rows, baseline)
        write_json(output / f"{cohort}-reference-differences.json", arithmetic)
        info = {"blocks": len(selected), "baseline_repeat_max_abs_error": repeat["max_abs_error"],
                "reference_max_abs_error": arithmetic["max_abs_error"], "challenges": {},
                "baseline_passed": repeat["max_abs_error"] <= policy["repeat_abs_tolerance"]
                    and arithmetic["max_abs_error"] <= policy["reference_abs_tolerance"]}
        report["cohorts"][cohort] = info
        if not info["baseline_passed"]:
            report["failure"] = f"{cohort}-baseline-gate"
            break
        for name, options in definitions:
            rows, _ = parent.score_pass(runner, selected, plan, output / f"{cohort}-b1-{name}", batch_size=1, **options)
            write_json(output / f"{cohort}-{name}-geometry-proof.json", validate_geometry(rows, selected, plan, **options))
            if name == "replica":
                write_json(output / f"{cohort}-replica-producer-proof.json", parent.replica_proof(baseline, rows))
            comparison = base.compare_passes(baseline, rows)
            write_json(output / f"{cohort}-{name}-differences.json", comparison)
            info["challenges"][name] = {"max_abs_error": comparison["max_abs_error"],
                                        "largest_error": comparison["largest_error"],
                                        "passed": comparison["max_abs_error"] <= source["epsilon"]}
            if not info["challenges"][name]["passed"]:
                report["failure"] = f"{cohort}-{name}-compatibility-gate"
                break
        info["observed_fallback_max"] = max((row["max_abs_error"] for row in info["challenges"].values()), default=0.0)
        info["complete"] = set(info["challenges"]) == set(CHALLENGES)
        info["passed"] = info["complete"] and all(row["passed"] for row in info["challenges"].values())
        if not info["passed"]:
            break
    report["observed_fallback_max"] = max((row.get("observed_fallback_max", 0.0) for row in report["cohorts"].values()), default=0.0)
    report["complete"] = len(report["cohorts"]) == 2 and all(row.get("complete", False) for row in report["cohorts"].values())
    report["passed"] = report["complete"] and all(row["baseline_passed"] and row["passed"] for row in report["cohorts"].values())
    report["validation_executed"] = "validation" in report["cohorts"]
    report["files"] = {path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file()
        and (path.name in {"scores.jsonl", "manifest.json", "pool_binding.json"}
             or path.name.endswith(("-differences.json", "-proof.json")))}
    base.atomic_json(output / "preflight_report.json", report)
    base.progress("numerical-v3-fallback-preflight-finished", passed=report["passed"], failure=report.get("failure"),
                  E8=report["E8"], epsilon=report["epsilon"], observed_fallback_max=report["observed_fallback_max"])
    return report


def _verified_preflight(plan: dict, output: Path, run: dict) -> dict:
    report = parent._verified_preflight(plan, output, run)
    source = plan["source_failure"]
    expected = {
        "schema_version": "general-model-ld-numeric-calibration/v3", "E8": source["E8"],
        "epsilon": source["epsilon"], "source_calibration_sha256": source["calibration_sha256"],
        "source_failed_run": plan["source_failed_run"], "calibration_mode": CALIBRATION_MODE,
        "validation_cohort_status": COHORT_STATUS, "error_families": list(CHALLENGES),
        "not_applicable": NOT_APPLICABLE,
        "query_gold_loaded": False, "test_content_read": False, "scientific_effect_checked": False,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise PackageError("fallback preflight changed the inherited policy or data boundary")
    if canonical_json_sha256(report["runtime_identity"]) != source["runtime_identity_sha256"]:
        raise PackageError("fallback gate no longer matches the source calibration runtime")
    if report["passed"]:
        if report.get("complete") is not True or set(report["cohorts"]) != {"regression", "validation"}:
            raise PackageError("passed fallback report is incomplete")
        for cohort, info in report["cohorts"].items():
            if (info.get("blocks") != len(plan["cohorts"][cohort]) * 12
                    or info.get("complete") is not True or info.get("baseline_passed") is not True
                    or info.get("passed") is not True or set(info.get("challenges", {})) != set(CHALLENGES)):
                raise PackageError("passed fallback cohort is incomplete")
            for key, limit in (("baseline_repeat_max_abs_error", 1e-4), ("reference_max_abs_error", 1e-4)):
                value = info.get(key, math.nan)
                if not math.isfinite(value) or not 0 <= value <= limit:
                    raise PackageError("passed fallback arithmetic or repeat exceeded its limit")
            for check in info["challenges"].values():
                value = check.get("max_abs_error", math.nan)
                if check.get("passed") is not True or not math.isfinite(value) or not 0 <= value <= source["epsilon"]:
                    raise PackageError("passed fallback challenge exceeded the inherited epsilon")
        _verify_passed_evidence(plan, output / "preflight", report)
    return report


def _verify_passed_evidence(plan: dict, directory: Path, report: dict) -> None:
    from diagnostics.general_model_numeric_pool import validate_sealed_pass

    actual_files = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
        if path.is_file() and (path.name in {"scores.jsonl", "manifest.json", "pool_binding.json"}
        or path.name.endswith(("-differences.json", "-proof.json")))}
    if set(report["files"]) != actual_files:
        raise PackageError("fallback preflight sealed evidence inventory differs")
    definitions = (
        ("r0", {"reference": True}, "baseline"), ("r1", {}, "repeat"),
        ("padding", {"padding_extra": 64}, "padding"), ("prefix", {"prefix": True}, "prefix"),
        ("members", {"permuted": True}, "members"), ("replica", {"replica_shift": 1}, "replica"),
    )
    required = set()
    for cohort in ("regression", "validation"):
        for suffix, _, label in definitions:
            required.update(f"{cohort}-b1-{suffix}/{name}" for name in ("scores.jsonl", "manifest.json", "pool_binding.json"))
            required.add(f"{cohort}-{label}-geometry-proof.json")
        required.update(f"{cohort}-{name}-differences.json" for name in ("repeat", "reference", *CHALLENGES))
        required.add(f"{cohort}-replica-producer-proof.json")
    if not required <= set(report["files"]):
        raise PackageError("passed fallback report lacks required twelve-pass evidence")
    for cohort in ("regression", "validation"):
        ids = set(plan["cohorts"][cohort])
        contexts = [row for row in plan["blocks"] if row["query_id"] in ids]
        baseline = None
        for suffix, options, label in definitions:
            path = directory / f"{cohort}-b1-{suffix}"
            receipt = validate_sealed_pass(path, plan)
            expected = {"plan_id": plan["plan_id"], "runtime": report["runtime_identity"], "batch_size": 1,
                        "reference": options.get("reference", False), "pass_name": path.name,
                        "records": [row["record_id"] for row in contexts],
                        "scoring_profile": parent.scoring_profile(**{k: v for k, v in options.items() if k != "reference"})}
            if (any(receipt.get("identity", {}).get(key) != value for key, value in expected.items())
                    or receipt.get("execution") != "data-parallel-identical-fp32"
                    or any(receipt.get(key) is not False for key in
                           ("query_gold_loaded", "test_content_read", "mixed_execution_modes"))):
                raise PackageError("fallback preflight pass identity or data boundary differs")
            rows = read_jsonl(path / "scores.jsonl")
            proof = validate_geometry(rows, contexts, plan, **options)
            if proof != read_json(directory / f"{cohort}-{label}-geometry-proof.json"):
                raise PackageError("fallback preflight geometry proof differs")
            if suffix == "r0":
                baseline = rows
                reference_rows = copy.deepcopy(baseline)
                for row in reference_rows:
                    for candidate in row["candidates"]:
                        candidate.update(candidate["reference_scores"])
                        candidate["scores"] = candidate_scores(candidate["token_logprobs"], candidate["eos_logprob"])
                comparison = base.compare_passes(reference_rows, baseline)
                comparison_name = "reference"
                reported = report["cohorts"][cohort]["reference_max_abs_error"]
            else:
                comparison = base.compare_passes(baseline, rows)
                comparison_name = "repeat" if suffix == "r1" else suffix
                reported = (report["cohorts"][cohort]["baseline_repeat_max_abs_error"] if suffix == "r1"
                            else report["cohorts"][cohort]["challenges"][suffix]["max_abs_error"])
            if (comparison != read_json(directory / f"{cohort}-{comparison_name}-differences.json")
                    or comparison["max_abs_error"] != reported):
                raise PackageError("fallback preflight readout evidence disagrees with the passed gate")
            if suffix == "replica" and parent.replica_proof(baseline, rows) != read_json(directory / f"{cohort}-replica-producer-proof.json"):
                raise PackageError("fallback physical replica proof differs")


def validate_raw_identity(plan: dict, report: dict, receipt: dict) -> None:
    expected = {"plan_id": plan["plan_id"], "runtime": report["runtime_identity"], "batch_size": 1,
                "reference": False, "pass_name": "dev-b1", "scoring_profile": parent.scoring_profile(),
                "records": [row["record_id"] for row in plan["blocks"]]}
    if any(receipt.get("identity", {}).get(key) != value for key, value in expected.items()):
        raise PackageError("development raw differs from the validated true batch-one runtime or profile")
    if (receipt.get("execution") != "data-parallel-identical-fp32"
            or any(receipt.get(key) is not False for key in ("query_gold_loaded", "test_content_read", "mixed_execution_modes"))):
        raise PackageError("fallback raw data or execution boundary differs")


def analyze_run(plan: dict, output: Path, *, blocks=None) -> dict:
    from diagnostics.general_model_numeric_pool import validate_sealed_pass

    run = read_json(output / "run_manifest.json")
    if run.get("status") not in {"raw_complete", "complete"}:
        raise PackageError("fallback analysis requires a sealed complete raw run")
    report = _verified_preflight(plan, output, run)
    if not report["passed"]:
        raise PackageError("analysis requires a passed batch-one fallback preflight")
    raw = output / "dev-b1"
    receipt = read_json(raw / "manifest.json")
    if (receipt.get("status") != "complete" or receipt.get("blocks") != 7716 or receipt.get("candidates") != 131172
            or sha256_file(raw / "manifest.json") != run["raw_manifest_sha256"]
            or sha256_file(raw / "scores.jsonl") != receipt["scores_sha256"]):
        raise PackageError("full development fallback raw is incomplete or changed")
    validate_raw_identity(plan, report, receipt)
    validate_sealed_pass(raw, plan)
    stored = read_jsonl(raw / "scores.jsonl")
    if blocks is not None and stored != blocks:
        raise PackageError("analysis input differs from sealed fallback raw")
    geometry = validate_geometry(stored, plan["blocks"], plan)
    directory = output / "analysis"
    if directory.exists():
        existing = read_json(directory / "manifest.json")
        if (existing.get("plan_id") != plan["plan_id"]
                or existing.get("raw_manifest_sha256") != run["raw_manifest_sha256"]
                or existing.get("gold_join_after_raw_sealed") is not True or existing.get("test_content_read") is not False
                or sha256_file(directory / "analysis.json") != existing["analysis_sha256"]):
            raise PackageError("existing fallback analysis identity differs")
        return existing
    package = Path(plan["package_path"])
    package_manifest = read_json(package / "manifest.json")
    gold_entry = next(row for row in package_manifest["files"] if row["path"] == "queries.dev.jsonl")
    if (sha256_file(package / "manifest.json") != plan["package_manifest_sha256"]
            or sha256_file(package / "queries.dev.jsonl") != gold_entry["sha256"]):
        raise PackageError("gold source changed after fallback scoring")
    gold = {str(row["id"]): row["projection"] for row in read_jsonl(package / "queries.dev.jsonl")}
    bootstrap = plan["config"]["analysis"]["bootstrap"]
    analysis = analyze_blocks(stored, frame=plan["frame"], gold_by_query=gold, epsilon=report["epsilon"],
                              bootstrap_replicates=bootstrap["repetitions"], bootstrap_seed=bootstrap["seed"])
    staging = Path(tempfile.mkdtemp(prefix=".analyzing-", dir=output))
    try:
        write_json(staging / "analysis.json", analysis)
        manifest = {"schema_version": "general-model-ld-numeric-analysis/v3", "plan_id": plan["plan_id"],
                    "raw_manifest_sha256": run["raw_manifest_sha256"], "gold_join_after_raw_sealed": True,
                    "test_content_read": False, "production_geometry": geometry,
                    "analysis_sha256": sha256_file(staging / "analysis.json")}
        write_json(staging / "manifest.json", manifest)
        os.rename(staging, directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def run_pipeline(plan_path: Path, output: Path, *, device="cuda:0", root: Path = ROOT) -> dict:
    from diagnostics.general_model_numeric_pool import PersistentNumericPool

    plan, contexts = load_plan(plan_path, root=root)
    if device != "cuda:0":
        raise PackageError("physical fallback devices are fixed by the registered configuration")
    if output.resolve().is_relative_to(Path(plan["source_failed_run"]).resolve()):
        raise PackageError("fallback requires a new run; source failure must remain unchanged")
    devices = plan["fallback_config"]["device_indices"]
    output.mkdir(parents=True, exist_ok=True)
    binding = {"plan_id": plan["plan_id"], "physical_device_indices": devices,
               "execution": "data-parallel-identical-fp32", "phase": "registered-batch-one-fallback",
               "raw_path": "dev-b1", "production_batch_size": 1, "source_failed_run": plan["source_failed_run"],
               "source_calibration_sha256": plan["source_failure"]["calibration_sha256"]}
    binding_path = output / "binding.json"
    if binding_path.exists() and read_json(binding_path) != binding:
        raise PackageError("fallback run directory belongs to a different execution")
    lock = (output / ".writer.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runner = None
    execution_started = False
    manifest_path = output / "run_manifest.json"
    result = {**binding, "schema_version": "general-model-ld-numeric-run/v3", "status": "running",
              "test_content_read": False, "query_gold_loaded_during_scoring": False,
              "automatic_profile_search": False}
    try:
        if not binding_path.exists():
            if manifest_path.exists() or (output / "preflight").exists() or (output / "dev-b1").exists():
                raise PackageError("refusing an unbound existing fallback artifact")
            base.atomic_json(binding_path, binding)
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if any(previous.get(key) != value for key, value in binding.items()):
                raise PackageError("fallback run manifest binding differs")
            if previous.get("status") in {"complete", "preflight_failed"}:
                report = _verified_preflight(plan, output, previous)
                if (previous["status"] == "complete") != report["passed"]:
                    raise PackageError("terminal fallback execution disagrees with its preflight")
                if previous["status"] == "complete":
                    analyze_run(plan, output)
                return previous
            if previous.get("status") == "raw_complete":
                result = previous
                execution_started = True
                analyze_run(plan, output)
                result.update(status="complete", analysis_published=True)
                base.atomic_json(manifest_path, result)
                return result
            if previous.get("status") == "failed":
                raise PackageError("failed fallback is sealed; no automatic retry or profile search is permitted")
        execution_started = True
        base.atomic_json(manifest_path, result)
        runner = PersistentNumericPool(plan["parent_plan"], plan["config"]["runtime"], devices, root)
        write_json(output / "runtime_identity.json", runner.identity)
        report = preflight(runner, plan, contexts, output / "preflight")
        result["preflight_report_sha256"] = sha256_file(output / "preflight/preflight_report.json")
        if report["passed"]:
            report = _verified_preflight(plan, output, result)
        if not report["passed"]:
            result.update(status="preflight_failed", failure=report.get("failure"), full_dev_started=False,
                          analysis_published=False, fallback_automatic=False)
        else:
            base.progress("numerical-v3-fallback-preflight-passed-starting-dev", blocks=7716, candidates=131172, batch_size=1)
            blocks, receipt = parent.score_pass(runner, contexts, plan, output / "dev-b1", batch_size=1)
            result.update(status="raw_complete", full_dev_started=True, raw_blocks=receipt["blocks"],
                          raw_manifest_sha256=sha256_file(output / "dev-b1/manifest.json"))
            base.atomic_json(manifest_path, result)
            runner.close()
            runner = None
            analyze_run(plan, output, blocks=blocks)
            result.update(status="complete", analysis_published=True)
        base.atomic_json(manifest_path, result)
        return result
    except BaseException as error:
        if not execution_started:
            raise
        status = "raw_complete" if result.get("raw_manifest_sha256") else (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        result.update(status=status, error_type=type(error).__name__, error=str(error), analysis_published=False)
        base.atomic_json(manifest_path, result)
        if not (output / "preflight/preflight_report.json").exists():
            directory = output / "preflight"
            directory.mkdir(exist_ok=True)
            base.atomic_json(directory / "failure.json", {"plan_id": plan["plan_id"], "passed": False,
                "error_type": type(error).__name__, "error": str(error), "full_dev_started": False})
        raise
    finally:
        if runner is not None:
            runner.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("validate", "run", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        if name != "validate":
            command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "build-plan":
        result = build_plan(args.config)
    elif args.command == "run":
        result = run_pipeline(args.plan, args.output, device=args.device)
    else:
        plan, contexts = load_plan(args.plan)
        result = analyze_run(plan, args.output) if args.command == "analyze" else {
            "plan_id": plan["plan_id"], "valid": True, "blocks": len(contexts), "query_gold_loaded": False,
            "production_batch_size": 1, "E8": plan["source_failure"]["E8"], "epsilon": plan["source_failure"]["epsilon"]}
    print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
    return 2 if result.get("status") == "preflight_failed" else 0
