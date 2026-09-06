"""Versioned numerical execution with independent shape and prefix checks."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from data.stage1_data import canonical_json_bytes, canonical_json_sha256, sha256_file
from diagnostics import general_model_numeric as parent
from diagnostics.general_model_numeric_analysis import analyze_blocks, candidate_scores
from diagnostics.general_model_package import ROOT, PackageError, read_json, read_jsonl, write_json, write_jsonl


DEFAULT_CONFIG = ROOT / "config/stage1/general_model_ld_numeric_v2.json"
CODE_FILES = (
    "src/diagnostics/general_model_numeric_v2.py",
    "src/diagnostics/general_model_numeric_kernel_v2.py",
    "scripts/stage1/general_model_numeric_v2.py",
    "src/tests/test_general_model_numeric_v2.py",
    "src/tests/test_general_model_numeric_kernel_v2.py",
    "src/diagnostics/general_model_numeric_pool.py",
    "src/tests/test_general_model_numeric_pool.py",
)
CHALLENGES = ("batch", "tail", "padding", "prefix", "members", "replica")


def validate_config(config: dict, original: dict) -> None:
    if config.get("schema_version") != "general-model-ld-numeric-config/v2":
        raise PackageError("unknown numerical execution amendment")
    for key in ("model_key", "tasks", "conditions", "candidate_order", "analysis",
                "test_access", "expected_package_id", "package_ref", "generation_preflight"):
        if config.get(key) != original.get(key):
            raise PackageError(f"scientific or input scope differs from parent: {key}")
    runtime = config["runtime"]
    for key in ("seed", "max_sequence_tokens", "baseline_batch_size", "accelerated_batch_size",
                "enable_thinking", "cpu_threads", "local_files_only", "trust_remote_code",
                "padding_side", "main_score", "auxiliary_scores", "overflow_policy", "batch_geometry"):
        if runtime.get(key) != original["runtime"].get(key):
            raise PackageError(f"registered numerical runtime invariant differs: {key}")
    if runtime.get("use_cache") is not False:
        raise PackageError("v2 scoring requires an uncached reference and forward")
    runtime_contract = {
        "dtype": "float32", "head_float32": True, "bf16_reduced_precision_reduction": False,
        "attention_implementation": "eager", "padding_policy": "dynamic",
        "score_mode": "full-sequence-teacher-forcing-selected-output-projection",
        "projection": "all-answer-and-eos-prediction-positions-only",
        "parameter_conversion": "verified-bfloat16-checkpoint-values-exactly-widened-to-float32",
    }
    if any(runtime.get(key) != value for key, value in runtime_contract.items()):
        raise PackageError("v2 forward precision or projection contract differs")
    policy = config["validation"]
    for key in ("regression_query_count", "validation_query_count", "baseline_repetitions",
                "accelerated_repetitions", "repeat_abs_tolerance", "reference_abs_tolerance",
                "calibration_max_abs_error", "epsilon_floor", "epsilon_multiplier",
                "epsilon_ceiling", "required_coverage"):
        if policy.get(key) != original["validation"].get(key):
            raise PackageError(f"registered numerical validation invariant differs: {key}")
    if policy.get("padding_challenge_extra") != 64:
        raise PackageError("v2 padding challenge must add 64 tokens")
    validation_contract = {
        "reference": "cpu-float64-from-identical-float32-forward-logits",
        "prefix_reference": "all-candidates-all-answer-and-eos-positions-unique-prefixes-uncached-unpadded-batch-one",
        "member_challenge": "group-rotate-one-then-reverse-hate-reverse-before-batching-restore-canonical-order-before-commit",
        "replica_challenge_shift": 1,
        "replica_comparison": "batch-one-and-identical-batch-four-on-another-physical-device",
    }
    if any(policy.get(key) != value for key, value in validation_contract.items()):
        raise PackageError("v2 numerical reference contract differs")
    for key in ("error_scope", "tail_batch", "frame_source", "cohorts_must_be_disjoint",
                "scientific_success_requires_positive_effect", "accuracy_or_generation_agreement_is_gate"):
        if policy.get(key) != original["validation"].get(key):
            raise PackageError(f"registered validation scope differs: {key}")
    execution = config["execution"]
    for key in ("expected_dev_queries", "expected_dev_blocks", "expected_dev_candidates",
                "auto_expand_on_pass", "stop_on_any_gate_failure"):
        if execution.get(key) != original["execution"].get(key):
            raise PackageError(f"registered execution invariant differs: {key}")
    if execution.get("automatic_dev_batch_size") != 4 or execution.get("raw_path", "dev-b4") != "dev-b4":
        raise PackageError("v2 requires a separately validated batch-four development run")
    if execution.get("device_indices") != [0, 1, 2, 3]:
        raise PackageError("v2 numerical replica device inventory differs")


def _parent_ref(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_plan(config_path: Path = DEFAULT_CONFIG, *, root: Path = ROOT) -> dict:
    config = read_json(config_path)
    reference = _parent_ref(Path(config["parent_plan_ref"]), root)
    original, _ = parent.load_plan(reference, root=root)
    validate_config(config, original["config"])
    plan = {
        "schema_version": "general-model-ld-numeric-plan/v2", "config": config,
        "parent_plan_ref": str(reference.resolve()), "parent_ref_sha256": sha256_file(reference),
        "parent_plan_id": original["plan_id"],
        "parent_plan_sha256": canonical_json_sha256(original),
        "code_sha256": {name: sha256_file(root / name) for name in CODE_FILES},
        "protocol_sha256": sha256_file(root / config["protocol_path"]),
        "scientific_scope_sha256": canonical_json_sha256({key: original["config"][key]
            for key in ("model_key", "tasks", "conditions", "candidate_order", "analysis")}),
        "query_gold_loaded": False, "test_content_read": False,
    }
    plan["plan_id"] = "gmlnum2-" + canonical_json_sha256(plan)
    output = root / config["output_root"]
    plans = output / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    destination = plans / plan["plan_id"]
    if destination.exists():
        if read_json(destination / "plan.json") != plan:
            raise PackageError("existing amended plan differs")
    else:
        staging = Path(tempfile.mkdtemp(prefix=".building-", dir=plans))
        try:
            write_json(staging / "plan.json", plan)
            (staging / "protocol.md").write_bytes((root / config["protocol_path"]).read_bytes())
            for name in CODE_FILES:
                target = staging / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((root / name).read_bytes())
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    ref = {"plan_id": plan["plan_id"], "target_path": str(destination),
           "plan_sha256": sha256_file(destination / "plan.json")}
    parent.atomic_json(output / "plan_ref.json", ref)
    parent.progress("amended-numerical-plan-frozen", **ref)
    return ref


def load_plan(path: Path, *, root: Path = ROOT) -> tuple[dict, list[dict]]:
    if path.is_dir():
        directory = path
    else:
        ref = read_json(path)
        directory = Path(ref["target_path"])
        if sha256_file(directory / "plan.json") != ref["plan_sha256"]:
            raise PackageError("amended plan reference differs")
    plan = read_json(directory / "plan.json")
    expected = "gmlnum2-" + canonical_json_sha256({k: v for k, v in plan.items() if k != "plan_id"})
    if plan.get("schema_version") != "general-model-ld-numeric-plan/v2" or plan["plan_id"] != expected:
        raise PackageError("amended plan identity differs")
    if set(plan["code_sha256"]) != set(CODE_FILES):
        raise PackageError("amended source inventory differs")
    for name, digest in plan["code_sha256"].items():
        if sha256_file(root / name) != digest or sha256_file(directory / "source" / name) != digest:
            raise PackageError(f"amended numerical source differs: {name}")
    if sha256_file(directory / "protocol.md") != plan["protocol_sha256"]:
        raise PackageError("amended protocol snapshot differs")
    reference = Path(plan["parent_plan_ref"])
    if sha256_file(reference) != plan["parent_ref_sha256"]:
        raise PackageError("parent plan reference changed")
    original, contexts = parent.load_plan(reference, root=root)
    if (original["plan_id"] != plan["parent_plan_id"]
            or canonical_json_sha256(original) != plan["parent_plan_sha256"]):
        raise PackageError("parent numerical plan changed")
    validate_config(plan["config"], original["config"])
    scientific = canonical_json_sha256({key: original["config"][key]
        for key in ("model_key", "tasks", "conditions", "candidate_order", "analysis")})
    if scientific != plan["scientific_scope_sha256"]:
        raise PackageError("amended scientific binding differs")
    effective = {**original, **plan, "parent_plan": original}
    return effective, contexts


def _kernel_scores(runner, items, *, reference=False):
    from diagnostics.general_model_numeric_kernel_v2 import score_batch
    return score_batch(runner, items, reference=reference)


def _prefix_scores(runner, context, catalog, *, reference=False):
    from diagnostics.general_model_numeric_kernel_v2 import score_prefix_block
    return score_prefix_block(runner, context, catalog, reference=reference)


def scoring_profile(*, prefix=False, padding_extra=0, permuted=False, replica_shift=0) -> dict:
    return {"prefix": prefix, "padding_extra": padding_extra, "replica_shift": replica_shift,
            "candidate_permutation": "group-rotate-one-then-reverse-hate-reverse" if permuted else "canonical"}


def score_pass(runner, contexts: list[dict], plan: dict, output: Path, *, batch_size: int,
               reference: bool = False, padding_extra: int = 0, prefix: bool = False,
               permuted: bool = False, replica_shift: int = 0,
               scorer=None, prefix_scorer=None) -> tuple[list[dict], dict]:
    if type(replica_shift) is not int or replica_shift < 0:
        raise PackageError("invalid numerical replica shift")
    if getattr(runner, "is_numeric_pool", False):
        if scorer is not None or prefix_scorer is not None:
            raise PackageError("replica pool uses its bound worker scorers")
        return runner.score_pass(contexts, plan, output, batch_size=batch_size,
                                 reference=reference, padding_extra=padding_extra,
                                 prefix=prefix, permuted=permuted, replica_shift=replica_shift)
    scorer = scorer or _kernel_scores
    prefix_scorer = prefix_scorer or _prefix_scores
    if prefix and (batch_size != 1 or permuted or padding_extra):
        raise PackageError("prefix reference requires its own unpadded batch-one pass")
    output.mkdir(parents=True, exist_ok=True)
    profile = scoring_profile(prefix=prefix, padding_extra=padding_extra,
                              permuted=permuted, replica_shift=replica_shift)
    identity = {"plan_id": plan["plan_id"], "runtime": runner.identity, "batch_size": batch_size,
                "reference": reference, "pass_name": output.name,
                "records": [r["record_id"] for r in contexts], "scoring_profile": profile}
    checkpoint = parent.Checkpoint(output / "checkpoint.sqlite3", identity)
    previous_padding = getattr(runner, "padding_extra", 0)
    runner.padding_extra = padding_extra
    started = time.monotonic()
    try:
        existing = checkpoint.rows()
        by_id = {r["record_id"]: r for r in contexts}
        if len(by_id) != len(contexts) or set(existing) - set(by_id):
            raise PackageError("checkpoint or requested frame has invalid blocks")
        for key, row in existing.items():
            parent.validate_block(row, by_id[key], plan["catalog"])
            if (row.get("runtime_sha256") != canonical_json_sha256(runner.identity)
                    or row.get("scoring_profile") != profile or row.get("plan_id") != plan["plan_id"]):
                raise PackageError("checkpoint block execution identity differs")
        invocation = str(uuid.uuid4())
        with (output / "invocations.jsonl").open("ab") as handle:
            handle.write(canonical_json_bytes({"invocation_id": invocation, "reused_blocks": len(existing),
                                              "identity_sha256": canonical_json_sha256(identity)}) + b"\n")
        catalog = copy.deepcopy(plan["catalog"])
        if permuted:
            catalog = {task: list(reversed(rows if task == "hate" else rows[1:] + rows[:1]))
                       for task, rows in catalog.items()}
        for rows, batches in parent.batch_groups(contexts, catalog, batch_size):
            found = [r["record_id"] in existing for r in rows]
            if all(found):
                continue
            if any(found):
                raise PackageError("checkpoint splits a frozen batch group")
            attempt = checkpoint.attempt(invocation, [r["record_id"] for r in rows])
            cohort = next((c for c in ("regression", "validation") if output.name.startswith(c)), "dev")
            blocks = {r["record_id"]: {**{k: r[k] for k in
                ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256")},
                "candidates": [], "plan_id": plan["plan_id"], "execution_batch_size": batch_size,
                "pass_name": output.name, "cohort": cohort, "scoring_profile": profile,
                "repetition": 1 if output.name.endswith("r1") else 0, "attempt_ordinal": attempt,
                "runtime_sha256": canonical_json_sha256(runner.identity)} for r in rows}
            if prefix:
                context = rows[0]
                items = [{"context": context, "candidate": c} for c in catalog[context["task"]]]
                evaluations = [(0, items, prefix_scorer(runner, context, catalog[context["task"]], reference=reference))]
            else:
                evaluations = ((ordinal, items, scorer(runner, items, reference=reference)) for ordinal, items in batches)
            for ordinal, items, scores in evaluations:
                if len(scores) != len(items):
                    raise PackageError("scorer output batch length differs")
                members = [f"{item['context']['record_id']}:{item['candidate']['candidate_id']}" for item in items]
                for index, (item, score) in enumerate(zip(items, scores, strict=True)):
                    blocks[item["context"]["record_id"]]["candidates"].append({
                        **item["candidate"], **score,
                        "scores": candidate_scores(score["token_logprobs"], score["eos_logprob"]),
                        "effective_batch_size": 1 if prefix else len(items),
                        "batch_ordinal": ordinal, "batch_member_ordinal": index,
                        "batch_members": members, "prefix_reference": prefix,
                    })
            committed = [blocks[r["record_id"]] for r in rows]
            for block, context in zip(committed, rows, strict=True):
                block["candidates"].sort(key=lambda item: item["ordinal"])
                parent.validate_block(block, context, plan["catalog"])
            checkpoint.commit(committed)
            checkpoint.finish_attempt(attempt)
            existing.update({r["record_id"]: r for r in committed})
            if len(existing) % 12 == 0 or len(existing) == len(contexts):
                parent.progress("scoring-v2", pass_name=output.name, completed_blocks=len(existing),
                                expected_blocks=len(contexts), elapsed_seconds=round(time.monotonic() - started, 2))
        ordered = [existing[r["record_id"]] for r in contexts]
        raw = output / "scores.jsonl"
        if raw.exists():
            if read_jsonl(raw) != ordered:
                raise PackageError("sealed score pass differs from checkpoint")
        else:
            temporary = output / ".scores.jsonl"
            write_jsonl(temporary, ordered)
            os.replace(temporary, raw)
        candidates = [candidate for block in ordered for candidate in block["candidates"]]
        forward = math.fsum(c.get("forward_seconds", 0.0) / c["effective_batch_size"] for c in candidates)
        peaks = [c["peak_memory_allocated_bytes"] for c in candidates if c.get("peak_memory_allocated_bytes") is not None]
        receipt = {"schema_version": "general-model-ld-numeric-pass/v2", "status": "complete",
                   "identity": identity, "blocks": len(ordered), "candidates": len(candidates),
                   "scores_sha256": sha256_file(raw), "query_gold_loaded": False,
                   "test_content_read": False, "mixed_execution_modes": False,
                   "performance": {"forward_seconds": forward,
                       "normalization_seconds": math.fsum(c.get("normalization_seconds", 0.0) for c in candidates),
                       "reference_seconds": math.fsum(c.get("reference_seconds", 0.0) for c in candidates),
                       "peak_memory_allocated_bytes": max(peaks) if peaks else None,
                       "forward_candidates_per_second": len(candidates) / forward if forward else None}}
        manifest = output / "manifest.json"
        if manifest.exists() and read_json(manifest) != receipt:
            raise PackageError("sealed score pass manifest differs")
        if not manifest.exists():
            parent.atomic_json(manifest, receipt)
        return ordered, receipt
    finally:
        runner.padding_extra = previous_padding
        checkpoint.close()


def replica_proof(standard: list[dict], shifted: list[dict]) -> dict:
    if len(standard) != len(shifted):
        raise PackageError("replica challenge block coverage differs")
    producers = {}
    count = 0
    geometry = ("batch_members", "batch_member_ordinal", "effective_batch_size", "batch_size",
                "padded_sequence_tokens", "sequence_tokens", "prompt_token_ids_sha256",
                "answer_token_ids", "eos_token_id", "causal_shift", "use_cache", "padding_side")
    for original, observed in zip(standard, shifted, strict=True):
        if any(original[key] != observed[key] for key in ("record_id", "context_sha256", "prompt_sha256")):
            raise PackageError("replica challenge context identity differs")
        if len(original["candidates"]) != len(observed["candidates"]):
            raise PackageError("replica challenge candidate coverage differs")
        for first, second in zip(original["candidates"], observed["candidates"], strict=True):
            if first["candidate_id"] != second["candidate_id"] or any(
                    key not in first or key not in second or first[key] != second[key] for key in geometry):
                raise PackageError("replica challenge altered candidate batch geometry")
            first_uuid, second_uuid = first.get("physical_gpu_uuid"), second.get("physical_gpu_uuid")
            first_index, second_index = first.get("physical_gpu_index"), second.get("physical_gpu_index")
            if (not isinstance(first_uuid, str) or not first_uuid
                    or not isinstance(second_uuid, str) or not second_uuid or first_uuid == second_uuid
                    or type(first_index) is not int or type(second_index) is not int or first_index == second_index):
                raise PackageError("replica challenge did not change the actual physical GPU")
            key = (first_uuid, second_uuid, first_index, second_index)
            producers[key] = producers.get(key, 0) + 1
            count += 1
    return {"passed": True, "blocks": len(standard), "candidates": count,
            "all_candidates_changed_physical_gpu": True, "batch_geometry_unchanged": True,
            "producer_pairs": [{"from_uuid": a, "to_uuid": b, "from_index": c, "to_index": d,
                                "candidates": count} for (a, b, c, d), count in sorted(producers.items())]}


def preflight(runner, plan: dict, contexts: list[dict], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    policy = plan["config"]["validation"]
    report = {"schema_version": "general-model-ld-numeric-calibration/v2", "plan_id": plan["plan_id"],
              "runtime_identity": runner.identity, "passed": False, "cohorts": {}, "epsilon": None,
              "query_gold_loaded": False, "test_content_read": False, "scientific_effect_checked": False,
              "formal_test_authorized": False, "error_families": list(CHALLENGES)}
    for cohort in ("regression", "validation"):
        ids = set(plan["cohorts"][cohort])
        selected = [r for r in contexts if r["query_id"] in ids]
        expected = {f"{qid}:{task}:{condition}" for qid in ids for task in plan["config"]["tasks"]
                    for condition in plan["config"]["conditions"]}
        if (len(ids) != policy[f"{cohort}_query_count"]
                or len(plan["cohorts"][cohort]) != len(ids)
                or set(plan["cohorts"]["regression"]) & set(plan["cohorts"]["validation"])
                or len(selected) != len(expected) or {r["record_id"] for r in selected} != expected):
            raise PackageError("preflight cohort matrix is incomplete or overlapping")
        baseline, _ = score_pass(runner, selected, plan, output / f"{cohort}-b1-r0", batch_size=1, reference=True)
        repeated, _ = score_pass(runner, selected, plan, output / f"{cohort}-b1-r1", batch_size=1)
        repeat = parent.compare_passes(baseline, repeated)
        write_json(output / f"{cohort}-repeat-differences.json", repeat)
        reference_rows = copy.deepcopy(baseline)
        for row in reference_rows:
            for candidate in row["candidates"]:
                candidate.update(candidate["reference_scores"])
                candidate["scores"] = candidate_scores(candidate["token_logprobs"], candidate["eos_logprob"])
        arithmetic = parent.compare_passes(reference_rows, baseline)
        write_json(output / f"{cohort}-reference-differences.json", arithmetic)
        info = {"blocks": len(selected), "baseline_repeat_max_abs_error": repeat["max_abs_error"],
                "reference_max_abs_error": arithmetic["max_abs_error"], "challenges": {},
                "baseline_passed": repeat["max_abs_error"] <= policy["repeat_abs_tolerance"]
                    and arithmetic["max_abs_error"] <= policy["reference_abs_tolerance"]}
        report["cohorts"][cohort] = info
        if not info["baseline_passed"]:
            report["failure"] = f"{cohort}-baseline-gate"
            break
        tail = [next(r for r in reversed(selected) if r["task"] == "hate" and r["condition"] == c)
                for c in plan["config"]["conditions"]]
        definitions = (
            ("batch", selected, {"batch_size": 4}, "b4-r0"),
            ("tail", tail, {"batch_size": 4}, "b4-tail2"),
            ("padding", selected, {"batch_size": 4, "padding_extra": policy["padding_challenge_extra"]}, "b4-padding"),
            ("prefix", selected, {"batch_size": 1, "prefix": True}, "b1-prefix"),
            ("members", selected, {"batch_size": 4, "permuted": True}, "b4-members"),
            ("replica", selected, {"batch_size": 4, "replica_shift": policy["replica_challenge_shift"]}, "b4-replica"),
        )
        limit = policy["calibration_max_abs_error"] if cohort == "regression" else report["epsilon"]
        standard_batch = None
        for name, frame, options, suffix in definitions:
            rows, _ = score_pass(runner, frame, plan, output / f"{cohort}-{suffix}", **options)
            if name == "batch":
                standard_batch = rows
            records = {r["record_id"] for r in frame}
            comparison = parent.compare_passes([r for r in baseline if r["record_id"] in records], rows)
            write_json(output / f"{cohort}-{name}-differences.json", comparison)
            if name == "replica":
                if standard_batch is None:
                    raise PackageError("replica challenge requires the standard batch-four pass")
                proof = replica_proof(standard_batch, rows)
                write_json(output / f"{cohort}-replica-producer-proof.json", proof)
                production = parent.compare_passes(standard_batch, rows)
                write_json(output / f"{cohort}-replica-vs-batch4-differences.json", production)
                largest = max((comparison, production), key=lambda result: result["max_abs_error"])
                info["replica_comparisons"] = {"versus_batch_one": comparison["max_abs_error"],
                                               "versus_standard_batch_four": production["max_abs_error"]}
                comparison = largest
            info["challenges"][name] = {"max_abs_error": comparison["max_abs_error"],
                                          "largest_error": comparison["largest_error"],
                                          "passed": comparison["max_abs_error"] <= limit}
            if not info["challenges"][name]["passed"]:
                report["failure"] = f"{cohort}-{name}-compatibility-gate"
                break
        info["max_abs_error"] = max((r["max_abs_error"] for r in info["challenges"].values()), default=0.0)
        info["complete"] = set(info["challenges"]) == set(CHALLENGES)
        info["passed"] = info["complete"] and all(r["passed"] for r in info["challenges"].values())
        if cohort == "regression":
            report["E8"] = info["max_abs_error"]
            if info["passed"]:
                report["epsilon"] = min(policy["epsilon_ceiling"], max(policy["epsilon_floor"],
                    policy["epsilon_multiplier"] * report["E8"]))
        if not info["passed"]:
            break
    report["complete"] = len(report["cohorts"]) == 2 and all(r.get("complete", False) for r in report["cohorts"].values())
    report["passed"] = report["complete"] and all(r["baseline_passed"] and r["passed"] for r in report["cohorts"].values())
    report["validation_executed"] = "validation" in report["cohorts"]
    report["files"] = {path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file()
        and (path.name in {"scores.jsonl", "manifest.json", "pool_binding.json"}
             or path.name.endswith(("-differences.json", "-proof.json")))}
    parent.atomic_json(output / "preflight_report.json", report)
    parent.progress("numerical-v2-preflight-finished", passed=report["passed"],
                    failure=report.get("failure"), E8=report.get("E8"), epsilon=report["epsilon"])
    return report


def _verified_preflight(plan: dict, output: Path, run: dict) -> dict:
    path = output / "preflight/preflight_report.json"
    if sha256_file(path) != run["preflight_report_sha256"]:
        raise PackageError("sealed preflight report changed")
    report = read_json(path)
    if report["plan_id"] != plan["plan_id"]:
        raise PackageError("preflight plan differs")
    for name, digest in report["files"].items():
        payload = output / "preflight" / name
        if not payload.resolve().is_relative_to((output / "preflight").resolve()) or sha256_file(payload) != digest:
            raise PackageError("sealed preflight payload changed")
    return report


def validate_raw_identity(plan: dict, report: dict, receipt: dict) -> None:
    expected = {"plan_id": plan["plan_id"], "runtime": report["runtime_identity"],
                "batch_size": 4, "reference": False, "pass_name": "dev-b4",
                "scoring_profile": scoring_profile(),
                "records": [row["record_id"] for row in plan["blocks"]]}
    if any(receipt.get("identity", {}).get(key) != value for key, value in expected.items()):
        raise PackageError("development raw does not match the validated production runtime and profile")
    if (receipt.get("execution") != "data-parallel-identical-fp32"
            or any(receipt.get(key) is not False for key in
                   ("query_gold_loaded", "test_content_read", "mixed_execution_modes"))):
        raise PackageError("development raw data or execution boundary differs")


def analyze_run(plan: dict, output: Path, *, blocks=None) -> dict:
    from diagnostics.general_model_numeric_pool import validate_sealed_pass

    run = read_json(output / "run_manifest.json")
    report = _verified_preflight(plan, output, run)
    if not report["passed"]:
        raise PackageError("analysis requires a passed numerical preflight")
    raw = output / "dev-b4"
    receipt = read_json(raw / "manifest.json")
    if (receipt["status"] != "complete" or receipt["blocks"] != 7716 or receipt["candidates"] != 131172
            or receipt["identity"]["plan_id"] != plan["plan_id"]
            or sha256_file(raw / "manifest.json") != run["raw_manifest_sha256"]
            or sha256_file(raw / "scores.jsonl") != receipt["scores_sha256"]):
        raise PackageError("full development raw artifact is incomplete or changed")
    validate_raw_identity(plan, report, receipt)
    validate_sealed_pass(raw, plan)
    stored = read_jsonl(raw / "scores.jsonl")
    if blocks is not None and stored != blocks:
        raise PackageError("analysis input differs from sealed raw")
    for row, context in zip(stored, plan["blocks"], strict=True):
        parent.validate_block(row, context, plan["catalog"])
    directory = output / "analysis"
    if directory.exists():
        existing = read_json(directory / "manifest.json")
        if (existing["plan_id"] != plan["plan_id"]
                or existing["raw_manifest_sha256"] != run["raw_manifest_sha256"]
                or sha256_file(directory / "analysis.json") != existing["analysis_sha256"]):
            raise PackageError("existing analysis identity differs")
        return existing
    package = Path(plan["package_path"])
    package_manifest = read_json(package / "manifest.json")
    gold_entry = next(row for row in package_manifest["files"] if row["path"] == "queries.dev.jsonl")
    if (sha256_file(package / "manifest.json") != plan["package_manifest_sha256"]
            or sha256_file(package / "queries.dev.jsonl") != gold_entry["sha256"]):
        raise PackageError("gold source changed after scoring")
    gold = {str(row["id"]): row["projection"] for row in read_jsonl(package / "queries.dev.jsonl")}
    bootstrap = plan["config"]["analysis"]["bootstrap"]
    analysis = analyze_blocks(stored, frame=plan["frame"], gold_by_query=gold, epsilon=report["epsilon"],
                              bootstrap_replicates=bootstrap["repetitions"], bootstrap_seed=bootstrap["seed"])
    staging = Path(tempfile.mkdtemp(prefix=".analyzing-", dir=output))
    write_json(staging / "analysis.json", analysis)
    manifest = {"schema_version": "general-model-ld-numeric-analysis/v2", "plan_id": plan["plan_id"],
                "raw_manifest_sha256": run["raw_manifest_sha256"], "gold_join_after_raw_sealed": True,
                "test_content_read": False, "analysis_sha256": sha256_file(staging / "analysis.json")}
    write_json(staging / "manifest.json", manifest)
    os.rename(staging, directory)
    return manifest


def run_pipeline(plan_path: Path, output: Path, *, device="cuda:0", root: Path = ROOT) -> dict:
    from diagnostics.general_model_numeric_pool import PersistentNumericPool
    plan, contexts = load_plan(plan_path, root=root)
    if device != "cuda:0":
        raise PackageError("physical devices are fixed by the numerical replica configuration")
    device_indices = plan["config"]["execution"]["device_indices"]
    output.mkdir(parents=True, exist_ok=True)
    binding = {"plan_id": plan["plan_id"], "physical_device_indices": device_indices,
               "execution": "data-parallel-identical-fp32",
               "phase": "preflight-then-dev-on-pass"}
    binding_path = output / "binding.json"
    if binding_path.exists() and read_json(binding_path) != binding:
        raise PackageError("run directory belongs to a different execution")
    if not binding_path.exists():
        parent.atomic_json(binding_path, binding)
    lock = (output / ".writer.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runner = None
    execution_started = False
    manifest_path = output / "run_manifest.json"
    result = {**binding, "schema_version": "general-model-ld-numeric-run/v2", "status": "running",
              "test_content_read": False, "query_gold_loaded_during_scoring": False}
    try:
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if previous.get("status") in {"complete", "preflight_failed"}:
                report = _verified_preflight(plan, output, previous)
                if (previous["status"] == "complete") != report["passed"]:
                    raise PackageError("terminal execution disagrees with its preflight")
                if previous["status"] == "complete":
                    analyze_run(plan, output)
                return previous
            if previous.get("status") == "raw_complete":
                result = previous
                execution_started = True
                analyze_run(plan, output)
                previous.update(status="complete", analysis_published=True)
                parent.atomic_json(manifest_path, previous)
                return previous
        execution_started = True
        parent.atomic_json(manifest_path, result)
        runner = PersistentNumericPool(plan["parent_plan"], plan["config"]["runtime"], device_indices, root)
        write_json(output / "runtime_identity.json", runner.identity)
        report = preflight(runner, plan, contexts, output / "preflight")
        result["preflight_report_sha256"] = sha256_file(output / "preflight/preflight_report.json")
        if not report["passed"]:
            result.update(status="preflight_failed", failure=report.get("failure"), full_dev_started=False,
                          analysis_published=False, fallback_automatic=False)
        else:
            parent.progress("numerical-v2-preflight-passed-starting-dev", blocks=7716, candidates=131172)
            blocks, receipt = score_pass(runner, contexts, plan, output / "dev-b4", batch_size=4)
            result.update(status="raw_complete", full_dev_started=True, raw_blocks=receipt["blocks"],
                          raw_manifest_sha256=sha256_file(output / "dev-b4/manifest.json"))
            parent.atomic_json(manifest_path, result)
            runner.close()
            runner = None
            analyze_run(plan, output, blocks=blocks)
            result.update(status="complete", analysis_published=True)
        parent.atomic_json(manifest_path, result)
        return result
    except BaseException as error:
        if not execution_started:
            raise
        status = "raw_complete" if result.get("raw_manifest_sha256") else (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        result.update(status=status,
                      error_type=type(error).__name__, error=str(error), analysis_published=False)
        parent.atomic_json(manifest_path, result)
        if not (output / "preflight/preflight_report.json").exists():
            directory = output / "preflight"
            directory.mkdir(exist_ok=True)
            parent.atomic_json(directory / "failure.json", {"plan_id": plan["plan_id"], "passed": False,
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
            command.add_argument("--device", default="cuda:0", help="logical worker device; physical replicas are registered in config")
    args = parser.parse_args(argv)
    if args.command == "build-plan":
        result = build_plan(args.config)
    elif args.command == "run":
        result = run_pipeline(args.plan, args.output, device=args.device)
    else:
        plan, contexts = load_plan(args.plan)
        result = analyze_run(plan, args.output) if args.command == "analyze" else {
            "plan_id": plan["plan_id"], "valid": True, "blocks": len(contexts), "query_gold_loaded": False}
    print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
    return 2 if result.get("status") == "preflight_failed" else 0
