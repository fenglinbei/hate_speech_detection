"""Audit only top-level sealed preflight passes, with independent CPU math."""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HELPERS = ROOT / "exps/causal_context/general_model_ld_numeric_v1/audits"
sys.path.insert(0, str(HELPERS))
from independent_full_dev_audit import expected_catalog, load_effective_plan, margin_values
from independent_numeric_audit import canonical_hash, digest, read, recompute
from independent_parallel_audit import audit_parallel_pass, independent_assignments, validate_cross_gpu_challenge
from independent_reference_pass_audit import audit as audit_reference


CHALLENGES = ("batch", "tail", "padding", "prefix", "members", "replica")
SUFFIXES = {"baseline": "b1-r0", "repeat": "b1-r1", "batch": "b4-r0", "tail": "b4-tail2",
            "padding": "b4-padding", "prefix": "b1-prefix", "members": "b4-members", "replica": "b4-replica"}
FIELDS = ("answer_sum", "answer_mean", "eos_logprob", "total_with_eos", "mean_with_eos")
TOLERANCE = 1e-9


def close(first, second):
    assert math.isfinite(first) and math.isfinite(second)
    assert abs(first - second) <= TOLERANCE, (first, second)


def flat_readouts(row, *, reference=False):
    values, candidates = {}, []
    for candidate in row["candidates"]:
        raw = candidate["reference_scores"] if reference else candidate
        computed = recompute(raw["token_logprobs"], raw["eos_logprob"])
        assert all(raw[name] == computed[name] for name in FIELDS)
        if not reference:
            assert candidate["scores"] == computed
        stem = "candidate/" + candidate["candidate_id"]
        values.update({f"{stem}/token/{position}": number for position, number in enumerate(raw["token_logprobs"])})
        values.update({stem + "/" + name: computed[name] for name in FIELDS})
        candidates.append({"scores": computed})
    values.update({f"margin/{mode}/{label}": number for (mode, label), number in margin_values(row["task"], candidates).items()})
    return values


def validate_geometry(rows, contexts, identity, plan):
    profile = identity["scoring_profile"]
    expected = {}
    context_by_id = {context["record_id"]: context for context in contexts}
    assignments = independent_assignments(contexts, identity["runtime"]["device_indices"], identity["batch_size"], profile["replica_shift"])
    for assignment in assignments:
        batch_ordinal = 0
        for group in assignment["groups"]:
            items = []
            for record_id in group:
                context = context_by_id[record_id]
                candidates = list(plan["catalog"][context["task"]])
                if profile["candidate_permutation"] != "canonical":
                    assert profile["candidate_permutation"] == "group-rotate-one-then-reverse-hate-reverse"
                    candidates = list(reversed(candidates if context["task"] == "hate" else candidates[1:] + candidates[:1]))
                items.extend((context, candidate) for candidate in candidates)
            batches = [items] if profile["prefix"] else [items[start:start + identity["batch_size"]]
                       for start in range(0, len(items), identity["batch_size"])]
            for batch in batches:
                members = [context["record_id"] + ":" + candidate["candidate_id"] for context, candidate in batch]
                padded = None if profile["prefix"] else max(context["prompt_tokens"] + len(candidate["answer_token_ids"]) + 1
                           for context, candidate in batch) + profile["padding_extra"]
                for position, (context, candidate) in enumerate(batch):
                    expected[(context["record_id"], candidate["candidate_id"])] = {
                        "batch_members": members, "batch_member_ordinal": position,
                        "batch_ordinal": 0 if profile["prefix"] else batch_ordinal,
                        "effective_batch_size": 1 if profile["prefix"] else len(batch),
                        "batch_size": 1 if profile["prefix"] else len(batch),
                        "padded_sequence_tokens": padded, "physical_gpu_index": assignment["physical_gpu_index"],
                    }
                batch_ordinal += 1
    prefix_values = {}
    candidate_count = token_count = 0
    for row, context in zip(rows, contexts, strict=True):
        assert all(row[name] == context[name] for name in ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256"))
        assert row["execution_batch_size"] == identity["batch_size"]
        assert row["pass_name"] == identity["pass_name"]
        assert row["repetition"] == (1 if row["pass_name"].endswith("r1") else 0)
        assert row["cohort"] == row["pass_name"].split("-")[0]
        catalog = plan["catalog"][row["task"]]
        assert len(row["candidates"]) == len(catalog)
        prefixes = {tuple(candidate["answer_token_ids"][:index]) for candidate in catalog
                    for index in range(len(candidate["answer_token_ids"]) + 1)}
        for candidate, frozen, canonical in zip(row["candidates"], catalog, expected_catalog(row["task"]), strict=True):
            assert all(candidate[name] == value for name, value in {**frozen, **canonical}.items())
            assert all(candidate[name] == value for name, value in expected[(row["record_id"], candidate["candidate_id"])].items())
            assert candidate["reference_checked"] is identity["reference"]
            assert candidate["prompt_token_ids_sha256"] == context["prompt_token_ids_sha256"]
            assert candidate["prompt_tokens"] == context["prompt_tokens"]
            assert candidate["answer_token_ids_sha256"] == canonical_hash(candidate["answer_token_ids"])
            assert candidate["sequence_tokens"] == context["prompt_tokens"] + candidate["answer_tokens"] + 1
            assert candidate["answer_tokens"] == len(candidate["answer_token_ids"]) == len(candidate["token_logprobs"])
            assert candidate["eos_token_id"] == plan["eos_token_id"] not in candidate["answer_token_ids"]
            assert candidate["token_boundary_checked"] is True and candidate["finite_target_logits_checked"] is True
            assert candidate["causal_shift"] == 1 and candidate["use_cache"] is False
            assert candidate["padding_side"] == "right" and candidate["prefix_reference"] is profile["prefix"]
            assert candidate["model_logits_dtype"] == candidate["logprob_arithmetic_dtype"] == "torch.float32"
            if profile["prefix"]:
                assert candidate["prefix_padding"] is False
                assert candidate["prefix_unique_forward_count"] == len(prefixes)
                assert candidate["scoring_implementation"] == "uncached-prefix-only"
                targets = candidate["answer_token_ids"] + [candidate["eos_token_id"]]
                probabilities = candidate["token_logprobs"] + [candidate["eos_logprob"]]
                for index, (target, probability) in enumerate(zip(targets, probabilities, strict=True)):
                    key = (row["record_id"], tuple(targets[:index]), target)
                    if key in prefix_values:
                        assert probability == prefix_values[key], "shared mathematical prefix was not reused exactly"
                    prefix_values[key] = probability
            else:
                assert candidate["scoring_implementation"] == "full-sequence-selected-projection"
                assert candidate["sequence_tokens"] <= candidate["padded_sequence_tokens"] <= 8192
            candidate_count += 1
            token_count += candidate["answer_tokens"]
        flat_readouts(row)
    return {"candidates": candidate_count, "answer_tokens": token_count,
            "reconstructed_batch_geometry_exact": True, "candidate_and_prompt_token_identity_exact": True,
            "canonical_catalog_restored": True, "all_candidate_scores_math_fsum_exact": True,
            "prefix_unique_token_conditionals": len(prefix_values) if profile["prefix"] else None}


def validate_pass(directory, contexts, plan, name):
    manifest = read(directory / "manifest.json")
    assert manifest["status"] == "complete"
    assert all(manifest[flag] is False for flag in ("query_gold_loaded", "test_content_read", "mixed_execution_modes"))
    identity = manifest["identity"]
    expected_profile = {"candidate_permutation": "group-rotate-one-then-reverse-hate-reverse" if name == "members" else "canonical",
                        "padding_extra": 64 if name == "padding" else 0,
                        "prefix": name == "prefix", "replica_shift": 1 if name == "replica" else 0}
    assert identity["scoring_profile"] == expected_profile
    assert identity["reference"] is (name == "baseline")
    expected_batch = 1 if plan.get("fallback_config") or name in ("baseline", "repeat", "prefix") else 4
    assert identity["batch_size"] == expected_batch
    assert identity["pass_name"] == directory.name
    manifest_sha, scores_sha = digest(directory / "manifest.json"), digest(directory / "scores.jsonl")
    assert scores_sha == manifest["scores_sha256"]
    rows = [json.loads(line) for line in (directory / "scores.jsonl").read_text().splitlines()]
    shard_audit = audit_parallel_pass(directory, contexts, plan, merged_rows=rows)
    geometry = validate_geometry(rows, contexts, identity, plan)
    assert geometry["candidates"] == manifest["candidates"]
    assert digest(directory / "manifest.json") == manifest_sha and digest(directory / "scores.jsonl") == scores_sha
    return rows, {"blocks": len(rows), **geometry, "manifest_sha256": manifest_sha,
                  "scores_sha256": scores_sha, "parallel_identity_and_shards": shard_audit}


def compare(reference_rows, observed_rows, recorded_path, *, cpu_reference=False):
    assert len(reference_rows) == len(observed_rows)
    saved = read(recorded_path) if recorded_path.exists() else None
    if saved is not None:
        assert len(saved["blocks"]) == len(observed_rows)
    maximum, largest, count, nonzero = 0.0, None, 0, 0
    summaries = defaultdict(lambda: {"readouts": 0, "max_abs_error": 0.0, "nonzero_readouts": 0})
    recorded_numeric_max = 0.0
    for position, (baseline, observed) in enumerate(zip(reference_rows, observed_rows, strict=True)):
        assert baseline["record_id"] == observed["record_id"]
        assert all(baseline[name] == observed[name] for name in ("query_id", "task", "condition", "context_sha256", "prompt_sha256"))
        before, after = flat_readouts(baseline, reference=cpu_reference), flat_readouts(observed)
        assert before.keys() == after.keys()
        stored = saved["blocks"][position] if saved is not None else None
        if stored is not None:
            assert stored["record_id"] == observed["record_id"]
            assert stored["differences"].keys() == before.keys()
        for metric in before:
            difference = after[metric] - before[metric]
            assert math.isfinite(difference)
            absolute = abs(difference)
            count += 1
            nonzero += absolute != 0
            if absolute > maximum:
                maximum, largest = absolute, {"record_id": observed["record_id"], "metric": metric, "difference": difference}
            family = "margin/" + metric.split("/")[1] if metric.startswith("margin/") else (
                "answer_token_logprob" if "/token/" in metric else metric.rsplit("/", 1)[-1])
            for group in ("metric/" + family, "task/" + observed["task"],
                          "task_condition/" + observed["task"] + "/" + observed["condition"]):
                summary = summaries[group]
                summary["readouts"] += 1
                summary["nonzero_readouts"] += absolute != 0
                summary["max_abs_error"] = max(summary["max_abs_error"], absolute)
            if stored is not None:
                close(stored["differences"][metric], difference)
                recorded_numeric_max = max(recorded_numeric_max, abs(stored["differences"][metric]))
    if saved is not None:
        close(saved["max_abs_error"], maximum)
        assert saved["max_abs_error"] == recorded_numeric_max
        if maximum == 0:
            assert saved["largest_error"] is None
        else:
            detail = saved["largest_error"]
            raw = next(block for block in saved["blocks"] if block["record_id"] == detail["record_id"])
            assert raw["differences"][detail["metric"]] == detail["difference"]
            assert abs(detail["difference"]) == recorded_numeric_max
    return {"blocks": len(observed_rows), "readouts": count, "nonzero_readouts": nonzero,
            "max_abs_error": maximum, "largest_error": largest, "numerical_error_summaries": dict(summaries),
            "stored_difference_file_verified": saved is not None,
            "stored_difference_sha256": digest(recorded_path) if saved is not None else None}


def audit(reference, directory):
    plan = load_effective_plan(reference, root=ROOT)
    passes, results, frames = {}, {}, {}
    for cohort in ("regression", "validation"):
        full = [entry for entry in plan["blocks"] if entry["query_id"] in plan["cohorts"][cohort]]
        tail = [next(entry for entry in reversed(full) if entry["task"] == "hate" and entry["condition"] == condition)
                for condition in plan["config"]["conditions"]]
        frames[cohort] = full
        results[cohort] = {"sealed_passes": {}, "pending_passes": [], "challenges": {}}
        for name, suffix in SUFFIXES.items():
            path = directory / f"{cohort}-{suffix}"
            if not (path / "manifest.json").is_file():
                results[cohort]["pending_passes"].append(name)
                continue
            rows, checked = validate_pass(path, tail if name == "tail" else full, plan, name)
            passes[(cohort, name)] = rows
            results[cohort]["sealed_passes"][name] = checked

    runtime_hashes = {entry["parallel_identity_and_shards"]["common_runtime_sha256"]
                      for result in results.values() for entry in result["sealed_passes"].values()}
    assert len(runtime_hashes) <= 1
    epsilon, e8 = None, None
    for cohort in ("regression", "validation"):
        result = results[cohort]
        if (cohort, "baseline") not in passes:
            result.update(six_challenges_complete=False, baseline_passed=None, passed=None)
            continue
        baseline = passes[(cohort, "baseline")]
        checked_reference = audit_reference(directory / f"{cohort}-b1-r0", reference)
        result["reference_raw_audit"] = checked_reference
        result["reference"] = compare(baseline, baseline, directory / f"{cohort}-reference-differences.json", cpu_reference=True)
        if (cohort, "repeat") in passes:
            result["repeat"] = compare(baseline, passes[(cohort, "repeat")], directory / f"{cohort}-repeat-differences.json")
            result["baseline_passed"] = result["reference"]["max_abs_error"] <= .0001 and result["repeat"]["max_abs_error"] <= .0001
        else:
            result["baseline_passed"] = None
        for name in CHALLENGES:
            if (cohort, name) not in passes:
                continue
            observed = passes[(cohort, name)]
            record_ids = {row["record_id"] for row in observed}
            comparison = compare([row for row in baseline if row["record_id"] in record_ids], observed,
                                 directory / f"{cohort}-{name}-differences.json")
            if name == "replica":
                standard = passes[(cohort, "batch")]
                replica_check = validate_cross_gpu_challenge(standard, observed,
                    result["sealed_passes"][name]["parallel_identity_and_shards"]["physical_gpu_uuid_by_index"])
                production = compare(standard, observed, directory / f"{cohort}-replica-vs-batch4-differences.json")
                proof_path = directory / f"{cohort}-replica-producer-proof.json"
                proof = read(proof_path) if proof_path.is_file() else None
                if proof is not None:
                    assert proof["passed"] and proof["all_candidates_changed_physical_gpu"] and proof["batch_geometry_unchanged"]
                    assert proof["blocks"] == len(observed) and proof["candidates"] == replica_check["candidates"]
                    observed_pairs = {(pair["from_uuid"], pair["to_uuid"]): pair["candidates"] for pair in proof["producer_pairs"]}
                    recomputed_pairs = {(pair["from_uuid"], pair["to_uuid"]): pair["candidates"] for pair in replica_check["transitions"]}
                    assert observed_pairs == recomputed_pairs
                    uuids = result["sealed_passes"][name]["parallel_identity_and_shards"]["physical_gpu_uuid_by_index"]
                    for pair in proof["producer_pairs"]:
                        assert uuids[pair["from_index"]] == pair["from_uuid"] and uuids[pair["to_index"]] == pair["to_uuid"]
                comparison = {"max_abs_error": max(comparison["max_abs_error"], production["max_abs_error"]),
                              "versus_batch_one": comparison, "versus_standard_batch_four": production,
                              "actual_physical_gpu_change": replica_check, "producer_proof_verified": proof is not None}
            result["challenges"][name] = comparison
        result["six_challenges_complete"] = set(result["challenges"]) == set(CHALLENGES)
        maximum = max((comparison["max_abs_error"] for comparison in result["challenges"].values()), default=0.0)
        result["observed_challenge_max_abs_error"] = maximum
        threshold = .005 if cohort == "regression" else epsilon
        result["threshold"] = threshold
        for comparison in result["challenges"].values():
            comparison["passed"] = comparison["max_abs_error"] <= threshold if threshold is not None else None
        result["passed"] = result["baseline_passed"] is True and result["six_challenges_complete"] and threshold is not None and maximum <= threshold
        if cohort == "regression" and result["passed"]:
            e8 = maximum
            epsilon = min(.005, max(.0001, 2 * e8))

    final_path = directory / "preflight_report.json"
    final = read(final_path) if final_path.is_file() else None
    final_verification = {"published": final is not None, "report_sha256": digest(final_path) if final is not None else None}
    if final is not None:
        assert final["plan_id"] == plan["plan_id"]
        assert canonical_hash(final["runtime_identity"]) in runtime_hashes
        assert final["error_families"] == list(CHALLENGES)
        assert all(final[flag] is False for flag in ("query_gold_loaded", "test_content_read", "scientific_effect_checked", "formal_test_authorized"))
        payload_paths = {path.relative_to(directory).as_posix(): path for path in directory.rglob("*")
                         if path.is_file() and (path.name in ("scores.jsonl", "manifest.json", "pool_binding.json")
                         or path.name.endswith(("-differences.json", "-proof.json")))}
        assert payload_paths.keys() == final["files"].keys()
        assert all(digest(path) == final["files"][name] for name, path in payload_paths.items())
        for cohort, reported in final["cohorts"].items():
            checked = results[cohort]
            assert reported["blocks"] == len(frames[cohort])
            close(reported["baseline_repeat_max_abs_error"], checked["repeat"]["max_abs_error"])
            close(reported["reference_max_abs_error"], checked["reference"]["max_abs_error"])
            assert reported["baseline_passed"] is checked["baseline_passed"]
            assert reported["challenges"].keys() == checked["challenges"].keys()
            for name, comparison in reported["challenges"].items():
                close(comparison["max_abs_error"], checked["challenges"][name]["max_abs_error"])
                assert comparison["passed"] is checked["challenges"][name]["passed"]
                stored_comparisons = [read(directory / f"{cohort}-{name}-differences.json")]
                if name == "replica":
                    stored_comparisons.append(read(directory / f"{cohort}-replica-vs-batch4-differences.json"))
                largest_comparison = max(stored_comparisons, key=lambda entry: entry["max_abs_error"])
                assert comparison["largest_error"] == largest_comparison["largest_error"]
            if "complete" in reported:
                assert reported["complete"] is checked["six_challenges_complete"]
                assert reported["passed"] is checked["passed"]
                close(reported["max_abs_error"], checked["observed_challenge_max_abs_error"])
            if "replica_comparisons" in reported:
                replica = checked["challenges"]["replica"]
                close(reported["replica_comparisons"]["versus_batch_one"], replica["versus_batch_one"]["max_abs_error"])
                close(reported["replica_comparisons"]["versus_standard_batch_four"], replica["versus_standard_batch_four"]["max_abs_error"])
        expected_complete = all(result["six_challenges_complete"] for result in results.values())
        expected_passed = all(result["passed"] for result in results.values())
        assert final["complete"] is expected_complete and final["passed"] is expected_passed
        assert final["validation_executed"] is (("validation", "baseline") in passes)
        if final.get("E8") is not None:
            close(final["E8"], results["regression"]["observed_challenge_max_abs_error"])
        if epsilon is not None:
            close(final["epsilon"], epsilon)
        else:
            assert final["epsilon"] is None
        for result in results.values():
            for name in ("reference", "repeat"):
                if name in result:
                    assert result[name]["stored_difference_file_verified"]
            for name, comparison in result["challenges"].items():
                if name == "replica":
                    assert comparison["producer_proof_verified"]
                    assert comparison["versus_batch_one"]["stored_difference_file_verified"]
                    assert comparison["versus_standard_batch_four"]["stored_difference_file_verified"]
                else:
                    assert comparison["stored_difference_file_verified"]
        run_manifest = read(directory.parent / "run_manifest.json")
        assert canonical_hash(read(directory.parent / "runtime_identity.json")) in runtime_hashes
        assert run_manifest["plan_id"] == plan["plan_id"]
        assert run_manifest["test_content_read"] is False
        assert run_manifest["query_gold_loaded_during_scoring"] is False
        bound = "preflight_report_sha256" in run_manifest
        if bound:
            assert run_manifest["preflight_report_sha256"] == digest(final_path)
        final_verification.update(all_payload_file_hashes_verified=True, payload_file_count=len(payload_paths),
                                  complete_and_passed_independently_verified=True, run_manifest_binds_report=bound)
    return {"schema_version": "independent-six-challenge-preflight-audit/v2", "plan_id": plan["plan_id"],
            "preflight_path": str(directory.resolve()), "cohorts": results,
            "E8_frozen_after_complete_regression": e8, "epsilon": epsilon,
            "all_six_regression_challenges_complete": results["regression"]["six_challenges_complete"],
            "all_six_validation_challenges_complete": results["validation"]["six_challenges_complete"],
            "validation_passed": results["validation"]["passed"],
            "preflight_passed": bool(final is not None and final["passed"]), "final_report": final_verification,
            "sealed_pass_count": len(passes),
            "scope": {"gpu_used": False, "query_gold_read": False, "test_content_read": False,
                      "scientific_effects_computed": False, "partial_passes_read": False,
                      "production_comparison_or_gate_functions_imported": False},
            "limits": ["Numerical margins are compared only between engineering passes of the same context; no L/D scientific contrast is computed.",
                       "Original full-vocabulary logits are not stored; CPU reference normalization cannot be re-executed from saved raw.",
                       "During a running dev pass the in-memory preflight binding may not yet be persisted in run_manifest; this is reported explicitly."],
            "audit_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
                Path(__file__), Path(__file__).with_name("independent_parallel_audit.py"),
                Path(__file__).with_name("independent_reference_pass_audit.py"),
                HELPERS / "independent_full_dev_audit.py", HELPERS / "independent_numeric_audit.py")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.plan, args.preflight_dir)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("sealed_pass_count", "E8_frozen_after_complete_regression", "epsilon",
          "all_six_regression_challenges_complete", "all_six_validation_challenges_complete", "validation_passed", "preflight_passed")}))
