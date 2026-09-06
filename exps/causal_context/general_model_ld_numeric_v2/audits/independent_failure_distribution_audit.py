"""Describe numerical gate failures only; no gold, scientific effects, or GPU."""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HELPERS = ROOT / "exps/causal_context/general_model_ld_numeric_v1/audits"
sys.path.insert(0, str(HELPERS))
from independent_numeric_audit import digest, read
from independent_preflight_audit import flat_readouts


def category(metric):
    pieces = metric.split("/")
    if pieces[0] == "margin":
        return "/".join(pieces[:2])
    return "answer_token_logprob" if pieces[2] == "token" else pieces[2]


def numerical_summary(entries, epsilon):
    absolute = np.asarray([abs(entry["difference"]) for entry in entries])
    exceeded = [entry for entry in entries if abs(entry["difference"]) > epsilon]
    candidates = {(entry["record_id"], entry["candidate_id"]) for entry in exceeded if entry["candidate_id"] is not None}
    return {"readouts": len(entries), "nonzero_readouts": int(np.count_nonzero(absolute)),
            "exceeded_readouts": len(exceeded), "exceeded_candidate_observations": len(candidates),
            "exceeded_candidate_ids": sorted({candidate for _, candidate in candidates}),
            "exceeded_records": len({entry["record_id"] for entry in exceeded}),
            "exceeded_queries": len({entry["query_id"] for entry in exceeded}),
            "maximum_absolute_error": float(absolute.max()),
            "absolute_error_quantiles": dict(zip(("q50", "q90", "q95", "q99"), np.quantile(absolute, [.5, .9, .95, .99]).tolist())),
            "largest": max(entries, key=lambda entry: abs(entry["difference"]))}


def audit(run, verified_path):
    verified = read(verified_path)
    assert verified["final_report"]["all_payload_file_hashes_verified"]
    assert verified["final_report"]["run_manifest_binds_report"]
    assert verified["all_six_regression_challenges_complete"] is True
    assert verified["validation_passed"] is False and verified["preflight_passed"] is False
    preflight = run / "preflight"
    report_path, run_path = preflight / "preflight_report.json", run / "run_manifest.json"
    report_sha, run_sha = digest(report_path), digest(run_path)
    report, terminal = read(report_path), read(run_path)
    assert report_sha == verified["final_report"]["report_sha256"] == terminal["preflight_report_sha256"]
    assert terminal["status"] == "preflight_failed" and terminal["failure"] == "validation-batch-compatibility-gate"
    assert terminal["full_dev_started"] is False and terminal["analysis_published"] is False
    assert report["passed"] is False and report["complete"] is False
    assert report["plan_id"] == verified["plan_id"] == terminal["plan_id"]
    epsilon = report["epsilon"]
    assert epsilon == min(.005, max(.0001, 2 * report["E8"])) == verified["epsilon"]
    passes, pass_hashes = {}, {}
    for name, suffix in (("baseline", "b1-r0"), ("repeat", "b1-r1"), ("batch", "b4-r0")):
        directory = preflight / ("validation-" + suffix)
        manifest = read(directory / "manifest.json")
        assert manifest["status"] == "complete"
        recorded = verified["cohorts"]["validation"]["sealed_passes"][name]
        assert digest(directory / "manifest.json") == recorded["manifest_sha256"]
        assert digest(directory / "scores.jsonl") == recorded["scores_sha256"] == manifest["scores_sha256"]
        passes[name] = [json.loads(line) for line in (directory / "scores.jsonl").read_text().splitlines()]
        pass_hashes[name] = {"manifest_sha256": recorded["manifest_sha256"], "scores_sha256": recorded["scores_sha256"]}
    baseline, batched = passes["baseline"], passes["batch"]
    assert len(baseline) == len(batched) == 288
    entries, candidate_pairs = [], {}
    for first, second in zip(baseline, batched, strict=True):
        assert first["record_id"] == second["record_id"]
        before, after = flat_readouts(first), flat_readouts(second)
        first_candidates = {candidate["candidate_id"]: candidate for candidate in first["candidates"]}
        second_candidates = {candidate["candidate_id"]: candidate for candidate in second["candidates"]}
        for candidate_id in first_candidates:
            candidate_pairs[(first["record_id"], candidate_id)] = (first, first_candidates[candidate_id], second_candidates[candidate_id])
        for metric in before:
            candidate_id = metric.split("/")[1] if metric.startswith("candidate/") else None
            entry = {"record_id": first["record_id"], "query_id": first["query_id"], "task": first["task"],
                     "condition": first["condition"], "metric": metric, "category": category(metric),
                     "candidate_id": candidate_id, "difference": after[metric] - before[metric]}
            if candidate_id is not None:
                a, b = first_candidates[candidate_id], second_candidates[candidate_id]
                entry.update(answer_tokens=a["answer_tokens"], prompt_tokens=a["prompt_tokens"],
                             sequence_tokens=a["sequence_tokens"], batch_one_padded_tokens=a["padded_sequence_tokens"],
                             batch_four_padded_tokens=b["padded_sequence_tokens"], physical_gpu_one=a["physical_gpu_index"],
                             physical_gpu_four=b["physical_gpu_index"])
            entries.append(entry)
    by_category, by_task, by_condition = defaultdict(list), defaultdict(list), defaultdict(list)
    for entry in entries:
        by_category[entry["category"]].append(entry)
        by_task[entry["task"]].append(entry)
        by_condition[entry["task"] + "/" + entry["condition"]].append(entry)
    exceeded = [entry for entry in entries if abs(entry["difference"]) > epsilon]
    exceeded_pairs = sorted({(entry["record_id"], entry["candidate_id"]) for entry in exceeded if entry["candidate_id"] is not None})
    decompositions = []
    for key in exceeded_pairs:
        row, first, second = candidate_pairs[key]
        token_differences = [b - a for a, b in zip(first["token_logprobs"], second["token_logprobs"], strict=True)]
        eos_difference = second["eos_logprob"] - first["eos_logprob"]
        sum_difference = second["answer_sum"] - first["answer_sum"]
        assert math.fsum(token_differences) == sum_difference
        assert math.fsum([*token_differences, eos_difference]) == second["total_with_eos"] - first["total_with_eos"]
        assert first["answer_token_ids"] == second["answer_token_ids"]
        decompositions.append({"record_id": row["record_id"], "candidate_id": first["candidate_id"],
            "prompt_token_ids_sha256": first["prompt_token_ids_sha256"], "prompt_tokens": first["prompt_tokens"],
            "answer_token_ids": first["answer_token_ids"], "answer_tokens": first["answer_tokens"],
            "sequence_tokens": first["sequence_tokens"], "b1_padded_tokens": first["padded_sequence_tokens"],
            "b4_padded_tokens": second["padded_sequence_tokens"], "b1_physical_gpu": first["physical_gpu_index"],
            "b4_physical_gpu": second["physical_gpu_index"],
            "same_actual_gpu_uuid": first["physical_gpu_uuid"] == second["physical_gpu_uuid"],
            "b1_batch_size": first["batch_size"], "b4_batch_size": second["batch_size"],
            "b4_batch_members": second["batch_members"], "b4_batch_member_ordinal": second["batch_member_ordinal"],
            "token_differences": token_differences, "eos_difference": eos_difference,
            "maximum_token_absolute_error": max(abs(value) for value in token_differences),
            "answer_sum_difference": sum_difference, "total_with_eos_difference": second["total_with_eos"] - first["total_with_eos"],
            "sum_difference_is_exact_sum_of_token_differences": True,
            "all_individual_tokens_within_epsilon": all(abs(value) <= epsilon for value in token_differences)})
    lengths = defaultdict(lambda: {"candidate_observations": 0, "answer_sum_exceeded": 0, "maximum_answer_sum_error": 0.0})
    for row, first, second in candidate_pairs.values():
        size = first["sequence_tokens"]
        label = next(label for upper, label in ((256, "0-256"), (512, "257-512"), (768, "513-768"),
                    (1024, "769-1024"), (1536, "1025-1536"), (2048, "1537-2048"), (8192, "2049-8192")) if size <= upper)
        bucket = lengths[row["task"] + "/" + label]
        error = abs(second["answer_sum"] - first["answer_sum"])
        bucket["candidate_observations"] += 1
        bucket["answer_sum_exceeded"] += error > epsilon
        bucket["maximum_answer_sum_error"] = max(bucket["maximum_answer_sum_error"], error)
    remaining = {name: {"executed": False, "directory_created": (preflight / ("validation-" + suffix)).exists()}
                 for name, suffix in (("tail", "b4-tail2"), ("padding", "b4-padding"), ("prefix", "b1-prefix"),
                                      ("members", "b4-members"), ("replica", "b4-replica"))}
    assert all(not entry["directory_created"] for entry in remaining.values())
    assert not (run / "dev-b4").exists() and not (run / "analysis").exists()
    assert max(abs(entry["difference"]) for entry in entries) == report["cohorts"]["validation"]["challenges"]["batch"]["max_abs_error"]
    assert digest(report_path) == report_sha and digest(run_path) == run_sha
    return {"schema_version": "independent-numerical-failure-distribution/v1", "audit_passed": True,
        "numerical_preflight_passed": False, "plan_id": verified["plan_id"], "run": str(run.resolve()),
        "run_manifest_sha256": run_sha, "preflight_report_sha256": report_sha,
        "verified_full_engineering_audit_sha256": digest(verified_path), "pass_hashes": pass_hashes,
        "epsilon": epsilon, "E8": report["E8"], "failure": terminal["failure"],
        "all_readouts": numerical_summary(entries, epsilon),
        "by_readout_category": {name: numerical_summary(values, epsilon) for name, values in sorted(by_category.items())},
        "by_task": {name: numerical_summary(values, epsilon) for name, values in sorted(by_task.items())},
        "by_task_condition": {name: numerical_summary(values, epsilon) for name, values in sorted(by_condition.items())},
        "all_exceeded_readouts": exceeded, "exceeded_candidate_decompositions": decompositions,
        "candidate_length_distribution": dict(lengths),
        "baseline_repeat_maximum_absolute_error": verified["cohorts"]["validation"]["repeat"]["max_abs_error"],
        "baseline_cpu_reference_maximum_absolute_error": verified["cohorts"]["validation"]["reference"]["max_abs_error"],
        "baseline_shared_prefix_geometry": verified["cohorts"]["validation"]["reference_raw_audit"]["within_pass_prefix_geometry"],
        "remaining_validation_challenges": remaining,
        "assessment": {
            "implementation_mismatch_detected": False,
            "all_exceeded_candidate_observations_same_actual_gpu": all(entry["same_actual_gpu_uuid"] for entry in decompositions),
            "all_exceeded_sums_reconstructed_exactly": True,
            "all_exceeded_candidate_tokens_individually_within_epsilon": all(entry["all_individual_tokens_within_epsilon"] for entry in decompositions),
            "explanation": "Observed failure is consistent with residual FP32 shape-dependent score variation accumulating across answer tokens. Registered candidate/token/context associations, EOS separation, arithmetic aggregation, producer and shard identity all pass. This does not prove the mathematical correctness of every underlying GPU operation.",
            "identification_limit": "The failed B1/B4 comparisons change batch size, members, position, and padding together. Their contributions cannot be separated because the five remaining validation challenges were not run.",
            "fallback_status": "No fallback authorization, validation, or scientific result is inferred by this failure audit.",
        },
        "scope": {"gpu_used": False, "gold_read": False, "scientific_effects_computed": False,
                  "threshold_changed": False, "full_dev_started": False},
        "audit_source_sha256": {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
                                str((Path(__file__).parent / "independent_preflight_audit.py").relative_to(ROOT)):
                                    digest(Path(__file__).parent / "independent_preflight_audit.py")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--verified-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.verified_preflight)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"audit_passed": result["audit_passed"], "numerical_preflight_passed": False,
                      "epsilon": result["epsilon"], "all_readouts": result["all_readouts"], "output": str(args.output)}))
