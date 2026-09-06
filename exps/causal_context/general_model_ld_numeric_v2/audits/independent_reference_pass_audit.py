"""Audit one sealed B1 CPU-reference pass without gold or scientific effects."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[4]
HELPERS = ROOT / "exps/causal_context/general_model_ld_numeric_v1/audits"
sys.path.insert(0, str(HELPERS))
from independent_full_dev_audit import expected_catalog, load_effective_plan, margin_values
from independent_numeric_audit import canonical_hash, digest, read, recompute
from independent_parallel_audit import audit_parallel_pass


def prefix_geometry(rows):
    unrestricted = defaultdict(list)
    fixed = defaultdict(list)
    for row in rows:
        for candidate in row["candidates"]:
            targets = candidate["answer_token_ids"] + [candidate["eos_token_id"]]
            probabilities = candidate["token_logprobs"] + [candidate["eos_logprob"]]
            for position, probability in enumerate(probabilities):
                prefix_and_target = tuple(targets[:position + 1])
                key = (row["record_id"], prefix_and_target)
                item = {"candidate_id": candidate["candidate_id"], "token_position_zero_based": position,
                        "sequence_tokens": candidate["sequence_tokens"],
                        "padded_sequence_tokens": candidate["padded_sequence_tokens"],
                        "physical_gpu_index": candidate["physical_gpu_index"], "logprob": probability}
                unrestricted[key].append(item)
                fixed[(key, candidate["padded_sequence_tokens"], candidate["effective_batch_size"],
                       candidate["physical_gpu_uuid"])].append(item)

    def summarize(groups, *, fixed_geometry):
        reports = []
        for key, observations in groups.items():
            if len(observations) < 2:
                continue
            minimum = min(observations, key=lambda item: item["logprob"])
            maximum = max(observations, key=lambda item: item["logprob"])
            record, prefix = key[0] if fixed_geometry else key
            reports.append({"record_id": record, "answer_prefix_including_target": list(prefix),
                            "observations": len(observations),
                            "distinct_sequence_lengths": len({item["sequence_tokens"] for item in observations}),
                            "spread": maximum["logprob"] - minimum["logprob"],
                            "minimum": minimum, "maximum": maximum})
        reports.sort(key=lambda item: item["spread"], reverse=True)
        spreads = np.asarray([item["spread"] for item in reports])
        return {"shared_prefix_groups": len(reports), "maximum": reports[0] if reports else None,
                "maximum_spread": float(spreads.max()) if len(spreads) else 0,
                "nonzero_spread_groups": int(np.count_nonzero(spreads)),
                "spread_quantiles": dict(zip(("q50", "q90", "q99"), np.quantile(spreads, [.5, .9, .99]).tolist())),
                "groups_with_multiple_sequence_lengths": sum(item["distinct_sequence_lengths"] > 1 for item in reports),
                "largest_five": reports[:5]}

    return {"grouping": "same-context-and-prompt-plus-same-preceding-answer-tokens-and-target-token",
            "across_full_candidate_lengths": summarize(unrestricted, fixed_geometry=False),
            "fixed_padded_length_batch_size_and_physical_gpu": summarize(fixed, fixed_geometry=True),
            "interpretation": "Descriptive within-pass numerical geometry check, not a substitute for registered prefix challenges or E8."}


def audit(directory, reference):
    plan = load_effective_plan(reference, root=ROOT)
    manifest_path, scores_path = directory / "manifest.json", directory / "scores.jsonl"
    manifest_sha, scores_sha = digest(manifest_path), digest(scores_path)
    manifest = read(manifest_path)
    assert manifest["status"] == "complete" and manifest["scores_sha256"] == scores_sha
    assert manifest["query_gold_loaded"] is False and manifest["test_content_read"] is False
    identity = manifest["identity"]
    assert identity["batch_size"] == 1 and identity["reference"] is True
    assert identity["pass_name"] == directory.name
    assert identity["scoring_profile"] == {"candidate_permutation": "canonical", "padding_extra": 0,
                                           "prefix": False, "replica_shift": 0}
    cohort = directory.name.split("-")[0]
    selected_ids = set(plan["cohorts"][cohort])
    contexts = [entry for entry in plan["blocks"] if entry["query_id"] in selected_ids]
    rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
    assert len(rows) == len(selected_ids) * 12 == manifest["blocks"]
    shard_audit = audit_parallel_pass(directory, contexts, plan, merged_rows=rows)

    package = Path(plan["package_path"])
    package_manifest = read(package / "manifest.json")
    assert digest(package / "manifest.json") == plan["package_manifest_sha256"]
    prompt_path = package / "contexts.dev.jsonl"
    expected_prompt_sha = next(item["sha256"] for item in package_manifest["files"] if item["path"] == prompt_path.name)
    assert digest(prompt_path) == expected_prompt_sha
    selected_records = {entry["record_id"] for entry in contexts}
    stored_prompts = {}
    with prompt_path.open() as handle:
        for line in handle:
            entry = json.loads(line)
            if entry["record_id"] in selected_records:
                assert entry["record_id"] not in stored_prompts
                stored_prompts[entry["record_id"]] = entry
    assert set(stored_prompts) == selected_records
    tokenizer_path = ROOT / "models/base/Qwen3-8B/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    maxima, counts = {}, Counter()
    lengths = {"hate": Counter(), "group": Counter()}
    worker_ordinals = Counter()
    runtime_sha = canonical_hash(identity["runtime"])

    def observe(family, error, row, metric):
        error = float(error)
        assert math.isfinite(error)
        counts[family] += 1
        item = {"max_abs_error": abs(error), "record_id": row["record_id"],
                "metric": metric, "difference": error}
        if family not in maxima or abs(error) > maxima[family]["max_abs_error"]:
            maxima[family] = item

    for row, context in zip(rows, contexts, strict=True):
        assert all(row[key] == context[key] for key in ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256"))
        assert row["plan_id"] == plan["plan_id"] and row["runtime_sha256"] == runtime_sha
        assert row["execution_batch_size"] == 1 and row["cohort"] == cohort
        assert row["pass_name"] == directory.name and row["repetition"] == 0
        prompt = stored_prompts[row["record_id"]]
        assert prompt["context_sha256"] == canonical_hash({key: value for key, value in prompt.items() if key != "context_sha256"})
        assert all(prompt[key] == value for key, value in context.items())
        assert hashlib.sha256(prompt["prompt_text"].encode()).hexdigest() == context["prompt_sha256"]
        prompt_ids = tokenizer.encode(prompt["prompt_text"], add_special_tokens=False).ids
        assert canonical_hash(prompt_ids) == context["prompt_token_ids_sha256"]
        assert len(prompt_ids) == context["prompt_tokens"]
        task = row["task"]
        candidates = row["candidates"]
        assert len(candidates) == (2 if task == "hate" else 32)
        cpu_candidates, observed_candidates = [], []
        for candidate, expected, frozen in zip(candidates, expected_catalog(task), plan["catalog"][task], strict=True):
            assert all(candidate[key] == value for key, value in expected.items())
            assert all(candidate[key] == value for key, value in frozen.items())
            answer = candidate["canonical_answer"]
            token_ids = tokenizer.encode(answer, add_special_tokens=False).ids
            assert candidate["answer_token_ids"] == token_ids
            assert tokenizer.encode(prompt["prompt_text"] + answer, add_special_tokens=False).ids == prompt_ids + token_ids
            assert canonical_hash(token_ids) == candidate["answer_token_ids_sha256"]
            assert hashlib.sha256(answer.encode()).hexdigest() == candidate["answer_sha256"] == candidate["canonical_answer_sha256"]
            assert candidate["eos_token_id"] == plan["eos_token_id"] == 151645
            assert plan["pad_token_id"] == 151643 and candidate["eos_token_id"] not in token_ids
            size = len(token_ids)
            assert size == len(candidate["token_logprobs"]) == candidate["answer_tokens"]
            assert candidate["prompt_tokens"] == len(prompt_ids)
            assert candidate["prompt_token_ids_sha256"] == context["prompt_token_ids_sha256"]
            assert candidate["sequence_tokens"] == len(prompt_ids) + size + 1
            assert candidate["padded_sequence_tokens"] == candidate["sequence_tokens"] <= 8192
            physical = candidate["physical_gpu_index"]
            assert candidate["batch_ordinal"] == worker_ordinals[physical]
            worker_ordinals[physical] += 1
            assert candidate["batch_members"] == [row["record_id"] + ":" + candidate["candidate_id"]]
            assert candidate["batch_member_ordinal"] == 0
            assert candidate["effective_batch_size"] == candidate["batch_size"] == 1
            assert candidate["causal_shift"] == 1 and candidate["use_cache"] is False
            assert candidate["padding_side"] == "right" and candidate["prefix_reference"] is False
            assert candidate["token_boundary_checked"] and candidate["finite_target_logits_checked"]
            assert candidate["model_logits_dtype"] == candidate["logprob_arithmetic_dtype"] == "torch.float32"
            counts["answer_tokens"] += size
            counts["candidates"] += 1
            lengths[task][size] += 1
            computed = recompute(candidate["token_logprobs"], candidate["eos_logprob"])
            cpu = candidate["reference_scores"]
            assert candidate["reference_checked"] and candidate["reference_arithmetic_dtype"] == "cpu.torch.float64"
            assert cpu["token_logprobs"] == candidate["reference_token_logprobs"]
            assert cpu["eos_logprob"] == candidate["reference_eos_logprob"]
            recomputed_cpu = recompute(cpu["token_logprobs"], cpu["eos_logprob"])
            assert len(cpu["token_logprobs"]) == size
            local_errors = []
            for name, value in computed.items():
                assert candidate["scores"][name] == value == candidate[name]
                assert cpu[name] == recomputed_cpu[name]
                error = value - recomputed_cpu[name]
                assert candidate["reference_differences"][name] == error
                local_errors.append(error)
                observe("candidate_" + name, error, row, candidate["candidate_id"])
                observe("all_registered_readouts", error, row, candidate["candidate_id"] + "/" + name)
            token_errors = np.asarray(candidate["token_logprobs"]) - np.asarray(cpu["token_logprobs"])
            np.testing.assert_array_equal(token_errors, candidate["reference_differences"]["token_logprobs"])
            for position, error in enumerate(token_errors):
                local_errors.append(float(error))
                metric = candidate["candidate_id"] + f"/token/{position}"
                observe("answer_token_logprob", error, row, metric)
                observe("all_registered_readouts", error, row, metric)
            assert candidate["reference_abs_error_max"] == max(abs(error) for error in local_errors)
            observed_candidates.append({"scores": computed})
            cpu_candidates.append({"scores": recomputed_cpu})
        observed_margins = margin_values(task, observed_candidates)
        cpu_margins = margin_values(task, cpu_candidates)
        for (mode, label), value in observed_margins.items():
            error = value - cpu_margins[(mode, label)]
            observe("margin_" + mode, error, row, label)
            observe("all_registered_readouts", error, row, mode + "/margin/" + label)

    assert counts["candidates"] == manifest["candidates"] == len(selected_ids) * 6 * 34
    assert digest(scores_path) == scores_sha and digest(manifest_path) == manifest_sha
    tolerance = plan["config"]["validation"]["reference_abs_tolerance"]
    assert tolerance == .0001
    return {
        "schema_version": "independent-reference-pass-audit/v2",
        "passed": maxima["all_registered_readouts"]["max_abs_error"] <= tolerance,
        "plan_id": plan["plan_id"], "pass_path": str(directory.resolve()), "blocks": len(rows),
        "manifest_sha256": manifest_sha, "scores_sha256": scores_sha,
        "tokenizer_json_sha256": digest(tokenizer_path), "gold_free_contexts_sha256": expected_prompt_sha,
        "counts": dict(counts), "answer_token_length_histograms": {task: dict(hist) for task, hist in lengths.items()},
        "candidates_per_physical_gpu": dict(worker_ordinals), "reference_abs_tolerance": tolerance,
        "maxima": maxima, "parallel_identity_and_shards": shard_audit,
        "within_pass_prefix_geometry": prefix_geometry(rows),
        "checks": {"plan_and_source_hashes": True, "complete_frozen_frame": True,
                   "canonical_candidates_and_independent_tokenizer_ids": True,
                   "gold_free_prompt_context_hashes_tokenization_and_all_candidate_boundaries": True,
                   "causal_boundary_and_worker_batch_metadata": True, "math_fsum_recomputation_exact": True,
                   "eos_excluded_from_primary_sum": True, "stored_reference_recomputation_exact": True,
                   "reference_differences_recomputed_exact": True, "all_margins_independently_recomputed_with_scipy": True,
                   "raw_unchanged_during_audit": True},
        "scope": {"query_gold_read": False, "test_content_read": False, "gpu_used": False,
                  "scientific_effects_analyzed": False, "imports_runtime_comparison_code": False},
        "limitations": ["Full vocabulary logits are intentionally not stored; this verifies saved CPU reference values, aggregation and error propagation, not a fresh normalization of original forward logits.",
                        "Single-pass prefix geometry does not replace the registered six-challenge calibration and held-out validation gates."],
        "audit_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), Path(__file__).with_name("independent_parallel_audit.py"),
            HELPERS / "independent_full_dev_audit.py", HELPERS / "independent_numeric_audit.py")},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.pass_dir, args.plan)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "blocks": result["blocks"], "counts": result["counts"],
                      "reference_max": result["maxima"]["all_registered_readouts"],
                      "prefix_spread_max": result["within_pass_prefix_geometry"]["across_full_candidate_lengths"]["maximum_spread"],
                      "output": str(args.output)}))
