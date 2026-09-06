"""Read-only raw audit with independent NumPy/SciPy math and no gold access."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from tokenizers import Tokenizer


def digest(path):
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def recompute(tokens, eos):
    tokens = [float(value) for value in tokens]
    assert tokens and all(math.isfinite(value) and value <= 1e-9 for value in tokens)
    assert math.isfinite(eos) and eos <= 1e-9
    total = math.fsum(tokens)
    with_eos = math.fsum([*tokens, eos])
    return {"answer_sum": total, "answer_mean": total / len(tokens), "eos_logprob": eos,
            "total_with_eos": with_eos, "mean_with_eos": with_eos / (len(tokens) + 1)}


def audit(root, pass_dir, plan_ref):
    labels = ("Racism", "Region", "LGBTQ", "Sexism", "others")
    modes = ("answer_sum", "answer_mean", "total_with_eos", "mean_with_eos")
    reference = read(plan_ref)
    plan_path = Path(reference["target_path"]) / "plan.json"
    assert digest(plan_path) == reference["plan_sha256"]
    plan = read(plan_path)
    assert plan["plan_id"] == reference["plan_id"]
    assert plan["plan_id"] == "gmlnum-" + canonical_hash({k: v for k, v in plan.items() if k != "plan_id"})
    manifest_path = pass_dir / "manifest.json"
    raw_path = pass_dir / "scores.jsonl"
    manifest_hash = digest(manifest_path)
    manifest = read(manifest_path)
    raw_hash = digest(raw_path)
    assert raw_hash == manifest["scores_sha256"]
    assert manifest["status"] == "complete"
    assert manifest["identity"]["plan_id"] == plan["plan_id"]
    assert manifest["identity"]["batch_size"] == 1 and manifest["identity"]["reference"] is True
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    cohort = pass_dir.name.split("-")[0]
    ids = set(plan["cohorts"][cohort])
    expected = [entry for entry in plan["blocks"] if entry["query_id"] in ids]
    assert [row["record_id"] for row in rows] == [entry["record_id"] for entry in expected]
    assert manifest["identity"]["records"] == [entry["record_id"] for entry in expected]
    assert len(rows) == len(ids) * 12 == manifest["blocks"]
    tokenizer_path = root / "models/base/Qwen3-8B/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    maxima = {}
    counts = Counter()
    lengths = {"hate": Counter(), "group": Counter()}

    def observe(family, error, record, metric):
        counts[family] += 1
        assert math.isfinite(error)
        item = {"max_abs_error": abs(error), "record_id": record,
                "metric": metric, "difference": error}
        if family not in maxima or item["max_abs_error"] > maxima[family]["max_abs_error"]:
            maxima[family] = item

    ordinal = 0
    for row, context in zip(rows, expected, strict=True):
        for key in ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256"):
            assert row[key] == context[key]
        assert row["plan_id"] == plan["plan_id"]
        assert row["runtime_sha256"] == canonical_hash(manifest["identity"]["runtime"])
        assert row["execution_batch_size"] == 1 and row["cohort"] == cohort
        assert row["pass_name"] == pass_dir.name and row["repetition"] == 0
        task = row["task"]
        candidates = row["candidates"]
        assert len(candidates) == (2 if task == "hate" else 32)
        values = {mode: [] for mode in modes}
        reference_values = {mode: [] for mode in modes}
        for index, candidate in enumerate(candidates):
            expected_labels = (["hate", "non-hate"][index:index + 1] if task == "hate" else
                               [label for bit, label in enumerate(labels) if index & (1 << bit)])
            answer = json.dumps(expected_labels[0] if task == "hate" else expected_labels,
                                separators=(",", ":"))
            candidate_id = expected_labels[0] if task == "hate" else f"group-{index:02d}"
            assert candidate["candidate_id"] == candidate_id and candidate["ordinal"] == index
            assert candidate["labels"] == expected_labels and candidate["canonical_answer"] == answer
            assert candidate["label_bitmap"] == (None if task == "hate" else index)
            assert candidate["cardinality"] == (None if task == "hate" else index.bit_count())
            assert candidate["answer_token_ids"] == tokenizer.encode(answer, add_special_tokens=False).ids
            assert candidate["answer_token_ids_sha256"] == canonical_hash(candidate["answer_token_ids"])
            assert candidate["answer_sha256"] == hashlib.sha256(answer.encode()).hexdigest()
            assert all(candidate[key] == value for key, value in plan["catalog"][task][index].items())
            assert candidate["eos_token_id"] == plan["eos_token_id"] == 151645
            assert plan["pad_token_id"] == 151643
            n_tokens = len(candidate["answer_token_ids"])
            assert n_tokens == len(candidate["token_logprobs"]) == candidate["answer_tokens"]
            assert candidate["prompt_tokens"] == context["prompt_tokens"]
            assert candidate["prompt_token_ids_sha256"] == context["prompt_token_ids_sha256"]
            assert candidate["sequence_tokens"] == candidate["prompt_tokens"] + n_tokens + 1
            assert candidate["padded_sequence_tokens"] == candidate["sequence_tokens"] <= 8192
            assert candidate["batch_ordinal"] == ordinal
            assert candidate["batch_members"] == [row["record_id"] + ":" + candidate_id]
            assert candidate["effective_batch_size"] == candidate["batch_size"] == 1
            assert candidate["batch_member_ordinal"] == 0
            assert candidate["causal_shift"] == 1 and candidate["use_cache"] is False
            assert candidate["token_boundary_checked"] and candidate["finite_target_logits_checked"]
            ordinal += 1
            lengths[task][n_tokens] += 1
            counts["answer_tokens"] += n_tokens
            counts["candidates"] += 1
            computed = recompute(candidate["token_logprobs"], candidate["eos_logprob"])
            cpu = candidate["reference_scores"]
            assert candidate["reference_checked"] and candidate["reference_arithmetic_dtype"] == "cpu.torch.float64"
            assert cpu["token_logprobs"] == candidate["reference_token_logprobs"]
            assert cpu["eos_logprob"] == candidate["reference_eos_logprob"]
            recomputed_cpu = recompute(cpu["token_logprobs"], cpu["eos_logprob"])
            assert len(cpu["token_logprobs"]) == n_tokens
            local_errors = []
            for key, value in computed.items():
                assert candidate["scores"][key] == value == candidate[key]
                assert cpu[key] == recomputed_cpu[key]
                error = value - recomputed_cpu[key]
                local_errors.append(error)
                assert candidate["reference_differences"][key] == error
                observe("candidate_" + key, error, row["record_id"], candidate_id)
                observe("all_registered_readouts", error, row["record_id"], candidate_id + "/" + key)
            token_errors = np.asarray(candidate["token_logprobs"], dtype=np.float64) - np.asarray(cpu["token_logprobs"], dtype=np.float64)
            np.testing.assert_array_equal(token_errors, candidate["reference_differences"]["token_logprobs"])
            for position, error in enumerate(token_errors):
                local_errors.append(float(error))
                observe("answer_token_logprob", float(error), row["record_id"], candidate_id + f"/token/{position}")
                observe("all_registered_readouts", float(error), row["record_id"], candidate_id + f"/token/{position}")
            assert candidate["reference_abs_error_max"] == max(abs(error) for error in local_errors)
            for mode in modes:
                values[mode].append(computed[mode])
                reference_values[mode].append(recomputed_cpu[mode])
        for mode in modes:
            gpu_scores = np.asarray(values[mode], dtype=np.float64)
            cpu_scores = np.asarray(reference_values[mode], dtype=np.float64)
            if task == "hate":
                margins = [("hate", gpu_scores[0] - gpu_scores[1], cpu_scores[0] - cpu_scores[1])]
            else:
                masks = np.arange(32)
                margins = []
                for bit, label in enumerate(labels):
                    includes = (masks & (1 << bit)) > 0
                    margins.append((label, float(logsumexp(gpu_scores[includes]) - logsumexp(gpu_scores[~includes])),
                                    float(logsumexp(cpu_scores[includes]) - logsumexp(cpu_scores[~includes]))))
            for label, observed, cpu in margins:
                error = observed - cpu
                observe("margin_" + mode, error, row["record_id"], label)
                observe("all_registered_readouts", error, row["record_id"], mode + "/margin/" + label)
    assert ordinal == manifest["candidates"] == len(ids) * 6 * 34
    assert digest(raw_path) == raw_hash and digest(manifest_path) == manifest_hash
    limit = plan["config"]["validation"]["reference_abs_tolerance"]
    assert limit == 1e-4
    return {
        "schema_version": "independent-numeric-raw-audit/v1", "passed": maxima["all_registered_readouts"]["max_abs_error"] <= limit,
        "plan_id": plan["plan_id"], "pass_path": str(pass_dir), "blocks": len(rows),
        "manifest_sha256": manifest_hash, "scores_sha256": raw_hash,
        "tokenizer_json_sha256": digest(tokenizer_path), "audit_source_sha256": digest(Path(__file__)),
        "counts": dict(counts), "answer_token_length_histograms": {task: dict(hist) for task, hist in lengths.items()},
        "reference_abs_tolerance": limit, "maxima": maxima,
        "checks": {"manifest_and_plan_hashes": True, "complete_frozen_frame": True,
                   "canonical_candidates_and_independent_tokenizer_ids": True, "causal_boundary_and_batch_metadata": True,
                   "math_fsum_recomputation_exact": True, "eos_excluded_from_primary_sum": True,
                   "stored_reference_recomputation_exact": True, "reference_differences_recomputed_exact": True,
                   "all_margins_independently_recomputed_with_scipy": True, "raw_unchanged_during_audit": True},
        "scope": {"query_gold_read": False, "test_content_read": False, "gpu_used": False,
                  "scientific_effects_analyzed": False, "imports_runtime_comparison_code": False},
        "limitation": "Full vocabulary logits were intentionally not stored; this independently verifies saved CPU reference values, score aggregation and error propagation, not a fresh normalization of the original forward logits.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pass_dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--plan-ref", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.pass_dir.resolve(), args.plan_ref.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"audit_path": str(args.output), "passed": result["passed"],
                      "blocks": result["blocks"], "candidates": result["counts"]["candidates"],
                      "max_abs_error": result["maxima"]["all_registered_readouts"]["max_abs_error"]}))
