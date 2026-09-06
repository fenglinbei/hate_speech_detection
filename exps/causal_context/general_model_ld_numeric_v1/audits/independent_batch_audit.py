"""Audit sealed batch compatibility with independent CPU arithmetic."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from tokenizers import Tokenizer

from independent_numeric_audit import canonical_hash, digest, read, recompute


LABELS = ("Racism", "Region", "LGBTQ", "Sexism", "others")
MODES = ("answer_sum", "answer_mean", "total_with_eos", "mean_with_eos")
CONDITIONS = ("C0", "CL", "CD", "CLD", "PL", "PD")


def flat(row, reference=False):
    result = {}
    candidate_scores = []
    for candidate in row["candidates"]:
        source = candidate["reference_scores"] if reference else candidate
        values = recompute(source["token_logprobs"], source["eos_logprob"])
        candidate_scores.append(values)
        prefix = "candidate/" + candidate["candidate_id"] + "/"
        for index, value in enumerate(source["token_logprobs"]):
            result[prefix + f"token/{index}"] = value
        for key, value in values.items():
            result[prefix + key] = value
    for mode in MODES:
        scores = np.asarray([value[mode] for value in candidate_scores], dtype=np.float64)
        if row["task"] == "hate":
            result[f"margin/{mode}/hate"] = float(scores[0] - scores[1])
        else:
            for bit, label in enumerate(LABELS):
                inside = (np.arange(32) & (1 << bit)) > 0
                result[f"margin/{mode}/{label}"] = float(logsumexp(scores[inside]) - logsumexp(scores[~inside]))
    return result


def summarize(items):
    magnitudes = np.abs(np.asarray([item[0] for item in items], dtype=np.float64))
    largest = int(magnitudes.argmax())
    value, record_id, metric = items[largest]
    return {"n": len(items), "nonzero": int(np.count_nonzero(magnitudes)),
            "above_1e4": int(np.count_nonzero(magnitudes > 1e-4)),
            "above_005": int(np.count_nonzero(magnitudes > .005)),
            "max_abs_error": float(magnitudes.max()), "mean_abs_error": float(magnitudes.mean()),
            "median_abs_error": float(np.median(magnitudes)),
            "p95_abs_error": float(np.quantile(magnitudes, .95)),
            "largest": {"record_id": record_id, "metric": metric, "difference": value}}


def compare(baseline, observed, runtime_difference_path, reference=False):
    runtime = read(runtime_difference_path)
    runtime_rows = {row["record_id"]: row["differences"] for row in runtime["blocks"]}
    assert set(runtime_rows) == set(observed)
    grouped = defaultdict(list)
    largest_detail = None
    maximum = -1
    max_runtime_formula_discrepancy = 0
    for record_id, row in observed.items():
        before = baseline[record_id]
        first = flat(before, reference=reference)
        second = flat(row)
        assert first.keys() == second.keys() == runtime_rows[record_id].keys()
        for metric, value in second.items():
            difference = value - first[metric]
            max_runtime_formula_discrepancy = max(max_runtime_formula_discrepancy,
                                                  abs(difference - runtime_rows[record_id][metric]))
            family = ("candidate/token" if "/token/" in metric else
                      "candidate/" + metric.split("/")[-1] if metric.startswith("candidate/") else
                      "/".join(metric.split("/")[:2]))
            item = (difference, record_id, metric)
            for key in ("all", "task/" + row["task"], "condition/" + row["condition"],
                        "task_condition/" + row["task"] + "/" + row["condition"], "metric/" + family,
                        "task_metric/" + row["task"] + "/" + family):
                grouped[key].append(item)
            if abs(difference) > maximum:
                maximum = abs(difference)
                largest_detail = {"record_id": record_id, "metric": metric, "difference": difference,
                                  "baseline": first[metric], "observed": value}
                if metric.startswith("candidate/"):
                    candidate_id = metric.split("/")[1]
                    a = next(candidate for candidate in before["candidates"] if candidate["candidate_id"] == candidate_id)
                    b = next(candidate for candidate in row["candidates"] if candidate["candidate_id"] == candidate_id)
                    largest_detail.update(
                        canonical_answer=b["canonical_answer"], answer_token_ids=b["answer_token_ids"],
                        baseline_token_logprobs=a["token_logprobs"], observed_token_logprobs=b["token_logprobs"],
                        token_differences=(np.asarray(b["token_logprobs"]) - np.asarray(a["token_logprobs"])).tolist(),
                        baseline_eos_logprob=a["eos_logprob"], observed_eos_logprob=b["eos_logprob"],
                        prompt_token_ids_sha256=b["prompt_token_ids_sha256"], prompt_tokens=b["prompt_tokens"],
                        sequence_tokens=b["sequence_tokens"], padded_sequence_tokens=b["padded_sequence_tokens"],
                        batch_members=b["batch_members"], batch_member_ordinal=b["batch_member_ordinal"])
    assert max_runtime_formula_discrepancy < 1e-10
    assert math.isclose(maximum, runtime["max_abs_error"], abs_tol=1e-10, rel_tol=0)
    return {"summary": summarize(grouped["all"]), "largest_detail": largest_detail,
            "independent_vs_runtime_formula_max_error": max_runtime_formula_discrepancy,
            "by_dimension": {key: summarize(items) for key, items in sorted(grouped.items()) if key != "all"}}


def audit(root, run, plan_ref):
    ref = read(plan_ref)
    plan_path = Path(ref["target_path"]) / "plan.json"
    assert digest(plan_path) == ref["plan_sha256"]
    plan = read(plan_path)
    assert plan["plan_id"] == "gmlnum-" + canonical_hash({k: v for k, v in plan.items() if k != "plan_id"})
    report_path = run / "preflight/preflight_report.json"
    report = read(report_path)
    terminal = read(run / "run_manifest.json")
    assert digest(report_path) == terminal["preflight_report_sha256"]
    assert terminal["plan_id"] == report["plan_id"] == plan["plan_id"]
    for name, checksum in report["files"].items():
        path = run / "preflight" / name
        assert path.resolve().is_relative_to((run / "preflight").resolve())
        assert digest(path) == checksum
    contexts = {row["record_id"]: row for row in
                map(json.loads, (Path(plan["package_path"]) / "contexts.dev.jsonl").read_text().splitlines())}
    tokenizer = Tokenizer.from_file(str(root / "models/base/Qwen3-8B/tokenizer.json"))
    regression_ids = set(plan["cohorts"]["regression"])
    expected_full = [row for row in plan["blocks"] if row["query_id"] in regression_ids]
    tail_records = [next(row["record_id"] for row in reversed(expected_full)
                         if row["task"] == "hate" and row["condition"] == condition) for condition in CONDITIONS]
    passes = {}
    pass_summaries = {}
    for name in ("regression-b1-r0", "regression-b1-r1", "regression-b4-r0", "regression-b4-tail2"):
        directory = run / "preflight" / name
        manifest = read(directory / "manifest.json")
        raw = directory / "scores.jsonl"
        assert digest(raw) == manifest["scores_sha256"]
        assert manifest["status"] == "complete" and manifest["identity"]["plan_id"] == plan["plan_id"]
        assert manifest["identity"]["runtime"] == report["runtime_identity"]
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        expected = ([entry for entry in expected_full if entry["record_id"] in tail_records]
                    if "tail" in name else expected_full)
        record_ids = [row["record_id"] for row in rows]
        assert record_ids == [entry["record_id"] for entry in expected] == manifest["identity"]["records"]
        assert len(rows) == len(set(record_ids)) == manifest["blocks"]
        batch_size = 1 if "b1" in name else 4
        flat_members = []
        for row, descriptor in zip(rows, expected, strict=True):
            context = contexts[row["record_id"]]
            assert row["plan_id"] == plan["plan_id"]
            assert row["runtime_sha256"] == canonical_hash(manifest["identity"]["runtime"])
            assert row["execution_batch_size"] == batch_size and row["pass_name"] == name
            for key in ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256"):
                assert row[key] == context[key] == descriptor[key]
            assert hashlib.sha256(context["prompt_text"].encode()).hexdigest() == row["prompt_sha256"]
            prompt_tokens = tokenizer.encode(context["prompt_text"], add_special_tokens=False).ids
            assert canonical_hash(prompt_tokens) == descriptor["prompt_token_ids_sha256"]
            assert len(prompt_tokens) == descriptor["prompt_tokens"]
            assert len(row["candidates"]) == len(plan["catalog"][row["task"]])
            for candidate, registered in zip(row["candidates"], plan["catalog"][row["task"]], strict=True):
                assert all(candidate[key] == value for key, value in registered.items())
                answer_tokens = tokenizer.encode(candidate["canonical_answer"], add_special_tokens=False).ids
                assert answer_tokens == candidate["answer_token_ids"]
                assert tokenizer.encode(context["prompt_text"] + candidate["canonical_answer"], add_special_tokens=False).ids == prompt_tokens + answer_tokens
                assert len(answer_tokens) == len(candidate["token_logprobs"]) == candidate["answer_tokens"]
                assert candidate["prompt_tokens"] == len(prompt_tokens)
                assert candidate["prompt_token_ids_sha256"] == descriptor["prompt_token_ids_sha256"]
                assert candidate["eos_token_id"] == plan["eos_token_id"] == 151645
                assert candidate["sequence_tokens"] == len(prompt_tokens) + len(answer_tokens) + 1
                assert candidate["causal_shift"] == 1 and candidate["use_cache"] is False
                assert candidate["padding_side"] == "right" and candidate["token_boundary_checked"]
                assert candidate["reference_checked"] == name.endswith("b1-r0")
                computed = recompute(candidate["token_logprobs"], candidate["eos_logprob"])
                assert candidate["scores"] == computed
                assert all(candidate[key] == value for key, value in computed.items())
                flat_members.append((row, candidate))
        assert len(flat_members) == manifest["candidates"]
        # Geometry is reconstructed from the frozen ordered records, not raw batch metadata.
        batches = []
        for position in range(0, len(flat_members)):
            row, candidate = flat_members[position]
            if candidate["ordinal"] != 0:
                continue
            if batch_size == 1:
                batches.extend([[entry] for entry in flat_members[position:position + len(row["candidates"])]])
            elif row["task"] == "group":
                batches.extend([flat_members[start:start + 4] for start in range(position, position + 32, 4)])
            else:
                if batches and any(member[0]["record_id"] == row["record_id"] for member in batches[-1]):
                    continue
                take = 2
                if position + 2 < len(flat_members):
                    neighbor = flat_members[position + 2][0]
                    if (neighbor["task"], neighbor["condition"]) == (row["task"], row["condition"]):
                        take = 4
                batches.append(flat_members[position:position + take])
        for ordinal, members in enumerate(batches):
            expected_members = [row["record_id"] + ":" + candidate["candidate_id"] for row, candidate in members]
            max_length = max(candidate["sequence_tokens"] for row, candidate in members)
            for position, (row, candidate) in enumerate(members):
                assert candidate["batch_ordinal"] == ordinal
                assert candidate["batch_member_ordinal"] == position
                assert candidate["batch_members"] == expected_members
                assert candidate["batch_size"] == candidate["effective_batch_size"] == len(members)
                assert candidate["padded_sequence_tokens"] == max_length <= 8192
        passes[name] = {row["record_id"]: row for row in rows}
        pass_summaries[name] = {"blocks": len(rows), "candidates": len(flat_members), "batches": len(batches),
                                "manifest_sha256": digest(directory / "manifest.json"), "scores_sha256": digest(raw),
                                "all_prompt_candidate_token_and_geometry_checks_passed": True}
    baseline = passes["regression-b1-r0"]
    comparisons = {}
    for name, target, filename, is_reference in (
        ("baseline_repeat", "regression-b1-r1", "regression-repeat-differences.json", False),
        ("cpu_reference", "regression-b1-r0", "regression-reference-differences.json", True),
        ("batch4", "regression-b4-r0", "regression-batch-differences.json", False),
        ("tail2", "regression-b4-tail2", "regression-tail-differences.json", False),
    ):
        comparisons[name] = compare(baseline, passes[target], run / "preflight" / filename, reference=is_reference)
    assert comparisons["baseline_repeat"]["summary"]["max_abs_error"] == 0
    e8 = max(comparisons["batch4"]["summary"]["max_abs_error"], comparisons["tail2"]["summary"]["max_abs_error"])
    epsilon = min(.005, max(.0001, 2 * e8))
    assert e8 == report["E8"] and epsilon == report["epsilon"]
    assert report["passed"] is False and report["validation_executed"] is False
    assert report["failure"] == "regression-batch-compatibility-gate"
    assert terminal["status"] == "preflight_failed" and terminal["full_dev_started"] is False
    assert terminal["analysis_published"] is False
    assert not (run / "dev-b4").exists() and not (run / "analysis").exists()
    assert not any(path.name.startswith("validation-") for path in (run / "preflight").iterdir())
    for name, checksum in report["files"].items():
        assert digest(run / "preflight" / name) == checksum
    return {
        "schema_version": "independent-numeric-batch-audit/v1", "audit_passed": True,
        "preflight_numerical_passed": False, "plan_id": plan["plan_id"],
        "preflight_report_sha256": digest(report_path), "run_manifest_sha256": digest(run / "run_manifest.json"),
        "audit_source_sha256": digest(Path(__file__)),
        "audit_helper_sha256": digest(Path(__file__).with_name("independent_numeric_audit.py")),
        "passes": pass_summaries, "comparisons": comparisons, "E8": e8, "epsilon": epsilon,
        "checks": {"all_report_file_hashes_match": True, "baseline_repetition_exact": True,
                   "full_frame_and_tail_match_frozen_inputs": True, "token_boundaries_and_candidate_identities_match": True,
                   "batch_members_and_padding_geometry_match": True, "all_saved_differences_independently_reproduced": True,
                   "failed_gate_and_stop_before_validation_and_dev_confirmed": True},
        "scope": {"query_gold_read": False, "gpu_used": False, "scientific_effects_computed": False},
        "conclusion": "The large accelerated-batch differences exist in stored per-token scores with matching prompts, candidates and batch geometry. No artifact association or score aggregation mismatch was found. Saved target logprobs do not identify which model forward operation caused the batch dependence.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--plan-ref", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.run.resolve(), args.plan_ref.resolve())
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"audit_passed": result["audit_passed"], "E8": result["E8"], "epsilon": result["epsilon"],
                      "largest": result["comparisons"]["batch4"]["largest_detail"], "path": str(args.output)}))
