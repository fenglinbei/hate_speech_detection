"""Read sealed engineering cohorts using independent math and shard auditors."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
V1 = ROOT / "exps/causal_context/general_model_ld_numeric_v1/audits"
V2 = ROOT / "exps/causal_context/general_model_ld_numeric_v2/audits"
V3 = ROOT / "exps/causal_context/general_model_ld_numeric_v3/audits"
sys.path[:0] = [str(V1), str(V2), str(V3)]
from independent_numeric_audit import canonical_hash, digest, read, recompute
from independent_parallel_audit import audit_parallel_pass, validate_cross_gpu_challenge
from independent_preflight_audit import compare, validate_geometry
from independent_fallback_preflight_audit import verify_replica_proof


DEFINITIONS = (("baseline", "r0"), ("repeat", "r1"), ("padding", "padding"),
               ("prefix", "prefix"), ("members", "members"), ("replica", "replica"))
FIELDS = ("answer_sum", "answer_mean", "eos_logprob", "total_with_eos", "mean_with_eos")


def audit(reference, run, cohort):
    ref = read(reference)
    directory = Path(ref["target_path"])
    assert digest(directory / "plan.json") == ref["plan_sha256"]
    plan = read(directory / "plan.json")
    assert plan["plan_id"] == ref["plan_id"] == "gmlcoverage-" + canonical_hash(
        {key: value for key, value in plan.items() if key != "plan_id"})
    assert plan["numeric_policy"] == {"E8": .00067138671875, "epsilon": .0013427734375,
        "repeat_abs_tolerance": .0001, "reference_abs_tolerance": .0001, "padding_extra": 64, "replica_shift": 1}
    for name, checksum in plan["code_sha256"].items():
        assert digest(ROOT / name) == digest(directory / "source" / name) == checksum
    for name, checksum in plan["input_files"].items():
        assert digest(directory / name) == checksum
    assert len(plan["cohorts"][cohort]) in ({8} if cohort == "regression" else {24} if cohort == "validation" else {1, 2, 3, 4})
    contexts = [row for row in plan["blocks"] if row["query_id"] in plan["cohorts"][cohort]]
    assert len(contexts) == len(plan["cohorts"][cohort]) * 16
    assert {row["record_id"] for row in contexts} == {f"{qid}:{task}:{condition}"
        for qid in plan["cohorts"][cohort] for task in ["hate", "group"] for condition in plan["config"]["conditions"]}
    runtime = read(run / "runtime_identity.json")
    runtime_hash = canonical_hash(runtime)
    preflight = run / "preflight"
    passes, checked = {}, {}
    evidence = {"runtime_identity.json": digest(run / "runtime_identity.json")}
    for label, suffix in DEFINITIONS:
        path = preflight / f"{cohort}-b1-{suffix}"
        manifest = read(path / "manifest.json")
        identity = manifest["identity"]
        profile = {"candidate_permutation": "group-rotate-one-then-reverse-hate-reverse" if label == "members" else "canonical",
                   "padding_extra": 64 if label == "padding" else 0,
                   "prefix": label == "prefix", "replica_shift": 1 if label == "replica" else 0}
        assert identity["scoring_profile"] == profile
        assert identity["runtime"] == runtime and canonical_hash(identity["runtime"]) == runtime_hash
        assert identity["plan_id"] == plan["plan_id"] and identity["pass_name"] == path.name
        assert identity["batch_size"] == 1 and identity["reference"] is (label == "baseline")
        assert manifest["status"] == "complete"
        assert all(manifest[key] is False for key in ("query_gold_loaded", "test_content_read", "mixed_execution_modes"))
        before = {file.relative_to(run).as_posix(): digest(file) for file in path.rglob("*")
                  if file.is_file() and file.name in {"manifest.json", "scores.jsonl", "pool_binding.json"}}
        rows = [json.loads(line) for line in (path / "scores.jsonl").read_text().splitlines()]
        parallel = audit_parallel_pass(path, contexts, plan, merged_rows=rows)
        geometry_rows = rows
        if cohort == "boundary":
            # The frozen v2 worker labels any non-regression/validation pass as "dev".
            assert all(row["cohort"] == "dev" and row["pass_name"] == path.name for row in rows)
            geometry_rows = copy.deepcopy(rows)
            for row in geometry_rows:
                row["cohort"] = "boundary"
        geometry = validate_geometry(geometry_rows, contexts, identity, plan)
        assert len(rows) == manifest["blocks"] == len(contexts)
        assert geometry["candidates"] == manifest["candidates"]
        proof_path = preflight / f"{cohort}-{label}-geometry-proof.json"
        proof = {"passed": True, "blocks": len(rows), "candidates": geometry["candidates"],
                 "scoring_profile": profile, "true_batch_one": True, "prefix_is_reference_only": label == "prefix",
                 "execution_order_verified": label != "prefix", "within_batch_row_position_claimed": False}
        assert read(proof_path) == proof
        assert all(digest(run / name) == checksum for name, checksum in before.items())
        evidence.update(before)
        evidence[proof_path.relative_to(run).as_posix()] = digest(proof_path)
        checked[label] = {"blocks": len(rows), **geometry, "parallel_identity_and_shards": parallel,
                          "geometry_proof_independently_verified": True}
        passes[label] = rows
    reference_candidates = 0
    baseline = passes["baseline"]
    for row in baseline:
        for candidate in row["candidates"]:
            ref_scores = candidate["reference_scores"]
            actual = recompute(candidate["token_logprobs"], candidate["eos_logprob"])
            expected = recompute(ref_scores["token_logprobs"], ref_scores["eos_logprob"])
            assert candidate["reference_arithmetic_dtype"] == "cpu.torch.float64"
            assert ref_scores["token_logprobs"] == candidate["reference_token_logprobs"]
            assert ref_scores["eos_logprob"] == candidate["reference_eos_logprob"]
            differences = {field: actual[field] - expected[field] for field in FIELDS}
            tokens = [a - b for a, b in zip(candidate["token_logprobs"], ref_scores["token_logprobs"], strict=True)]
            assert candidate["reference_differences"] == {**differences, "token_logprobs": tokens}
            assert candidate["reference_abs_error_max"] == max(abs(value) for value in [*differences.values(), *tokens])
            reference_candidates += 1
    comparisons = {}
    for name in ("reference", "repeat", "padding", "prefix", "members", "replica"):
        path = preflight / f"{cohort}-{name}-differences.json"
        comparison = compare(baseline, baseline if name == "reference" else passes[name], path,
                             cpu_reference=name == "reference")
        assert comparison["stored_difference_file_verified"]
        threshold = .0001 if name in {"reference", "repeat"} else plan["numeric_policy"]["epsilon"]
        assert math.isfinite(comparison["max_abs_error"])
        comparison.update(threshold=threshold, passed=comparison["max_abs_error"] <= threshold)
        comparisons[name] = comparison
        evidence[path.relative_to(run).as_posix()] = digest(path)
    uuids = checked["replica"]["parallel_identity_and_shards"]["physical_gpu_uuid_by_index"]
    movement = validate_cross_gpu_challenge(baseline, passes["replica"], uuids)
    proof_path = preflight / f"{cohort}-replica-producer-proof.json"
    verify_replica_proof(proof_path, movement, uuids, len(contexts))
    evidence[proof_path.relative_to(run).as_posix()] = digest(proof_path)
    assert all(digest(run / name) == checksum for name, checksum in evidence.items())
    return {"schema_version": "independent-coverage-stage-review/v1", "plan_id": plan["plan_id"],
        "plan_ref_sha256": digest(reference), "plan_sha256": ref["plan_sha256"],
        "run_path": str(run.resolve()), "cohort": cohort, "query_ids": plan["cohorts"][cohort],
        "runtime_identity_sha256": runtime_hash, "sealed_pass_count": len(passes),
        "blocks_per_pass": len(contexts), "candidate_reference_fields_verified": reference_candidates,
        "E8_inherited": plan["numeric_policy"]["E8"], "epsilon": plan["numeric_policy"]["epsilon"],
        "epsilon_recalibrated": False, "passes": checked, "comparisons": comparisons,
        "replica_producer_proof_independently_verified": movement,
        "stage_passed": all(value["passed"] for value in comparisons.values()),
        "full_preflight_pass_claimed": False, "development_release_authorized_by_this_review": False,
        "final_preflight_report_present_at_review": (preflight / "preflight_report.json").exists(),
        "legacy_boundary_cohort_metadata": "raw cohort=dev; boundary identity verified through pass_name and frozen query IDs" if cohort == "boundary" else None,
        "scope": {"gpu_used": False, "query_gold_read": False, "test_content_read": False,
                  "scientific_effects_computed": False, "partial_passes_read": False,
                  "frozen_sources_modified": False, "run_state_modified": False,
                  "production_comparison_used_as_expected": False},
        "evidence_sha256": evidence,
        "audit_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), V1 / "independent_numeric_audit.py", V1 / "independent_full_dev_audit.py",
            V2 / "independent_preflight_audit.py", V2 / "independent_parallel_audit.py",
            V3 / "independent_fallback_preflight_audit.py")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cohort", choices=["regression", "validation", "boundary"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.plan, args.run, args.cohort)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"cohort": result["cohort"], "stage_passed": result["stage_passed"],
        "sealed_pass_count": result["sealed_pass_count"], "runtime_sha256": result["runtime_identity_sha256"],
        "maxima": {key: value["max_abs_error"] for key, value in result["comparisons"].items()},
        "audit_sha256": digest(args.output)}, sort_keys=True))
