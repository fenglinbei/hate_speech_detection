"""Independently check four batch-one fallback challenges at inherited epsilon."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
V1 = ROOT / "exps/causal_context/general_model_ld_numeric_v1/audits"
V2 = ROOT / "exps/causal_context/general_model_ld_numeric_v2/audits"
sys.path[:0] = [str(V1), str(V2)]
from independent_full_dev_audit import load_effective_plan
from independent_numeric_audit import canonical_hash, digest, read
from independent_parallel_audit import validate_cross_gpu_challenge
from independent_preflight_audit import close, compare, validate_pass
from independent_reference_pass_audit import audit as audit_reference


CHALLENGES = ("padding", "prefix", "members", "replica")
SUFFIXES = {"baseline": "b1-r0", "repeat": "b1-r1", "padding": "b1-padding",
            "prefix": "b1-prefix", "members": "b1-members", "replica": "b1-replica"}
NOT_APPLICABLE = ["batch-four", "tail-size-two", "within-batch-row-position"]


def validate_report_scope(report, plan):
    assert report["schema_version"] == "general-model-ld-numeric-calibration/v3"
    assert report["plan_id"] == plan["plan_id"]
    assert report["calibration_mode"] == "inherited-source-failure-no-recalibration"
    assert report["source_calibration_sha256"] == plan["source_failure"]["calibration_sha256"]
    assert canonical_hash(report["runtime_identity"]) == plan["source_failure"]["runtime_identity_sha256"]
    assert report["source_failed_run"] == plan["source_failed_run"]
    assert report["E8"] == plan["source_failure"]["E8"]
    assert report["epsilon"] == plan["source_failure"]["epsilon"]
    assert report["error_families"] == list(CHALLENGES)
    assert report["not_applicable"] == NOT_APPLICABLE
    assert report["validation_cohort_status"] == "previously-exposed-preregistered-fallback-revalidation"
    assert all(report[flag] is False for flag in ("query_gold_loaded", "test_content_read", "scientific_effect_checked", "formal_test_authorized"))


def verify_replica_proof(path, check, uuids, blocks):
    proof = read(path)
    assert proof["passed"] is True and proof["all_candidates_changed_physical_gpu"] is True
    assert proof["batch_geometry_unchanged"] is True
    assert proof["blocks"] == blocks and proof["candidates"] == check["candidates"]
    pairs = proof["producer_pairs"]
    assert len(pairs) == len({(row["from_uuid"], row["to_uuid"]) for row in pairs})
    assert {(row["from_uuid"], row["to_uuid"]): row["candidates"] for row in pairs} == {
        (row["from_uuid"], row["to_uuid"]): row["candidates"] for row in check["transitions"]}
    assert all(uuids[row["from_index"]] == row["from_uuid"] and uuids[row["to_index"]] == row["to_uuid"] for row in pairs)


def audit(reference, directory):
    plan = load_effective_plan(reference, root=ROOT)
    assert plan["schema_version"] == "general-model-ld-numeric-plan/v3"
    epsilon = plan["source_failure"]["epsilon"]
    passes, results = {}, {}
    runtime_hashes = set()
    for cohort in ("regression", "validation"):
        contexts = [row for row in plan["blocks"] if row["query_id"] in plan["cohorts"][cohort]]
        result = {"sealed_passes": {}, "pending_passes": [], "challenges": {}, "threshold": epsilon}
        results[cohort] = result
        for name, suffix in SUFFIXES.items():
            path = directory / f"{cohort}-{suffix}"
            if not (path / "manifest.json").is_file():
                result["pending_passes"].append(name)
                continue
            rows, checked = validate_pass(path, contexts, plan, name)
            assert checked["parallel_identity_and_shards"]["common_runtime_sha256"] == plan["source_failure"]["runtime_identity_sha256"]
            proof_path = directory / f"{cohort}-{name}-geometry-proof.json"
            proof_present = proof_path.is_file()
            if proof_present:
                profile = read(path / "manifest.json")["identity"]["scoring_profile"]
                assert read(proof_path) == {"passed": True, "blocks": len(rows), "candidates": checked["candidates"],
                    "scoring_profile": profile, "true_batch_one": True, "prefix_is_reference_only": name == "prefix",
                    "execution_order_verified": name != "prefix", "within_batch_row_position_claimed": False}
            checked["geometry_proof_independently_verified"] = proof_present
            result["sealed_passes"][name] = checked
            passes[(cohort, name)] = rows
            runtime_hashes.add(checked["parallel_identity_and_shards"]["common_runtime_sha256"])
        if (cohort, "baseline") not in passes:
            result.update(baseline_passed=None, four_challenges_complete=False, passed=None)
            continue
        baseline = passes[(cohort, "baseline")]
        result["reference_raw_audit"] = audit_reference(directory / f"{cohort}-b1-r0", reference)
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
            comparison = compare(baseline, observed, directory / f"{cohort}-{name}-differences.json")
            comparison["passed"] = comparison["max_abs_error"] <= epsilon
            if name == "replica":
                uuids = result["sealed_passes"][name]["parallel_identity_and_shards"]["physical_gpu_uuid_by_index"]
                check = validate_cross_gpu_challenge(baseline, observed, uuids)
                proof_path = directory / f"{cohort}-replica-producer-proof.json"
                if proof_path.exists():
                    verify_replica_proof(proof_path, check, uuids, len(observed))
                comparison.update(actual_physical_gpu_change=check, producer_proof_verified=proof_path.exists())
            result["challenges"][name] = comparison
        result["four_challenges_complete"] = set(result["challenges"]) == set(CHALLENGES)
        result["observed_fallback_max"] = max((entry["max_abs_error"] for entry in result["challenges"].values()), default=0.0)
        result["passed"] = result["baseline_passed"] is True and result["four_challenges_complete"] and all(entry["passed"] for entry in result["challenges"].values())
    assert len(runtime_hashes) <= 1
    final_path = directory / "preflight_report.json"
    report = read(final_path) if final_path.is_file() else None
    verification = {"published": report is not None}
    if report is not None:
        validate_report_scope(report, plan)
        assert canonical_hash(report["runtime_identity"]) in runtime_hashes
        payloads = {path.relative_to(directory).as_posix(): path for path in directory.rglob("*")
                    if path.is_file() and (path.name in ("scores.jsonl", "manifest.json", "pool_binding.json")
                    or path.name.endswith(("-differences.json", "-proof.json")))}
        assert payloads.keys() == report["files"].keys()
        assert all(digest(path) == report["files"][name] for name, path in payloads.items())
        if report["passed"]:
            required = set()
            for cohort in ("regression", "validation"):
                for name, suffix in SUFFIXES.items():
                    required.update(f"{cohort}-{suffix}/{file}" for file in ("scores.jsonl", "manifest.json", "pool_binding.json"))
                    required.add(f"{cohort}-{name}-geometry-proof.json")
                required.update(f"{cohort}-{name}-differences.json" for name in ("reference", "repeat", *CHALLENGES))
                required.add(f"{cohort}-replica-producer-proof.json")
            assert required <= set(report["files"])
            assert len(passes) == 12
        for cohort, reported in report["cohorts"].items():
            checked = results[cohort]
            assert reported["blocks"] == len(plan["cohorts"][cohort]) * 12
            assert reported["baseline_passed"] is checked["baseline_passed"]
            close(reported["baseline_repeat_max_abs_error"], checked["repeat"]["max_abs_error"])
            close(reported["reference_max_abs_error"], checked["reference"]["max_abs_error"])
            assert reported["challenges"].keys() == checked["challenges"].keys()
            for name, entry in reported["challenges"].items():
                close(entry["max_abs_error"], checked["challenges"][name]["max_abs_error"])
                assert entry["passed"] is checked["challenges"][name]["passed"]
                assert entry["largest_error"] == read(directory / f"{cohort}-{name}-differences.json")["largest_error"]
            if "complete" in reported:
                assert reported["complete"] is checked["four_challenges_complete"]
                assert reported["passed"] is checked["passed"]
                close(reported["observed_fallback_max"], checked["observed_fallback_max"])
        expected_complete = all(entry["four_challenges_complete"] for entry in results.values())
        expected_passed = all(entry["passed"] for entry in results.values())
        assert report["complete"] is expected_complete and report["passed"] is expected_passed
        assert report["validation_executed"] is (("validation", "baseline") in passes)
        close(report["observed_fallback_max"], max(entry.get("observed_fallback_max", 0.0) for entry in results.values()))
        for checked in results.values():
            assert all(entry["geometry_proof_independently_verified"] for entry in checked["sealed_passes"].values())
            for name in ("reference", "repeat"):
                if name in checked:
                    assert checked[name]["stored_difference_file_verified"]
            for name, entry in checked["challenges"].items():
                assert entry["stored_difference_file_verified"]
                if name == "replica":
                    assert entry["producer_proof_verified"]
        terminal = read(directory.parent / "run_manifest.json")
        assert terminal["plan_id"] == plan["plan_id"]
        assert terminal["raw_path"] == "dev-b1" and terminal["production_batch_size"] == 1
        assert terminal["source_calibration_sha256"] == plan["source_failure"]["calibration_sha256"]
        assert terminal["source_failed_run"] == plan["source_failed_run"]
        assert canonical_hash(read(directory.parent / "runtime_identity.json")) == plan["source_failure"]["runtime_identity_sha256"]
        assert terminal["query_gold_loaded_during_scoring"] is False and terminal["test_content_read"] is False
        bound = "preflight_report_sha256" in terminal
        if bound:
            assert terminal["preflight_report_sha256"] == digest(final_path)
        verification.update(report_sha256=digest(final_path), all_payload_file_hashes_verified=True,
                            payload_file_count=len(payloads), run_manifest_binds_report=bound,
                            complete_and_passed_flags_independently_verified=True)
    return {"schema_version": "independent-batch-one-fallback-audit/v1", "plan_id": plan["plan_id"],
            "preflight_path": str(directory.resolve()), "source_failed_run": plan["source_failed_run"],
            "source_calibration_sha256": plan["source_failure"]["calibration_sha256"],
            "source_runtime_identity_sha256": plan["source_failure"]["runtime_identity_sha256"],
            "E8_inherited": plan["source_failure"]["E8"], "epsilon": epsilon, "epsilon_recalibrated": False,
            "cohorts": results, "sealed_pass_count": len(passes), "applicable_challenges": list(CHALLENGES),
            "not_applicable": NOT_APPLICABLE,
            "all_four_regression_challenges_complete": results["regression"]["four_challenges_complete"],
            "all_four_validation_challenges_complete": results["validation"]["four_challenges_complete"],
            "validation_passed": results["validation"]["passed"],
            "preflight_passed": bool(report is not None and report["passed"]), "final_report": verification,
            "validation_cohort_status": "previously-exposed-preregistered-fallback-revalidation",
            "scope": {"gpu_used": False, "query_gold_read": False, "scientific_effects_computed": False,
                      "partial_passes_read": False, "test_content_read": False, "original_six_challenges_claimed_passed": False},
            "audit_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
                Path(__file__), V1 / "independent_full_dev_audit.py", V1 / "independent_numeric_audit.py",
                V2 / "independent_preflight_audit.py", V2 / "independent_parallel_audit.py", V2 / "independent_reference_pass_audit.py")}}


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
    print(json.dumps({name: result[name] for name in ("sealed_pass_count", "E8_inherited", "epsilon",
        "all_four_regression_challenges_complete", "all_four_validation_challenges_complete", "validation_passed", "preflight_passed")}))
