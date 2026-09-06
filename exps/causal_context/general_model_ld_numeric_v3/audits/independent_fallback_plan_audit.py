"""Audit the frozen fallback plan and source failure, without running scoring."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HELPERS = ROOT / "exps/causal_context/general_model_ld_numeric_v1/audits"
sys.path.insert(0, str(HELPERS))
from independent_full_dev_audit import CONDITIONS, CONTRASTS, expected_catalog, load_effective_plan
from independent_numeric_audit import canonical_hash, digest, read


def audit(reference):
    plan = load_effective_plan(reference, root=ROOT)
    assert plan["schema_version"] == "general-model-ld-numeric-plan/v3"
    stored_ref = read(reference)
    parent = plan["parent_plan"]
    config = plan["fallback_config"]
    assert config == read(ROOT / "config/stage1/general_model_ld_numeric_v3.json")
    assert digest(ROOT / config["protocol_path"]) == plan["protocol_sha256"]
    assert plan["config"] == parent["config"]
    assert plan["config"]["model_key"] == "qwen3-8b"
    assert plan["config"]["tasks"] == ["hate", "group"]
    assert plan["config"]["conditions"] == list(CONDITIONS)
    assert {**plan["config"]["analysis"]["primary_comparisons"],
            **plan["config"]["analysis"]["secondary_comparisons"]} == CONTRASTS
    assert config["production_batch_size"] == 1 and config["raw_path"] == "dev-b1"
    assert config["auto_expand_on_pass"] is True and config["stop_on_any_gate_failure"] is True
    assert plan["config"]["runtime"]["dtype"] == "float32"
    assert plan["config"]["analysis"]["gold_join"] == "analysis-only-after-raw-seal"
    assert plan["query_gold_loaded"] is False and plan["test_content_read"] is False
    assert plan["formal_test_authorized"] is False
    qids = [entry["query_id"] for entry in plan["frame"]]
    assert len(qids) == len(set(qids)) == 643
    assert qids == sorted(qids, key=int)
    assert all(set(entry) == {"query_id", "lex_hit"} and type(entry["lex_hit"]) is bool for entry in plan["frame"])
    expected_records = [f"{qid}:{task}:{condition}" for task in ("hate", "group") for condition in CONDITIONS for qid in qids]
    assert [entry["record_id"] for entry in plan["blocks"]] == expected_records
    assert len(plan["blocks"]) == plan["expected_blocks"] == 7716
    assert plan["expected_candidates"] == plan["boundary_checks"] == 131172
    assert plan["catalog_sha256"] == canonical_hash(plan["catalog"])
    for task in ("hate", "group"):
        assert len(plan["catalog"][task]) == len(expected_catalog(task))
        for recorded, expected in zip(plan["catalog"][task], expected_catalog(task), strict=True):
            assert all(recorded[name] == value for name, value in expected.items())
            assert recorded["answer_token_ids_sha256"] == canonical_hash(recorded["answer_token_ids"])
            assert recorded["answer_tokens"] == len(recorded["answer_token_ids"])
            assert plan["eos_token_id"] not in recorded["answer_token_ids"]
    regression, validation = plan["cohorts"]["regression"], plan["cohorts"]["validation"]
    assert len(regression) == len(set(regression)) == 8
    assert len(validation) == len(set(validation)) == 24
    assert set(regression).isdisjoint(validation) and set(regression + validation) <= set(qids)
    failed = Path(plan["source_failed_run"])
    previous_audit_path = failed.parents[1] / "audits" / (failed.name + "-preflight-engineering-failure.json")
    previous = read(previous_audit_path)
    assert previous["plan_id"] == parent["plan_id"]
    assert previous["preflight_passed"] is False and previous["validation_passed"] is False
    assert previous["final_report"]["report_sha256"] == plan["source_failure"]["hashes"]["preflight/preflight_report.json"]
    assert previous["epsilon"] == plan["source_failure"]["epsilon"]
    assert previous["E8_frozen_after_complete_regression"] == plan["source_failure"]["E8"]
    return {
        "schema_version": "independent-fallback-plan-audit/v1", "audit_passed": True,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "plan_id": plan["plan_id"],
        "plan_ref_sha256": digest(reference), "plan_file_sha256": stored_ref["plan_sha256"],
        "parent_plan_id": parent["plan_id"], "parent_effective_semantic_sha256": plan["parent_plan_sha256"],
        "full_parent_config_scope_sha256": plan["scientific_scope_sha256"],
        "source_files_checked": len(plan["code_sha256"]), "source_sha256": plan["code_sha256"],
        "source_failed_run": str(failed), "source_failure_bound_files": len(plan["source_failure"]["hashes"]),
        "source_calibration_sha256": plan["source_failure"]["calibration_sha256"],
        "source_runtime_identity_sha256": plan["source_failure"]["runtime_identity_sha256"],
        "previous_independent_failure_audit_sha256": digest(previous_audit_path),
        "E8_inherited": plan["source_failure"]["E8"], "epsilon_inherited": plan["source_failure"]["epsilon"],
        "production": {"raw_path": "dev-b1", "batch_size": 1, "physical_gpu_indices": config["device_indices"],
                       "dtype": "float32", "queries": 643, "blocks": 7716, "candidates": 131172},
        "applicable_challenges": config["validation"]["challenges"],
        "not_applicable": ["batch-four", "tail-size-two", "within-batch-row-position"],
        "checks": {"recursive_plan_and_reference_hashes": True, "all_source_current_and_all_layer_snapshots": True,
                   "fallback_config_and_protocol_snapshots": True, "complete_failure_source_hash_inventory": True,
                   "source_actual_runtime_and_regression_calibration_hashes": True,
                   "source_failure_still_failed_no_dev_or_analysis": True,
                   "scientific_config_and_candidate_order_unchanged": True,
                   "exact_frame_catalog_tokens_and_separate_eos": True,
                   "gold_after_raw_seal_only_and_test_disallowed": True,
                   "inherited_epsilon_not_recalibrated": True},
        "scope": {"gold_read": False, "test_content_read": False, "gpu_used": False,
                  "new_run_scores_read": False, "scientific_effects_computed": False},
        "limitations": ["This verifies registration and source binding, not successful execution of the new fallback.",
                        "Physical batch-one geometry and actual runtime equality require the new sealed passes.",
                        "The additional 24 queries are previously exposed fallback revalidation, not a new unseen cohort.",
                        "No claim is made that the original batch-four six-challenge validation passed."],
        "audit_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), HELPERS / "independent_full_dev_audit.py", HELPERS / "independent_numeric_audit.py")},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.plan)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("audit_passed", "plan_id", "source_files_checked",
          "source_failure_bound_files", "E8_inherited", "epsilon_inherited")}))
