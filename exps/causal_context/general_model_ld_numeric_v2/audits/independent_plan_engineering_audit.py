"""Read-only plan/source audit; never opens scoring, query gold, or test data."""

import argparse
import hashlib
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
    parent = plan["parent_plan"]
    config = plan["config"]
    previous = parent["config"]
    assert config == read(ROOT / "config/stage1/general_model_ld_numeric_v2.json")
    assert digest(ROOT / config["protocol_path"]) == plan["protocol_sha256"]
    assert digest(ROOT / previous["protocol_path"]) == parent["protocol_sha256"]
    assert digest(Path(plan["package_path"]) / "manifest.json") == plan["package_manifest_sha256"]
    assert digest(Path(plan["generation_preflight_path"]) / "run_manifest.json") == plan["generation_manifest_sha256"]
    scope_fields = ("model_key", "tasks", "conditions", "candidate_order", "analysis")
    assert all(config[name] == previous[name] for name in scope_fields)
    assert config["model_key"] == "qwen3-8b"
    assert config["tasks"] == ["hate", "group"]
    assert config["conditions"] == list(CONDITIONS)
    comparisons = {**config["analysis"]["primary_comparisons"], **config["analysis"]["secondary_comparisons"]}
    assert comparisons == CONTRASTS
    assert config["analysis"]["gold_join"] == "analysis-only-after-raw-seal"
    bootstrap = config["analysis"]["bootstrap"]
    assert bootstrap["repetitions"] == 10000 and bootstrap["seed"] == 42
    assert bootstrap["joint_across_tasks_and_readouts"] and bootstrap["paired_across_conditions"]
    assert bootstrap["confirmatory_p_values"] is False

    query_ids = [entry["query_id"] for entry in plan["frame"]]
    assert len(query_ids) == len(set(query_ids)) == 643
    assert query_ids == sorted(query_ids, key=int)
    assert all(set(entry) == {"query_id", "lex_hit"} and type(entry["lex_hit"]) is bool for entry in plan["frame"])
    expected_records = [f"{qid}:{task}:{condition}" for task in config["tasks"]
                        for condition in config["conditions"] for qid in query_ids]
    assert [entry["record_id"] for entry in plan["blocks"]] == expected_records
    block_fields = {"condition", "context_sha256", "prompt_sha256", "prompt_token_ids_sha256",
                    "prompt_tokens", "query_id", "record_id", "task"}
    assert all(set(entry) == block_fields for entry in plan["blocks"])
    assert len(plan["blocks"]) == plan["expected_blocks"] == 7716
    assert plan["expected_candidates"] == plan["boundary_checks"] == 643 * 6 * 34 == 131172
    assert plan["catalog_sha256"] == canonical_hash(plan["catalog"])
    for task in config["tasks"]:
        assert len(plan["catalog"][task]) == len(expected_catalog(task))
        for stored, expected in zip(plan["catalog"][task], expected_catalog(task), strict=True):
            assert all(stored[name] == value for name, value in expected.items())
            assert stored["answer_sha256"] == hashlib.sha256(stored["canonical_answer"].encode()).hexdigest()
            assert stored["answer_token_ids_sha256"] == canonical_hash(stored["answer_token_ids"])
            assert stored["answer_tokens"] == len(stored["answer_token_ids"]) > 0
            assert plan["eos_token_id"] not in stored["answer_token_ids"]
    regression = plan["cohorts"]["regression"]
    validation = plan["cohorts"]["validation"]
    assert len(regression) == len(set(regression)) == 8
    assert len(validation) == len(set(validation)) == 24
    assert set(regression).isdisjoint(validation)
    assert set(regression + validation) <= set(query_ids)

    for flag in ("query_gold_loaded", "test_content_read", "formal_test_authorized"):
        assert plan[flag] is False and parent[flag] is False
    assert config["test_access"] is False
    execution = config["execution"]
    for flag in ("analysis_requires_raw_seal", "raw_seal_requires_full_coverage", "stop_on_any_gate_failure",
                 "auto_expand_on_pass", "early_stop_before_validation_on_regression_failure"):
        assert execution[flag] is True
    assert execution["save_query_gold_in_scoring_artifacts"] is False
    assert execution["fallback"]["automatic"] is False
    assert execution["device_indices"] == [0, 1, 2, 3]
    assert execution["automatic_dev_batch_size"] == 4
    assert execution["atomic_unit"] == "query-task-condition-all-candidates"
    runtime = config["runtime"]
    assert runtime["dtype"] == "float32" and runtime["head_float32"] is True
    assert runtime["bf16_reduced_precision_reduction"] is False
    assert runtime["use_cache"] is False and runtime["enable_thinking"] is False
    assert runtime["main_score"] == "answer-token-logprob-sum-excluding-eos"
    for name in ("batch_geometry", "max_sequence_tokens", "baseline_batch_size", "accelerated_batch_size",
                 "main_score", "auxiliary_scores", "padding_side", "seed"):
        assert runtime[name] == previous["runtime"][name]
    gates = config["validation"]
    unchanged_gates = ("repeat_abs_tolerance", "reference_abs_tolerance", "calibration_max_abs_error",
                       "epsilon_floor", "epsilon_ceiling", "epsilon_multiplier", "epsilon_formula",
                       "required_coverage", "regression_query_count", "validation_query_count",
                       "tail_batch", "error_scope", "scientific_success_requires_positive_effect")
    assert all(gates[name] == previous["validation"][name] for name in unchanged_gates)
    assert gates["repeat_abs_tolerance"] == gates["reference_abs_tolerance"] == .0001
    assert gates["calibration_max_abs_error"] == gates["epsilon_ceiling"] == .005
    assert gates["epsilon_floor"] == .0001 and gates["epsilon_multiplier"] == 2
    assert gates["replica_challenge_shift"] == 1 and gates["padding_challenge_extra"] == 64
    assert gates["member_challenge"] == "group-rotate-one-then-reverse-hate-reverse-before-batching-restore-canonical-order-before-commit"
    assert gates["replica_comparison"] == "batch-one-and-identical-batch-four-on-another-physical-device"
    assert gates["prefix_reference"] == "all-candidates-all-answer-and-eos-positions-unique-prefixes-uncached-unpadded-batch-one"
    assert set(previous["validation"]["required_checks"]) <= set(gates["required_checks"])

    sources = {**parent["code_sha256"], **plan["code_sha256"]}
    ref = read(reference)
    return {
        "schema_version": "independent-plan-engineering-audit/v1", "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_ref": str(reference.resolve()), "plan_ref_sha256": digest(reference),
        "plan_id": plan["plan_id"], "plan_file_sha256": ref["plan_sha256"],
        "parent_plan_id": plan["parent_plan_id"], "parent_semantic_sha256": plan["parent_plan_sha256"],
        "scientific_scope_sha256": plan["scientific_scope_sha256"],
        "source_files_checked": len(sources), "source_sha256": sources,
        "checks": {
            "plan_reference_file_hash_and_self_canonical_id": True,
            "parent_reference_file_hash_and_parent_canonical_semantic_hash": True,
            "all_parent_and_new_source_snapshots_equal_current_files": True,
            "both_protocol_snapshots_and_current_protocol_hashes": True,
            "current_config_equals_embedded_config": True,
            "upstream_package_and_generation_manifest_hashes": True,
            "unchanged_scientific_scope_and_unchanged_registered_math_source": True,
            "complete_gold_free_frame_and_ordered_record_id_cartesian_product": True,
            "canonical_catalog_answer_token_hashes_and_separate_eos": True,
            "disjoint_original_eight_and_additional_twenty_four": True,
            "numerical_thresholds_not_relaxed": True,
            "six_full_coverage_challenges_including_nonidentity_hate_and_physical_replica": True,
            "raw_seal_required_before_gold_join_and_test_disallowed": True,
        },
        "registered_counts": {"queries": 643, "blocks": 7716, "candidates": 131172,
                              "regression_queries": 8, "validation_queries": 24},
        "engineering_profile": {"dtype": "float32", "physical_device_indices": [0, 1, 2, 3],
                                "batch_size": 4, "raw_path": "dev-b4", "e8_cap": .005,
                                "epsilon_floor": .0001, "epsilon_ceiling": .005},
        "config_changed_top_level_fields": sorted(name for name in set(previous) | set(config)
                                                  if previous.get(name) != config.get(name)),
        "scope_limitations": ["No GPU computation or runtime realization was audited in this plan-only check.",
                              "No preflight scores, scientific L/D effects, query gold, or test content were read.",
                              "Formal preflight and full-dev raw are not declared complete by this artifact.",
                              "The registered four-GPU identity and shard chains still require live-pass validation."],
        "query_gold_read": False, "test_content_read": False, "scientific_effects_computed": False,
        "audit_source_sha256": {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
                                str((HELPERS / "independent_full_dev_audit.py").relative_to(ROOT)):
                                    digest(HELPERS / "independent_full_dev_audit.py"),
                                str((HELPERS / "independent_numeric_audit.py").relative_to(ROOT)):
                                    digest(HELPERS / "independent_numeric_audit.py")},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.plan)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "plan_id": result["plan_id"],
                      "source_files_checked": result["source_files_checked"], "output": str(args.output)}))
