"""Synthetic source-chain and inherited-threshold tests; no actual gold or GPU."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from independent_fallback_preflight_audit import CHALLENGES, NOT_APPLICABLE, validate_report_scope
from independent_full_dev_audit import load_effective_plan
from independent_numeric_audit import canonical_hash, digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def seal(directory, value, prefix):
    value = {key: item for key, item in value.items() if key != "plan_id"}
    value["plan_id"] = prefix + canonical_hash(value)
    write(directory / "plan.json", value)
    return value


def reference(path, directory, plan):
    write(path, {"target_path": str(directory), "plan_id": plan["plan_id"], "plan_sha256": digest(directory / "plan.json")})


class FallbackAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        first, second, third = [self.root / name for name in ("v1", "v2", "v3")]
        for directory in (first, second, third):
            directory.mkdir()
            (directory / "protocol.md").write_text("Synthetic protocol\n")
        config = {"model_key": "synthetic", "tasks": ["hate", "group"],
                  "conditions": ["C0", "CL", "CD", "CLD", "PL", "PD"],
                  "candidate_order": {}, "analysis": {}, "execution": {"device_indices": [0, 1, 2, 3]}}
        parent = seal(first, {"schema_version": "general-model-ld-numeric-plan/v1", "config": config,
                             "code_sha256": {}, "protocol_sha256": digest(first / "protocol.md")}, "gmlnum-")
        first_ref = self.root / "v1_ref.json"
        reference(first_ref, first, parent)
        scientific = {name: config[name] for name in ("model_key", "tasks", "conditions", "candidate_order", "analysis")}
        amended = seal(second, {"schema_version": "general-model-ld-numeric-plan/v2", "config": config,
            "parent_plan_ref": str(first_ref), "parent_ref_sha256": digest(first_ref), "parent_plan_id": parent["plan_id"],
            "parent_plan_sha256": canonical_hash(parent), "scientific_scope_sha256": canonical_hash(scientific),
            "protocol_sha256": digest(second / "protocol.md"), "code_sha256": {}}, "gmlnum2-")
        second_ref = self.root / "v2_ref.json"
        reference(second_ref, second, amended)
        self.second, self.third = second, third
        effective = load_effective_plan(second_ref, root=self.root)
        failed = self.root / "failed"
        self.failure_payload = failed / "preflight/synthetic-scores.jsonl"
        self.failure_payload.parent.mkdir(parents=True)
        self.failure_payload.write_text("synthetic opaque scores\n")
        write(failed / "runtime_identity.json", {})
        original_challenges = ["batch", "tail", "padding", "prefix", "members", "replica"]
        report = {"schema_version": "general-model-ld-numeric-calibration/v2", "plan_id": amended["plan_id"],
                  "passed": False, "complete": False, "failure": "validation-batch-compatibility-gate",
                  "E8": .00067138671875, "epsilon": .0013427734375, "runtime_identity": {},
                  "error_families": original_challenges,
                  "query_gold_loaded": False, "test_content_read": False, "scientific_effect_checked": False, "formal_test_authorized": False,
                  "cohorts": {"regression": {"passed": True, "complete": True, "baseline_passed": True,
                      "baseline_repeat_max_abs_error": 0.0, "reference_max_abs_error": .00001,
                      "max_abs_error": .00067138671875, "challenges": {name: {"passed": True, "max_abs_error": .00067138671875}
                                                                        for name in original_challenges}},
                      "validation": {"baseline_passed": True, "baseline_repeat_max_abs_error": 0.0, "reference_max_abs_error": .00001,
                                     "challenges": {"batch": {"passed": False, "max_abs_error": .001621246337890625}}}},
                  "files": {"synthetic-scores.jsonl": digest(self.failure_payload)}}
        write(failed / "preflight/preflight_report.json", report)
        write(third / "source_preflight_report.json", report)
        write(failed / "run_manifest.json", {"status": "preflight_failed", "plan_id": amended["plan_id"],
              "failure": report["failure"], "full_dev_started": False, "analysis_published": False,
              "query_gold_loaded_during_scoring": False, "test_content_read": False,
              "preflight_report_sha256": digest(failed / "preflight/preflight_report.json")})
        hashes = {"preflight/" + name: checksum for name, checksum in report["files"].items()}
        hashes.update({name: digest(failed / name) for name in ("run_manifest.json", "runtime_identity.json", "preflight/preflight_report.json")})
        fallback = {"parent_plan_ref": str(second_ref), "source_failed_run": str(failed),
                    "production_batch_size": 1, "raw_path": "dev-b1", "device_indices": [0, 1, 2, 3],
                    "validation": {"challenges": list(CHALLENGES), "inherit_epsilon": True, "epsilon_recalibration": False,
                                   "repeat_abs_tolerance": .0001, "reference_abs_tolerance": .0001,
                                   "padding_extra": 64, "replica_shift": 1},
                    "automatic_profile_search": False, "test_access": False, "query_gold_in_scoring": False}
        write(third / "fallback_config.json", fallback)
        self.stored = seal(third, {"schema_version": "general-model-ld-numeric-plan/v3", "fallback_config": fallback,
            "fallback_config_sha256": canonical_hash(fallback), "scientific_scope_sha256": canonical_hash(config),
            "query_gold_loaded": False, "test_content_read": False,
            "parent_plan_ref": str(second_ref), "parent_ref_sha256": digest(second_ref),
            "parent_plan_id": amended["plan_id"], "parent_plan_sha256": canonical_hash(effective),
            "source_failed_run": str(failed), "source_failure": {"E8": report["E8"], "epsilon": report["epsilon"],
                "calibration_sha256": canonical_hash(report["cohorts"]["regression"]), "hashes": hashes,
                "runtime_identity_sha256": canonical_hash({}), "failure": report["failure"]},
            "inherited_code_sha256": {}, "code_sha256": {}, "protocol_sha256": digest(third / "protocol.md")}, "gmlnum3-")

    def test_recursive_v3_plan_binds_effective_parent_and_failed_source(self):
        actual = load_effective_plan(self.third, root=self.root)
        self.assertEqual(actual["parent_plan"]["schema_version"], "general-model-ld-numeric-plan/v2")
        self.assertEqual(actual["parent_plan"]["parent_plan"]["schema_version"], "general-model-ld-numeric-plan/v1")
        self.assertEqual(actual["config"], actual["parent_plan"]["config"])
        self.assertEqual(actual["source_failure"]["epsilon"], .0013427734375)

    def test_v3_parent_file_hash_cannot_replace_effective_semantic_hash(self):
        changed = copy.deepcopy(self.stored)
        changed["parent_plan_sha256"] = digest(self.second / "plan.json")
        seal(self.third, changed, "gmlnum3-")
        with self.assertRaises(AssertionError):
            load_effective_plan(self.third, root=self.root)

    def test_source_failed_scores_hash_drift_is_rejected(self):
        self.failure_payload.write_text("different synthetic scores\n")
        with self.assertRaises(AssertionError):
            load_effective_plan(self.third, root=self.root)

    def test_fallback_cannot_enable_epsilon_recalibration(self):
        changed = copy.deepcopy(self.stored)
        changed["fallback_config"]["validation"]["epsilon_recalibration"] = True
        seal(self.third, changed, "gmlnum3-")
        with self.assertRaises(AssertionError):
            load_effective_plan(self.third, root=self.root)

    def test_fallback_report_scope_preserves_four_challenges_and_exposure(self):
        plan = load_effective_plan(self.third, root=self.root)
        report = {"schema_version": "general-model-ld-numeric-calibration/v3", "plan_id": plan["plan_id"],
                  "calibration_mode": "inherited-source-failure-no-recalibration", "source_calibration_sha256": plan["source_failure"]["calibration_sha256"],
                  "runtime_identity": {}, "source_failed_run": plan["source_failed_run"],
                  "E8": plan["source_failure"]["E8"], "epsilon": plan["source_failure"]["epsilon"],
                  "error_families": list(CHALLENGES), "not_applicable": NOT_APPLICABLE,
                  "validation_cohort_status": "previously-exposed-preregistered-fallback-revalidation",
                  "query_gold_loaded": False, "test_content_read": False, "scientific_effect_checked": False, "formal_test_authorized": False}
        validate_report_scope(report, plan)
        for field, wrong in (("epsilon", .005), ("error_families", [*CHALLENGES, "batch"]), ("runtime_identity", {"drift": True}),
                             ("validation_cohort_status", "unseen-validation")):
            changed = {**report, field: wrong}
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_report_scope(changed, plan)


if __name__ == "__main__":
    unittest.main()
