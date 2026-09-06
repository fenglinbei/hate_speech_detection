"""Synthetic-only tests; no real score or query-gold access."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SPEC = importlib.util.spec_from_file_location("test_replication_renderer", Path(__file__).with_name("render_replication_report.py"))
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def analysis_fixture():
    result = {"schema_version": "general-model-numeric-coverage-analysis/v1", "query_count": 643,
              "block_count": 10288, "epsilon": R.CSV.POLICY["epsilon"], "score_modes": list(R.CSV.MODES),
              "conditions": list(R.CSV.CONDITIONS), "contrasts": R.CSV.CONTRASTS,
              "primary_contrasts": list(R.CSV.PRIMARY), "reference_contrasts": list(R.CSV.REFERENCE),
              "group_order": list(R.CSV.LABELS), "stratum_counts": R.CSV.CI_COUNTS,
              "per_query": [{"query_id": str(i)} for i in range(643)],
              "bootstrap": {"endpoint_count": 240, "replicates": 10000, "seed": 42, "scope": "descriptive",
                            "unit": "query", "interval": "percentile_95",
                            "shared_draw_across_tasks_conditions_endpoints_within_stratum": True,
                            "resampled_model_seeds": False}}
    for family, cells in (("contrast", R.CSV.CONTRASTS), ("condition", R.CSV.CONDITIONS)):
        rows = []
        for task, label in R.CSV.READOUTS:
            for mode in R.CSV.MODES:
                for cell in cells:
                    for stratum, count in R.CSV.CI_COUNTS.items():
                        row = {"task": task, "metric": f"{mode}/margin/{label}", family: cell,
                               "stratum": stratum, "n": count, "mean": .2, "median": .2}
                        if family == "contrast":
                            row.update(descriptive_ci95=[.1, .3], direction="positive",
                                       error_bound=R.CSV.POLICY["epsilon"] * sum(abs(v) for v in cells[cell].values()),
                                       positive_query_count=count, negative_query_count=0, unresolved_query_count=0)
                        rows.append(row)
        result[family + "_summaries"] = rows
    return result


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.audit_dir = self.root / "audit"
        self.audit_dir.mkdir()
        self.plan_id = "gmlrep-" + "1" * 64
        self.runtimes = {}
        for shift, name in ((0, "runtime-baseline.json"), (1, "runtime-replica.json")):
            runtime = {"execution": "whole-layer-sharded-fp32", "model": {"key": "qwen3-14b"},
                       "device_map": {f"module{i}": 2 * shift + (i % 2) for i in range(44)}}
            self.runtimes[shift] = runtime
            write_json(self.run / name, runtime)
        self.gate = {"schema_version": "general-model-coverage-replication-preflight/v1",
                     "plan_id": self.plan_id, "complete": True, "passed": True, "numeric_policy": R.CSV.POLICY,
                     "inherited_E8_is_not_new_model_calibration": True,
                     "placement_challenge": "same-layer-partition-different-physical-GPUs",
                     "query_gold_loaded": False, "test_content_read": False, "scientific_effect_checked": False,
                     "checks": {key: {"passed": True, "max_abs_error": 0.0,
                                      "limit": .0001 if key.rsplit("-", 1)[1] in ("reference", "repeat")
                                      else R.CSV.POLICY["epsilon"]} for key in R.CHECKS},
                     "runtime_sha256": {str(shift): R.digest(self.run / name) for shift, name in
                                        ((0, "runtime-baseline.json"), (1, "runtime-replica.json"))}}
        write_json(self.run / "preflight/preflight_report.json", self.gate)
        scores = self.run / "dev-b1/scores.jsonl"
        scores.parent.mkdir()
        scores.write_bytes(b"synthetic raw: deliberately not valid JSON\n")
        self.raw = {"schema_version": "general-model-ld-numeric-pass/v2", "status": "complete",
                    "blocks": 10288, "candidates": 174896, "scores_sha256": R.digest(scores),
                    "query_gold_loaded": False, "test_content_read": False, "mixed_execution_modes": False,
                    "identity": {"batch_size": 1, "reference": False, "pass_name": "dev-b1",
                                 "scoring_profile": R.PROFILE, "runtime": self.runtimes[0], "plan_id": self.plan_id}}
        write_json(self.run / "dev-b1/manifest.json", self.raw)
        self.analysis = analysis_fixture()
        write_json(self.run / "analysis/analysis.json", self.analysis)
        self.analysis_manifest = {"schema_version": "general-model-coverage-replication-analysis/v1",
                                  "model_key": "qwen3-14b", "plan_id": self.plan_id,
                                  "raw_manifest_sha256": R.digest(self.run / "dev-b1/manifest.json"),
                                  "analysis_sha256": R.digest(self.run / "analysis/analysis.json"),
                                  "gold_join_after_raw_sealed": True, "test_content_read": False,
                                  "production_geometry": {"passed": True, "blocks": 10288, "candidates": 174896,
                                                          "true_batch_one": True, "scoring_profile": R.PROFILE,
                                                          "prefix_is_reference_only": False,
                                                          "execution_order_verified": True,
                                                          "within_batch_row_position_claimed": False}}
        write_json(self.run / "analysis/manifest.json", self.analysis_manifest)
        self.terminal = {"schema_version": "general-model-coverage-replication-run/v1", "status": "complete",
                         "analysis_published": True, "full_dev_started": True, "raw_blocks": 10288,
                         "model_key": "qwen3-14b", "plan_id": self.plan_id, "execution": "model-parallel-fp32",
                         "query_gold_loaded_during_scoring": False, "test_content_read": False,
                         "automatic_profile_search": False,
                         "preflight_report_sha256": R.digest(self.run / "preflight/preflight_report.json"),
                         "raw_manifest_sha256": R.digest(self.run / "dev-b1/manifest.json"),
                         "analysis_manifest_sha256": R.digest(self.run / "analysis/manifest.json")}
        write_json(self.run / "run_manifest.json", self.terminal)
        gold = self.root / "queries.dev.jsonl"
        gold.write_bytes(b"synthetic gold: deliberately not valid JSON\n")
        self.audit = {"schema_version": "independent-coverage-replication-full-dev-audit/v1",
                      "audit_passed": True, "passed": True, "run_status": "complete", "model_key": "qwen3-14b",
                      "scientific_tables_written": True, "query_gold_read": True,
                      "raw_validated_before_gold_access": True, "gpu_used": False, "test_content_read": False,
                      "queries": 643, "blocks": 10288, "candidates": 174896, "plan_id": self.plan_id,
                      "all_ci_verified": True, "ci_endpoint_count": 240,
                      "verification": {"independently_recomputed_ci_endpoints_per_stratum": 240,
                                       "independently_recomputed_ci_strata": 6, "max_ci_endpoint_discrepancy": 0.0},
                      "source_hashes": {str(self.run / name): R.digest(self.run / name) for name in R.RUN_FILES},
                      "production_geometry": {**{key: True for key in R.GEOMETRY_FLAGS}, "blocks": 10288,
                                              "candidates": 174896, "scoring_profile": R.PROFILE},
                      "independent_preflight": {"passed": True, "sealed_pass_count": 18,
                                                "epsilon_recalibrated": False,
                                                "all_registered_numeric_readouts_recomputed": True,
                                                "E8": R.CSV.POLICY["E8"], "epsilon": R.CSV.POLICY["epsilon"],
                                                "checks": self.gate["checks"],
                                                "cohorts": {"regression": {"queries": 8}, "validation": {"queries": 24},
                                                            "boundary": {"queries": 4}},
                                                "runtime_remapping": {"passed": True, "modules_changed": 44,
                                                                      "baseline_devices": [0, 1],
                                                                      "replica_devices": [2, 3]}}, "files": {}}
        self.audit["source_hashes"][str(gold)] = R.digest(gold)
        for name in R.CSV.EXTRA_TABLES:
            (self.audit_dir / name).write_bytes(b"synthetic,value\n1,2\n")
            self.audit["files"][name] = R.digest(self.audit_dir / name)
        self.save_audit()

    def save_audit(self):
        write_json(self.audit_dir / "audit.json", self.audit)

    def test_complete_binding_and_render_do_not_parse_raw_or_gold(self):
        with mock.patch.object(R, "read_json", wraps=R.read_json) as read:
            receipt = R.render(self.run, self.audit_dir, self.root / "report")
        names = [Path(call.args[0]).name for call in read.call_args_list]
        self.assertNotIn("scores.jsonl", names)
        self.assertNotIn("queries.dev.jsonl", names)
        self.assertFalse(receipt["recomputed_statistics"])
        self.assertEqual(len(receipt["files"]), 11)
        text = (self.root / "report/REPORT.md").read_text()
        self.assertIn("Qwen3-14B", text)
        self.assertIn("not a pure causal parameter-count comparison", text)
        self.assertNotIn("Qwen3-8B", text)
        for name in R.CSV.EXTRA_TABLES:
            self.assertEqual((self.root / "report" / name).read_bytes(), (self.audit_dir / name).read_bytes())

    def test_incomplete_run_refused_before_audit(self):
        self.terminal["status"] = "running"
        write_json(self.run / "run_manifest.json", self.terminal)
        with mock.patch.object(R, "read_json", wraps=R.read_json) as read:
            with self.assertRaisesRegex(R.ReportError, "complete sealed"):
                R.verify_inputs(self.run, self.audit_dir)
        self.assertEqual(len(read.call_args_list), 1)

    def test_source_tampering_refused(self):
        (self.run / "dev-b1/scores.jsonl").write_bytes(b"tampered")
        with self.assertRaisesRegex(R.ReportError, "audited source changed"):
            R.verify_inputs(self.run, self.audit_dir)

    def test_missing_source_refused(self):
        del self.audit["source_hashes"][str(self.run / "runtime-replica.json")]
        self.save_audit()
        with self.assertRaisesRegex(R.ReportError, "audited run artifacts"):
            R.verify_inputs(self.run, self.audit_dir)

    def test_missing_check_refused(self):
        del self.audit["independent_preflight"]["checks"]["boundary-replica"]
        self.save_audit()
        with self.assertRaisesRegex(R.ReportError, "all 18"):
            R.verify_inputs(self.run, self.audit_dir)

    def test_replication_geometry_required(self):
        self.audit["production_geometry"]["global_candidate_ordinals_verified"] = False
        self.save_audit()
        with self.assertRaisesRegex(R.ReportError, "whole-layer production geometry"):
            R.verify_inputs(self.run, self.audit_dir)

    def test_all_ci_required(self):
        self.audit["verification"]["independently_recomputed_ci_strata"] = 5
        self.save_audit()
        with self.assertRaisesRegex(R.ReportError, "all registered CI"):
            R.verify_inputs(self.run, self.audit_dir)

    def test_legacy_schema_refused(self):
        self.audit["schema_version"] = "independent-coverage-full-dev-audit/v1"
        self.save_audit()
        with self.assertRaisesRegex(R.ReportError, "independent replication audit"):
            R.verify_inputs(self.run, self.audit_dir)

    def test_table_escape_refused(self):
        self.audit["files"]["../queries.dev.jsonl"] = self.audit["source_hashes"][str(self.root / "queries.dev.jsonl")]
        self.save_audit()
        with self.assertRaisesRegex(R.ReportError, "escapes"):
            R.verify_inputs(self.run, self.audit_dir)

    def test_overwrite_refused(self):
        with self.assertRaisesRegex(R.ReportError, "refusing overwrite"):
            R.render(self.run, self.audit_dir, self.audit_dir)


if __name__ == "__main__":
    unittest.main()
