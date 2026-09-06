"""Synthetic-only report export and fail-closed sealed artifact tests."""

import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SPEC = importlib.util.spec_from_file_location("render_coverage_report", Path(__file__).with_name("render_coverage_report.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def synthetic_analysis():
    conditions, contrasts = [], []
    for task, label in report.READOUTS:
        for mode in report.MODES:
            for stratum, n in report.CI_COUNTS.items():
                base = {"task": task, "metric": f"{mode}/margin/{label}", "stratum": stratum,
                        "n": n, "mean": .125, "median": .1, "p05": -.5, "p95": .6,
                        "q25": -.1, "q75": .3, "iqr": .4, "min": -1., "max": 1., "status": "defined"}
                for condition in report.CONDITIONS:
                    conditions.append({**base, "condition": condition})
                for contrast, weights in report.CONTRASTS.items():
                    contrasts.append({**base, "contrast": contrast, "descriptive_ci95": [-.05, .25],
                        "error_bound": report.POLICY["epsilon"] * sum(abs(v) for v in weights.values()),
                        "raw_sign": 1, "direction": "positive", "positive_query_count": n,
                        "negative_query_count": 0, "unresolved_query_count": 0,
                        "positive_query_fraction": 1., "negative_query_fraction": 0., "unresolved_query_fraction": 0.})
    conditions.extend([
        {"task": "group", "metric": "answer_sum/within_k/natural/Racism/0", "condition": "C0",
         "stratum": "all", "n": 0, "mean": None, "median": None, "status": "undefined"},
        {"task": "hate", "metric": "answer_sum/gold/mass", "condition": "CLnew",
         "stratum": "all", "n": 643, "mean": .5, "median": .5, "status": "defined"},
    ])
    return {"schema_version": "general-model-numeric-coverage-analysis/v1", "query_count": 643,
            "block_count": 10288, "epsilon": report.POLICY["epsilon"], "conditions": list(report.CONDITIONS),
            "score_modes": list(report.MODES), "group_order": list(report.LABELS),
            "contrasts": copy.deepcopy(report.CONTRASTS), "primary_contrasts": list(report.PRIMARY),
            "reference_contrasts": list(report.REFERENCE),
            "bootstrap": {"endpoint_count": 240, "replicates": 10000, "seed": 42, "scope": "descriptive",
                          "unit": "query", "interval": "percentile_95", "resampled_model_seeds": False,
                          "shared_draw_across_tasks_conditions_endpoints_within_stratum": True},
            "stratum_counts": dict(report.CI_COUNTS), "condition_summaries": conditions,
            "contrast_summaries": contrasts, "per_query": [{"query_id": str(i)} for i in range(643)]}


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "coverage-synthetic"
        self.audit_dir = self.root / "independent-audit"
        self.output = self.root / "report"
        self.plan_id = "gmlcoverage-synthetic"
        self.analysis = synthetic_analysis()
        self.runtime = {"synthetic": True}
        self.terminal = {"schema_version": "general-model-coverage-run/v1", "phase": "merged-lexicon-coverage",
                         "plan_id": self.plan_id, "status": "complete", "analysis_published": True,
                         "full_dev_started": True, "raw_path": "dev-b1", "production_batch_size": 1,
                         "numeric_policy": copy.deepcopy(report.POLICY), "test_content_read": False,
                         "query_gold_loaded_during_scoring": False, "automatic_profile_search": False}
        self.gate = {"schema_version": "general-model-coverage-preflight/v1", "plan_id": self.plan_id,
                     "passed": True, "complete": True, "numeric_policy": copy.deepcopy(report.POLICY),
                     "E8": report.POLICY["E8"], "epsilon": report.POLICY["epsilon"],
                     "runtime_identity": self.runtime, "error_families": list(report.CHALLENGES),
                     "calibration_mode": "inherited-fixed-tolerance-new-input-revalidation",
                     "not_applicable": ["batch-four", "tail-size-two", "within-batch-row-position"],
                     "query_gold_loaded": False, "test_content_read": False,
                     "scientific_effect_checked": False, "formal_test_authorized": False,
                     "observed_max_abs_error": .00002, "cohorts": {}}
        independent = {"passed": True, "epsilon_recalibrated": False,
                       "all_registered_numeric_readouts_recomputed": True,
                       "E8": report.POLICY["E8"], "epsilon": report.POLICY["epsilon"],
                       "sealed_pass_count": 18, "observed_max_abs_error": .00002, "cohorts": {}}
        for cohort, count in (("regression", 8), ("validation", 24), ("boundary", 4)):
            self.gate["cohorts"][cohort] = {"blocks": count * 16, "baseline_passed": True,
                "complete": True, "passed": True, "reference_max_abs_error": .00001,
                "baseline_repeat_max_abs_error": 0.,
                "challenges": {name: {"passed": True, "max_abs_error": .00002} for name in report.CHALLENGES}}
            independent["cohorts"][cohort] = {"queries": count, "passed": True,
                "passes": {name: {"synthetic": True} for name in ("baseline", "repeat", *report.CHALLENGES)},
                "comparisons": {name: {"max_abs_error": 0. if name == "repeat" else .00001 if name == "reference" else .00002}
                                for name in ("reference", "repeat", *report.CHALLENGES)}}
        self.raw = {"status": "complete", "blocks": 10288, "candidates": 174896,
                    "query_gold_loaded": False, "test_content_read": False, "mixed_execution_modes": False,
                    "identity": {"plan_id": self.plan_id, "batch_size": 1, "reference": False, "pass_name": "dev-b1",
                                 "runtime": self.runtime, "scoring_profile": {"candidate_permutation": "canonical",
                                    "padding_extra": 0, "prefix": False, "replica_shift": 0}}}
        self.audit = {"schema_version": "independent-coverage-full-dev-audit/v1", "plan_id": self.plan_id,
                      "audit_passed": True, "scientific_tables_written": True, "run_status": "complete",
                      "preflight_passed": True, "raw_path": "dev-b1", "production_batch_size": 1,
                      "full_raw_true_batch_one_geometry_verified": True, "raw_validated_before_gold_access": True,
                      "test_content_read": False, "queries": 643, "blocks": 10288, "candidates": 174896,
                      "all_ci_verified": True, "ci_endpoint_count": 240, "independent_preflight": independent,
                      "verification": {"independently_recomputed_ci_endpoints_per_stratum": 240,
                          "independently_recomputed_ci_strata": 6, "max_ci_endpoint_discrepancy": 0.}}
        self.audit_dir.mkdir()
        (self.run / "dev-b1").mkdir(parents=True)
        # Opaque raw intentionally cannot be deserialized; only its bytes may be hashed.
        (self.run / "dev-b1/scores.jsonl").write_text("opaque synthetic payload\n", encoding="utf-8")
        for name in report.EXTRA_TABLES:
            (self.audit_dir / name).write_bytes(b"field,value\r\nsynthetic,0.25\r\n")
        self.seal()

    def seal(self):
        write_json(self.run / "runtime_identity.json", self.runtime)
        write_json(self.run / "preflight/preflight_report.json", self.gate)
        self.raw["scores_sha256"] = report.digest(self.run / "dev-b1/scores.jsonl")
        write_json(self.run / "dev-b1/manifest.json", self.raw)
        write_json(self.run / "analysis/analysis.json", self.analysis)
        manifest = {"schema_version": "general-model-coverage-analysis/v1", "plan_id": self.plan_id,
                    "raw_manifest_sha256": report.digest(self.run / "dev-b1/manifest.json"),
                    "analysis_sha256": report.digest(self.run / "analysis/analysis.json"),
                    "gold_join_after_raw_sealed": True, "test_content_read": False}
        write_json(self.run / "analysis/manifest.json", manifest)
        self.terminal.update(preflight_report_sha256=report.digest(self.run / "preflight/preflight_report.json"),
                             raw_manifest_sha256=report.digest(self.run / "dev-b1/manifest.json"),
                             analysis_manifest_sha256=report.digest(self.run / "analysis/manifest.json"))
        write_json(self.run / "run_manifest.json", self.terminal)
        self.audit.update(report.artifact_hashes(self.run))
        self.audit["files"] = {name: report.digest(self.audit_dir / name) for name in report.EXTRA_TABLES}
        write_json(self.audit_dir / "audit.json", self.audit)

    def render(self):
        return report.render(self.run, self.audit_dir, self.output)

    def rows(self, filename):
        with (self.output / filename).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_sealed_complete_fixture_exports_all_registered_results_without_raw_deserialization(self):
        manifest = self.render()
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(manifest["raw_scores_deserialized"])
        self.assertFalse(manifest["query_gold_file_read"])
        self.assertFalse(manifest["recomputed_statistics"])
        for name, count in (("primary_effects.csv", 24), ("secondary_effects.csv", 24), ("reference_effects.csv", 12)):
            rows = self.rows(name)
            self.assertEqual(len(rows), count)
            self.assertEqual({row["mean"] for row in rows}, {"0.125"})
            self.assertEqual({row["ci95_low"] for row in rows}, {"-0.05"})
        self.assertEqual(len(self.rows("margin_effects.csv")), 240 * 6)
        self.assertEqual(len(self.rows("condition_margins.csv")), 4 * 6 * 8 * 6)
        self.assertEqual(len(self.rows("gold_readouts.csv")), 1)
        missing = self.rows("distribution_cardinality_readouts.csv")[0]
        self.assertEqual(missing["mean"], "")
        self.assertEqual(missing["status"], "undefined")
        self.assertEqual(missing["ci95_low"], "")
        for name in report.EXTRA_TABLES:
            self.assertEqual((self.output / name).read_bytes(), (self.audit_dir / name).read_bytes())
        body = (self.output / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("Lnew=Lq∪Ld", body)
        self.assertIn("Lnew 仍保留示例带来的 Ld", body)
        self.assertIn("1,440", body)
        self.assertIn("不是区间上下界的数量", body)
        self.assertIn("平均 log-odds 变化不等于平均包含概率变化", body)

    def test_incomplete_or_old_run_is_rejected_before_loading_analysis(self):
        self.terminal["status"] = "raw_complete"
        self.seal()
        original_read = report.read_json
        with mock.patch.object(report, "read_json", wraps=original_read) as reader:
            with self.assertRaisesRegex(report.ReportError, "run must be complete"):
                self.render()
        self.assertEqual([Path(call.args[0]).name for call in reader.call_args_list], ["run_manifest.json"])
        self.assertFalse(self.output.exists())

    def test_failed_preflight_and_incomplete_raw_fail(self):
        self.gate["passed"] = False
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "preflight"):
            self.render()
        self.gate["passed"] = True
        self.raw["candidates"] -= 1
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "raw coverage"):
            self.render()

    def test_partial_ci_audit_and_missing_reference_interval_fail(self):
        self.audit["all_ci_verified"] = False
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "240 CI"):
            self.render()
        self.audit["all_ci_verified"] = True
        row = next(row for row in self.analysis["contrast_summaries"] if row["contrast"] == report.REFERENCE[0])
        row.pop("descriptive_ci95")
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "missing interval"):
            self.render()

    def test_missing_condition_or_contrast_summary_fails(self):
        removed = self.analysis["condition_summaries"].pop(0)
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "eight-condition"):
            self.render()
        self.analysis["condition_summaries"].append(removed)
        self.analysis["contrast_summaries"].pop()
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "240-by-six"):
            self.render()

    def test_no_extra_auxiliary_ci_is_accepted(self):
        self.analysis["condition_summaries"][-1]["descriptive_ci95"] = [.2, .8]
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "CI outside registered"):
            self.render()

    def test_stale_artifact_and_audit_table_hashes_fail(self):
        (self.run / "dev-b1/scores.jsonl").write_text("changed opaque raw")
        with self.assertRaisesRegex(report.ReportError, "audit hashes"):
            self.render()
        self.seal()
        (self.audit_dir / report.EXTRA_TABLES[0]).write_text("changed table")
        with self.assertRaisesRegex(report.ReportError, "audited table changed"):
            self.render()

    def test_terminal_analysis_manifest_hash_is_required(self):
        self.terminal["analysis_manifest_sha256"] = "wrong"
        write_json(self.run / "run_manifest.json", self.terminal)
        self.audit.update(report.artifact_hashes(self.run))
        write_json(self.audit_dir / "audit.json", self.audit)
        with self.assertRaisesRegex(report.ReportError, "terminal hash"):
            self.render()

    def test_numerical_policy_cannot_be_loosened(self):
        self.gate["epsilon"] *= 2
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "tolerance changed"):
            self.render()

    def test_boundary_pass_coverage_and_failed_challenge_are_required(self):
        self.audit["independent_preflight"]["sealed_pass_count"] = 12
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "sealed pass count"):
            self.render()
        self.audit["independent_preflight"]["sealed_pass_count"] = 18
        self.gate["cohorts"]["boundary"]["challenges"]["padding"]["passed"] = False
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "challenge differs"):
            self.render()

    def test_original_lq_strata_cannot_be_renamed_to_generic_lexicon_hits(self):
        self.analysis["stratum_counts"]["lex_hit"] = self.analysis["stratum_counts"].pop("Lq_hit")
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "Lq or gold CI population"):
            self.render()

    def test_no_overwrite_or_output_inside_sealed_inputs(self):
        self.output.mkdir()
        sentinel = self.output / "existing.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(report.ReportError, "refusing overwrite"):
            self.render()
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaisesRegex(report.ReportError, "inside sealed run"):
            report.render(self.run, self.audit_dir, self.run / "new-report")

    def test_numerically_unresolved_and_empty_appendix_preserve_values(self):
        rendered = report.estimate({"n": 1, "mean": .00001, "descriptive_ci95": [-.1, .1],
                                    "direction": "numerically_unresolved"})
        self.assertIn("1e-05", rendered)
        self.assertIn("数值未分辨", rendered)
        self.analysis["stratum_counts"]["gold_size_3_appendix"] = 0
        self.analysis["contrast_summaries"].append({"task": "group", "metric": "answer_sum/margin/Racism",
            "contrast": "L", "stratum": "gold_size_3_appendix", "n": 0, "mean": None,
            "median": None, "status": "undefined"})
        self.seal()
        self.render()
        rows = [row for row in self.rows("margin_effects.csv") if row["stratum"] == "gold_size_3_appendix"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mean"], rows[0]["ci95_low"])
        self.assertEqual(rows[0]["mean"], "")

    def test_duplicate_summary_and_borrowed_plan_audit_fail(self):
        self.analysis["contrast_summaries"].append(copy.deepcopy(self.analysis["contrast_summaries"][0]))
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "duplicate"):
            self.render()
        self.analysis["contrast_summaries"].pop()
        self.audit["plan_id"] = "gmlcoverage-other"
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "plan identities"):
            self.render()


if __name__ == "__main__":
    unittest.main()
