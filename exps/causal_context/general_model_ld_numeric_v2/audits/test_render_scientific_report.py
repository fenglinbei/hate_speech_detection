import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("render_scientific_report.py")
SPEC = importlib.util.spec_from_file_location("render_scientific_report", PATH)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def synthetic_analysis():
    counts = {"all": 643, "lex_hit": 223, "lex_no_hit": 420,
              "gold_size_0": 252, "gold_size_1": 328, "gold_size_2plus": 63}
    conditions, contrasts = [], []
    for task, label in report.READOUTS:
        for mode in report.MODES:
            for stratum, n in counts.items():
                base = {"task": task, "metric": f"{mode}/margin/{label}", "stratum": stratum,
                        "n": n, "mean": 0.125, "median": 0.1, "p05": -0.5, "p95": 0.6,
                        "q25": -0.1, "q75": 0.3, "iqr": 0.4, "min": -1, "max": 1, "status": "defined"}
                for condition in report.CONDITIONS:
                    conditions.append({**base, "condition": condition})
                for contrast in report.PRIMARY + report.SECONDARY:
                    contrasts.append({**base, "contrast": contrast, "descriptive_ci95": [-0.05, 0.25],
                        "error_bound": 0.0004 if contrast == "LxD" else 0.0002,
                        "raw_sign": 1, "direction": "positive", "positive_query_count": n,
                        "negative_query_count": 0, "unresolved_query_count": 0,
                        "positive_query_fraction": 1.0, "negative_query_fraction": 0.0,
                        "unresolved_query_fraction": 0.0})
    conditions.append({"task": "group", "metric": "answer_sum/within_k/natural/Racism/0",
                       "condition": "C0", "stratum": "all", "n": 0, "mean": None,
                       "median": None, "status": "undefined"})
    return {"query_count": 643, "block_count": 7716, "epsilon": 0.0001,
            "conditions": list(report.CONDITIONS), "score_modes": list(report.MODES),
            "contrasts": copy.deepcopy(report.CONTRAST_WEIGHTS),
            "bootstrap": {"endpoint_count": 192, "replicates": 10000, "seed": 42, "scope": "descriptive"},
            "stratum_counts": counts, "condition_summaries": conditions, "contrast_summaries": contrasts,
            "per_query": [{"query_id": str(index)} for index in range(643)]}


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "numeric-03"
        self.audit_dir = self.root / "audit"
        self.output = self.root / "report"
        self.plan_id = "synthetic-plan"
        self.analysis = synthetic_analysis()
        self.terminal = {"plan_id": self.plan_id, "status": "complete", "analysis_published": True,
                         "test_content_read": False, "query_gold_loaded_during_scoring": False}
        self.gate = {"plan_id": self.plan_id, "passed": True, "complete": True, "epsilon": 0.0001}
        self.raw = {"status": "complete", "blocks": 7716, "candidates": 131172,
                    "identity": {"plan_id": self.plan_id, "batch_size": 4, "reference": False, "pass_name": "dev-b4",
                                 "scoring_profile": {"candidate_permutation": "canonical", "padding_extra": 0,
                                                     "prefix": False, "replica_shift": 0}}}
        self.audit = {"plan_id": self.plan_id, "audit_passed": True, "scientific_tables_written": True,
                      "run_status": "complete", "preflight_passed": True, "queries": 643,
                      "raw_path": "dev-b4", "production_batch_size": 4,
                      "blocks": 7716, "candidates": 131172, "all_ci_verified": True, "ci_endpoint_count": 192,
                      "verification": {"independently_recomputed_ci_endpoints_per_stratum": 192,
                          "independently_recomputed_ci_strata": 6, "max_ci_endpoint_discrepancy": 0.0}}
        self.audit_dir.mkdir()
        (self.run / "dev-b4").mkdir(parents=True)
        # This deliberately is not parseable JSON: the renderer may hash raw but must never deserialize it.
        (self.run / "dev-b4/scores.jsonl").write_text("synthetic opaque raw payload\n", encoding="utf-8")
        for name in report.EXTRA_TABLES:
            (self.audit_dir / name).write_bytes(b"field,value\r\nsynthetic,0.25\r\n")
        self.seal()

    def seal(self):
        raw_name = self.terminal.get("raw_path", "dev-b4")
        write_json(self.run / "preflight/preflight_report.json", self.gate)
        if "independent_fallback_preflight" in self.audit:
            self.audit["independent_fallback_preflight"]["final_report"]["report_sha256"] = report.digest(self.run / "preflight/preflight_report.json")
        self.raw["scores_sha256"] = report.digest(self.run / raw_name / "scores.jsonl")
        write_json(self.run / raw_name / "manifest.json", self.raw)
        write_json(self.run / "analysis/analysis.json", self.analysis)
        analysis_manifest = {"plan_id": self.plan_id,
            "raw_manifest_sha256": report.digest(self.run / raw_name / "manifest.json"),
            "analysis_sha256": report.digest(self.run / "analysis/analysis.json"),
            "gold_join_after_raw_sealed": True, "test_content_read": False}
        write_json(self.run / "analysis/manifest.json", analysis_manifest)
        self.terminal.update(preflight_report_sha256=report.digest(self.run / "preflight/preflight_report.json"),
                             raw_manifest_sha256=report.digest(self.run / raw_name / "manifest.json"))
        write_json(self.run / "run_manifest.json", self.terminal)
        self.audit.update(report.artifact_hashes(self.run))
        self.audit["files"] = {name: report.digest(self.audit_dir / name) for name in report.EXTRA_TABLES}
        write_json(self.audit_dir / "audit.json", self.audit)

    def render(self):
        return report.render(self.run, self.audit_dir, self.output)

    def fallback_fixture(self):
        (self.run / "dev-b4").rename(self.run / "dev-b1")
        source = "synthetic-source-calibration"
        challenges = ["padding", "prefix", "members", "replica"]
        self.terminal.update(schema_version="general-model-ld-numeric-run/v3", raw_path="dev-b1",
                             production_batch_size=1, source_calibration_sha256=source, source_failed_run="synthetic-failed-run")
        self.raw["identity"].update(batch_size=1, pass_name="dev-b1", runtime={})
        self.audit.update(raw_path="dev-b1", production_batch_size=1, source_failed_run="synthetic-failed-run",
                          source_calibration_sha256=source, source_E8=.00003, source_epsilon=.0001,
                          applicable_challenges=challenges, source_runtime_identity_sha256=report.canonical_hash({}),
                          full_raw_true_batch_one_geometry_verified=True,
                          independent_fallback_preflight={"preflight_passed": True, "sealed_pass_count": 12,
                              "all_four_regression_challenges_complete": True, "all_four_validation_challenges_complete": True,
                              "final_report": {}})
        self.gate.update(schema_version="general-model-ld-numeric-calibration/v3", E8=.00003,
                         calibration_mode="inherited-source-failure-no-recalibration", source_calibration_sha256=source,
                         runtime_identity={},
                         error_families=challenges, not_applicable=["batch-four", "tail-size-two", "within-batch-row-position"],
                         validation_cohort_status="previously-exposed-preregistered-fallback-revalidation",
                         cohorts={cohort: {"blocks": count, "baseline_passed": True, "complete": True, "passed": True,
                                           "baseline_repeat_max_abs_error": 0.0, "reference_max_abs_error": .00001,
                                           "challenges": {name: {"passed": True, "max_abs_error": .00002} for name in challenges}}
                                  for cohort, count in (("regression", 96), ("validation", 288))})
        self.seal()

    def test_complete_audited_fixture_renders_compact_report_and_exact_exports(self):
        manifest = self.render()
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(manifest["raw_scores_deserialized"])
        self.assertFalse(manifest["query_gold_file_read"])
        self.assertFalse(manifest["recomputed_statistics"])
        self.assertLessEqual(len((self.output / "REPORT.md").read_text().splitlines()), 150)
        for filename in ("primary_effects.csv", "secondary_effects.csv"):
            with (self.output / filename).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 24)
            self.assertEqual({row["mean"] for row in rows}, {"0.125"})
            self.assertEqual({row["ci95_low"] for row in rows}, {"-0.05"})
        with (self.output / "margin_effects.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 192 * 6)
        self.assertEqual((self.output / report.EXTRA_TABLES[0]).read_bytes(),
                         (self.audit_dir / report.EXTRA_TABLES[0]).read_bytes())
        with (self.output / "auxiliary_readouts.csv").open() as handle:
            missing = list(csv.DictReader(handle))[0]
        self.assertEqual(missing["mean"], "")
        self.assertEqual(missing["status"], "undefined")

    def test_incomplete_run_is_rejected_before_creating_output(self):
        self.terminal["status"] = "raw_complete"
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "run must be complete"):
            self.render()
        self.assertFalse(self.output.exists())

    def test_failed_preflight_or_incomplete_raw_cannot_render(self):
        self.gate["passed"] = False
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "preflight"):
            self.render()
        self.gate["passed"] = True
        self.raw["candidates"] = 131171
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "raw coverage"):
            self.render()

    def test_partial_ci_audit_is_rejected_even_when_audit_passed(self):
        self.audit["all_ci_verified"] = False
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "192"):
            self.render()
        self.assertFalse(self.output.exists())

    def test_missing_endpoint_or_interval_is_rejected(self):
        removed = self.analysis["contrast_summaries"].pop()
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "endpoint matrix"):
            self.render()
        self.analysis["contrast_summaries"].append(removed)
        removed.pop("descriptive_ci95")
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "confidence interval"):
            self.render()

    def test_hash_drift_is_rejected_without_parsing_raw(self):
        (self.run / "dev-b4/scores.jsonl").write_text("changed opaque raw\n")
        with self.assertRaisesRegex(report.ReportError, "audit hashes"):
            self.render()
        self.assertFalse(self.output.exists())

    def test_audited_auxiliary_file_drift_is_rejected(self):
        (self.audit_dir / report.EXTRA_TABLES[0]).write_text("changed\n")
        with self.assertRaisesRegex(report.ReportError, "audited table changed"):
            self.render()

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "user.txt"
        sentinel.write_text("preserve")
        with self.assertRaisesRegex(report.ReportError, "refusing overwrite"):
            self.render()
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_duplicate_summaries_are_rejected(self):
        self.analysis["contrast_summaries"].append(copy.deepcopy(self.analysis["contrast_summaries"][0]))
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "duplicate"):
            self.render()

    def test_mismatched_plan_cannot_borrow_another_audit(self):
        self.audit["plan_id"] = "different-plan"
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "plan identities"):
            self.render()

    def test_numerically_unresolved_cell_is_marked_without_changing_value(self):
        row = {"n": 1, "mean": 0.00001, "descriptive_ci95": [-0.2, 0.2], "direction": "numerically_unresolved"}
        rendered = report.estimate(row)
        self.assertIn("1e-05", rendered)
        self.assertIn("数值未分辨", rendered)

    def test_empty_appendix_is_exported_as_undefined_without_inventing_zero(self):
        name = "gold_size_3_appendix"
        self.analysis["stratum_counts"][name] = 0
        for family in ("condition", "contrast"):
            self.analysis[family + "_summaries"].append({"task": "group", "metric": "answer_sum/margin/Racism",
                family: "C0" if family == "condition" else "L", "stratum": name,
                "n": 0, "mean": None, "median": None, "status": "undefined"})
        self.seal()
        self.render()
        with (self.output / "margin_effects.csv").open() as handle:
            rows = [row for row in csv.DictReader(handle) if row["stratum"] == name]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mean"], "")
        self.assertEqual(rows[0]["ci95_low"], "")
        self.assertEqual(rows[0]["n"], "0")

    def test_empty_main_stratum_keeps_all_endpoints_and_renders_na(self):
        self.analysis["stratum_counts"]["gold_size_0"] = 0
        self.analysis["stratum_counts"]["gold_size_1"] += 252
        for family in ("condition", "contrast"):
            for row in self.analysis[family + "_summaries"]:
                if row["stratum"] == "gold_size_1":
                    row["n"] += 252
                    if family == "contrast":
                        row["positive_query_count"] += 252
                if row["stratum"] != "gold_size_0":
                    continue
                for key in report.SUMMARY_FIELDS:
                    row.pop(key, None)
                row.update(n=0, mean=None, median=None, status="undefined")
                if family == "contrast":
                    row["descriptive_ci95"] = None
        self.seal()
        self.render()
        body = (self.output / "REPORT.md").read_text()
        self.assertIn("| gold 大小 0 | hate | 0 | NA | NA | NA | NA |", body)
        with (self.output / "margin_effects.csv").open() as handle:
            rows = [row for row in csv.DictReader(handle) if row["stratum"] == "gold_size_0"]
        self.assertEqual(len(rows), 192)
        self.assertTrue(all(row["mean"] == row["ci95_low"] == row["direction"] == "" for row in rows))

    def test_batch_one_fallback_uses_dynamic_raw_path_and_four_challenge_disclosure(self):
        self.fallback_fixture()
        manifest = self.render()
        self.assertEqual(manifest["raw_path"], "dev-b1")
        self.assertEqual(manifest["production_batch_size"], 1)
        self.assertFalse(manifest["raw_scores_deserialized"])
        text = (self.output / "REPORT.md").read_text()
        self.assertIn("真实 batch 1", text)
        self.assertIn("不宣称原六项通过", text)
        self.assertIn("已暴露样本", text)

    def test_batch_one_raw_cannot_claim_batch_four_geometry(self):
        self.fallback_fixture()
        self.raw["identity"]["batch_size"] = 4
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "raw scoring geometry"):
            self.render()

    def test_fallback_cannot_recalibrate_epsilon(self):
        self.fallback_fixture()
        self.gate["epsilon"] = .0002
        self.analysis["epsilon"] = .0002
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "inherited E8 and epsilon"):
            self.render()

    def test_fallback_requires_its_four_applicable_challenges(self):
        self.fallback_fixture()
        self.gate["cohorts"]["validation"]["challenges"].pop("prefix")
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "four-challenge"):
            self.render()

    def test_fallback_source_calibration_binding_cannot_drift(self):
        self.fallback_fixture()
        self.gate["source_calibration_sha256"] = "different-source"
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "source calibration"):
            self.render()

    def test_fallback_runtime_cannot_drift_from_inherited_calibration(self):
        self.fallback_fixture()
        self.gate["runtime_identity"] = {"dtype": "different"}
        self.raw["identity"]["runtime"] = self.gate["runtime_identity"]
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "actual runtime"):
            self.render()

    def test_fallback_requires_independent_twelve_pass_proof(self):
        self.fallback_fixture()
        self.audit["independent_fallback_preflight"]["sealed_pass_count"] = 11
        self.seal()
        with self.assertRaisesRegex(report.ReportError, "twelve-pass"):
            self.render()


if __name__ == "__main__":
    unittest.main()
