import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics.general_model_package import PackageError, read_json, read_jsonl, write_json, write_jsonl
from diagnostics.general_model_runtime import (
    _verify_run,
    evaluate_run,
    run_local,
    select_contexts,
    summarize_preflight,
)
from model.stage1_registry import ModelRegistryError, ResolvedModelSourceContract, verified_model_source_lease


CONFIG = {
    "matrix": {"core_conditions": ["C0", "CD"], "primary_conditions": ["C0", "CD", "L-Category"]},
    "runtime": {"determinism_repetitions": 2, "classification_valid_rate_min": 0.95},
    "models": [{"key": "qwen3-8b"}],
}
ENVIRONMENT = {"python": "fixture", "packages": {}}
RUNTIME_IDENTITY = {"model_tree_sha256": "synthetic", "device": "mock-device"}


def context(query_id, task, condition):
    row = {
        "record_id": f"{query_id}:{task}:{condition}",
        "query_id": query_id, "task": task, "condition": condition,
        "overflow": False, "control_valid": True,
    }
    return {**row, "context_sha256": canonical_json_sha256(row)}


def generated(row, repetition=0):
    prediction = {"hate": "hate", "group": '["Region"]', "extraction": "[]"}[row["task"]]
    return {
        "record_id": row["record_id"], "context_sha256": row["context_sha256"],
        "repetition": repetition, "query_id": row["query_id"],
        "task": row["task"], "condition": row["condition"],
        "prediction": prediction, "output_token_ids": [7, 9], "output_tokens": 2,
        "termination": "eos", "finite_logits_checked": True, "elapsed_seconds": 0.0,
    }


def prediction_payload(row):
    return {key: value for key, value in generated(row).items()
            if key not in {"record_id", "context_sha256", "repetition", "query_id", "task", "condition"}}


class GeneralModelPreflightSummaryTest(unittest.TestCase):
    def setUp(self):
        self.contexts = [context("q", task, condition)
                         for task in ("hate", "group", "extraction") for condition in ("C0", "CD")]
        self.rows = [generated(row, rep) for rep in range(2) for row in self.contexts]

    def test_complete_deterministic_finite_preflight_passes_without_effect_gate(self):
        report = summarize_preflight(self.contexts, self.rows, CONFIG)
        self.assertTrue(report["passed"])
        self.assertTrue(report["complete"])
        self.assertTrue(report["deterministic"])
        self.assertTrue(report["finite_generation_logits"])
        self.assertFalse(report["scientific_effect_checked"])
        self.assertFalse(report["formal_test_authorized"])
        self.assertEqual(len(report["classification_valid_rates"]), 4)

    def test_missing_or_duplicate_record_cannot_count_as_complete(self):
        for changed in (self.rows[:-1], self.rows[:-1] + [copy.deepcopy(self.rows[0])]):
            report = summarize_preflight(self.contexts, changed, CONFIG)
            self.assertFalse(report["complete"])
            self.assertFalse(report["passed"])

    def test_changed_output_tokens_fail_determinism(self):
        self.rows[-1]["output_token_ids"] = [7, 10]
        report = summarize_preflight(self.contexts, self.rows, CONFIG)
        self.assertFalse(report["passed"])
        self.assertFalse(report["deterministic"])
        self.assertIn(self.rows[-1]["record_id"], report["unstable_record_ids"])

    def test_unchecked_finite_logits_fail_even_with_valid_answers(self):
        self.rows[0]["finite_logits_checked"] = False
        report = summarize_preflight(self.contexts, self.rows, CONFIG)
        self.assertFalse(report["finite_generation_logits"])
        self.assertFalse(report["passed"])

    def test_invalid_classification_fails_its_own_condition_rate(self):
        self.rows[0]["prediction"] = "unknown"
        report = summarize_preflight(self.contexts, self.rows, CONFIG)
        self.assertFalse(report["passed"])
        self.assertEqual(report["classification_valid_rates"]["hate:C0"], 0.0)

    def test_extraction_failure_and_length_are_reported_without_classification_inference(self):
        for row in self.rows:
            if row["task"] == "extraction":
                row["prediction"] = "malformed extraction"
                row["termination"] = "length"
        report = summarize_preflight(self.contexts, self.rows, CONFIG)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["length_limited_records"]), 4)
        self.assertFalse(any(key.startswith("extraction:") for key in report["classification_valid_rates"]))

    def test_recovery_is_not_reported_as_canonical_format(self):
        for row in self.rows:
            if row["task"] == "hate":
                row["prediction"] = '{"result":"hate"}'
        report = summarize_preflight(self.contexts, self.rows, CONFIG)
        self.assertTrue(report["passed"])
        cell = report["classification_format_diagnostics"]["hate:C0"]
        self.assertEqual(cell["valid_rate"], 1.0)
        self.assertEqual(cell["canonical_rate"], 0.0)
        self.assertEqual(cell["recovered_rate"], 1.0)
        self.assertEqual(cell["recovery_rule_counts"], {"object-field:result": 1})

    def test_length_limited_classification_cannot_pass_even_if_label_parses(self):
        for row in self.rows:
            if row["task"] == "hate":
                row["termination"] = "length"
        report = summarize_preflight(self.contexts, self.rows, CONFIG)
        self.assertFalse(report["passed"])
        self.assertEqual(report["classification_valid_rates"]["hate:C0"], 0.0)


class GeneralModelCohortPreflightTest(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(CONFIG)
        self.config["matrix"].update(preflight_query_count=8, preflight_validation_query_count=24)
        regression = [f"r{i}" for i in range(8)]
        validation = [f"v{i}" for i in range(24)]
        self.frames = {"preflight_query_ids": regression + validation,
                       "preflight_regression_query_ids": regression,
                       "preflight_validation_query_ids": validation}
        self.contexts = [context(qid, task, condition)
                         for qid in regression + validation
                         for task in (["hate", "group", "extraction"] if qid in regression else ["hate", "group"])
                         for condition in self.config["matrix"]["core_conditions"]]
        self.rows = [generated(row, rep) for rep in range(2) for row in self.contexts]

    def test_pooled_success_cannot_hide_regression_failure(self):
        for row in self.rows:
            if row["record_id"] == "r0:hate:C0":
                row["prediction"] = "unknown"
        report = summarize_preflight(self.contexts, self.rows, self.config, self.frames)
        self.assertGreaterEqual(report["classification_valid_rates"]["hate:C0"], 0.95)
        self.assertFalse(report["passed"])
        self.assertEqual(report["classification_gate_rates"]["regression:hate:C0"], 7 / 8)
        self.assertEqual(report["classification_cohort_diagnostics"]["validation:hate:C0"]["query_count"], 24)

    def test_validation_threshold_is_applied_to_its_own_denominator(self):
        for invalid_count, expected_pass in ((1, True), (2, False)):
            changed = copy.deepcopy(self.rows)
            for row in changed:
                if row["task"] == "hate" and row["condition"] == "C0" and row["query_id"] in {f"v{i}" for i in range(invalid_count)}:
                    row["prediction"] = "unknown"
            report = summarize_preflight(self.contexts, changed, self.config, self.frames)
            self.assertEqual(report["passed"], expected_pass)
            self.assertEqual(report["classification_gate_rates"]["validation:hate:C0"], (24 - invalid_count) / 24)

    def test_missing_overlapping_or_partial_cohort_is_rejected(self):
        with self.assertRaisesRegex(PackageError, "registered cohort"):
            summarize_preflight(self.contexts, self.rows, self.config)
        changed = copy.deepcopy(self.frames)
        changed["preflight_validation_query_ids"][0] = "r0"
        with self.assertRaisesRegex(PackageError, "cohorts differ"):
            summarize_preflight(self.contexts, self.rows, self.config, changed)
        with self.assertRaisesRegex(PackageError, "complete cohort matrix"):
            summarize_preflight(self.contexts[:-1], self.rows[:-1], self.config, self.frames)


class GeneralModelRuntimeWorkflowTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.package = self.root / "package"
        self.package.mkdir()
        self.contexts = [context(qid, task, condition)
                         for qid in ("q1", "q2") for task in ("hate", "group", "extraction")
                         for condition in (CONFIG["matrix"]["core_conditions"] if task == "extraction"
                                           else CONFIG["matrix"]["primary_conditions"])]
        write_json(self.package / "config.resolved.json", CONFIG)
        write_json(self.package / "frames.dev.json", {"preflight_query_ids": ["q1"]})
        write_json(self.package / "build_inputs.json", {"environment": ENVIRONMENT})
        write_jsonl(self.package / "contexts.dev.jsonl", self.contexts)
        write_jsonl(self.package / "queries.dev.jsonl", [
            {"id": qid, "projection": {"hate": "hate", "group": ["Region"], "extraction": [
                {"target": "target", "argument": "argument", "targeted_group": ["Region"], "hateful": "hate"}
            ]}}
            for qid in ("q1", "q2")
        ])
        self.validation_patch = patch("diagnostics.general_model_runtime.validate_package", return_value={"package_id": "fixture-package"})
        self.environment_patch = patch("diagnostics.general_model_runtime._environment", return_value=ENVIRONMENT)
        self.runner_patch = patch("diagnostics.general_model_runtime.LocalRunner")
        self.validate = self.validation_patch.start()
        self.environment_patch.start()
        self.runner = self.runner_patch.start()
        self.addCleanup(self.validation_patch.stop)
        self.addCleanup(self.environment_patch.stop)
        self.addCleanup(self.runner_patch.stop)
        self.runner.return_value.identity = copy.deepcopy(RUNTIME_IDENTITY)
        self.runner.return_value.generate.side_effect = prediction_payload

    def execute(self, output, **kwargs):
        with redirect_stdout(io.StringIO()):
            return run_local(self.package, output, root=self.root, device="mock-device", **kwargs)

    def preflight(self, name="preflight"):
        output = self.root / name
        self.execute(output, phase="preflight")
        return output

    def refresh_hashes(self, run):
        manifest = read_json(run / "run_manifest.json")
        manifest["files"] = {path.name: sha256_file(path) for path in run.iterdir()
                             if path.is_file() and path.name != "run_manifest.json"}
        write_json(run / "run_manifest.json", manifest)

    def test_select_contexts_requires_complete_preflight_but_allows_dev_subsets(self):
        preflight = select_contexts(self.package, "preflight")
        self.assertEqual(len(preflight), 6)
        self.assertEqual({row["query_id"] for row in preflight}, {"q1"})
        for kwargs in ({"tasks": ["hate"]}, {"conditions": ["C0"]}):
            with self.assertRaisesRegex(PackageError, "complete core matrix"):
                select_contexts(self.package, "preflight", **kwargs)
        dev = select_contexts(self.package, "dev", tasks=["hate"], conditions=["L-Category"])
        self.assertEqual(len(dev), 2)
        self.assertEqual({row["query_id"] for row in dev}, {"q1", "q2"})

    def test_test_phase_and_duplicate_selection_are_rejected(self):
        with self.assertRaisesRegex(PackageError, "test remains sealed"):
            select_contexts(self.package, "test")
        for kwargs in ({"tasks": ["hate", "hate"]}, {"conditions": ["C0", "C0"]}):
            with self.assertRaises(PackageError):
                select_contexts(self.package, "dev", **kwargs)

    def test_validation_queries_do_not_expand_extraction_frame(self):
        config = copy.deepcopy(CONFIG)
        config["matrix"].update(preflight_query_count=1, preflight_validation_query_count=1)
        write_json(self.package / "config.resolved.json", config)
        write_json(self.package / "frames.dev.json", {
            "preflight_query_ids": ["q1", "q2"], "preflight_regression_query_ids": ["q1"],
            "preflight_validation_query_ids": ["q2"],
        })
        contexts = select_contexts(self.package, "preflight")
        self.assertEqual(len(contexts), 10)
        self.assertEqual({row["query_id"] for row in contexts if row["task"] == "extraction"}, {"q1"})
        run = self.preflight()
        self.assertEqual(len(read_jsonl(run / "predictions.jsonl")), 20)
        output = self.root / "evaluation.json"
        evaluate_run(self.package, run, output, root=self.root)
        result = read_json(output)
        self.assertEqual(len(result["cohort_evaluations"]), 10)
        self.assertEqual(result["evaluations"]["hate:C0"]["query_count"], 2)
        self.assertEqual(result["cohort_evaluations"]["validation:hate:C0"]["query_count"], 1)
        self.assertNotIn("validation:extraction:C0", result["cohort_evaluations"])

    def test_dev_requires_preflight_before_constructing_a_runner(self):
        output = self.root / "dev"
        with self.assertRaisesRegex(PackageError, "requires a completed matching preflight"):
            self.execute(output, phase="dev")
        self.assertFalse(output.exists())
        self.runner.assert_not_called()

    def test_invalid_context_or_environment_is_rejected_before_runner(self):
        self.contexts[0]["overflow"] = True
        write_jsonl(self.package / "contexts.dev.jsonl", self.contexts)
        with self.assertRaisesRegex(PackageError, "invalid/overflowing"):
            self.execute(self.root / "overflow", phase="preflight")
        self.runner.assert_not_called()
        with patch("diagnostics.general_model_runtime._environment", return_value={"different": True}):
            with self.assertRaisesRegex(PackageError, "environment differs"):
                self.execute(self.root / "environment", phase="preflight")
        self.runner.assert_not_called()

    def test_full_preflight_and_dev_evaluation_use_only_planned_repetition_zero(self):
        preflight = self.preflight()
        report = read_json(preflight / "preflight_report.json")
        self.assertTrue(report["passed"])
        self.assertEqual(len(read_jsonl(preflight / "predictions.jsonl")), 12)
        dev = self.root / "dev"
        result = self.execute(dev, phase="dev", preflight=preflight, tasks=["hate"], conditions=["C0"])
        self.assertEqual(result["completed_record_count"], 2)
        output = self.root / "eval.json"
        evaluated = evaluate_run(self.package, dev, output, root=self.root)
        self.assertEqual(evaluated["condition_task_count"], 1)
        payload = read_json(output)
        self.assertEqual(payload["repetition_scored"], 0)
        self.assertFalse(payload["test_content_read"])
        self.assertEqual(set(payload["evaluations"]), {"hate:C0"})
        with self.assertRaisesRegex(PackageError, "already exists"):
            evaluate_run(self.package, dev, output, root=self.root)

    def test_failed_preflight_blocks_dev(self):
        self.runner.return_value.generate.side_effect = lambda row: {**prediction_payload(row), "prediction": "invalid"}
        preflight = self.preflight()
        self.assertFalse(read_json(preflight / "preflight_report.json")["passed"])
        self.runner.reset_mock()
        with self.assertRaisesRegex(PackageError, "preflight has not passed"):
            self.execute(self.root / "dev", phase="dev", preflight=preflight)
        self.runner.assert_not_called()

    def test_preflight_pass_flag_cannot_override_failed_prediction_replay(self):
        self.runner.return_value.generate.side_effect = lambda row: {**prediction_payload(row), "prediction": "invalid"}
        preflight = self.preflight()
        report = read_json(preflight / "preflight_report.json")
        report["passed"] = True
        write_json(preflight / "preflight_report.json", report)
        self.refresh_hashes(preflight)
        self.runner.reset_mock()
        with self.assertRaises(PackageError):
            self.execute(self.root / "dev", phase="dev", preflight=preflight)
        self.runner.assert_not_called()

    def test_runtime_identity_change_fails_before_any_dev_generation(self):
        preflight = self.preflight()
        self.runner.return_value.identity = {**RUNTIME_IDENTITY, "device": "different-device"}
        self.runner.return_value.generate.reset_mock()
        output = self.root / "dev"
        with self.assertRaisesRegex(PackageError, "identity differs"):
            self.execute(output, phase="dev", preflight=preflight)
        self.runner.return_value.generate.assert_not_called()
        self.assertEqual(read_json(output / "run_manifest.json")["status"], "failed")

    def test_generation_error_preserves_failed_partial_run_and_evaluation_refuses_it(self):
        emitted = prediction_payload(self.contexts[0])
        self.runner.return_value.generate.side_effect = [emitted, RuntimeError("synthetic failure")]
        output = self.root / "failed"
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            self.execute(output, phase="preflight")
        manifest = read_json(output / "run_manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["completed_record_count"], 1)
        with self.assertRaisesRegex(PackageError, "incomplete"):
            evaluate_run(self.package, output, self.root / "eval.json", root=self.root)

    def test_corrupt_run_payload_is_rejected(self):
        run = self.preflight()
        with (run / "predictions.jsonl").open("ab") as handle:
            handle.write(b"\n")
        with self.assertRaisesRegex(PackageError, "payload differs"):
            evaluate_run(self.package, run, self.root / "eval.json", root=self.root)

    def test_completed_manifest_must_hash_all_required_payload_files(self):
        run = self.preflight()
        for missing in ("planned_records.json", "predictions.jsonl", "runtime_identity.json"):
            manifest = read_json(run / "run_manifest.json")
            original = copy.deepcopy(manifest)
            del manifest["files"][missing]
            write_json(run / "run_manifest.json", manifest)
            with self.subTest(missing=missing), self.assertRaises(PackageError):
                _verify_run(run, "fixture-package")
            write_json(run / "run_manifest.json", original)

    def test_missing_prediction_and_wrong_context_fail_even_with_updated_checksums(self):
        run = self.preflight()
        original = read_jsonl(run / "predictions.jsonl")
        write_jsonl(run / "predictions.jsonl", original[:-1])
        self.refresh_hashes(run)
        with self.assertRaisesRegex(PackageError, "frame differs"):
            evaluate_run(self.package, run, self.root / "eval.json", root=self.root)
        altered = copy.deepcopy(original)
        altered[0]["query_id"] = "q2"
        write_jsonl(run / "predictions.jsonl", altered)
        self.refresh_hashes(run)
        with self.assertRaisesRegex(PackageError, "different context"):
            evaluate_run(self.package, run, self.root / "eval.json", root=self.root)

    def test_consistent_plan_and_predictions_subset_cannot_replace_selected_dev_frame(self):
        preflight = self.preflight()
        run = self.root / "dev"
        self.execute(run, phase="dev", preflight=preflight, tasks=["hate"], conditions=["C0"])
        rows = read_jsonl(run / "predictions.jsonl")[:1]
        planned = read_json(run / "planned_records.json")[:1]
        write_jsonl(run / "predictions.jsonl", rows)
        write_json(run / "planned_records.json", planned)
        manifest = read_json(run / "run_manifest.json")
        manifest["planned_record_count"] = 1
        manifest["completed_record_count"] = 1
        write_json(run / "run_manifest.json", manifest)
        self.refresh_hashes(run)
        with self.assertRaises(PackageError):
            evaluate_run(self.package, run, self.root / "eval.json", root=self.root)

    def test_model_lease_member_names_match_registry_contract(self):
        contract = ResolvedModelSourceContract(
            workspace_root=self.root, checkpoint_inventory={}, tokenizer_inventory={}, base_inventory={},
        )
        lease = verified_model_source_lease(contract, source_names=("checkpoint", "tokenizer"))
        self.assertIsNotNone(lease)
        with self.assertRaises(ModelRegistryError):
            verified_model_source_lease(contract, source_names=("checkpoint_inventory",))


if __name__ == "__main__":
    unittest.main()
