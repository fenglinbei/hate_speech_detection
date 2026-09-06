import copy
import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from data.stage1_data import canonical_json_sha256
from diagnostics.general_model_numeric import (
    CONDITIONS, Checkpoint, batch_groups, compare_passes, load_plan, preflight,
    run_pipeline, score_pass, validate_block,
)
from diagnostics.general_model_numeric_analysis import candidate_catalog, candidate_scores
from diagnostics.general_model_package import PackageError, write_json


VALIDATION = {
    "regression_query_count": 8, "validation_query_count": 24,
    "baseline_repetitions": 2, "accelerated_repetitions": 1,
    "repeat_abs_tolerance": 1e-4, "reference_abs_tolerance": 1e-4,
    "calibration_max_abs_error": 0.005, "epsilon_floor": 0.0001,
    "epsilon_multiplier": 2, "epsilon_ceiling": 0.005, "required_coverage": 1.0,
}


def catalog_fixture():
    catalog = candidate_catalog()
    for candidates in catalog.values():
        for candidate in candidates:
            ids = [candidate["ordinal"] + 1]
            candidate.update(answer_token_ids=ids, answer_tokens=1,
                             answer_token_ids_sha256=canonical_json_sha256(ids))
    return catalog


def context(query_id="1", task="hate", condition="C0"):
    row = {"record_id": f"{query_id}:{task}:{condition}", "query_id": query_id,
           "task": task, "condition": condition, "prompt_sha256": "fixture-prompt",
           "prompt_token_ids_sha256": "fixture-tokens", "prompt_tokens": 3}
    return {**row, "context_sha256": canonical_json_sha256(row)}


def plan_fixture():
    return {
        "plan_id": "fixture-plan", "catalog": catalog_fixture(), "package_path": "fixture-package",
        "config": {"tasks": ["hate", "group"], "conditions": CONDITIONS, "validation": copy.deepcopy(VALIDATION),
                   "runtime": {"cpu_threads": 4}},
        "cohorts": {"regression": [str(index) for index in range(1, 9)],
                    "validation": [str(index) for index in range(9, 33)]},
        "generation_runtime_identity": {"model": "fixture"},
    }


def full_preflight_contexts():
    return [context(str(index), task, condition)
            for task in ("hate", "group") for condition in CONDITIONS for index in range(1, 33)]


def score_fixture(candidate, *, shift=0.0, reference=False, reference_shift=0.0):
    values = [-3.0 - candidate["ordinal"] / 100 + shift]
    eos = -0.3
    scores = candidate_scores(values, eos)
    result = {
        "answer_token_ids": candidate["answer_token_ids"],
        "answer_token_ids_sha256": candidate["answer_token_ids_sha256"],
        "answer_tokens": 1, "token_logprobs": values, "eos_token_id": 127,
        **scores, "scores": scores, "finite": True,
        "finite_target_logits_checked": True, "token_boundary_checked": True,
        "reference_checked": reference,
    }
    if reference:
        ref_values = [value + reference_shift for value in values]
        ref_scores = candidate_scores(ref_values, eos)
        result.update(reference_scores={"token_logprobs": ref_values, **ref_scores},
                      reference_token_logprobs=ref_values, reference_eos_logprob=eos,
                      reference_abs_error_max=max(abs(scores[key] - ref_scores[key]) for key in scores))
    return result


def block_fixture(row, catalog, *, shift=0.0, reference=False, reference_margin_error=False):
    candidates = []
    for candidate in catalog[row["task"]]:
        delta = 0.0
        if reference_margin_error and row["task"] == "hate":
            delta = 7.5e-5 if candidate["ordinal"] == 0 else -7.5e-5
        candidates.append({**candidate, **score_fixture(candidate, shift=shift,
                                                       reference=reference, reference_shift=delta)})
    return {**row, "candidates": candidates}


class NumericBatchGeometryTests(unittest.TestCase):
    def test_group_candidates_stay_in_one_query_and_hate_pairs_share_only_a_cell(self):
        contexts = [context("1"), context("2"), context("3"), context("4", condition="CL"),
                    context("1", task="group"), context("2", task="group")]
        groups = list(batch_groups(contexts, catalog_fixture(), 4))
        self.assertEqual([len(rows) for rows, _ in groups], [2, 1, 1, 1, 1])
        ordinals = [ordinal for _, batches in groups for ordinal, _ in batches]
        self.assertEqual(ordinals, list(range(len(ordinals))))
        self.assertEqual([len(items) for _, items in groups[0][1]], [4])
        self.assertEqual([len(items) for _, items in groups[1][1]], [2])
        for rows, batches in groups[3:]:
            self.assertEqual(len(batches), 8)
            self.assertEqual([item["candidate"]["ordinal"] for _, batch in batches for item in batch], list(range(32)))
            self.assertTrue(all(item["context"]["record_id"] == rows[0]["record_id"] for _, batch in batches for item in batch))

    def test_batch_one_preserves_canonical_order_and_contains_one_sequence(self):
        contexts = [context("1"), context("1", task="group")]
        groups = list(batch_groups(contexts, catalog_fixture(), 1))
        self.assertEqual([len(batches) for _, batches in groups], [2, 32])
        self.assertTrue(all(len(items) == 1 for _, batches in groups for _, items in batches))
        with self.assertRaises(PackageError):
            list(batch_groups(contexts, catalog_fixture(), 2))


class NumericCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_identity_prevents_cross_runtime_or_batch_reuse(self):
        path = self.root / "checkpoint.sqlite3"
        original = {"plan_id": "p", "batch_size": 4, "runtime": {"model": "m"}, "reference": False}
        checkpoint = Checkpoint(path, original)
        checkpoint.close()
        for field, changed in (("plan_id", "q"), ("batch_size", 1),
                               ("runtime", {"model": "other"}), ("reference", True)):
            with self.subTest(field=field), self.assertRaises(PackageError):
                Checkpoint(path, {**original, field: changed})

    def test_group_commit_rolls_back_all_rows_on_failure(self):
        checkpoint = Checkpoint(self.root / "checkpoint.sqlite3", {"id": 1})
        self.addCleanup(checkpoint.close)
        checkpoint.commit([{"record_id": "existing"}])
        with self.assertRaises(sqlite3.IntegrityError):
            checkpoint.commit([{"record_id": "new"}, {"record_id": "existing"}])
        self.assertEqual(set(checkpoint.rows()), {"existing"})

    def test_modified_payload_is_rejected(self):
        checkpoint = Checkpoint(self.root / "checkpoint.sqlite3", {"id": 1})
        self.addCleanup(checkpoint.close)
        checkpoint.commit([{"record_id": "q", "value": 1}])
        checkpoint.connection.execute("UPDATE blocks SET payload=? WHERE key='q'", ('{"record_id":"q","value":2}',))
        checkpoint.connection.commit()
        with self.assertRaisesRegex(PackageError, "hash"):
            checkpoint.rows()


class NumericScorePassTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = plan_fixture()
        self.runner = SimpleNamespace(identity={"model": "fixture"})
        self.contexts = [context("1"), context("2"), context("1", task="group")]
        self.scorer = patch("diagnostics.general_model_numeric.score_batch", side_effect=self.fake_scores).start()
        self.addCleanup(patch.stopall)

    def fake_scores(self, runner, items, *, reference=False):
        return [score_fixture(item["candidate"], reference=reference) for item in items]

    def execute(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return score_pass(self.runner, self.contexts, self.plan, self.root / "pass", batch_size=4, **kwargs)

    def test_complete_pass_and_identical_resume_do_not_rescore(self):
        rows, receipt = self.execute()
        self.assertEqual(receipt["blocks"], 3)
        self.assertEqual(receipt["candidates"], 36)
        self.assertEqual(self.scorer.call_count, 9)
        self.scorer.reset_mock()
        resumed, resumed_receipt = self.execute()
        self.assertEqual(resumed, rows)
        self.assertEqual(resumed_receipt, receipt)
        self.scorer.assert_not_called()

    def test_failed_group_does_not_commit_a_partial_query(self):
        calls = 0
        def fail_mid_group(runner, items, *, reference=False):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("fixture interrupted")
            return self.fake_scores(runner, items, reference=reference)
        self.scorer.side_effect = fail_mid_group
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.execute()
        with sqlite3.connect(self.root / "pass/checkpoint.sqlite3") as connection:
            self.assertEqual({row[0] for row in connection.execute("SELECT key FROM blocks")}, {"1:hate:C0", "2:hate:C0"})
        self.assertFalse((self.root / "pass/manifest.json").exists())

    def test_missing_candidate_result_blocks_sealing(self):
        self.scorer.side_effect = lambda runner, items, reference=False: self.fake_scores(runner, items, reference=reference)[:-1]
        with self.assertRaisesRegex(PackageError, "length"):
            self.execute()
        self.assertFalse((self.root / "pass/manifest.json").exists())

    def test_tampered_sealed_raw_fails_resume(self):
        self.execute()
        path = self.root / "pass/scores.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"eos_logprob":-0.3', b'"eos_logprob":-0.4', 1))
        with self.assertRaisesRegex(PackageError, "sealed score pass"):
            self.execute()

    def test_candidate_omission_and_wrong_identity_fail_block_validation(self):
        row = block_fixture(self.contexts[0], self.plan["catalog"])
        incomplete = copy.deepcopy(row)
        incomplete["candidates"].pop()
        with self.assertRaises(PackageError):
            validate_block(incomplete, self.contexts[0], self.plan["catalog"])
        wrong = copy.deepcopy(row)
        wrong["candidates"][0]["candidate_id"] = "non-hate"
        with self.assertRaises(PackageError):
            validate_block(wrong, self.contexts[0], self.plan["catalog"])


class NumericPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan, self.contexts = plan_fixture(), full_preflight_contexts()
        self.runner = SimpleNamespace(identity={"model": "fixture"})
        self.regression_shift = 4e-5
        self.validation_shift = 8e-5
        self.repeat_shift = 0.0
        self.reference_margin_error = False
        self.calls = []
        patcher = patch("diagnostics.general_model_numeric.score_pass", side_effect=self.fake_pass)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_pass(self, runner, contexts, plan, output, *, batch_size, reference=False):
        self.calls.append((output.name, {r["query_id"] for r in contexts}, batch_size))
        shift = self.repeat_shift if output.name.endswith("b1-r1") else 0.0
        if batch_size == 4:
            shift = self.regression_shift if output.name.startswith("regression") else self.validation_shift
        rows = [block_fixture(row, plan["catalog"], shift=shift, reference=reference,
                              reference_margin_error=self.reference_margin_error) for row in contexts]
        return rows, {"status": "complete", "blocks": len(rows)}

    def execute(self):
        with redirect_stdout(io.StringIO()):
            return preflight(self.runner, self.plan, self.contexts, self.root)

    def test_validation_is_disjoint_and_cannot_recalibrate_epsilon(self):
        self.validation_shift = 1.5e-4
        report = self.execute()
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["E8"], self.regression_shift, places=12)
        self.assertEqual(report["epsilon"], 1e-4)
        self.assertEqual(report["failure"], "validation-batch-compatibility-gate")
        for name, ids, _ in self.calls:
            cohort = "regression" if name.startswith("regression") else "validation"
            if name.endswith("tail2"):
                self.assertEqual(len(ids), 1)
                self.assertTrue(ids.issubset(self.plan["cohorts"][cohort]))
            else:
                self.assertEqual(ids, set(self.plan["cohorts"][cohort]))

    def test_complete_numerically_stable_preflight_passes_without_effect_gate(self):
        report = self.execute()
        self.assertTrue(report["passed"])
        self.assertTrue(report["complete"])
        self.assertFalse(report["scientific_effect_checked"])
        self.assertEqual(report["cohorts"]["regression"]["blocks"], 96)
        self.assertEqual(report["cohorts"]["validation"]["blocks"], 288)

    def test_regression_batch_failure_never_runs_validation(self):
        self.regression_shift = 0.006
        report = self.execute()
        self.assertFalse(report["passed"])
        self.assertFalse(report["validation_executed"])
        self.assertTrue(all(name.startswith("regression") for name, _, _ in self.calls))

    def test_baseline_repeat_failure_stops_before_accelerated_pass(self):
        self.repeat_shift = 2e-4
        report = self.execute()
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "regression-baseline-gate")
        self.assertEqual([batch for _, _, batch in self.calls], [1, 1])

    def test_cpu_reference_gate_includes_margin_error_not_just_candidate_scores(self):
        self.reference_margin_error = True
        report = self.execute()
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "regression-baseline-gate")
        self.assertGreater(report["cohorts"]["regression"]["reference_max_abs_error"], 1e-4)

    def test_missing_context_duplicate_and_overlapping_cohort_fail_closed(self):
        for modification in ("missing", "duplicate", "overlap"):
            self.contexts = full_preflight_contexts()
            self.plan = plan_fixture()
            if modification == "missing":
                self.contexts.pop()
            elif modification == "duplicate":
                self.contexts.append(copy.deepcopy(self.contexts[0]))
            else:
                self.plan["cohorts"]["validation"][0] = "1"
            with self.subTest(modification=modification), self.assertRaises(PackageError):
                self.execute()

    def test_comparison_includes_eos_and_each_token(self):
        row = context()
        baseline = block_fixture(row, self.plan["catalog"])
        observed = copy.deepcopy(baseline)
        changed = observed["candidates"][0]
        changed["eos_logprob"] += 0.004
        changed["scores"] = candidate_scores(changed["token_logprobs"], changed["eos_logprob"])
        compared = compare_passes([baseline], [observed])
        self.assertAlmostEqual(compared["max_abs_error"], 0.004)
        self.assertIn("candidate/hate/eos_logprob", compared["blocks"][0]["differences"])


class NumericLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_failed_preflight_never_starts_full_dev_or_analysis(self):
        plan = plan_fixture()
        model_runner = SimpleNamespace(identity=plan["generation_runtime_identity"], model=object(), torch=MagicMock())
        def failed_preflight(runner, actual_plan, contexts, output):
            output.mkdir(parents=True)
            report = {"plan_id": plan["plan_id"], "passed": False, "failure": "regression-baseline-gate"}
            write_json(output / "preflight_report.json", report)
            return report
        with patch("diagnostics.general_model_numeric.load_plan", return_value=(plan, [])), \
                patch("diagnostics.general_model_numeric.LocalRunner", return_value=model_runner), \
                patch("diagnostics.general_model_numeric.preflight", side_effect=failed_preflight), \
                patch("diagnostics.general_model_numeric.score_pass") as full_pass, \
                patch("diagnostics.general_model_numeric.analyze_run") as analysis, redirect_stdout(io.StringIO()):
            result = run_pipeline(self.root / "plan", self.root / "run")
        self.assertEqual(result["status"], "preflight_failed")
        self.assertFalse(result["full_dev_started"])
        self.assertFalse(result["analysis_published"])
        full_pass.assert_not_called()
        analysis.assert_not_called()

    def test_terminal_complete_status_without_artifacts_is_not_trusted(self):
        output = self.root / "run"
        output.mkdir()
        write_json(output / "run_manifest.json", {"status": "complete"})
        with patch("diagnostics.general_model_numeric.load_plan", return_value=(plan_fixture(), [])), \
                patch("diagnostics.general_model_numeric.LocalRunner") as model_loader:
            with self.assertRaises((PackageError, FileNotFoundError, KeyError)):
                run_pipeline(self.root / "plan", output)
        model_loader.assert_not_called()

    def test_plan_hash_and_ref_tampering_fail_before_model_loading(self):
        directory = self.root / "plan"
        directory.mkdir()
        write_json(directory / "plan.json", {"plan_id": "wrong", "value": 1})
        with self.assertRaisesRegex(PackageError, "identity"):
            load_plan(directory)
        ref = self.root / "plan_ref.json"
        write_json(ref, {"target_path": str(directory), "plan_sha256": "wrong"})
        with self.assertRaisesRegex(PackageError, "ref"):
            load_plan(ref)


if __name__ == "__main__":
    unittest.main()
