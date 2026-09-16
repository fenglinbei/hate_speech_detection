import copy
import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from diagnostics import general_model_numeric_v2 as numeric
from diagnostics.general_model_package import PackageError, read_json, write_json
from tests.test_general_model_numeric import (
    block_fixture, catalog_fixture, context, full_preflight_contexts,
    plan_fixture, score_fixture,
)


def amended_config(original):
    amended = copy.deepcopy(original)
    amended["schema_version"] = "general-model-ld-numeric-config/v2"
    amended["runtime"].update(
        dtype="float32", head_float32=True, bf16_reduced_precision_reduction=False,
        padding_policy="dynamic", score_mode="full-sequence-teacher-forcing-selected-output-projection",
        projection="all-answer-and-eos-prediction-positions-only",
        parameter_conversion="verified-bfloat16-checkpoint-values-exactly-widened-to-float32")
    amended["validation"].update(padding_challenge_extra=64,
        reference="cpu-float64-from-identical-float32-forward-logits",
        prefix_reference="all-candidates-all-answer-and-eos-positions-unique-prefixes-uncached-unpadded-batch-one",
        member_challenge="rotate-candidate-order-one-place-then-reverse-before-batching-restore-canonical-order-before-commit")
    return amended


class AmendedScorePassTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = plan_fixture()
        self.runner = SimpleNamespace(identity={"model": "fixture"}, padding_extra=0)
        self.contexts = [context("1"), context("2"), context("1", task="group")]
        self.calls = []

    def scorer(self, runner, items, *, reference=False):
        self.calls.append([item["candidate"]["ordinal"] for item in items])
        return [score_fixture(item["candidate"], reference=reference) for item in items]

    def execute(self, **options):
        with redirect_stdout(io.StringIO()):
            return numeric.score_pass(self.runner, self.contexts, self.plan, self.root / "pass",
                                      batch_size=4, scorer=self.scorer, **options)

    def test_permutation_changes_group_members_but_seals_canonical_order(self):
        rows, receipt = self.execute(permuted=True)
        group = rows[-1]["candidates"]
        self.assertEqual([row["ordinal"] for row in group], list(range(32)))
        self.assertEqual(self.calls[1], [0, 31, 30, 29])
        self.assertEqual(receipt["identity"]["scoring_profile"]["candidate_permutation"], "rotate-one-then-reverse")
        self.calls.clear()
        resumed, resumed_receipt = self.execute(permuted=True)
        self.assertEqual(resumed, rows)
        self.assertEqual(resumed_receipt, receipt)
        self.assertEqual(self.calls, [])

    def test_padding_geometry_is_in_checkpoint_identity_and_is_restored(self):
        self.execute(padding_extra=64)
        self.assertEqual(self.runner.padding_extra, 0)
        with self.assertRaisesRegex(PackageError, "different runtime"):
            self.execute(padding_extra=0)

    def test_failed_group_never_commits_partial_candidates_and_restores_padding(self):
        def fail(runner, items, *, reference=False):
            if len(self.calls) == 2:
                raise RuntimeError("interrupted")
            return self.scorer(runner, items, reference=reference)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "interrupted"):
            numeric.score_pass(self.runner, self.contexts, self.plan, self.root / "pass", batch_size=4,
                               scorer=fail, padding_extra=64)
        self.assertEqual(self.runner.padding_extra, 0)
        with sqlite3.connect(self.root / "pass/checkpoint.sqlite3") as connection:
            self.assertEqual({row[0] for row in connection.execute("SELECT key FROM blocks")},
                             {"1:hate:C0", "2:hate:C0"})
        self.assertFalse((self.root / "pass/manifest.json").exists())

    def test_prefix_scores_one_complete_block_and_does_not_call_full_scorer(self):
        calls = []
        def prefix(runner, row, candidates, *, reference=False):
            calls.append(row["record_id"])
            return [score_fixture(candidate, reference=reference) for candidate in candidates]
        with redirect_stdout(io.StringIO()):
            rows, receipt = numeric.score_pass(self.runner, self.contexts, self.plan, self.root / "prefix",
                batch_size=1, prefix=True, scorer=self.scorer, prefix_scorer=prefix)
        self.assertEqual(calls, [r["record_id"] for r in self.contexts])
        self.assertEqual(self.calls, [])
        self.assertEqual(receipt["candidates"], 36)
        self.assertTrue(all(candidate["prefix_reference"] for row in rows for candidate in row["candidates"]))


class AmendedPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = plan_fixture()
        self.plan["config"]["validation"]["padding_challenge_extra"] = 64
        self.contexts = full_preflight_contexts()
        self.runner = SimpleNamespace(identity={"model": "fixture"})
        self.shifts = {"regression": {}, "validation": {}}
        self.calls = []
        patcher = patch.object(numeric, "score_pass", side_effect=self.fake_pass)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_pass(self, runner, contexts, plan, output, *, batch_size, reference=False, **options):
        self.calls.append(output.name)
        cohort = output.name.split("-", 1)[0]
        suffix = output.name[len(cohort) + 1:]
        delta = self.shifts[cohort].get(suffix, 0.0)
        return [block_fixture(row, plan["catalog"], shift=delta, reference=reference) for row in contexts], {}

    def execute(self):
        with redirect_stdout(io.StringIO()):
            return numeric.preflight(self.runner, self.plan, self.contexts, self.root)

    def test_all_five_challenges_contribute_and_validation_does_not_recalibrate(self):
        self.shifts["regression"]["b1-prefix"] = 0.0001
        self.shifts["validation"]["b4-padding"] = 0.00019
        report = self.execute()
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["E8"], 0.0001)
        self.assertAlmostEqual(report["epsilon"], 0.0002)
        self.assertEqual(set(report["cohorts"]["validation"]["challenges"]), set(numeric.CHALLENGES))
        self.assertEqual(len(self.calls), 14)

    def test_shape_failure_blocks_validation_even_when_batch_check_passes(self):
        self.shifts["regression"]["b4-padding"] = 0.006
        report = self.execute()
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "regression-padding-compatibility-gate")
        self.assertFalse(report["validation_executed"])
        self.assertIsNone(report["epsilon"])
        self.assertNotIn("regression-b1-prefix", self.calls)

    def test_prefix_failure_blocks_development(self):
        self.shifts["regression"]["b1-prefix"] = 0.006
        report = self.execute()
        self.assertEqual(report["failure"], "regression-prefix-compatibility-gate")
        self.assertFalse(report["validation_executed"])
        self.assertNotIn("regression-b4-members", self.calls)

    def test_validation_must_meet_frozen_epsilon_not_calibration_ceiling(self):
        self.shifts["regression"]["b4-r0"] = 0.0001
        self.shifts["validation"]["b1-prefix"] = 0.0003
        report = self.execute()
        self.assertAlmostEqual(report["epsilon"], 0.0002)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "validation-prefix-compatibility-gate")

    def test_repeat_failure_stops_before_geometry_tests(self):
        self.shifts["regression"]["b1-r1"] = 0.0002
        report = self.execute()
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "regression-baseline-gate")
        self.assertEqual(self.calls, ["regression-b1-r0", "regression-b1-r1"])

    def test_duplicate_cohort_cannot_pass_coverage(self):
        self.plan["cohorts"]["validation"][0] = self.plan["cohorts"]["regression"][0]
        with self.assertRaisesRegex(PackageError, "overlapping"):
            self.execute()


class AmendmentScopeTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.original = read_json(root / "config/stage1/general_model_ld_numeric_v1.json")
        self.config = amended_config(self.original)

    def test_runtime_precision_amendment_preserves_scientific_registration(self):
        self.config["runtime"].update(dtype="float32", head_float32=True, bf16_reduced_precision_reduction=False)
        numeric.validate_config(self.config, self.original)

    def test_scientific_changes_and_relaxed_thresholds_are_rejected(self):
        changes = [(["analysis", "query_weighting"], "nonuniform"),
                   (["validation", "calibration_max_abs_error"], 0.01),
                   (["test_access"], True), (["runtime", "main_score"], "mean")]
        for path, value in changes:
            amended = copy.deepcopy(self.config)
            target = amended
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(PackageError):
                numeric.validate_config(amended, self.original)


class AmendmentPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        project = Path(__file__).resolve().parents[2]
        original = read_json(project / "config/stage1/general_model_ld_numeric_v1.json")
        self.parent_plan = {"plan_id": "parent-fixture", "config": original}
        config = amended_config(original)
        config.update(schema_version="general-model-ld-numeric-config/v2", parent_plan_ref="parent-ref.json",
                      output_root="numeric-v2", protocol_path="protocol.md")
        config["validation"]["padding_challenge_extra"] = 64
        write_json(self.root / "config.json", config)
        write_json(self.root / "parent-ref.json", {"parent": "fixture"})
        (self.root / "protocol.md").write_text("fixture registration\n")
        for name in numeric.CODE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture source\n")
        patcher = patch.object(numeric.parent, "load_plan", return_value=(self.parent_plan, []))
        patcher.start()
        self.addCleanup(patcher.stop)
        with redirect_stdout(io.StringIO()):
            self.reference = numeric.build_plan(self.root / "config.json", root=self.root)
        self.ref_path = self.root / "numeric-v2/plan_ref.json"

    def test_plan_binds_parent_and_new_sources_and_can_be_loaded(self):
        effective, contexts = numeric.load_plan(self.ref_path, root=self.root)
        self.assertEqual(effective["parent_plan"], self.parent_plan)
        self.assertEqual(effective["parent_plan_id"], "parent-fixture")
        self.assertEqual(contexts, [])
        self.assertEqual(effective["plan_id"], self.reference["plan_id"])

    def test_current_source_change_is_rejected_even_with_unchanged_snapshot(self):
        (self.root / numeric.CODE_FILES[0]).write_text("# changed source\n")
        with self.assertRaisesRegex(PackageError, "source differs"):
            numeric.load_plan(self.ref_path, root=self.root)

    def test_parent_reference_change_is_rejected(self):
        write_json(self.root / "parent-ref.json", {"parent": "changed"})
        with self.assertRaisesRegex(PackageError, "parent plan reference"):
            numeric.load_plan(self.ref_path, root=self.root)


class AmendedPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = plan_fixture()
        self.plan["parent_plan"] = {"plan_id": "parent"}
        self.runner = SimpleNamespace(identity={"model": "fixture"}, model=object(), torch=MagicMock())
        self.loader = patch.object(numeric, "load_plan", return_value=(self.plan, [])).start()
        self.constructor = patch("diagnostics.general_model_numeric_kernel_v2.NumericRunner", return_value=self.runner).start()
        self.addCleanup(patch.stopall)

    def test_failed_preflight_stops_before_development(self):
        def failed(runner, plan, contexts, output):
            output.mkdir(parents=True)
            report = {"passed": False, "plan_id": plan["plan_id"], "files": {}, "failure": "shape-gate"}
            write_json(output / "preflight_report.json", report)
            return report
        with patch.object(numeric, "preflight", side_effect=failed), patch.object(numeric, "score_pass") as score:
            result = numeric.run_pipeline(self.root / "plan.json", self.root / "run")
        self.assertEqual(result["status"], "preflight_failed")
        self.assertFalse(result["full_dev_started"])
        score.assert_not_called()
        self.assertFalse(hasattr(self.runner, "model"))
        self.runner.torch.cuda.empty_cache.assert_called_once()

    def test_terminal_verification_failure_does_not_overwrite_existing_receipt(self):
        output = self.root / "run"
        output.mkdir()
        prior = {"status": "complete", "plan_id": self.plan["plan_id"], "preflight_report_sha256": "fixture"}
        write_json(output / "run_manifest.json", prior)
        before = (output / "run_manifest.json").read_bytes()
        with patch.object(numeric, "_verified_preflight", side_effect=PackageError("tampered")):
            with self.assertRaisesRegex(PackageError, "tampered"):
                numeric.run_pipeline(self.root / "plan.json", output)
        self.assertEqual((output / "run_manifest.json").read_bytes(), before)
        self.constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
