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
from data.stage1_data import sha256_file
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
        member_challenge="group-rotate-one-then-reverse-hate-reverse-before-batching-restore-canonical-order-before-commit",
        replica_challenge_shift=1,
        replica_comparison="batch-one-and-identical-batch-four-on-another-physical-device")
    amended["execution"]["device_indices"] = [0, 1, 2, 3]
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
        self.assertEqual(receipt["identity"]["scoring_profile"]["candidate_permutation"], "group-rotate-one-then-reverse-hate-reverse")
        self.calls.clear()
        resumed, resumed_receipt = self.execute(permuted=True)
        self.assertEqual(resumed, rows)
        self.assertEqual(resumed_receipt, receipt)
        self.assertEqual(self.calls, [])

    def test_hate_permutation_actually_swaps_candidates_and_restores_canonical_output(self):
        rows, _ = self.execute(permuted=True)
        self.assertEqual(self.calls[0], [1, 0, 1, 0])
        for row in rows[:2]:
            self.assertEqual([c["ordinal"] for c in row["candidates"]], [0, 1])
        self.assertEqual([c["batch_member_ordinal"] for c in rows[0]["candidates"]], [1, 0])

    def test_replica_shift_is_bound_to_checkpoint_identity(self):
        self.execute(replica_shift=1)
        with self.assertRaisesRegex(PackageError, "different runtime"):
            self.execute(replica_shift=0)

    def test_pool_receives_exact_scoring_profile_without_local_forward(self):
        pool = SimpleNamespace(is_numeric_pool=True, score_pass=MagicMock(return_value=([], {"fixture": True})))
        rows, receipt = numeric.score_pass(pool, self.contexts, self.plan, self.root / "pool", batch_size=4,
                                          padding_extra=64, permuted=True, replica_shift=1)
        self.assertEqual((rows, receipt), ([], {"fixture": True}))
        pool.score_pass.assert_called_once_with(self.contexts, self.plan, self.root / "pool", batch_size=4,
            reference=False, padding_extra=64, prefix=False, permuted=True, replica_shift=1)

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
        self.plan["config"]["validation"]["replica_challenge_shift"] = 1
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
        rows = [block_fixture(row, plan["catalog"], shift=delta, reference=reference) for row in contexts]
        for row in rows:
            for candidate in row["candidates"]:
                candidate.update(batch_members=[candidate["candidate_id"]], batch_member_ordinal=0,
                    batch_size=1, effective_batch_size=1, padded_sequence_tokens=8, sequence_tokens=5,
                    prompt_token_ids_sha256="fixture", causal_shift=1, use_cache=False, padding_side="right",
                    physical_gpu_index=1 if suffix == "b4-replica" else 0,
                    physical_gpu_uuid="GPU-one" if suffix == "b4-replica" else "GPU-zero")
        return rows, {}

    def execute(self):
        with redirect_stdout(io.StringIO()):
            return numeric.preflight(self.runner, self.plan, self.contexts, self.root)

    def test_all_six_challenges_contribute_and_validation_does_not_recalibrate(self):
        self.shifts["regression"]["b1-prefix"] = 0.0001
        self.shifts["validation"]["b4-padding"] = 0.00019
        report = self.execute()
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["E8"], 0.0001)
        self.assertAlmostEqual(report["epsilon"], 0.0002)
        self.assertEqual(set(report["cohorts"]["validation"]["challenges"]), set(numeric.CHALLENGES))
        self.assertEqual(len(self.calls), 16)

    def test_replica_error_sets_calibration_and_is_checked_on_validation(self):
        self.shifts["regression"]["b4-replica"] = 0.0002
        self.shifts["validation"]["b4-replica"] = 0.0005
        report = self.execute()
        self.assertAlmostEqual(report["E8"], 0.0002)
        self.assertAlmostEqual(report["epsilon"], 0.0004)
        self.assertEqual(report["failure"], "validation-replica-compatibility-gate")
        self.assertFalse(report["passed"])

    def test_replica_error_includes_comparison_to_standard_batch_four(self):
        self.shifts["regression"]["b4-r0"] = 0.0002
        self.shifts["regression"]["b4-replica"] = -0.0002
        report = self.execute()
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["E8"], 0.0004)
        self.assertAlmostEqual(report["epsilon"], 0.0008)
        self.assertTrue((self.root / "regression-replica-vs-batch4-differences.json").exists())
        proof = read_json(self.root / "regression-replica-producer-proof.json")
        self.assertTrue(proof["all_candidates_changed_physical_gpu"])
        self.assertIn("regression-replica-producer-proof.json", report["files"])

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


class ReplicaProofTests(unittest.TestCase):
    def setUp(self):
        self.standard = [block_fixture(context(), catalog_fixture())]
        for candidate in self.standard[0]["candidates"]:
            candidate.update(batch_members=["hate-0", "hate-1"], batch_member_ordinal=candidate["ordinal"],
                batch_size=2, effective_batch_size=2, padded_sequence_tokens=8, sequence_tokens=5,
                prompt_token_ids_sha256="fixture", causal_shift=1, use_cache=False, padding_side="right",
                physical_gpu_index=0, physical_gpu_uuid="GPU-zero")
        self.shifted = copy.deepcopy(self.standard)
        for candidate in self.shifted[0]["candidates"]:
            candidate.update(physical_gpu_index=1, physical_gpu_uuid="GPU-one")

    def test_actual_producer_changes_and_identical_geometry_pass(self):
        proof = numeric.replica_proof(self.standard, self.shifted)
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["candidates"], 2)

    def test_metadata_shift_cannot_substitute_for_different_gpu_uuid(self):
        self.shifted[0]["candidates"][0]["physical_gpu_uuid"] = "GPU-zero"
        with self.assertRaisesRegex(PackageError, "actual physical GPU"):
            numeric.replica_proof(self.standard, self.shifted)

    def test_actual_device_change_does_not_allow_changed_padding_or_members(self):
        for key, value in (("padded_sequence_tokens", 9), ("batch_members", ["different"]),
                           ("batch_member_ordinal", 1)):
            shifted = copy.deepcopy(self.shifted)
            shifted[0]["candidates"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(PackageError, "batch geometry"):
                numeric.replica_proof(self.standard, shifted)


class RawIdentityTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"plan_id": "fixture", "blocks": [{"record_id": "1:hate:C0"}]}
        self.report = {"runtime_identity": {"replicas": ["GPU-zero", "GPU-one"]}}
        self.receipt = {"identity": {"plan_id": "fixture", "runtime": self.report["runtime_identity"],
            "batch_size": 4, "reference": False, "pass_name": "dev-b4",
            "scoring_profile": numeric.scoring_profile(), "records": ["1:hate:C0"]},
            "execution": "data-parallel-identical-fp32", "query_gold_loaded": False,
            "test_content_read": False, "mixed_execution_modes": False}

    def test_production_runtime_and_geometry_must_match_passed_preflight(self):
        numeric.validate_raw_identity(self.plan, self.report, self.receipt)
        variants = {"runtime": {"replicas": ["other"]}, "batch_size": 1, "reference": True,
                    "pass_name": "validation-b4-r0", "records": [],
                    "scoring_profile": numeric.scoring_profile(replica_shift=1)}
        for key, value in variants.items():
            receipt = copy.deepcopy(self.receipt)
            receipt["identity"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(PackageError, "validated production"):
                numeric.validate_raw_identity(self.plan, self.report, receipt)

    def test_raw_cannot_claim_gold_or_mixed_execution(self):
        for key in ("query_gold_loaded", "test_content_read", "mixed_execution_modes"):
            receipt = {**self.receipt, key: True}
            with self.subTest(key=key), self.assertRaisesRegex(PackageError, "boundary"):
                numeric.validate_raw_identity(self.plan, self.report, receipt)

    def test_changed_shard_stops_analysis_before_reading_raw_or_gold_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            raw = output / "dev-b4"
            raw.mkdir()
            (raw / "scores.jsonl").write_text("")
            receipt = {**self.receipt, "status": "complete", "blocks": 7716, "candidates": 131172,
                       "scores_sha256": sha256_file(raw / "scores.jsonl")}
            write_json(raw / "manifest.json", receipt)
            write_json(output / "run_manifest.json", {"raw_manifest_sha256": sha256_file(raw / "manifest.json")})
            with patch.object(numeric, "_verified_preflight", return_value={**self.report, "passed": True}), \
                    patch("diagnostics.general_model_numeric_pool.validate_sealed_pass", side_effect=PackageError("changed shard")) as check, \
                    patch.object(numeric, "read_jsonl", side_effect=AssertionError("rows must not be loaded")):
                with self.assertRaisesRegex(PackageError, "changed shard"):
                    numeric.analyze_run(self.plan, output)
            check.assert_called_once_with(raw, self.plan)


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
        self.plan["config"]["execution"] = {"device_indices": [0, 1, 2, 3]}
        self.runner = SimpleNamespace(identity={"model": "fixture"}, close=MagicMock())
        self.loader = patch.object(numeric, "load_plan", return_value=(self.plan, [])).start()
        self.constructor = patch("diagnostics.general_model_numeric_pool.PersistentNumericPool", return_value=self.runner).start()
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
        self.runner.close.assert_called_once()

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

    def test_sealed_raw_analysis_resume_does_not_load_gpu_pool(self):
        output = self.root / "run"
        output.mkdir()
        prior = {"status": "raw_complete", "plan_id": self.plan["plan_id"],
                 "raw_manifest_sha256": "raw-fixture", "preflight_report_sha256": "preflight-fixture"}
        write_json(output / "run_manifest.json", prior)
        with patch.object(numeric, "analyze_run", return_value={"analysis": "fixture"}) as analysis:
            result = numeric.run_pipeline(self.root / "plan.json", output)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["analysis_published"])
        analysis.assert_called_once_with(self.plan, output)
        self.constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
