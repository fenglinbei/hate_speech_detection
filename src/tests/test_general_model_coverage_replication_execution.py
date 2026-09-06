"""Synthetic CPU-only checks of replication seals, gates, and restart behavior."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_coverage_replication_execution as execution
from diagnostics import general_model_numeric_sharded as sharded
from diagnostics.general_model_package import PackageError, read_json, write_json, write_jsonl
from tests.test_general_model_coverage_execution import coverage_plan
from tests.test_general_model_numeric_v3 import rows_fixture


def plan_fixture():
    plan = coverage_plan()
    plan["model"] = {"key": "qwen3-14b"}
    return plan


def runtime_fixture(shift=0):
    mapping = {"model.layers.0": shift * 2, "model.layers.1": shift * 2 + 1}
    return {"device_map": mapping, "device_map_sha256": canonical_json_sha256(mapping),
            "hardware": [{"physical_gpu_index": index, "uuid": f"GPU-fixture-{index}"}
                         for index in sorted(set(mapping.values()))], "replica_shift": shift}


def write_pass(plan, contexts, path, runtime, *, shift=0.0, **options):
    rows = rows_fixture(contexts, plan, shift=shift, **options)
    for row in rows:
        row.update(runtime_sha256=canonical_json_sha256(runtime), pass_name=path.name,
                   cohort=next((name for name in ("regression", "validation")
                                if path.name.startswith(name)), "dev"))
        for candidate in row["candidates"]:
            candidate.update(physical_gpu_indices=sorted(set(runtime["device_map"].values())),
                             physical_gpu_uuids=[item["uuid"] for item in runtime["hardware"]],
                             model_device_map_sha256=runtime["device_map_sha256"],
                             replica_shift=options.get("replica_shift", 0),
                             fp32_operator_dispatch_checked=True, fp32_operator_count=10,
                             peak_memory_by_device={str(index): {"allocated_bytes": 128, "reserved_bytes": 256}
                                                    for index in set(runtime["device_map"].values())})
    path.mkdir(parents=True, exist_ok=True)
    write_jsonl(path / "scores.jsonl", rows)
    receipt = {"schema_version": "general-model-ld-numeric-pass/v2", "status": "complete",
               "identity": {"plan_id": plan["plan_id"], "runtime": runtime, "batch_size": 1,
                            "reference": options.get("reference", False), "pass_name": path.name,
                            "records": [row["record_id"] for row in contexts],
                            "scoring_profile": execution.scoring.scoring_profile(
                                **{key: value for key, value in options.items() if key != "reference"})},
               "blocks": len(rows), "candidates": sum(len(row["candidates"]) for row in rows),
               "scores_sha256": sha256_file(path / "scores.jsonl"),
               "query_gold_loaded": False, "test_content_read": False, "mixed_execution_modes": False}
    write_json(path / "manifest.json", receipt)
    return rows, receipt


class PassBindingTests(unittest.TestCase):
    def test_all_registered_profiles_retain_candidate_device_evidence(self):
        plan = plan_fixture()
        for _, options, label in execution.PASSES:
            with self.subTest(profile=label), tempfile.TemporaryDirectory() as tmp:
                runtime = runtime_fixture(options.get("replica_shift", 0))
                contexts, path = plan["blocks"][:2], Path(tmp) / "dev-b1"
                write_pass(plan, contexts, path, runtime, **options)
                with patch.object(sharded, "validate_runtime_identity") as validate:
                    rows, _, proof = execution.read_pass(plan, path, contexts, options, runtime)
                self.assertTrue(proof["passed"])
                self.assertEqual(len(rows), 2)
                validate.assert_called_once_with(plan, runtime, replica_shift=options.get("replica_shift", 0))

    def test_resealed_wrong_device_dispatch_or_memory_evidence_is_rejected(self):
        changes = (("physical_gpu_indices", [0]), ("physical_gpu_uuids", ["GPU-other", "GPU-fixture-1"]),
                   ("model_device_map_sha256", "other"), ("replica_shift", 1),
                   ("fp32_operator_dispatch_checked", False), ("fp32_operator_count", 0),
                   ("fp32_operator_count", True), ("peak_memory_by_device", {}),
                   ("peak_memory_by_device", {"0": {"allocated_bytes": -1, "reserved_bytes": 256},
                                               "1": {"allocated_bytes": 128, "reserved_bytes": 256}}))
        for key, value in changes:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as tmp:
                plan, runtime = plan_fixture(), runtime_fixture()
                contexts, path = plan["blocks"][:1], Path(tmp) / "dev-b1"
                rows, receipt = write_pass(plan, contexts, path, runtime)
                rows[0]["candidates"][0][key] = value
                write_jsonl(path / "scores.jsonl", rows)
                receipt["scores_sha256"] = sha256_file(path / "scores.jsonl")
                write_json(path / "manifest.json", receipt)
                with patch.object(sharded, "validate_runtime_identity"), self.assertRaises(PackageError):
                    execution.read_pass(plan, path, contexts, {}, runtime)

    def test_runtime_changed_at_resume_is_rejected_before_pass_reuse(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(sharded, "validate_runtime_identity"):
            root, plan = Path(tmp), plan_fixture()
            execution.save_runtime(plan, root, SimpleNamespace(identity=runtime_fixture()), 0)
            changed = runtime_fixture()
            changed["hardware"][0]["uuid"] = "GPU-other"
            with self.assertRaisesRegex(PackageError, "resumed sharded runtime"):
                execution.save_runtime(plan, root, SimpleNamespace(identity=changed), 0)

    def test_resealed_wrong_pass_identity_or_score_attribution_is_rejected(self):
        for mutation in ("manifest-runtime", "block-runtime", "block-pass", "block-cohort"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                plan, runtime = plan_fixture(), runtime_fixture()
                contexts, path = plan["blocks"][:1], Path(tmp) / "dev-b1"
                rows, receipt = write_pass(plan, contexts, path, runtime)
                if mutation == "manifest-runtime":
                    receipt["identity"]["runtime"] = {"other": True}
                else:
                    key = {"block-runtime": "runtime_sha256", "block-pass": "pass_name",
                           "block-cohort": "cohort"}[mutation]
                    rows[0][key] = "other"
                    write_jsonl(path / "scores.jsonl", rows)
                    receipt["scores_sha256"] = sha256_file(path / "scores.jsonl")
                write_json(path / "manifest.json", receipt)
                with patch.object(sharded, "validate_runtime_identity"), self.assertRaises(PackageError):
                    execution.read_pass(plan, path, contexts, {}, runtime)


class PreflightTests(unittest.TestCase):
    def run_fixture(self, root, *, failure=None, empty_boundary=False):
        plan, calls, closed = plan_fixture(), [], []
        if empty_boundary:
            plan["cohorts"]["boundary"] = []

        def factory(actual, replica_shift):
            return SimpleNamespace(identity=runtime_fixture(replica_shift),
                                   close=lambda: closed.append(replica_shift))

        def scorer(runner, contexts, actual, path, *, batch_size, scorer, prefix_scorer, **options):
            self.assertEqual(batch_size, 1)
            calls.append(path.name)
            return write_pass(actual, contexts, path, runner.identity,
                              shift=0.002 if path.name == failure else 0.0, **options)

        with patch.object(sharded, "validate_runtime_identity"), \
                patch.object(execution.scoring, "score_pass", side_effect=scorer), redirect_stdout(io.StringIO()):
            report = execution.run_preflight(plan, plan["blocks"], root, factory)
        return plan, report, calls, closed

    def verify(self, root, plan):
        with patch.object(sharded, "validate_runtime_identity"):
            return execution.verify_preflight(plan, plan["blocks"], root)

    def reseal(self, root, report):
        report["files"] = execution._evidence(root / "preflight")
        write_json(root / "preflight/preflight_report.json", report)

    def test_all_eighteen_passes_are_recomputed_in_two_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, calls, closed = self.run_fixture(root)
            self.assertTrue(report["passed"])
            self.assertTrue(report["complete"])
            self.assertEqual(len(calls), 18)
            self.assertEqual(closed, [0, 1])
            self.assertEqual(calls[-3:], ["regression-b1-replica", "validation-b1-replica", "boundary-b1-replica"])
            self.assertEqual(report, self.verify(root, plan))
            never_load = Mock(side_effect=AssertionError("must not load a model"))
            with patch.object(sharded, "validate_runtime_identity"):
                self.assertEqual(report, execution.run_preflight(plan, plan["blocks"], root, never_load))
            never_load.assert_not_called()

    def test_early_failure_is_sealed_prefix_and_last_failure_is_complete(self):
        for failure, count, complete in (("regression-b1-padding", 3, False),
                                         ("boundary-b1-replica", 18, True)):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan, report, calls, _ = self.run_fixture(root, failure=failure)
                self.assertFalse(report["passed"])
                self.assertIs(report["complete"], complete)
                self.assertEqual(len(calls), count)
                self.assertEqual(report, self.verify(root, plan))

    def test_failure_cannot_hide_a_skipped_earlier_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, _, _ = self.run_fixture(root, failure="regression-b1-padding")
            del report["checks"]["regression-repeat"]
            self.reseal(root, report)
            with self.assertRaisesRegex(PackageError, "execution prefix"):
                self.verify(root, plan)

    def test_checks_after_first_failure_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, _, _ = self.run_fixture(root)
            path = root / "preflight/regression-b1-padding"
            options = {"padding_extra": 64}
            rows, _ = write_pass(plan, execution.selected(plan, plan["blocks"], "regression"), path,
                                 runtime_fixture(), shift=0.002, **options)
            baseline = execution.read_jsonl(root / "preflight/regression-b1-r0/scores.jsonl")
            comparison = execution.base.compare_passes(baseline, rows)
            report["checks"]["regression-padding"].update(max_abs_error=comparison["max_abs_error"], passed=False)
            report.update(passed=False, failure="regression-padding")
            write_json(root / "preflight/regression-padding-differences.json", comparison)
            self.reseal(root, report)
            with self.assertRaisesRegex(PackageError, "pass/failure claim"):
                self.verify(root, plan)

    def test_empty_boundary_is_twelve_passes_not_eighteen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, calls, _ = self.run_fixture(root, empty_boundary=True)
            self.assertTrue(report["passed"])
            self.assertEqual(len(calls), 12)
            self.assertEqual(report, self.verify(root, plan))

    def test_extra_unchecked_pass_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, _, _ = self.run_fixture(root)
            path = root / "preflight/unregistered"
            path.mkdir()
            write_json(path / "manifest.json", {"unexpected": True})
            self.reseal(root, report)
            with self.assertRaisesRegex(PackageError, "unregistered pass evidence"):
                self.verify(root, plan)

    def test_physical_uuid_aliases_cannot_claim_a_replica_challenge(self):
        baseline, replica = runtime_fixture(), runtime_fixture(1)
        execution._verify_remapping(baseline, replica)
        replica["hardware"][0]["uuid"] = baseline["hardware"][0]["uuid"]
        with self.assertRaisesRegex(PackageError, "physical GPU"):
            execution._verify_remapping(baseline, replica)

    def test_same_physical_index_cannot_change_uuid_between_snapshots(self):
        baseline = runtime_fixture()
        replica = {"device_map": {"model.layers.0": 1, "model.layers.1": 0},
                   "hardware": [{"physical_gpu_index": 0, "uuid": "GPU-other-0"},
                                {"physical_gpu_index": 1, "uuid": "GPU-other-1"}]}
        with self.assertRaisesRegex(PackageError, "physical GPU"):
            execution._verify_remapping(baseline, replica)


class RestartAndGoldGateTests(unittest.TestCase):
    def prepare(self, root, status):
        plan = plan_fixture()
        output = root / plan["config"]["output_root"] / "runs/test-run"
        (output / "preflight").mkdir(parents=True)
        write_json(output / "preflight/preflight_report.json", {"synthetic": True})
        run = {"schema_version": execution.SCHEMA, "plan_id": plan["plan_id"],
               "model_key": plan["model"]["key"], "status": status, "execution": "model-parallel-fp32",
               "query_gold_loaded_during_scoring": False, "test_content_read": False,
               "automatic_profile_search": False, "full_dev_started": status in {"raw_complete", "complete"},
               "analysis_published": status == "complete",
               "preflight_report_sha256": sha256_file(output / "preflight/preflight_report.json")}
        if status in {"raw_complete", "complete"}:
            (output / "dev-b1").mkdir()
            write_json(output / "dev-b1/manifest.json", {"synthetic": True})
            run.update(raw_blocks=len(plan["blocks"]), raw_manifest_sha256=sha256_file(output / "dev-b1/manifest.json"))
            write_json(execution.runtime_path(output, 0), runtime_fixture())
        if status == "complete":
            (output / "analysis").mkdir()
            write_json(output / "analysis/manifest.json", {"synthetic": True})
            run["analysis_manifest_sha256"] = sha256_file(output / "analysis/manifest.json")
        write_json(output / "run_manifest.json", run)
        return plan, output, run

    def test_complete_reentry_does_not_load_gpu_or_mutate_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, output, run = self.prepare(root, "complete")
            before = sha256_file(output / "run_manifest.json")
            never_load = Mock(side_effect=AssertionError("must not load a model"))
            with patch.object(execution, "ROOT", root), patch.object(execution, "analyze", return_value={}) as analyze:
                self.assertEqual(run, execution.execute(plan, plan["blocks"], output, phase="dev", runner_factory=never_load))
            analyze.assert_called_once()
            never_load.assert_not_called()
            self.assertEqual(before, sha256_file(output / "run_manifest.json"))

    def test_failed_states_never_retry_or_open_gold(self):
        for status in ("failed", "preflight_failed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan, output, _ = self.prepare(root, status)
                never_load = Mock()
                with patch.object(execution, "ROOT", root), patch.object(execution, "read_jsonl") as gold, \
                        self.assertRaisesRegex(PackageError, "failed replication is sealed"):
                    execution.execute(plan, plan["blocks"], output, phase="dev", runner_factory=never_load)
                never_load.assert_not_called()
                gold.assert_not_called()

    def test_premature_analyze_preserves_valid_preflight_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, output, _ = self.prepare(root, "preflight_passed")
            before = sha256_file(output / "run_manifest.json")
            never_load = Mock()
            with patch.object(execution, "ROOT", root), \
                    patch.object(execution, "verify_preflight", return_value={"passed": True}), \
                    self.assertRaisesRegex(PackageError, "analysis cannot initiate"):
                execution.execute(plan, plan["blocks"], output, phase="analyze", runner_factory=never_load)
            self.assertEqual(before, sha256_file(output / "run_manifest.json"))
            never_load.assert_not_called()

    def test_raw_complete_resumes_analysis_without_gpu_and_preserves_raw_on_error(self):
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan, output, _ = self.prepare(root, "raw_complete")
                never_load = Mock(side_effect=AssertionError("must not load a model"))

                def analyze(*args):
                    if fails:
                        raise RuntimeError("synthetic analysis interruption")
                    (output / "analysis").mkdir()
                    write_json(output / "analysis/manifest.json", {"synthetic": True})
                    return {}

                with patch.object(execution, "ROOT", root), \
                        patch.object(execution, "verify_preflight", return_value={"passed": True}), \
                        patch.object(execution, "analyze", side_effect=analyze):
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "analysis interruption"):
                            execution.execute(plan, plan["blocks"], output, phase="analyze", runner_factory=never_load)
                    else:
                        self.assertEqual(execution.execute(plan, plan["blocks"], output, phase="dev",
                                                           runner_factory=never_load)["status"], "complete")
                never_load.assert_not_called()
                self.assertEqual(read_json(output / "run_manifest.json")["status"], "raw_complete" if fails else "complete")

    def test_keyboard_interrupt_remains_resumable_not_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, plan = Path(tmp), plan_fixture()
            output = root / plan["config"]["output_root"] / "runs/test-run"
            never_load = Mock()
            with patch.object(execution, "ROOT", root), \
                    patch.object(execution, "run_preflight", side_effect=KeyboardInterrupt), \
                    self.assertRaises(KeyboardInterrupt):
                execution.execute(plan, plan["blocks"], output, phase="preflight", runner_factory=never_load)
            self.assertEqual(read_json(output / "run_manifest.json")["status"], "interrupted")
            never_load.assert_not_called()

    def test_contradictory_terminal_flags_are_rejected_before_gpu_or_analysis(self):
        for key, value in (("analysis_published", False), ("full_dev_started", False),
                           ("raw_blocks", 1), ("execution", "data-parallel")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan, output, run = self.prepare(root, "complete")
                run[key] = value
                write_json(output / "run_manifest.json", run)
                never_load = Mock()
                with patch.object(execution, "ROOT", root), patch.object(execution, "analyze") as analyze, \
                        self.assertRaises(PackageError):
                    execution.execute(plan, plan["blocks"], output, phase="dev", runner_factory=never_load)
                analyze.assert_not_called()
                never_load.assert_not_called()

    def test_wrong_preflight_or_raw_hash_blocks_query_gold_deserialization(self):
        for mutation in ("preflight-failed", "preflight-hash", "raw-hash"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                plan, output, run = self.prepare(Path(tmp), "raw_complete")
                if mutation == "preflight-hash":
                    run["preflight_report_sha256"] = "other"
                if mutation == "raw-hash":
                    run["raw_manifest_sha256"] = "other"
                with patch.object(execution, "verify_preflight", return_value={"passed": mutation != "preflight-failed"}), \
                        patch.object(execution, "read_jsonl") as gold, self.assertRaises(PackageError):
                    execution.analyze(plan, plan["blocks"], output, run)
                gold.assert_not_called()

    def test_unsealed_status_cannot_enter_gold_analysis(self):
        for status in ("running", "interrupted", "preflight_passed", "preflight_failed", "failed"):
            with self.subTest(status=status), patch.object(execution, "read_jsonl") as gold, self.assertRaises(PackageError):
                execution.analyze(plan_fixture(), [], Path("/synthetic-unread"), {"status": status})
            gold.assert_not_called()

    def test_existing_analysis_model_or_geometry_mutation_is_rejected_without_gold(self):
        for mutation in ("model_key", "production_geometry", "schema_version"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                plan, output, run = self.prepare(Path(tmp), "complete")
                write_json(output / "analysis/analysis.json", {"synthetic": True})
                receipt = {"schema_version": "general-model-coverage-replication-analysis/v1",
                           "model_key": plan["model"]["key"], "production_geometry": {"passed": True},
                           "plan_id": plan["plan_id"], "raw_manifest_sha256": run["raw_manifest_sha256"],
                           "analysis_sha256": sha256_file(output / "analysis/analysis.json"),
                           "gold_join_after_raw_sealed": True, "test_content_read": False}
                receipt[mutation] = "other"
                write_json(output / "analysis/manifest.json", receipt)
                with patch.object(execution, "verify_preflight", return_value={"passed": True}), \
                        patch.object(execution, "read_pass", return_value=([], {}, {"passed": True})), \
                        patch.object(execution, "read_jsonl") as gold, \
                        self.assertRaisesRegex(PackageError, "existing analysis"):
                    execution.analyze(plan, plan["blocks"], output, run)
                gold.assert_not_called()

    def test_existing_valid_analysis_is_reused_without_query_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, output, run = self.prepare(Path(tmp), "complete")
            write_json(output / "analysis/analysis.json", {"synthetic": True})
            receipt = {"schema_version": "general-model-coverage-replication-analysis/v1",
                       "model_key": plan["model"]["key"], "production_geometry": {"passed": True},
                       "plan_id": plan["plan_id"], "raw_manifest_sha256": run["raw_manifest_sha256"],
                       "analysis_sha256": sha256_file(output / "analysis/analysis.json"),
                       "gold_join_after_raw_sealed": True, "test_content_read": False}
            write_json(output / "analysis/manifest.json", receipt)
            with patch.object(execution, "verify_preflight", return_value={"passed": True}), \
                    patch.object(execution, "read_pass", return_value=([], {}, {"passed": True})), \
                    patch.object(execution, "read_jsonl") as gold:
                self.assertEqual(receipt, execution.analyze(plan, plan["blocks"], output, run))
            gold.assert_not_called()

    def test_completed_missing_analysis_does_not_rejoin_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, output, run = self.prepare(Path(tmp), "raw_complete")
            run["status"] = "complete"
            with patch.object(execution, "verify_preflight", return_value={"passed": True}), \
                    patch.object(execution, "read_pass", return_value=([], {}, {"passed": True})), \
                    patch.object(execution, "read_jsonl") as gold, \
                    self.assertRaisesRegex(PackageError, "lost its analysis"):
                execution.analyze(plan, plan["blocks"], output, run)
            gold.assert_not_called()

    def test_unbound_replica_runtime_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, plan = Path(tmp), plan_fixture()
            output = root / plan["config"]["output_root"] / "runs/test-run"
            output.mkdir(parents=True)
            write_json(execution.runtime_path(output, 1), runtime_fixture(1))
            never_load = Mock()
            with patch.object(execution, "ROOT", root), self.assertRaisesRegex(PackageError, "unbound"):
                execution.execute(plan, plan["blocks"], output, phase="preflight", runner_factory=never_load)
            never_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
