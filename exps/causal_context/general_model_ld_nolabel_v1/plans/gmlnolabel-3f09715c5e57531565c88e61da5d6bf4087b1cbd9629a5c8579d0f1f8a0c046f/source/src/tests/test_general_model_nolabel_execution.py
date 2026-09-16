import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_nolabel_execution as execution
from diagnostics.general_model_package import PackageError, read_json, write_json
from tests.test_general_model_numeric import context, plan_fixture
from tests.test_general_model_numeric_v3 import rows_fixture


CONDITIONS = ["C0", "CLnew", "CD", "CLDnew", "CLnewNoCat", "CLDnewNoCat"]


def coverage_plan(full=False):
    plan = plan_fixture()
    qids = [str(i) for i in range(1, 644 if full else 4)]
    plan["config"].update(conditions=CONDITIONS, output_root="coverage-output",
        execution={"device_indices": [0, 1, 2, 3]}, analysis={"bootstrap": {"repetitions": 10000, "seed": 42}})
    plan["config"]["runtime"].update(dtype="float32", padding_policy="dynamic", use_cache=False,
        attention_implementation="eager", max_sequence_tokens=8192, enable_thinking=False, seed=42,
        padding_side="right", bf16_reduced_precision_reduction=False)
    plan["numeric_policy"] = copy.deepcopy(execution.POLICY)
    plan["eos_token_id"] = 127
    plan["cohorts"] = ({"regression": [str(i) for i in range(1, 9)],
        "validation": [str(i) for i in range(9, 33)], "boundary": ["33"]} if full else
        {"regression": ["1"], "validation": ["2"], "boundary": ["3"]})
    plan["frame"] = [{"query_id": qid, "lex_hit": False} for qid in qids]
    plan["blocks"] = [context(qid, task, condition) for task in ["hate", "group"]
                       for condition in CONDITIONS for qid in qids]
    plan["runtime_parent_plan"] = {key: plan[key] for key in
        ("package_path", "generation_runtime_identity", "blocks", "catalog")}
    plan["code_sha256"] = {"worker.py": "source-digest"}
    return plan


def runtime_fixture(plan):
    runtime = plan["config"]["runtime"]
    lengths = {}
    for row in plan["blocks"]:
        key = (row["query_id"], row["task"])
        lengths[key] = max(lengths.get(key, 0), row["prompt_tokens"] + max(
            candidate["answer_tokens"] for candidate in plan["catalog"][row["task"]]) + 1)
    numeric = {
        "source_generation_runtime": plan["generation_runtime_identity"], "numeric_runtime": runtime,
        "transformer_dtype": "torch.float32", "lm_head_dtype": "torch.float32",
        "tf32_matmul": False, "tf32_cudnn": False, "use_cache": False,
        "projection": "answer-and-eos-prediction-positions-only", "logprob_arithmetic": "float32",
        "aggregation": "float64", "cpu_threads": runtime["cpu_threads"],
        "bf16_reduced_precision_reduction": runtime["bf16_reduced_precision_reduction"],
        "fixed_length_table": [{"query_id": q, "task": t, "tokens": n} for (q, t), n in sorted(lengths.items())],
        "global_padding_length": max(lengths.values()),
    }
    return {"numeric_runtime": runtime, "device_indices": [0, 1, 2, 3],
        "execution": "data-parallel-identical-fp32", "replicas": [
            {"physical_gpu_index": i, "actual_numeric_identity": copy.deepcopy(numeric),
             "actual_numeric_identity_sha256": canonical_json_sha256(numeric),
             "hardware": {"physical_gpu_index": i, "uuid": f"GPU-{i}"},
             "source_sha256": plan["code_sha256"]} for i in range(4)]}


class ExecutionContractTests(unittest.TestCase):
    def test_complete_six_condition_matrix_and_kernel_parent(self):
        plan = coverage_plan(full=True)
        execution.validate_execution_plan(plan, plan["blocks"])
        self.assertEqual(len(plan["blocks"]), 7716)
        self.assertEqual(sum(len(plan["catalog"][row["task"]]) for row in plan["blocks"]), 131172)
        with self.assertRaises(PackageError):
            execution.validate_execution_plan(plan, plan["blocks"][:-1])

    def test_old_conditions_or_looser_tolerance_are_not_accepted(self):
        plan = coverage_plan(full=True)
        plan["numeric_policy"]["epsilon"] *= 2
        with self.assertRaisesRegex(PackageError, "recalibrate"):
            execution.validate_execution_plan(plan, plan["blocks"])

    def test_runtime_accepts_new_context_lengths_not_old_identity_sha(self):
        plan = coverage_plan()
        identity = runtime_fixture(plan)
        execution.validate_runtime(plan, identity)
        plan["blocks"][0]["prompt_tokens"] += 100
        with self.assertRaisesRegex(PackageError, "new input geometry"):
            execution.validate_runtime(plan, identity)
        execution.validate_runtime(plan, runtime_fixture(plan))

    def test_tf32_bf16_replica_and_source_changes_are_rejected(self):
        for key, value in (("tf32_matmul", True), ("lm_head_dtype", "torch.bfloat16"),
                           ("use_cache", True), ("projection", "full-logits")):
            with self.subTest(key=key):
                plan = coverage_plan()
                identity = runtime_fixture(plan)
                identity["replicas"][0]["actual_numeric_identity"][key] = value
                with self.assertRaises(PackageError):
                    execution.validate_runtime(plan, identity)
        plan = coverage_plan()
        identity = runtime_fixture(plan)
        identity["replicas"][1]["hardware"]["uuid"] = "GPU-0"
        with self.assertRaises(PackageError):
            execution.validate_runtime(plan, identity)

    def test_run_namespace_rejects_plan_and_historical_paths_before_pool(self):
        plan = coverage_plan(full=True)
        with tempfile.TemporaryDirectory() as tmp, patch.object(execution, "PersistentNumericPool") as pool:
            root = Path(tmp)
            for path in (root / "historical", root / "coverage-output/plans/x", root / "coverage-output/runs"):
                with self.subTest(path=path), self.assertRaises(PackageError):
                    execution.run_pipeline(plan, plan["blocks"], path, root=root)
            pool.assert_not_called()


class PreflightEvidenceTests(unittest.TestCase):
    def execute(self, root, *, error_pass=None, error=0.0, empty_boundary=False):
        plan = coverage_plan()
        if empty_boundary:
            plan["cohorts"]["boundary"] = []
        runtime = runtime_fixture(plan)
        saved, calls = {}, []

        def scorer(runner, contexts, actual, path, *, batch_size, **options):
            self.assertEqual(batch_size, 1)
            calls.append(path.name)
            rows = rows_fixture(contexts, actual, shift=error if path.name == error_pass else 0, **options)
            saved[path.name] = rows
            path.mkdir(parents=True)
            write_json(path / "manifest.json", {"test_fixture": path.name})
            return rows, {}

        with patch.object(execution.scoring, "score_pass", side_effect=scorer), redirect_stdout(io.StringIO()):
            report = execution.preflight(SimpleNamespace(identity=runtime), plan, plan["blocks"], root / "preflight")
        write_json(root / "runtime_identity.json", runtime)
        run = {"preflight_report_sha256": sha256_file(root / "preflight/preflight_report.json")}
        return plan, report, run, saved, calls

    def verify(self, root, plan, run, saved):
        def verify_pass(actual, path, contexts, runtime, options):
            rows = saved[path.name]
            return rows, {}, execution.validate_geometry(rows, contexts, actual, **options)
        with patch.object(execution, "_verify_pass", side_effect=verify_pass):
            return execution.verified_preflight(plan, root, run)

    def test_all_eighteen_passes_are_fresh_and_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, run, saved, calls = self.execute(root)
            self.assertTrue(report["passed"])
            self.assertEqual(len(calls), 18)
            self.assertEqual(report["epsilon"], 0.0013427734375)
            self.assertEqual(report["E8"], 0.00067138671875)
            self.assertEqual(report, self.verify(root, plan, run, saved))

    def test_boundary_last_failure_is_complete_but_not_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, run, saved, calls = self.execute(root, error_pass="boundary-b1-replica", error=0.002)
            self.assertFalse(report["passed"])
            self.assertTrue(report["complete"])
            self.assertEqual(report["failure"], "boundary-replica-gate")
            self.assertEqual(len(calls), 18)
            self.assertEqual(report, self.verify(root, plan, run, saved))

    def test_failed_regression_stops_before_validation_without_recalibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, run, saved, calls = self.execute(root, error_pass="regression-b1-padding", error=0.002)
            self.assertFalse(report["passed"])
            self.assertEqual(calls, ["regression-b1-r0", "regression-b1-r1", "regression-b1-padding"])
            self.assertEqual(report, self.verify(root, plan, run, saved))

    def test_repeat_tolerance_is_independent_one_e_minus_four(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, run, saved, calls = self.execute(root, error_pass="regression-b1-r1", error=0.00015)
            self.assertEqual(report["failure"], "regression-repeat-gate")
            self.assertEqual(len(calls), 2)
            self.assertEqual(report, self.verify(root, plan, run, saved))

    def test_empty_boundary_not_claimed_as_gpu_tested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, run, saved, calls = self.execute(root, empty_boundary=True)
            self.assertTrue(report["passed"])
            self.assertEqual(len(calls), 12)
            self.assertEqual(report["cohorts"]["boundary"]["blocks"], 0)
            self.assertTrue(report["cohorts"]["boundary"]["skipped"])
            self.assertEqual(report, self.verify(root, plan, run, saved))

    def test_forged_success_cannot_replace_failed_raw_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, run, saved, _ = self.execute(root, error_pass="boundary-b1-replica", error=0.002)
            report["passed"] = True
            report.pop("failure")
            path = root / "preflight/preflight_report.json"
            write_json(path, report)
            run["preflight_report_sha256"] = sha256_file(path)
            with self.assertRaisesRegex(PackageError, "terminal preflight"):
                self.verify(root, plan, run, saved)

    def test_changed_geometry_proof_is_rejected_even_with_rebound_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, run, saved, _ = self.execute(root)
            proof = root / "preflight/regression-padding-geometry-proof.json"
            write_json(proof, {"passed": True})
            report["files"] = execution._evidence_files(root / "preflight")
            path = root / "preflight/preflight_report.json"
            write_json(path, report)
            run["preflight_report_sha256"] = sha256_file(path)
            with self.assertRaisesRegex(PackageError, "geometry proof"):
                self.verify(root, plan, run, saved)


class AnalysisBoundaryTests(unittest.TestCase):
    def test_no_query_gold_read_before_raw_complete(self):
        plan = coverage_plan()
        for status in ("running", "preflight_failed", "failed", "interrupted"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_json(root / "run_manifest.json", {"plan_id": plan["plan_id"], "status": status})
                with patch.object(execution, "read_jsonl") as read, self.assertRaises(PackageError):
                    execution.analyze_run(plan, root)
                read.assert_not_called()

    def test_raw_manifest_mutation_blocks_gold_join(self):
        plan = coverage_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dev-b1").mkdir()
            write_json(root / "dev-b1/manifest.json", {})
            write_json(root / "run_manifest.json", {"plan_id": plan["plan_id"], "status": "raw_complete",
                       "raw_manifest_sha256": "changed"})
            with patch.object(execution, "verified_preflight", return_value={"passed": True}), \
                    patch.object(execution, "read_jsonl") as read, self.assertRaisesRegex(PackageError, "raw manifest"):
                execution.analyze_run(plan, root)
            read.assert_not_called()

    def test_complete_missing_analysis_must_not_rejoin_gold(self):
        plan = coverage_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dev-b1").mkdir()
            write_json(root / "dev-b1/manifest.json", {})
            write_json(root / "run_manifest.json", {"plan_id": plan["plan_id"], "status": "complete",
                       "raw_manifest_sha256": sha256_file(root / "dev-b1/manifest.json")})
            with patch.object(execution, "verified_preflight", return_value={"passed": True, "runtime_identity": {}}), \
                    patch.object(execution, "_verify_pass", return_value=([], {}, {})), \
                    patch.object(execution, "read_jsonl") as read, self.assertRaisesRegex(PackageError, "lost its sealed analysis"):
                execution.analyze_run(plan, root)
            read.assert_not_called()


class TerminalIntegrityTests(unittest.TestCase):
    def prepare(self, root, status):
        plan = coverage_plan(full=True)
        output = root / "coverage-output/runs/run-01"
        output.mkdir(parents=True)
        binding = {"plan_id": plan["plan_id"], "physical_device_indices": [0, 1, 2, 3],
                   "execution": "data-parallel-identical-fp32", "phase": "explicit-category-removal",
                   "raw_path": "dev-b1", "production_batch_size": 1, "numeric_policy": execution.POLICY}
        run = {**binding, "schema_version": "general-model-nolabel-run/v1", "status": status,
               "test_content_read": False, "query_gold_loaded_during_scoring": False,
               "automatic_profile_search": False, "full_dev_started": status == "complete",
               "analysis_published": status == "complete", "preflight_report_sha256": "sealed-digest"}
        if status == "preflight_failed":
            run["failure"] = "boundary-replica-gate"
        write_json(output / "binding.json", binding)
        write_json(output / "run_manifest.json", run)
        return plan, output, run

    def test_complete_reentry_verifies_without_gpu_or_manifest_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, output, run = self.prepare(root, "complete")
            before = sha256_file(output / "run_manifest.json")
            with patch.object(execution, "PersistentNumericPool") as pool, \
                    patch.object(execution, "verified_preflight", return_value={"passed": True}) as verify, \
                    patch.object(execution, "analyze_run", return_value={}) as analyze:
                self.assertEqual(run, execution.run_pipeline(plan, plan["blocks"], output, root=root))
            verify.assert_called_once()
            analyze.assert_called_once()
            pool.assert_not_called()
            self.assertEqual(before, sha256_file(output / "run_manifest.json"))

    def test_failed_last_gate_remains_terminal_and_cannot_start_dev(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, output, run = self.prepare(root, "preflight_failed")
            with patch.object(execution, "PersistentNumericPool") as pool, \
                    patch.object(execution, "verified_preflight", return_value={"passed": False, "failure": run["failure"]}):
                self.assertEqual(run, execution.run_pipeline(plan, plan["blocks"], output, root=root))
                (output / "dev-b1").mkdir()
                with self.assertRaisesRegex(PackageError, "downstream artifacts"):
                    execution.run_pipeline(plan, plan["blocks"], output, root=root)
            pool.assert_not_called()

    def test_changed_resume_preflight_rejected_before_gpu_and_without_rebinding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, output, _ = self.prepare(root, "interrupted")
            before = sha256_file(output / "run_manifest.json")
            with patch.object(execution, "PersistentNumericPool") as pool, \
                    patch.object(execution, "verified_preflight", side_effect=PackageError("changed sealed report")), \
                    self.assertRaisesRegex(PackageError, "changed sealed report"):
                execution.run_pipeline(plan, plan["blocks"], output, root=root)
            pool.assert_not_called()
            self.assertEqual(before, sha256_file(output / "run_manifest.json"))

    def test_failed_run_has_no_automatic_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, output, _ = self.prepare(root, "failed")
            with patch.object(execution, "PersistentNumericPool") as pool, self.assertRaisesRegex(PackageError, "automatic retry"):
                execution.run_pipeline(plan, plan["blocks"], output, root=root)
            pool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
