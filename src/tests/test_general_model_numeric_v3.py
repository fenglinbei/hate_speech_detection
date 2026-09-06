import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_numeric_v3 as fallback
from diagnostics.general_model_package import PackageError, read_json, write_json
from tests.test_general_model_numeric import block_fixture, context, full_preflight_contexts, plan_fixture


def config_fixture():
    return {
        "schema_version": "general-model-ld-numeric-fallback-config/v1",
        "parent_plan_ref": "parent_ref.json", "source_failed_run": "failed",
        "protocol_path": "protocol.md", "output_root": "fallback-output",
        "production_batch_size": 1, "raw_path": "dev-b1", "device_indices": [0, 1, 2, 3],
        "validation": {"repeat_abs_tolerance": 1e-4, "reference_abs_tolerance": 1e-4,
                       "inherit_epsilon": True, "epsilon_recalibration": False,
                       "padding_extra": 64, "replica_shift": 1,
                       "challenges": list(fallback.CHALLENGES)},
        "auto_expand_on_pass": True, "stop_on_any_gate_failure": True,
        "automatic_profile_search": False, "test_access": False, "query_gold_in_scoring": False,
    }


def fallback_plan():
    plan = plan_fixture()
    plan["config"]["runtime"].update(dtype="float32", padding_policy="dynamic")
    plan["config"]["execution"] = {"device_indices": [0, 1, 2, 3], "automatic_dev_batch_size": 4,
        "fallback": {"batch_size": 1, "new_run_required": True, "automatic": False,
                     "mixed_batch_artifacts_allowed": False}}
    plan["generation_runtime_identity"]["generation_config"] = {"eos_token_id": 127}
    plan["eos_token_id"] = 127
    plan["fallback_config"] = config_fixture()
    plan["source_failed_run"] = "/unread-source-failure"
    plan["source_failure"] = {"E8": 0.00067138671875, "epsilon": 0.0013427734375,
        "calibration_sha256": "frozen-regression", "runtime_identity_sha256": canonical_json_sha256({"fixture": True})}
    plan["blocks"] = full_preflight_contexts()
    plan["parent_plan"] = {"fixture_parent": True}
    return plan


def rows_fixture(contexts, plan, *, reference=False, prefix=False, padding_extra=0,
                 permuted=False, replica_shift=0, shift=0.0):
    rows = []
    profile = fallback.parent.scoring_profile(prefix=prefix, padding_extra=padding_extra,
                                              permuted=permuted, replica_shift=replica_shift)
    for ctx in contexts:
        row = block_fixture(ctx, plan["catalog"], reference=reference, shift=shift)
        row.update(plan_id=plan["plan_id"], execution_batch_size=1, scoring_profile=profile)
        ids = [candidate["candidate_id"] for candidate in row["candidates"]]
        execution = list(reversed(ids if ctx["task"] == "hate" else ids[1:] + ids[:1])) if permuted else ids
        members = [f"{ctx['record_id']}:{cid}" for cid in ids]
        for index, candidate in enumerate(row["candidates"]):
            candidate.update(
                batch_size=1, effective_batch_size=1, prefix_reference=prefix,
                batch_member_ordinal=index if prefix else 0,
                batch_members=members if prefix else [members[index]],
                prompt_tokens=ctx["prompt_tokens"], prompt_token_ids_sha256=ctx["prompt_token_ids_sha256"],
                sequence_tokens=ctx["prompt_tokens"] + candidate["answer_tokens"] + 1,
                causal_shift=1, use_cache=False, padding_side="right", model_logits_dtype="torch.float32",
                logprob_arithmetic_dtype="torch.float32", physical_gpu_index=replica_shift,
                physical_gpu_uuid=f"GPU-fixture-{replica_shift}",
                scoring_implementation="uncached-prefix-only" if prefix else "full-sequence-selected-projection",
                batch_ordinal=0 if prefix else execution.index(candidate["candidate_id"]),
            )
            if reference:
                candidate["reference_arithmetic_dtype"] = "cpu.torch.float64"
            if prefix:
                unique = {tuple(c["answer_token_ids"][:i]) for c in row["candidates"] for i in range(c["answer_tokens"] + 1)}
                candidate.update(padded_sequence_tokens=None, prefix_padding=False, prefix_unique_forward_count=len(unique))
            else:
                candidate.update(padded_sequence_tokens=candidate["sequence_tokens"] + padding_extra,
                                 padding_challenge_extra=padding_extra)
        rows.append(row)
    return rows


class FallbackConfigTests(unittest.TestCase):
    def test_inherits_science_without_copying_runtime_or_old_batch_four_dispatch(self):
        plan = fallback_plan()
        fallback.validate_config(plan["fallback_config"], plan["config"])
        self.assertEqual(plan["config"]["execution"]["automatic_dev_batch_size"], 4)
        self.assertEqual(plan["fallback_config"]["production_batch_size"], 1)

    def test_rejects_policy_or_scientific_override(self):
        for key, value in (("production_batch_size", 4), ("production_batch_size", True),
                           ("raw_path", "dev-b4"), ("automatic_profile_search", True),
                           ("test_access", True), ("runtime", {}), ("analysis", {})):
            with self.subTest(key=key, value=value):
                plan = fallback_plan()
                plan["fallback_config"][key] = value
                with self.assertRaises(PackageError):
                    fallback.validate_config(plan["fallback_config"], plan["config"])

    def test_requires_prior_registered_unmixed_fallback(self):
        for key, value in (("batch_size", 4), ("new_run_required", False),
                           ("automatic", True), ("mixed_batch_artifacts_allowed", True)):
            with self.subTest(key=key):
                plan = fallback_plan()
                plan["config"]["execution"]["fallback"][key] = value
                with self.assertRaises(PackageError):
                    fallback.validate_config(plan["fallback_config"], plan["config"])


class FallbackGeometryTests(unittest.TestCase):
    def setUp(self):
        self.plan = fallback_plan()
        self.contexts = [context(), context(task="group")]

    def test_all_registered_batch_one_profiles_have_real_geometry(self):
        for options in ({}, {"reference": True}, {"padding_extra": 64}, {"prefix": True},
                        {"permuted": True}, {"replica_shift": 1}):
            with self.subTest(options=options):
                rows = rows_fixture(self.contexts, self.plan, **options)
                proof = fallback.validate_geometry(rows, self.contexts, self.plan, **options)
                self.assertTrue(proof["passed"])
                self.assertFalse(proof["within_batch_row_position_claimed"])

    def test_configuration_label_cannot_hide_batch_padding_or_precision_change(self):
        mutations = (("batch_size", 4), ("effective_batch_size", 2), ("batch_members", ["other"]),
                     ("batch_member_ordinal", 1), ("padded_sequence_tokens", 500),
                     ("model_logits_dtype", "torch.bfloat16"), ("padding_side", "left"),
                     ("causal_shift", 0), ("use_cache", True), ("prefix_reference", True),
                     ("eos_token_id", 99), ("prompt_token_ids_sha256", "other"))
        for key, value in mutations:
            with self.subTest(key=key):
                rows = rows_fixture(self.contexts, self.plan)
                rows[0]["candidates"][0][key] = value
                with self.assertRaises(PackageError):
                    fallback.validate_geometry(rows, self.contexts, self.plan)

    def test_members_checks_actual_execution_ordinals_not_saved_canonical_order(self):
        rows = rows_fixture(self.contexts, self.plan, permuted=True)
        for row in rows:
            self.assertEqual([c["ordinal"] for c in row["candidates"]], list(range(len(row["candidates"]))))
        self.assertEqual([c["batch_ordinal"] for c in rows[0]["candidates"]], [1, 0])
        rows[0]["candidates"][0]["batch_ordinal"] = 0
        rows[0]["candidates"][1]["batch_ordinal"] = 1
        with self.assertRaisesRegex(PackageError, "execution order"):
            fallback.validate_geometry(rows, self.contexts, self.plan, permuted=True)

    def test_prefix_reference_metadata_is_not_misrepresented_as_batched_candidates(self):
        rows = rows_fixture(self.contexts, self.plan, prefix=True)
        self.assertEqual(len(rows[1]["candidates"][0]["batch_members"]), 32)
        fallback.validate_geometry(rows, self.contexts, self.plan, prefix=True)
        rows[1]["candidates"][0]["prefix_unique_forward_count"] -= 1
        with self.assertRaises(PackageError):
            fallback.validate_geometry(rows, self.contexts, self.plan, prefix=True)

    def test_replica_proves_different_uuid_and_identical_real_batch_one(self):
        baseline = rows_fixture(self.contexts, self.plan, reference=True)
        shifted = rows_fixture(self.contexts, self.plan, replica_shift=1)
        self.assertTrue(fallback.parent.replica_proof(baseline, shifted)["batch_geometry_unchanged"])
        shifted[0]["candidates"][0]["physical_gpu_uuid"] = "GPU-fixture-0"
        with self.assertRaises(PackageError):
            fallback.parent.replica_proof(baseline, shifted)


class FallbackPreflightTests(unittest.TestCase):
    def run_preflight(self, *, error_pass=None, error=0.0, runtime=None):
        plan = fallback_plan()
        calls = []
        def score(runner, contexts, actual_plan, output, *, batch_size, **options):
            self.assertEqual(batch_size, 1)
            calls.append(output.name)
            shift = error if output.name == error_pass else 0.0
            return rows_fixture(contexts, actual_plan, shift=shift, **options), {}
        with tempfile.TemporaryDirectory() as tmp, patch.object(fallback.parent, "score_pass", side_effect=score), redirect_stdout(io.StringIO()):
            result = fallback.preflight(SimpleNamespace(identity={"fixture": True} if runtime is None else runtime),
                                         plan, plan["blocks"], Path(tmp))
            return result, calls

    def test_pass_keeps_old_epsilon_and_e8_even_when_observed_error_is_zero(self):
        report, calls = self.run_preflight()
        self.assertTrue(report["passed"])
        self.assertEqual(len(calls), 12)
        self.assertFalse(any("b4" in name or "tail" in name for name in calls))
        self.assertEqual(report["E8"], 0.00067138671875)
        self.assertEqual(report["epsilon"], 0.0013427734375)
        self.assertEqual(report["observed_fallback_max"], 0.0)
        self.assertEqual(report["validation_cohort_status"], fallback.COHORT_STATUS)
        self.assertEqual(report["not_applicable"], fallback.NOT_APPLICABLE)

    def test_regression_is_a_gate_against_inherited_epsilon_not_a_new_calibration(self):
        report, calls = self.run_preflight(error_pass="regression-b1-padding", error=0.0015)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "regression-padding-compatibility-gate")
        self.assertEqual(report["epsilon"], 0.0013427734375)
        self.assertEqual(calls, ["regression-b1-r0", "regression-b1-r1", "regression-b1-padding"])

    def test_validation_failure_stops_without_profile_search(self):
        report, calls = self.run_preflight(error_pass="validation-b1-padding", error=0.0015)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "validation-padding-compatibility-gate")
        self.assertNotIn("validation-b1-prefix", calls)

    def test_repeat_limit_remains_one_e_minus_four(self):
        report, calls = self.run_preflight(error_pass="regression-b1-r1", error=0.00015)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "regression-baseline-gate")
        self.assertEqual(len(calls), 2)

    def test_rejects_actual_runtime_drift_before_any_scoring(self):
        with self.assertRaisesRegex(PackageError, "actual source calibration runtime"):
            self.run_preflight(runtime={"fixture": False})


class FallbackSourceTests(unittest.TestCase):
    def source(self, root):
        plan = fallback_plan()
        original = {**plan, "plan_id": "source-plan"}
        regression = {
            "complete": True, "passed": True, "baseline_passed": True,
            "baseline_repeat_max_abs_error": 0.0, "reference_max_abs_error": 0.0,
            "max_abs_error": plan["source_failure"]["E8"],
            "challenges": {key: {"max_abs_error": plan["source_failure"]["E8"], "passed": True}
                           for key in fallback.parent.CHALLENGES},
        }
        report = {
            "schema_version": "general-model-ld-numeric-calibration/v2", "plan_id": "source-plan",
            "runtime_identity": {"fixture": True}, "passed": False, "complete": False,
            "failure": "validation-batch-compatibility-gate", "E8": plan["source_failure"]["E8"],
            "epsilon": plan["source_failure"]["epsilon"], "files": {},
            "error_families": list(fallback.parent.CHALLENGES),
            "query_gold_loaded": False, "test_content_read": False,
            "scientific_effect_checked": False, "formal_test_authorized": False,
            "cohorts": {"regression": regression, "validation": {
                "baseline_passed": True, "baseline_repeat_max_abs_error": 0.0, "reference_max_abs_error": 0.0,
                "challenges": {"batch": {"passed": False, "max_abs_error": 0.001621246337890625}}}},
        }
        (root / "preflight").mkdir(parents=True, exist_ok=True)
        write_json(root / "preflight/preflight_report.json", report)
        write_json(root / "runtime_identity.json", {"fixture": True})
        run = {"plan_id": "source-plan", "status": "preflight_failed", "failure": report["failure"],
               "full_dev_started": False, "analysis_published": False, "test_content_read": False,
               "query_gold_loaded_during_scoring": False,
               "preflight_report_sha256": sha256_file(root / "preflight/preflight_report.json")}
        write_json(root / "run_manifest.json", run)
        return original, report, run

    def test_binds_failed_source_and_exact_regression_preimage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, _ = self.source(root)
            binding = fallback.source_failure_binding(plan, root)
            self.assertEqual(binding["calibration_sha256"], canonical_json_sha256(report["cohorts"]["regression"]))
            self.assertEqual(binding["epsilon"], 0.0013427734375)
            self.assertEqual(binding["runtime_identity_sha256"], canonical_json_sha256({"fixture": True}))

    def test_recalibrated_epsilon_is_rejected_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, report, run = self.source(root)
            report["epsilon"] = 0.005
            write_json(root / "preflight/preflight_report.json", report)
            run["preflight_report_sha256"] = sha256_file(root / "preflight/preflight_report.json")
            write_json(root / "run_manifest.json", run)
            with self.assertRaisesRegex(PackageError, "epsilon formula"):
                fallback.source_failure_binding(plan, root)

    def test_source_with_any_development_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, _, _ = self.source(root)
            (root / "dev-b4").mkdir()
            with self.assertRaises(PackageError):
                fallback.source_failure_binding(plan, root)

    def test_new_plan_snapshots_full_parent_sources_and_revalidates_source_failure(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            original, _, _ = self.source(root / "failed")
            original.pop("parent_plan")
            original["code_sha256"] = {"parent.py": "pending"}
            (root / "parent.py").write_text("frozen parent\n")
            original["code_sha256"]["parent.py"] = sha256_file(root / "parent.py")
            for name in fallback.CODE_FILES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("synthetic source\n")
            write_json(root / "parent_ref.json", {"fixture": True})
            (root / "protocol.md").write_text("synthetic registered fallback\n")
            write_json(root / "config.json", config_fixture())
            with patch.object(fallback.parent, "load_plan", return_value=(original, original["blocks"])):
                ref = fallback.build_plan(root / "config.json", root=root)
                loaded, contexts = fallback.load_plan(Path(ref["target_path"]), root=root)
                self.assertEqual(loaded["config"], original["config"])
                self.assertEqual(loaded["config"]["execution"]["automatic_dev_batch_size"], 4)
                self.assertEqual(set(loaded["code_sha256"]), {"parent.py", *fallback.CODE_FILES})
                self.assertEqual(loaded["parent_plan"], original)
                self.assertEqual(contexts, original["blocks"])
                snapshot = Path(ref["target_path"]) / "source/parent.py"
                snapshot.write_text("changed source\n")
                with self.assertRaisesRegex(PackageError, "source snapshot"):
                    fallback.load_plan(Path(ref["target_path"]), root=root)


class FallbackEvidenceTests(unittest.TestCase):
    def test_complete_summary_without_twelve_sealed_passes_is_rejected(self):
        plan = fallback_plan()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(PackageError, "twelve-pass"):
                fallback._verify_passed_evidence(plan, Path(tmp), {"files": {}})

    def test_unlisted_payload_is_rejected_even_when_existing_hashes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "unexpected-proof.json", {})
            with self.assertRaisesRegex(PackageError, "inventory"):
                fallback._verify_passed_evidence(fallback_plan(), root, {"files": {}})


class FallbackPipelineTests(unittest.TestCase):
    def test_success_order_and_complete_reentry_are_explicitly_read_only(self):
        plan = fallback_plan()
        events = []
        runner = SimpleNamespace(identity={"fixture": True}, close=lambda: events.append("close"))
        def gate(runner, plan, contexts, output):
            events.append("preflight")
            output.mkdir(parents=True)
            write_json(output / "preflight_report.json", {"passed": True})
            return {"passed": True}
        def verified(plan, output, run):
            events.append("verified")
            return {"passed": True}
        def score(runner, contexts, plan, output, *, batch_size):
            events.append("dev-b1")
            self.assertEqual(batch_size, 1)
            output.mkdir()
            write_json(output / "manifest.json", {"blocks": 7716})
            return [], {"blocks": 7716}
        def analyze(plan, output, **kwargs):
            events.append("analysis")
            self.assertIn(read_json(output / "run_manifest.json")["status"], {"raw_complete", "complete"})
            self.assertIn("close", events)
            return {}
        with tempfile.TemporaryDirectory() as tmp, patch.object(fallback, "load_plan", return_value=(plan, plan["blocks"])), \
                patch("diagnostics.general_model_numeric_pool.PersistentNumericPool", return_value=runner) as pool, \
                patch.object(fallback, "preflight", side_effect=gate), \
                patch.object(fallback, "_verified_preflight", side_effect=verified), \
                patch.object(fallback.parent, "score_pass", side_effect=score), \
                patch.object(fallback, "analyze_run", side_effect=analyze), redirect_stdout(io.StringIO()):
            output = Path(tmp)
            result = fallback.run_pipeline(Path("plan"), output)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(events, ["preflight", "verified", "dev-b1", "close", "analysis"])
            old = (output / "run_manifest.json").read_bytes()
            self.assertEqual(fallback.run_pipeline(Path("plan"), output), result)
            self.assertEqual(events[-2:], ["verified", "analysis"])
            self.assertEqual((output / "run_manifest.json").read_bytes(), old)
            self.assertEqual(pool.call_count, 1)

    def test_gate_failure_closes_pool_and_never_starts_dev_or_gold_analysis(self):
        plan = fallback_plan()
        runner = SimpleNamespace(identity={"fixture": True}, close=lambda: closed.append(True))
        closed = []
        def gate(runner, plan, contexts, output):
            report = {"passed": False, "failure": "regression-padding-compatibility-gate"}
            output.mkdir(parents=True)
            write_json(output / "preflight_report.json", report)
            return report
        with tempfile.TemporaryDirectory() as tmp, patch.object(fallback, "load_plan", return_value=(plan, plan["blocks"])), \
                patch("diagnostics.general_model_numeric_pool.PersistentNumericPool", return_value=runner), \
                patch.object(fallback, "preflight", side_effect=gate), patch.object(fallback.parent, "score_pass") as score, \
                patch.object(fallback, "analyze_run") as analyze:
            result = fallback.run_pipeline(Path("plan"), Path(tmp))
            self.assertEqual(result["status"], "preflight_failed")
            self.assertEqual(result["production_batch_size"], 1)
            score.assert_not_called()
            analyze.assert_not_called()
            self.assertEqual(closed, [True])

    def test_terminal_failed_execution_is_not_restarted_or_rewritten(self):
        plan = fallback_plan()
        with tempfile.TemporaryDirectory() as tmp, patch.object(fallback, "load_plan", return_value=(plan, plan["blocks"])), \
                patch("diagnostics.general_model_numeric_pool.PersistentNumericPool", side_effect=RuntimeError("fixture crash")) as pool:
            output = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "fixture crash"):
                fallback.run_pipeline(Path("plan"), output)
            old = (output / "run_manifest.json").read_bytes()
            with self.assertRaisesRegex(PackageError, "sealed"):
                fallback.run_pipeline(Path("plan"), output)
            self.assertEqual((output / "run_manifest.json").read_bytes(), old)
            self.assertEqual(pool.call_count, 1)

    def test_source_failure_directory_cannot_be_reused(self):
        plan = fallback_plan()
        with tempfile.TemporaryDirectory() as tmp, patch.object(fallback, "load_plan", return_value=(plan, plan["blocks"])), \
                patch("diagnostics.general_model_numeric_pool.PersistentNumericPool") as pool:
            plan["source_failed_run"] = tmp
            with self.assertRaises(PackageError):
                fallback.run_pipeline(Path("plan"), Path(tmp))
            pool.assert_not_called()

    def test_first_expansion_requires_verified_evidence_and_raw_complete_resume_needs_no_gpu(self):
        plan = fallback_plan()
        closed = []
        runner = SimpleNamespace(identity={"fixture": True}, close=lambda: closed.append(True))
        def gate(runner, plan, contexts, output):
            output.mkdir(parents=True)
            report = {"passed": True}
            write_json(output / "preflight_report.json", report)
            return report
        def score(runner, contexts, plan, output, *, batch_size):
            self.assertEqual(batch_size, 1)
            self.assertEqual(output.name, "dev-b1")
            output.mkdir()
            write_json(output / "manifest.json", {"blocks": 7716})
            return [], {"blocks": 7716}
        with tempfile.TemporaryDirectory() as tmp, patch.object(fallback, "load_plan", return_value=(plan, plan["blocks"])), \
                patch("diagnostics.general_model_numeric_pool.PersistentNumericPool", return_value=runner) as pool, \
                patch.object(fallback, "preflight", side_effect=gate), \
                patch.object(fallback, "_verified_preflight", return_value={"passed": True}) as verified, \
                patch.object(fallback.parent, "score_pass", side_effect=score), \
                patch.object(fallback, "analyze_run", side_effect=RuntimeError("analysis only")), redirect_stdout(io.StringIO()):
            output = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "analysis only"):
                fallback.run_pipeline(Path("plan"), output)
            self.assertEqual(read_json(output / "run_manifest.json")["status"], "raw_complete")
            self.assertEqual(closed, [True])
            self.assertEqual(verified.call_count, 1)
            with patch.object(fallback, "analyze_run", return_value={}):
                result = fallback.run_pipeline(Path("plan"), output)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(pool.call_count, 1)
            old = (output / "run_manifest.json").read_bytes()
            with patch.object(fallback, "_verified_preflight", side_effect=PackageError("changed gate")):
                with self.assertRaisesRegex(PackageError, "changed gate"):
                    fallback.run_pipeline(Path("plan"), output)
            self.assertEqual((output / "run_manifest.json").read_bytes(), old)

    def test_missing_evidence_does_not_expand_passed_boolean_to_development(self):
        plan = fallback_plan()
        runner = SimpleNamespace(identity={"fixture": True}, close=lambda: None)
        def gate(runner, plan, contexts, output):
            output.mkdir(parents=True)
            write_json(output / "preflight_report.json", {"passed": True})
            return {"passed": True}
        with tempfile.TemporaryDirectory() as tmp, patch.object(fallback, "load_plan", return_value=(plan, plan["blocks"])), \
                patch("diagnostics.general_model_numeric_pool.PersistentNumericPool", return_value=runner), \
                patch.object(fallback, "preflight", side_effect=gate), \
                patch.object(fallback, "_verified_preflight", side_effect=PackageError("missing evidence")), \
                patch.object(fallback.parent, "score_pass") as score:
            with self.assertRaisesRegex(PackageError, "missing evidence"):
                fallback.run_pipeline(Path("plan"), Path(tmp))
            score.assert_not_called()

    def test_raw_identity_requires_batch_one_pass_and_passed_runtime(self):
        plan = fallback_plan()
        report = {"runtime_identity": {"fixture": True}}
        receipt = {"identity": {"plan_id": plan["plan_id"], "runtime": report["runtime_identity"],
            "batch_size": 1, "reference": False, "pass_name": "dev-b1",
            "records": [row["record_id"] for row in plan["blocks"]],
            "scoring_profile": fallback.parent.scoring_profile()},
            "execution": "data-parallel-identical-fp32", "query_gold_loaded": False,
            "test_content_read": False, "mixed_execution_modes": False}
        fallback.validate_raw_identity(plan, report, receipt)
        for key, value in (("batch_size", 4), ("pass_name", "dev-b4"), ("runtime", {"other": True}), ("reference", True)):
            with self.subTest(key=key):
                changed = copy.deepcopy(receipt)
                changed["identity"][key] = value
                with self.assertRaises(PackageError):
                    fallback.validate_raw_identity(plan, report, changed)


if __name__ == "__main__":
    unittest.main()
