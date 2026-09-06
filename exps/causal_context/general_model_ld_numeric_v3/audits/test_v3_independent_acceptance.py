"""Independent synthetic acceptance checks; no model, GPU, or real query gold."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_numeric_v3 as numeric
from diagnostics.general_model_package import PackageError, read_json, write_json, write_jsonl
from tests.test_general_model_numeric import block_fixture, context, full_preflight_contexts, plan_fixture


EPSILON = 0.0013427734375


def fallback_plan():
    plan = plan_fixture()
    plan["generation_runtime_identity"]["generation_config"] = {"eos_token_id": 127}
    plan.update(eos_token_id=127, source_failed_run="synthetic-failed-run",
                source_failure={"E8": EPSILON / 2, "epsilon": EPSILON,
                                "calibration_sha256": "synthetic-calibration",
                                "runtime_identity_sha256": canonical_json_sha256({"synthetic": True})},
                fallback_config={"validation": {"repeat_abs_tolerance": 1e-4,
                    "reference_abs_tolerance": 1e-4, "padding_extra": 64, "replica_shift": 1}})
    plan["blocks"] = full_preflight_contexts()
    return plan


def geometry_rows(contexts, plan, *, reference=False, prefix=False, padding_extra=0,
                  permuted=False, replica_shift=0, shift=0.0):
    rows = []
    for block_index, ctx in enumerate(contexts):
        row = block_fixture(ctx, plan["catalog"], reference=reference, shift=shift)
        row.update(plan_id=plan["plan_id"], execution_batch_size=1,
                   scoring_profile=numeric.parent.scoring_profile(prefix=prefix,
                       padding_extra=padding_extra, permuted=permuted, replica_shift=replica_shift))
        catalog = plan["catalog"][ctx["task"]]
        ids = [candidate["candidate_id"] for candidate in catalog]
        order = list(reversed(ids if ctx["task"] == "hate" else ids[1:] + ids[:1])) if permuted else ids
        members = [ctx["record_id"] + ":" + candidate_id for candidate_id in ids]
        prefixes = {tuple(c["answer_token_ids"][:i]) for c in catalog for i in range(c["answer_tokens"] + 1)}
        for index, candidate in enumerate(row["candidates"]):
            length = ctx["prompt_tokens"] + candidate["answer_tokens"] + 1
            device = (block_index + replica_shift) % 4
            candidate.update(batch_size=1, effective_batch_size=1, prefix_reference=prefix,
                batch_member_ordinal=index if prefix else 0,
                batch_members=members if prefix else [members[index]],
                batch_ordinal=0 if prefix else order.index(candidate["candidate_id"]),
                prompt_tokens=ctx["prompt_tokens"], prompt_token_ids_sha256=ctx["prompt_token_ids_sha256"],
                sequence_tokens=length, padded_sequence_tokens=None if prefix else length + padding_extra,
                causal_shift=1, use_cache=False, padding_side="right", model_logits_dtype="torch.float32",
                logprob_arithmetic_dtype="torch.float32", physical_gpu_index=device,
                physical_gpu_uuid=f"GPU-synthetic-{device}",
                scoring_implementation="uncached-prefix-only" if prefix else "full-sequence-selected-projection")
            if prefix:
                candidate.update(prefix_padding=False, prefix_unique_forward_count=len(prefixes))
            else:
                candidate["padding_challenge_extra"] = padding_extra
            if reference:
                candidate["reference_arithmetic_dtype"] = "cpu.torch.float64"
        rows.append(row)
    return rows


class GeometryIndependentTests(unittest.TestCase):
    def setUp(self):
        self.plan = fallback_plan()
        self.contexts = [context("1"), context("1", task="group")]

    def test_exact_registered_modes_are_accepted(self):
        for options in ({}, {"reference": True}, {"padding_extra": 64}, {"prefix": True},
                        {"permuted": True}, {"replica_shift": 1}):
            with self.subTest(options=options):
                rows = geometry_rows(self.contexts, self.plan, **options)
                proof = numeric.validate_geometry(rows, self.contexts, self.plan, **options)
                self.assertTrue(proof["passed"])
                self.assertEqual(proof["candidates"], 34)

    def test_metadata_cannot_emulate_real_batch_one(self):
        mutations = {"batch_size": 4, "effective_batch_size": 4, "batch_member_ordinal": 1,
                     "batch_members": ["dummy"], "prefix_reference": True,
                     "prompt_tokens": 4, "prompt_token_ids_sha256": "changed",
                     "sequence_tokens": 99, "padded_sequence_tokens": 99,
                     "causal_shift": 0, "use_cache": True, "padding_side": "left",
                     "model_logits_dtype": "torch.bfloat16", "logprob_arithmetic_dtype": "torch.float64",
                     "token_boundary_checked": False, "reference_checked": True,
                     "scoring_implementation": "uncached-prefix-only", "padding_challenge_extra": 64}
        for key, value in mutations.items():
            with self.subTest(field=key):
                rows = geometry_rows(self.contexts, self.plan)
                rows[0]["candidates"][0][key] = value
                with self.assertRaises(PackageError):
                    numeric.validate_geometry(rows, self.contexts, self.plan)

    def test_eos_must_equal_the_frozen_plan_eos(self):
        rows = geometry_rows(self.contexts, self.plan)
        rows[0]["candidates"][0]["eos_token_id"] = self.plan["eos_token_id"] + 1
        with self.assertRaises(PackageError):
            numeric.validate_geometry(rows, self.contexts, self.plan)

    def test_permutation_label_alone_does_not_prove_changed_execution_order(self):
        rows = geometry_rows(self.contexts, self.plan)
        for row in rows:
            row["scoring_profile"] = numeric.parent.scoring_profile(permuted=True)
        with self.assertRaisesRegex(PackageError, "execution order"):
            numeric.validate_geometry(rows, self.contexts, self.plan, permuted=True)

    def test_prefix_reference_cannot_be_accepted_as_production_raw(self):
        rows = geometry_rows(self.contexts, self.plan, prefix=True)
        with self.assertRaises(PackageError):
            numeric.validate_geometry(rows, self.contexts, self.plan)


class PreflightIndependentTests(unittest.TestCase):
    def setUp(self):
        self.plan = fallback_plan()
        self.contexts = full_preflight_contexts()
        self.runner = SimpleNamespace(identity={"synthetic": True})
        self.calls = []
        self.shifts = {}

    def fake_pass(self, runner, contexts, plan, output, *, batch_size, reference=False, **options):
        self.assertEqual(batch_size, 1)
        self.calls.append((output.name, [row["query_id"] for row in contexts]))
        return geometry_rows(contexts, plan, reference=reference,
                             shift=self.shifts.get(output.name, 0.0), **options), {}

    def execute(self, contexts=None):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(numeric.parent, "score_pass", side_effect=self.fake_pass):
            return numeric.preflight(self.runner, self.plan, contexts or self.contexts, Path(directory))

    def test_all_modes_cover_original_disjoint_cohorts_without_recalibration(self):
        self.shifts["regression-b1-prefix"] = EPSILON / 10
        self.shifts["validation-b1-prefix"] = EPSILON / 2
        report = self.execute()
        self.assertTrue(report["passed"])
        self.assertEqual(report["epsilon"], EPSILON)
        self.assertEqual(report["E8"], EPSILON / 2)
        self.assertEqual(report["calibration_mode"], numeric.CALIBRATION_MODE)
        self.assertEqual(len(self.calls), 12)
        for name, query_ids in self.calls:
            cohort = name.split("-", 1)[0]
            self.assertEqual(set(query_ids), set(self.plan["cohorts"][cohort]))
            self.assertEqual(len(query_ids), len(set(query_ids)) * 12)
        self.assertEqual(set(report["not_applicable"]), set(numeric.NOT_APPLICABLE))

    def test_each_production_challenge_stops_on_either_cohort_failure(self):
        for cohort in ("regression", "validation"):
            for challenge in numeric.CHALLENGES:
                with self.subTest(cohort=cohort, challenge=challenge):
                    self.calls.clear()
                    failed = f"{cohort}-b1-{challenge}"
                    self.shifts = {failed: EPSILON * 2}
                    report = self.execute()
                    self.assertFalse(report["passed"])
                    self.assertEqual(report["failure"], f"{cohort}-{challenge}-compatibility-gate")
                    self.assertEqual(self.calls[-1][0], failed)
                    self.assertEqual(report["epsilon"], EPSILON)
                    if cohort == "regression":
                        self.assertFalse(report["validation_executed"])

    def test_incomplete_cohort_is_rejected_before_any_forward(self):
        with self.assertRaisesRegex(PackageError, "incomplete or overlapping"):
            self.execute(self.contexts[1:])
        self.assertEqual(self.calls, [])

    def test_duplicated_query_membership_is_rejected_before_any_forward(self):
        self.plan["cohorts"]["regression"].append("1")
        with self.assertRaisesRegex(PackageError, "incomplete or overlapping"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_runtime_different_from_source_failure_is_rejected_before_forward(self):
        self.runner.identity = {"synthetic": "changed"}
        with self.assertRaisesRegex(PackageError, "runtime"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_margin_gate_can_fail_when_every_candidate_token_error_is_below_epsilon(self):
        original = self.fake_pass

        def margin_shift(runner, contexts, plan, output, **options):
            rows, receipt = original(runner, contexts, plan, output, **options)
            if output.name == "regression-b1-prefix":
                for row in rows:
                    if row["task"] == "hate":
                        for index, candidate in enumerate(row["candidates"]):
                            delta = EPSILON * 0.75 * (1 if index == 0 else -1)
                            candidate["token_logprobs"][0] += delta
                            candidate["scores"] = numeric.candidate_scores(candidate["token_logprobs"],
                                                                           candidate["eos_logprob"])
                            candidate.update(candidate["scores"])
            return rows, receipt

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(numeric.parent, "score_pass", side_effect=margin_shift):
            report = numeric.preflight(self.runner, self.plan, self.contexts, Path(directory))
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "regression-prefix-compatibility-gate")
        error = report["cohorts"]["regression"]["challenges"]["prefix"]
        self.assertIn("margin/", error["largest_error"]["metric"])
        self.assertAlmostEqual(error["max_abs_error"], EPSILON * 1.5)

    def test_complete_passed_evidence_roundtrip_and_missing_pass_rejection(self):
        def sealed_fake(runner, contexts, plan, output, *, batch_size, reference=False, **options):
            rows, _ = self.fake_pass(runner, contexts, plan, output, batch_size=batch_size,
                                     reference=reference, **options)
            output.mkdir(parents=True)
            write_jsonl(output / "scores.jsonl", rows)
            receipt = {"identity": {"plan_id": plan["plan_id"], "runtime": runner.identity,
                "batch_size": 1, "reference": reference, "pass_name": output.name,
                "records": [row["record_id"] for row in contexts],
                "scoring_profile": numeric.parent.scoring_profile(**options)},
                "execution": "data-parallel-identical-fp32", "query_gold_loaded": False,
                "test_content_read": False, "mixed_execution_modes": False}
            write_json(output / "manifest.json", receipt)
            write_json(output / "pool_binding.json", {"synthetic_shard_validation_mocked": True})
            return rows, receipt

        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch.object(numeric.parent, "score_pass", side_effect=sealed_fake), \
                patch("diagnostics.general_model_numeric_pool.validate_sealed_pass",
                      side_effect=lambda path, plan: read_json(path / "manifest.json")):
            output = Path(temporary)
            report = numeric.preflight(self.runner, self.plan, self.contexts, output / "preflight")
            report_path = output / "preflight/preflight_report.json"
            run = {"preflight_report_sha256": sha256_file(report_path)}
            verified = numeric._verified_preflight(self.plan, output, run)
            self.assertTrue(verified["passed"])
            missing = "validation-b1-prefix/scores.jsonl"
            (output / "preflight" / missing).unlink()
            del report["files"][missing]
            write_json(report_path, report)
            run["preflight_report_sha256"] = sha256_file(report_path)
            with self.assertRaisesRegex(PackageError, "required twelve-pass evidence"):
                numeric._verified_preflight(self.plan, output, run)


class SealedPreflightIndependentTests(unittest.TestCase):
    def setUp(self):
        self.plan = fallback_plan()
        self.report = {
            "schema_version": "general-model-ld-numeric-calibration/v3",
            "plan_id": self.plan["plan_id"], "passed": True, "complete": True,
            "E8": EPSILON / 2, "epsilon": EPSILON,
            "source_calibration_sha256": "synthetic-calibration",
            "source_failed_run": self.plan["source_failed_run"],
            "runtime_identity": {"synthetic": True},
            "calibration_mode": numeric.CALIBRATION_MODE,
            "validation_cohort_status": numeric.COHORT_STATUS,
            "error_families": list(numeric.CHALLENGES), "not_applicable": numeric.NOT_APPLICABLE,
            "query_gold_loaded": False, "test_content_read": False,
            "scientific_effect_checked": False, "formal_test_authorized": False,
            "files": {}, "cohorts": {},
        }
        for cohort in ("regression", "validation"):
            self.report["cohorts"][cohort] = {
                "blocks": len(self.plan["cohorts"][cohort]) * 12,
                "complete": True, "passed": True, "baseline_passed": True,
                "baseline_repeat_max_abs_error": 0.0, "reference_max_abs_error": 0.0,
                "challenges": {name: {"passed": True, "max_abs_error": 0.0}
                               for name in numeric.CHALLENGES},
            }

    def verify(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(numeric.parent, "_verified_preflight", return_value=self.report):
            return numeric._verified_preflight(self.plan, Path(directory), {})

    def test_passed_summary_without_mandatory_evidence_is_rejected(self):
        with self.assertRaises(PackageError):
            self.verify()

    def test_changed_epsilon_or_runtime_cannot_be_accepted_as_inherited(self):
        for field, value in {"epsilon": EPSILON * 2, "E8": EPSILON,
                             "runtime_identity": {"synthetic": "changed"},
                             "calibration_mode": "recalibrated"}.items():
            with self.subTest(field=field):
                original = copy.deepcopy(self.report)
                self.report[field] = value
                with self.assertRaises(PackageError):
                    self.verify()
                self.report = original


class GoldBoundaryIndependentTests(unittest.TestCase):
    def test_failed_preflight_status_cannot_reach_gold_or_raw_reads(self):
        calls = []

        def reader(path):
            calls.append(str(path))
            self.assertEqual(path.name, "run_manifest.json")
            return {"status": "preflight_failed"}

        with patch.object(numeric, "read_json", side_effect=reader), \
                patch.object(numeric, "read_jsonl", side_effect=AssertionError("unexpected content read")):
            with self.assertRaisesRegex(PackageError, "sealed complete raw"):
                numeric.analyze_run(fallback_plan(), Path("synthetic-output"))
        self.assertEqual(len(calls), 1)

    def test_incomplete_raw_cannot_reach_gold(self):
        def reader(path):
            if path.name == "run_manifest.json":
                return {"status": "raw_complete"}
            if path.name == "manifest.json" and path.parent.name == "dev-b1":
                return {"status": "complete", "blocks": 7715, "candidates": 131172}
            raise AssertionError("unexpected input access: " + str(path))

        with patch.object(numeric, "read_json", side_effect=reader), \
                patch.object(numeric, "_verified_preflight", return_value={"passed": True}), \
                patch.object(numeric, "read_jsonl", side_effect=AssertionError("unexpected gold read")):
            with self.assertRaisesRegex(PackageError, "incomplete or changed"):
                numeric.analyze_run(fallback_plan(), Path("synthetic-output"))

    def test_raw_identity_rejects_different_runtime_plan_or_batch(self):
        plan = fallback_plan()
        report = {"runtime_identity": {"synthetic": True}}
        receipt = {"identity": {"plan_id": plan["plan_id"], "runtime": report["runtime_identity"],
            "batch_size": 1, "reference": False, "pass_name": "dev-b1",
            "scoring_profile": numeric.parent.scoring_profile(),
            "records": [row["record_id"] for row in plan["blocks"]]},
            "execution": "data-parallel-identical-fp32", "query_gold_loaded": False,
            "test_content_read": False, "mixed_execution_modes": False}
        numeric.validate_raw_identity(plan, report, receipt)
        for key, value in {"plan_id": "other-plan", "runtime": {"other": True}, "batch_size": 4,
                           "reference": True, "pass_name": "dev-b4", "records": []}.items():
            with self.subTest(key=key):
                changed = copy.deepcopy(receipt)
                changed["identity"][key] = value
                with self.assertRaises(PackageError):
                    numeric.validate_raw_identity(plan, report, changed)


if __name__ == "__main__":
    unittest.main()
