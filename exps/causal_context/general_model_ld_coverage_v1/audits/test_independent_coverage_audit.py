"""Synthetic-only checks; no model, query data, gold file or run is opened."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import independent_coverage_audit as audit
from independent_preflight_audit import flat_readouts
from independent_preflight_audit import validate_geometry as frozen_geometry
from diagnostics import general_model_numeric_coverage_analysis as production
from tests.test_general_model_numeric_coverage_analysis import fixture


def raw_fixture(*, empty_resources=False):
    frame, gold, blocks = fixture(empty_resources=empty_resources)
    catalog = production.candidate_catalog()
    for task, candidates in catalog.items():
        for candidate in candidates:
            candidate["answer_token_ids"] = list(range(1, 2 + candidate["ordinal"] % 3))
    contexts = []
    for index, block in enumerate(blocks):
        record_id = ":".join(block[key] for key in ("query_id", "task", "condition"))
        context = {key: block[key] for key in ("query_id", "task", "condition")}
        context.update(record_id=record_id, context_sha256=audit.canonical_hash([record_id, "context"]),
                       prompt_sha256=audit.canonical_hash([record_id, "prompt"]), prompt_tokens=100 + index,
                       prompt_token_ids_sha256=audit.canonical_hash([record_id, "prompt_tokens"]))
        contexts.append(context)
        block.update(context, plan_id="synthetic-plan")
        for candidate in block["candidates"]:
            candidate.update(candidate["scores"])
            candidate.update(answer_token_ids_sha256=audit.canonical_hash(candidate["answer_token_ids"]),
                             prompt_token_ids_sha256=context["prompt_token_ids_sha256"], eos_token_id=999,
                             prompt_tokens=context["prompt_tokens"],
                             sequence_tokens=context["prompt_tokens"] + len(candidate["answer_token_ids"]) + 1)
            candidate["padded_sequence_tokens"] = candidate["sequence_tokens"]
    plan = {"frame": frame, "blocks": contexts, "catalog": catalog, "eos_token_id": 999}
    manifest = {"blocks": len(frame) * 16, "candidates": len(frame) * 272,
                "identity": {"plan_id": "synthetic-plan"}}
    return frame, gold, blocks, plan, manifest


def sealed_metadata():
    plan = {"plan_id": "synthetic-plan", "numeric_policy": dict(audit.EXPECTED_POLICY)}
    gate = {"schema_version": "general-model-coverage-preflight/v1", "plan_id": plan["plan_id"],
            "passed": True, "complete": True, "runtime_identity": {"synthetic": True},
            "E8": audit.EXPECTED_E8, "epsilon": audit.EXPECTED_EPSILON,
            "numeric_policy": dict(audit.EXPECTED_POLICY),
            "calibration_mode": "inherited-fixed-tolerance-new-input-revalidation",
            "error_families": list(audit.CHALLENGES),
            "not_applicable": ["batch-four", "tail-size-two", "within-batch-row-position"],
            "query_gold_loaded": False, "test_content_read": False,
            "scientific_effect_checked": False, "formal_test_authorized": False}
    terminal = {"schema_version": "general-model-coverage-run/v1", "plan_id": plan["plan_id"],
                "status": "complete", "raw_path": "dev-b1", "production_batch_size": 1,
                "numeric_policy": dict(audit.EXPECTED_POLICY), "full_dev_started": True,
                "analysis_published": True, "query_gold_loaded_during_scoring": False,
                "test_content_read": False, "automatic_profile_search": False}
    manifest = {"status": "complete", "blocks": 10288, "candidates": 174896,
                "query_gold_loaded": False, "test_content_read": False, "mixed_execution_modes": False,
                "identity": {"plan_id": plan["plan_id"], "batch_size": 1, "reference": False,
                             "pass_name": "dev-b1", "runtime": gate["runtime_identity"],
                             "scoring_profile": {"candidate_permutation": "canonical", "padding_extra": 0,
                                                 "prefix": False, "replica_shift": 0}}}
    return plan, terminal, manifest, gate


def geometry_fixture(*, cohort="boundary", suffix="r0"):
    frame, _, blocks, plan, _ = raw_fixture()
    plan["cohorts"] = {name: [frame[index]["query_id"]]
                       for index, name in enumerate(("regression", "validation", "boundary"))}
    query_ids = {row["query_id"] for row in frame} if cohort == "dev" else set(plan["cohorts"][cohort])
    contexts = [row for row in plan["blocks"] if row["query_id"] in query_ids]
    rows = [row for row in blocks if row["query_id"] in query_ids]
    pass_name = "dev-b1" if cohort == "dev" else f"{cohort}-b1-{suffix}"
    profile = {"candidate_permutation": "group-rotate-one-then-reverse-hate-reverse" if suffix == "members" else "canonical",
               "padding_extra": 64 if suffix == "padding" else 0,
               "prefix": suffix == "prefix", "replica_shift": 1 if suffix == "replica" else 0}
    identity = {"pass_name": pass_name, "records": [row["record_id"] for row in contexts],
                "runtime": {"device_indices": [0, 1, 2, 3]}, "batch_size": 1,
                "scoring_profile": profile, "reference": cohort != "dev" and suffix == "r0"}
    ordinals = [0, 0, 0, 0]
    for index, row in enumerate(rows):
        physical = (index + profile["replica_shift"]) % 4
        row.update(execution_batch_size=1, pass_name=pass_name, repetition=int(suffix == "r1"),
                   cohort=cohort if cohort in ("regression", "validation") else "dev")
        execution = list(row["candidates"])
        if suffix == "members":
            execution = list(reversed(execution if row["task"] == "hate" else execution[1:] + execution[:1]))
        prefixes = {tuple(candidate["answer_token_ids"][:position]) for candidate in row["candidates"]
                    for position in range(len(candidate["answer_token_ids"]) + 1)}
        for position, candidate in enumerate(execution):
            candidate.update(answer_tokens=len(candidate["answer_token_ids"]), reference_checked=identity["reference"],
                             token_boundary_checked=True, finite_target_logits_checked=True,
                             causal_shift=1, use_cache=False, padding_side="right", prefix_reference=profile["prefix"],
                             model_logits_dtype="torch.float32", logprob_arithmetic_dtype="torch.float32",
                             physical_gpu_index=physical, effective_batch_size=1, batch_size=1,
                             batch_ordinal=0 if profile["prefix"] else ordinals[physical],
                             batch_member_ordinal=position if profile["prefix"] else 0)
            if profile["prefix"]:
                candidate.update(batch_members=[row["record_id"] + ":" + item["candidate_id"] for item in execution],
                                 padded_sequence_tokens=None, prefix_padding=False,
                                 prefix_unique_forward_count=len(prefixes), scoring_implementation="uncached-prefix-only")
            else:
                candidate.update(batch_members=[row["record_id"] + ":" + candidate["candidate_id"]],
                                 padded_sequence_tokens=candidate["sequence_tokens"] + profile["padding_extra"],
                                 scoring_implementation="full-sequence-selected-projection")
                ordinals[physical] += 1
            # Shared synthetic answer prefixes have exactly shared conditionals.
            candidate["token_logprobs"] = [-0.5 * (position + 1) for position in range(candidate["answer_tokens"])]
            candidate["scores"] = audit.recompute(candidate["token_logprobs"], -0.25 * (candidate["answer_tokens"] + 1))
            candidate.update(candidate["scores"])
    return rows, contexts, identity, plan


class LegacyBoundaryGeometryTests(unittest.TestCase):
    def test_legacy_helper_remains_unmodified_and_reproduces_the_boundary_false_rejection(self):
        rows, contexts, identity, plan = geometry_fixture()
        self.assertIsNot(audit.validate_geometry, frozen_geometry)
        with self.assertRaises(AssertionError):
            frozen_geometry(rows, contexts, identity, plan)
        self.assertTrue(audit.validate_geometry(rows, contexts, identity, plan)["legacy_boundary_cohort_metadata_accepted"])

    def test_all_six_boundary_passes_accept_dev_metadata_without_mutating_raw(self):
        for _, suffix in audit.PASSES:
            rows, contexts, identity, plan = geometry_fixture(suffix=suffix)
            original = copy.deepcopy(rows)
            with self.subTest(suffix=suffix):
                result = audit.validate_geometry(rows, contexts, identity, plan)
                self.assertEqual(result["candidates"], 272)
                self.assertEqual(result["registered_pass_cohort"], "boundary")
                self.assertEqual(result["raw_cohort_metadata"], "dev")
                self.assertTrue(result["record_matrix_matches_registered_cohort"])
                self.assertTrue(result["legacy_boundary_cohort_metadata_accepted"])
                self.assertEqual(rows, original)

    def test_regression_validation_and_full_dev_keep_exact_registered_metadata(self):
        for cohort in ("regression", "validation", "dev"):
            rows, contexts, identity, plan = geometry_fixture(cohort=cohort)
            with self.subTest(cohort=cohort):
                result = audit.validate_geometry(rows, contexts, identity, plan)
                self.assertEqual(result["raw_cohort_metadata"], cohort)
                self.assertFalse(result["legacy_boundary_cohort_metadata_accepted"])

    def test_boundary_or_original_cohort_mislabel_is_rejected(self):
        for cohort, incorrect in (("boundary", "boundary"), ("boundary", "validation"),
                                  ("regression", "dev"), ("validation", "dev"), ("dev", "boundary")):
            rows, contexts, identity, plan = geometry_fixture(cohort=cohort)
            rows[0]["cohort"] = incorrect
            with self.subTest(cohort=cohort, incorrect=incorrect), self.assertRaises(AssertionError):
                audit.validate_geometry(rows, contexts, identity, plan)

    def test_boundary_cannot_borrow_another_registered_cohorts_records(self):
        rows, contexts, identity, plan = geometry_fixture(cohort="regression")
        identity["pass_name"] = "boundary-b1-r0"
        for row in rows:
            row.update(pass_name=identity["pass_name"], cohort="dev")
        with self.assertRaises(AssertionError):
            audit.validate_geometry(rows, contexts, identity, plan)

    def test_unknown_pass_or_incomplete_record_matrix_is_rejected(self):
        for pass_name in ("unknown-b1-r0", "boundary-b4-r0", "boundary-b1-unknown", "dev-b1-r0"):
            rows, contexts, identity, plan = geometry_fixture()
            identity["pass_name"] = pass_name
            for row in rows:
                row["pass_name"] = pass_name
            with self.subTest(pass_name=pass_name), self.assertRaises(AssertionError):
                audit.validate_geometry(rows, contexts, identity, plan)
        rows, contexts, identity, plan = geometry_fixture()
        identity["records"] = identity["records"][:-1]
        with self.assertRaises(AssertionError):
            audit.validate_geometry(rows[:-1], contexts[:-1], identity, plan)

    def test_boundary_compatibility_does_not_relax_numerical_or_batch_geometry_checks(self):
        for field, value in (("padded_sequence_tokens", 8193), ("effective_batch_size", 4),
                             ("batch_members", ["different-record:candidate"]), ("eos_logprob", -99)):
            rows, contexts, identity, plan = geometry_fixture()
            rows[0]["candidates"][0][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit.validate_geometry(rows, contexts, identity, plan)


class IndependentRawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.gold, cls.blocks, cls.plan, cls.manifest = raw_fixture()

    def validate(self, rows=None, plan=None, manifest=None):
        return audit._raw_validation(self.blocks if rows is None else rows,
                                     self.plan if plan is None else plan,
                                     self.manifest if manifest is None else manifest,
                                     expected_query_count=len(self.frame))

    def test_all_eight_conditions_candidate_tokens_eos_and_cardinality_evidence(self):
        margins, candidates, eos, size, counts = self.validate()
        self.assertEqual(len(margins), 64)
        self.assertEqual(len(candidates), 64)
        self.assertEqual(counts["candidates"], 1088)
        self.assertEqual(len(eos), 1088)
        self.assertEqual(len(size), 4 * 8 * 4 * 6)
        for row in size:
            self.assertAlmostEqual(row["logsumexp_evidence"] - row["logmeanexp_evidence"],
                                   np.log(row["candidate_count"]), places=12)

    def test_candidate_raw_score_or_eos_corruption_is_rejected(self):
        for field in ("answer_sum", "answer_mean", "eos_logprob", "total_with_eos", "mean_with_eos"):
            rows = copy.deepcopy(self.blocks)
            rows[0]["candidates"][0]["scores"][field] += 0.01
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.validate(rows)

    def test_candidate_prompt_and_canonical_identity_corruption_is_rejected(self):
        for field, value in (("candidate_id", "unregistered"), ("labels", ["non-hate"]),
                             ("canonical_answer", '"non-hate"'), ("answer_token_ids", [888]),
                             ("prompt_tokens", 1), ("eos_token_id", 1),
                             ("prompt_token_ids_sha256", "wrong"), ("padded_sequence_tokens", 9000)):
            rows = copy.deepcopy(self.blocks)
            rows[0]["candidates"][0][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.validate(rows)

    def test_incomplete_duplicate_or_legacy_six_condition_frame_is_rejected(self):
        for rows in (self.blocks[:-1], self.blocks[:-1] + self.blocks[:1],
                     [r for r in self.blocks if r["condition"] not in ("CLq", "CLqD")]):
            with self.subTest(length=len(rows)), self.assertRaises(AssertionError):
                self.validate(rows)


class IndependentAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.gold, cls.blocks, cls.plan, cls.manifest = raw_fixture()
        cls.result = production.analyze_blocks(cls.blocks, frame=cls.frame, gold_by_query=cls.gold,
                                               epsilon=audit.EXPECTED_EPSILON, expected_query_count=4)
        cls.margins, cls.candidates, *_ = audit._raw_validation(
            cls.blocks, cls.plan, cls.manifest, expected_query_count=4)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.table, cls.verification = audit.verify_analysis(
                cls.result, cls.frame, cls.margins, cls.candidates, cls.gold,
                audit.EXPECTED_EPSILON, expected_query_count=4)

    def test_all_240_targets_six_ci_strata_and_all_60480_auxiliary_summaries(self):
        self.assertEqual(len(self.table), 1440)
        self.assertTrue(all(row["ci_independently_recomputed"] for row in self.table))
        self.assertEqual(self.verification["independently_recomputed_ci_endpoints_per_stratum"], 240)
        self.assertEqual(self.verification["independently_recomputed_ci_strata"], 6)
        self.assertLess(self.verification["max_ci_endpoint_discrepancy"], 1e-12)
        self.assertEqual(self.verification["all_descriptive_summaries_verified"],
                         {"condition": 26880, "contrast": 33600})
        self.assertEqual(self.verification["all_paired_readouts_verified"], 4 * 10 * 420)
        self.assertEqual({row["contrast"] for row in self.table}, set(audit.CONTRASTS))
        self.assertEqual({row["stratum"] for row in self.table},
                         {"all", "Lq_hit", "Lq_no_hit", "gold_size_0", "gold_size_1", "gold_size_2plus"})

    def test_expected_statistics_never_call_production_analyzer(self):
        with mock.patch.object(production, "analyze_blocks", side_effect=AssertionError("must be independent")):
            audit._verify_gold_metrics({r["query_id"]: r for r in self.result["per_query"]},
                                       self.candidates, self.gold)
            audit._verify_distribution_metrics({r["query_id"]: r for r in self.result["per_query"]},
                                               self.candidates)
            audit.sample_ci(np.array([[1.0, 2.0], [3.0, 5.0]]), np.array([True, True]),
                            stratum="synthetic", replicates=100)

    def test_reference_difference_and_gold_auxiliary_corruption_is_rejected(self):
        for metric in ("answer_sum/margin/hate", "answer_mean/gold/nll"):
            result = copy.deepcopy(self.result)
            result["per_query"][0]["tasks"]["hate"]["contrasts"]["CLnew_minus_CLq"][metric] += 0.1
            with self.subTest(metric=metric), self.assertRaises(AssertionError):
                audit.verify_all_descriptive_summaries(result, self.frame, self.gold, audit.EXPECTED_EPSILON)
        result = copy.deepcopy(self.result)
        result["per_query"][0]["tasks"]["group"]["conditions"]["CLDnew"]["answer_sum/gold/toggle_margin/Racism"] += 0.1
        with self.assertRaises(AssertionError):
            audit._verify_gold_metrics({r["query_id"]: r for r in result["per_query"]}, self.candidates, self.gold)

    def test_size_evidence_and_undefined_endpoint_corruption_is_rejected(self):
        for metric, value in (("answer_sum/cardinality/equal_k/expected_size", -1),
                             ("answer_mean/within_k/natural/Racism/0", 0)):
            result = copy.deepcopy(self.result)
            result["per_query"][0]["tasks"]["group"]["conditions"]["CLnew"][metric] = value
            with self.subTest(metric=metric), self.assertRaises(AssertionError):
                audit._verify_distribution_metrics({r["query_id"]: r for r in result["per_query"]}, self.candidates)

    def test_registered_reference_ci_endpoint_corruption_is_rejected(self):
        result = copy.deepcopy(self.result)
        row = next(row for row in result["contrast_summaries"] if row["contrast"] == "CLDnew_minus_CLqD"
                   and row["metric"] == "answer_sum/margin/hate" and row["stratum"] == "all")
        row["descriptive_ci95"][0] += 0.01
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(AssertionError):
            audit.verify_analysis(result, self.frame, self.margins, self.candidates,
                                  self.gold, audit.EXPECTED_EPSILON, expected_query_count=4)


class EmptyResourceTests(unittest.TestCase):
    def test_three_empty_lnew_queries_have_zero_lexical_effect_and_empty_lq_hit_ci_is_na(self):
        frame, gold, blocks, plan, manifest = raw_fixture(empty_resources=True)
        result = production.analyze_blocks(blocks, frame=frame, gold_by_query=gold,
                                           epsilon=audit.EXPECTED_EPSILON, expected_query_count=3)
        margins, candidates, *_ = audit._raw_validation(blocks, plan, manifest, expected_query_count=3)
        with contextlib.redirect_stdout(io.StringIO()):
            table, verification = audit.verify_analysis(result, frame, margins, candidates, gold,
                                                        audit.EXPECTED_EPSILON, expected_query_count=3)
        empty = [row for row in table if row["stratum"] == "Lq_hit"]
        self.assertEqual(len(empty), 240)
        self.assertTrue(all(row["n"] == 0 and row["mean"] is None and row["ci95_low"] is None for row in empty))
        self.assertTrue(all(row["ci_independently_recomputed"] for row in empty))
        lexical = {"L", "LxD", "L_given_D", "CL_minus_PL", *audit.REFERENCES}
        self.assertTrue(all(row["mean"] == 0 and row["ci95_low"] == row["ci95_high"] == 0
                            for row in table if row["n"] and row["contrast"] in lexical))
        self.assertEqual(verification["independently_recomputed_ci_endpoints_per_stratum"], 240)


class GoldSealTests(unittest.TestCase):
    def test_complete_seal_permits_explicit_gold_audit(self):
        audit.require_scoring_seal(*sealed_metadata(), allow_gold=True)

    def test_raw_only_mode_accepts_raw_complete_but_gold_requires_published_analysis(self):
        plan, terminal, manifest, gate = sealed_metadata()
        terminal.update(status="raw_complete", analysis_published=False)
        audit.require_scoring_seal(plan, terminal, manifest, gate, allow_gold=False)
        with self.assertRaises(AssertionError):
            audit.require_scoring_seal(plan, terminal, manifest, gate, allow_gold=True)

    def test_failed_gate_incomplete_raw_batch4_or_changed_epsilon_stops_before_gold(self):
        mutations = ((1, "status", "running"), (1, "full_dev_started", False),
                     (1, "query_gold_loaded_during_scoring", True), (1, "analysis_published", False),
                     (2, "status", "partial"), (2, "blocks", 10287), (2, "candidates", 174895),
                     (3, "passed", False), (3, "complete", False), (3, "epsilon", 0.005),
                     (3, "error_families", ["batch", "tail"]), (1, "production_batch_size", 4))
        for index, field, value in mutations:
            parts = sealed_metadata()
            parts[index][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(AssertionError):
                audit.require_scoring_seal(*parts, allow_gold=True)

    def test_fully_consistent_but_recalibrated_policy_is_still_rejected(self):
        plan, terminal, manifest, gate = sealed_metadata()
        for metadata in (plan, terminal, gate):
            metadata["numeric_policy"]["epsilon"] = 0.005
        gate["epsilon"] = 0.005
        with self.assertRaises(AssertionError):
            audit.require_scoring_seal(plan, terminal, manifest, gate, allow_gold=True)


class PreflightOrchestrationTests(unittest.TestCase):
    """Real independent readout math; replica/geometry helpers tested separately."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        frame, _, blocks, self.plan, _ = raw_fixture()
        self.plan.update(plan_id="synthetic-plan", cohorts={name: [frame[index]["query_id"]]
                         for index, name in enumerate(("regression", "validation", "boundary"))})
        self.report = {"runtime_identity": {"synthetic": True}, "observed_max_abs_error": 0.0, "cohorts": {}}
        for cohort, ids in self.plan["cohorts"].items():
            rows = [row for row in blocks if row["query_id"] in ids]
            self.report["cohorts"][cohort] = {
                "blocks": 16, "passed": True, "complete": True, "baseline_passed": True,
                "reference_max_abs_error": 0.0, "baseline_repeat_max_abs_error": 0.0,
                "challenges": {name: {"passed": True, "max_abs_error": 0.0} for name in audit.CHALLENGES}}
            differences = {"blocks": [{"record_id": row["record_id"],
                            "differences": {key: 0.0 for key in flat_readouts(row)}}
                           for row in rows], "max_abs_error": 0.0, "largest_error": None}
            for name, suffix in audit.PASSES:
                directory = self.run / "preflight" / f"{cohort}-b1-{suffix}"
                directory.mkdir(parents=True)
                profile = {"candidate_permutation": "group-rotate-one-then-reverse-hate-reverse" if name == "members" else "canonical",
                           "padding_extra": 64 if name == "padding" else 0,
                           "prefix": name == "prefix", "replica_shift": 1 if name == "replica" else 0}
                identity = {"plan_id": self.plan["plan_id"], "pass_name": directory.name,
                            "runtime": self.report["runtime_identity"], "batch_size": 1,
                            "reference": name == "baseline", "scoring_profile": profile}
                stored = copy.deepcopy(rows)
                if name == "baseline":
                    for row in stored:
                        for candidate in row["candidates"]:
                            candidate["reference_scores"] = {**candidate["scores"],
                                                              "token_logprobs": list(candidate["token_logprobs"])}
                with (directory / "scores.jsonl").open("w") as handle:
                    handle.writelines(json.dumps(row) + "\n" for row in stored)
                manifest = {"identity": identity, "status": "complete", "blocks": 16, "candidates": 272,
                            "query_gold_loaded": False, "test_content_read": False, "mixed_execution_modes": False,
                            "scores_sha256": audit.digest(directory / "scores.jsonl")}
                self.write_json(directory / "manifest.json", manifest)
                self.write_json(directory.parent / f"{cohort}-{name}-geometry-proof.json", {
                    "passed": True, "blocks": 16, "candidates": 272, "scoring_profile": profile,
                    "true_batch_one": True, "prefix_is_reference_only": name == "prefix",
                    "execution_order_verified": name != "prefix", "within_batch_row_position_claimed": False})
                difference_name = "reference" if name == "baseline" else name
                self.write_json(directory.parent / f"{cohort}-{difference_name}-differences.json", differences)

    @staticmethod
    def write_json(path, value):
        with path.open("w") as handle:
            json.dump(value, handle)

    def run_audit(self):
        with mock.patch.object(audit, "audit_parallel_pass", return_value={"physical_gpu_uuid_by_index": {0: "GPU-a", 1: "GPU-b"}}), \
                mock.patch.object(audit, "validate_geometry", return_value={"candidates": 272}), \
                mock.patch.object(audit, "validate_cross_gpu_challenge", return_value={"all_candidates_changed_physical_gpu": True}):
            return audit.independent_preflight(self.plan, self.run, self.report)

    def test_all_three_cohorts_six_passes_are_numerically_recomputed_without_recalibration(self):
        result = self.run_audit()
        self.assertEqual(result["sealed_pass_count"], 18)
        self.assertEqual(result["epsilon"], audit.EXPECTED_EPSILON)
        self.assertFalse(result["epsilon_recalibrated"])
        for cohort in result["cohorts"].values():
            self.assertEqual(set(cohort["comparisons"]), {"reference", "repeat", *audit.CHALLENGES})
            self.assertTrue(all(comparison["readouts"] > 0 and comparison["stored_difference_file_verified"]
                                for comparison in cohort["comparisons"].values()))

    def test_empty_boundary_skips_only_its_six_passes(self):
        self.plan["cohorts"]["boundary"] = []
        self.report["cohorts"]["boundary"] = {"blocks": 0, "skipped": True, "reason": "no-new-boundary-query"}
        result = self.run_audit()
        self.assertEqual(result["sealed_pass_count"], 12)
        self.assertTrue(result["cohorts"]["boundary"]["skipped"])

    def test_missing_or_numerically_corrupt_difference_evidence_is_rejected(self):
        path = self.run / "preflight/regression-padding-differences.json"
        recorded = audit.read(path)
        key = next(iter(recorded["blocks"][0]["differences"]))
        recorded["blocks"][0]["differences"][key] = 0.01
        self.write_json(path, recorded)
        with self.assertRaises(AssertionError):
            self.run_audit()
        path.unlink()
        with self.assertRaises(AssertionError):
            self.run_audit()

    def test_changed_raw_file_hash_or_incomplete_pass_is_rejected(self):
        path = self.run / "preflight/regression-b1-r0/manifest.json"
        manifest = audit.read(path)
        manifest["scores_sha256"] = "wrong"
        self.write_json(path, manifest)
        with self.assertRaises(AssertionError):
            self.run_audit()
        manifest["status"] = "partial"
        self.write_json(path, manifest)
        with self.assertRaises(AssertionError):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
