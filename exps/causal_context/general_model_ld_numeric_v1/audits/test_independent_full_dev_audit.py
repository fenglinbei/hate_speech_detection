"""Synthetic tests only; no dataset or run payload is opened."""

import math
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from independent_full_dev_audit import (
    CONDITIONS, CONTRASTS, LABELS, MODES, _verify_distribution_metrics,
    expected_catalog, load_effective_plan, margin_values, paired, sample_ci,
    verify_all_descriptive_summaries,
)
from independent_numeric_audit import canonical_hash, digest, recompute


class IndependentFullDevAuditTests(unittest.TestCase):
    def test_parent_semantic_hash_is_distinct_from_serialized_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir, child_dir = root / "parent", root / "child"
            parent_dir.mkdir()
            child_dir.mkdir()
            for directory in (parent_dir, child_dir):
                (directory / "protocol.md").write_text("Synthetic protocol\n")
            scientific = {"model_key": "synthetic", "tasks": ["hate", "group"], "conditions": list(CONDITIONS),
                          "candidate_order": {}, "analysis": {}}
            parent = {"schema_version": "general-model-ld-numeric-plan/v1", "config": scientific,
                      "frame": [{"query_id": "synthetic", "lex_hit": False}], "code_sha256": {},
                      "protocol_sha256": digest(parent_dir / "protocol.md")}
            parent["plan_id"] = "gmlnum-" + canonical_hash(parent)
            (parent_dir / "plan.json").write_text(json.dumps(parent, indent=2) + "\n")
            parent_ref = root / "parent_ref.json"
            parent_ref.write_text(json.dumps({"plan_id": parent["plan_id"], "target_path": str(parent_dir),
                                              "plan_sha256": digest(parent_dir / "plan.json")}) + "\n")
            child = {"schema_version": "general-model-ld-numeric-plan/v2", "config": scientific,
                     "parent_plan_ref": str(parent_ref), "parent_ref_sha256": digest(parent_ref),
                     "parent_plan_id": parent["plan_id"], "parent_plan_sha256": canonical_hash(parent),
                     "scientific_scope_sha256": canonical_hash(scientific), "code_sha256": {},
                     "protocol_sha256": digest(child_dir / "protocol.md")}
            child["plan_id"] = "gmlnum2-" + canonical_hash(child)
            (child_dir / "plan.json").write_text(json.dumps(child, indent=2) + "\n")
            effective = load_effective_plan(child_dir, root=root)
            self.assertEqual(effective["frame"], parent["frame"])
            self.assertEqual(effective["parent_plan"], parent)
            child.pop("plan_id")
            child["parent_plan_sha256"] = digest(parent_dir / "plan.json")
            self.assertNotEqual(child["parent_plan_sha256"], canonical_hash(parent))
            child["plan_id"] = "gmlnum2-" + canonical_hash(child)
            (child_dir / "plan.json").write_text(json.dumps(child) + "\n")
            with self.assertRaises(AssertionError):
                load_effective_plan(child_dir, root=root)

    def test_factorized_32_set_marginals_and_exact_contrasts(self):
        factors = [0.2, -0.3, 0.5, -0.7, 1.0]
        candidates = expected_catalog("group")
        for candidate in candidates:
            score = -50 + sum(factors[bit] for bit in range(5) if candidate["ordinal"] & (1 << bit))
            candidate["scores"] = {mode: score for mode in MODES}
        actual = margin_values("group", candidates)
        for mode in MODES:
            for label, factor in zip(LABELS, factors, strict=True):
                self.assertAlmostEqual(actual[(mode, label)], factor)
        cells = dict(zip(CONDITIONS, [1, 3, 4, 10, 2, 2.5], strict=True))
        self.assertEqual({key: paired(cells, key) for key in CONTRASTS},
                         {"L": 2, "D": 3, "LD": 9, "LxD": 4, "L_given_D": 6,
                          "D_given_L": 7, "CL_minus_PL": 1, "CD_minus_PD": 1.5})

    def test_direct_bootstrap_is_joint_deterministic_and_preserves_constants(self):
        matrix = np.asarray([[1, 2, 7], [2, 4, 7], [5, 10, 7], [9, 18, 7]], dtype=float)
        first = sample_ci(matrix, [True] * 4, stratum="synthetic", replicates=500)
        second = sample_ci(matrix, [True] * 4, stratum="synthetic", replicates=500)
        self.assertEqual(first, second)
        np.testing.assert_allclose(first[1], 2 * np.asarray(first[0]))
        self.assertEqual(first[2], [7, 7])
        self.assertEqual(sample_ci(matrix, [False] * 4, stratum="empty", replicates=10), [None] * 3)

    def test_distribution_checks_cover_combinatorics_and_undefined_edges(self):
        candidates = expected_catalog("group")
        for candidate in candidates:
            candidate["scores"] = {mode: -10.0 for mode in MODES}
        recorded = {}
        for mode in MODES:
            prefix = mode + "/"
            recorded[prefix + "candidate_entropy"] = math.log(32)
            for label in LABELS:
                recorded[prefix + "inclusion_mass/" + label] = .5
                for size in range(6):
                    recorded[prefix + f"within_k/natural/{label}/{size}"] = None if size in (0, 5) else math.log(size / (5 - size))
                    recorded[prefix + f"within_k/count_corrected/{label}/{size}"] = None if size in (0, 5) else 0
            for weighting in ("natural", "equal_k"):
                probabilities = [math.comb(5, size) / 32 if weighting == "natural" else 1 / 6 for size in range(6)]
                for size, value in enumerate(probabilities):
                    recorded[prefix + f"cardinality/{weighting}/p{size}"] = value
                recorded[prefix + f"cardinality/{weighting}/expected_size"] = 2.5
                recorded[prefix + f"cardinality/{weighting}/entropy"] = -sum(value * math.log(value) for value in probabilities)
        query = {"synthetic": {"tasks": {"group": {"conditions": {"C0": recorded}}}}}
        self.assertEqual(_verify_distribution_metrics(query, {("synthetic", "group", "C0"): candidates}), 328)
        recorded["answer_sum/within_k/natural/Racism/0"] = 0
        with self.assertRaises(AssertionError):
            _verify_distribution_metrics(query, {("synthetic", "group", "C0"): candidates})


class IndependentAuxiliarySummaryAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from diagnostics.general_model_numeric_analysis import analyze_blocks

        cls.frame = [{"query_id": "synthetic-0", "lex_hit": False}, {"query_id": "synthetic-1", "lex_hit": True}]
        cls.gold = {"synthetic-0": {"hate": "non-hate", "group": []},
                    "synthetic-1": {"hate": "hate", "group": ["Racism"]}}
        blocks = []
        for query_index, query in enumerate(cls.frame):
            for task in ("hate", "group"):
                for condition_index, condition in enumerate(CONDITIONS):
                    candidates = expected_catalog(task)
                    for candidate in candidates:
                        tokens = [-1.0 - .03 * candidate["ordinal"] - .01 * condition_index * (candidate["ordinal"] % 3),
                                  -.05 - .01 * query_index]
                        candidate.update(answer_token_ids=[1, 2], token_logprobs=tokens,
                                         scores=recompute(tokens, -.01))
                    blocks.append({"query_id": query["query_id"], "task": task, "condition": condition, "candidates": candidates})
        cls.analysis = analyze_blocks(blocks, frame=cls.frame, gold_by_query=cls.gold, epsilon=.001,
                                      expected_query_count=2, bootstrap_replicates=5, bootstrap_seed=42)

    def test_all_auxiliary_pairs_and_summaries_with_na_pass(self):
        result = verify_all_descriptive_summaries(self.analysis, self.frame, self.gold, .001)
        self.assertEqual(result["all_paired_readouts_verified"], 2 * 8 * 420)
        self.assertEqual(result["all_descriptive_summaries_verified"], {"condition": 20160, "contrast": 26880})
        self.assertGreater(result["undefined_descriptive_summaries_verified"]["condition"], 0)

    def test_auxiliary_paired_gold_value_tampering_is_rejected(self):
        changed = copy.deepcopy(self.analysis)
        changed["per_query"][0]["tasks"]["hate"]["contrasts"]["L"]["answer_sum/gold/nll"] += .01
        with self.assertRaises(AssertionError):
            verify_all_descriptive_summaries(changed, self.frame, self.gold, .001)

    def test_auxiliary_summary_mean_tampering_is_rejected(self):
        changed = copy.deepcopy(self.analysis)
        row = next(row for row in changed["condition_summaries"] if row["metric"] == "answer_sum/candidate_entropy" and row["stratum"] == "all")
        row["mean"] += .01
        with self.assertRaises(AssertionError):
            verify_all_descriptive_summaries(changed, self.frame, self.gold, .001)

    def test_empty_stratum_ci_cannot_be_replaced_with_zero_interval(self):
        changed = copy.deepcopy(self.analysis)
        row = next(row for row in changed["contrast_summaries"] if row["metric"] == "answer_sum/margin/hate" and row["stratum"] == "gold_size_2plus")
        self.assertIsNone(row["descriptive_ci95"])
        row["descriptive_ci95"] = [0.0, 0.0]
        with self.assertRaises(AssertionError):
            verify_all_descriptive_summaries(changed, self.frame, self.gold, .001)

    def test_unknown_metric_outside_registered_schema_is_rejected(self):
        changed = copy.deepcopy(self.analysis)
        changed["per_query"][0]["tasks"]["hate"]["conditions"]["C0"]["answer_sum/unregistered"] = 0
        with self.assertRaises(AssertionError):
            verify_all_descriptive_summaries(changed, self.frame, self.gold, .001)


if __name__ == "__main__":
    unittest.main()
