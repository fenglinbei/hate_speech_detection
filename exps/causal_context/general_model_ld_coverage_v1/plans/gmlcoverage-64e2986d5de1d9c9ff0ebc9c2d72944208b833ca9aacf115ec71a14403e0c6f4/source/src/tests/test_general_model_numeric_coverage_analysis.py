import copy
import hashlib
import inspect
import json
import math
import unittest
from collections import Counter

import numpy as np

from diagnostics import general_model_numeric_analysis as frozen
from diagnostics import general_model_numeric_coverage_analysis as coverage


def candidates(task, strength=0.0):
    result = coverage.candidate_catalog()[task]
    for candidate in result:
        ordinal = candidate["ordinal"]
        length = 1 + ordinal % 3
        preference = float(ordinal == 0) if task == "hate" else (
            float(bool(ordinal & 1)) + 0.5 * float(bool(ordinal & 8)))
        score = -40 + ordinal * 0.1 + strength * preference
        values = [score / length] * length
        eos = -0.25 - ordinal * 0.03 + strength * 0.01 * (ordinal % 2)
        candidate.update(answer_token_ids=list(range(1, length + 1)),
                         token_logprobs=values, scores=coverage.candidate_scores(values, eos))
    return result


def fixture(*, empty_resources=False):
    count = 3 if empty_resources else 4
    sizes = (0, 1, 2) if empty_resources else (2, 0, 1, 3)
    frame = [{"query_id": f"synthetic-{index}", "lex_hit": False if empty_resources else index % 2 == 0}
             for index in range(count)]
    gold = {row["query_id"]: {"hate": "hate" if index % 2 == 0 else "non-hate",
                              "group": list(coverage.GROUP_LABELS[:sizes[index]])}
            for index, row in enumerate(frame)}
    blocks = []
    for index, row in enumerate(frame):
        strengths = {"C0": 0, "CLnew": 1, "CD": 2, "CLDnew": 2.5,
                     "PLnew": 0.25, "PD": 0.5, "CLq": 0.5, "CLqD": 2.2}
        if not row["lex_hit"]:
            strengths.update(CLq=0, CLqD=2)
        if empty_resources:
            strengths.update(CLnew=0, CLDnew=2, PLnew=0)
        for task in coverage.TASKS:
            for condition in coverage.CONDITIONS:
                blocks.append({"query_id": row["query_id"], "task": task, "condition": condition,
                               "candidates": candidates(task, strengths[condition] * (index + 1))})
    return frame, gold, blocks


class FrozenMathReuseTests(unittest.TestCase):
    def test_all_core_math_and_bootstrap_are_unmodified_frozen_functions(self):
        for name in ("candidate_catalog", "candidate_scores", "block_readouts", "block_analysis",
                     "logsumexp", "joint_bootstrap_mean_ci"):
            with self.subTest(name=name):
                self.assertIs(getattr(coverage, name), getattr(frozen, name))
        self.assertEqual(frozen.CONDITIONS, ("C0", "CL", "CD", "CLD", "PL", "PD"))
        self.assertEqual(len(frozen.CONTRASTS), 8)
        self.assertEqual(coverage.GROUP_LABELS, frozen.GROUP_LABELS)
        self.assertEqual(coverage.SCORE_MODES, frozen.SCORE_MODES)

    def test_catalog_eos_raw_tokens_and_all_auxiliary_readouts_are_preserved(self):
        self.assertEqual(coverage.candidate_catalog(), frozen.candidate_catalog())
        self.assertEqual(coverage.candidate_scores([-1, -2, -3], -4),
                         {"answer_sum": -6, "answer_mean": -2, "eos_logprob": -4,
                          "total_with_eos": -10, "mean_with_eos": -2.5})
        for task, gold, metric_count in (("hate", "hate", 32), ("group", ["Racism", "Sexism"], 388)):
            values = candidates(task, 0.7)
            before = copy.deepcopy(values)
            with self.subTest(task=task):
                self.assertEqual(coverage.block_readouts(task, values), frozen.block_readouts(task, values))
                metrics = coverage.block_analysis(task, values, gold=gold)
                self.assertEqual(metrics, frozen.block_analysis(task, values, gold=gold))
                self.assertEqual(len(metrics), metric_count)
                self.assertFalse(any("gold/" in key for key in coverage.block_analysis(task, values)))
                self.assertEqual(values, before)
                for mode in coverage.SCORE_MODES:
                    self.assertIn(f"{mode}/gold/tie_count", metrics)
                    self.assertIn(f"{mode}/gold/nll", metrics)
                    if task == "group":
                        for weighting in ("natural", "equal_k"):
                            self.assertIn(f"{mode}/cardinality/{weighting}/entropy", metrics)
                            self.assertIn(f"{mode}/cardinality/{weighting}/expected_size", metrics)
                        for label in coverage.GROUP_LABELS:
                            self.assertIn(f"{mode}/gold/toggle_margin/{label}", metrics)
                            for size in (0, 5):
                                self.assertIsNone(metrics[f"{mode}/within_k/natural/{label}/{size}"])
                                self.assertIsNone(metrics[f"{mode}/within_k/count_corrected/{label}/{size}"])

    def test_analyze_signature_preserves_registered_defaults(self):
        self.assertEqual(inspect.signature(coverage.analyze_blocks), inspect.signature(frozen.analyze_blocks))


class CoverageContrastTests(unittest.TestCase):
    def test_exact_eight_conditions_and_ten_registered_contrasts(self):
        self.assertEqual(coverage.CONDITIONS,
                         ("C0", "CLnew", "CD", "CLDnew", "PLnew", "PD", "CLq", "CLqD"))
        result = coverage.paired_contrasts({"C0": 1, "CLnew": 3, "CD": 4, "CLDnew": 10,
                                           "PLnew": 2, "PD": 2.5, "CLq": 1.5, "CLqD": 6})
        self.assertEqual(result, {"L": 2, "D": 3, "LD": 9, "LxD": 4,
                                  "L_given_D": 6, "D_given_L": 7, "CL_minus_PL": 1,
                                  "CD_minus_PD": 1.5, "CLnew_minus_CLq": 1.5,
                                  "CLDnew_minus_CLqD": 4})
        self.assertEqual(tuple(coverage.CONTRASTS)[8:], coverage.REFERENCE_CONTRASTS)
        self.assertEqual(len(coverage.CONTRASTS), 10)

    def test_error_band_for_references_is_two_epsilon_not_interaction_band(self):
        for contrast in coverage.CONTRASTS:
            with self.subTest(contrast=contrast):
                bound = 0.04 if contrast == "LxD" else 0.02
                self.assertEqual(coverage.numerical_direction(bound, epsilon=0.01, contrast=contrast),
                                 {"raw_sign": 1, "direction": "numerically_unresolved", "error_bound": bound})
                self.assertEqual(coverage.numerical_direction(-bound - 0.001, epsilon=0.01,
                                                              contrast=contrast)["direction"], "negative")
        for epsilon, contrast in ((-1, "L"), (math.nan, "L"), (0.1, "unregistered")):
            with self.subTest(epsilon=epsilon, contrast=contrast), self.assertRaises(ValueError):
                coverage.numerical_direction(1, epsilon=epsilon, contrast=contrast)

    def test_legacy_missing_extra_and_nonfinite_cells_are_rejected(self):
        cells = {condition: 0.0 for condition in coverage.CONDITIONS}
        for invalid in ({condition: 0 for condition in frozen.CONDITIONS},
                        {key: value for key, value in cells.items() if key != "CLq"},
                        {**cells, "extra": 0}, {**cells, "CLqD": math.inf}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                coverage.paired_contrasts(invalid)


class CoverageAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.gold, cls.blocks = fixture()
        cls.result = coverage.analyze_blocks(cls.blocks, frame=cls.frame, gold_by_query=cls.gold,
                                             epsilon=0.01, expected_query_count=4, bootstrap_replicates=100)
        cls.summaries = {(r["task"], r["metric"], r["contrast"], r["stratum"]): r
                         for r in cls.result["contrast_summaries"]}

    def test_complete_eight_condition_itt_and_lq_specific_strata(self):
        result = self.result
        self.assertEqual(result["query_count"], 4)
        self.assertEqual(result["block_count"], 64)
        self.assertEqual(result["conditions"], list(coverage.CONDITIONS))
        self.assertEqual(result["stratum_counts"], {"all": 4, "Lq_hit": 2, "Lq_no_hit": 2,
                                                  "gold_size_0": 1, "gold_size_1": 1,
                                                  "gold_size_2plus": 2, "gold_size_2_appendix": 1,
                                                  "gold_size_3_appendix": 1})
        nohit = result["per_query"][1]
        self.assertFalse(nohit["lex_hit"])
        self.assertGreater(nohit["tasks"]["hate"]["contrasts"]["L"]["answer_sum/margin/hate"], 0)
        self.assertIn("query Lq", result["interpretation"]["lexicon_hit_strata"])
        self.assertIn("not sequence probability", result["interpretation"]["score_normalization"]["answer_mean"])
        self.assertIn("demo-derived", result["interpretation"]["demo_ablation"])
        self.assertEqual(result["reference_contrasts"], list(coverage.REFERENCE_CONTRASTS))

    def test_240_targets_have_six_pointwise_ci_strata_and_no_extra_ci(self):
        result = self.result
        self.assertEqual(result["bootstrap"]["endpoint_count"], 240)
        ci_rows = [row for row in result["contrast_summaries"] if "descriptive_ci95" in row]
        counts = Counter(row["stratum"] for row in ci_rows)
        self.assertEqual(counts, {name: 240 for name in
                                 ("all", "Lq_hit", "Lq_no_hit", "gold_size_0", "gold_size_1", "gold_size_2plus")})
        self.assertEqual(len(ci_rows), 1440)
        main = [r for r in ci_rows if r["stratum"] == "all" and r["metric"].startswith("answer_sum/")]
        self.assertEqual(sum(r["contrast"] in coverage.PRIMARY_CONTRASTS for r in main), 24)
        self.assertEqual(sum(r["contrast"] not in coverage.PRIMARY_CONTRASTS for r in main), 36)
        self.assertEqual(sum(r["stratum"] == "all" and not r["metric"].startswith("answer_sum/")
                             for r in ci_rows), 180)
        self.assertTrue(all("/margin/" in row["metric"] for row in ci_rows))
        self.assertFalse(any("descriptive_ci95" in row for row in result["condition_summaries"]))
        self.assertNotIn("p_value", json.dumps(result, allow_nan=False))

    def test_all_readouts_and_paired_auxiliary_distributions_remain_present(self):
        self.assertEqual(len(self.result["condition_summaries"]), 8 * 420 * 8)
        self.assertEqual(len(self.result["contrast_summaries"]), 10 * 420 * 8)
        for query in self.result["per_query"]:
            for task in coverage.TASKS:
                cells = query["tasks"][task]["conditions"]
                for contrast, weights in coverage.CONTRASTS.items():
                    for metric, observed in query["tasks"][task]["contrasts"][contrast].items():
                        if cells["C0"][metric] is None:
                            self.assertIsNone(observed)
                        else:
                            self.assertEqual(observed, math.fsum(weight * cells[cell][metric]
                                                                for cell, weight in weights.items()),
                                             msg=f"{task}/{contrast}/{metric}")
        for name in coverage.REFERENCE_CONTRASTS:
            summary = self.summaries[("hate", "answer_sum/margin/hate", name, "all")]
            self.assertAlmostEqual(sum(summary[d + "_query_fraction"]
                                       for d in ("positive", "negative", "unresolved")), 1)
            self.assertAlmostEqual(summary["iqr"], summary["q75"] - summary["q25"])

    def test_joint_bootstrap_all_240_targets_matches_direct_query_resampling(self):
        endpoints = [(task, f"{mode}/margin/{label}", contrast)
                     for task in coverage.TASKS for mode in coverage.SCORE_MODES
                     for label in (("hate",) if task == "hate" else coverage.GROUP_LABELS)
                     for contrast in coverage.CONTRASTS]
        matrix = np.array([[q["tasks"][task]["contrasts"][contrast][metric]
                            for task, metric, contrast in endpoints] for q in self.result["per_query"]])
        for stratum, positions in (("all", [0, 1, 2, 3]), ("Lq_hit", [0, 2]), ("Lq_no_hit", [1, 3])):
            selected = matrix[positions]
            stream = int.from_bytes(hashlib.sha256(f"42\0{stratum}".encode()).digest(), "big")
            rng = np.random.Generator(np.random.PCG64(stream))
            indices = rng.integers(0, len(selected), size=(100, len(selected)))
            direct = np.quantile(selected[indices].mean(axis=1), [.025, .975], axis=0, method="linear").T
            stored = np.array([self.summaries[(task, metric, contrast, stratum)]["descriptive_ci95"]
                               for task, metric, contrast in endpoints])
            np.testing.assert_allclose(stored, direct, rtol=1e-12, atol=1e-12)

    def test_incomplete_duplicate_wrong_frame_and_extra_condition_fail_closed(self):
        wrong_condition = {**self.blocks[0], "condition": "CL"}
        examples = (self.blocks[:-1], [*self.blocks, self.blocks[0]],
                    [wrong_condition, *self.blocks[1:]])
        for rows in examples:
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                coverage.analyze_blocks(rows, frame=self.frame, gold_by_query=self.gold, epsilon=0.01,
                                        expected_query_count=4, bootstrap_replicates=2)
        with self.assertRaises(ValueError):
            coverage.analyze_blocks(self.blocks, frame=self.frame, gold_by_query=self.gold, epsilon=0.01)
        with self.assertRaises(ValueError):
            coverage.analyze_blocks(self.blocks, frame=self.frame, gold_by_query={}, epsilon=0.01,
                                    expected_query_count=4)
        with self.assertRaises(ValueError):
            coverage.analyze_blocks(self.blocks, frame=[{**self.frame[0], "lex_hit": None}, *self.frame[1:]],
                                    gold_by_query=self.gold, epsilon=0.01, expected_query_count=4)

    def test_result_contrast_metadata_cannot_mutate_registered_constants(self):
        weights = self.result["contrasts"]["L"]
        try:
            weights["CLnew"] = 99
            self.assertEqual(coverage.CONTRASTS["L"], {"CLnew": 1, "C0": -1})
            self.assertEqual(frozen.CONTRASTS["L"], {"CL": 1, "C0": -1})
        finally:
            weights["CLnew"] = 1


class EmptyResourceTests(unittest.TestCase):
    def test_three_empty_lnew_queries_keep_exact_zero_but_empty_strata_are_na(self):
        frame, gold, blocks = fixture(empty_resources=True)
        result = coverage.analyze_blocks(blocks, frame=frame, gold_by_query=gold, epsilon=0.01,
                                         expected_query_count=3, bootstrap_replicates=20)
        self.assertEqual(result["query_count"], 3)
        self.assertEqual(result["block_count"], 48)
        self.assertEqual(result["stratum_counts"]["Lq_hit"], 0)
        for query in result["per_query"]:
            for task in coverage.TASKS:
                cells = query["tasks"][task]["conditions"]
                self.assertEqual(cells["CLnew"], cells["C0"])
                self.assertEqual(cells["CLq"], cells["C0"])
                self.assertEqual(cells["PLnew"], cells["C0"])
                self.assertEqual(cells["CLDnew"], cells["CD"])
                self.assertEqual(cells["CLqD"], cells["CD"])
        for summary in result["contrast_summaries"]:
            if "descriptive_ci95" not in summary:
                continue
            if summary["stratum"] == "Lq_hit":
                self.assertEqual(summary["status"], "undefined")
                self.assertEqual(summary["n"], 0)
                self.assertIsNone(summary["mean"])
                self.assertIsNone(summary["descriptive_ci95"])
            elif summary["contrast"] in ("L", "LxD", "L_given_D", "CL_minus_PL", *coverage.REFERENCE_CONTRASTS):
                self.assertEqual(summary["mean"], 0)
                self.assertEqual(summary["descriptive_ci95"], [0, 0])
                self.assertEqual(summary["unresolved_query_fraction"], 1)
        d = next(r for r in result["contrast_summaries"] if r["stratum"] == "all" and
                 r["task"] == "hate" and r["metric"] == "answer_sum/margin/hate" and r["contrast"] == "D")
        self.assertGreater(d["mean"], 0)


if __name__ == "__main__":
    unittest.main()
