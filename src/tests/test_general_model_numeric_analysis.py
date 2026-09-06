import copy
import itertools
import json
import math
import unittest

import numpy as np

from diagnostics.general_model_numeric_analysis import (
    CONDITIONS,
    GROUP_LABELS,
    SCORE_MODES,
    analyze_blocks,
    block_analysis,
    block_readouts,
    candidate_catalog,
    candidate_scores,
    joint_bootstrap_mean_ci,
    logsumexp,
    numerical_direction,
    paired_contrasts,
)


def candidates(task, scores=None, token_count=1):
    result = candidate_catalog()[task]
    if scores is None:
        scores = [-10.0] * len(result)
    for candidate, score in zip(result, scores):
        token_logprobs = [score / token_count] * token_count
        candidate.update(answer_token_ids=list(range(1, token_count + 1)),
                         token_logprobs=token_logprobs,
                         scores=candidate_scores(token_logprobs, -0.25))
    return result


class CandidateMathTests(unittest.TestCase):
    def test_catalog_covers_each_subset_once_in_fixed_order(self):
        catalog = candidate_catalog()
        self.assertEqual([entry["canonical_answer"] for entry in catalog["hate"]],
                         ['"hate"', '"non-hate"'])
        groups = catalog["group"]
        expected = {tuple(GROUP_LABELS[index] for index in range(5) if bits[index])
                    for bits in itertools.product((0, 1), repeat=5)}
        self.assertEqual({tuple(entry["labels"]) for entry in groups}, expected)
        for mask, candidate in enumerate(groups):
            self.assertEqual(candidate["ordinal"], mask)
            self.assertEqual(candidate["label_bitmap"], mask)
            self.assertEqual(candidate["cardinality"], mask.bit_count())
            self.assertEqual(json.loads(candidate["canonical_answer"]), candidate["labels"])
            self.assertNotIn(" ", candidate["canonical_answer"])
        groups[0]["labels"].append("Racism")
        self.assertEqual(candidate_catalog()["group"][0]["labels"], [])

    def test_eos_is_separate_and_mean_has_correct_denominator(self):
        actual = candidate_scores([-1, -2, -3], -4)
        self.assertEqual(actual, {"answer_sum": -6, "answer_mean": -2, "eos_logprob": -4,
                                  "total_with_eos": -10, "mean_with_eos": -2.5})
        for values, eos in (([], -1), ([math.nan], -1), ([-1], math.inf)):
            with self.subTest(values=values, eos=eos), self.assertRaises(ValueError):
                candidate_scores(values, eos)

    def test_extreme_scores_are_stable(self):
        self.assertAlmostEqual(logsumexp([-10000, -10000]), -10000 + math.log(2))
        metrics = block_analysis("group", candidates("group", [-10000] * 32))
        for label in GROUP_LABELS:
            self.assertAlmostEqual(metrics[f"answer_sum/margin/{label}"], 0)
            self.assertAlmostEqual(metrics[f"answer_sum/inclusion_mass/{label}"], 0.5)

    def test_multilabel_marginals_use_all_32_sets_and_can_sum_above_one(self):
        logits = [0.25, 0.5, 1.0, 1.5, 2.0]
        scores = [-100 + sum(value for bit, value in enumerate(logits) if mask & (1 << bit))
                  for mask in range(32)]
        readouts = block_readouts("group", candidates("group", scores))
        for label, expected in zip(GROUP_LABELS, logits):
            self.assertAlmostEqual(readouts[f"margin/answer_sum/{label}"], expected)
        metrics = block_analysis("group", candidates("group", scores))
        self.assertGreater(sum(metrics[f"answer_sum/inclusion_mass/{label}"] for label in GROUP_LABELS), 2.5)

    def test_uniform_sets_show_combinatorics_and_equal_k_sensitivity(self):
        metrics = block_analysis("group", candidates("group"))
        for size in range(6):
            self.assertAlmostEqual(metrics[f"answer_sum/cardinality/natural/p{size}"], math.comb(5, size) / 32)
            self.assertAlmostEqual(metrics[f"answer_sum/cardinality/equal_k/p{size}"], 1 / 6)
        for label in GROUP_LABELS:
            for size in (0, 5):
                self.assertIsNone(metrics[f"answer_sum/within_k/natural/{label}/{size}"])
                self.assertIsNone(metrics[f"answer_sum/within_k/count_corrected/{label}/{size}"])
            for size in range(1, 5):
                self.assertAlmostEqual(metrics[f"answer_sum/within_k/natural/{label}/{size}"],
                                       math.log(size / (5 - size)))
                self.assertAlmostEqual(metrics[f"answer_sum/within_k/count_corrected/{label}/{size}"], 0)

    def test_within_k_count_correction_cancels_in_paired_difference(self):
        baseline = block_analysis("group", candidates("group"))
        changed = block_analysis("group", candidates("group", [-10 + 0.6 * bool(mask & 1) for mask in range(32)]))
        for size in range(1, 5):
            raw = f"answer_sum/within_k/natural/Racism/{size}"
            corrected = f"answer_sum/within_k/count_corrected/Racism/{size}"
            self.assertAlmostEqual(changed[raw] - baseline[raw], 0.6)
            self.assertAlmostEqual(changed[corrected] - baseline[corrected], 0.6)

    def test_gold_is_explicit_auxiliary_and_toggle_favors_gold_in_both_directions(self):
        scored = candidates("group", [-20 - mask for mask in range(32)])
        before = copy.deepcopy(scored)
        without_gold = block_analysis("group", scored)
        metrics = block_analysis("group", scored, gold=["Racism"])
        self.assertFalse(any("gold/" in key for key in without_gold))
        self.assertEqual(scored, before)
        for key, value in without_gold.items():
            self.assertEqual(metrics[key], value)
        self.assertEqual(metrics["answer_sum/gold/rank"], 2)
        self.assertEqual(metrics["answer_sum/gold/best_nongold_margin"], -1)
        self.assertEqual(metrics["answer_sum/gold/toggle_margin/Racism"], -1)
        self.assertEqual(metrics["answer_sum/gold/toggle_margin/Region"], 2)
        self.assertAlmostEqual(metrics["answer_sum/gold/nll"], -math.log(metrics["answer_sum/gold/mass"]))

    def test_score_modes_and_gate_include_each_raw_token(self):
        scored = candidates("hate", [-6, -8], token_count=2)
        readouts = block_readouts("hate", scored)
        self.assertEqual(readouts["candidate/hate/token/1"], -3)
        self.assertEqual(readouts["candidate/non-hate/eos_logprob"], -0.25)
        self.assertEqual(readouts["margin/answer_sum/hate"], 2)
        self.assertEqual(readouts["margin/answer_mean/hate"], 1)
        self.assertEqual(readouts["margin/total_with_eos/hate"], 2)
        self.assertAlmostEqual(readouts["margin/mean_with_eos/hate"], 2 / 3)

    def test_invalid_identity_missing_score_boundary_and_nonfinite_are_rejected(self):
        examples = []
        bad = candidates("group")
        examples.append(bad[:-1])
        bad = candidates("group")
        bad[0], bad[1] = bad[1], bad[0]
        examples.append(bad)
        bad = candidates("group")
        bad[3]["scores"]["answer_mean"] += 0.1
        examples.append(bad)
        bad = candidates("group")
        bad[3]["token_logprobs"].append(-1)
        examples.append(bad)
        bad = candidates("group")
        bad[3]["scores"]["eos_logprob"] = math.nan
        examples.append(bad)
        for example in examples:
            with self.subTest(example=example[:1]), self.assertRaises(ValueError):
                block_readouts("group", example)


class PairedAnalysisTests(unittest.TestCase):
    def test_eight_contrasts_and_error_propagation(self):
        actual = paired_contrasts({"C0": 1, "CL": 3, "CD": 4, "CLD": 10, "PL": 2, "PD": 2.5})
        self.assertEqual(actual, {"L": 2, "D": 3, "LD": 9, "LxD": 4,
                                  "L_given_D": 6, "D_given_L": 7, "CL_minus_PL": 1,
                                  "CD_minus_PD": 1.5})
        self.assertEqual(numerical_direction(0.02, epsilon=0.01, contrast="L")["direction"],
                         "numerically_unresolved")
        self.assertEqual(numerical_direction(-0.021, epsilon=0.01, contrast="L")["direction"], "negative")
        self.assertEqual(numerical_direction(-0.03, epsilon=0.01, contrast="LxD")["raw_sign"], -1)
        with self.assertRaises(ValueError):
            paired_contrasts({"C0": 0})

    def test_bootstrap_reuses_queries_for_all_endpoints_and_subgroups(self):
        values = np.asarray([[1, 2], [3, 6], [7, 14], [9, 18]], dtype=float)
        strata = {"all": [True] * 4, "first": [True, False, False, False], "empty": [False] * 4}
        first = joint_bootstrap_mean_ci(values, strata=strata, replicates=500, seed=42, batch_size=17)
        second = joint_bootstrap_mean_ci(values, strata=strata, replicates=500, seed=42, batch_size=256)
        self.assertEqual(first, second)
        np.testing.assert_allclose(np.asarray(first["all"])[1], np.asarray(first["all"])[0] * 2)
        self.assertEqual(first["first"], [[1, 1], [2, 2]])
        self.assertEqual(first["empty"], [None, None])

    def test_complete_analysis_retains_nonhit_and_reports_descriptive_only(self):
        frame = [{"query_id": "a", "lex_hit": True}, {"query_id": "b", "lex_hit": False}]
        gold = {"a": {"hate": "hate", "group": ["Racism", "Region"]},
                "b": {"hate": "non-hate", "group": []}}
        blocks = []
        for query in frame:
            for task in ("hate", "group"):
                for condition in CONDITIONS:
                    scores = [-10.0] * (2 if task == "hate" else 32)
                    if task == "hate" and condition == "CL":
                        scores[0] += 1 if query["query_id"] == "a" else 3
                    blocks.append({"query_id": query["query_id"], "task": task, "condition": condition,
                                   "candidates": candidates(task, scores)})
        result = analyze_blocks(blocks, frame=frame, gold_by_query=gold, epsilon=0.01,
                                expected_query_count=2, bootstrap_replicates=50)
        summary = next(row for row in result["contrast_summaries"]
                       if row["task"] == "hate" and row["metric"] == "answer_sum/margin/hate"
                       and row["contrast"] == "L" and row["stratum"] == "all")
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["mean"], 2)
        self.assertEqual(summary["positive_query_count"], 2)
        self.assertEqual(summary["positive_query_fraction"], 1)
        self.assertEqual(summary["negative_query_fraction"], 0)
        self.assertEqual(summary["unresolved_query_fraction"], 0)
        self.assertEqual(summary["q25"], 1.5)
        self.assertEqual(summary["q75"], 2.5)
        self.assertEqual(summary["iqr"], 1)
        self.assertIn("not sequence probability", result["interpretation"]["score_normalization"]["answer_mean"])
        self.assertIn("descriptive_ci95", summary)
        self.assertEqual(result["stratum_counts"]["lex_no_hit"], 1)
        self.assertEqual(result["stratum_counts"]["gold_size_2plus"], 1)
        self.assertEqual(len(result["per_query"]), 2)
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("p_value", serialized)
        self.assertNotIn("p_two_sided", serialized)
        for invalid_blocks in (blocks[:-1], [*blocks, blocks[0]]):
            with self.assertRaises(ValueError):
                analyze_blocks(invalid_blocks, frame=frame, gold_by_query=gold, epsilon=0.01,
                               expected_query_count=2, bootstrap_replicates=2)
        with self.assertRaises(ValueError):
            analyze_blocks(blocks, frame=frame, gold_by_query=gold, epsilon=0.01)


if __name__ == "__main__":
    unittest.main()
