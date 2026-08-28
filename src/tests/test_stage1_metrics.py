import copy
import unittest

from metrics.stage1_metrics import (
    aggregate_query_metrics,
    evaluate_query,
    exact_mcnemar_p,
    flip_table,
    optimal_assignment,
    similarity_v1,
)
from metrics.stage1_statistics import (
    classify_behavior_test,
    factorial_effects,
    fixed_seed_paired_bootstrap,
    holm_adjust,
    paired_dz,
)
from utils.quadruple import serialize_quadruples


def quad(target, argument, group="Racism", hateful="hate"):
    return {
        "target": target,
        "argument": argument,
        "targeted_group": [group],
        "hateful": hateful,
    }


class SimilarityAndAssignmentTests(unittest.TestCase):
    def test_similarity_none_and_autojunk_policy(self):
        self.assertEqual(similarity_v1(None, None), 1.0)
        self.assertEqual(similarity_v1(None, "x"), 0.0)
        self.assertEqual(similarity_v1("x", None), 0.0)
        # A repeated sequence exercises the explicitly disabled autojunk path.
        left = "a" * 220 + "b"
        right = "a" * 220 + "c"
        self.assertGreater(similarity_v1(left, right), 0.99)

    def test_maximum_weight_beats_greedy_and_pair_tie_is_lexicographic(self):
        matrix = {
            (0, 0): 0.9,
            (0, 1): 0.8,
            (1, 0): 0.85,
            (1, 1): 0.1,
        }
        assignment = optimal_assignment(2, 2, lambda i, j: matrix[i, j])
        self.assertEqual(assignment.pairs, ((0, 1), (1, 0)))
        self.assertEqual(assignment.total_weight, 1.65)

        tied = optimal_assignment(2, 2, lambda _i, _j: 1.0)
        self.assertEqual(tied.pairs, ((0, 0), (1, 1)))

    def test_assignment_rounds_each_weight_to_twelve_places(self):
        assignment = optimal_assignment(1, 1, lambda _i, _j: 0.12345678901249)
        self.assertEqual(assignment.weights, (0.123456789012,))


class QueryMetricTests(unittest.TestCase):
    def setUp(self):
        self.gold = [
            quad("甲", "论点甲", "Racism", "hate"),
            quad("乙", "论点乙", "Sexism", "hate"),
        ]

    def evaluate(self, raw, **kwargs):
        return evaluate_query(
            query_id="1",
            condition="C0",
            raw_output=raw,
            gold=self.gold,
            content_sha256="a" * 64,
            prompt_sha256="b" * 64,
            context_record_sha256="c" * 64,
            **kwargs,
        )

    def test_tuple_reordering_does_not_change_metrics(self):
        first = self.evaluate(serialize_quadruples(self.gold))
        second = self.evaluate(serialize_quadruples(list(reversed(self.gold))))
        for key in ("hard", "soft"):
            self.assertEqual(first["tuple"][key]["tp"], 2)
            self.assertEqual(second["tuple"][key]["tp"], 2)
            self.assertTrue(second["tuple"][key]["correct"])
        self.assertTrue(second["field_bound"]["targeted_group"]["correct"])

    def test_group_array_order_is_canonical_for_hard_match(self):
        gold = [quad("甲", "论点", "Racism", "hate")]
        gold[0]["targeted_group"] = ["Racism", "Sexism"]
        raw = '[{"target":"甲","argument":"论点","targeted_group":["Sexism","Racism"],"hateful":"hate"}]'
        result = evaluate_query(query_id="2", condition="CL", raw_output=raw, gold=gold)
        self.assertTrue(result["tuple"]["hard"]["correct"])

    def test_missing_and_extra_tuples_are_penalized(self):
        missing = self.evaluate(serialize_quadruples(self.gold[:1]))
        self.assertEqual(missing["tuple"]["hard"]["fn"], 1)
        self.assertFalse(missing["tuple"]["hard"]["correct"])

        extra_values = [*self.gold, quad("丙", "论点丙", "others", "hate")]
        extra = self.evaluate(serialize_quadruples(extra_values))
        self.assertEqual(extra["tuple"]["hard"]["fp"], 1)
        self.assertFalse(extra["field_unbound"]["target"]["exact_correct"])

    def test_invalid_output_is_empty_prediction_and_stays_in_denominator(self):
        invalid = self.evaluate("```json\n[]\n```")
        self.assertFalse(invalid["strict_format_valid"])
        self.assertTrue(invalid["recoverable_parse_valid"])
        self.assertEqual(invalid["scored_pred_tuple_count"], 0)
        self.assertEqual(invalid["tuple"]["hard"]["fn"], 2)
        self.assertFalse(invalid["field_bound"]["targeted_group"]["correct"])

        valid = self.evaluate(serialize_quadruples(self.gold))
        summary = aggregate_query_metrics([invalid, {**valid, "id": "2"}])
        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["format"]["strict_format_rate"], 0.5)
        self.assertAlmostEqual(summary["tuple"]["hard"]["f1"], 2 / 3)

    def test_unbound_and_bound_namespaces_disagree_when_labels_are_swapped(self):
        swapped = [
            quad("甲", "论点甲", "Sexism", "hate"),
            quad("乙", "论点乙", "Racism", "hate"),
        ]
        result = self.evaluate(serialize_quadruples(swapped))
        self.assertTrue(result["field_unbound"]["targeted_group"]["exact_correct"])
        self.assertFalse(result["field_bound"]["targeted_group"]["correct"])
        self.assertEqual(result["field_bound"]["targeted_group"]["tp"], 0)
        self.assertEqual(result["field_bound"]["targeted_group"]["fp"], 2)
        self.assertEqual(result["field_bound"]["targeted_group"]["fn"], 2)

    def test_unmatched_values_contribute_zero_to_similarity(self):
        result = self.evaluate(serialize_quadruples(self.gold[:1]))
        self.assertEqual(result["field_unbound"]["target"]["similarity_denominator"], 2)
        self.assertEqual(result["field_unbound"]["target"]["similarity"], 0.5)

    def test_runner_failure_never_scores_even_if_payload_is_valid(self):
        result = self.evaluate(serialize_quadruples(self.gold), runner_status="backend_error")
        self.assertFalse(result["strict_format_valid"])
        self.assertEqual(result["tuple"]["hard"]["fn"], 2)
        self.assertIn("runner_status_not_ok", {item["code"] for item in result["errors"]})


class FlipTests(unittest.TestCase):
    def base(self, query_id, correct):
        return {
            "id": str(query_id),
            "content_sha256": f"{query_id:064x}",
            "gold_sha256": "f" * 64,
            "correctness": {"tuple/hard": correct},
        }

    def test_flip_cells_and_mcnemar(self):
        a = [self.base(1, False), self.base(2, True), self.base(3, False), self.base(4, True)]
        b = [self.base(1, True), self.base(2, False), self.base(3, False), self.base(4, True)]
        table = flip_table(a, b, "tuple/hard")
        self.assertEqual((table["n00"], table["n01"], table["n10"], table["n11"]), (1, 1, 1, 1))
        self.assertEqual(table["net_flip"], 0.0)
        self.assertEqual(table["exact_mcnemar_p"], 1.0)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)

    def test_pair_hash_mismatch_fails_closed(self):
        a = [self.base(1, False)]
        b = copy.deepcopy(a)
        b[0]["gold_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "gold_sha256"):
            flip_table(a, b, "tuple/hard")


class PairedStatisticsTests(unittest.TestCase):
    def test_factorial_hand_calculation(self):
        self.assertEqual(
            factorial_effects({"C0": 1.0, "CL": 2.0, "CD": 4.0, "CLD": 8.0}),
            {"L": 1.0, "D": 3.0, "interaction": 3.0},
        )

    def test_a_equals_a_bootstrap_is_exactly_zero(self):
        seed_cells = {
            42: {name: [1.0, 2.0, 3.0] for name in ("C0", "CL", "CD", "CLD")},
            43: {name: [3.0, 2.0, 1.0] for name in ("C0", "CL", "CD", "CLD")},
            44: {name: [2.0, 2.0, 2.0] for name in ("C0", "CL", "CD", "CLD")},
        }
        result = fixed_seed_paired_bootstrap(
            seed_cells=seed_cells,
            metric=lambda values: sum(values) / len(values),
            effect=factorial_effects,
            conditions=("C0", "CL", "CD", "CLD"),
            family="unit-test",
            replicates=50,
        )
        for item in result["effects"].values():
            self.assertEqual(item["point"], 0.0)
            self.assertEqual(item["ci"], [0.0, 0.0])
            self.assertTrue(all(value == 0.0 for value in item["replicates"]))

    def test_holm_tie_break_step_down_and_direction_guard(self):
        adjusted = holm_adjust(
            [
                {"name": "late", "p": 0.01, "contrast_order": 1, "endpoint_order": 0},
                {"name": "early", "p": 0.01, "contrast_order": 0, "endpoint_order": 0},
                {"name": "third", "p": 0.04, "contrast_order": 2, "endpoint_order": 0},
            ]
        )
        by_name = {item["name"]: item for item in adjusted}
        self.assertEqual(by_name["early"]["holm_rank"], 1)
        self.assertEqual(by_name["late"]["holm_rank"], 2)
        self.assertAlmostEqual(by_name["early"]["holm_p"], 0.03)
        self.assertAlmostEqual(by_name["late"]["holm_p"], 0.03)
        self.assertAlmostEqual(by_name["third"]["holm_p"], 0.04)

        classified = classify_behavior_test(
            point=0.02,
            ci=[0.005, 0.03],
            holm_p=0.01,
            seed_effects={42: 0.02, 43: 0.03, 44: -0.01},
            sesoi=0.01,
        )
        self.assertTrue(classified["stat_positive"])
        self.assertFalse(classified["confirmatory_positive"])

    def test_paired_dz_degeneracy_is_fail_closed(self):
        self.assertEqual(paired_dz([0.0, 0.0]), 0.0)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            paired_dz([1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
