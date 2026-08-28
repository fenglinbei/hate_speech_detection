import copy
import unittest

from data.counterfactual_manifest import (
    CounterfactualError,
    build_proposal_rows,
    candidate_id,
    finalize_rows,
    review_template,
)


def quad(target, argument, group, hateful):
    return {
        "target": target,
        "argument": argument,
        "targeted_group": [group],
        "hateful": hateful,
    }


def context(query_id="10"):
    return {
        "query": {
            "id": query_id,
            "content": "坏甲说法，坏乙观点。",
            "gold": [
                quad("坏甲", "坏甲说法", "Racism", "hate"),
                quad("坏乙", "坏乙观点", "Sexism", "hate"),
            ],
        },
        "record_sha256": "a" * 64,
    }


def train_records():
    return [
        {
            "id": "1",
            "content": "x",
            "quadruples": [quad("x", "x", "Racism", "hate")],
        },
        {
            "id": "2",
            "content": "y",
            "quadruples": [quad("y", "y", "Sexism", "hate")],
        },
        {
            "id": "3",
            "content": "z",
            "quadruples": [quad(None, "z", "non-hate", "non-hate")],
        },
    ]


class ProposalTests(unittest.TestCase):
    def test_candidate_id_is_stable_and_value_sensitive(self):
        kwargs = {
            "query_id": "10",
            "gold_sha256": "f" * 64,
            "tuple_index": 0,
            "field": "hateful",
            "candidate_value": "non-hate",
            "family": "binary-label-flip",
            "source": "deterministic-label-catalog",
            "source_id": "v1",
        }
        first = candidate_id(**kwargs)
        self.assertEqual(first, candidate_id(**kwargs))
        self.assertRegex(first, r"^cf:v1:[0-9a-f]{64}$")
        changed = candidate_id(**{**kwargs, "source_id": "v2"})
        self.assertNotEqual(first, changed)

    def test_proposal_has_full_automatic_coverage_and_bounded_review_cohort(self):
        rows, meta = build_proposal_rows(
            cf_proposal_id="cfp-test",
            context_records=[context()],
            train_records=train_records(),
            field_quota={"target": 2, "argument": 2},
        )
        automatic = [row for row in rows if not row["review_required"]]
        review = [row for row in rows if row["review_required"]]
        self.assertEqual(len(automatic), 4)  # 2 tuples × group/hate
        self.assertLessEqual(len(review), 4)
        self.assertEqual(meta["gold_tuple_count"], 2)
        self.assertEqual({row["field"] for row in automatic}, {"targeted_group", "hateful"})
        for row in rows:
            self.assertNotEqual(row["gold_value"], row["candidate_value"])


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.proposals, _ = build_proposal_rows(
            cf_proposal_id="cfp-test",
            context_records=[context()],
            train_records=train_records(),
            field_quota={"target": 2, "argument": 2},
        )

    def completed_reviews(self):
        rows = review_template(self.proposals, reviewer_id="panel")
        for row in rows:
            if row["decision"] == "":
                row["decision"] = "pass"
                row["reason_code"] = "valid-local-foil"
        return rows

    def test_finalize_selects_only_passed_and_group_hate_gate_is_one(self):
        final, summary = finalize_rows(
            self.proposals,
            self.completed_reviews(),
            cf_build_id="cf-test",
            review_id="review:v1:" + "b" * 64,
        )
        self.assertTrue(summary["group_hate_coverage_gate"])
        self.assertTrue(
            all(
                row["selected_cf_id"] is not None
                for row in final
                if row["field"] in {"targeted_group", "hateful"}
            )
        )
        for row in final:
            self.assertEqual(len(row["record_sha256"]), 64)

    def test_exact_review_candidate_set_is_required(self):
        reviews = self.completed_reviews()[:-1]
        with self.assertRaisesRegex(CounterfactualError, "exactly"):
            finalize_rows(
                self.proposals,
                reviews,
                cf_build_id="cf-test",
                review_id="review",
            )

    def test_review_cannot_change_proposal_id(self):
        reviews = self.completed_reviews()
        reviews = copy.deepcopy(reviews)
        reviews[0]["cf_proposal_id"] = "cfp-other"
        with self.assertRaisesRegex(CounterfactualError, "proposal ID"):
            finalize_rows(
                self.proposals,
                reviews,
                cf_build_id="cf-test",
                review_id="review",
            )


if __name__ == "__main__":
    unittest.main()
