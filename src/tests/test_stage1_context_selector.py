import math
import unittest

from data.context_selector import (
    CandidateIdentityError,
    NonFiniteScoreError,
    QuotaUnsatisfiedError,
    round_similarity_half_even,
    select_demos,
    select_lexicons,
)
from rag.types import RetrievalHit, content_sha256, sha256_text


def make_hit(demo_id, source_id, content, score, source_class, gold="[]"):
    return RetrievalHit(
        id=demo_id,
        source_record_id=source_id,
        content=content,
        output=gold,
        content_sha256=content_sha256(content),
        gold_sha256=sha256_text(gold),
        score=score,
        rank=99,
        source_class=source_class,
        method="cosine",
        provenance={"fixture": True},
    )


def make_lex_hit(lexicon_id, term, content, score, rank, method):
    return RetrievalHit(
        id=lexicon_id,
        source_record_id=term,
        content=content,
        output=None,
        content_sha256=content_sha256(content),
        gold_sha256=None,
        score=score,
        rank=rank,
        source_class="Racism",
        method=method,
        provenance={"match_spans": [[1, 3]]} if score is None else {},
    )


class Stage1ContextSelectorTest(unittest.TestCase):
    def test_half_even_rounding_precedes_threshold_and_rank(self):
        self.assertEqual(round_similarity_half_even(0.123456785), 0.12345678)
        self.assertEqual(round_similarity_half_even(0.123456795), 0.1234568)

        hits = {
            "A": [
                make_hit("demo-b", "2", "b", 0.5000000049, "A"),
                make_hit("demo-a", "1", "a", 0.5000000048, "A"),
            ]
        }
        trace = select_demos(
            hits,
            source_class_order=["A"],
            allocated_class_top_k={"A": 1},
            similarity_threshold=0.5,
            candidate_multiplier=2,
        )

        self.assertEqual(trace.class_candidate_ids["A"], ("demo-a", "demo-b"))
        self.assertEqual(trace.selected_ids, ("demo-a",))
        self.assertEqual(trace.candidates[0].evidence[0].source_rank, 0)

        with self.assertRaises(QuotaUnsatisfiedError):
            select_demos(
                {"A": [make_hit("boundary", "3", "boundary", 0.5000000049, "A")]},
                source_class_order=["A"],
                allocated_class_top_k={"A": 1},
                similarity_threshold=0.5000000049,
            )

    def test_cross_class_dedupe_refills_quota(self):
        shared_a = make_hit("shared", "10", "same", 0.90, "A")
        shared_b = make_hit("shared", "10", "same", 0.95, "B")
        trace = select_demos(
            {
                "A": [shared_a, make_hit("a2", "11", "a2", 0.80, "A")],
                "B": [shared_b, make_hit("b2", "12", "b2", 0.70, "B")],
            },
            source_class_order=["A", "B"],
            allocated_class_top_k={"A": 1, "B": 1},
            candidate_multiplier=2,
        )

        self.assertEqual(trace.selected_ids, ("shared", "b2"))
        self.assertEqual(
            [(item.assigned_quota_class, item.quota_round) for item in trace.quota_assignments],
            [("A", 0), ("B", 0)],
        )
        self.assertEqual(len(set(trace.selected_ids)), 2)
        shared = next(candidate for candidate in trace.candidates if candidate.demo_id == "shared")
        self.assertEqual(len(shared.evidence), 2)
        self.assertEqual(shared.selection_score, 0.95)
        self.assertEqual(shared.tie_source_class, "B")
        self.assertEqual(trace.duplicate_occurrences_merged, 1)

    def test_assignment_order_and_prompt_order_are_independent(self):
        trace = select_demos(
            {
                "A": [make_hit("slow", "1", "slow", 0.5, "A")],
                "B": [make_hit("fast", "2", "fast", 0.9, "B")],
            },
            source_class_order=["A", "B"],
            allocated_class_top_k={"A": 1, "B": 1},
            candidate_multiplier=1,
        )

        self.assertEqual(trace.selected_ids, ("slow", "fast"))
        self.assertEqual(trace.prompt_order, ("fast", "slow"))

    def test_query_overlap_is_excluded_with_trace(self):
        query_content = content_sha256("query")
        trace = select_demos(
            {
                "A": [
                    make_hit("same-id", "q1", "other", 0.9, "A"),
                    make_hit("same-content", "2", "query", 0.8, "A"),
                    make_hit("kept", "3", "kept", 0.7, "A"),
                ]
            },
            source_class_order=["A"],
            allocated_class_top_k={"A": 1},
            candidate_multiplier=3,
            query_source_record_id="q1",
            query_content_sha256=query_content,
        )

        self.assertEqual(trace.selected_ids, ("kept",))
        self.assertEqual(
            {item["reason"] for item in trace.excluded},
            {"source_record_id_overlap", "content_sha256_overlap"},
        )

    def test_invalid_scores_identity_and_unfillable_quota_hard_fail(self):
        with self.assertRaises(NonFiniteScoreError):
            select_demos(
                {"A": [make_hit("x", "1", "x", math.nan, "A")]},
                source_class_order=["A"],
                allocated_class_top_k={"A": 1},
            )

        with self.assertRaises(CandidateIdentityError):
            select_demos(
                {
                    "A": [make_hit("x", "1", "x", 0.9, "A")],
                    "B": [make_hit("x", "2", "different", 0.8, "B")],
                },
                source_class_order=["A", "B"],
                allocated_class_top_k={"A": 1, "B": 1},
                candidate_multiplier=2,
            )

        with self.assertRaises(QuotaUnsatisfiedError):
            select_demos(
                {"A": [make_hit("x", "1", "x", 0.9, "A")]},
                source_class_order=["A"],
                allocated_class_top_k={"A": 2},
                candidate_multiplier=2,
            )

    def test_input_order_does_not_change_trace(self):
        first = make_hit("a", "1", "a", 0.5, "A")
        second = make_hit("b", "2", "b", 0.5, "A")
        kwargs = {
            "source_class_order": ["A"],
            "allocated_class_top_k": {"A": 2},
            "candidate_multiplier": 1,
        }

        left = select_demos({"A": [first, second]}, **kwargs)
        right = select_demos({"A": [second, first]}, **kwargs)

        self.assertEqual(left.to_dict(), right.to_dict())

    def test_lexicon_exact_first_semantic_dedupe_preserves_both_evidence(self):
        shared_exact = make_lex_hit("lex-shared", "shared", "shared block", None, 0, "substring")
        shared_semantic = make_lex_hit("lex-shared", "shared", "shared block", 0.99, 8, "cosine")
        semantic = make_lex_hit("lex-semantic", "semantic", "semantic block", 0.8, 0, "cosine")

        trace = select_lexicons(
            [shared_exact],
            [semantic, shared_semantic],
            semantic_top_k=1,
        )

        self.assertEqual(trace.selected_ids, ("lex-shared", "lex-semantic"))
        self.assertEqual(trace.prompt_order, trace.selected_ids)
        self.assertEqual(len(trace.candidates[0].evidence), 2)
        self.assertIsNone(trace.candidates[0].evidence[0].written_similarity)
        self.assertEqual(trace.candidates[0].evidence[0].match_spans, ((1, 3),))
        self.assertEqual(trace.duplicate_occurrences_merged, 1)

    def test_lexicon_semantic_score_ties_preserve_label_blind_upstream_rank(self):
        rank_first = make_lex_hit("lex-z", "first", "first block", 0.8, 0, "cosine")
        id_first = make_lex_hit("lex-a", "second", "second block", 0.8, 1, "cosine")

        trace = select_lexicons(
            [],
            [id_first, rank_first],
            semantic_top_k=1,
        )

        self.assertEqual(trace.selected_ids, ("lex-z",))
        self.assertEqual(trace.candidates[0].evidence[0].source_rank, 0)


if __name__ == "__main__":
    unittest.main()
