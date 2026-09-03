from __future__ import annotations

import unittest

from rag.controlled_lexicon_matcher import (
    ControlledLexiconError,
    ControlledLexiconMatcher,
    normalize_surface,
)


LEXICON_SHA = "a" * 64


def entry(
    lexicon_id: str,
    term: str,
    *,
    variants: list[str] | None = None,
    match_policy: dict | None = None,
) -> dict:
    return {
        "lexicon_id": lexicon_id,
        "term": term,
        "category": "others",
        "definition": "fixture",
        "variants": variants or [],
        "match_policy": match_policy or {},
    }


class ControlledLexiconMatcherTests(unittest.TestCase):
    def test_normalization_is_conservative(self) -> None:
        self.assertEqual(normalize_surface("ＡＢＣe\u0301"), "abcé")
        self.assertEqual(normalize_surface("臺灣"), "臺灣")

    def test_policy_filters_before_global_longest(self) -> None:
        matcher = ControlledLexiconMatcher(
            [
                entry(
                    "lex-base",
                    "基",
                    match_policy={
                        "exclude_any": [
                            {"rule_id": "exclude-jiben", "target": "right", "pattern": "^本"}
                        ]
                    },
                ),
                entry("lex-long", "基本盘"),
            ],
            lexicon_sha256=LEXICON_SHA,
        )

        result = matcher.match("基本尊重，不是基本盘")

        self.assertEqual(
            [(row["term"], row["match_spans"]) for row in result["selected_hits"]],
            [("基本盘", [[7, 10]])],
        )
        excluded = [
            row for row in result["candidates"] if row["term"] == "基"
        ]
        self.assertTrue(all(row["selection"] == "policy_excluded" for row in excluded))
        self.assertTrue(
            all(row["policy"]["matched_exclude_rule_ids"] == ["exclude-jiben"] for row in excluded)
        )

    def test_longest_nonoverlapping_span_wins_globally(self) -> None:
        matcher = ControlledLexiconMatcher(
            [entry("lex-short", "妈宝"), entry("lex-long", "妈宝女")],
            lexicon_sha256=LEXICON_SHA,
        )

        result = matcher.match("妈宝女和妈宝")

        self.assertEqual(
            result["selected_spans"],
            [
                {
                    "candidate_id": result["selected_spans"][0]["candidate_id"],
                    "lexicon_id": "lex-long",
                    "term": "妈宝女",
                    "raw_surface": "妈宝女",
                    "span": [0, 3],
                },
                {
                    "candidate_id": result["selected_spans"][1]["candidate_id"],
                    "lexicon_id": "lex-short",
                    "term": "妈宝",
                    "raw_surface": "妈宝",
                    "span": [4, 6],
                },
            ],
        )
        nested = next(
            row
            for row in result["candidates"]
            if row["term"] == "妈宝" and row["span"] == [0, 2]
        )
        self.assertEqual(nested["selection"], "overlap_lost")

    def test_explicit_variant_generates_but_regex_cannot_resize_span(self) -> None:
        matcher = ControlledLexiconMatcher(
            [
                entry(
                    "lex-one",
                    "瞎逼逼",
                    variants=["瞎ＢＢ"],
                    match_policy={
                        "require_any": [
                            {"rule_id": "surface", "target": "surface", "pattern": "^瞎bb$"}
                        ]
                    },
                )
            ],
            lexicon_sha256=LEXICON_SHA,
        )

        result = matcher.match("是在瞎bb，不是别的")

        self.assertEqual(result["selected_spans"][0]["span"], [2, 5])
        self.assertEqual(result["selected_spans"][0]["raw_surface"], "瞎bb")
        sources = result["candidates"][0]["generation_sources"]
        self.assertEqual(sources[0]["kind"], "variant")
        self.assertTrue(sources[0]["normalization_applied"])

    def test_normalized_surface_collision_is_rejected(self) -> None:
        with self.assertRaisesRegex(ControlledLexiconError, "collision"):
            ControlledLexiconMatcher(
                [entry("lex-a", "ABC"), entry("lex-b", "ａｂｃ")],
                lexicon_sha256=LEXICON_SHA,
            )

    def test_exclude_rule_wins_over_require_rule(self) -> None:
        matcher = ControlledLexiconMatcher(
            [
                entry(
                    "lex-one",
                    "批",
                    match_policy={
                        "require_any": [
                            {"rule_id": "always", "target": "surface", "pattern": "^批$"}
                        ],
                        "exclude_any": [
                            {"rule_id": "ordinary", "target": "right", "pattern": "^(判|评)"}
                        ],
                    },
                )
            ],
            lexicon_sha256=LEXICON_SHA,
        )

        result = matcher.match("批判和一批")

        self.assertEqual(result["selected_spans"], [
            {
                "candidate_id": result["selected_spans"][0]["candidate_id"],
                "lexicon_id": "lex-one",
                "term": "批",
                "raw_surface": "批",
                "span": [4, 5],
            }
        ])


if __name__ == "__main__":
    unittest.main()
