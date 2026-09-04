"""CPU-only integration checks for the immutable mechanism lexicon resource.

These tests exercise the actual frozen JSON, not a second fixture lexicon.  The
legacy retriever smoke runs in a child process with model imports stubbed so it
cannot construct embeddings or initialize a GPU and cannot pollute other tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from diagnostics.exploratory_qwen3_ld_v0 import PilotError, exact_hits, load_lexicon
from rag.controlled_lexicon_matcher import (
    MATCHER_POLICY_VERSION,
    ControlledLexiconError,
    ControlledLexiconMatcher,
    normalize_surface,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN_PATH = ROOT / "data/lexicon/annotated_lexicon_mechanism_frozen_v1.json"
SOURCE_PATH = ROOT / "data/lexicon/annotated_lexicon.json"
SOURCE_SHA256 = "a4a2d1e7826419a365962ded70806a610345d21a998454adeda8b1f999738565"
EMPTY_DEFINITION_MARKER = "未提供释义（冻结时为空；未补写）"


class LexiconMechanismFreezeRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        cls.entries = cls.payload["terms"]
        cls.file_sha256 = hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest()
        cls.matcher = ControlledLexiconMatcher(
            cls.entries,
            lexicon_sha256=cls.file_sha256,
            policy_sha256=cls.payload["matcher_policy_sha256"],
        )
        cls.pilot_entries = load_lexicon(
            {
                "sources": {
                    "lexicon": str(FROZEN_PATH),
                    "expected_lexicon_count": len(cls.entries),
                }
            }
        )

    def test_source_is_unchanged_and_frozen_inventory_is_unique(self) -> None:
        self.assertEqual(hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(), SOURCE_SHA256)
        self.assertEqual(len(self.entries), 833)
        self.assertEqual(len({row["lexicon_id"] for row in self.entries}), 833)
        self.assertEqual(len({normalize_surface(row["term"]) for row in self.entries}), 833)
        self.assertEqual(self.payload["matcher_policy_version"], MATCHER_POLICY_VERSION)
        self.assertTrue(self.payload["lexicon_build_id"])
        self.assertEqual(len(self.pilot_entries), len(self.entries))

    def test_empty_definitions_are_preserved_and_only_rendered_as_missing(self) -> None:
        missing = [
            (row, ordinal, sense)
            for row in self.entries
            for ordinal, sense in enumerate(row["senses"], start=1)
            if not sense["definition"].strip()
        ]
        self.assertEqual(len(missing), 7)
        for row, ordinal, sense in missing:
            with self.subTest(term=row["term"], sense_id=sense["sense_id"]):
                self.assertEqual(sense["definition"], "")
                self.assertIn(f"[{ordinal}] {EMPTY_DEFINITION_MARKER}", row["definition"])
        for row in self.entries:
            with self.subTest(term=row["term"]):
                expected = "\n".join(
                    f"[{ordinal}] {sense['definition'].strip() or EMPTY_DEFINITION_MARKER}"
                    for ordinal, sense in enumerate(row["senses"], start=1)
                )
                self.assertEqual(row["definition"], expected)

    def test_sense_category_associations_survive_pilot_loading(self) -> None:
        pilot_by_id = {row["lexicon_id"]: row for row in self.pilot_entries}
        for row in self.entries:
            with self.subTest(term=row["term"]):
                self.assertEqual(pilot_by_id[row["lexicon_id"]]["senses"], row["senses"])
                categories = {value for sense in row["senses"] for value in sense["categories"]}
                self.assertEqual({value.strip() for value in row["category"].split(",")}, categories)
                self.assertTrue(categories)
                self.assertEqual(len({sense["sense_id"] for sense in row["senses"]}), len(row["senses"]))

    def test_longest_matching_deduplicates_injection_but_preserves_occurrences(self) -> None:
        trace = self.matcher.match("妈宝女和妈宝女和妈宝")
        hits = {row["term"]: row for row in trace["selected_hits"]}
        self.assertEqual(hits["妈宝女"]["match_spans"], [[0, 3], [4, 7]])
        self.assertEqual(hits["妈宝"]["match_spans"], [[8, 10]])
        nested = [
            row for row in trace["candidates"]
            if row["term"] == "妈宝" and row["span"] in ([0, 2], [4, 6])
        ]
        self.assertEqual(len(nested), 2)
        self.assertTrue(all(row["selection"] == "overlap_lost" for row in nested))

    def test_normalization_preserves_raw_spans_without_duplicate_injection(self) -> None:
        trace = self.matcher.match("ＹＰ、yp")
        yp_hits = [row for row in trace["selected_hits"] if row["term"] == "YP"]
        self.assertEqual(len(yp_hits), 1)
        self.assertEqual(yp_hits[0]["match_spans"], [[0, 2], [3, 5]])
        self.assertEqual(yp_hits[0]["matched_surfaces"], ["ＹＰ", "yp"])

    def test_context_filters_existing_candidates_before_selection(self) -> None:
        trace = self.matcher.match("基本和基因、批判和其它")
        for term in ("基", "批", "它"):
            candidates = [row for row in trace["candidates"] if row["term"] == term]
            self.assertTrue(candidates, term)
            self.assertTrue(all(row["selection"] == "policy_excluded" for row in candidates), term)

    def test_pilot_uses_identical_full_matcher_and_rejects_top_k(self) -> None:
        query = "妈宝女和妈宝女和妈宝；ＹＰ、yp"
        expected = self.matcher.match(query)["selected_hits"]
        actual = exact_hits(query, self.pilot_entries, top_k=-1, lexicon_sha256=self.file_sha256)
        self.assertEqual(
            [(row["lexicon_id"], row["match_spans"]) for row in actual],
            [(row["lexicon_id"], row["match_spans"]) for row in expected],
        )
        for top_k in (0, 1, 5):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(PilotError, "top-k"):
                exact_hits(query, self.pilot_entries, top_k=top_k)

    def test_policy_hash_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ControlledLexiconError, "policy hash"):
            ControlledLexiconMatcher(
                self.entries,
                lexicon_sha256=self.file_sha256,
                policy_sha256="0" * 64,
            )

    def test_actual_retriever_load_and_legacy_api_without_model_initialization(self) -> None:
        program = r'''
import hashlib
import json
import sys
from pathlib import Path

root, frozen = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / "src" / "tests"))
from test_stage1_retrieval_hits import install_import_stubs
install_import_stubs()
from rag.core import LexiconRetriever

retriever = LexiconRetriever.__new__(LexiconRetriever)
retriever.lexicon_schema = "cold"
retriever.include_variants = False
retriever.load_datas(str(frozen))
assert not hasattr(retriever, "model")
assert retriever.controlled_matcher is not None
payload = json.loads(frozen.read_text(encoding="utf-8"))
assert retriever.matcher_policy_sha256 == payload["matcher_policy_sha256"]
assert retriever.lexicon_sha256 == hashlib.sha256(frozen.read_bytes()).hexdigest()
query = "妈宝女和妈宝女和妈宝"
hits = retriever.including_retrieve_hits(query, top_k=-1)
legacy = retriever.including_retrieve(query, top_k=-1)
assert len(hits) == len(legacy) == 2
assert all(hit.method == "controlled_longest" for hit in hits)
assert [hit.content for hit in hits] == legacy
assert hits[0].provenance["match_spans"] == [[0, 3], [4, 7]]
assert all(hit.provenance["lexicon_sha256"] == retriever.lexicon_sha256 for hit in hits)
for method in (retriever.including_retrieve_hits, retriever.including_retrieve):
    for top_k in (0, 1, 5):
        try:
            method(query, top_k=top_k)
        except ValueError as exc:
            assert "top-k" in str(exc)
        else:
            raise AssertionError("controlled matcher accepted top-k truncation")
print(json.dumps({"entry_count": len(retriever.entries), "model_initialized": False}))
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", program, str(ROOT), str(FROZEN_PATH)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertEqual(json.loads(result.stdout), {"entry_count": 833, "model_initialized": False})


if __name__ == "__main__":
    unittest.main()
