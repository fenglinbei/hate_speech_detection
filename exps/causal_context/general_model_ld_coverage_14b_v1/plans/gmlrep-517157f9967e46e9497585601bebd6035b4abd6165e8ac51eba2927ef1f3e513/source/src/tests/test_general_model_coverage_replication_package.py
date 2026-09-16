"""Synthetic retokenization checks; no models, matching changes or GPU access."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_coverage_replication_package as replication
from diagnostics.general_model_package import PackageError, write_json, write_jsonl
from src.tests.test_general_model_coverage_package import Fixture, Tokenizer


class ReplicationTokenizer(Tokenizer):
    eos_token_id = 250001
    pad_token_id = 250002

    def encode(self, text, add_special_tokens=False):
        return [value + 100 for value in text.encode("utf-8")]

    def apply_chat_template(self, messages, **kwargs):
        return "replication-template\n" + super().apply_chat_template(messages, **kwargs)


class ReplicationPackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = Fixture(self.root)
        self.old, self.inventory = self.fixture.build()
        self.parent_dir = self.root / "parent"
        self.parent_dir.mkdir()
        write_json(self.parent_dir / "resource_inventory.json", self.inventory)
        write_jsonl(self.parent_dir / "contexts.dev.jsonl", self.old)
        self.plan = {**copy.deepcopy(self.fixture.plan), "plan_id": "gmlcoverage-synthetic-parent",
                     "plan_dir": str(self.parent_dir),
                     "input_files": {name: sha256_file(self.parent_dir / name)
                                     for name in ("resource_inventory.json", "contexts.dev.jsonl")},
                     "blocks": [{key: row[key] for key in ("record_id", "context_sha256")} for row in self.old]}

    def build(self, tokenizer=None, generation_eos=None):
        with mock.patch.object(replication, "ROOT", self.root), \
                mock.patch.object(replication, "FROZEN_LEXICON_SHA256", self.fixture.lexicon_sha), \
                mock.patch.object(replication, "FROZEN_ENTRY_COUNT", 2):
            return replication.build_contexts(self.plan, self.old, self.root, tokenizer or ReplicationTokenizer(),
                model_key="qwen3.8-27b", generation_eos_token_ids=[250001, 250002] if generation_eos is None else generation_eos)

    def test_retokenization_preserves_six_semantic_cells_and_accepts_multiple_generation_stops(self):
        snapshot = copy.deepcopy(self.old)
        contexts, catalog, inventory, eos, pad = self.build()
        self.assertEqual((eos, pad), (250001, 250002))
        self.assertEqual(inventory["eos"]["generation_eos_token_ids"], [250001, 250002])
        self.assertTrue(inventory["eos"]["multiple_generation_stop_ids_are_not_canonical_score_ambiguity"])
        self.assertEqual(len(contexts), 3 * 2 * 8)
        self.assertEqual(inventory["boundary_checks"], 3 * 8 * 34)
        self.assertEqual(inventory["counts"]["unchanged_non_placebo_messages"], 36)
        old = {row["record_id"]: row for row in self.old}
        for row in contexts:
            if row["condition"] not in ("PLnew", "PD"):
                self.assertEqual(row["messages"], old[row["record_id"]]["messages"])
            self.assertEqual(row["model_key"], "qwen3.8-27b")
            self.assertTrue(row["prompt_text"].startswith("replication-template"))
            self.assertEqual(row["context_sha256"], canonical_json_sha256({k: v for k, v in row.items() if k != "context_sha256"}))
        self.assertNotEqual(catalog["hate"][0]["answer_token_ids"], self.fixture.plan["catalog"]["hate"][0]["answer_token_ids"])
        self.assertEqual(self.old, snapshot)

    def test_no_matching_retrieval_or_gold_reads_and_boundaries_reselected(self):
        from rag.controlled_lexicon_matcher import ControlledLexiconMatcher
        with mock.patch.object(ControlledLexiconMatcher, "match", side_effect=AssertionError("must not rematch")), \
                mock.patch.object(replication, "read_jsonl", wraps=replication.read_jsonl) as reads:
            _, _, inventory, _, _ = self.build()
        self.assertEqual({Path(call.args[0]).name for call in reads.call_args_list},
                         {"retrieval.dev.jsonl", "lexicon.dev.jsonl", "fit_catalog.jsonl"})
        self.assertFalse(inventory["matching_reexecuted"])
        self.assertFalse(inventory["retrieval_reexecuted"])
        self.assertFalse(inventory["query_gold_loaded"])
        self.assertFalse(inventory["test_content_read"])
        self.assertEqual(inventory["original_cohorts"], self.fixture.plan["cohorts"])
        self.assertEqual(inventory["boundary_query_ids"], ["3"])
        self.assertNotEqual(inventory["limits"]["max_prompt_tokens"], self.inventory["limits"]["max_prompt_tokens"])

    def test_parent_context_resource_or_demo_mutation_is_rejected(self):
        self.old[0]["messages"][0]["content"] += "changed"
        with self.assertRaisesRegex(PackageError, "parent context identity changed"):
            self.build()
        self.old[0]["messages"][0]["content"] = self.old[0]["messages"][0]["content"][:-7]
        self.inventory["per_query"][0]["union_ids"] = []
        write_json(self.parent_dir / "resource_inventory.json", self.inventory)
        with self.assertRaisesRegex(PackageError, "resource inventory changed"):
            self.build()

    def test_candidate_boundary_and_sequence_budget_fail_closed(self):
        class Broken(ReplicationTokenizer):
            def encode(self, text, add_special_tokens=False):
                result = super().encode(text, add_special_tokens)
                return result + [888] if "assistant:" in text and text.endswith('"hate"') else result

        with self.assertRaisesRegex(PackageError, "token boundary changed"):
            self.build(Broken())
        with mock.patch.object(replication, "MAX_SEQUENCE_TOKENS", 200):
            with self.assertRaisesRegex(PackageError, "input plus padding exceeds"):
                self.build()


if __name__ == "__main__":
    unittest.main()
