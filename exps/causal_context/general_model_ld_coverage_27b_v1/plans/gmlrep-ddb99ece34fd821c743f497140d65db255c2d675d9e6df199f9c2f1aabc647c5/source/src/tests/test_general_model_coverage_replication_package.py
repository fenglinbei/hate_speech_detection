"""Synthetic retokenization checks; no models, matching changes or GPU access."""

import copy
import json
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
        self.assertEqual(inventory["placebo_adapter_policy"], replication.compact_placebo_policy())
        self.assertEqual(inventory["placebo_adapter_applied_record_ids"], [])
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


class CompactPlaceboTests(unittest.TestCase):
    def setUp(self):
        self.demos = [{"id": str(index), "content": "source-demo",
                       "quadruples": [{"target": "target", "argument": "argument",
                                       "targeted_group": ["Region"], "hateful": "non-hate"}]} for index in range(10)]
        self.hits = [{"lexicon_id": f"lex-{index}", "term": f"source-term-{index}", "senses": [
            {"sense_id": f"sense-{index}-{sense}", "definition": "source-definition", "categories": ["Region"]}
            for sense in range(5)]} for index in range(2)]
        self.tolerance = {"absolute_tokens": 8, "relative": .02}

    def counter(self, compact_tokens=194):
        def count(text):
            if not text:
                return 0
            if "source-demo" in text or "source-definition" in text:
                return 188
            addition_count = sum(text.count(sentence) for sentence in replication.renderer.NEUTRAL_SENTENCES)
            if any(sentence in text for sentence in replication.COMPACT_SENTENCES):
                return compact_tokens + addition_count * 4
            return 206 + max(0, addition_count - 10) * 4
        return count

    def render(self, counter, condition="PD"):
        return replication.render_with_placebo_adapter("group", condition, "query", self.hits, self.demos,
            token_count=counter, placebo_tolerance=self.tolerance)

    def test_oversized_original_base_uses_compact_sentence_without_relaxing_tolerance(self):
        rendered = self.render(self.counter())
        trace = rendered["trace"]["placebo"]
        self.assertTrue(rendered["control_valid"])
        self.assertEqual(trace["source_tokens"], 188)
        self.assertEqual(trace["placebo_tokens"], 194)
        self.assertEqual(trace["absolute_difference"], 6)
        self.assertEqual(trace["allowed_difference"], 8)
        self.assertEqual(trace["construction_steps"], 0)
        self.assertEqual(trace["original_failure"]["placebo_tokens"], 206)
        self.assertEqual(trace["original_failure"]["construction_steps"], 0)
        self.assertEqual(trace["original_failure"]["status"], "invalid")
        self.assertEqual(trace["material_version"], replication.COMPACT_MATERIAL_VERSION)
        self.assertEqual(trace["material_sha256"], replication.compact_placebo_policy()["material_sha256"])
        self.assertEqual(trace["selected_base_sentence_index"], 0)
        self.assertEqual(len(trace["compact_candidates"]), 4)
        self.assertEqual(rendered["trace"]["reasons"], [])
        user = rendered["messages"][1]["content"]
        self.assertEqual(user.count(replication.COMPACT_SENTENCES[0]), 10)
        self.assertEqual(user.count(replication.renderer._json(json.loads(replication.renderer._SCHEMAS["group"]))), 10)
        self.assertNotIn("source-demo", user)
        self.assertEqual(len(rendered["trace"]["demo"]), 10)

    def test_compact_lexicon_preserves_every_entry_and_sense(self):
        rendered = self.render(self.counter(), "PL")
        self.assertTrue(rendered["control_valid"])
        user = rendered["messages"][1]["content"]
        self.assertEqual(user.count(replication.COMPACT_SENTENCES[0]), 10)
        self.assertEqual(user.count('\u7c7b\u522b\uff1a["<\u7c7b\u522b>"]'), 10)
        self.assertEqual(user.count("\u8bcd\u6761\uff1a"), 2)
        self.assertEqual(len(rendered["trace"]["lexicon"]), 10)
        self.assertNotIn("source-definition", user)

    def test_compact_base_still_uses_frozen_whole_sentence_append_search(self):
        rendered = self.render(self.counter(compact_tokens=174))
        trace = rendered["trace"]["placebo"]
        self.assertTrue(rendered["control_valid"])
        self.assertEqual(trace["construction_steps"], 2)
        self.assertEqual(trace["placebo_tokens"], 182)
        self.assertEqual(trace["absolute_difference"], 6)

    def test_compact_still_too_long_fails_closed(self):
        rendered = self.render(self.counter(compact_tokens=210))
        self.assertFalse(rendered["control_valid"])
        self.assertEqual(rendered["control_status"], "invalid")
        self.assertEqual(rendered["trace"]["placebo"]["allowed_difference"], 8)
        self.assertEqual(rendered["trace"]["placebo"]["absolute_difference"], 22)
        self.assertEqual(rendered["trace"]["reasons"], ["whole-sentence-match-failed"])

    def test_valid_original_and_non_placebo_results_are_exactly_unchanged(self):
        for condition in ("C0", "CL", "CD", "CLD", "PL", "PD"):
            with self.subTest(condition=condition):
                counter = lambda text: 188 if text else 0
                expected = replication.render_condition("group", condition, "query", self.hits, self.demos,
                    token_count=counter, placebo_tolerance=self.tolerance)
                with mock.patch.object(replication, "_compact_material", side_effect=AssertionError("must not adapt")):
                    self.assertEqual(expected, self.render(counter, condition))

    def test_adapter_never_mutates_frozen_renderer_materials(self):
        sentences, terms = replication.renderer.NEUTRAL_SENTENCES, replication.renderer.NEUTRAL_TERMS
        schemas = copy.deepcopy(replication.renderer._SCHEMAS)
        self.render(self.counter())
        self.assertIs(sentences, replication.renderer.NEUTRAL_SENTENCES)
        self.assertIs(terms, replication.renderer.NEUTRAL_TERMS)
        self.assertEqual(schemas, replication.renderer._SCHEMAS)


if __name__ == "__main__":
    unittest.main()
