"""Synthetic-only checks for the global dictionary coverage input contract."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_coverage_package as coverage
from diagnostics.general_model_contexts import render_condition
from diagnostics.general_model_numeric_analysis import candidate_catalog
from diagnostics.general_model_package import PackageError, write_json, write_jsonl
from rag.controlled_lexicon_matcher import ControlledLexiconMatcher


class Tokenizer:
    eos_token_id = 900001
    pad_token_id = 900002

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False, separators=(",", ":")) + "\nassistant:"


class Fixture:
    def __init__(self, root):
        self.root = root
        self.tokenizer = Tokenizer()
        self.entries = [
            {"lexicon_id": "lex-2", "term": "alpha", "category": "Region", "definition": "unused",
             "variants": [], "match_policy": {}, "senses": [
                 {"sense_id": "alpha-sense", "categories": ["Region"], "definition": "alpha meaning " * 12}]},
            {"lexicon_id": "lex-10", "term": "beta", "category": "Sexism", "definition": "unused",
             "variants": [], "match_policy": {}, "senses": [
                 {"sense_id": "beta-sense-1", "categories": ["Sexism"], "definition": "beta meaning one " * 12},
                 {"sense_id": "beta-sense-2", "categories": ["others"], "definition": "beta meaning two " * 12}]},
        ]
        provisional = ControlledLexiconMatcher(self.entries, lexicon_sha256="0" * 64)
        write_json(root / "lexicon.json", {"terms": self.entries, "matcher_policy_sha256": provisional.policy_sha256})
        self.lexicon_sha = sha256_file(root / "lexicon.json")
        self.matcher = ControlledLexiconMatcher(self.entries, lexicon_sha256=self.lexicon_sha,
                                                policy_sha256=provisional.policy_sha256)
        write_json(root / "config.resolved.json", {
            "sources": {"lexicon": "lexicon.json", "lexicon_sha256": self.lexicon_sha},
            "controls": {"absolute_tokens": 8, "relative": 0.02}, "retrieval": {"demo_top_k": 10},
        })
        self.retrieval, self.lexicon_rows, self.fit = [], [], []
        self.old = []
        for qid, query in (("1", "alpha query"), ("2", "unmatched query"), ("3", "unmatched query")):
            demos = []
            for ordinal in range(10):
                prefix = ("beta" if ordinal < 2 else "alpha") if qid == "1" else ("beta" if qid == "2" else "plain")
                row = {"id": str(int(qid) * 100 + ordinal),
                       "content": prefix + " ordinary material" * 12,
                       "quadruples": [{"target": "target", "argument": "argument",
                                       "targeted_group": ["Region"], "hateful": "hate"}]}
                self.fit.append(copy.deepcopy(row))
                demos.append({**row, "prompt_rank": ordinal})
            trace = self.matcher.match(query)
            self.retrieval.append({"query_id": qid, "demos": demos})
            self.lexicon_rows.append({"query_id": qid, "trace": trace})
            for task in coverage.TASKS:
                for condition in ("C0", "CL", "CD", "CLD", "PL", "PD"):
                    rendered = render_condition(task, condition, query, trace["selected_hits"], demos,
                                                token_count=lambda text: len(self.tokenizer.encode(text)))
                    row = {"record_id": f"{qid}:{task}:{condition}", "query_id": qid,
                           "task": task, "condition": condition, **rendered}
                    row["context_sha256"] = canonical_json_sha256(row)
                    self.old.append(row)
        self.save_resources()
        catalog = candidate_catalog()
        for candidates in catalog.values():
            for candidate in candidates:
                answer = candidate["canonical_answer"]
                ids = self.tokenizer.encode(answer)
                candidate.update(answer_token_ids=ids, answer_tokens=len(ids),
                                 answer_token_ids_sha256=canonical_json_sha256(ids),
                                 answer_sha256=hashlib.sha256(answer.encode()).hexdigest())
        self.plan = {"catalog": catalog, "eos_token_id": self.tokenizer.eos_token_id,
                     "pad_token_id": self.tokenizer.pad_token_id,
                     "frame": [{"query_id": str(i), "lex_hit": i == 1} for i in range(1, 4)],
                     "cohorts": {"regression": ["1"], "validation": ["2"]}}

    def save_resources(self):
        write_jsonl(self.root / "retrieval.dev.jsonl", self.retrieval)
        write_jsonl(self.root / "lexicon.dev.jsonl", self.lexicon_rows)
        write_jsonl(self.root / "fit_catalog.jsonl", self.fit)

    def build(self, tokenizer=None):
        with mock.patch.object(coverage, "ROOT", self.root), \
                mock.patch.object(coverage, "FROZEN_LEXICON_SHA256", self.lexicon_sha), \
                mock.patch.object(coverage, "FROZEN_ENTRY_COUNT", len(self.entries)):
            return coverage.build_contexts(self.plan, self.old, self.root, tokenizer or self.tokenizer)


class CoveragePackageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = Fixture(Path(self.directory.name))

    def test_global_union_id_order_all_senses_and_hidden_source_information(self):
        snapshot = copy.deepcopy(self.fixture.old)
        contexts, inventory = self.fixture.build()
        by_id = {row["record_id"]: row for row in contexts}
        metadata = {row["query_id"]: row for row in inventory["per_query"]}
        self.assertEqual(metadata["1"]["union_ids"], ["lex-10", "lex-2"])
        self.assertEqual(metadata["1"]["lq_ids"], ["lex-2"])
        self.assertEqual(metadata["1"]["ld_ids"], ["lex-10", "lex-2"])
        for condition in ("CLnew", "CLDnew"):
            text = by_id[f"1:hate:{condition}"]["messages"][1]["content"]
            self.assertEqual(text.count("beta meaning one"), 12)
            self.assertEqual(text.count("beta meaning two"), 12)
            self.assertLess(text.index("beta meaning one"), text.index("alpha meaning"))
            self.assertNotIn("lex-10", text)
            self.assertNotIn("beta-sense", text)
            self.assertNotIn("prompt_rank", text)
        self.assertEqual(by_id["1:hate:CLnew"]["trace"]["injected_demo_ids"], [])
        self.assertEqual(by_id["1:hate:CLDnew"]["trace"]["injected_demo_ids"], metadata["1"]["demo_ids"])
        self.assertEqual(by_id["1:hate:CLnew"]["trace"]["injected_blocks"]["lexicon_sha256"],
                         by_id["1:hate:CLDnew"]["trace"]["injected_blocks"]["lexicon_sha256"])
        self.assertEqual(self.fixture.old, snapshot)

    def test_no_query_hit_can_have_new_dictionary_and_empty_union_degenerates(self):
        contexts, inventory = self.fixture.build()
        by_id = {row["record_id"]: row for row in contexts}
        metadata = {row["query_id"]: row for row in inventory["per_query"]}
        self.assertFalse(metadata["2"]["lex_hit"])
        self.assertTrue(metadata["2"]["union_hit"])
        self.assertNotEqual(by_id["2:hate:CLnew"]["messages"], by_id["2:hate:C0"]["messages"])
        self.assertEqual(by_id["2:hate:CLq"]["messages"], by_id["2:hate:C0"]["messages"])
        for task in coverage.TASKS:
            for condition in ("CLnew", "PLnew", "CLq"):
                self.assertEqual(by_id[f"3:{task}:{condition}"]["messages"], by_id[f"3:{task}:C0"]["messages"])
            for condition in ("CLDnew", "CLqD"):
                self.assertEqual(by_id[f"3:{task}:{condition}"]["messages"], by_id[f"3:{task}:CD"]["messages"])
        self.assertEqual(inventory["boundary_query_ids"], ["3"])

    def test_matrix_hashes_boundary_coverage_and_gold_free_reads(self):
        original_read = coverage.read_jsonl
        read_names = []

        def checked_read(path):
            self.assertIn(path.name, {"retrieval.dev.jsonl", "lexicon.dev.jsonl", "fit_catalog.jsonl"})
            read_names.append(path.name)
            return original_read(path)

        with mock.patch.object(coverage, "read_jsonl", side_effect=checked_read):
            contexts, inventory = self.fixture.build()
        self.assertEqual(len(read_names), 3)
        self.assertEqual(len(contexts), 48)
        self.assertEqual(inventory["boundary_checks"], 3 * 8 * 34)
        self.assertFalse(inventory["query_gold_loaded"])
        self.assertFalse(inventory["test_content_read"])
        self.assertFalse(inventory["model_forward_executed"])
        self.assertLess(inventory["limits"]["max_full_sequence_tokens"] + 64, 8192)
        self.assertEqual([(r["task"], r["condition"]) for r in contexts[::3]],
                         [(task, condition) for task in coverage.TASKS for condition in coverage.CONDITIONS])
        for row in contexts:
            self.assertEqual(row["context_sha256"], canonical_json_sha256({k: v for k, v in row.items() if k != "context_sha256"}))
            self.assertTrue(row["control_valid"])
            if row["condition"] in ("PLnew", "PD") and row["trace"]["placebo"]["source_tokens"] is not None:
                placebo = row["trace"]["placebo"]
                self.assertLessEqual(placebo["absolute_difference"], placebo["allowed_difference"])

    def test_same_id_payload_mismatch_fails(self):
        self.fixture.lexicon_rows[0]["trace"]["selected_hits"][0]["senses"][0]["definition"] = "changed"
        self.fixture.save_resources()
        with self.assertRaisesRegex(PackageError, "matching does not replay"):
            self.fixture.build()

    def test_demo_content_answer_and_order_drift_fail(self):
        original = copy.deepcopy(self.fixture.retrieval)
        for field, changed in (("content", "changed text"), ("quadruples", []), ("prompt_rank", 8)):
            with self.subTest(field=field):
                self.fixture.retrieval = copy.deepcopy(original)
                self.fixture.retrieval[0]["demos"][0][field] = changed
                self.fixture.save_resources()
                with self.assertRaises(PackageError):
                    self.fixture.build()

    def test_parent_mutation_and_catalog_drift_fail(self):
        self.fixture.old[0]["messages"][0]["content"] += "changed"
        with self.assertRaisesRegex(PackageError, "parent context hash"):
            self.fixture.build()
        self.fixture.old[0]["messages"][0]["content"] = self.fixture.old[0]["messages"][0]["content"][:-7]
        self.fixture.plan["catalog"]["hate"][0]["answer_token_ids"] = [4]
        with self.assertRaisesRegex(PackageError, "candidate catalog changed"):
            self.fixture.build()

    def test_token_boundary_change_fails(self):
        class BrokenTokenizer(Tokenizer):
            def encode(self, text, add_special_tokens=False):
                tokens = super().encode(text, add_special_tokens)
                return tokens + [999] if "assistant:" in text and text.endswith('"hate"') else tokens

        with self.assertRaisesRegex(PackageError, "token boundary changed"):
            self.fixture.build(BrokenTokenizer())

    def test_padding_budget_is_enforced_without_trimming(self):
        with mock.patch.object(coverage, "MAX_SEQUENCE_TOKENS", 200):
            with self.assertRaisesRegex(PackageError, "sequence plus padding exceeds budget"):
                self.fixture.build()

    def test_invalid_neutral_control_fails_closed(self):
        original = coverage.render_condition

        def invalid(*args, **kwargs):
            result = original(*args, **kwargs)
            if args[1] == "PL":
                result.update(control_valid=False, control_status="invalid")
            return result

        with mock.patch.object(coverage, "render_condition", side_effect=invalid):
            with self.assertRaisesRegex(PackageError, "neutral-control construction failed"):
                self.fixture.build()


class CoverageBoundaryTests(unittest.TestCase):
    def test_lexicographic_ties_are_chosen_before_deduplication(self):
        rows = [
            {"query_id": "2", "union_count": 9, "dictionary_tokens": 100, "max_full_sequence_tokens": 1000},
            {"query_id": "10", "union_count": 9, "dictionary_tokens": 100, "max_full_sequence_tokens": 1000},
            {"query_id": "3", "union_count": 0, "dictionary_tokens": 0, "max_full_sequence_tokens": 100},
        ]
        selected, records = coverage.select_boundary_queries(rows, {"regression": ["10"], "validation": []})
        self.assertEqual(selected, ["3"])
        self.assertEqual([row["representative_query_id"] for row in records], ["3", "10", "10", "10"])
        self.assertEqual(records[1]["reason"], "already-in-original-preflight")
        selected, records = coverage.select_boundary_queries(rows, {"regression": [], "validation": []})
        self.assertEqual(selected, ["3", "10"])
        self.assertEqual(records[2]["reason"], "already-selected-boundary")

    def test_missing_empty_union_is_recorded_without_substituting(self):
        rows = [{"query_id": "1", "union_count": 1, "dictionary_tokens": 8, "max_full_sequence_tokens": 100}]
        selected, records = coverage.select_boundary_queries(rows, {"regression": [], "validation": []})
        self.assertEqual(selected, ["1"])
        self.assertEqual(records[0]["reason"], "no-empty-union")
        self.assertIsNone(records[0]["representative_query_id"])

    def test_overlapping_original_cohorts_fail(self):
        rows = [{"query_id": "1", "union_count": 0, "dictionary_tokens": 0, "max_full_sequence_tokens": 10}]
        with self.assertRaisesRegex(PackageError, "overlap"):
            coverage.select_boundary_queries(rows, {"regression": ["1"], "validation": ["1"]})


if __name__ == "__main__":
    unittest.main()
