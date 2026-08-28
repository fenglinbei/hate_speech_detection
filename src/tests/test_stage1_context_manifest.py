import copy
import unittest

from data.context_manifest import (
    ContextManifestError,
    ContextOverflow,
    finalize_context_budget,
    render_condition_item,
    render_sft_item,
    stable_demo_id,
    stable_lexicon_id,
    stable_term_evidence_id,
)


class CharacterTokenizer:
    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, enable_thinking=False):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "".join(f"<{row['role']}>{row['content']}" for row in conversation) + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text)


SYSTEM = "quad-json"
USER = "L={lexicons}\nD={examples}\nQ={text}"


def fixture():
    lex_id = "lex:v2:" + "a" * 64
    demo_ids = ["demo:v1:" + character * 64 for character in "bcd"]
    lexicons = {lex_id: {"lexicon_id": lex_id, "rendered_block": "LEX", "rendered_block_sha256": None}}
    demos = {
        demo_id: {"demo_id": demo_id, "rendered_block": "DEMO-" + str(index) * 20, "rendered_block_sha256": None}
        for index, demo_id in enumerate(demo_ids)
    }
    record = {
        "context_build_id": "ctx-test",
        "query": {"id": "1", "content": "QUERY", "gold": []},
        "selection": {
            "lexicons": {"prompt_order_before_budget": [lex_id]},
            "demos": {"prompt_order_before_budget": demo_ids},
        },
    }
    return record, lexicons, demos, lex_id, demo_ids


class ContextBudgetTests(unittest.TestCase):
    def test_cld_is_trimmed_once_and_other_conditions_are_subsets(self):
        record, lexicons, demos, lex_id, demo_ids = fixture()
        tokenizer = CharacterTokenizer()
        # Force at least one demo to be removed while retaining a non-empty prefix.
        full = finalize_context_budget(
            record,
            lexicon_catalog=lexicons,
            demo_catalog=demos,
            system_prompt=SYSTEM,
            user_prompt_template=USER,
            tokenizer=tokenizer,
            max_sequence_tokens=125,
            completion_reserve_tokens=20,
        )
        final_ids = full["selection"]["demos"]["prompt_order_final"]
        self.assertEqual(final_ids, demo_ids[: len(final_ids)])
        self.assertLess(len(final_ids), len(demo_ids))
        self.assertEqual(full["conditions"]["CL"]["lexicon_ids"], [lex_id])
        self.assertEqual(full["conditions"]["CL"]["demo_ids"], [])
        self.assertEqual(full["conditions"]["CD"]["demo_ids"], final_ids)
        self.assertEqual(full["conditions"]["CLD"]["demo_ids"], final_ids)
        self.assertEqual(full["conditions"]["C0"]["lexicon_ids"], [])

        item = render_condition_item(full, "CD")
        self.assertEqual(item["context_manifest"]["demo_ids"], final_ids)
        sft = render_sft_item(full, "CD")
        self.assertEqual(sft["output"], "[]")
        self.assertEqual(sft["metadata"]["output_protocol"], "quad-json-v1")

    def test_base_plus_lexicon_overflow_hard_fails(self):
        record, lexicons, demos, _, _ = fixture()
        only_lex = copy.deepcopy(lexicons)
        next(iter(only_lex.values()))["rendered_block"] = "X" * 500
        with self.assertRaises(ContextOverflow):
            finalize_context_budget(
                record,
                lexicon_catalog=only_lex,
                demo_catalog=demos,
                system_prompt=SYSTEM,
                user_prompt_template=USER,
                tokenizer=CharacterTokenizer(),
                max_sequence_tokens=100,
                completion_reserve_tokens=20,
            )

    def test_condition_or_record_tampering_is_detected(self):
        record, lexicons, demos, _, _ = fixture()
        frozen = finalize_context_budget(
            record,
            lexicon_catalog=lexicons,
            demo_catalog=demos,
            system_prompt=SYSTEM,
            user_prompt_template=USER,
            tokenizer=CharacterTokenizer(),
            max_sequence_tokens=300,
            completion_reserve_tokens=20,
        )
        tampered = copy.deepcopy(frozen)
        tampered["conditions"]["CD"]["demo_ids"] = []
        with self.assertRaises(ContextManifestError):
            render_condition_item(tampered, "CD")


class StableIdTests(unittest.TestCase):
    def test_stable_ids_are_full_sha256_and_content_sensitive(self):
        demo = stable_demo_id(source_record_id="7", content_sha256="a" * 64, gold_sha256="b" * 64)
        legacy_lex = stable_lexicon_id(
            term="x", category="Racism", definition="d", variants=["y"]
        )
        lex = stable_term_evidence_id(
            term="x",
            definition="d",
            variants=["y"],
            usage_notes="usage",
            ambiguity_notes="ambiguity",
        )
        self.assertRegex(demo, r"^demo:v1:[0-9a-f]{64}$")
        self.assertRegex(legacy_lex, r"^lex:v1:[0-9a-f]{64}$")
        self.assertRegex(lex, r"^lex:v2:[0-9a-f]{64}$")
        self.assertNotEqual(
            lex,
            stable_term_evidence_id(
                term="x",
                definition="changed",
                variants=["y"],
                usage_notes="usage",
                ambiguity_notes="ambiguity",
            ),
        )


if __name__ == "__main__":
    unittest.main()
