import math
import unittest
from types import SimpleNamespace

import torch

from metrics.stage1_margin import (
    minimal_overlap_cover,
    prepare_field_pair,
    prepare_score_input,
    replace_one_field,
    score_field_margin,
    score_inputs,
)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        ids = [2 + (ord(char) % 120) for char in text]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


class DeterministicModel(torch.nn.Module):
    def __init__(self, vocab_size=128):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, position_ids):
        vocab = torch.arange(self.vocab_size, device=input_ids.device, dtype=torch.float32)
        logits = vocab.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        # Depend on position without changing relative logits; this exercises the
        # explicit position-id surface while retaining easy batch invariance.
        logits = logits + position_ids.unsqueeze(-1).float() * 0.001 + self.anchor
        return SimpleNamespace(logits=logits)


def gold_quad():
    return [
        {
            "target": "甲",
            "argument": "论点",
            "targeted_group": ["Racism"],
            "hateful": "hate",
        }
    ]


class SpanMaskTests(unittest.TestCase):
    def test_minimal_overlap_and_boundary_crossing(self):
        indices, left, right = minimal_overlap_cover([(0, 3), (3, 5), (5, 5)], (1, 4))
        self.assertEqual(indices, (0, 1))
        self.assertTrue(left)
        self.assertTrue(right)

    def test_full_response_is_tokenized_before_token_crop(self):
        tokenizer = CharacterTokenizer()
        item = prepare_score_input(
            tokenizer=tokenizer,
            rendered_chat_prompt="P",
            canonical_response='{"x":"abc","suffix":99}',
            character_span=(5, 10),
            max_sequence_tokens=100,
        )
        self.assertEqual(len(item.response_ids), len(item.response_text))
        self.assertLess(len(item.cropped_input_ids), len(item.prompt_ids) + len(item.response_ids))
        self.assertEqual(len(item.cropped_input_ids), item.global_token_indices[-1] + 1)

    def test_overflow_is_not_truncated(self):
        with self.assertRaises(OverflowError):
            prepare_score_input(
                tokenizer=CharacterTokenizer(),
                rendered_chat_prompt="prompt",
                canonical_response='["long"]',
                character_span=(1, 7),
                max_sequence_tokens=4,
            )


class CounterfactualTests(unittest.TestCase):
    def test_exactly_one_field_must_change(self):
        foil = replace_one_field(gold_quad(), tuple_index=0, field="hateful", candidate_value="non-hate")
        self.assertEqual(foil[0].hateful, "non-hate")
        self.assertEqual(foil[0].targeted_group, ("Racism",))
        with self.assertRaisesRegex(ValueError, "exactly"):
            replace_one_field(gold_quad(), tuple_index=0, field="hateful", candidate_value="hate")

    def test_gold_and_foil_use_complete_canonical_value_spans(self):
        gold_input, foil_input = prepare_field_pair(
            tokenizer=CharacterTokenizer(),
            rendered_chat_prompt="PROMPT",
            gold=gold_quad(),
            tuple_index=0,
            field="targeted_group",
            candidate_value=["Sexism", "Racism"],
        )
        gold_literal = gold_input.response_text[slice(*gold_input.character_span)]
        foil_literal = foil_input.response_text[slice(*foil_input.character_span)]
        self.assertEqual(gold_literal, '["Racism"]')
        self.assertEqual(foil_literal, '["Racism","Sexism"]')


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.model = DeterministicModel()

    def test_batched_and_unbatched_scores_are_equal(self):
        items = prepare_field_pair(
            tokenizer=self.tokenizer,
            rendered_chat_prompt="PROMPT",
            gold=gold_quad(),
            tuple_index=0,
            field="argument",
            candidate_value="观点",
        )
        batched = score_inputs(model=self.model, score_inputs=items, pad_token_id=0)
        unbatched = [
            score_inputs(model=self.model, score_inputs=[item], pad_token_id=0)[0]
            for item in items
        ]
        for left, right in zip(batched, unbatched, strict=True):
            self.assertEqual(left["token_ids"], right["token_ids"])
            self.assertEqual(left["token_logprobs"], right["token_logprobs"])
            self.assertEqual(left["mean_logprob"], right["mean_logprob"])

    def test_field_margin_records_mean_and_sum(self):
        result = score_field_margin(
            model=self.model,
            tokenizer=self.tokenizer,
            rendered_chat_prompt="PROMPT",
            gold=gold_quad(),
            tuple_index=0,
            field="hateful",
            candidate_value="non-hate",
        )
        self.assertTrue(math.isfinite(result["mean_margin"]))
        self.assertTrue(math.isfinite(result["sum_margin_sensitivity"]))
        self.assertGreater(result["gold"]["token_count"], 0)
        self.assertGreater(result["counterfactual"]["token_count"], 0)

    def test_fp32_log_softmax_matches_manual_token_sum(self):
        item = prepare_field_pair(
            tokenizer=self.tokenizer,
            rendered_chat_prompt="PROMPT",
            gold=gold_quad(),
            tuple_index=0,
            field="target",
            candidate_value="乙",
        )[0]
        result = score_inputs(
            model=self.model,
            score_inputs=[item],
            pad_token_id=self.tokenizer.pad_token_id,
        )[0]
        normalizer = float(
            torch.logsumexp(torch.arange(128, dtype=torch.float32), dim=0).item()
        )
        expected = sum(token_id - normalizer for token_id in result["token_ids"])
        self.assertLessEqual(abs(result["sum_logprob"] - expected), 1e-5)
        self.assertLessEqual(
            abs(result["mean_logprob"] - expected / result["token_count"]),
            1e-5,
        )


if __name__ == "__main__":
    unittest.main()
