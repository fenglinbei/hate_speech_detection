import hashlib
import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from data.stage1_data import canonical_json_sha256
from diagnostics.general_model_numeric_kernel import NumericKernelError
from diagnostics.general_model_numeric_kernel_v2 import score_batch, score_prefix_block


class Tokenizer:
    eos_token_id = 127
    pad_token_id = 0

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(c) - 31 for c in text]


class NumericKernelV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=16, max_position_embeddings=256,
                             eos_token_id=127, pad_token_id=0, attention_dropout=0.0,
                             attn_implementation="eager")
        model = Qwen3ForCausalLM(config).eval()
        tokenizer = Tokenizer()
        prompt = "PROMPT:"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        self.context = {
            "query_id": "1", "task": "group", "prompt_text": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_token_ids_sha256": canonical_json_sha256(ids),
            "prompt_tokens": len(ids), "control_valid": True, "overflow": False,
        }
        self.catalog = [{"canonical_answer": text} for text in ('["R"]', '["Region"]', '["R","X"]', '[]')]
        self.items = [{"context": self.context, "candidate": c} for c in self.catalog]
        self.runner = SimpleNamespace(
            model=model, tokenizer=tokenizer, eos_ids={127}, device=torch.device("cpu"),
            numeric_runtime={"padding_policy": "query-task-six-condition-max"},
            fixed_lengths={("1", "group"): 24}, global_length=32, padding_extra=0,
        )

    def assert_scores_close(self, left, right, tolerance=1e-4):
        for a, b in zip(left, right, strict=True):
            for key in ("answer_sum", "answer_mean", "eos_logprob", "total_with_eos", "mean_with_eos"):
                self.assertLessEqual(abs(a[key] - b[key]), tolerance)
            self.assertEqual(a["answer_token_ids"], b["answer_token_ids"])
            for x, y in zip(a["token_logprobs"], b["token_logprobs"], strict=True):
                self.assertLessEqual(abs(x - y), tolerance)

    def test_batch_prefix_padding_and_reference_agree(self):
        baseline = [score_batch(self.runner, [item], reference=True)[0] for item in self.items]
        batched = score_batch(self.runner, self.items)
        prefix = score_prefix_block(self.runner, self.context, self.catalog, reference=True)
        self.runner.padding_extra = 64
        padded = score_batch(self.runner, self.items)
        for other in (batched, prefix, padded):
            self.assert_scores_close(baseline, other)
        self.assertEqual(padded[0]["padded_sequence_tokens"], 88)
        self.assertTrue(all(row["reference_abs_error_max"] < 1e-4 for row in baseline + prefix))
        self.assertTrue(all(row["use_cache"] is False for row in prefix))
        self.assertTrue(all(row["prefix_padding"] is False for row in prefix))

    def test_selected_projection_matches_original_full_logits(self):
        from diagnostics.general_model_numeric_kernel import score_batch as full_logits_score
        self.runner.config = {"runtime": {"max_sequence_tokens": 8192}}
        self.runner.numeric_runtime["padding_policy"] = "dynamic"
        original = full_logits_score(self.runner, self.items)
        selected = score_batch(self.runner, self.items)
        self.assert_scores_close(original, selected)

    def test_permutation_restores_identical_scores(self):
        baseline = score_batch(self.runner, self.items)
        permuted = score_batch(self.runner, list(reversed(self.items)))
        self.assert_scores_close(baseline, list(reversed(permuted)))

    def test_rejects_invalid_geometry(self):
        with self.assertRaises(NumericKernelError):
            score_batch(self.runner, self.items + self.items[:1])
        self.runner.padding_extra = -1
        with self.assertRaises(NumericKernelError):
            score_batch(self.runner, self.items)
        self.runner.padding_extra = 8192
        with self.assertRaises(NumericKernelError):
            score_batch(self.runner, self.items)

    def test_empty_batch(self):
        self.assertEqual(score_batch(self.runner, []), [])

    def test_bfloat_transformer_float_head(self):
        self.runner.model.bfloat16()
        self.runner.model.lm_head.float()
        rows = score_batch(self.runner, self.items, reference=True)
        self.assertTrue(all(row["model_logits_dtype"] == "torch.float32" for row in rows))
        self.assertTrue(all(row["reference_abs_error_max"] < 1e-4 for row in rows))


if __name__ == "__main__":
    unittest.main()
