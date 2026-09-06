import copy
import hashlib
import math
import unittest
from types import SimpleNamespace

import torch

from data.stage1_data import canonical_json_sha256
from diagnostics.general_model_numeric_kernel import NumericKernelError, score_batch


class CharacterTokenizer:
    eos_token_id = 127
    pad_token_id = 0

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("special tokens must not be added")
        return [ord(char) - 31 for char in text]


class CausalFixtureModel:
    def __init__(self, corrupt=None):
        self.calls = []
        self.corrupt = corrupt

    def __call__(self, *, input_ids, attention_mask, use_cache):
        self.calls.append((input_ids.clone(), attention_mask.clone(), use_cache))
        vocabulary = torch.arange(128)[None, None, :]
        prefix = (input_ids * attention_mask).cumsum(dim=1)
        positions = torch.arange(input_ids.shape[1])[None, :]
        preferred = (prefix * 3 + positions * 7) % 128
        logits = (-(vocabulary - preferred[:, :, None]).abs() / 4).to(torch.bfloat16)
        if self.corrupt is not None:
            self.corrupt(logits)
        self.last_logits = logits
        return SimpleNamespace(logits=logits)


def runner(corrupt=None):
    return SimpleNamespace(
        tokenizer=CharacterTokenizer(), model=CausalFixtureModel(corrupt),
        device=torch.device("cpu"), config={"runtime": {"max_sequence_tokens": 8192}},
        eos_ids={127},
    )


def item(prompt="PROMPT:", answer='"hate"'):
    tokenizer = CharacterTokenizer()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    return {
        "context": {
            "prompt_text": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
            "prompt_tokens": len(prompt_ids), "control_valid": True, "overflow": False,
        },
        "candidate": {
            "canonical_answer": answer,
            "canonical_answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "answer_token_ids": answer_ids,
            "answer_token_ids_sha256": canonical_json_sha256(answer_ids),
            "answer_tokens": len(answer_ids),
        },
    }


class NumericKernelTests(unittest.TestCase):
    def test_causal_shift_and_eos_match_independent_full_softmax(self):
        model_runner = runner()
        example = item(answer='["Region","others"]')
        result = score_batch(model_runner, [example])[0]
        prompt_length = example["context"]["prompt_tokens"]
        target_ids = example["candidate"]["answer_token_ids"] + [127]
        expected = []
        for offset, token_id in enumerate(target_ids):
            logits = model_runner.model.last_logits[0, prompt_length - 1 + offset].float()
            expected.append(logits.log_softmax(dim=0)[token_id].item())
        torch.testing.assert_close(torch.tensor(result["token_logprobs"]), torch.tensor(expected[:-1]))
        self.assertAlmostEqual(result["eos_logprob"], expected[-1], places=5)
        self.assertAlmostEqual(result["answer_sum"], math.fsum(expected[:-1]), places=4)
        self.assertAlmostEqual(result["total_with_eos"], math.fsum(expected), places=4)
        self.assertEqual(result["answer_mean"], result["answer_sum"] / len(target_ids[:-1]))
        self.assertEqual(result["mean_with_eos"], result["total_with_eos"] / len(target_ids))
        self.assertNotEqual(result["answer_sum"], result["total_with_eos"])
        self.assertTrue(result["finite_target_logits_checked"])
        self.assertFalse(result["reference_checked"])

    def test_right_padding_does_not_score_prompt_or_padding_and_matches_singletons(self):
        examples = [item("A:", "[]"), item("LONG PROMPT:", '"non-hate"'),
                    item("B:", '["Racism"]'), item("OTHER:", '["Sexism"]')]
        model_runner = runner()
        observed = score_batch(model_runner, examples)
        inputs, mask, use_cache = model_runner.model.calls[0]
        self.assertFalse(use_cache)
        self.assertEqual(inputs.shape[0], 4)
        for index, example in enumerate(examples):
            expected_sequence = model_runner.tokenizer.encode(
                example["context"]["prompt_text"] + example["candidate"]["canonical_answer"],
                add_special_tokens=False,
            ) + [127]
            length = len(expected_sequence)
            self.assertEqual(inputs[index, :length].tolist(), expected_sequence)
            self.assertEqual(inputs[index, length:].tolist(), [0] * (inputs.shape[1] - length))
            self.assertEqual(mask[index].tolist(), [1] * length + [0] * (inputs.shape[1] - length))
            singleton = score_batch(runner(), [example])[0]
            for key in ("token_logprobs", "answer_sum", "answer_mean", "eos_logprob", "total_with_eos"):
                self.assertEqual(observed[index][key], singleton[key])

    def test_cpu_float64_reference_uses_the_same_bfloat16_logits(self):
        model_runner = runner()
        example = item(answer='["Region"]')
        result = score_batch(model_runner, [example], reference=True)[0]
        expected = []
        for offset, token_id in enumerate(example["candidate"]["answer_token_ids"] + [127]):
            vector = model_runner.model.last_logits[0, example["context"]["prompt_tokens"] - 1 + offset].tolist()
            maximum = max(vector)
            expected.append(vector[token_id] - maximum - math.log(math.fsum(math.exp(x - maximum) for x in vector)))
        for observed, target in zip(result["reference_token_logprobs"] + [result["reference_eos_logprob"]], expected):
            self.assertAlmostEqual(observed, target, places=12)
        self.assertLess(result["reference_abs_error_max"], 1e-4)
        self.assertEqual(result["model_logits_dtype"], "torch.bfloat16")
        self.assertEqual(len(model_runner.model.calls), 1)
        self.assertEqual(result["reference_scores"]["answer_sum"], math.fsum(result["reference_token_logprobs"]))
        self.assertAlmostEqual(result["reference_differences"]["answer_sum"],
                               result["answer_sum"] - math.fsum(expected[:-1]), places=12)

    def test_all_vocabulary_logits_at_each_target_must_be_finite(self):
        example = item()
        for bad in (math.nan, math.inf, -math.inf):
            for position in (example["context"]["prompt_tokens"] - 1,
                             example["context"]["prompt_tokens"] + example["candidate"]["answer_tokens"] - 1):
                with self.subTest(value=bad, position=position):
                    def corrupt(logits):
                        logits[0, position, 126] = bad
                    with self.assertRaisesRegex(NumericKernelError, "non-finite logits"):
                        score_batch(runner(corrupt), [example])

    def test_unscored_prompt_final_eos_and_padding_logits_are_outside_finite_scope(self):
        examples = [item("SHORT:", "[]"), item("LONGER PROMPT:", '"non-hate"')]
        length = examples[0]["context"]["prompt_tokens"] + examples[0]["candidate"]["answer_tokens"] + 1
        def corrupt(logits):
            logits[0, 0, :] = math.nan
            logits[0, length - 1:, :] = math.nan
        observed = score_batch(runner(corrupt), examples)
        self.assertEqual(observed[0]["token_logprobs"], score_batch(runner(), [examples[0]])[0]["token_logprobs"])

    def test_frozen_context_identity_mismatch_is_rejected_before_forward(self):
        for key, value in (("prompt_sha256", "wrong"), ("prompt_token_ids_sha256", "wrong"),
                           ("prompt_tokens", 1), ("overflow", True), ("control_valid", False)):
            with self.subTest(key=key):
                example, model_runner = item(), runner()
                example["context"][key] = value
                with self.assertRaises(NumericKernelError):
                    score_batch(model_runner, [example])
                self.assertEqual(model_runner.model.calls, [])

    def test_frozen_candidate_identity_mismatch_is_rejected(self):
        for key, value in (("answer_token_ids", [1]), ("answer_token_ids_sha256", "wrong"),
                           ("answer_tokens", 1), ("canonical_answer_sha256", "wrong")):
            with self.subTest(key=key):
                example = item()
                example["candidate"][key] = value
                with self.assertRaisesRegex(NumericKernelError, "candidate"):
                    score_batch(runner(), [example])

    def test_nonconcatenative_token_boundary_is_rejected(self):
        example, model_runner = item(), runner()
        original_encode = model_runner.tokenizer.encode
        joint = example["context"]["prompt_text"] + example["candidate"]["canonical_answer"]
        def unstable(text, *, add_special_tokens):
            return [23] if text == joint else original_encode(text, add_special_tokens=add_special_tokens)
        model_runner.tokenizer.encode = unstable
        with self.assertRaisesRegex(NumericKernelError, "boundary"):
            score_batch(model_runner, [example])
        self.assertEqual(model_runner.model.calls, [])

    def test_overflow_includes_eos_and_honors_hard_limit(self):
        example, model_runner = item("P", "A"), runner()
        model_runner.config["runtime"]["max_sequence_tokens"] = 2
        with self.assertRaisesRegex(NumericKernelError, "3 > 2"):
            score_batch(model_runner, [example])
        model_runner.config["runtime"]["max_sequence_tokens"] = 16384
        with self.assertRaisesRegex(NumericKernelError, "8193 > 8192"):
            score_batch(model_runner, [item("P" * 8191, "A")])
        self.assertEqual(model_runner.model.calls, [])

    def test_empty_answer_prompt_ambiguous_eos_and_oversized_batch_are_rejected(self):
        for example in (item(answer=""), item(prompt="")):
            with self.assertRaisesRegex(NumericKernelError, "nonempty"):
                score_batch(runner(), [example])
        model_runner = runner()
        model_runner.eos_ids = {126, 127}
        with self.assertRaisesRegex(NumericKernelError, "unambiguous"):
            score_batch(model_runner, [item()])
        with self.assertRaisesRegex(NumericKernelError, "at most four"):
            score_batch(runner(), [item()] * 5)
        self.assertEqual(score_batch(None, []), [])

    def test_gold_fields_do_not_affect_scoring(self):
        original = item()
        changed = copy.deepcopy(original)
        changed["context"]["gold"] = {"hate": "non-hate", "group": ["Region"]}
        expected, observed = score_batch(runner(), [original])[0], score_batch(runner(), [changed])[0]
        for key in ("token_logprobs", "eos_logprob", "answer_sum", "answer_mean", "total_with_eos"):
            self.assertEqual(expected[key], observed[key])

    def test_cpu_timings_are_recorded_without_cuda_memory_measurement(self):
        rows = score_batch(runner(), [item(), item(answer="[]")], reference=True)
        self.assertEqual(rows[0]["forward_seconds"], rows[1]["forward_seconds"])
        for row in rows:
            self.assertGreater(row["forward_seconds"], 0.0)
            self.assertGreater(row["normalization_seconds"], 0.0)
            self.assertGreater(row["reference_seconds"], 0.0)
            self.assertIsNone(row["peak_memory_allocated_bytes"])


if __name__ == "__main__":
    unittest.main()
