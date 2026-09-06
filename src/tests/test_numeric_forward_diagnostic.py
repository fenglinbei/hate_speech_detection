import copy
import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from data.stage1_data import canonical_json_sha256
from diagnostics.general_model_numeric_analysis import candidate_catalog
from diagnostics.general_model_package import read_json
from diagnostics.numeric_forward_diagnostic import (
    _head_to_float32, fixed_padding_lengths, forward_probe, run, vector_differences,
)


class Tokenizer:
    eos_token_id = 127
    pad_token_id = 0

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(char) - 31 for char in text]


def example(answer='["Racism"]', *, record_id="1322:group:CD", prompt="INPUT:"):
    query_id, task, condition = record_id.split(":")
    tokenizer = Tokenizer()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    context = {"record_id": record_id, "query_id": query_id, "task": task, "condition": condition,
               "prompt_text": prompt, "prompt_tokens": len(prompt_ids),
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
               "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
               "control_valid": True, "overflow": False}
    context["context_sha256"] = canonical_json_sha256(context)
    candidate = {"candidate_id": answer, "canonical_answer": answer,
                 "answer_token_ids": answer_ids, "answer_tokens": len(answer_ids),
                 "answer_token_ids_sha256": canonical_json_sha256(answer_ids)}
    return {"context": context, "candidate": candidate}


class ForwardDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=16, max_position_embeddings=256, pad_token_id=0,
                             eos_token_id=127, use_cache=False, attention_dropout=0.0,
                             attn_implementation="eager")
        with patch("torch.cuda.is_available", return_value=False):
            cls.model = Qwen3ForCausalLM(config).eval()

    def runner(self, *, dtype=torch.float32):
        return SimpleNamespace(model=copy.deepcopy(self.model).to(dtype=dtype), tokenizer=Tokenizer(),
                               eos_ids={127}, device=torch.device("cpu"), identity={"device": "cpu", "model": "tiny"},
                               config={"runtime": {"max_sequence_tokens": 8192}})

    def test_fixed_geometry_uses_all_six_conditions_and_largest_candidate(self):
        contexts = [example(prompt="X" * (10 + index), record_id=f"1322:group:{condition}")["context"]
                    for index, condition in enumerate(("C0", "CL", "CD", "CLD", "PL", "PD"))]
        contexts.append(example(record_id="1322:hate:C0", prompt="X" * 25)["context"])
        result = fixed_padding_lengths(contexts, {"group": [{"answer_tokens": 1}, {"answer_tokens": 16}],
                                                 "hate": [{"answer_tokens": 3}, {"answer_tokens": 5}]})
        self.assertEqual(result, {("1322", "group"): 32, ("1322", "hate"): 31})
        with self.assertRaisesRegex(ValueError, "8192"):
            fixed_padding_lengths([{**contexts[0], "prompt_tokens": 8190}], {"group": [{"answer_tokens": 4}]})

    def test_prefix_probe_excludes_target_and_future_tokens(self):
        item = example()
        rows, states = forward_probe(self.runner(), [item], prefix_only=True, probe_index=2, capture_layers=True)
        row = rows[0]
        self.assertEqual(row["sequence_tokens"], item["context"]["prompt_tokens"] + 2)
        self.assertEqual(row["scored_target_ids"], [item["candidate"]["answer_token_ids"][2]])
        self.assertEqual(row["probe_logit_position"], row["sequence_tokens"] - 1)
        self.assertIsNone(row["answer_sum"])
        self.assertEqual(set(states[0]), {"embedding", "layer_00", "layer_01", "lm_head_input"})

    def test_full_and_prefix_fixed_padding_match_at_the_identical_causal_position(self):
        runner, item = self.runner(), example()
        full, hidden = forward_probe(runner, [item], pad_to=64, probe_index=2, capture_layers=True)
        prefix, reference = forward_probe(runner, [item], pad_to=64, prefix_only=True,
                                         probe_index=2, capture_layers=True)
        self.assertEqual(full[0]["probe_logprob"], prefix[0]["probe_logprob"])
        self.assertEqual(full[0]["probe_target_logit"], prefix[0]["probe_target_logit"])
        self.assertTrue(all(value["max_abs"] == 0 for value in vector_differences(reference[0], hidden[0]).values()))
        self.assertNotEqual(full[0]["input_ids_sha256"], prefix[0]["input_ids_sha256"])
        self.assertNotEqual(full[0]["attention_mask_sha256"], prefix[0]["attention_mask_sha256"])

    def test_batch_padding_preserves_targets_and_hooks_are_removed(self):
        runner = self.runner()
        items = [example("[]"), example('["Region"]'), example('["Racism","others"]'), example('["Sexism"]')]
        batched, hidden = forward_probe(runner, items, pad_to=64, capture_layers=True)
        for item, observed in zip(items, batched):
            baseline, _ = forward_probe(runner, [item], pad_to=64)
            self.assertEqual(observed["scored_target_ids"], item["candidate"]["answer_token_ids"] + [127])
            self.assertLess(max(abs(a - b) for a, b in zip(observed["target_logprobs"], baseline[0]["target_logprobs"])), 1e-5)
            self.assertEqual(observed["effective_batch_size"], 4)
            self.assertIsNone(observed["peak_memory_allocated_bytes"])
        self.assertEqual(len(runner.model.lm_head._forward_pre_hooks), 0)
        self.assertTrue(all(len(layer._forward_hooks) == 0 for layer in runner.model.model.layers))

    def test_fp32_head_preserves_bfloat16_trunk_and_records_head_output_dtype(self):
        runner = self.runner(dtype=torch.bfloat16)
        handle = _head_to_float32(runner.model)
        try:
            rows, _ = forward_probe(runner, [example()], pad_to=64)
            self.assertEqual(runner.model.model.embed_tokens.weight.dtype, torch.bfloat16)
            self.assertEqual(runner.model.lm_head.weight.dtype, torch.float32)
            self.assertEqual(rows[0]["model_logits_dtype"], "torch.float32")
        finally:
            handle.remove()

    def test_vectors_report_known_localized_error(self):
        delta = vector_differences({"layer_00": torch.tensor([1.0, 2.0])},
                                   {"layer_00": torch.tensor([1.0, 4.0])})
        self.assertEqual(delta["layer_00"]["max_abs"], 2.0)
        self.assertAlmostEqual(delta["layer_00"]["rms"], 2 ** 0.5)
        with self.assertRaisesRegex(ValueError, "keys"):
            vector_differences({"a": torch.tensor([1.0])}, {"b": torch.tensor([1.0])})

    def test_end_to_end_diagnostic_writes_separate_manifest_without_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic"
            item = example()
            catalog = candidate_catalog()
            tokenizer = Tokenizer()
            for candidates in catalog.values():
                for candidate in candidates:
                    ids = tokenizer.encode(candidate["canonical_answer"], add_special_tokens=False)
                    candidate.update(answer_token_ids=ids, answer_tokens=len(ids),
                                     answer_token_ids_sha256=canonical_json_sha256(ids))
            plan = {"plan_id": "synthetic", "package_path": "unused", "catalog": catalog,
                    "cohorts": {"regression": ["1322"]}}
            args = SimpleNamespace(plan=Path("unused"), contexts=["1322:group:CD"],
                                   output=output, cpu_threads=2, disable_bf16_reduction=True,
                                   dtype="float32", device="cpu", head_fp32=False,
                                   candidate_ids=["group-01", "group-02"], padding="both",
                                   batch_sizes=[1, 2], capture_layers=True, probe_token_index=1,
                                   skip_prefix=False)
            with patch("diagnostics.numeric_forward_diagnostic.load_plan", return_value=(plan, [item["context"]])), \
                    patch("diagnostics.numeric_forward_diagnostic.LocalRunner", return_value=self.runner()), \
                    patch("torch.cuda.empty_cache"), redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["scientific_result"])
            self.assertFalse(result["query_gold_loaded"])
            self.assertFalse(result["test_content_read"])
            self.assertEqual(len(result["files"]), 5)
            fixed = read_json(output / "1322-group-CD-fixed-b2.json")["records"]
            self.assertTrue(all("prefix_reference_differences" in row and "vs_dynamic_batch1" in row for row in fixed))


if __name__ == "__main__":
    unittest.main()
