from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from finetune.train import (
    CustomTrainer,
    STAGE1_EARLY_STOPPING_POLICY,
    calibration_bucket,
    encode_training_example,
    get_early_stopping_settings,
    split_train_only_calibration,
    validate_early_stopping_training_args,
    validate_stage1_data_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(relative_path: str) -> dict:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        self.last_template_kwargs = kwargs
        return "rendered-prompt"

    def __call__(self, text, add_special_tokens=False):
        if text == "rendered-prompt":
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        ids = list(range(10, 10 + len(text)))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class CalibrationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.data_config = load_json("config/stage1/train_m_ld.json")["data"]

    def test_hash_partition_is_order_invariant_disjoint_and_exhaustive(self):
        rows = [
            {"id": str(index), "instruction": "s", "input": "q", "output": "[]"}
            for index in range(500)
        ]
        forward_fit, forward_calibration = split_train_only_calibration(
            pd.DataFrame(rows), self.data_config
        )
        reverse_fit, reverse_calibration = split_train_only_calibration(
            pd.DataFrame(list(reversed(rows))), self.data_config
        )

        fit_ids = set(forward_fit["id"])
        calibration_ids = set(forward_calibration["id"])
        self.assertFalse(fit_ids & calibration_ids)
        self.assertEqual(fit_ids | calibration_ids, {row["id"] for row in rows})
        self.assertEqual(fit_ids, set(reverse_fit["id"]))
        self.assertEqual(calibration_ids, set(reverse_calibration["id"]))

    def test_bucket_is_stable_and_policy_bound(self):
        calibration = self.data_config["calibration"]
        self.assertEqual(
            calibration_bucket("query-17", calibration),
            calibration_bucket("query-17", calibration),
        )
        changed = dict(calibration, salt="different")
        self.assertNotEqual(
            calibration_bucket("query-17", calibration),
            calibration_bucket("query-17", changed),
        )

    def test_scientific_dev_path_is_rejected(self):
        invalid = dict(self.data_config, val_data_path="data/dev.jsonl")
        with self.assertRaisesRegex(ValueError, "scientific dev"):
            split_train_only_calibration(
                pd.DataFrame(
                    [
                        {"id": str(index), "instruction": "s", "input": "q", "output": "[]"}
                        for index in range(100)
                    ]
                ),
                invalid,
            )


class GoldTokenPolicyTests(unittest.TestCase):
    def test_structured_gold_uses_compact_json_not_python_repr(self):
        tokenizer = FakeTokenizer()
        value = [
            {
                "target": "甲",
                "argument": None,
                "targeted_group": ["Racism"],
                "hateful": "hate",
            }
        ]
        compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"))

        encoded = encode_training_example(
            {"id": "q-json", "instruction": "system", "input": "query", "output": value},
            tokenizer,
            max_length=256,
        )

        self.assertEqual(sum(label != -100 for label in encoded["labels"]), len(compact) + 1)

    def test_complete_gold_and_eos_are_supervised(self):
        tokenizer = FakeTokenizer()
        encoded = encode_training_example(
            {"id": "q1", "instruction": "system", "input": "query", "output": "[]"},
            tokenizer,
            max_length=16,
        )

        self.assertEqual(encoded["input_ids"], [1, 2, 3, 10, 11, 99])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 10, 11, 99])
        self.assertFalse(tokenizer.last_template_kwargs["enable_thinking"])

    def test_overflow_hard_fails_instead_of_truncating_gold(self):
        with self.assertRaisesRegex(ValueError, "gold truncation is forbidden"):
            encode_training_example(
                {"id": "q-overflow", "instruction": "system", "input": "query", "output": "abcdef"},
                FakeTokenizer(),
                max_length=7,
            )


class EarlyStoppingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_json("config/stage1/train_m_ld.json")

    def test_frozen_policy_and_training_arguments_are_consistent(self):
        settings = get_early_stopping_settings(self.config)
        self.assertIsNotNone(settings)
        for key, expected in STAGE1_EARLY_STOPPING_POLICY.items():
            self.assertEqual(settings[key], expected)
        self.assertEqual(
            validate_early_stopping_training_args(
                self.config,
                dict(self.config["training"]),
                save_inference_only=False,
            ),
            settings,
        )

    def test_stage1_cannot_claim_calibration_while_routing_to_dev(self):
        invalid = json.loads(json.dumps(self.config))
        invalid["data"]["selection_split"] = "scientific-dev"
        invalid["data"]["val_data_path"] = "data/dev.jsonl"
        with self.assertRaisesRegex(ValueError, "train-only-calibration"):
            validate_stage1_data_contract(invalid)

    def test_formal_runtime_is_blocked_until_schedule_loader_exists(self):
        self.assertEqual(
            self.config["execution_status"],
            "runtime-gated-by-immutable-training-refs",
        )
        with self.assertRaisesRegex(RuntimeError, "schedule-aware loader"):
            validate_stage1_data_contract(self.config, require_runtime_ready=True)

    def test_best_metric_requires_threshold_and_keeps_earliest_tie(self):
        trainer = object.__new__(CustomTrainer)
        trainer.early_stopping_config = get_early_stopping_settings(self.config)
        trainer.args = SimpleNamespace(metric_for_best_model="eval_loss")
        trainer.state = SimpleNamespace(best_metric=None, best_global_step=None, global_step=10)

        self.assertTrue(trainer._determine_best_metric({"eval_loss": 1.0}, None))
        self.assertEqual(trainer.state.best_global_step, 10)

        trainer.state.global_step = 20
        self.assertFalse(trainer._determine_best_metric({"eval_loss": 0.9995}, None))
        self.assertEqual(trainer.state.best_global_step, 10)

        trainer.state.global_step = 30
        self.assertTrue(trainer._determine_best_metric({"eval_loss": 0.998}, None))
        self.assertEqual(trainer.state.best_global_step, 30)

        trainer.state.global_step = 40
        self.assertFalse(trainer._determine_best_metric({"eval_loss": 0.998}, None))
        self.assertEqual(trainer.state.best_global_step, 30)


class FrozenProfileTests(unittest.TestCase):
    def test_roles_use_qwen3_full_finetune_and_deepspeed_zero3(self):
        m_ld = load_json("config/stage1/train_m_ld.json")
        m_drop = load_json("config/stage1/train_m_drop.json")
        deepspeed = load_json("config/stage1/deepspeed_zero3.json")

        for config in (m_ld, m_drop):
            self.assertEqual(config["model_name"], "Qwen/Qwen3-8B")
            self.assertEqual(config["model_path"], "models/base/Qwen3-8B")
            self.assertFalse(config["lora"])
            self.assertEqual(config["training"]["deepspeed"], "config/stage1/deepspeed_zero3.json")
            self.assertNotIn("fsdp", config["training"])
        self.assertEqual(deepspeed["zero_optimization"]["stage"], 3)
        self.assertTrue(
            deepspeed["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"]
        )
        self.assertEqual(m_ld["context_policy"]["lexicon_dropout_probability"], 0.0)
        self.assertEqual(m_drop["context_policy"]["lexicon_dropout_probability"], 0.5)
        self.assertEqual(m_drop["context_policy"]["demonstration_dropout_probability"], 0.5)
        self.assertTrue(m_drop["context_policy"]["dropout_independent"])

    def test_source_recipe_has_exact_matched_slots_without_future_model_refs(self):
        recipe = load_json("exps/specs/stage1_context_factorial.json")
        slots = recipe["ordered_model_slots"]
        self.assertEqual(
            recipe["execution_status"],
            "runtime-gated-by-immutable-training-refs",
        )
        self.assertEqual(len(slots), 6)
        self.assertEqual({slot["seed"] for slot in slots}, {42, 43, 44})
        self.assertEqual(
            recipe["pilot_slot_keys"],
            ["M_LD/seed-42", "M_drop/seed-42"],
        )
        serialized = json.dumps(recipe, sort_keys=True)
        self.assertNotIn("model_ref", serialized)
        self.assertNotIn("checkpoint_path", serialized)
        self.assertNotIn("schedule_build_id", serialized)

    def test_generation_prompt_and_completion_fit_model_budget(self):
        profile = load_json("config/stage1/generation_greedy.json")
        runtime = profile["model_runtime"]
        sampling = profile["sampling"]
        self.assertEqual(
            runtime["max_prompt_tokens"] + runtime["completion_reserve_tokens"],
            runtime["max_model_len"],
        )
        self.assertEqual(
            sampling["max_new_tokens"],
            runtime["completion_reserve_tokens"],
        )
        self.assertEqual(runtime["overflow_policy"], "hard-fail-no-truncation")


if __name__ == "__main__":
    unittest.main()
