import json
import tempfile
import unittest
from pathlib import Path

from data.build_data import make_data
from data.config import Config
from metrics.metric_llm import BinaryClassificationMetrics
from prompt import (
    COLD_BINARY_PROMPT_USER_V2,
    COLD_BINARY_RAG_PROMPT_USER_V2_WO_EXAMPLES,
    COLD_BINARY_RAG_PROMPT_USER_V2_WO_LEX,
)
from utils.parser import parse_binary_label


class ColdBinaryParserTest(unittest.TestCase):
    def test_parse_binary_label_normalizes_common_outputs(self):
        cases = {
            "hate": "hate",
            " Hate\n": "hate",
            "1": "hate",
            "hateful": "hate",
            "non-hate": "non-hate",
            "non_hate": "non-hate",
            "non hate": "non-hate",
            "0": "non-hate",
        }
        for raw, expected in cases.items():
            self.assertEqual(parse_binary_label(raw), expected)

    def test_parse_binary_label_rejects_invalid_output(self):
        self.assertIsNone(parse_binary_label("maybe"))


class ColdBinaryMetricsTest(unittest.TestCase):
    def test_binary_metrics_count_invalid_as_wrong(self):
        rows = [
            {"gt_label": "hate", "pred_label": "hate", "status": "success"},
            {"gt_label": "hate", "pred_label": "non-hate", "status": "success"},
            {"gt_label": "non-hate", "pred_label": "non-hate", "status": "success"},
            {"gt_label": "non-hate", "pred_label": None, "status": "invalid"},
        ]

        metrics = BinaryClassificationMetrics().run(datas_list=rows)

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_precision"], 0.75)
        self.assertEqual(metrics["macro_recall"], 0.5)
        self.assertEqual(metrics["hate_f1"], 0.6667)
        self.assertEqual(metrics["non_hate_f1"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5834)
        self.assertEqual(metrics["confusion_matrix"]["non-hate"]["invalid"], 1)


def make_record(sample_id, content, label, metadata=None):
    return {
        "id": sample_id,
        "content": content,
        "metadata": metadata or {},
        "quadruples": [
            {
                "target": "NULL",
                "argument": content,
                "targeted_group": "Racism" if label == "hate" else "non-hate",
                "hateful": label,
            }
        ],
    }


class ColdBinaryBuildTest(unittest.TestCase):
    def _render_train_prompt(self, root: Path, prompt_template: str, retrieval_settings=None) -> str:
        out_dir = root / "out"
        cache_dir = root / "cache"
        train_path = root / "train.json"
        val_path = root / "val.json"
        test_path = root / "test.json"
        retrieval_settings = retrieval_settings or {
            "use_srag": False,
            "srag_top_k": 0,
            "use_lex": False,
        }

        train_path.write_text(
            json.dumps([make_record("train_1", "讨厌某群体", "hate")], ensure_ascii=False),
            encoding="utf-8",
        )
        val_path.write_text(
            json.dumps([make_record("val_1", "普通讨论", "non-hate")], ensure_ascii=False),
            encoding="utf-8",
        )
        test_path.write_text(
            json.dumps([make_record("test_1", "恶意泛化某群体", "hate")], ensure_ascii=False),
            encoding="utf-8",
        )

        config_path = root / "config.json"
        config_path.write_text(
            json.dumps({
                "task_type": "cold_binary",
                "data_paths": {
                    "raw_data_path": str(train_path),
                    "val_data_path": str(val_path),
                    "test_data_path": str(test_path),
                    "train_output_path": str(out_dir / "train.jsonl"),
                    "val_output_path": str(out_dir / "val.jsonl"),
                    "val_runner_output_path": str(out_dir / "val_runner.json"),
                    "test_output_path": str(out_dir / "test.json"),
                    "lexicon_data_path": "",
                    "tokenizer_path": None,
                },
                "prompt_templates": {
                    "prompt_template": prompt_template,
                    "example_template": "COLD_BINARY_EXAMPLE_PROMPT",
                    "system_prompt": "COLD_BINARY_SYSTEM_PROMPT",
                },
                "retrieval_settings": retrieval_settings,
                "training_settings": {
                    "auto_length": False,
                    "split_ratio": 0.9,
                },
                "cache_settings": {
                    "enable_build_cache": False,
                    "build_cache_dir": str(cache_dir),
                },
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        make_data(Config(str(config_path)))
        train_row = json.loads((out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        return train_row["input"]

    def test_v2_prompt_variants_omit_only_missing_resource_sections(self):
        lexicons = "关键词：坏词\n类别：Racism\n定义：测试背景"
        examples = "文本：示例文本\n标签：hate"

        no_examples = (
            COLD_BINARY_RAG_PROMPT_USER_V2_WO_EXAMPLES
            .replace("{lexicons}", lexicons)
            .replace("{text}", "待判断文本")
        )
        self.assertIn("背景知识：", no_examples)
        self.assertIn("关键词：坏词", no_examples)
        self.assertNotIn("示例：", no_examples)
        self.assertIn("边界规则", no_examples)
        self.assertIn("只输出一个标签：hate 或 non-hate", no_examples)

        no_lex = (
            COLD_BINARY_RAG_PROMPT_USER_V2_WO_LEX
            .replace("{examples}", examples)
            .replace("{text}", "待判断文本")
        )
        self.assertIn("示例：", no_lex)
        self.assertIn("示例文本", no_lex)
        self.assertNotIn("背景知识：", no_lex)
        self.assertIn("边界规则", no_lex)

        no_resources = COLD_BINARY_PROMPT_USER_V2.replace("{text}", "待判断文本")
        self.assertNotIn("背景知识：", no_resources)
        self.assertNotIn("示例：", no_resources)
        self.assertIn("边界规则", no_resources)
        self.assertIn("文本：待判断文本", no_resources)

    def test_make_data_writes_label_outputs_and_gt_label(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.json"
            val_path = root / "val.json"
            test_path = root / "test.json"
            out_dir = root / "out"
            cache_dir = root / "cache"

            train_path.write_text(
                json.dumps([
                    make_record("train_1", "讨厌某群体", "hate"),
                    make_record("train_2", "普通评论", "non-hate"),
                ], ensure_ascii=False),
                encoding="utf-8",
            )
            val_path.write_text(
                json.dumps([
                    make_record(
                        "val_1",
                        "友好讨论",
                        "non-hate",
                        {"topic": "race", "fine_grained_label": "0"},
                    )
                ], ensure_ascii=False),
                encoding="utf-8",
            )
            test_path.write_text(
                json.dumps([
                    make_record(
                        "test_1",
                        "恶意泛化某群体",
                        "hate",
                        {"topic": "race", "fine_grained_label": "2"},
                    )
                ], ensure_ascii=False),
                encoding="utf-8",
            )

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({
                    "task_type": "cold_binary",
                    "data_paths": {
                        "raw_data_path": str(train_path),
                        "val_data_path": str(val_path),
                        "test_data_path": str(test_path),
                        "train_output_path": str(out_dir / "train.jsonl"),
                        "val_output_path": str(out_dir / "val.jsonl"),
                        "val_runner_output_path": str(out_dir / "val_runner.json"),
                        "test_output_path": str(out_dir / "test.json"),
                        "lexicon_data_path": "",
                        "tokenizer_path": None,
                    },
                    "prompt_templates": {
                        "prompt_template": "COLD_BINARY_RAG_PROMPT_USER",
                        "example_template": "COLD_BINARY_EXAMPLE_PROMPT",
                        "system_prompt": "COLD_BINARY_SYSTEM_PROMPT",
                    },
                    "retrieval_settings": {
                        "use_srag": False,
                        "use_lex": False,
                    },
                    "training_settings": {
                        "auto_length": False,
                        "split_ratio": 0.9,
                    },
                    "cache_settings": {
                        "enable_build_cache": False,
                        "build_cache_dir": str(cache_dir),
                    },
                }, ensure_ascii=False),
                encoding="utf-8",
            )

            make_data(Config(str(config_path)))

            train_rows = [
                json.loads(line)
                for line in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["output"] for row in train_rows], ["hate", "non-hate"])
            self.assertNotIn("三元组", train_rows[0]["input"])
            self.assertNotIn("四元组", train_rows[0]["input"])

            test_rows = json.loads((out_dir / "test.json").read_text(encoding="utf-8"))
            self.assertEqual(test_rows[0]["gt_label"], "hate")
            self.assertEqual(test_rows[0]["metadata"]["topic"], "race")
            self.assertIn("messages_list", test_rows[0])
            self.assertEqual(test_rows[0]["gt_quadruples"][0]["hateful"], "hate")

            val_runner_rows = json.loads((out_dir / "val_runner.json").read_text(encoding="utf-8"))
            self.assertEqual(val_runner_rows[0]["gt_label"], "non-hate")
            self.assertEqual(val_runner_rows[0]["metadata"]["fine_grained_label"], "0")
            self.assertIn("messages_list", val_runner_rows[0])

    def test_zero_shot_v2_prompt_omits_empty_resource_sections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt = self._render_train_prompt(Path(tmp_dir), "COLD_BINARY_PROMPT_USER_V2")

        self.assertIn("边界规则", prompt)
        self.assertIn("只输出一个标签：hate 或 non-hate", prompt)
        self.assertIn("文本：讨厌某群体", prompt)
        self.assertNotIn("背景知识", prompt)
        self.assertNotIn("示例：", prompt)

    def test_explicit_full_rag_v2_prompt_is_not_auto_switched(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt = self._render_train_prompt(
                Path(tmp_dir),
                "COLD_BINARY_RAG_PROMPT_USER_V2",
                {
                    "use_srag": False,
                    "srag_top_k": 0,
                    "use_lex": False,
                    "lex_top_k": 0,
                },
            )

        self.assertIn("边界规则", prompt)
        self.assertIn("背景知识：", prompt)
        self.assertIn("示例：", prompt)
        self.assertIn("文本：讨厌某群体", prompt)


if __name__ == "__main__":
    unittest.main()
