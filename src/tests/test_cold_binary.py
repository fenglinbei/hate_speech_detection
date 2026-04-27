import json
import tempfile
import unittest
from pathlib import Path

from data.build_data import make_data
from data.config import Config
from metrics.metric_llm import BinaryClassificationMetrics
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
        self.assertEqual(metrics["hate_f1"], 0.6667)
        self.assertEqual(metrics["non_hate_f1"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5834)
        self.assertEqual(metrics["confusion_matrix"]["non-hate"]["invalid"], 1)


def make_record(sample_id, content, label):
    return {
        "id": sample_id,
        "content": content,
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
                json.dumps([make_record("val_1", "友好讨论", "non-hate")], ensure_ascii=False),
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
            self.assertEqual(test_rows[0]["gt_quadruples"][0]["hateful"], "hate")


if __name__ == "__main__":
    unittest.main()
