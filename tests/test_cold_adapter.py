import csv
import json
import tempfile
import unittest
from pathlib import Path

from data.cold_adapter import (
    ALLOWED_GROUPS,
    convert_cold_dataset,
    convert_cold_splits,
    normalize_records,
    validate_standard_records,
)


class ColdAdapterTest(unittest.TestCase):
    def test_normalize_records_maps_binary_cold_rows_to_quadruples(self):
        records = [
            {"id": "a1", "text": "plain comment", "label": "0"},
            {"id": "a2", "text": "hostile gender comment", "label": "1", "category": "gender"},
            {"id": "a3", "text": "racial category comment", "label": "Racism"},
        ]

        normalized = normalize_records(records)

        validate_standard_records(normalized)
        self.assertEqual(
            normalized[0]["quadruples"][0],
            {
                "target": "NULL",
                "argument": "plain comment",
                "targeted_group": "non-hate",
                "hateful": "non-hate",
            },
        )
        self.assertEqual(normalized[1]["quadruples"][0]["targeted_group"], "Sexism")
        self.assertEqual(normalized[1]["quadruples"][0]["hateful"], "hate")
        self.assertEqual(normalized[2]["quadruples"][0]["targeted_group"], "Racism")
        self.assertEqual(normalized[2]["quadruples"][0]["hateful"], "hate")

    def test_normalize_records_deduplicates_ids_and_keeps_allowed_groups(self):
        records = [
            {"id": "dup", "content": "regional hate text", "hateful": "hate", "target_group": "region"},
            {"id": "dup", "content": "unknown hate text", "hateful": "hate", "target_group": "unknown"},
        ]

        normalized = normalize_records(records)

        self.assertEqual(normalized[0]["id"], "cold_dup")
        self.assertEqual(normalized[1]["id"], "cold_dup_1")
        for sample in normalized:
            quad = sample["quadruples"][0]
            groups = {group.strip() for group in quad["targeted_group"].split(",")}
            self.assertTrue(groups.issubset(ALLOWED_GROUPS))

    def test_convert_cold_dataset_respects_existing_split_column(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "cold.csv"
            output_dir = root / "std"
            rows = [
                {"id": "1", "text": "train sample", "label": "0", "category": "", "split": "train"},
                {"id": "2", "text": "test sample", "label": "1", "category": "race", "split": "test"},
            ]
            with input_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["id", "text", "label", "category", "split"])
                writer.writeheader()
                writer.writerows(rows)

            train, test = convert_cold_dataset(str(input_path), str(output_dir))

            self.assertEqual(len(train), 1)
            self.assertEqual(len(test), 1)
            saved_train = json.loads((output_dir / "train.json").read_text(encoding="utf-8"))
            saved_test = json.loads((output_dir / "test.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_train, train)
            self.assertEqual(saved_test, test)
            self.assertEqual(saved_test[0]["quadruples"][0]["targeted_group"], "Racism")

    def test_convert_cold_splits_preserves_train_dev_test_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            output_dir = root / "std"
            raw_dir.mkdir()

            def write_split(path, row_id, split, topic, label, text):
                with path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=["", "split", "topic", "label", "TEXT"])
                    writer.writeheader()
                    writer.writerow({"": row_id, "split": split, "topic": topic, "label": label, "TEXT": text})

            write_split(raw_dir / "train.csv", "10", "train", "race", "1", "train text")
            write_split(raw_dir / "dev.csv", "20", "dev", "gender", "1", "dev text")
            write_split(raw_dir / "test.csv", "30", "test", "region", "0", "test text")

            splits = convert_cold_splits(str(raw_dir), str(output_dir))

            self.assertEqual(set(splits), {"train", "val", "test"})
            self.assertEqual(splits["train"][0]["id"], "cold_train_10")
            self.assertEqual(splits["val"][0]["quadruples"][0]["targeted_group"], "Sexism")
            self.assertEqual(splits["test"][0]["quadruples"][0]["hateful"], "non-hate")
            self.assertTrue((output_dir / "train.json").exists())
            self.assertTrue((output_dir / "val.json").exists())
            self.assertTrue((output_dir / "test.json").exists())


if __name__ == "__main__":
    unittest.main()
