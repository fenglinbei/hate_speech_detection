import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.make_cold_smoke_subset import build_smoke_subset


GROUPS = ("non-hate", "Region", "Racism", "Sexism")


def make_record(split, group, idx):
    return {
        "id": f"{split}_{group}_{idx}",
        "content": f"{split} {group} sample {idx}",
        "quadruples": [
            {
                "target": "NULL",
                "argument": f"{split} {group} sample {idx}",
                "targeted_group": group,
                "hateful": "non-hate" if group == "non-hate" else "hate",
            }
        ],
    }


def write_split(path, split, per_group):
    records = []
    for group in GROUPS:
        for idx in range(per_group):
            records.append(make_record(split, group, idx))
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def selected_groups(records):
    return {
        quad["targeted_group"]
        for record in records
        for quad in record["quadruples"]
    }


class ColdSmokeSubsetTest(unittest.TestCase):
    def test_build_smoke_subset_writes_deterministic_stratified_splits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "std"
            output_dir = root / "smoke"
            input_dir.mkdir()

            write_split(input_dir / "train.json", "train", per_group=8)
            write_split(input_dir / "val.json", "val", per_group=4)
            write_split(input_dir / "test.json", "test", per_group=4)

            counts = build_smoke_subset(
                input_dir=input_dir,
                output_dir=output_dir,
                split_sizes={"train": 12, "val": 4, "test": 4},
                seed=7,
                required_groups=GROUPS,
            )

            self.assertEqual(counts, {"train": 12, "val": 4, "test": 4})
            first_outputs = {}
            for split, expected_count in counts.items():
                records = json.loads((output_dir / f"{split}.json").read_text(encoding="utf-8"))
                first_outputs[split] = records
                self.assertEqual(len(records), expected_count)
                self.assertEqual(len({record["id"] for record in records}), expected_count)
                self.assertEqual(selected_groups(records), set(GROUPS))
                for record in records:
                    self.assertTrue(record["content"])
                    self.assertTrue(record["quadruples"])

            build_smoke_subset(
                input_dir=input_dir,
                output_dir=output_dir,
                split_sizes={"train": 12, "val": 4, "test": 4},
                seed=7,
                required_groups=GROUPS,
            )
            for split in counts:
                records = json.loads((output_dir / f"{split}.json").read_text(encoding="utf-8"))
                self.assertEqual(records, first_outputs[split])


if __name__ == "__main__":
    unittest.main()
