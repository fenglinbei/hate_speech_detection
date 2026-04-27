import argparse
import csv
import json
import os
import random
from typing import Any, Iterable, Optional


ALLOWED_GROUPS = {"non-hate", "Region", "Racism", "Sexism", "LGBTQ", "others"}
TEXT_FIELDS = (
    "content",
    "text",
    "sentence",
    "comment",
    "tweet",
    "post",
    "weibo",
    "raw_text",
)
ID_FIELDS = ("id", "", "uid", "sid", "sample_id", "index", "unnamed: 0")
LABEL_FIELDS = ("hateful", "hate", "label", "offensive", "toxic", "is_hate", "is_offensive")
GROUP_FIELDS = ("targeted_group", "target_group", "group", "category", "topic", "hate_type", "type")
TARGET_FIELDS = ("target", "target_span", "target_text", "targeted", "object", "entity")
ARGUMENT_FIELDS = ("argument", "argument_span", "opinion", "rationale", "span", "evidence")
SPLIT_FIELDS = ("split", "set", "subset", "partition")
FINE_GRAINED_LABEL_FIELDS = ("fine-grained-label", "fine_grained_label", "fine_label", "sub_label")

HATE_VALUES = {
    "1",
    "true",
    "yes",
    "y",
    "hate",
    "hateful",
    "offensive",
    "toxic",
    "abusive",
    "insult",
    "discrimination",
}
NON_HATE_VALUES = {
    "0",
    "false",
    "no",
    "n",
    "normal",
    "clean",
    "non-hate",
    "non_hate",
    "nonhate",
    "not_hate",
    "not offensive",
    "not_offensive",
}

GROUP_ALIASES = {
    "non-hate": "non-hate",
    "non_hate": "non-hate",
    "nonhate": "non-hate",
    "normal": "non-hate",
    "clean": "non-hate",
    "region": "Region",
    "regional": "Region",
    "location": "Region",
    "area": "Region",
    "racism": "Racism",
    "race": "Racism",
    "racial": "Racism",
    "ethnicity": "Racism",
    "ethnic": "Racism",
    "sexism": "Sexism",
    "sexist": "Sexism",
    "gender": "Sexism",
    "women": "Sexism",
    "female": "Sexism",
    "lgbtq": "LGBTQ",
    "lgbt": "LGBTQ",
    "homosexual": "LGBTQ",
    "sexual_orientation": "LGBTQ",
    "other": "others",
    "others": "others",
    "misc": "others",
    "abusive": "others",
    "offensive": "others",
    "toxic": "others",
    "discrimination": "others",
}


def _first_value(record: dict[str, Any], fields: Iterable[str]) -> Optional[Any]:
    lower_to_key = {str(key).lower(): key for key in record.keys()}
    for field in fields:
        key = lower_to_key.get(field.lower())
        if key is None:
            continue
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _metadata_from_record(record: dict[str, Any]) -> dict[str, str]:
    metadata = {}
    topic = _first_value(record, GROUP_FIELDS)
    fine_grained_label = _first_value(record, FINE_GRAINED_LABEL_FIELDS)
    split = _first_value(record, SPLIT_FIELDS)

    if topic is not None:
        metadata["topic"] = _as_text(topic)
    if fine_grained_label is not None:
        metadata["fine_grained_label"] = _as_text(fine_grained_label)
    if split is not None:
        metadata["split"] = _as_text(split)
    return metadata


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "hate" if value else "non-hate"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "hate" if int(value) == 1 else "non-hate"

    text = _as_text(value).lower()
    if text in HATE_VALUES:
        return "hate"
    if text in NON_HATE_VALUES:
        return "non-hate"
    if any(token in text for token in ("hate", "offensive", "toxic", "abusive", "discrimination")):
        return "hate"
    return "non-hate"


def _normalize_group(value: Any, is_hate: bool) -> str:
    if not is_hate:
        return "non-hate"
    if value is None:
        return "others"

    if isinstance(value, (list, tuple, set)):
        raw_parts = value
    else:
        raw_parts = str(value).replace(";", ",").replace("/", ",").replace("|", ",").split(",")

    groups = []
    for part in raw_parts:
        key = _as_text(part).lower().replace(" ", "_")
        group = GROUP_ALIASES.get(key)
        if group and group != "non-hate" and group not in groups:
            groups.append(group)

    return ", ".join(groups) if groups else "others"


def _maps_to_hate_group(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        raw_parts = value
    else:
        raw_parts = str(value).replace(";", ",").replace("/", ",").replace("|", ",").split(",")

    for part in raw_parts:
        key = _as_text(part).lower().replace(" ", "_")
        group = GROUP_ALIASES.get(key)
        if group and group != "non-hate":
            return True
    return False


def load_records(path: str) -> list[dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8-sig") as file:
            data = json.load(file)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "records", "samples", "items"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(f"Unsupported JSON layout in {path}")

    if ext == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8-sig") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if ext in {".csv", ".tsv"}:
        delimiter = "\t" if ext == ".tsv" else ","
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file, delimiter=delimiter))

    raise ValueError(f"Unsupported COLD input extension: {ext}")


def normalize_record(record: dict[str, Any], index: int, id_prefix: str = "cold") -> dict[str, Any]:
    content = _as_text(_first_value(record, TEXT_FIELDS))
    if not content:
        raise ValueError(f"Missing text/content field at record index {index}")

    raw_id = _first_value(record, ID_FIELDS)
    sample_id = f"{id_prefix}_{raw_id if raw_id is not None else index}"

    raw_label = _first_value(record, LABEL_FIELDS)
    raw_group = _first_value(record, GROUP_FIELDS)
    label = _normalize_label(raw_label)
    if label == "non-hate" and (
            _maps_to_hate_group(raw_label) or (raw_label is None and _maps_to_hate_group(raw_group))
            ):
        label = "hate"
    is_hate = label == "hate"
    targeted_group = _normalize_group(raw_group if raw_group is not None else raw_label, is_hate)
    target = _as_text(_first_value(record, TARGET_FIELDS)) or "NULL"
    argument = _as_text(_first_value(record, ARGUMENT_FIELDS)) or content

    if not is_hate:
        target = target or "NULL"
        targeted_group = "non-hate"
        argument = argument or content

    return {
        "id": sample_id,
        "content": content,
        "metadata": _metadata_from_record(record),
        "quadruples": [
            {
                "target": target,
                "argument": argument,
                "targeted_group": targeted_group,
                "hateful": label,
            }
        ],
    }


def normalize_records(records: list[dict[str, Any]], id_prefix: str = "cold") -> list[dict[str, Any]]:
    normalized = [normalize_record(record, idx, id_prefix=id_prefix) for idx, record in enumerate(records)]
    seen = {}
    for sample in normalized:
        sample_id = str(sample["id"])
        if sample_id not in seen:
            seen[sample_id] = 0
            continue
        seen[sample_id] += 1
        sample["id"] = f"{sample_id}_{seen[sample_id]}"
    validate_standard_records(normalized)
    return normalized


def _record_split(record: dict[str, Any]) -> Optional[str]:
    split = _first_value(record, SPLIT_FIELDS)
    if split is None:
        return None
    text = _as_text(split).lower()
    if text in {"train", "training"}:
        return "train"
    if text in {"val", "valid", "validation", "dev", "development"}:
        return "val"
    if text in {"test", "testing"}:
        return "test"
    return None


def split_records(
        raw_records: list[dict[str, Any]],
        normalized_records: list[dict[str, Any]],
        train_ratio: float,
        seed: int
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_values = [_record_split(record) for record in raw_records]
    if any(split_values):
        train = [sample for sample, split in zip(normalized_records, split_values) if split == "train"]
        test = [sample for sample, split in zip(normalized_records, split_values) if split == "test"]
        if train and test:
            return train, test

    shuffled = list(normalized_records)
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def validate_standard_records(records: list[dict[str, Any]]) -> None:
    ids = set()
    for idx, sample in enumerate(records):
        sample_id = sample.get("id")
        if sample_id in ids:
            raise ValueError(f"Duplicate id: {sample_id}")
        ids.add(sample_id)
        if not _as_text(sample.get("content")):
            raise ValueError(f"Missing content at normalized record {idx}")
        quads = sample.get("quadruples")
        if not isinstance(quads, list) or not quads:
            raise ValueError(f"Missing quadruples at normalized record {idx}")
        for quad in quads:
            group = quad.get("targeted_group")
            groups = [g.strip() for g in str(group).split(",") if g.strip()]
            if not groups or any(g not in ALLOWED_GROUPS for g in groups):
                raise ValueError(f"Invalid targeted_group at record {idx}: {group}")
            if quad.get("hateful") not in {"hate", "non-hate"}:
                raise ValueError(f"Invalid hateful label at record {idx}: {quad.get('hateful')}")


def save_json(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def _find_split_file(raw_dir: str, split: str) -> Optional[str]:
    aliases = {
        "train": ("train", "training"),
        "val": ("val", "valid", "validation", "dev", "development"),
        "test": ("test", "testing"),
    }
    extensions = (".csv", ".tsv", ".json", ".jsonl")
    for name in aliases[split]:
        for ext in extensions:
            path = os.path.join(raw_dir, f"{name}{ext}")
            if os.path.exists(path):
                return path
    return None


def convert_split_file(input_path: str, output_path: str, id_prefix: str) -> list[dict[str, Any]]:
    records = normalize_records(load_records(input_path), id_prefix=id_prefix)
    validate_standard_records(records)
    save_json(output_path, records)
    return records


def convert_cold_splits(
        raw_dir: str = "data/cold/raw",
        output_dir: str = "data/cold/std",
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        id_prefix: str = "cold",
        ) -> dict[str, list[dict[str, Any]]]:
    split_paths = {
        "train": train_path or _find_split_file(raw_dir, "train"),
        "val": val_path or _find_split_file(raw_dir, "val"),
        "test": test_path or _find_split_file(raw_dir, "test"),
    }

    missing = [split for split in ("train", "test") if split_paths[split] is None]
    if missing:
        raise FileNotFoundError(f"Missing required COLD split file(s): {', '.join(missing)}")

    converted = {}
    for split, input_path in split_paths.items():
        if input_path is None:
            continue
        output_path = os.path.join(output_dir, f"{split}.json")
        converted[split] = convert_split_file(
            input_path=input_path,
            output_path=output_path,
            id_prefix=f"{id_prefix}_{split}",
        )
    return converted


def convert_cold_dataset(
        input_path: str,
        output_dir: str = "data/cold/std",
        train_ratio: float = 0.8,
        seed: int = 42,
        id_prefix: str = "cold",
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_records = load_records(input_path)
    normalized = normalize_records(raw_records, id_prefix=id_prefix)
    train, test = split_records(raw_records, normalized, train_ratio=train_ratio, seed=seed)
    validate_standard_records(train)
    validate_standard_records(test)
    save_json(os.path.join(output_dir, "train.json"), train)
    save_json(os.path.join(output_dir, "test.json"), test)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COLD-style data to this project's quadruple JSON format.")
    parser.add_argument("--input", help="Path to one raw COLD json/jsonl/csv/tsv file. This mode may split data.")
    parser.add_argument("--raw-dir", help="Directory containing train/dev(or val)/test raw COLD files.")
    parser.add_argument("--train-file", help="Explicit train split path.")
    parser.add_argument("--val-file", help="Explicit validation split path.")
    parser.add_argument("--test-file", help="Explicit test split path.")
    parser.add_argument("--output-dir", default="data/cold/std", help="Directory for normalized split JSON files.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio when input has no split column.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-prefix", default="cold")
    args = parser.parse_args()

    if args.raw_dir or args.train_file or args.val_file or args.test_file:
        splits = convert_cold_splits(
            raw_dir=args.raw_dir or "data/cold/raw",
            output_dir=args.output_dir,
            train_path=args.train_file,
            val_path=args.val_file,
            test_path=args.test_file,
            id_prefix=args.id_prefix,
        )
        print(json.dumps({split: len(records) for split, records in splits.items()}, ensure_ascii=False, indent=2))
        return

    if not args.input:
        parser.error("Provide either --input for single-file conversion or --raw-dir/--train-file/--test-file for fixed splits.")

    train, test = convert_cold_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        id_prefix=args.id_prefix,
    )
    print(json.dumps({"train": len(train), "test": len(test)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
