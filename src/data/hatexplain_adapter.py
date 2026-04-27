from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_URL = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/post_id_divisions.json"
LABELS = {"hatespeech", "offensive", "normal"}


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(str(token) for token in tokens)
    text = re.sub(r"\s+([.,!?;:%)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_source(post_id: str) -> str:
    if post_id.endswith("_twitter"):
        return "twitter"
    if post_id.endswith("_gab"):
        return "gab"
    return "unknown"


def resolve_split_url(script_path: str | Path | None) -> str:
    if not script_path or not Path(script_path).exists():
        return DEFAULT_SPLIT_URL

    text = Path(script_path).read_text(encoding="utf-8")
    url_match = re.search(r'_URL\s*=\s*["\']([^"\']+)["\']', text)
    base_url = url_match.group(1) if url_match else DEFAULT_SPLIT_URL.rsplit("/", 1)[0] + "/"
    split_match = re.search(r'["\']post_id_divisions["\']\s*:\s*_URL\s*\+\s*["\']([^"\']+)["\']', text)
    if split_match:
        return base_url + split_match.group(1)
    return DEFAULT_SPLIT_URL


def load_split_divisions(split_path: str | Path, script_path: str | Path | None = None) -> dict[str, list[str]]:
    split_path = Path(split_path)
    if split_path.exists():
        return load_json(split_path)

    split_url = resolve_split_url(script_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(split_url, split_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"HateXplain split file not found at {split_path} and could not be downloaded from {split_url}."
        ) from exc
    return load_json(split_path)


def majority_label(annotators: list[dict[str, Any]], tie_policy: str) -> tuple[str | None, dict[str, int], bool]:
    votes = Counter(str(ann.get("label", "")).strip().lower() for ann in annotators)
    votes = Counter({label: count for label, count in votes.items() if label in LABELS})
    if not votes:
        return None, {}, False

    max_votes = max(votes.values())
    winners = [label for label, count in votes.items() if count == max_votes]
    if len(winners) == 1:
        return winners[0], dict(votes), False

    if tie_policy == "drop":
        return None, dict(votes), True
    if tie_policy == "ambiguous":
        return "ambiguous", dict(votes), True
    if tie_policy == "prefer_hate":
        for label in ("hatespeech", "offensive", "normal"):
            if label in winners:
                return label, dict(votes), True

    raise ValueError(f"Unsupported tie policy: {tie_policy}")


def target_groups(annotators: list[dict[str, Any]], label: str) -> tuple[list[str], dict[str, int]]:
    if label == "normal":
        return [], {}

    votes: Counter[str] = Counter()
    for ann in annotators:
        if str(ann.get("label", "")).strip().lower() == "normal":
            continue
        for target in ann.get("target", []) or []:
            target_text = str(target).strip()
            if target_text and target_text.lower() != "none":
                votes[target_text] += 1

    majority_targets = sorted(target for target, count in votes.items() if count >= 2)
    return majority_targets or sorted(votes), dict(votes)


def mask_to_spans(tokens: list[str], selected_indices: list[int]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    if not selected_indices:
        return spans

    cur: list[int] = []
    for idx in selected_indices:
        if not cur or idx == cur[-1] + 1:
            cur.append(idx)
            continue
        spans.append({"token_indices": cur, "text": detokenize([tokens[i] for i in cur])})
        cur = [idx]
    if cur:
        spans.append({"token_indices": cur, "text": detokenize([tokens[i] for i in cur])})
    return spans


def rationale_spans(tokens: list[str], masks: list[list[int]], label: str) -> tuple[list[dict[str, Any]], int]:
    valid_masks: list[list[int]] = []
    invalid_count = 0
    for mask in masks or []:
        if not isinstance(mask, list) or len(mask) != len(tokens):
            invalid_count += 1
            continue
        valid_masks.append([1 if int(value) else 0 for value in mask])

    if not valid_masks:
        return [], invalid_count

    threshold = len(valid_masks) // 2 + 1
    selected = [
        idx
        for idx in range(len(tokens))
        if sum(mask[idx] for mask in valid_masks) >= threshold
    ]
    if not selected and label != "normal":
        selected = [
            idx
            for idx in range(len(tokens))
            if any(mask[idx] for mask in valid_masks)
        ]
    return mask_to_spans(tokens, selected), invalid_count


def convert_record(
        post_id: str,
        raw: dict[str, Any],
        split: str,
        tie_policy: str,
        report: Counter,
        ) -> dict[str, Any] | None:
    tokens = [str(token) for token in raw.get("post_tokens", [])]
    annotators = raw.get("annotators", []) or []
    label, label_votes, tied = majority_label(annotators, tie_policy)
    if tied:
        report["tied_label_records"] += 1
    if label is None:
        report["dropped_tie_records"] += 1
        return None
    if label == "ambiguous":
        report["dropped_ambiguous_records"] += 1
        return None

    targets, target_votes = target_groups(annotators, label)
    rationales, invalid_masks = rationale_spans(tokens, raw.get("rationales", []) or [], label)
    report["invalid_rationale_masks"] += invalid_masks
    report[f"label:{label}"] += 1
    report[f"split:{split}"] += 1

    return {
        "id": post_id,
        "content": detokenize(tokens),
        "tokens": tokens,
        "annotation": {
            "label": label,
            "target_groups": targets,
            "rationales": rationales,
        },
        "metadata": {
            "dataset": "hatexplain",
            "split": split,
            "source": infer_source(post_id),
            "label_votes": label_votes,
            "target_votes": target_votes,
        },
    }


def convert_dataset(
        input_path: str | Path,
        split_path: str | Path,
        output_dir: str | Path,
        script_path: str | Path | None = None,
        tie_policy: str = "drop",
        ) -> dict[str, Any]:
    dataset = load_json(input_path)
    if not isinstance(dataset, dict):
        raise ValueError("Expected HateXplain dataset.json to be a post_id -> record dictionary.")

    split_divisions = load_split_divisions(split_path, script_path)
    output_dir = Path(output_dir)
    report: Counter[str] = Counter()
    report["input_records"] = len(dataset)
    split_ids = {
        post_id
        for split_name in ("train", "val", "test")
        for post_id in split_divisions.get(split_name, [])
    }
    unassigned_ids = set(dataset) - split_ids
    report["unassigned_records"] = len(unassigned_ids)
    for post_id in unassigned_ids:
        raw = dataset[post_id]
        _label, _votes, tied = majority_label(raw.get("annotators", []) or [], "drop")
        if tied:
            report["unassigned_tie_records"] += 1

    for split in ("train", "val", "test"):
        if split not in split_divisions:
            raise KeyError(f"Missing split '{split}' in {split_path}.")

        converted: list[dict[str, Any]] = []
        for post_id in split_divisions[split]:
            raw = dataset.get(post_id)
            if raw is None:
                report[f"missing_id:{split}"] += 1
                continue
            record = convert_record(post_id, raw, split, tie_policy, report)
            if record is not None:
                converted.append(record)

        dump_json(converted, output_dir / f"{split}.json")
        report[f"output_{split}_records"] = len(converted)

    report_data = {
        "input": str(input_path),
        "split_path": str(split_path),
        "tie_policy": tie_policy,
        "counts": dict(report),
    }
    dump_json(report_data, output_dir / "conversion_report.json")
    return report_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local HateXplain data to the project standard format.")
    parser.add_argument("--input", default="data/hateXplain/raw/dataset.json")
    parser.add_argument("--split-path", default="data/hateXplain/raw/post_id_divisions.json")
    parser.add_argument("--script-path", default="data/hateXplain/raw/hatexplain.py")
    parser.add_argument("--output-dir", default="data/hateXplain/std")
    parser.add_argument("--tie-policy", choices=["drop", "prefer_hate", "ambiguous"], default="drop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_dataset(
        input_path=args.input,
        split_path=args.split_path,
        output_dir=args.output_dir,
        script_path=args.script_path,
        tie_policy=args.tie_policy,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
