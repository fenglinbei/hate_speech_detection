#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_GROUPS = ("non-hate", "Region", "Racism", "Sexism")
DEFAULT_SPLIT_SIZES = {"train": 80, "val": 20, "test": 20}


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def save_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def sample_groups(record: dict) -> set[str]:
    groups: set[str] = set()
    for quad in record.get("quadruples", []):
        raw_group = quad.get("targeted_group")
        if raw_group is None:
            continue
        for group in str(raw_group).replace("/", ",").replace("|", ",").split(","):
            group = group.strip()
            if group:
                groups.add(group)
    return groups


def primary_group(record: dict, preferred_groups: Iterable[str]) -> str:
    groups = sample_groups(record)
    for group in preferred_groups:
        if group in groups:
            return group
    return sorted(groups)[0] if groups else "__missing_group__"


def validate_records(records: list[dict], source: Path) -> None:
    seen_ids = set()
    for idx, record in enumerate(records):
        sample_id = record.get("id")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate id in {source}: {sample_id}")
        seen_ids.add(sample_id)
        if not str(record.get("content", "")).strip():
            raise ValueError(f"Missing content in {source} at index {idx}")
        if not isinstance(record.get("quadruples"), list) or not record["quadruples"]:
            raise ValueError(f"Missing quadruples in {source} at index {idx}")


def allocate_by_largest_remainder(
    buckets: dict[str, list[int]],
    total_size: int,
    required_groups: tuple[str, ...],
) -> dict[str, int]:
    available_groups = [group for group in required_groups if buckets.get(group)]
    if total_size < len(available_groups):
        raise ValueError(
            f"Sample size {total_size} is too small to cover groups: {available_groups}"
        )

    allocations = {group: 1 for group in available_groups}
    remaining = total_size - len(available_groups)
    if remaining <= 0:
        return allocations

    total_available = sum(len(buckets[group]) for group in available_groups)
    remainders: list[tuple[float, str]] = []
    for group in available_groups:
        quota = remaining * len(buckets[group]) / total_available
        extra = int(quota)
        allocations[group] += min(extra, len(buckets[group]) - allocations[group])
        remainders.append((quota - extra, group))

    while sum(allocations.values()) < total_size:
        progressed = False
        for _, group in sorted(remainders, reverse=True):
            if allocations[group] >= len(buckets[group]):
                continue
            allocations[group] += 1
            progressed = True
            if sum(allocations.values()) >= total_size:
                break
        if not progressed:
            break

    return allocations


def stratified_sample(
    records: list[dict],
    total_size: int,
    seed: int,
    required_groups: Iterable[str] = DEFAULT_GROUPS,
) -> list[dict]:
    if total_size <= 0:
        return []
    if total_size > len(records):
        raise ValueError(f"Requested {total_size} records from only {len(records)} records")

    required = tuple(required_groups)
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        buckets[primary_group(record, required)].append(idx)

    missing_groups = [group for group in required if not buckets.get(group)]
    if missing_groups:
        raise ValueError(f"Cannot cover missing group(s): {missing_groups}")

    rng = random.Random(seed)
    shuffled_buckets = {group: list(indices) for group, indices in buckets.items()}
    for indices in shuffled_buckets.values():
        rng.shuffle(indices)

    allocations = allocate_by_largest_remainder(shuffled_buckets, total_size, required)
    selected_indices: set[int] = set()
    for group, count in allocations.items():
        selected_indices.update(shuffled_buckets[group][:count])

    if len(selected_indices) < total_size:
        remaining = [idx for idx in range(len(records)) if idx not in selected_indices]
        rng.shuffle(remaining)
        selected_indices.update(remaining[: total_size - len(selected_indices)])

    selected = [records[idx] for idx in sorted(selected_indices)]
    validate_selected_groups(selected, required)
    return selected


def validate_selected_groups(records: list[dict], required_groups: tuple[str, ...]) -> None:
    selected_groups = set()
    for record in records:
        selected_groups.update(sample_groups(record))
    missing = [group for group in required_groups if group not in selected_groups]
    if missing:
        raise ValueError(f"Selected subset does not cover group(s): {missing}")


def build_smoke_subset(
    input_dir: Path,
    output_dir: Path,
    split_sizes: dict[str, int],
    seed: int,
    required_groups: Iterable[str] = DEFAULT_GROUPS,
) -> dict[str, int]:
    counts = {}
    required = tuple(required_groups)
    for split, size in split_sizes.items():
        source_path = input_dir / f"{split}.json"
        records = load_json(source_path)
        validate_records(records, source_path)
        selected = stratified_sample(records, size, seed=seed, required_groups=required)
        save_json(output_dir / f"{split}.json", selected)
        counts[split] = len(selected)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic stratified COLD smoke-test subset."
    )
    parser.add_argument("--input-dir", default="data/cold/std")
    parser.add_argument("--output-dir", default="data/cold/smoke")
    parser.add_argument("--train-size", type=int, default=DEFAULT_SPLIT_SIZES["train"])
    parser.add_argument("--val-size", type=int, default=DEFAULT_SPLIT_SIZES["val"])
    parser.add_argument("--test-size", type=int, default=DEFAULT_SPLIT_SIZES["test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_smoke_subset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        split_sizes={
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size,
        },
        seed=args.seed,
        required_groups=args.groups,
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
