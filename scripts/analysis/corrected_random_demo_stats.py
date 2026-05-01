from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/full/std/train.json"
TEST_PATH = ROOT / "data/full/std/test.json"
CORRECTED_RANDOM_PATH = (
    ROOT
    / "exps/emonstration_selection/random/exp_c3700e5fe8/data/"
    / "test_unstratified_random10_s42.json"
)

RUNNER_OUTPUTS = {
    "Global": ROOT
    / "exps/emonstration_selection/global/exp_744c07ab8e/runner_output/"
    / "exp_744c07ab8e_s42.json",
    "MMR": ROOT
    / "exps/emonstration_selection/mmr/exp_22101d2758/runner_output/"
    / "exp_22101d2758_s42.json",
    "Cluster": ROOT
    / "exps/emonstration_selection/cluster/exp_bd8221b10b/runner_output/"
    / "exp_bd8221b10b_s42.json",
    "Uniform": ROOT
    / "exps/emonstration_selection/uniform/exp_c44a603058/runner_output/"
    / "exp_c44a603058_s42.json",
    "Class-quota": ROOT / "output/runner/method_comparison/ours_prompt_al1280.json",
}

METHOD_ORDER = ["Random", "Global", "MMR", "Cluster", "Uniform", "Class-quota"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_category(category: Any) -> str:
    text = str(category).replace("[END]", "").strip()
    if text == "others":
        return "Others"
    return text


def split_categories(raw: Any) -> list[str]:
    if raw is None:
        return []
    text = str(raw).replace("，", ",").replace(";", ",").replace("/", ",")
    return [cat for part in text.split(",") if (cat := normalize_category(part))]


def categories_from_quadruples(quadruples: list[dict[str, Any]] | None) -> list[str]:
    categories: list[str] = []
    for quadruple in quadruples or []:
        categories.extend(split_categories(quadruple.get("targeted_group")))
    return categories


def format_demo_output(record: dict[str, Any]) -> str:
    triples = []
    for quadruple in record.get("quadruples", []) or []:
        target = str(quadruple.get("target", "")).strip()
        argument = str(quadruple.get("argument", "")).strip()
        group = str(quadruple.get("targeted_group", "")).strip()
        triples.append(f"{target} | {argument} | {group}")
    return " [SEP] ".join(triples) + " [END]" if triples else ""


def generate_corrected_random(
    train_path: Path,
    test_path: Path,
    output_path: Path,
    seed: int,
    shot_num: int,
) -> list[dict[str, Any]]:
    train_data = load_json(train_path)
    test_data = load_json(test_path)
    rng = np.random.RandomState(seed)
    rows: list[dict[str, Any]] = []

    if shot_num > len(train_data):
        raise ValueError(f"shot_num={shot_num} exceeds train size={len(train_data)}")

    for item in test_data:
        sampled_indices = rng.choice(len(train_data), size=shot_num, replace=False)
        sampled_demos = []
        for idx in sampled_indices.tolist():
            demo = train_data[int(idx)]
            sampled_demos.append(
                {
                    "id": demo.get("id"),
                    "content": demo.get("content", ""),
                    "quadruples": demo.get("quadruples", []),
                    "output": format_demo_output(demo),
                }
            )

        rows.append(
            {
                "id": item.get("id"),
                "content": item.get("content", ""),
                "gt_quadruples": item.get("quadruples", item.get("gt_quadruples", [])),
                "sampled_demos": sampled_demos,
            }
        )

    dump_json(rows, output_path)
    return rows


def shannon_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count)


def compute_demo_stats(
    rows: list[dict[str, Any]],
    demo_categories_fn,
    gold_categories_fn,
    avg_f1: str | float,
) -> dict[str, Any]:
    coverage_values: list[int] = []
    entropy_values: list[float] = []
    gold_coverage_values: list[int] = []
    missing = defaultdict(lambda: [0, 0])

    for row in rows:
        demo_categories = demo_categories_fn(row)
        demo_set = set(demo_categories)
        gold_set = set(gold_categories_fn(row))
        coverage_values.append(len(demo_set))
        entropy_values.append(shannon_entropy(Counter(demo_categories)))
        gold_coverage_values.append(1 if gold_set & demo_set else 0)

        for category in ("LGBTQ", "Others"):
            if category in gold_set:
                missing[category][1] += 1
                if category not in demo_set:
                    missing[category][0] += 1

    total = len(rows)
    if total == 0:
        raise ValueError("No rows to summarize.")

    return {
        "avg_groups": sum(coverage_values) / total,
        "demo_entropy": sum(entropy_values) / total,
        "gold_demo_coverage": sum(gold_coverage_values) / total,
        "missing_lgbtq": missing_rate(missing["LGBTQ"]),
        "missing_others": missing_rate(missing["Others"]),
        "avg_f1": avg_f1,
        "coverage_distribution": dict(sorted(Counter(coverage_values).items())),
    }


def missing_rate(pair: list[int]) -> float:
    missing_count, total_count = pair
    return missing_count / total_count if total_count else 0.0


def corrected_random_demo_categories(row: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    for demo in row.get("sampled_demos", []) or []:
        categories.extend(categories_from_quadruples(demo.get("quadruples", [])))
    return categories


def corrected_random_gold_categories(row: dict[str, Any]) -> list[str]:
    return categories_from_quadruples(row.get("gt_quadruples", []))


def runner_prompt_text(row: dict[str, Any]) -> str:
    for conversation in row.get("messages_list", []) or []:
        for message in conversation:
            if message.get("role") == "user":
                return str(message.get("content", ""))
    return ""


def runner_demo_blocks(prompt_text: str) -> list[str]:
    if "示例：" not in prompt_text:
        return []
    examples = prompt_text.split("示例：", 1)[1]
    blocks = re.findall(
        r"### 句子：\n.*?\n### 三元组：\n(.*?)(?=\n### 句子：\n|\Z)",
        examples,
        flags=re.S,
    )
    return [block.strip() for block in blocks if "[END]" in block]


def categories_from_demo_output(output: str) -> list[str]:
    categories: list[str] = []
    for triple in re.split(r"\s*\[SEP\]\s*|\s*\[END\]\s*", output):
        if "|" not in triple:
            continue
        fields = [field.strip() for field in triple.split("|")]
        if len(fields) >= 3:
            categories.extend(split_categories(fields[-1]))
    return categories


def runner_demo_categories(row: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    for block in runner_demo_blocks(runner_prompt_text(row)):
        categories.extend(categories_from_demo_output(block))
    return categories


def runner_gold_categories(row: dict[str, Any]) -> list[str]:
    return categories_from_quadruples(row.get("gt_quadruples", []))


def compute_runner_stats(path: Path) -> dict[str, Any]:
    data = load_json(path)
    return compute_demo_stats(
        rows=data["results"],
        demo_categories_fn=runner_demo_categories,
        gold_categories_fn=runner_gold_categories,
        avg_f1=float(data["metric"]["f1_avg"]),
    )


def render_number(value: str | float) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.4f}"


def render_markdown_table(stats: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Retrieval   | Avg #groups in demos | Demo entropy | Gold-demo coverage | Missing rate: LGBTQ | Missing rate: Others | Avg-F1 |",
        "| ----------- | -------------------: | -----------: | -----------------: | ------------------: | -------------------: | -----: |",
    ]
    for method in METHOD_ORDER:
        row = stats[method]
        lines.append(
            "| {method:<11} | {avg_groups:>20} | {entropy:>12} | {gold:>18} | "
            "{lgbtq:>19} | {others:>20} | {f1:>6} |".format(
                method=method,
                avg_groups=render_number(row["avg_groups"]),
                entropy=render_number(row["demo_entropy"]),
                gold=render_number(row["gold_demo_coverage"]),
                lgbtq=render_number(row["missing_lgbtq"]),
                others=render_number(row["missing_others"]),
                f1=render_number(row["avg_f1"]),
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate corrected unstratified random demos and summarize demo statistics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot-num", type=int, default=10)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--corrected-random-path", type=Path, default=CORRECTED_RANDOM_PATH)
    parser.add_argument("--table-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corrected_rows = generate_corrected_random(
        train_path=args.train_path,
        test_path=args.test_path,
        output_path=args.corrected_random_path,
        seed=args.seed,
        shot_num=args.shot_num,
    )

    stats: dict[str, dict[str, Any]] = {
        "Random": compute_demo_stats(
            rows=corrected_rows,
            demo_categories_fn=corrected_random_demo_categories,
            gold_categories_fn=corrected_random_gold_categories,
            avg_f1="N/A",
        )
    }
    for method, path in RUNNER_OUTPUTS.items():
        stats[method] = compute_runner_stats(path)

    markdown_table = render_markdown_table(stats)
    if args.table_output is not None:
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        args.table_output.write_text(markdown_table + "\n", encoding="utf-8")

    demo_counts = Counter(len(row.get("sampled_demos", [])) for row in corrected_rows)
    print(f"corrected_random_path: {args.corrected_random_path.relative_to(ROOT)}")
    print(f"corrected_random_sha256: {file_sha256(args.corrected_random_path)}")
    print(f"corrected_random_records: {len(corrected_rows)}")
    print(f"corrected_random_demo_count_distribution: {dict(sorted(demo_counts.items()))}")
    print(f"corrected_random_coverage_distribution: {stats['Random']['coverage_distribution']}")
    print()
    print(markdown_table)
    print()
    print("Note: corrected Random Avg-F1 is N/A because no LLM inference was rerun.")


if __name__ == "__main__":
    main()
