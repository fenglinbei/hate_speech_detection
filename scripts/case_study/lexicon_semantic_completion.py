#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build case studies for lexicon semantic completion.

The script analyzes an existing runner output and reconstructs where prompt
lexicon entries came from:

- exact entries: reproduced with LexiconRetriever's substring inclusion logic
- semantic entries: prompt background entries not explained by exact inclusion

It does not call the LLM or the embedding model.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_RUNNER_OUTPUT = (
    "exps/emonstration_selection/uniform/exp_c44a603058/"
    "runner_output/exp_c44a603058_s42.json"
)
DEFAULT_SELECTED_IDS = "6925,5427,5439,5609,5109"


BACKGROUND_RE = re.compile(
    r"背景知识：\n(?P<body>.*?)(?:\n\n示例：|\n\n### 句子：)",
    flags=re.S,
)
LEXICON_ENTRY_RE = re.compile(
    r"###\s*\n"
    r"关键词：(?P<term>.*?)\n"
    r"类别：(?P<category>.*?)\n"
    r"定义：(?P<definition>.*?)(?=\n\n###|\Z)",
    flags=re.S,
)
LABEL_SPLIT_RE = re.compile(r"[,，/\\|;；]+")


@dataclass(frozen=True)
class LexiconEntry:
    term: str
    category: str
    definition: str

    def to_json(self) -> dict[str, str]:
        return {
            "term": self.term,
            "category": self.category,
            "definition": self.definition,
        }


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_label(text: Any) -> str:
    return normalize_space(text).lower().replace("_", "-")


def split_labels(text: Any) -> set[str]:
    raw = normalize_space(text)
    if not raw:
        return set()
    return {normalize_label(part) for part in LABEL_SPLIT_RE.split(raw) if part.strip()}


def parse_id_list(value: str) -> list[int]:
    ids: list[int] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    return ids


def infer_exp_dir(runner_output: Path) -> Path:
    if runner_output.parent.name == "runner_output":
        return runner_output.parent.parent
    return runner_output.parent


def resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_lexicon_entries(lexicon_path: Path) -> list[LexiconEntry]:
    data = read_json(lexicon_path)
    terms = data.get("terms", data if isinstance(data, list) else [])
    entries: list[LexiconEntry] = []
    for item in terms:
        entries.append(
            LexiconEntry(
                term=str(item.get("term", "")).strip(),
                category=str(item.get("category", "")).strip(),
                definition=normalize_space(item.get("definition", "")),
            )
        )
    return [entry for entry in entries if entry.term]


def parse_prompt_lexicon_entries(row: dict[str, Any]) -> list[LexiconEntry]:
    try:
        prompt = row["messages_list"][0][1]["content"]
    except (KeyError, IndexError, TypeError):
        return []

    match = BACKGROUND_RE.search(prompt)
    if not match:
        return []

    entries: list[LexiconEntry] = []
    for entry_match in LEXICON_ENTRY_RE.finditer(match.group("body").strip()):
        entries.append(
            LexiconEntry(
                term=entry_match.group("term").strip(),
                category=entry_match.group("category").strip(),
                definition=normalize_space(entry_match.group("definition")),
            )
        )
    return entries


def reconstruct_exact_entries(
    content: str,
    lexicon_entries: list[LexiconEntry],
    top_k: int,
    case_sensitive: bool,
) -> tuple[list[LexiconEntry], list[LexiconEntry]]:
    """Reproduce LexiconRetriever.including_retrieve for the cold lexicon."""
    exact_all: list[LexiconEntry] = []
    query = content if case_sensitive else content.lower()

    for entry in lexicon_entries:
        term = entry.term if case_sensitive else entry.term.lower()
        if term and term in query:
            exact_all.append(entry)

    exact_top = exact_all if top_k == -1 else exact_all[:top_k]
    return exact_top, exact_all


def entry_counter(entries: list[LexiconEntry]) -> Counter[str]:
    return Counter(entry.term for entry in entries)


def subtract_exact_terms(
    prompt_entries: list[LexiconEntry],
    exact_entries: list[LexiconEntry],
) -> list[LexiconEntry]:
    """Attribute prompt entries not explained by exact top-k to semantic retrieval."""
    remaining_exact = entry_counter(exact_entries)
    semantic_entries: list[LexiconEntry] = []
    for entry in prompt_entries:
        if remaining_exact[entry.term] > 0:
            remaining_exact[entry.term] -= 1
        else:
            semantic_entries.append(entry)
    return semantic_entries


def preprocess_soft_quad(quad: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": str(quad.get("target", "")).strip(),
        "argument": str(quad.get("argument", "")).strip(),
        "targeted_group": sorted(str(quad.get("targeted_group", "")).lower().split(", ")),
        "hateful": str(quad.get("hateful", "")).lower().strip(),
    }


def hard_key(quad: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(quad.get("target", "")).strip(),
        str(quad.get("argument", "")).strip(),
        str(quad.get("targeted_group", "")).strip().lower(),
        str(quad.get("hateful", "")).strip().lower(),
    )


def similarity(left: Any, right: Any) -> float:
    return SequenceMatcher(None, str(left or ""), str(right or "")).ratio()


def normalize_obfuscated_text(text: Any) -> str:
    """Normalize a few common visible obfuscations for bridge evidence only."""
    value = str(text or "")
    value = re.sub(r"[sS涩色]会", "社会", value)
    return value


def hard_match(pred: dict[str, Any], gold: dict[str, Any]) -> bool:
    return hard_key(pred) == hard_key(gold)


def soft_match(pred: dict[str, Any], gold: dict[str, Any], threshold: float) -> bool:
    pred_norm = preprocess_soft_quad(pred)
    gold_norm = preprocess_soft_quad(gold)
    if (
        pred_norm["targeted_group"] != gold_norm["targeted_group"]
        or pred_norm["hateful"] != gold_norm["hateful"]
    ):
        return False
    return (
        similarity(pred_norm["target"], gold_norm["target"]) > threshold
        and similarity(pred_norm["argument"], gold_norm["argument"]) > threshold
    )


def greedy_matches(
    preds: list[dict[str, Any]],
    golds: list[dict[str, Any]],
    match_func,
) -> list[dict[str, Any]]:
    matched_gold: set[int] = set()
    matches: list[dict[str, Any]] = []
    for pred_idx, pred in enumerate(preds):
        for gold_idx, gold in enumerate(golds):
            if gold_idx in matched_gold:
                continue
            if match_func(pred, gold):
                matched_gold.add(gold_idx)
                matches.append(
                    {
                        "pred_idx": pred_idx,
                        "gold_idx": gold_idx,
                        "target_similarity": round(
                            similarity(pred.get("target"), gold.get("target")), 4
                        ),
                        "argument_similarity": round(
                            similarity(pred.get("argument"), gold.get("argument")), 4
                        ),
                    }
                )
                break
    return matches


def group_set_from_quads(quads: list[dict[str, Any]]) -> set[str]:
    groups: set[str] = set()
    for quad in quads:
        if normalize_label(quad.get("hateful")) in {"non-hate", "nonhate"}:
            groups.add("non-hate")
        groups.update(split_labels(quad.get("targeted_group")))
    return groups


def category_set(entries: list[LexiconEntry]) -> set[str]:
    groups: set[str] = set()
    for entry in entries:
        groups.update(split_labels(entry.category))
    return groups


def ngrams(text: str, min_len: int = 2, max_len: int = 5) -> set[str]:
    grams: set[str] = set()
    text = normalize_space(text)
    if len(text) < min_len:
        return grams
    for size in range(min_len, min(max_len, len(text)) + 1):
        for idx in range(0, len(text) - size + 1):
            grams.add(text[idx : idx + size])
    return grams


def bridge_phrases_from_quads(quads: list[dict[str, Any]]) -> set[str]:
    phrases: set[str] = set()
    for quad in quads:
        for key in ("target", "argument"):
            raw = normalize_space(quad.get(key))
            normalized = normalize_obfuscated_text(raw)
            for value in {raw, normalized}:
                if not value:
                    continue
                chunks = [chunk for chunk in re.split(r"[^\w\u4e00-\u9fff]+", value) if chunk]
                for chunk in chunks:
                    if len(chunk) == 1 and key == "target":
                        phrases.add(chunk)
                    elif len(chunk) >= 2:
                        phrases.add(chunk)
                        if len(chunk) <= 12:
                            phrases.update(ngrams(chunk))
    return phrases


def keep_longest_phrases(phrases: set[str]) -> list[str]:
    kept: list[str] = []
    for phrase in sorted(phrases, key=lambda item: (-len(item), item)):
        if any(phrase != other and phrase in other for other in kept):
            continue
        kept.append(phrase)
    return kept


def semantic_bridge_evidence(
    semantic_entries: list[LexiconEntry],
    golds: list[dict[str, Any]],
    preds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find explicit links between semantic definitions and the predicted/GT spans."""
    phrases = bridge_phrases_from_quads(golds) | bridge_phrases_from_quads(preds)
    evidence: list[dict[str, Any]] = []
    for entry in semantic_entries:
        haystack = normalize_obfuscated_text(
            f"{entry.term} {entry.category} {entry.definition}"
        )
        matched = keep_longest_phrases(
            {
                phrase
                for phrase in phrases
                if phrase and phrase in haystack
            }
        )
        if matched:
            evidence.append(
                {
                    "term": entry.term,
                    "category": entry.category,
                    "matched_phrases": matched[:6],
                    "definition": entry.definition,
                }
            )
    return evidence


def longest_common_substring_len(left: str, right: str) -> int:
    best = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left) + 1):
            if j - i <= best:
                continue
            if left[i:j] in right:
                best = j - i
    return best


def partial_term_hits(entries: list[LexiconEntry], content: str) -> list[str]:
    hits: list[str] = []
    for entry in entries:
        if entry.term in content:
            hits.append(entry.term)
            continue
        if len(entry.term) <= 2:
            if any(char in content for char in entry.term):
                hits.append(entry.term)
        elif longest_common_substring_len(entry.term, content) >= 2:
            hits.append(entry.term)
    return hits


def quad_to_text(quad: dict[str, Any]) -> str:
    return (
        f"{quad.get('target', '')} | {quad.get('argument', '')} | "
        f"{quad.get('targeted_group', '')} | {quad.get('hateful', '')}"
    )


def match_summary(
    preds: list[dict[str, Any]],
    golds: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    hard = greedy_matches(preds, golds, hard_match)
    soft = greedy_matches(preds, golds, lambda pred, gold: soft_match(pred, gold, threshold))
    hard_pairs = {(item["pred_idx"], item["gold_idx"]) for item in hard}
    soft_only = [
        item for item in soft if (item["pred_idx"], item["gold_idx"]) not in hard_pairs
    ]
    return {
        "hard_tp": len(hard),
        "soft_tp": len(soft),
        "soft_only_tp": len(soft_only),
        "hard_pairs": hard,
        "soft_pairs": soft,
        "soft_only_pairs": soft_only,
    }


def exact_status(
    exact_entries: list[LexiconEntry],
    semantic_entries: list[LexiconEntry],
    golds: list[dict[str, Any]],
) -> str:
    if not exact_entries:
        return "empty"
    gold_groups = group_set_from_quads(golds)
    exact_overlap = category_set(exact_entries) & gold_groups
    semantic_overlap = category_set(semantic_entries) & gold_groups
    if not exact_overlap and semantic_overlap:
        return "category_unrelated"
    return "present"


def score_candidate(case: dict[str, Any]) -> float:
    score = 0.0
    if case["exact_status"] == "empty":
        score += 20
    elif case["exact_status"] == "category_unrelated":
        score += 8

    score += 5 * case["match"]["hard_tp"]
    score += 3 * case["match"]["soft_only_tp"]
    score += 2 * len(case["semantic_gold_group_overlap"])
    score += min(len(case["semantic_terms"]), 5)
    score += min(len(case["semantic_partial_term_hits"]), 3)
    score += 4 * len(case.get("semantic_bridge_evidence", []))

    # Prefer compact, easy-to-read cases for a paper table.
    if len(case["content"]) > 120:
        score -= 2
    return score


def build_explanation(case: dict[str, Any]) -> str:
    if case["exact_status"] == "empty":
        exact_part = "精确子串匹配没有返回任何词条"
    elif case["exact_status"] == "category_unrelated":
        exact_part = "精确子串匹配只返回了与金标类别不重合的词条"
    else:
        exact_part = "精确子串匹配已有词条，但语义检索继续补充了近邻背景"

    semantic_terms = "、".join(case["semantic_terms"][:5]) or "无"
    overlap = "、".join(case["semantic_gold_group_overlap"]) or "相关类别"
    bridge_items = []
    for item in case.get("semantic_bridge_evidence", [])[:3]:
        phrases = "、".join(item["matched_phrases"][:3])
        bridge_items.append(f"{item['term']} -> {phrases}")
    bridge_part = (
        f"其中 {'；'.join(bridge_items)} 明确把语义词条定义与待抽取片段连接起来。"
        if bridge_items
        else ""
    )
    match_part = (
        f"预测与 GT 的 hard 命中数为 {case['match']['hard_tp']}，"
        f"soft 命中数为 {case['match']['soft_tp']}。"
    )
    return (
        f"{exact_part}；semantic retrieval 补入 {semantic_terms} 等词条，"
        f"这些词条为 {overlap} 判断提供了背景线索。{bridge_part}{match_part}"
    )


def build_case(
    row: dict[str, Any],
    lexicon_entries: list[LexiconEntry],
    lex_top_k: int,
    threshold: float,
    case_sensitive: bool,
) -> dict[str, Any]:
    content = str(row.get("content", ""))
    prompt_entries = parse_prompt_lexicon_entries(row)
    exact_entries, exact_all = reconstruct_exact_entries(
        content=content,
        lexicon_entries=lexicon_entries,
        top_k=lex_top_k,
        case_sensitive=case_sensitive,
    )
    semantic_entries = subtract_exact_terms(prompt_entries, exact_entries)
    golds = row.get("gt_quadruples") or []
    preds = row.get("pred_quadruples") or []
    match = match_summary(preds, golds, threshold)

    gold_groups = group_set_from_quads(golds)
    exact_groups = category_set(exact_entries)
    semantic_groups = category_set(semantic_entries)
    semantic_overlap = sorted(semantic_groups & gold_groups)

    case = {
        "id": row.get("id"),
        "content": content,
        "exact_status": exact_status(exact_entries, semantic_entries, golds),
        "exact_terms": [entry.term for entry in exact_entries],
        "exact_all_terms": [entry.term for entry in exact_all],
        "semantic_terms": [entry.term for entry in semantic_entries],
        "prompt_terms": [entry.term for entry in prompt_entries],
        "exact_entries": [entry.to_json() for entry in exact_entries],
        "semantic_entries": [entry.to_json() for entry in semantic_entries],
        "semantic_entry_explanations": [
            f"{entry.term}（{entry.category}）：{entry.definition}"
            for entry in semantic_entries
        ],
        "exact_gold_group_overlap": sorted(exact_groups & gold_groups),
        "semantic_gold_group_overlap": semantic_overlap,
        "semantic_partial_term_hits": partial_term_hits(semantic_entries, content),
        "semantic_bridge_evidence": semantic_bridge_evidence(
            semantic_entries, golds, preds
        ),
        "gt_quadruples": golds,
        "pred_quadruples": preds,
        "match": match,
    }
    case["score"] = round(score_candidate(case), 4)
    case["explanation"] = build_explanation(case)
    return case


def is_candidate(case: dict[str, Any]) -> bool:
    return bool(case["semantic_terms"]) and case["match"]["soft_tp"] > 0


def markdown_quad_list(quads: list[dict[str, Any]]) -> str:
    if not quads:
        return "- 无"
    return "\n".join(f"- `{quad_to_text(quad)}`" for quad in quads)


def markdown_lexicon_entry_list(entries: list[dict[str, str]]) -> str:
    if not entries:
        return "- 无"
    lines: list[str] = []
    for entry in entries:
        lines.append(
            f"- `{entry.get('term', '')}` ({entry.get('category', '')})："
            f"{entry.get('definition', '')}"
        )
    return "\n".join(lines)


def markdown_bridge_evidence(items: list[dict[str, Any]]) -> str:
    if not items:
        return "- 未检测到定义与 GT/pred 片段的直接字面桥接；主要依据类别一致和语义近邻关系。"
    lines: list[str] = []
    for item in items:
        phrases = "、".join(f"`{phrase}`" for phrase in item["matched_phrases"][:6])
        lines.append(
            f"- `{item['term']}` ({item['category']}) 的定义/词面连接了 {phrases}"
        )
    return "\n".join(lines)


def render_markdown(
    *,
    runner_output: Path,
    build_config: Path,
    lexicon_path: Path,
    summary: dict[str, Any],
    selected_cases: list[dict[str, Any]],
    top_candidates: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# 词典语义补全 Case Study",
        "",
        "## Method",
        "",
        f"- Runner output: `{runner_output}`",
        f"- Build config: `{build_config}`",
        f"- Lexicon: `{lexicon_path}`",
        f"- `lex_top_k`: {summary['lex_top_k']}",
        f"- `lex_sim_top_k`: {summary['lex_sim_top_k']}",
        f"- Similarity threshold for prediction audit: {summary['similarity_threshold']}",
        "- Exact 来源按词表顺序复现子串匹配；prompt 中不能由 exact top-k 解释的背景词条记为 semantic 补全。",
        "",
        "## Summary",
        "",
        f"- Total results: {summary['total_results']}",
        f"- Candidates with semantic terms and soft/hard prediction hit: {summary['candidate_count']}",
        f"- Candidates with empty exact match: {summary['exact_empty_candidate_count']}",
        f"- Selected cases: {', '.join(str(case['id']) for case in selected_cases)}",
        "",
        "## Selected Cases",
        "",
    ]

    for idx, case in enumerate(selected_cases, start=1):
        exact_text = "无" if not case["exact_terms"] else "、".join(case["exact_terms"])
        lines.extend(
            [
                f"### Case {idx}: id={case['id']}",
                "",
                f"- 原句：{case['content']}",
                f"- Exact 精确匹配结果：{exact_text}",
                "- Semantic 补全词条解释：",
                markdown_lexicon_entry_list(case["semantic_entries"][:5]),
                "- 补全作用证据：",
                markdown_bridge_evidence(case.get("semantic_bridge_evidence", [])[:5]),
                f"- 匹配复核：hard_tp={case['match']['hard_tp']}, "
                f"soft_tp={case['match']['soft_tp']}, "
                f"soft_only_tp={case['match']['soft_only_tp']}",
                "- GT：",
                markdown_quad_list(case["gt_quadruples"]),
                "- 模型输出：",
                markdown_quad_list(case["pred_quadruples"]),
                f"- 解释：{case['explanation']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Top Candidates",
            "",
            "| rank | id | score | exact_status | hard_tp | soft_tp | exact_terms | semantic_terms |",
            "|---:|---:|---:|---|---:|---:|---|---|",
        ]
    )
    for rank, case in enumerate(top_candidates, start=1):
        exact_terms = "、".join(case["exact_terms"]) if case["exact_terms"] else "无"
        semantic_terms = "、".join(case["semantic_terms"][:5])
        lines.append(
            f"| {rank} | {case['id']} | {case['score']} | {case['exact_status']} | "
            f"{case['match']['hard_tp']} | {case['match']['soft_tp']} | "
            f"{exact_terms} | {semantic_terms} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate lexicon semantic-completion case studies."
    )
    parser.add_argument("--runner-output", default=DEFAULT_RUNNER_OUTPUT)
    parser.add_argument("--build-config", default=None)
    parser.add_argument("--lexicon-data", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--selected-ids", default=DEFAULT_SELECTED_IDS)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--case-insensitive", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    runner_output = resolve_path(args.runner_output, repo_root)
    if runner_output is None:
        raise ValueError("--runner-output is required")
    exp_dir = infer_exp_dir(runner_output)
    build_config = (
        resolve_path(args.build_config, repo_root)
        if args.build_config
        else exp_dir / "build_config.json"
    )
    config = read_json(build_config)

    lexicon_path = (
        resolve_path(args.lexicon_data, repo_root)
        if args.lexicon_data
        else resolve_path(config["data_paths"]["lexicon_data_path"], repo_root)
    )
    if lexicon_path is None:
        raise ValueError("Lexicon path could not be resolved")

    out_dir = (
        resolve_path(args.out_dir, repo_root)
        if args.out_dir
        else exp_dir / "case_study"
    )
    if out_dir is None:
        raise ValueError("Output directory could not be resolved")
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval_settings = config.get("retrieval_settings", {})
    lex_top_k = int(retrieval_settings.get("lex_top_k", 5))
    lex_sim_top_k = int(retrieval_settings.get("lex_sim_top_k", 5))

    runner_data = read_json(runner_output)
    results = runner_data.get("results", [])
    lexicon_entries = load_lexicon_entries(lexicon_path)

    cases = [
        build_case(
            row=row,
            lexicon_entries=lexicon_entries,
            lex_top_k=lex_top_k,
            threshold=args.similarity_threshold,
            case_sensitive=not args.case_insensitive,
        )
        for row in results
    ]
    candidates = [case for case in cases if is_candidate(case)]
    candidates.sort(key=lambda case: (-case["score"], int(case["id"])))

    cases_by_id = {int(case["id"]): case for case in candidates}
    selected_ids = parse_id_list(args.selected_ids)
    missing_ids = [case_id for case_id in selected_ids if case_id not in cases_by_id]
    if missing_ids:
        raise ValueError(f"Selected ids are not valid candidates: {missing_ids}")
    selected_cases = [cases_by_id[case_id] for case_id in selected_ids]
    top_candidates = candidates[: args.top_n]

    summary = {
        "runner_output": str(runner_output),
        "build_config": str(build_config),
        "lexicon_data": str(lexicon_path),
        "lex_top_k": lex_top_k,
        "lex_sim_top_k": lex_sim_top_k,
        "similarity_threshold": args.similarity_threshold,
        "case_sensitive_exact": not args.case_insensitive,
        "total_results": len(results),
        "candidate_count": len(candidates),
        "exact_empty_candidate_count": sum(
            1 for case in candidates if case["exact_status"] == "empty"
        ),
        "selected_ids": selected_ids,
    }

    json_payload = {
        "summary": summary,
        "selected_cases": selected_cases,
        "top_candidates": top_candidates,
        "candidates": candidates,
    }
    json_path = out_dir / "lexicon_semantic_completion_cases.json"
    md_path = out_dir / "lexicon_semantic_completion_cases.md"
    write_json(json_path, json_payload)
    md_path.write_text(
        render_markdown(
            runner_output=runner_output,
            build_config=build_config,
            lexicon_path=lexicon_path,
            summary=summary,
            selected_cases=selected_cases,
            top_candidates=top_candidates,
        ),
        encoding="utf-8",
    )

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote Markdown: {md_path}")
    print(
        "Summary: "
        f"{summary['candidate_count']} candidates, "
        f"{summary['exact_empty_candidate_count']} exact-empty candidates, "
        f"selected ids={selected_ids}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
