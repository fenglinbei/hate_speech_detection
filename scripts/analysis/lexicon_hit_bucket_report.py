#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build lexicon-hit bucket metrics for a runner output.

Buckets are defined by whether each test instance has exact lexicon matches and
high-confidence semantic lexicon retrieval matches. Metrics are recomputed on
the main runner output only; ablation/test prompt files are used only to recover
lexicon provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
BOOTSTRAP_DIR = REPO_ROOT / "scripts" / "paired_bootstrap"
for path in (SRC_DIR, BOOTSTRAP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paired_bootstrap_llm import (  # noqa: E402
    InstanceCounts,
    SystemData,
    TupleCounts,
    compute_instance_counts,
    load_system,
    observed_metrics,
    sum_counts,
)
from prompt import LEXICON_RAG_PROMPT  # noqa: E402


BUCKET_ORDER = ["Exact-hit", "Semantic-only-hit", "Both-hit", "No-hit"]
METRIC_COLUMNS = [
    ("f1_target", "Tar-F1"),
    ("f1_hate", "Hate-F1"),
    ("f1_avg", "Avg-F1"),
    ("f1_hard", "Hard-F1"),
    ("f1_soft", "Soft-F1"),
]

LEXICON_ENTRY_RE = re.compile(
    r"关键词：(?P<term>.*?)\n"
    r"类别：(?P<category>.*?)\n"
    r"定义：(?P<definition>.*?)(?=\n###|\Z)",
    re.S,
)


@dataclass(frozen=True)
class LexiconEntry:
    index: int
    term: str
    category: str
    definition: str
    prompt: str


@dataclass(frozen=True)
class SemanticCandidate:
    index: int
    term: str
    category: str
    score: float


@dataclass(frozen=True)
class SampleLexiconInfo:
    instance_id: str
    exact_terms: list[str]
    exact_hit: bool
    semantic_candidates: list[SemanticCandidate]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report bucketed F1 metrics by exact/semantic lexicon hits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runner-output",
        type=Path,
        default=Path("output/runner/method_comparison/ours_prompt_al1280.json"),
        help="Main runner output JSON to evaluate.",
    )
    parser.add_argument(
        "--exact-test",
        type=Path,
        default=Path("exps/ablation/wo_semantic_match/exp_3992dbfb11/data/test.json"),
        help="Exact-only test prompt JSON used to recover exact lexicon hits.",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=Path("data/lexicon/annotated_lexicon.json"),
        help="Annotated lexicon JSON.",
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=Path("models/base/bge-large-zh-v1.5"),
        help="SentenceTransformer model path for semantic lexicon scoring.",
    )
    parser.add_argument(
        "--semantic-top-k",
        type=int,
        default=5,
        help="Number of semantic lexicon candidates to consider before exact-term exclusion.",
    )
    parser.add_argument(
        "--threshold-grid",
        default="0.50:0.95:0.01",
        help="Inclusive threshold grid as start:end:step.",
    )
    parser.add_argument(
        "--min-bucket-size",
        type=int,
        default=30,
        help="Minimum desired size for every bucket when selecting threshold.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("output/analyse/lexicon_bucket/ours_prompt_al1280_bucket_metrics.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Audit JSON output path. Defaults to --out-md with .json suffix.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Soft-match threshold for metric recomputation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Embedding device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding model encoding.",
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for validating recomputed all-sample metrics.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable SentenceTransformer progress bars.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_lexicon(path: Path) -> list[LexiconEntry]:
    payload = read_json(path)
    terms = payload.get("terms")
    if not isinstance(terms, list):
        raise ValueError(f"{path} must contain a top-level list field named 'terms'.")

    entries: list[LexiconEntry] = []
    for index, row in enumerate(terms):
        term = str(row.get("term", "")).strip()
        category = str(row.get("category", "")).strip()
        definition = str(row.get("definition", "")).strip()
        if not term:
            continue
        prompt = (
            LEXICON_RAG_PROMPT.replace("{word}", term)
            .replace("{category}", category)
            .replace("{definition}", definition)
        )
        entries.append(
            LexiconEntry(
                index=index,
                term=term,
                category=category,
                definition=definition,
                prompt=prompt,
            )
        )
    if not entries:
        raise ValueError(f"{path} did not contain any usable lexicon terms.")
    return entries


def parse_prompt_lexicon_terms(row: dict[str, Any]) -> list[str]:
    messages_list = row.get("messages_list")
    if not messages_list:
        return []
    try:
        prompt = str(messages_list[0][1]["content"])
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Could not parse messages_list for id={row.get('id')!r}.") from exc

    background = prompt.split("示例：", 1)[0]
    return [match.group("term").strip() for match in LEXICON_ENTRY_RE.finditer(background)]


def reconstruct_exact_terms(content: str, lexicon_entries: Sequence[LexiconEntry]) -> list[str]:
    # LexiconRetriever.including_retrieve iterates word2item.keys(), so duplicate
    # lexicon terms are checked only once while preserving first insertion order.
    seen: set[str] = set()
    terms: list[str] = []
    for entry in lexicon_entries:
        if entry.term in seen:
            continue
        seen.add(entry.term)
        if entry.term and entry.term in content:
            terms.append(entry.term)
    return terms


def load_exact_test(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a JSON list.")

    by_id: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(payload):
        if "id" not in row:
            raise ValueError(f"{path} item {idx} is missing 'id'.")
        instance_id = str(row["id"])
        if instance_id in by_id:
            raise ValueError(f"{path} contains duplicate id {instance_id!r}.")
        by_id[instance_id] = row
    return by_id


def validate_id_alignment(system: SystemData, exact_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    runner_ids = set(system.ids)
    exact_ids = set(exact_by_id)
    missing_in_exact = sorted(runner_ids - exact_ids, key=sort_id)
    missing_in_runner = sorted(exact_ids - runner_ids, key=sort_id)
    if missing_in_exact or missing_in_runner:
        raise ValueError(
            "Runner/exact id mismatch: "
            f"missing_in_exact={missing_in_exact[:10]}, "
            f"missing_in_runner={missing_in_runner[:10]}"
        )
    return {
        "runner_total": len(system.ids),
        "exact_total": len(exact_by_id),
        "ids_aligned": True,
    }


def validate_exact_parser(
    exact_by_id: dict[str, dict[str, Any]],
    lexicon_entries: Sequence[LexiconEntry],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    exact_terms_by_id: dict[str, list[str]] = {}
    mismatches: list[dict[str, Any]] = []

    for instance_id, row in exact_by_id.items():
        prompt_terms = parse_prompt_lexicon_terms(row)
        reconstructed_terms = reconstruct_exact_terms(str(row.get("content", "")), lexicon_entries)
        expected_prefix = reconstructed_terms[: len(prompt_terms)]
        prompt_ok = prompt_terms == expected_prefix
        hit_ok = bool(prompt_terms) == bool(reconstructed_terms)
        if not (prompt_ok and hit_ok):
            mismatches.append(
                {
                    "id": instance_id,
                    "prompt_terms": prompt_terms,
                    "reconstructed_terms": reconstructed_terms,
                    "expected_prefix": expected_prefix,
                }
            )
        exact_terms_by_id[instance_id] = prompt_terms

    if mismatches:
        examples = json.dumps(mismatches[:5], ensure_ascii=False, indent=2)
        raise ValueError(f"Exact parser validation failed; examples:\n{examples}")

    exact_hit_count = sum(1 for terms in exact_terms_by_id.values() if terms)
    return exact_terms_by_id, {
        "exact_parser_ok": True,
        "exact_hit_count": exact_hit_count,
        "no_exact_count": len(exact_terms_by_id) - exact_hit_count,
    }


def sort_id(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def encode_texts(
    model: Any,
    texts: Sequence[str],
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return l2_normalize(embeddings)


def encode_texts_with_transformers(
    model_path: Path,
    device: str,
    texts: Sequence[str],
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Semantic lexicon scoring requires either sentence_transformers or transformers."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()

    pooling_config = {}
    pooling_config_path = model_path / "1_Pooling" / "config.json"
    if pooling_config_path.exists():
        pooling_config = read_json(pooling_config_path)
    use_cls = bool(pooling_config.get("pooling_mode_cls_token", False))

    vectors: list[np.ndarray] = []
    batches: Iterable[Sequence[str]] = (
        texts[start:start + batch_size]
        for start in range(0, len(texts), batch_size)
    )
    if show_progress:
        try:
            from tqdm import tqdm

            total = math.ceil(len(texts) / batch_size)
            batches = tqdm(batches, total=total, desc="Encoding", unit="batch")
        except Exception:
            pass

    with torch.no_grad():
        for batch in batches:
            encoded = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            token_embeddings = output.last_hidden_state
            if use_cls:
                pooled = token_embeddings[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))

    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return l2_normalize(np.vstack(vectors))


def compute_embeddings(
    model_path: Path,
    device: str,
    lexicon_prompts: Sequence[str],
    contents: Sequence[str],
    batch_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(model_path), device=device)
        lexicon_embeddings = encode_texts(model, lexicon_prompts, batch_size, show_progress)
        query_embeddings = encode_texts(model, contents, batch_size, show_progress)
        return lexicon_embeddings, query_embeddings
    except ImportError:
        lexicon_embeddings = encode_texts_with_transformers(
            model_path=model_path,
            device=device,
            texts=lexicon_prompts,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        query_embeddings = encode_texts_with_transformers(
            model_path=model_path,
            device=device,
            texts=contents,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return lexicon_embeddings, query_embeddings


def compute_semantic_candidates(
    system: SystemData,
    lexicon_entries: Sequence[LexiconEntry],
    exact_terms_by_id: dict[str, list[str]],
    model_path: Path,
    device: str,
    semantic_top_k: int,
    batch_size: int,
    show_progress: bool,
) -> dict[str, list[SemanticCandidate]]:
    if semantic_top_k <= 0:
        return {instance_id: [] for instance_id in system.ids}

    lexicon_prompts = [entry.prompt for entry in lexicon_entries]
    contents = [str(system.results_by_id[instance_id].get("content", "")) for instance_id in system.ids]
    lexicon_embeddings, query_embeddings = compute_embeddings(
        model_path=model_path,
        device=device,
        lexicon_prompts=lexicon_prompts,
        contents=contents,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    similarities = query_embeddings @ lexicon_embeddings.T
    k = min(semantic_top_k, len(lexicon_entries))
    candidates_by_id: dict[str, list[SemanticCandidate]] = {}

    for row_idx, instance_id in enumerate(system.ids):
        row_scores = similarities[row_idx]
        if k == len(lexicon_entries):
            top_indices = np.argsort(-row_scores)
        else:
            rough = np.argpartition(-row_scores, k - 1)[:k]
            top_indices = rough[np.argsort(-row_scores[rough])]

        exact_counter = Counter(exact_terms_by_id.get(instance_id, []))
        candidates: list[SemanticCandidate] = []
        for lex_idx in top_indices.tolist():
            entry = lexicon_entries[int(lex_idx)]
            if exact_counter[entry.term] > 0:
                exact_counter[entry.term] -= 1
                continue
            candidates.append(
                SemanticCandidate(
                    index=entry.index,
                    term=entry.term,
                    category=entry.category,
                    score=float(row_scores[int(lex_idx)]),
                )
            )
        candidates_by_id[instance_id] = candidates
    return candidates_by_id


def parse_threshold_grid(grid: str) -> list[float]:
    parts = grid.split(":")
    if len(parts) != 3:
        raise ValueError("--threshold-grid must be formatted as start:end:step.")
    start, end, step = [Decimal(part) for part in parts]
    if step <= 0:
        raise ValueError("--threshold-grid step must be positive.")
    if start > end:
        raise ValueError("--threshold-grid start must be <= end.")

    thresholds: list[float] = []
    value = start
    while value <= end:
        thresholds.append(float(value))
        value += step
    if not thresholds or thresholds[-1] < float(end):
        thresholds.append(float(end))
    return thresholds


def bucket_name(exact_hit: bool, semantic_hit: bool) -> str:
    if exact_hit and semantic_hit:
        return "Both-hit"
    if exact_hit:
        return "Exact-hit"
    if semantic_hit:
        return "Semantic-only-hit"
    return "No-hit"


def bucket_ids_for_threshold(
    sample_info: dict[str, SampleLexiconInfo],
    ids: Sequence[str],
    tau: float,
) -> dict[str, list[str]]:
    buckets = {bucket: [] for bucket in BUCKET_ORDER}
    for instance_id in ids:
        info = sample_info[instance_id]
        semantic_hit = any(candidate.score >= tau for candidate in info.semantic_candidates)
        buckets[bucket_name(info.exact_hit, semantic_hit)].append(instance_id)
    return buckets


def choose_threshold(
    sample_info: dict[str, SampleLexiconInfo],
    ids: Sequence[str],
    thresholds: Sequence[float],
    min_bucket_size: int,
) -> tuple[float, list[dict[str, Any]], str | None]:
    sweep: list[dict[str, Any]] = []
    first_valid: float | None = None

    for tau in thresholds:
        buckets = bucket_ids_for_threshold(sample_info, ids, tau)
        sizes = {bucket: len(buckets[bucket]) for bucket in BUCKET_ORDER}
        min_size = min(sizes.values()) if sizes else 0
        sweep.append(
            {
                "tau": round(tau, 6),
                "bucket_sizes": sizes,
                "min_bucket_size": min_size,
            }
        )
        if first_valid is None and min_size >= min_bucket_size:
            first_valid = tau

    if first_valid is not None:
        return first_valid, sweep, None

    best = max(sweep, key=lambda row: (row["min_bucket_size"], -row["tau"]))
    warning = (
        "No threshold in the requested grid gives every bucket at least "
        f"{min_bucket_size} samples; selected tau={best['tau']:.6g} because it "
        f"maximizes the smallest bucket size ({best['min_bucket_size']})."
    )
    return float(best["tau"]), sweep, warning


def counts_to_dict(counts: TupleCounts) -> dict[str, int]:
    return {"tp": int(counts.tp), "fp": int(counts.fp), "fn": int(counts.fn)}


def metric_counts_for_instances(instances: Sequence[InstanceCounts]) -> dict[str, dict[str, int]]:
    return {
        "hard": counts_to_dict(sum_counts(item.hard for item in instances)),
        "soft": counts_to_dict(sum_counts(item.soft for item in instances)),
        "target": counts_to_dict(sum_counts(item.target for item in instances)),
        "hate": counts_to_dict(sum_counts(item.hate for item in instances)),
    }


def validate_metrics_against_payload(
    system: SystemData,
    all_metrics: dict[str, float],
    tolerance: float,
) -> dict[str, Any]:
    payload = read_json(system.path)
    metric_payload = payload.get("metric", {})
    metric_paths = {
        "f1_hard": ("f1_hard",),
        "f1_soft": ("f1_soft",),
        "f1_avg": ("f1_avg",),
        "f1_target": ("field_metrics", "targeted_group", "f1"),
        "f1_hate": ("field_metrics", "hateful", "f1"),
    }
    diffs: dict[str, dict[str, float]] = {}

    for metric_name, path in metric_paths.items():
        expected = nested_get(metric_payload, path)
        if expected is None:
            continue
        observed = all_metrics[metric_name]
        diff = abs(float(expected) - observed)
        diffs[metric_name] = {
            "payload": float(expected),
            "recomputed": observed,
            "abs_diff": diff,
        }
        if diff > tolerance:
            raise ValueError(
                f"Metric validation failed for {metric_name}: payload={expected}, "
                f"recomputed={observed:.8f}, diff={diff:.8f}, tolerance={tolerance}."
            )
    return {"metric_validation_ok": True, "tolerance": tolerance, "diffs": diffs}


def nested_get(payload: dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def compute_bucket_metrics(
    buckets: dict[str, list[str]],
    instance_counts_by_id: dict[str, InstanceCounts],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for bucket in BUCKET_ORDER:
        ids = buckets[bucket]
        instances = [instance_counts_by_id[instance_id] for instance_id in ids]
        metrics = observed_metrics(instances)
        out[bucket] = {
            **{key: float(value) for key, value in metrics.items()},
            "counts": metric_counts_for_instances(instances),
        }
    return out


def format_float(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.4f}"


def render_markdown(
    args: argparse.Namespace,
    out_json: Path,
    selected_tau: float,
    threshold_warning: str | None,
    buckets: dict[str, list[str]],
    bucket_metrics: dict[str, dict[str, Any]],
    validations: dict[str, Any],
) -> str:
    lines = [
        "# Lexicon Hit Bucket Metrics Report",
        "",
        "## Settings",
        "",
        f"- Runner output: `{rel_path(resolve_path(args.runner_output))}`",
        f"- Exact source: `{rel_path(resolve_path(args.exact_test))}`",
        f"- Lexicon: `{rel_path(resolve_path(args.lexicon))}`",
        f"- Semantic model: `{rel_path(resolve_path(args.embedding_model))}`",
        f"- Semantic top-k: `{args.semantic_top_k}`",
        f"- Semantic threshold: `{selected_tau:.4f}`",
        f"- Threshold grid: `{args.threshold_grid}`",
        f"- Minimum bucket size target: `{args.min_bucket_size}`",
        f"- Audit JSON: `{rel_path(out_json)}`",
        "",
    ]
    if threshold_warning:
        lines.extend(["## Warning", "", threshold_warning, ""])

    lines.extend(
        [
            "## Metrics",
            "",
            "| Bucket | N | Semantic threshold | Tar-F1 | Hate-F1 | Avg-F1 | Hard-F1 | Soft-F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket in BUCKET_ORDER:
        metrics = bucket_metrics[bucket]
        row = [
            bucket,
            str(len(buckets[bucket])),
            f"{selected_tau:.4f}",
            *[format_float(metrics[key]) for key, _label in METRIC_COLUMNS],
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- ID alignment: `{validations['id_alignment']['runner_total']}` runner samples and "
            f"`{validations['id_alignment']['exact_total']}` exact-source samples.",
            f"- Exact parser: `{validations['exact_parser']['exact_hit_count']}` exact-hit samples and "
            f"`{validations['exact_parser']['no_exact_count']}` no-exact samples.",
            f"- Metric validation tolerance: `{validations['metric_validation']['tolerance']}`.",
            "- Bucket integrity: all samples assigned to exactly one bucket.",
        ]
    )
    return "\n".join(lines)


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def semantic_candidates_to_json(candidates: Sequence[SemanticCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "index": candidate.index,
            "term": candidate.term,
            "category": candidate.category,
            "score": round(candidate.score, 8),
        }
        for candidate in candidates
    ]


def build_audit_json(
    args: argparse.Namespace,
    out_md: Path,
    selected_tau: float,
    threshold_warning: str | None,
    threshold_sweep: list[dict[str, Any]],
    validations: dict[str, Any],
    buckets: dict[str, list[str]],
    bucket_metrics: dict[str, dict[str, Any]],
    sample_info: dict[str, SampleLexiconInfo],
    ids: Sequence[str],
) -> dict[str, Any]:
    samples: dict[str, Any] = {}
    for instance_id in ids:
        info = sample_info[instance_id]
        semantic_hit_terms = [
            {
                "term": candidate.term,
                "category": candidate.category,
                "score": round(candidate.score, 8),
            }
            for candidate in info.semantic_candidates
            if candidate.score >= selected_tau
        ]
        semantic_hit = bool(semantic_hit_terms)
        samples[instance_id] = {
            "bucket": bucket_name(info.exact_hit, semantic_hit),
            "exact_hit": info.exact_hit,
            "semantic_hit": semantic_hit,
            "exact_terms": info.exact_terms,
            "semantic_candidates": semantic_candidates_to_json(info.semantic_candidates),
            "semantic_hit_terms": semantic_hit_terms,
        }

    return {
        "runner_output": rel_path(resolve_path(args.runner_output)),
        "exact_test": rel_path(resolve_path(args.exact_test)),
        "lexicon": rel_path(resolve_path(args.lexicon)),
        "embedding_model": rel_path(resolve_path(args.embedding_model)),
        "markdown_report": rel_path(out_md),
        "semantic_top_k": args.semantic_top_k,
        "threshold_grid": args.threshold_grid,
        "min_bucket_size": args.min_bucket_size,
        "selected_tau": round(selected_tau, 8),
        "threshold_warning": threshold_warning,
        "threshold_sweep": threshold_sweep,
        "validations": validations,
        "bucket_order": BUCKET_ORDER,
        "bucket_sizes": {bucket: len(buckets[bucket]) for bucket in BUCKET_ORDER},
        "bucket_ids": buckets,
        "metrics": bucket_metrics,
        "samples": samples,
    }


def validate_bucket_integrity(buckets: dict[str, list[str]], ids: Sequence[str]) -> dict[str, Any]:
    assigned: list[str] = []
    for bucket in BUCKET_ORDER:
        assigned.extend(buckets[bucket])
    assigned_counter = Counter(assigned)
    duplicate_ids = sorted(
        [instance_id for instance_id, count in assigned_counter.items() if count != 1],
        key=sort_id,
    )
    missing_ids = sorted(set(ids) - set(assigned), key=sort_id)
    if duplicate_ids or missing_ids or len(assigned) != len(ids):
        raise ValueError(
            "Bucket integrity failed: "
            f"duplicates={duplicate_ids[:10]}, missing={missing_ids[:10]}, "
            f"assigned={len(assigned)}, expected={len(ids)}"
        )
    return {
        "bucket_integrity_ok": True,
        "assigned_total": len(assigned),
        "expected_total": len(ids),
    }


def main() -> int:
    args = parse_args()

    runner_output = resolve_path(args.runner_output)
    exact_test = resolve_path(args.exact_test)
    lexicon_path = resolve_path(args.lexicon)
    embedding_model = resolve_path(args.embedding_model)
    out_md = resolve_path(args.out_md)
    out_json = resolve_path(args.out_json) if args.out_json else out_md.with_suffix(".json")

    system = load_system(runner_output)
    exact_by_id = load_exact_test(exact_test)
    lexicon_entries = load_lexicon(lexicon_path)

    id_alignment = validate_id_alignment(system, exact_by_id)
    exact_terms_by_id, exact_parser_validation = validate_exact_parser(exact_by_id, lexicon_entries)

    device = resolve_device(args.device)
    semantic_candidates_by_id = compute_semantic_candidates(
        system=system,
        lexicon_entries=lexicon_entries,
        exact_terms_by_id=exact_terms_by_id,
        model_path=embedding_model,
        device=device,
        semantic_top_k=args.semantic_top_k,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )

    sample_info = {
        instance_id: SampleLexiconInfo(
            instance_id=instance_id,
            exact_terms=exact_terms_by_id[instance_id],
            exact_hit=bool(exact_terms_by_id[instance_id]),
            semantic_candidates=semantic_candidates_by_id[instance_id],
        )
        for instance_id in system.ids
    }

    thresholds = parse_threshold_grid(args.threshold_grid)
    selected_tau, threshold_sweep, threshold_warning = choose_threshold(
        sample_info=sample_info,
        ids=system.ids,
        thresholds=thresholds,
        min_bucket_size=args.min_bucket_size,
    )
    buckets = bucket_ids_for_threshold(sample_info, system.ids, selected_tau)
    bucket_integrity = validate_bucket_integrity(buckets, system.ids)

    instance_counts_by_id = {
        instance_id: compute_instance_counts(
            system.results_by_id[instance_id],
            args.similarity_threshold,
        )
        for instance_id in system.ids
    }
    all_metrics = observed_metrics([instance_counts_by_id[instance_id] for instance_id in system.ids])
    metric_validation = validate_metrics_against_payload(
        system=system,
        all_metrics=all_metrics,
        tolerance=args.metric_tolerance,
    )
    bucket_metrics = compute_bucket_metrics(buckets, instance_counts_by_id)

    validations = {
        "id_alignment": id_alignment,
        "exact_parser": exact_parser_validation,
        "metric_validation": metric_validation,
        "bucket_integrity": bucket_integrity,
        "embedding_device": device,
    }
    audit_payload = build_audit_json(
        args=args,
        out_md=out_md,
        selected_tau=selected_tau,
        threshold_warning=threshold_warning,
        threshold_sweep=threshold_sweep,
        validations=validations,
        buckets=buckets,
        bucket_metrics=bucket_metrics,
        sample_info=sample_info,
        ids=system.ids,
    )
    write_json(out_json, audit_payload)
    write_text(
        out_md,
        render_markdown(
            args=args,
            out_json=out_json,
            selected_tau=selected_tau,
            threshold_warning=threshold_warning,
            buckets=buckets,
            bucket_metrics=bucket_metrics,
            validations=validations,
        ),
    )

    print(f"Wrote Markdown report: {out_md}")
    print(f"Wrote audit JSON: {out_json}")
    print(f"Selected semantic threshold: {selected_tau:.4f}")
    print("Bucket sizes:", json.dumps(audit_payload["bucket_sizes"], ensure_ascii=False))
    if threshold_warning:
        print(f"Warning: {threshold_warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
