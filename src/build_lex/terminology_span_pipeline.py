"""Fail-closed lifecycle for category-free terminology span discovery.

The lifecycle deliberately keeps three concerns separate:

* cheap, label-free corpus census and deterministic pilot sampling;
* resumable local-Qwen span proposals whose offsets are resolved by code; and
* human Gold / statistical gates that must be approved before a full scan.

Provider output is never a lexicon entry.  It is only a proposal that may be
promoted after exact-boundary review.  All model-visible payloads contain the
raw ``content`` field and nothing from the task annotations.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import html
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from build_lex.train_only import FrozenTrainInput, resolve_train_input
from data.training_artifacts import (
    canonical_json_bytes,
    canonical_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    write_canonical_json,
    write_canonical_jsonl,
)


CONFIG_SCHEMA_VERSION = "stage1-terminology-span-pipeline-config/v1"
CENSUS_SCHEMA_VERSION = "terminology-span-census/v1"
SPAN_FRAME_SCHEMA_VERSION = "terminology-span-frame/v1"
PILOT_FRAME_SCHEMA_VERSION = "terminology-span-pilot-frame/v1"
QWEN_ANNOTATION_SCHEMA_VERSION = "terminology-qwen-span-annotation/v1"
HUMAN_REVIEW_SCHEMA_VERSION = "terminology-span-human-review/v1"
GOLD_SCHEMA_VERSION = "terminology-gold/v1"
GATE_SCHEMA_VERSION = "terminology-gate/v1"
PILOT_DECISION_SCHEMA_VERSION = "span-pilot-decision/v1"
FULL_AUDIT_DECISION_SCHEMA_VERSION = "full-span-audit-decision/v1"
CHECKPOINT_SCHEMA_VERSION = "terminology-span-checkpoint/v1"
REVIEW_PACKAGE_SCHEMA_VERSION = "terminology-span-review-package/v1"
RESOURCE_ROLE = "terminology-understanding-library/v1"
SOURCE_POLICY = "content-only-no-task-labels/v1"
PROMPT_VERSION = "category-free-exhaustive-minimal-span/v1"
SAMPLING_POLICY_VERSION = "content-feature-six-stratum-200-100-plus-60/v1"
RANKER_POLICY_VERSION = "l2-logistic-liblinear-c1/v1"

TASK_FIELD_KEYS = frozenset(
    {
        "annotation_count",
        "categories",
        "category",
        "category_counts",
        "category_purity",
        "hate_count",
        "hate_precision",
        "hateful",
        "label",
        "labels",
        "log_odds",
        "non_hate_count",
        "nonhate_penalty",
        "primary_category",
        "targeted_group",
        "target",
        "argument",
        "quadruples",
    }
)
PUBLIC_TASK_KEYS = frozenset({"task_id", "blind_alias", "content"})
TERMINAL_STATUSES = frozenset({"success", "empty", "exception_resolved"})
PILOT_PHASES = ("tune", "validation")
PILOT_STRATA = (
    "rule_empty",
    "candidate_dense",
    "mixed_obfuscated",
    "long_content",
    "candidate_sparse",
    "random_baseline",
)
FEATURE_NAMES = (
    "qwen_source",
    "rule_source",
    "both_sources",
    "surface_length",
    "log_doc_frequency",
    "pmi",
    "left_entropy",
    "right_entropy",
    "rule_score",
    "special_form",
    "mixed_char_classes",
)

RETRYABLE_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._'\-][A-Za-z0-9]+)*")
_HASHTAG_RE = re.compile(r"[#＃][^\s#＃]{1,20}")
_MIXED_TOKEN_RE = re.compile(r"(?<![\w\u3400-\u9fff])[A-Za-z0-9\u3400-\u9fff]{2,20}(?![\w\u3400-\u9fff])")
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "](?:[\uFE0E\uFE0F\u200D]|[\U0001F300-\U0001FAFF\u2600-\u27BF])*"
)

SYSTEM_PROMPT = """你是一名中文术语 span 提议员。你只会收到一条原始文本，不会收到数据集标签、类别或既有标注。

任务：找出为了正确理解这条文本而值得额外解释的最小连续原文片段。它可以是网络用语、缩写、隐语、谐音/替换写法、圈层表达、具有特殊语用的称谓，或语境中不能仅按普通字面组合稳定理解的表达。

约束：
1. 每个 surface 必须逐字复制自原文，并用 occurrence_ordinal 指明它在原文中第几次出现（从 1 开始）。
2. 只选最小且语义完整的连续片段；不要包含句尾语气词、无关修饰语、整句或分句。
3. 不要仅因词语低频就收录；普通可组合短语、纯人名以及不具有可复用解释的一次性乱码应排除。
4. 中性身份词、俚语或语境依赖表达都可以提议；不要判断仇恨、目标群体或任何任务类别。
5. 最多 8 个 span，可以返回空数组。

只输出一个 JSON 对象，不要 Markdown：
{
  "spans": [
    {
      "surface": "原文逐字子串",
      "occurrence_ordinal": 1,
      "reason": "简述为何需要额外理解"
    }
  ],
  "record_reason": "简述整体提取或空结果原因"
}"""


def _span_implementation_sha256() -> str:
    """Bind rules, offset parsing, ranking and provider handling to source."""

    return sha256_file(Path(__file__).resolve())


class TerminologySpanError(RuntimeError):
    """Raised when a span lifecycle invariant is violated."""


class SpanProviderError(TerminologySpanError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.http_status = http_status


@dataclass(frozen=True)
class QwenProviderConfig:
    model: str
    api_base: str
    api_key: str
    concurrency: int
    timeout_seconds: int
    max_attempts_per_task: int
    temperature: float
    top_p: float
    max_tokens: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "provider": "qwen",
            "model": self.model,
            "api_base_origin": _api_origin(self.api_base),
            "concurrency": self.concurrency,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts_per_task": self.max_attempts_per_task,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class AttemptReservation:
    task_id: str
    attempt_no: int
    request_sha256: str


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(inner) for inner in value]
    return value


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(_plain(value))).hexdigest()


def _normalize_content(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _sampling_key(seed: str, namespace: str, *values: str) -> str:
    return hashlib.sha256(
        "\x1f".join((seed, namespace, *values)).encode("utf-8")
    ).hexdigest()


def _api_origin(api_base: str) -> str:
    parsed = urlsplit(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise TerminologySpanError("qwen api_base is malformed")
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _chat_url(api_base: str) -> str:
    return api_base.rstrip("/") + "/chat/completions"


def _sanitize_error(error: BaseException) -> str:
    value = re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted>", str(error))
    return value[:1000]


def _forbidden_key_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, inner in value.items():
            child = (*path, str(key))
            if str(key).lower() in TASK_FIELD_KEYS:
                found.append(".".join(child))
            found.extend(_forbidden_key_paths(inner, child))
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            found.extend(_forbidden_key_paths(inner, (*path, str(index))))
    return found


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the pilot/full span protocol config."""

    value = load_json(path)
    required = {
        "schema_version",
        "resource_role",
        "source_policy",
        "expected_fit_count",
        "artifact_root",
        "sampling",
        "extraction",
        "qwen",
        "gates",
        "ranking",
        "full_audit",
        "resolution",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise TerminologySpanError("terminology span config fields are not canonical")
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise TerminologySpanError("unsupported terminology span config schema")
    if value.get("resource_role") != RESOURCE_ROLE:
        raise TerminologySpanError("span pipeline must target the terminology library")
    if value.get("source_policy") != SOURCE_POLICY:
        raise TerminologySpanError("span pipeline must be content-only")
    if _forbidden_key_paths(value):
        raise TerminologySpanError("span config contains a forbidden task field")
    expected = value.get("expected_fit_count")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise TerminologySpanError("expected_fit_count must be positive")
    artifact_root = value.get("artifact_root")
    if not isinstance(artifact_root, str) or not artifact_root:
        raise TerminologySpanError("artifact_root must be non-empty text")
    artifact_path = Path(artifact_root)
    if artifact_path.is_absolute() or ".." in artifact_path.parts:
        raise TerminologySpanError("artifact_root must be workspace-relative")

    sampling = value.get("sampling")
    if not isinstance(sampling, Mapping) or set(sampling) != {
        "seed",
        "tune_count",
        "validation_count",
        "hidden_repeat_count",
        "minimum_repeat_gap",
    }:
        raise TerminologySpanError("sampling config is not canonical")
    if (
        not isinstance(sampling.get("seed"), str)
        or not sampling["seed"]
        or sampling.get("tune_count") != 200
        or sampling.get("validation_count") != 100
        or sampling.get("hidden_repeat_count") != 60
        or not isinstance(sampling.get("minimum_repeat_gap"), int)
        or sampling["minimum_repeat_gap"] < 12
    ):
        raise TerminologySpanError("pilot sampling must be 200+100 with 60 repeats")

    extraction = value.get("extraction")
    expected_extraction = {
        "zh_min_chars",
        "zh_max_chars",
        "latin_max_tokens",
        "max_rule_spans_per_record",
        "max_qwen_spans_per_record",
        "max_surface_chars",
        "max_text_chars_per_record",
    }
    if not isinstance(extraction, Mapping) or set(extraction) != expected_extraction:
        raise TerminologySpanError("extraction config is not canonical")
    if (
        extraction.get("zh_min_chars") != 2
        or extraction.get("zh_max_chars") != 8
        or extraction.get("latin_max_tokens") != 4
        or extraction.get("max_qwen_spans_per_record") != 8
    ):
        raise TerminologySpanError("span length and Qwen cap differ from the frozen plan")
    for key in (
        "max_rule_spans_per_record",
        "max_surface_chars",
        "max_text_chars_per_record",
    ):
        if isinstance(extraction.get(key), bool) or not isinstance(extraction.get(key), int) or extraction[key] < 1:
            raise TerminologySpanError(f"extraction.{key} must be positive")

    qwen = value.get("qwen")
    qwen_keys = {
        "provider",
        "model",
        "api_base",
        "api_key_env",
        "concurrency",
        "timeout_seconds",
        "max_attempts_per_task",
        "temperature",
        "top_p",
        "max_tokens",
    }
    if not isinstance(qwen, Mapping) or set(qwen) != qwen_keys:
        raise TerminologySpanError("qwen config is not canonical")
    if (
        qwen.get("provider") != "qwen"
        or qwen.get("model") != "Qwen3.8-27B"
        or qwen.get("max_attempts_per_task") != 3
        or qwen.get("temperature") != 0
        or qwen.get("top_p") != 1
    ):
        raise TerminologySpanError("Qwen provider contract differs from the pilot plan")
    _api_origin(str(qwen.get("api_base", "")))
    if not isinstance(qwen.get("api_key_env"), str) or not qwen["api_key_env"]:
        raise TerminologySpanError("qwen.api_key_env must be non-empty")
    for key in ("concurrency", "timeout_seconds", "max_tokens"):
        if isinstance(qwen.get(key), bool) or not isinstance(qwen.get(key), int) or qwen[key] < 1:
            raise TerminologySpanError(f"qwen.{key} must be positive")

    gates = value.get("gates")
    gate_keys = {
        "max_terminal_failure_rate",
        "min_exact_recall",
        "min_exact_recall_ci_lower",
        "max_auto_error_ci_upper",
        "min_auto_coverage",
        "min_repeat_jaccard",
        "min_repeat_needs_explanation_agreement",
        "confidence_level",
    }
    if not isinstance(gates, Mapping) or set(gates) != gate_keys:
        raise TerminologySpanError("gate config is not canonical")
    frozen_gates = {
        "max_terminal_failure_rate": 0.01,
        "min_exact_recall": 0.95,
        "min_exact_recall_ci_lower": 0.90,
        "max_auto_error_ci_upper": 0.05,
        "min_auto_coverage": 0.60,
        "min_repeat_jaccard": 0.90,
        "min_repeat_needs_explanation_agreement": 0.90,
        "confidence_level": 0.95,
    }
    if dict(gates) != frozen_gates:
        raise TerminologySpanError("gate thresholds differ from the frozen plan")

    ranking = value.get("ranking")
    if not isinstance(ranking, Mapping) or dict(ranking) != {
        "policy": RANKER_POLICY_VERSION,
        "penalty": "l2",
        "C": 1.0,
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": 42,
        "max_candidates": 1000,
    }:
        raise TerminologySpanError("ranking policy differs from the frozen plan")

    audit = value.get("full_audit")
    if not isinstance(audit, Mapping) or dict(audit) != {
        "sample_size": 200,
        "high_score": 50,
        "near_threshold": 50,
        "qwen_only_rare": 50,
        "overlap_variant_mixed": 50,
    }:
        raise TerminologySpanError("full audit policy differs from the frozen plan")
    resolution = value.get("resolution")
    resolution_keys = {
        "qwen_primary",
        "deepseek_reviewer",
        "bge_similarity_is_feature_only",
        "bge_model_path",
        "bge_model_file_tree_sha256",
        "minimum_bge_similarity",
        "max_queries_per_term",
        "max_pages_per_term",
        "trusted_source_registry",
        "minimum_independent_sources",
        "public_suffix_policy",
        "public_suffix_sha1",
        "auto_qc_fraction",
        "live_web_during_formal_inference",
    }
    if not isinstance(resolution, Mapping) or set(resolution) != resolution_keys:
        raise TerminologySpanError("resolution config is not canonical")
    if (
        resolution.get("qwen_primary") is not True
        or resolution.get("deepseek_reviewer") is not True
        or resolution.get("bge_similarity_is_feature_only") is not True
        or resolution.get("live_web_during_formal_inference") is not False
        or not isinstance(resolution.get("bge_model_file_tree_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", resolution["bge_model_file_tree_sha256"]
        )
        is None
        or resolution.get("max_queries_per_term") != 3
        or resolution.get("max_pages_per_term") != 3
        or resolution.get("minimum_independent_sources") != 2
        or resolution.get("public_suffix_policy")
        != "libpsl-builtin-etld-plus-one/v1"
        or not isinstance(resolution.get("public_suffix_sha1"), str)
        or re.fullmatch(r"[0-9a-f]{40}", resolution["public_suffix_sha1"])
        is None
        or resolution.get("auto_qc_fraction") != 0.1
    ):
        raise TerminologySpanError("resolution gates differ from the frozen plan")
    model_path = resolution.get("bge_model_path")
    if (
        not isinstance(model_path, str)
        or not model_path
        or Path(model_path).is_absolute()
        or ".." in Path(model_path).parts
    ):
        raise TerminologySpanError("resolution BGE path must be workspace-relative")
    similarity_floor = resolution.get("minimum_bge_similarity")
    if (
        isinstance(similarity_floor, bool)
        or not isinstance(similarity_floor, (int, float))
        or not 0.0 <= float(similarity_floor) <= 1.0
    ):
        raise TerminologySpanError("minimum BGE similarity is invalid")
    registry = resolution.get("trusted_source_registry")
    if not isinstance(registry, list):
        raise TerminologySpanError("trusted_source_registry must be an array")
    domains: set[str] = set()
    for row in registry:
        if (
            not isinstance(row, Mapping)
            or not {"domain"}.issubset(row)
            or not set(row).issubset({"domain", "name"})
            or not isinstance(row["domain"], str)
            or not re.fullmatch(r"[A-Za-z0-9.-]+", row["domain"])
            or row["domain"].lower() in domains
            or ("name" in row and not isinstance(row["name"], str))
        ):
            raise TerminologySpanError("trusted source registry entry is invalid")
        domains.add(row["domain"].lower())
    return json.loads(json.dumps(value, ensure_ascii=False))


def qwen_provider_config(
    config: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
) -> QwenProviderConfig:
    qwen = config["qwen"]
    environment = os.environ if environ is None else environ
    key_name = str(qwen["api_key_env"])
    return QwenProviderConfig(
        model=str(qwen["model"]),
        api_base=str(qwen["api_base"]),
        api_key=str(environment.get(key_name) or "EMPTY"),
        concurrency=int(qwen["concurrency"]),
        timeout_seconds=int(qwen["timeout_seconds"]),
        max_attempts_per_task=int(qwen["max_attempts_per_task"]),
        temperature=float(qwen["temperature"]),
        top_p=float(qwen["top_p"]),
        max_tokens=int(qwen["max_tokens"]),
    )


def _char_class_count(value: str) -> int:
    classes = set()
    for char in value:
        if _CJK_RE.fullmatch(char):
            classes.add("cjk")
        elif char.isalpha():
            classes.add("latin")
        elif char.isdigit():
            classes.add("digit")
        elif _EMOJI_RE.fullmatch(char):
            classes.add("emoji")
        elif not char.isspace():
            classes.add("symbol")
    return len(classes)


def _is_special_form(surface: str, source: str) -> bool:
    return source in {"hashtag", "emoji", "mixed_token"} or (
        _char_class_count(surface) >= 2
        or (surface.isascii() and any(char.isdigit() for char in surface))
    )


def _iter_occurrences(
    content: str, extraction: Mapping[str, Any]
) -> Iterable[tuple[str, int, int, str]]:
    """Yield raw rule/statistical occurrences without semantic classification."""

    zh_min = int(extraction["zh_min_chars"])
    zh_max = int(extraction["zh_max_chars"])
    for match in _CJK_RE.finditer(content):
        sequence = match.group(0)
        for start_offset in range(len(sequence)):
            for length in range(zh_min, min(zh_max, len(sequence) - start_offset) + 1):
                start = match.start() + start_offset
                end = start + length
                yield content[start:end], start, end, "cjk_ngram"

    latin = list(_LATIN_TOKEN_RE.finditer(content))
    max_tokens = int(extraction["latin_max_tokens"])
    for left in range(len(latin)):
        for size in range(1, max_tokens + 1):
            right = left + size - 1
            if right >= len(latin):
                break
            between = content[latin[left].end() : latin[right].start()] if right > left else ""
            if right > left and (len(between) > 3 or not between.isspace()):
                break
            start, end = latin[left].start(), latin[right].end()
            surface = content[start:end]
            if len(surface) <= int(extraction["max_surface_chars"]):
                yield surface, start, end, "latin_token_ngram"

    for source, pattern in (
        ("hashtag", _HASHTAG_RE),
        ("mixed_token", _MIXED_TOKEN_RE),
        ("emoji", _EMOJI_RE),
    ):
        for match in pattern.finditer(content):
            surface = match.group(0)
            if source == "mixed_token" and _char_class_count(surface) < 2:
                continue
            if 1 <= len(surface) <= int(extraction["max_surface_chars"]):
                yield surface, match.start(), match.end(), source


def _unique_occurrences(
    content: str, extraction: Mapping[str, Any]
) -> list[dict[str, Any]]:
    combined: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for surface, start, end, source in _iter_occurrences(content, extraction):
        if not surface.strip() or content[start:end] != surface:
            continue
        combined[(start, end, surface)].add(source)
    return [
        {
            "surface": surface,
            "start": start,
            "end": end,
            "sources": sorted(sources),
        }
        for (start, end, surface), sources in sorted(combined.items())
    ]


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
        if count > 0
    )


def _piece_frequency(piece: str, counts: Counter[str], chars: Counter[str]) -> int:
    return chars[piece] if len(piece) == 1 else counts[piece]


def _term_pmi(
    term: str,
    *,
    counts: Counter[str],
    chars: Counter[str],
    total_chars: int,
) -> float:
    if len(term) < 2 or counts[term] <= 0 or total_chars <= 0:
        return 0.0
    values: list[float] = []
    for split in range(1, len(term)):
        left, right = term[:split], term[split:]
        left_count = _piece_frequency(left, counts, chars)
        right_count = _piece_frequency(right, counts, chars)
        if left_count and right_count:
            values.append(
                math.log2(
                    (counts[term] * total_chars) / (left_count * right_count)
                )
            )
    return min(values) if values else 0.0


def _rule_score(metrics: Mapping[str, Any], *, special_form: bool) -> float:
    return (
        math.log1p(float(metrics["document_frequency"]))
        + 0.35 * max(0.0, float(metrics["pmi"]))
        + 0.20
        * (float(metrics["left_entropy"]) + float(metrics["right_entropy"]))
        + (1.5 if special_form else 0.0)
    )


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _sampling_pools(
    records: Sequence[Mapping[str, Any]], *, dense_cutoff: int, long_cutoff: int
) -> dict[str, list[Mapping[str, Any]]]:
    pools = {
        "rule_empty": [row for row in records if row["raw_candidate_count"] == 0],
        "candidate_dense": [row for row in records if row["raw_candidate_count"] >= dense_cutoff],
        "mixed_obfuscated": [row for row in records if row["has_mixed_or_special"]],
        "long_content": [row for row in records if row["char_length"] >= long_cutoff],
        "candidate_sparse": [row for row in records if 0 < row["raw_candidate_count"] < dense_cutoff],
        "random_baseline": list(records),
    }
    return pools


def _balanced_quotas(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(PILOT_STRATA))
    return {
        stratum: base + (1 if index < remainder else 0)
        for index, stratum in enumerate(PILOT_STRATA)
    }


def _select_without_replacement(
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    all_records: Sequence[Mapping[str, Any]],
    *,
    total: int,
    seed: str,
    namespace: str,
    excluded_ids: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids = set(excluded_ids)
    quotas = _balanced_quotas(total)
    for stratum in PILOT_STRATA:
        eligible = sorted(
            (
                row
                for row in pools[stratum]
                if str(row["record_id"]) not in selected_ids
            ),
            key=lambda row: _sampling_key(
                seed,
                namespace,
                stratum,
                str(row["record_id"]),
                str(row["content_sha256"]),
            ),
        )
        chosen = list(eligible[: quotas[stratum]])
        for row in chosen:
            copy = dict(row)
            copy["sampling_stratum"] = stratum
            selected.append(copy)
            selected_ids.add(str(row["record_id"]))
    if len(selected) < total:
        fallback = sorted(
            (row for row in all_records if str(row["record_id"]) not in selected_ids),
            key=lambda row: _sampling_key(
                seed,
                namespace,
                "fallback",
                str(row["record_id"]),
                str(row["content_sha256"]),
            ),
        )
        for row in fallback[: total - len(selected)]:
            copy = dict(row)
            copy["sampling_stratum"] = "random_baseline"
            selected.append(copy)
            selected_ids.add(str(row["record_id"]))
    if len(selected) != total:
        raise TerminologySpanError(f"insufficient records for {namespace}: {len(selected)}/{total}")
    return selected


def _task_id(record_id: str, content_sha256: str) -> str:
    return "span-" + hashlib.sha256(
        f"{record_id}\x1f{content_sha256}".encode("utf-8")
    ).hexdigest()[:24]


def _order_review_cases(
    primary: Sequence[dict[str, Any]],
    repeats: Sequence[dict[str, Any]],
    *,
    seed: str,
    minimum_gap: int,
) -> tuple[list[dict[str, Any]], int, int]:
    repeat_ids = {str(row["record_id"]) for row in repeats}
    tasks: list[dict[str, Any]] = []
    for row in primary:
        record_id = str(row["record_id"])
        group = (
            "repeat-" + _sampling_key(seed, "repeat", record_id)[:20]
            if record_id in repeat_ids
            else None
        )
        for occurrence in range(2 if group else 1):
            tasks.append(
                {
                    "record_id": record_id,
                    "task_id": row["task_id"],
                    "phase": row["phase"],
                    "sampling_stratum": row["sampling_stratum"],
                    "repeat_group": group,
                    "occurrence": occurrence,
                }
            )
    for nonce in range(100_000):
        ordered = sorted(
            tasks,
            key=lambda row: _sampling_key(
                seed,
                "review-order",
                str(nonce),
                str(row["record_id"]),
                str(row["occurrence"]),
            ),
        )
        positions: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(ordered):
            if row["repeat_group"]:
                positions[str(row["repeat_group"])].append(index)
        observed = min(
            (abs(pair[1] - pair[0]) for pair in positions.values()),
            default=len(ordered),
        )
        if observed >= minimum_gap:
            for index, row in enumerate(ordered, start=1):
                row["case_id"] = f"TSP-{index:03d}"
                row["blind_alias"] = f"术语-{index:03d}"
            return ordered, nonce, observed
    raise TerminologySpanError("could not separate hidden repeats in review order")


def build_census_documents(
    frozen: FrozenTrainInput, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Build A0 corpus statistics and a sealed 200/100 pilot frame in memory."""

    if len(frozen.records) != int(config["expected_fit_count"]):
        raise TerminologySpanError(
            f"fit count differs: {len(frozen.records)} != {config['expected_fit_count']}"
        )
    extraction = config["extraction"]
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    char_counts: Counter[str] = Counter()
    left_boundaries: dict[str, Counter[str]] = defaultdict(Counter)
    right_boundaries: dict[str, Counter[str]] = defaultdict(Counter)
    term_sources: dict[str, set[str]] = defaultdict(set)
    total_chars = 0

    for source in frozen.records:
        record_id = str(source.get("id", ""))
        content_value = source.get("content")
        if not record_id or not isinstance(content_value, str):
            raise TerminologySpanError("fit record lacks id/content")
        content = _normalize_content(content_value)
        maximum = int(extraction["max_text_chars_per_record"])
        if len(content) > maximum:
            raise TerminologySpanError(
                f"fit content exceeds the frozen full-text limit: {record_id} "
                f"({len(content)}>{maximum}); increase and refreeze the config "
                "instead of truncating the record"
            )
        content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        raw = _unique_occurrences(content, extraction)
        seen_terms: set[str] = set()
        for occurrence in raw:
            term = str(occurrence["surface"])
            counts[term] += 1
            seen_terms.add(term)
            term_sources[term].update(occurrence["sources"])
            left = content[int(occurrence["start"]) - 1] if occurrence["start"] else "<BOS>"
            right = content[int(occurrence["end"])] if occurrence["end"] < len(content) else "<EOS>"
            left_boundaries[term][left] += 1
            right_boundaries[term][right] += 1
        document_frequency.update(seen_terms)
        char_counts.update(char for char in content if not char.isspace())
        total_chars += sum(1 for char in content if not char.isspace())
        records.append(
            {
                "record_id": record_id,
                "content": content,
                "content_sha256": content_sha,
                "char_length": len(content),
                "raw_candidate_count": len(raw),
                "has_mixed_or_special": any(
                    any(source_name in {"mixed_token", "hashtag", "emoji"} for source_name in occurrence["sources"])
                    for occurrence in raw
                ),
            }
        )

    metrics_by_term: dict[str, dict[str, Any]] = {}
    for term in sorted(counts):
        pmi = _term_pmi(
            term,
            counts=counts,
            chars=char_counts,
            total_chars=total_chars,
        )
        left_entropy = _entropy(left_boundaries[term])
        right_entropy = _entropy(right_boundaries[term])
        sources = sorted(term_sources[term])
        special = any(_is_special_form(term, source) for source in sources)
        metrics = {
            "term": term,
            "lookup_key": unicodedata.normalize("NFKC", term).casefold().strip(),
            "occurrence_count": counts[term],
            "document_frequency": document_frequency[term],
            "pmi": round(pmi, 8),
            "left_entropy": round(left_entropy, 8),
            "right_entropy": round(right_entropy, 8),
            "sources": sources,
            "special_form": special,
        }
        metrics["rule_score"] = round(_rule_score(metrics, special_form=special), 8)
        metrics_by_term[term] = metrics

    record_rows: list[dict[str, Any]] = []
    for record in records:
        raw = _unique_occurrences(str(record["content"]), extraction)
        proposals: list[dict[str, Any]] = []
        for occurrence in raw:
            metrics = metrics_by_term[str(occurrence["surface"])]
            proposals.append(
                {
                    **occurrence,
                    "lookup_key": metrics["lookup_key"],
                    "document_frequency": metrics["document_frequency"],
                    "pmi": metrics["pmi"],
                    "left_entropy": metrics["left_entropy"],
                    "right_entropy": metrics["right_entropy"],
                    "rule_score": metrics["rule_score"],
                    "special_form": metrics["special_form"],
                }
            )
        proposals.sort(
            key=lambda row: (
                -float(row["rule_score"]),
                int(row["start"]),
                int(row["end"]),
                str(row["surface"]),
            )
        )
        proposals = proposals[: int(extraction["max_rule_spans_per_record"])]
        record_rows.append(
            {
                key: record[key]
                for key in (
                    "record_id",
                    "content_sha256",
                    "char_length",
                    "raw_candidate_count",
                    "has_mixed_or_special",
                )
            }
            | {"rule_proposals": proposals}
        )

    dense_cutoff = max(1, _percentile([row["raw_candidate_count"] for row in records], 0.75))
    long_cutoff = max(1, _percentile([row["char_length"] for row in records], 0.75))
    pools = _sampling_pools(records, dense_cutoff=dense_cutoff, long_cutoff=long_cutoff)
    sampling_pool_counts = {
        name: len({str(row["record_id"]) for row in pools.get(name, [])})
        for name in PILOT_STRATA
    }
    sampling = config["sampling"]
    seed = str(sampling["seed"])
    tune = _select_without_replacement(
        pools,
        records,
        total=int(sampling["tune_count"]),
        seed=seed,
        namespace="tune",
        excluded_ids=set(),
    )
    tune_ids = {str(row["record_id"]) for row in tune}
    validation = _select_without_replacement(
        pools,
        records,
        total=int(sampling["validation_count"]),
        seed=seed,
        namespace="validation",
        excluded_ids=tune_ids,
    )
    by_id = {str(row["record_id"]): row for row in records}
    selected: list[dict[str, Any]] = []
    for phase, rows in (("tune", tune), ("validation", validation)):
        for row in rows:
            selected.append(
                {
                    "record_id": str(row["record_id"]),
                    "content_sha256": str(row["content_sha256"]),
                    "task_id": _task_id(str(row["record_id"]), str(row["content_sha256"])),
                    "phase": phase,
                    "sampling_stratum": str(row["sampling_stratum"]),
                }
            )

    repeat_count = int(sampling["hidden_repeat_count"])
    tune_repeat_count = round(repeat_count * len(tune) / len(selected))
    phase_repeat_quotas = {"tune": tune_repeat_count, "validation": repeat_count - tune_repeat_count}
    repeats: list[dict[str, Any]] = []
    for phase in PILOT_PHASES:
        phase_rows = [row for row in selected if row["phase"] == phase]
        repeats.extend(
            sorted(
                phase_rows,
                key=lambda row: _sampling_key(
                    seed, "repeat-select", phase, str(row["record_id"])
                ),
            )[: phase_repeat_quotas[phase]]
        )
    review_order, order_nonce, observed_gap = _order_review_cases(
        selected,
        repeats,
        seed=seed,
        minimum_gap=int(sampling["minimum_repeat_gap"]),
    )
    unavailable_sampling_strata = sorted(
        name for name, count in sampling_pool_counts.items() if count == 0
    )
    selected_sampling_stratum_counts = dict(
        sorted(Counter(row["sampling_stratum"] for row in selected).items())
    )

    public_tasks = []
    audit_tasks = []
    for row in selected:
        source = by_id[str(row["record_id"])]
        public_tasks.append(
            {
                "task_id": row["task_id"],
                "blind_alias": "Q-" + row["task_id"][-8:],
                "content": source["content"],
            }
        )
        audit_tasks.append(dict(row))
    public_tasks.sort(key=lambda row: str(row["task_id"]))
    audit_tasks.sort(key=lambda row: str(row["task_id"]))

    config_sha = _canonical_sha(config)
    identity = {
        "schema_version": CENSUS_SCHEMA_VERSION,
        "span_implementation_sha256": _span_implementation_sha256(),
        "config_sha256": config_sha,
        "data_build_id": frozen.data_build_id,
        "fit_data_sha256": frozen.train_data_sha256,
        "fit_ids_sha256": frozen.train_ids_sha256,
        "fit_record_count": len(frozen.records),
        "rule_terms_sha256": _canonical_sha(list(metrics_by_term.values())),
        "record_rows_sha256": _canonical_sha(record_rows),
        "pilot_public_sha256": _canonical_sha(public_tasks),
        "pilot_audit_sha256": _canonical_sha(audit_tasks),
        "review_order_sha256": _canonical_sha(review_order),
        "sampling_pool_counts": sampling_pool_counts,
        "unavailable_sampling_strata": unavailable_sampling_strata,
        "selected_sampling_stratum_counts": selected_sampling_stratum_counts,
    }
    census_id = "spancensus-" + _canonical_sha(identity)
    metadata = {
        **identity,
        "census_id": census_id,
        "source_policy": SOURCE_POLICY,
        "resource_role": RESOURCE_ROLE,
        "sampling_policy": SAMPLING_POLICY_VERSION,
        "dense_cutoff": dense_cutoff,
        "long_cutoff": long_cutoff,
        "pilot_counts": {
            "tune": len(tune),
            "validation": len(validation),
            "hidden_repeats": len(repeats),
            "review_cases": len(review_order),
        },
        "review_order_nonce": order_nonce,
        "observed_minimum_repeat_gap": observed_gap,
        "model_visible_fields": ["content"],
        "task_label_contribution_count": 0,
        "dev_test_contribution_count": 0,
    }
    pilot_public = {
        "schema_version": PILOT_FRAME_SCHEMA_VERSION,
        "census_id": census_id,
        "prompt_version": PROMPT_VERSION,
        "tasks": public_tasks,
    }
    pilot_audit = {
        "schema_version": PILOT_FRAME_SCHEMA_VERSION,
        "census_id": census_id,
        "tasks": audit_tasks,
        "review_order": review_order,
    }
    return {
        "metadata": metadata,
        "records": record_rows,
        "rule_terms": list(metrics_by_term.values()),
        "pilot_public": pilot_public,
        "pilot_audit": pilot_audit,
    }


def write_census_artifact(
    documents: Mapping[str, Any], *, output_root: str | Path
) -> Path:
    """Atomically materialize a content-addressed A0 artifact."""

    metadata = documents.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TerminologySpanError("census documents lack metadata")
    census_id = str(metadata.get("census_id", ""))
    if not re.fullmatch(r"spancensus-[0-9a-f]{64}", census_id):
        raise TerminologySpanError("census id is malformed")
    parent = Path(output_root).resolve()
    target = parent / census_id
    if target.exists():
        validate_census_artifact(target)
        return target
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".spancensus-", dir=parent))
    try:
        write_canonical_json(temporary / "census.meta.json", metadata)
        write_canonical_jsonl(
            temporary / "records.jsonl", documents["records"], key="record_id", numeric_key=True
        )
        write_canonical_jsonl(
            temporary / "rule_terms.jsonl", documents["rule_terms"], key="term"
        )
        write_canonical_json(temporary / "pilot.public.json", documents["pilot_public"])
        write_canonical_json(temporary / "pilot.audit.json", documents["pilot_audit"])
        files = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        write_canonical_json(
            temporary / "manifest.json",
            {
                "schema_version": CENSUS_SCHEMA_VERSION,
                "census_id": census_id,
                "files": files,
            },
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_census_artifact(target)
    return target


def validate_census_artifact(path: str | Path) -> dict[str, Any]:
    directory = Path(path)
    manifest = load_json(directory / "manifest.json")
    metadata = load_json(directory / "census.meta.json")
    pilot_public = load_json(directory / "pilot.public.json")
    pilot_audit = load_json(directory / "pilot.audit.json")
    records = load_jsonl(directory / "records.jsonl")
    rule_terms = load_jsonl(directory / "rule_terms.jsonl")
    if manifest.get("schema_version") != CENSUS_SCHEMA_VERSION:
        raise TerminologySpanError("census manifest schema is invalid")
    census_id = manifest.get("census_id")
    if metadata.get("census_id") != census_id or directory.name != census_id:
        raise TerminologySpanError("census identity differs")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, Mapping) or set(expected_files) != {
        "census.meta.json",
        "pilot.audit.json",
        "pilot.public.json",
        "records.jsonl",
        "rule_terms.jsonl",
    }:
        raise TerminologySpanError("census file set is not canonical")
    for name, digest in expected_files.items():
        if sha256_file(directory / name) != digest:
            raise TerminologySpanError(f"census file hash mismatch: {name}")
    computed_frames = {
        "rule_terms_sha256": _canonical_sha(rule_terms),
        "record_rows_sha256": _canonical_sha(records),
        "pilot_public_sha256": _canonical_sha(pilot_public.get("tasks")),
        "pilot_audit_sha256": _canonical_sha(pilot_audit.get("tasks")),
        "review_order_sha256": _canonical_sha(pilot_audit.get("review_order")),
    }
    if any(metadata.get(key) != digest for key, digest in computed_frames.items()):
        raise TerminologySpanError("census content hashes differ from payloads")
    identity_keys = (
        "schema_version",
        "span_implementation_sha256",
        "config_sha256",
        "data_build_id",
        "fit_data_sha256",
        "fit_ids_sha256",
        "fit_record_count",
        "rule_terms_sha256",
        "record_rows_sha256",
        "pilot_public_sha256",
        "pilot_audit_sha256",
        "review_order_sha256",
        "sampling_pool_counts",
        "unavailable_sampling_strata",
        "selected_sampling_stratum_counts",
    )
    expected_census_id = "spancensus-" + _canonical_sha(
        {key: metadata.get(key) for key in identity_keys}
    )
    if (
        census_id != expected_census_id
        or metadata.get("span_implementation_sha256")
        != _span_implementation_sha256()
    ):
        raise TerminologySpanError("census content-addressed identity is invalid")
    if pilot_public.get("schema_version") != PILOT_FRAME_SCHEMA_VERSION or pilot_audit.get("schema_version") != PILOT_FRAME_SCHEMA_VERSION:
        raise TerminologySpanError("pilot frame schema is invalid")
    if pilot_public.get("census_id") != census_id or pilot_audit.get("census_id") != census_id:
        raise TerminologySpanError("pilot frame census binding differs")
    tasks = pilot_public.get("tasks")
    audit_tasks = pilot_audit.get("tasks")
    if not isinstance(tasks, list) or not isinstance(audit_tasks, list) or len(tasks) != 300 or len(audit_tasks) != 300:
        raise TerminologySpanError("pilot frame must contain 300 unique tasks")
    if any(set(task) != PUBLIC_TASK_KEYS for task in tasks):
        raise TerminologySpanError("pilot public task exposes a non-public field")
    if _forbidden_key_paths(pilot_public):
        raise TerminologySpanError("pilot public frame contains a task field")
    public_by_id = {str(task["task_id"]): task for task in tasks}
    audit_by_id = {str(task["task_id"]): task for task in audit_tasks}
    if len(public_by_id) != 300 or set(public_by_id) != set(audit_by_id):
        raise TerminologySpanError("pilot public/audit task mapping differs")
    if Counter(str(row.get("phase")) for row in audit_tasks) != Counter({"tune": 200, "validation": 100}):
        raise TerminologySpanError("pilot phase counts differ")
    pool_counts = metadata.get("sampling_pool_counts")
    unavailable = metadata.get("unavailable_sampling_strata")
    selected_counts = metadata.get("selected_sampling_stratum_counts")
    observed_selected_counts = dict(
        sorted(Counter(str(row.get("sampling_stratum")) for row in audit_tasks).items())
    )
    if (
        not isinstance(pool_counts, Mapping)
        or set(pool_counts) != set(PILOT_STRATA)
        or any(not isinstance(count, int) or count < 0 for count in pool_counts.values())
        or unavailable
        != sorted(name for name in PILOT_STRATA if pool_counts.get(name) == 0)
        or selected_counts != observed_selected_counts
        or not set(observed_selected_counts).issubset(set(PILOT_STRATA))
    ):
        raise TerminologySpanError("pilot sampling coverage metadata differs")
    for task_id, audit in audit_by_id.items():
        public = public_by_id[task_id]
        if hashlib.sha256(public["content"].encode("utf-8")).hexdigest() != audit.get("content_sha256"):
            raise TerminologySpanError("pilot content hash differs")
    review_order = pilot_audit.get("review_order")
    if not isinstance(review_order, list) or len(review_order) != 360:
        raise TerminologySpanError("pilot review order must contain 300+60 cases")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(review_order):
        group = row.get("repeat_group")
        if group:
            groups[str(group)].append(index)
    if len(groups) != 60 or any(len(pair) != 2 for pair in groups.values()):
        raise TerminologySpanError("pilot hidden-repeat mapping differs")
    if min(abs(pair[1] - pair[0]) for pair in groups.values()) < 12:
        raise TerminologySpanError("pilot hidden repeats are too close")
    if len(records) != metadata.get("fit_record_count"):
        raise TerminologySpanError("census record count differs")
    if any(_forbidden_key_paths(row) for row in records):
        raise TerminologySpanError("census record exposes a task field")
    return {
        "census_id": census_id,
        "metadata": metadata,
        "records": records,
        "rule_terms": rule_terms,
        "pilot_public": pilot_public,
        "pilot_audit": pilot_audit,
        "manifest": manifest,
    }


def build_qwen_messages(content: str) -> list[dict[str, str]]:
    """Render a category-free request from content only."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"original_record": content},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def build_qwen_request(
    task: Mapping[str, Any], provider: QwenProviderConfig
) -> dict[str, Any]:
    if set(task) != PUBLIC_TASK_KEYS:
        raise TerminologySpanError("refusing to render a non-public Qwen task")
    if _forbidden_key_paths(task):
        raise TerminologySpanError("Qwen task contains a task-label field")
    return {
        "model": provider.model,
        "messages": build_qwen_messages(str(task["content"])),
        "temperature": provider.temperature,
        "top_p": provider.top_p,
        "max_tokens": provider.max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        },
    }


def qwen_contract(
    provider: QwenProviderConfig, *, config_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "span_implementation_sha256": _span_implementation_sha256(),
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "config_sha256": config_sha256,
        "provider": provider.public_dict(),
        "model_visible_fields": ["content"],
        "response_format": {"type": "json_object"},
        "max_spans": 8,
        "offset_policy": "surface-plus-occurrence-resolved-by-code/v1",
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```json"):
        value = value[7:]
    elif value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    value = value.strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise SpanProviderError(
                "Qwen response has no JSON object", retryable=True
            )
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SpanProviderError(
                "Qwen response JSON is malformed", retryable=True
            ) from exc
    if not isinstance(parsed, dict):
        raise SpanProviderError("Qwen response is not an object", retryable=True)
    return parsed


def exact_occurrences(content: str, surface: str) -> list[tuple[int, int]]:
    """Return overlapping Python-code-point offsets for one exact surface."""

    if not surface:
        return []
    positions: list[tuple[int, int]] = []
    offset = 0
    while offset <= len(content) - len(surface):
        start = content.find(surface, offset)
        if start < 0:
            break
        positions.append((start, start + len(surface)))
        offset = start + 1
    return positions


def resolve_exact_span(
    content: str, surface: str, occurrence_ordinal: int
) -> tuple[int, int]:
    """Resolve a copied surface and 1-based occurrence to authoritative offsets."""

    if isinstance(occurrence_ordinal, bool) or not isinstance(occurrence_ordinal, int):
        raise TerminologySpanError("occurrence_ordinal must be an integer")
    positions = exact_occurrences(content, surface)
    if occurrence_ordinal < 1 or occurrence_ordinal > len(positions):
        raise TerminologySpanError("surface occurrence cannot be resolved")
    return positions[occurrence_ordinal - 1]


def normalize_qwen_annotation(
    parsed: Mapping[str, Any],
    content: str,
    *,
    max_spans: int = 8,
    max_surface_chars: int = 80,
) -> dict[str, Any]:
    """Strictly normalize Qwen output; no model-provided numeric offset is used."""

    if set(parsed) != {"spans", "record_reason"}:
        raise SpanProviderError(
            "Qwen response keys are not canonical", retryable=True
        )
    raw_spans = parsed.get("spans")
    record_reason = parsed.get("record_reason")
    if not isinstance(raw_spans, list) or not isinstance(record_reason, str):
        raise SpanProviderError("Qwen response types are invalid", retryable=True)
    if len(raw_spans) > max_spans:
        raise SpanProviderError("Qwen returned too many spans", retryable=True)
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for index, row in enumerate(raw_spans):
        if not isinstance(row, Mapping) or set(row) != {
            "surface",
            "occurrence_ordinal",
            "reason",
        }:
            raise SpanProviderError(
                f"Qwen span {index} fields are invalid", retryable=True
            )
        surface = row.get("surface")
        ordinal = row.get("occurrence_ordinal")
        reason = row.get("reason")
        if (
            not isinstance(surface, str)
            or not surface
            or surface != surface.strip()
            or len(surface) > max_surface_chars
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise SpanProviderError(
                f"Qwen span {index} value is invalid", retryable=True
            )
        try:
            start, end = resolve_exact_span(content, surface, ordinal)
        except TerminologySpanError as exc:
            raise SpanProviderError(
                f"Qwen span {index} is not exactly locatable", retryable=True
            ) from exc
        identity = (start, end, surface)
        if identity in seen:
            raise SpanProviderError(
                f"Qwen span {index} is duplicated", retryable=True
            )
        seen.add(identity)
        normalized.append(
            {
                "surface": surface,
                "occurrence_ordinal": ordinal,
                "start": start,
                "end": end,
                "reason": reason.strip(),
            }
        )
    normalized.sort(
        key=lambda row: (row["start"], row["end"], row["surface"])
    )
    return {
        "schema_version": QWEN_ANNOTATION_SCHEMA_VERSION,
        "terminal_status": "success" if normalized else "empty",
        "spans": normalized,
        "record_reason": record_reason.strip(),
    }


class SpanResultStore:
    """Durable SQLite checkpoint bound to one frozen Qwen request contract."""

    def __init__(
        self,
        path: str | Path,
        *,
        contract: Mapping[str, Any],
        fit_data_sha256: str,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.contract = dict(contract)
        self.contract_sha256 = _canonical_sha(contract)
        self.fit_data_sha256 = fit_data_sha256
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        old_umask = os.umask(0o077)
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS attempts (
                        task_id TEXT NOT NULL,
                        attempt_no INTEGER NOT NULL,
                        request_sha256 TEXT NOT NULL,
                        status TEXT NOT NULL,
                        http_status INTEGER,
                        response_json TEXT,
                        error TEXT,
                        PRIMARY KEY(task_id, attempt_no)
                    );
                    CREATE TABLE IF NOT EXISTS results (
                        task_id TEXT PRIMARY KEY,
                        request_sha256 TEXT NOT NULL,
                        attempt_no INTEGER NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        response_model TEXT NOT NULL,
                        normalized_json TEXT NOT NULL,
                        raw_response_json TEXT NOT NULL,
                        usage_json TEXT
                    );
                    CREATE TABLE IF NOT EXISTS exception_resolutions (
                        task_id TEXT PRIMARY KEY,
                        content_sha256 TEXT NOT NULL,
                        normalized_json TEXT NOT NULL,
                        reviewer_id TEXT NOT NULL,
                        source_sha256 TEXT NOT NULL
                    );
                    """
                )
                expected = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "contract": json.dumps(
                        self.contract,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "contract_sha256": self.contract_sha256,
                    "fit_data_sha256": self.fit_data_sha256,
                }
                for key, value in expected.items():
                    row = connection.execute(
                        "SELECT value FROM meta WHERE key=?", (key,)
                    ).fetchone()
                    if row is None:
                        connection.execute(
                            "INSERT INTO meta(key,value) VALUES(?,?)", (key, value)
                        )
                    elif row["value"] != value:
                        raise TerminologySpanError(
                            f"span checkpoint binding changed: {key}"
                        )
            os.chmod(self.path, 0o600)
        finally:
            os.umask(old_umask)

    def has_terminal(self, task_id: str) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                "SELECT 1 FROM results WHERE task_id=?", (task_id,)
            ).fetchone()
            resolution = connection.execute(
                "SELECT 1 FROM exception_resolutions WHERE task_id=?", (task_id,)
            ).fetchone()
        return result is not None or resolution is not None

    def reserve(
        self,
        *,
        task_id: str,
        request_sha256: str,
        max_attempts: int,
    ) -> AttemptReservation | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            completed = connection.execute(
                "SELECT request_sha256 FROM results WHERE task_id=?", (task_id,)
            ).fetchone()
            if completed is not None:
                if completed["request_sha256"] != request_sha256:
                    raise TerminologySpanError("completed Qwen request changed")
                connection.commit()
                return None
            if connection.execute(
                "SELECT 1 FROM exception_resolutions WHERE task_id=?", (task_id,)
            ).fetchone() is not None:
                connection.commit()
                return None
            rows = connection.execute(
                "SELECT attempt_no,request_sha256 FROM attempts WHERE task_id=? ORDER BY attempt_no",
                (task_id,),
            ).fetchall()
            if any(row["request_sha256"] != request_sha256 for row in rows):
                raise TerminologySpanError("Qwen request changed across attempts")
            next_attempt = len(rows) + 1
            if next_attempt > max_attempts:
                connection.rollback()
                raise SpanProviderError(
                    f"attempt budget exhausted for {task_id}", retryable=False
                )
            connection.execute(
                "INSERT INTO attempts(task_id,attempt_no,request_sha256,status) VALUES(?,?,?,'reserved')",
                (task_id, next_attempt, request_sha256),
            )
            connection.commit()
        return AttemptReservation(task_id, next_attempt, request_sha256)

    def finish_failure(
        self,
        reservation: AttemptReservation,
        *,
        error: str,
        retryable: bool,
        http_status: int | None,
        response: Any,
    ) -> None:
        with self._connect() as connection:
            updated = connection.execute(
                "UPDATE attempts SET status=?,http_status=?,response_json=?,error=? "
                "WHERE task_id=? AND attempt_no=? AND status='reserved'",
                (
                    "retryable_failure" if retryable else "terminal_failure",
                    http_status,
                    json.dumps(response, ensure_ascii=False, sort_keys=True)
                    if response is not None
                    else None,
                    error,
                    reservation.task_id,
                    reservation.attempt_no,
                ),
            ).rowcount
            if updated != 1:
                raise TerminologySpanError("Qwen attempt reservation is not active")

    def finish_success(
        self,
        reservation: AttemptReservation,
        *,
        content_sha256: str,
        response_model: str,
        normalized: Mapping[str, Any],
        raw_response: Mapping[str, Any],
        usage: Mapping[str, Any] | None,
        http_status: int,
    ) -> None:
        raw_wire = json.dumps(raw_response, ensure_ascii=False, sort_keys=True)
        normalized_wire = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        usage_wire = (
            json.dumps(dict(usage), ensure_ascii=False, sort_keys=True)
            if isinstance(usage, Mapping)
            else None
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE attempts SET status='success',http_status=?,response_json=?,error=NULL "
                "WHERE task_id=? AND attempt_no=? AND status='reserved'",
                (
                    http_status,
                    raw_wire,
                    reservation.task_id,
                    reservation.attempt_no,
                ),
            ).rowcount
            if updated != 1:
                connection.rollback()
                raise TerminologySpanError("Qwen attempt reservation is not active")
            connection.execute(
                "INSERT INTO results(task_id,request_sha256,attempt_no,content_sha256,response_model,"
                "normalized_json,raw_response_json,usage_json) VALUES(?,?,?,?,?,?,?,?)",
                (
                    reservation.task_id,
                    reservation.request_sha256,
                    reservation.attempt_no,
                    content_sha256,
                    response_model,
                    normalized_wire,
                    raw_wire,
                    usage_wire,
                ),
            )
            connection.commit()

    def terminal_rows(self) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        with self._connect() as connection:
            for row in connection.execute("SELECT * FROM results ORDER BY task_id"):
                normalized = json.loads(row["normalized_json"])
                rows[row["task_id"]] = {
                    "task_id": row["task_id"],
                    "request_sha256": row["request_sha256"],
                    "attempt_no": row["attempt_no"],
                    "content_sha256": row["content_sha256"],
                    "response_model": row["response_model"],
                    "terminal_status": normalized["terminal_status"],
                    "annotation": normalized,
                    "usage": json.loads(row["usage_json"])
                    if row["usage_json"]
                    else None,
                }
            for row in connection.execute(
                "SELECT * FROM exception_resolutions ORDER BY task_id"
            ):
                normalized = json.loads(row["normalized_json"])
                rows[row["task_id"]] = {
                    "task_id": row["task_id"],
                    "request_sha256": None,
                    "attempt_no": None,
                    "content_sha256": row["content_sha256"],
                    "response_model": None,
                    "terminal_status": "exception_resolved",
                    "annotation": normalized,
                    "reviewer_id": row["reviewer_id"],
                    "source_sha256": row["source_sha256"],
                    "usage": None,
                }
        return rows

    def failures(self) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            completed = {
                row["task_id"]
                for row in connection.execute("SELECT task_id FROM results")
            } | {
                row["task_id"]
                for row in connection.execute("SELECT task_id FROM exception_resolutions")
            }
            rows = connection.execute(
                "SELECT task_id,MAX(attempt_no) AS attempts,MAX(error) AS error "
                "FROM attempts GROUP BY task_id ORDER BY task_id"
            ).fetchall()
        return {
            row["task_id"]: {
                "attempts": row["attempts"],
                "error": row["error"],
            }
            for row in rows
            if row["task_id"] not in completed
        }

    def status(self) -> dict[str, Any]:
        terminals = self.terminal_rows()
        failures = self.failures()
        with self._connect() as connection:
            attempts = {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status,COUNT(*) AS count FROM attempts GROUP BY status"
                )
            }
        return {
            "contract_sha256": self.contract_sha256,
            "terminal_count": len(terminals),
            "terminal_status_counts": dict(
                Counter(row["terminal_status"] for row in terminals.values())
            ),
            "failure_count": len(failures),
            "attempt_counts": attempts,
        }

    def add_exception_resolutions(
        self,
        *,
        tasks: Mapping[str, Mapping[str, Any]],
        resolution_document: Mapping[str, Any],
    ) -> int:
        if resolution_document.get("schema_version") != "terminology-span-exception-resolution/v1":
            raise TerminologySpanError("exception-resolution schema is invalid")
        reviewer_id = resolution_document.get("reviewer_id")
        rows = resolution_document.get("resolutions")
        if not isinstance(reviewer_id, str) or not reviewer_id or not isinstance(rows, list):
            raise TerminologySpanError("exception-resolution document is invalid")
        source_sha = _canonical_sha(resolution_document)
        inserted = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for row in rows:
                if not isinstance(row, Mapping) or set(row) != {"task_id", "spans", "notes"}:
                    connection.rollback()
                    raise TerminologySpanError("exception-resolution row is invalid")
                task_id = str(row["task_id"])
                task = tasks.get(task_id)
                if task is None:
                    connection.rollback()
                    raise TerminologySpanError("exception-resolution task is unknown")
                if connection.execute(
                    "SELECT 1 FROM results WHERE task_id=?", (task_id,)
                ).fetchone() is not None:
                    connection.rollback()
                    raise TerminologySpanError(
                        "exception-resolution cannot replace a Qwen terminal result"
                    )
                attempts = connection.execute(
                    "SELECT attempt_no,status FROM attempts WHERE task_id=? ORDER BY attempt_no",
                    (task_id,),
                ).fetchall()
                maximum = int(
                    self.contract.get("provider", {}).get(
                        "max_attempts_per_task", 3
                    )
                )
                if (
                    not attempts
                    or any(attempt["status"] == "success" for attempt in attempts)
                    or (
                        len(attempts) < maximum
                        and attempts[-1]["status"] != "terminal_failure"
                    )
                ):
                    connection.rollback()
                    raise TerminologySpanError(
                        "exception-resolution requires an exhausted or terminal Qwen failure"
                    )
                normalized = normalize_human_spans(
                    str(task["content"]), row["spans"]
                )
                annotation = {
                    "schema_version": QWEN_ANNOTATION_SCHEMA_VERSION,
                    "terminal_status": "exception_resolved",
                    "spans": [
                        {
                            **span,
                            "reason": "人工异常补充",
                        }
                        for span in normalized
                    ],
                    "record_reason": str(row.get("notes") or "人工异常补充"),
                }
                content_sha = hashlib.sha256(
                    str(task["content"]).encode("utf-8")
                ).hexdigest()
                normalized_wire = json.dumps(
                    annotation, ensure_ascii=False, sort_keys=True
                )
                existing = connection.execute(
                    "SELECT content_sha256,normalized_json,reviewer_id,source_sha256 "
                    "FROM exception_resolutions WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["content_sha256"] != content_sha
                        or existing["normalized_json"] != normalized_wire
                        or existing["reviewer_id"] != reviewer_id
                        or existing["source_sha256"] != source_sha
                    ):
                        connection.rollback()
                        raise TerminologySpanError(
                            "existing exception-resolution differs"
                        )
                    continue
                connection.execute(
                    "INSERT INTO exception_resolutions(task_id,content_sha256,normalized_json,reviewer_id,source_sha256) "
                    "VALUES(?,?,?,?,?)",
                    (task_id, content_sha, normalized_wire, reviewer_id, source_sha),
                )
                inserted += 1
            connection.commit()
        return inserted


def _request_qwen_once(
    task: Mapping[str, Any],
    provider: QwenProviderConfig,
    store: SpanResultStore,
    *,
    max_surface_chars: int,
) -> str:
    task_id = str(task["task_id"])
    if store.has_terminal(task_id):
        return "cached"
    request_payload = build_qwen_request(task, provider)
    request_sha = _canonical_sha(request_payload)
    last_error: SpanProviderError | None = None
    while True:
        reservation = store.reserve(
            task_id=task_id,
            request_sha256=request_sha,
            max_attempts=provider.max_attempts_per_task,
        )
        if reservation is None:
            return "cached"
        response_data: Any = None
        try:
            import requests

            response = requests.post(
                _chat_url(provider.api_base),
                json=request_payload,
                headers={
                    "Authorization": f"Bearer {provider.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=provider.timeout_seconds,
                allow_redirects=False,
            )
            try:
                response_data = response.json()
            except ValueError:
                response_data = {"body_excerpt": response.text[:1000]}
            if response.status_code != 200:
                raise SpanProviderError(
                    f"Qwen HTTP {response.status_code}",
                    retryable=response.status_code in RETRYABLE_HTTP_STATUS,
                    http_status=response.status_code,
                )
            if not isinstance(response_data, Mapping):
                raise SpanProviderError("Qwen response is not an object", retryable=True)
            choices = response_data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
                raise SpanProviderError("Qwen response lacks choices", retryable=True)
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            if finish_reason not in {"stop", None}:
                raise SpanProviderError(
                    f"Qwen finish_reason={finish_reason}",
                    retryable=finish_reason in {"length", "insufficient_system_resource"},
                )
            message = choice.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                raise SpanProviderError("Qwen returned empty content", retryable=True)
            parsed = _extract_json_object(content)
            normalized = normalize_qwen_annotation(
                parsed,
                str(task["content"]),
                max_spans=8,
                max_surface_chars=max_surface_chars,
            )
            response_model = response_data.get("model")
            if not isinstance(response_model, str) or not response_model:
                raise SpanProviderError("Qwen response lacks model id", retryable=True)
            store.finish_success(
                reservation,
                content_sha256=hashlib.sha256(
                    str(task["content"]).encode("utf-8")
                ).hexdigest(),
                response_model=response_model,
                normalized=normalized,
                raw_response=response_data,
                usage=response_data.get("usage")
                if isinstance(response_data.get("usage"), Mapping)
                else None,
                http_status=response.status_code,
            )
            return normalized["terminal_status"]
        except SpanProviderError as exc:
            last_error = exc
        except Exception as exc:
            last_error = SpanProviderError(
                _sanitize_error(exc), retryable=True, http_status=None
            )
        store.finish_failure(
            reservation,
            error=_sanitize_error(last_error),
            retryable=last_error.retryable,
            http_status=last_error.http_status,
            response=response_data,
        )
        if (
            not last_error.retryable
            or reservation.attempt_no >= provider.max_attempts_per_task
        ):
            raise last_error
        time.sleep(min(8.0, 1.25 * (2 ** (reservation.attempt_no - 1))))


def run_qwen_tasks(
    *,
    tasks: Sequence[Mapping[str, Any]],
    checkpoint_path: str | Path,
    provider: QwenProviderConfig,
    config: Mapping[str, Any],
    fit_data_sha256: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run a resumable set of tune, validation, or full Qwen tasks."""

    if provider.concurrency < 1 or provider.concurrency > 64:
        raise TerminologySpanError("Qwen concurrency must be in [1,64]")
    if provider.max_attempts_per_task != 3:
        raise TerminologySpanError("Qwen max attempts must remain three")
    config_sha = _canonical_sha(config)
    contract = qwen_contract(provider, config_sha256=config_sha)
    store = SpanResultStore(
        checkpoint_path,
        contract=contract,
        fit_data_sha256=fit_data_sha256,
    )
    unique_tasks = {str(task["task_id"]): dict(task) for task in tasks}
    if len(unique_tasks) != len(tasks):
        raise TerminologySpanError("Qwen task ids are not unique")
    if any(set(task) != PUBLIC_TASK_KEYS for task in unique_tasks.values()):
        raise TerminologySpanError("Qwen task frame contains private fields")
    pending = [
        task
        for _, task in sorted(unique_tasks.items())
        if not store.has_terminal(str(task["task_id"]))
    ]
    if limit is not None:
        if limit < 1:
            raise TerminologySpanError("Qwen task limit must be positive")
        pending = pending[:limit]
    failures: dict[str, str] = {}
    completed = 0
    lock = threading.Lock()

    def execute(task: Mapping[str, Any]) -> tuple[str, str]:
        try:
            status = _request_qwen_once(
                task,
                provider,
                store,
                max_surface_chars=int(config["extraction"]["max_surface_chars"]),
            )
            return str(task["task_id"]), status
        except Exception as exc:
            return str(task["task_id"]), "error:" + _sanitize_error(exc)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=provider.concurrency
    ) as executor:
        future_map = {executor.submit(execute, task): task for task in pending}
        for future in concurrent.futures.as_completed(future_map):
            task_id, status = future.result()
            with lock:
                if status.startswith("error:"):
                    failures[task_id] = status[6:]
                else:
                    completed += 1
                observed = completed + len(failures)
                if observed and (observed % 10 == 0 or observed == len(pending)):
                    print(
                        f"[qwen-span] batch={observed}/{len(pending)} "
                        f"terminal={store.status()['terminal_count']} errors={len(failures)}",
                        flush=True,
                    )
    status = store.status()
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "scheduled": len(pending),
        "completed_now": completed,
        "failures": failures,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "status": status,
    }


def pilot_tasks(
    census_dir: str | Path, *, phase: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if phase not in PILOT_PHASES:
        raise TerminologySpanError("pilot phase must be tune or validation")
    census = validate_census_artifact(census_dir)
    phase_ids = {
        str(row["task_id"])
        for row in census["pilot_audit"]["tasks"]
        if row["phase"] == phase
    }
    tasks = [
        dict(row)
        for row in census["pilot_public"]["tasks"]
        if str(row["task_id"]) in phase_ids
    ]
    expected = 200 if phase == "tune" else 100
    if len(tasks) != expected:
        raise TerminologySpanError(f"pilot {phase} task count differs")
    return census, tasks


def validation_extension_tasks(
    *,
    frozen: FrozenTrainInput,
    census_dir: str | Path,
    block: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return one deterministic, previously unused 100-record A2 extension.

    Extension blocks are derived from the frozen census only after an
    ``INCONCLUSIVE`` result.  They never overlap A1, the initial A2 block, or one
    another, and reuse the exact frozen Qwen contract.
    """

    if isinstance(block, bool) or not isinstance(block, int) or block < 1:
        raise TerminologySpanError("validation extension block must be positive")
    census = validate_census_artifact(census_dir)
    all_tasks = build_full_tasks(frozen=frozen, census=census)
    task_by_id = {str(row["task_id"]): row for row in all_tasks}
    base_task_ids = {
        str(row["task_id"]) for row in census["pilot_audit"]["tasks"]
    }
    seed = str(census["metadata"]["sampling_policy"])
    # Include the original configured seed through the census identity so a
    # sampling-policy change cannot preserve extension membership accidentally.
    extension_seed = f"{census['census_id']}:{seed}"
    eligible = sorted(
        (
            row
            for row in census["records"]
            if _task_id(str(row["record_id"]), str(row["content_sha256"]))
            not in base_task_ids
        ),
        key=lambda row: _sampling_key(
            extension_seed,
            "locked-validation-extension",
            str(row["record_id"]),
        ),
    )
    start = (block - 1) * 100
    selected = eligible[start : start + 100]
    if len(selected) != 100:
        raise TerminologySpanError("validation extension block exceeds unused fit records")
    tasks: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row in selected:
        record_id = str(row["record_id"])
        task_id = _task_id(record_id, str(row["content_sha256"]))
        tasks.append(dict(task_by_id[task_id]))
        audit.append(
            {
                "task_id": task_id,
                "record_id": record_id,
                "content_sha256": row["content_sha256"],
                "phase": "validation",
                "sampling_stratum": f"locked_extension_{block:02d}",
                "extension_block": block,
            }
        )
    tasks.sort(key=lambda row: str(row["task_id"]))
    audit.sort(key=lambda row: str(row["task_id"]))
    return census, tasks, audit


def _occurrence_ordinal(content: str, surface: str, start: int, end: int) -> int:
    positions = exact_occurrences(content, surface)
    try:
        return positions.index((start, end)) + 1
    except ValueError as exc:
        raise TerminologySpanError("candidate offsets do not match its surface") from exc


def _proposal_features(
    *,
    surface: str,
    sources: set[str],
    rule: Mapping[str, Any] | None,
) -> dict[str, float]:
    qwen_source = float("qwen" in sources)
    rule_source = float("rule" in sources)
    return {
        "qwen_source": qwen_source,
        "rule_source": rule_source,
        "both_sources": float(qwen_source == 1.0 and rule_source == 1.0),
        "surface_length": float(len(surface)),
        "log_doc_frequency": math.log1p(float(rule.get("document_frequency", 0)))
        if rule
        else 0.0,
        "pmi": float(rule.get("pmi", 0.0)) if rule else 0.0,
        "left_entropy": float(rule.get("left_entropy", 0.0)) if rule else 0.0,
        "right_entropy": float(rule.get("right_entropy", 0.0)) if rule else 0.0,
        "rule_score": float(rule.get("rule_score", 0.0)) if rule else 0.0,
        "special_form": float(bool(rule.get("special_form", False)))
        if rule
        else 0.0,
        "mixed_char_classes": float(_char_class_count(surface) >= 2),
    }


def combine_record_proposals(
    *,
    content: str,
    rule_proposals: Sequence[Mapping[str, Any]],
    qwen_annotation: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Union rule and Qwen proposals while retaining overlaps and provenance."""

    combined: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rule_proposals:
        surface = str(row.get("surface", ""))
        start, end = row.get("start"), row.get("end")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or content[start:end] != surface
        ):
            raise TerminologySpanError("rule proposal offsets are invalid")
        identity = (start, end, surface)
        combined[identity] = {
            "surface": surface,
            "occurrence_ordinal": _occurrence_ordinal(
                content, surface, start, end
            ),
            "start": start,
            "end": end,
            "lookup_key": unicodedata.normalize("NFKC", surface).casefold().strip(),
            "sources": {"rule"},
            "rule": dict(row),
            "qwen_reason": None,
        }
    if qwen_annotation is not None:
        for row in qwen_annotation.get("spans", []):
            surface = str(row.get("surface", ""))
            start, end = row.get("start"), row.get("end")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or content[start:end] != surface
            ):
                raise TerminologySpanError("Qwen proposal offsets are invalid")
            identity = (start, end, surface)
            item = combined.setdefault(
                identity,
                {
                    "surface": surface,
                    "occurrence_ordinal": _occurrence_ordinal(
                        content, surface, start, end
                    ),
                    "start": start,
                    "end": end,
                    "lookup_key": unicodedata.normalize("NFKC", surface)
                    .casefold()
                    .strip(),
                    "sources": set(),
                    "rule": None,
                    "qwen_reason": None,
                },
            )
            item["sources"].add("qwen")
            item["qwen_reason"] = str(row.get("reason") or "")
    result = []
    for item in combined.values():
        sources = set(item.pop("sources"))
        item["sources"] = sorted(sources)
        item["features"] = _proposal_features(
            surface=item["surface"], sources=sources, rule=item["rule"]
        )
        result.append(item)
    result.sort(key=lambda row: (row["start"], row["end"], row["surface"]))
    return result


def normalize_human_spans(
    content: str, spans: Any, *, max_surface_chars: int = 80
) -> list[dict[str, Any]]:
    if not isinstance(spans, list):
        raise TerminologySpanError("human spans must be an array")
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for index, row in enumerate(spans):
        if not isinstance(row, Mapping) or set(row) != {
            "surface",
            "occurrence_ordinal",
        }:
            raise TerminologySpanError(f"human span {index} fields are invalid")
        surface = row.get("surface")
        ordinal = row.get("occurrence_ordinal")
        if (
            not isinstance(surface, str)
            or not surface
            or surface != surface.strip()
            or len(surface) > max_surface_chars
        ):
            raise TerminologySpanError(f"human span {index} surface is invalid")
        start, end = resolve_exact_span(content, surface, ordinal)
        identity = (start, end, surface)
        if identity in seen:
            raise TerminologySpanError(f"human span {index} is duplicated")
        seen.add(identity)
        result.append(
            {
                "surface": surface,
                "occurrence_ordinal": ordinal,
                "start": start,
                "end": end,
            }
        )
    result.sort(key=lambda row: (row["start"], row["end"], row["surface"]))
    return result


def _review_annotation_schema(package_id: str, case_count: int) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Category-free terminology span review",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "package_id", "reviewer_id", "annotations"],
        "properties": {
            "schema_version": {"const": HUMAN_REVIEW_SCHEMA_VERSION},
            "package_id": {"const": package_id},
            "reviewer_id": {"type": "string", "minLength": 1, "maxLength": 100},
            "annotations": {
                "type": "array",
                "minItems": case_count,
                "maxItems": case_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "case_id",
                        "needs_explanation",
                        "spans",
                        "issue_tags",
                        "notes",
                    ],
                    "properties": {
                        "case_id": {"type": "string", "minLength": 1},
                        "needs_explanation": {"type": "boolean"},
                        "spans": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["surface", "occurrence_ordinal"],
                                "properties": {
                                    "surface": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 80,
                                    },
                                    "occurrence_ordinal": {
                                        "type": "integer",
                                        "minimum": 1,
                                    },
                                },
                            },
                        },
                        "issue_tags": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {
                                "enum": [
                                    "missed_span",
                                    "too_wide",
                                    "too_narrow",
                                    "ordinary_phrase",
                                    "sentence_fragment",
                                    "wrong_occurrence",
                                    "other",
                                ]
                            },
                        },
                        "notes": {"type": "string", "maxLength": 2000},
                    },
                },
            },
        },
    }


def _write_deterministic_zip(directory: Path) -> Path:
    archive_path = directory.with_suffix(".zip")
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(directory.name) / path.relative_to(directory)
            info = zipfile.ZipInfo(relative.as_posix())
            info.date_time = (2026, 8, 26, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    os.replace(temporary, archive_path)
    return archive_path


def _copy_review_assets(templates_dir: Path, target: Path) -> None:
    for name in ("index.html", "app.js", "styles.css"):
        source = templates_dir / name
        if not source.is_file():
            raise TerminologySpanError(f"span review asset is missing: {name}")
        shutil.copy2(source, target / name)


def _private_source_map_path(package_directory: Path) -> Path:
    return (
        package_directory.parent
        / ".private"
        / f"{package_directory.name}.source_map.json"
    )


def _write_private_source_map(
    package_directory: Path, source_map: Sequence[Mapping[str, Any]]
) -> None:
    path = _private_source_map_path(package_directory)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists():
        if load_json(path) != list(source_map):
            raise TerminologySpanError("private review source map differs")
    else:
        write_canonical_json(path, list(source_map))
    os.chmod(path, 0o600)


def _open_span_store(
    *,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    fit_data_sha256: str,
    environ: Mapping[str, str] | None = None,
) -> SpanResultStore:
    provider = qwen_provider_config(config, environ=environ)
    return SpanResultStore(
        checkpoint_path,
        contract=qwen_contract(provider, config_sha256=_canonical_sha(config)),
        fit_data_sha256=fit_data_sha256,
    )


def build_pilot_review_package(
    *,
    census_dir: str | Path,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    templates_dir: str | Path,
    output_root: str | Path,
    phase: str = "all",
) -> dict[str, Any]:
    """Build an A1-only tuning review or the frozen 300+60 gate review."""

    census = validate_census_artifact(census_dir)
    if phase not in {"all", "tune"}:
        raise TerminologySpanError("pilot review phase must be all or tune")
    if _canonical_sha(config) != census["metadata"]["config_sha256"]:
        raise TerminologySpanError("pilot config differs from census config")
    store = _open_span_store(
        checkpoint_path=checkpoint_path,
        config=config,
        fit_data_sha256=str(census["metadata"]["fit_data_sha256"]),
    )
    terminals = store.terminal_rows()
    failures = store.failures()
    public_by_id = {
        str(row["task_id"]): row for row in census["pilot_public"]["tasks"]
    }
    record_by_id = {
        str(row["record_id"]): row for row in census["records"]
    }
    audit_by_task = {
        str(row["task_id"]): row for row in census["pilot_audit"]["tasks"]
    }
    review_rows = list(census["pilot_audit"]["review_order"])
    if phase == "tune":
        # A1 must be reviewable before the locked A2 records are ever scanned or
        # exposed.  Hidden repeats are reserved for the final frozen review.
        review_rows = [
            row
            for row in review_rows
            if row.get("phase") == "tune" and row.get("occurrence") == 0
        ]
    expected_review_count = 360 if phase == "all" else 200
    if len(review_rows) != expected_review_count:
        raise TerminologySpanError(
            f"pilot {phase} review selection differs: "
            f"{len(review_rows)}/{expected_review_count}"
        )
    required_task_ids = {str(row["task_id"]) for row in review_rows}
    missing = sorted(required_task_ids - set(terminals) - set(failures))
    if missing:
        raise TerminologySpanError(
            f"pilot Qwen run is incomplete: {len(missing)} task(s) have no terminal attempt"
        )
    cases: list[dict[str, Any]] = []
    source_map: list[dict[str, Any]] = []
    for review in review_rows:
        task_id = str(review["task_id"])
        public = public_by_id[task_id]
        audit = audit_by_task[task_id]
        rule_row = record_by_id[str(audit["record_id"])]
        terminal = terminals.get(task_id)
        annotation = terminal["annotation"] if terminal else None
        proposals = combine_record_proposals(
            content=str(public["content"]),
            rule_proposals=rule_row["rule_proposals"],
            qwen_annotation=annotation,
        )
        case = {
            "case_id": review["case_id"],
            "blind_alias": review["blind_alias"],
            "content": public["content"],
            "proposals": [
                {
                    key: proposal[key]
                    for key in (
                        "surface",
                        "occurrence_ordinal",
                        "start",
                        "end",
                    )
                }
                for proposal in proposals
            ],
            "provider_terminal_status": terminal["terminal_status"]
            if terminal
            else "provider_failure",
        }
        cases.append(case)
        source_map.append(
            {
                "case_id": review["case_id"],
                "task_id": task_id,
                "record_id": audit["record_id"],
                "phase": audit["phase"],
                "sampling_stratum": audit["sampling_stratum"],
                "repeat_group": review.get("repeat_group"),
                "occurrence": review["occurrence"],
            }
        )
    hidden_repeat_count = len(
        {
            str(row["repeat_group"])
            for row in source_map
            if row.get("repeat_group") is not None
        }
    ) if phase == "all" else 0
    package_identity = {
        "schema_version": REVIEW_PACKAGE_SCHEMA_VERSION,
        "mode": "pilot" if phase == "all" else "pilot-tune",
        "census_id": census["census_id"],
        "qwen_contract_sha256": store.contract_sha256,
        "cases_sha256": _canonical_sha(cases),
        "source_map_sha256": _canonical_sha(source_map),
        "case_count": len(cases),
        "unique_record_count": len({str(row["record_id"]) for row in source_map}),
        "hidden_repeat_count": hidden_repeat_count,
    }
    package_id = "termspan-review-" + _canonical_sha(package_identity)[:20]
    manifest = {
        **package_identity,
        "package_id": package_id,
        "offline": True,
        "human_visible_fields": ["content", "unattributed span proposals"],
        "human_hidden_fields": [
            "task annotations",
            "candidate source",
            "fit record id",
            "tune/validation phase",
            "hidden repeat linkage",
        ],
    }
    output_parent = Path(output_root).resolve()
    target = output_parent / package_id
    _write_private_source_map(target, source_map)
    if target.exists():
        existing = _load_review_package(target)
        if existing["manifest"] != manifest:
            raise TerminologySpanError("existing review package identity differs")
        return {
            "package_id": package_id,
            "directory": str(target),
            "archive": str(target.with_suffix(".zip")),
            "case_count": len(cases),
        }
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".termspan-review-", dir=output_parent))
    try:
        _copy_review_assets(Path(templates_dir), temporary)
        write_canonical_json(temporary / "manifest.json", manifest)
        write_canonical_json(temporary / "cases.json", cases)
        write_canonical_json(
            temporary / "annotation_results.schema.json",
            _review_annotation_schema(package_id, len(cases)),
        )
        cases_wire = (
            "window.TERMINOLOGY_SPAN_REVIEW = "
            + json.dumps(
                {"manifest": manifest, "cases": cases},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + ";\n"
        )
        (temporary / "cases.js").write_text(cases_wire, encoding="utf-8")
        (temporary / "README.md").write_text(
            "# 无类别术语 span 试运行人工复审\n\n"
            "直接打开 `index.html`。对每条完整原文穷尽填写需要解释的最小连续 span；"
            "页面中的候选只作提示，必须主动补充遗漏。候选来源、阶段和隐藏复测关系不在"
            "评审包内，由研究协调者单独保管。\n",
            encoding="utf-8",
        )
        checksum_names = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{sha256_file(temporary / name)}  {name}\n"
                for name in checksum_names
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)
        archive = _write_deterministic_zip(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "package_id": package_id,
        "directory": str(target),
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "case_count": len(cases),
    }


def build_validation_extension_review_package(
    *,
    frozen: FrozenTrainInput,
    census_dir: str | Path,
    block: int,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    templates_dir: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build a blind exhaustive review package for one extra locked A2 block."""

    census, tasks, audit_rows = validation_extension_tasks(
        frozen=frozen, census_dir=census_dir, block=block
    )
    if _canonical_sha(config) != census["metadata"]["config_sha256"]:
        raise TerminologySpanError("validation extension config differs from census")
    store = _open_span_store(
        checkpoint_path=checkpoint_path,
        config=config,
        fit_data_sha256=str(census["metadata"]["fit_data_sha256"]),
    )
    terminals = store.terminal_rows()
    failures = store.failures()
    task_by_id = {str(row["task_id"]): row for row in tasks}
    audit_by_id = {str(row["task_id"]): row for row in audit_rows}
    task_ids = set(task_by_id)
    missing = sorted(task_ids - set(terminals) - set(failures))
    if missing:
        raise TerminologySpanError(
            f"validation extension Qwen run is incomplete: {len(missing)} task(s)"
        )
    record_by_id = {str(row["record_id"]): row for row in census["records"]}
    ordered_ids = sorted(
        task_ids,
        key=lambda task_id: _sampling_key(
            census["census_id"],
            "extension-review-order",
            str(block),
            task_id,
        ),
    )
    cases: list[dict[str, Any]] = []
    source_map: list[dict[str, Any]] = []
    for index, task_id in enumerate(ordered_ids, start=1):
        public = task_by_id[task_id]
        audit = audit_by_id[task_id]
        terminal = terminals.get(task_id)
        proposals = combine_record_proposals(
            content=str(public["content"]),
            rule_proposals=record_by_id[str(audit["record_id"])]["rule_proposals"],
            qwen_annotation=terminal["annotation"] if terminal else None,
        )
        case_id = f"TSE-{block:02d}-{index:03d}"
        cases.append(
            {
                "case_id": case_id,
                "blind_alias": f"追加锁定-{index:03d}",
                "content": public["content"],
                "proposals": [
                    {
                        key: proposal[key]
                        for key in (
                            "surface",
                            "occurrence_ordinal",
                            "start",
                            "end",
                        )
                    }
                    for proposal in proposals
                ],
                "provider_terminal_status": terminal["terminal_status"]
                if terminal
                else "provider_failure",
            }
        )
        source_map.append(
            {
                "case_id": case_id,
                "task_id": task_id,
                "record_id": audit["record_id"],
                "phase": "validation",
                "sampling_stratum": audit["sampling_stratum"],
                "repeat_group": None,
                "occurrence": 0,
                "extension_block": block,
            }
        )
    identity = {
        "schema_version": REVIEW_PACKAGE_SCHEMA_VERSION,
        "mode": "pilot-extension",
        "census_id": census["census_id"],
        "qwen_contract_sha256": store.contract_sha256,
        "extension_block": block,
        "cases_sha256": _canonical_sha(cases),
        "source_map_sha256": _canonical_sha(source_map),
        "case_count": 100,
        "unique_record_count": 100,
        "hidden_repeat_count": 0,
    }
    package_id = "termspan-extension-" + _canonical_sha(identity)[:20]
    manifest = {
        **identity,
        "package_id": package_id,
        "offline": True,
        "human_visible_fields": ["content", "unattributed span proposals"],
        "human_hidden_fields": [
            "task annotations",
            "candidate source",
            "fit record id",
            "extension block provenance",
        ],
    }
    parent = Path(output_root).resolve()
    target = parent / package_id
    _write_private_source_map(target, source_map)
    if target.exists():
        existing = _load_review_package(target)
        if existing["manifest"] != manifest:
            raise TerminologySpanError("existing extension review package differs")
        return {
            "package_id": package_id,
            "directory": str(target),
            "archive": str(target.with_suffix(".zip")),
            "case_count": 100,
        }
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".termspan-extension-", dir=parent))
    try:
        _copy_review_assets(Path(templates_dir), temporary)
        write_canonical_json(temporary / "manifest.json", manifest)
        write_canonical_json(temporary / "cases.json", cases)
        write_canonical_json(
            temporary / "annotation_results.schema.json",
            _review_annotation_schema(package_id, 100),
        )
        (temporary / "cases.js").write_text(
            "window.TERMINOLOGY_SPAN_REVIEW = "
            + json.dumps(
                {"manifest": manifest, "cases": cases},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + ";\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# A2 追加锁定 span 复审\n\n"
            "仅在上一轮为 INCONCLUSIVE 时使用。对 100 条完整原文穷尽标注；"
            "候选来源与扩展块身份在盲审期间保持隐藏。\n",
            encoding="utf-8",
        )
        checksum_names = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{sha256_file(temporary / name)}  {name}\n"
                for name in checksum_names
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)
        archive = _write_deterministic_zip(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "package_id": package_id,
        "directory": str(target),
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "case_count": 100,
    }


def _load_review_package(path: str | Path) -> dict[str, Any]:
    directory = Path(path)
    manifest = load_json(directory / "manifest.json")
    cases = load_json(directory / "cases.json")
    source_map = load_json(_private_source_map_path(directory))
    if (
        manifest.get("schema_version") != REVIEW_PACKAGE_SCHEMA_VERSION
        or not isinstance(cases, list)
        or not isinstance(source_map, list)
        or manifest.get("cases_sha256") != _canonical_sha(cases)
        or manifest.get("source_map_sha256") != _canonical_sha(source_map)
    ):
        raise TerminologySpanError("span review package is invalid")
    if len(cases) != manifest.get("case_count") or len(source_map) != len(cases):
        raise TerminologySpanError("span review package counts differ")
    case_ids = [str(row.get("case_id")) for row in cases if isinstance(row, Mapping)]
    source_ids = [
        str(row.get("case_id")) for row in source_map if isinstance(row, Mapping)
    ]
    if (
        len(case_ids) != len(cases)
        or len(source_ids) != len(source_map)
        or len(set(case_ids)) != len(case_ids)
        or len(set(source_ids)) != len(source_ids)
        or set(case_ids) != set(source_ids)
        or any(not isinstance(row.get("content"), str) for row in cases)
    ):
        raise TerminologySpanError("span review package case mapping differs")
    return {"directory": directory, "manifest": manifest, "cases": cases, "source_map": source_map}


def validate_human_review(
    package_dir: str | Path, annotation_path: str | Path
) -> dict[str, Any]:
    package = _load_review_package(package_dir)
    document = load_json(annotation_path)
    if (
        not isinstance(document, Mapping)
        or set(document) != {
            "schema_version",
            "package_id",
            "reviewer_id",
            "annotations",
        }
        or document.get("schema_version") != HUMAN_REVIEW_SCHEMA_VERSION
        or document.get("package_id") != package["manifest"]["package_id"]
        or not isinstance(document.get("reviewer_id"), str)
        or not document["reviewer_id"].strip()
        or not isinstance(document.get("annotations"), list)
    ):
        raise TerminologySpanError("human span review document is invalid")
    case_by_id = {str(case["case_id"]): case for case in package["cases"]}
    normalized: dict[str, dict[str, Any]] = {}
    for row in document["annotations"]:
        if not isinstance(row, Mapping) or set(row) != {
            "case_id",
            "needs_explanation",
            "spans",
            "issue_tags",
            "notes",
        }:
            raise TerminologySpanError("human span review row is invalid")
        case_id = str(row["case_id"])
        case = case_by_id.get(case_id)
        if case is None or case_id in normalized:
            raise TerminologySpanError("human span review case id is unknown or duplicated")
        if not isinstance(row["needs_explanation"], bool):
            raise TerminologySpanError("needs_explanation must be boolean")
        spans = normalize_human_spans(str(case["content"]), row["spans"])
        if row["needs_explanation"] != bool(spans):
            raise TerminologySpanError("needs_explanation must agree with the span set")
        allowed_tags = {
            "missed_span",
            "too_wide",
            "too_narrow",
            "ordinary_phrase",
            "sentence_fragment",
            "wrong_occurrence",
            "other",
        }
        tags = row["issue_tags"]
        notes = row["notes"]
        if (
            not isinstance(tags, list)
            or any(tag not in allowed_tags for tag in tags)
            or len(tags) != len(set(tags))
            or not isinstance(notes, str)
            or len(notes) > 2000
        ):
            raise TerminologySpanError("human review tags/notes are invalid")
        normalized[case_id] = {
            "case_id": case_id,
            "needs_explanation": row["needs_explanation"],
            "spans": spans,
            "issue_tags": list(tags),
            "notes": notes,
        }
    if set(normalized) != set(case_by_id):
        raise TerminologySpanError(
            f"human review is incomplete: {len(normalized)}/{len(case_by_id)}"
        )
    return {
        "schema_version": GOLD_SCHEMA_VERSION,
        "package_id": package["manifest"]["package_id"],
        "reviewer_id": document["reviewer_id"].strip(),
        "annotation_sha256": _canonical_sha(document),
        "annotations": normalized,
        "package": package,
    }


def validate_tune_review_prerequisite(
    *,
    census_dir: str | Path,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    review_package_dir: str | Path,
    annotation_path: str | Path,
) -> dict[str, Any]:
    """Require a complete, current-contract A1 review before any A2 call."""

    census = validate_census_artifact(census_dir)
    if _canonical_sha(config) != census["metadata"]["config_sha256"]:
        raise TerminologySpanError(
            "A2 config differs from census; rerun A0 and A1 before validation"
        )
    reviewed = validate_human_review(review_package_dir, annotation_path)
    manifest = reviewed["package"]["manifest"]
    store = _open_span_store(
        checkpoint_path=checkpoint_path,
        config=config,
        fit_data_sha256=str(census["metadata"]["fit_data_sha256"]),
    )
    source_map = reviewed["package"]["source_map"]
    expected_task_ids = {
        str(row["task_id"])
        for row in census["pilot_audit"]["tasks"]
        if row.get("phase") == "tune"
    }
    observed_task_ids = {str(row.get("task_id")) for row in source_map}
    if (
        manifest.get("mode") != "pilot-tune"
        or manifest.get("census_id") != census["census_id"]
        or manifest.get("qwen_contract_sha256") != store.contract_sha256
        or len(source_map) != 200
        or observed_task_ids != expected_task_ids
        or any(
            row.get("phase") != "tune" or row.get("occurrence") != 0
            for row in source_map
        )
    ):
        raise TerminologySpanError(
            "A2 requires a complete A1 review under the current frozen contract"
        )
    return {
        "package_id": reviewed["package_id"],
        "annotation_sha256": reviewed["annotation_sha256"],
        "qwen_contract_sha256": store.contract_sha256,
    }


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    if math.isinf(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    values = [
        math.lgamma(n + 1)
        - math.lgamma(index + 1)
        - math.lgamma(n - index + 1)
        + index * log_p
        + (n - index) * log_q
        for index in range(k + 1)
    ]
    return min(1.0, math.exp(_logsumexp(values)))


def clopper_pearson_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Exact two-sided binomial interval without a SciPy runtime dependency."""

    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or not isinstance(successes, int)
        or not isinstance(total, int)
        or total < 0
        or successes < 0
        or successes > total
        or not 0.0 < confidence < 1.0
    ):
        raise TerminologySpanError("invalid binomial interval inputs")
    if total == 0:
        return 0.0, 1.0
    alpha_tail = (1.0 - confidence) / 2.0
    if successes == 0:
        lower = 0.0
    else:
        low, high = 0.0, 1.0
        for _ in range(80):
            mid = (low + high) / 2.0
            survival = 1.0 - _binomial_cdf(successes - 1, total, mid)
            if survival < alpha_tail:
                low = mid
            else:
                high = mid
        lower = (low + high) / 2.0
    if successes == total:
        upper = 1.0
    else:
        low, high = 0.0, 1.0
        for _ in range(80):
            mid = (low + high) / 2.0
            cdf = _binomial_cdf(successes, total, mid)
            if cdf > alpha_tail:
                low = mid
            else:
                high = mid
        upper = (low + high) / 2.0
    return lower, upper


def _span_identity(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return int(row["start"]), int(row["end"]), str(row["surface"])


def _build_terminology_gold_document(
    *,
    census: Mapping[str, Any],
    reviewed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize exhaustive human Gold plus machine-to-Gold edits."""

    review_bindings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for gold in reviewed:
        package = gold["package"]
        manifest = package["manifest"]
        review_bindings.append(
            {
                "package_id": gold["package_id"],
                "annotation_sha256": gold["annotation_sha256"],
                "reviewer_id": gold["reviewer_id"],
                "mode": manifest["mode"],
                "extension_block": manifest.get("extension_block"),
            }
        )
        cases = {str(row["case_id"]): row for row in package["cases"]}
        source = {str(row["case_id"]): row for row in package["source_map"]}
        for case_id, annotation in gold["annotations"].items():
            if case_id in seen_case_ids:
                raise TerminologySpanError("Gold case id is duplicated across packages")
            seen_case_ids.add(case_id)
            case = cases[case_id]
            mapping = source[case_id]
            machine = {_span_identity(row): row for row in case["proposals"]}
            truth = {_span_identity(row): row for row in annotation["spans"]}
            additions = [truth[key] for key in sorted(set(truth) - set(machine))]
            deletions = [machine[key] for key in sorted(set(machine) - set(truth))]
            revisions: list[dict[str, Any]] = []
            used_additions: set[tuple[int, int, str]] = set()
            for deleted in deletions:
                deleted_key = _span_identity(deleted)
                candidates = [
                    added
                    for added in additions
                    if _span_identity(added) not in used_additions
                    and int(deleted["start"]) < int(added["end"])
                    and int(added["start"]) < int(deleted["end"])
                ]
                if not candidates:
                    continue
                added = sorted(
                    candidates,
                    key=lambda row: (
                        abs(int(row["start"]) - int(deleted["start"]))
                        + abs(int(row["end"]) - int(deleted["end"])),
                        int(row["start"]),
                        int(row["end"]),
                    ),
                )[0]
                used_additions.add(_span_identity(added))
                revisions.append({"from": deleted, "to": added})
            rows.append(
                {
                    "case_id": case_id,
                    "record_id": mapping["record_id"],
                    "task_id": mapping["task_id"],
                    "phase": mapping["phase"],
                    "extension_block": mapping.get("extension_block"),
                    "hidden_repeat_group": mapping.get("repeat_group"),
                    "hidden_repeat_occurrence": mapping.get("occurrence", 0),
                    "content_sha256": hashlib.sha256(
                        str(case["content"]).encode("utf-8")
                    ).hexdigest(),
                    "needs_explanation": annotation["needs_explanation"],
                    "gold_spans": annotation["spans"],
                    "machine_proposals": case["proposals"],
                    "changes": {
                        "additions": additions,
                        "deletions": deletions,
                        "boundary_revisions": revisions,
                    },
                    "issue_tags": annotation["issue_tags"],
                    "notes": annotation["notes"],
                }
            )
    rows.sort(key=lambda row: str(row["case_id"]))
    document = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "census_id": census["census_id"],
        "source_policy": SOURCE_POLICY,
        "review_bindings": review_bindings,
        "case_count": len(rows),
        "unique_task_count": len({str(row["task_id"]) for row in rows}),
        "hidden_repeat_case_count": sum(
            row["hidden_repeat_group"] is not None for row in rows
        ),
        "cases": rows,
        "task_label_contribution_count": 0,
        "dev_test_contribution_count": 0,
    }
    document["gold_sha256"] = _canonical_sha(document)
    return document


def validate_terminology_gold(path: str | Path) -> dict[str, Any]:
    value = load_json(path)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "census_id",
            "source_policy",
            "review_bindings",
            "case_count",
            "unique_task_count",
            "hidden_repeat_case_count",
            "cases",
            "task_label_contribution_count",
            "dev_test_contribution_count",
            "gold_sha256",
        }
        or value.get("schema_version") != GOLD_SCHEMA_VERSION
        or value.get("source_policy") != SOURCE_POLICY
        or value.get("task_label_contribution_count") != 0
        or value.get("dev_test_contribution_count") != 0
    ):
        raise TerminologySpanError("terminology Gold schema is invalid")
    declared = value.get("gold_sha256")
    unhashed = {key: inner for key, inner in value.items() if key != "gold_sha256"}
    if declared != _canonical_sha(unhashed):
        raise TerminologySpanError("terminology Gold hash is invalid")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != value.get("case_count"):
        raise TerminologySpanError("terminology Gold case count differs")
    if len({str(row.get("case_id")) for row in cases}) != len(cases):
        raise TerminologySpanError("terminology Gold case ids are duplicated")
    if (
        value.get("unique_task_count")
        != len({str(row.get("task_id")) for row in cases})
        or value.get("hidden_repeat_case_count")
        != sum(row.get("hidden_repeat_group") is not None for row in cases)
        or not isinstance(value.get("review_bindings"), list)
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "package_id",
                "annotation_sha256",
                "reviewer_id",
                "mode",
                "extension_block",
            }
            for row in value["review_bindings"]
        )
    ):
        raise TerminologySpanError("terminology Gold counts/bindings differ")
    case_keys = {
        "case_id",
        "record_id",
        "task_id",
        "phase",
        "extension_block",
        "hidden_repeat_group",
        "hidden_repeat_occurrence",
        "content_sha256",
        "needs_explanation",
        "gold_spans",
        "machine_proposals",
        "changes",
        "issue_tags",
        "notes",
    }
    span_keys = {"surface", "occurrence_ordinal", "start", "end"}
    for row in cases:
        if (
            not isinstance(row, Mapping)
            or set(row) != case_keys
            or row.get("phase") not in {"tune", "validation"}
            or not isinstance(row.get("needs_explanation"), bool)
            or not isinstance(row.get("gold_spans"), list)
            or not isinstance(row.get("machine_proposals"), list)
            or any(
                not isinstance(span, Mapping) or set(span) != span_keys
                for span in row["gold_spans"] + row["machine_proposals"]
            )
            or row["needs_explanation"] != bool(row["gold_spans"])
            or not isinstance(row.get("changes"), Mapping)
            or set(row["changes"])
            != {"additions", "deletions", "boundary_revisions"}
            or any(
                not isinstance(row["changes"].get(key), list)
                for key in ("additions", "deletions", "boundary_revisions")
            )
            or any(
                not isinstance(span, Mapping) or set(span) != span_keys
                for span in row["changes"].get("additions", [])
                + row["changes"].get("deletions", [])
            )
        ):
            raise TerminologySpanError("terminology Gold case is invalid")
        truth = {_span_identity(span) for span in row["gold_spans"]}
        machine = {_span_identity(span) for span in row["machine_proposals"]}
        additions = {
            _span_identity(span) for span in row["changes"]["additions"]
        }
        deletions = {
            _span_identity(span) for span in row["changes"]["deletions"]
        }
        if additions != truth - machine or deletions != machine - truth:
            raise TerminologySpanError("terminology Gold changes differ")
        for revision in row["changes"]["boundary_revisions"]:
            if (
                not isinstance(revision, Mapping)
                or set(revision) != {"from", "to"}
                or not isinstance(revision.get("from"), Mapping)
                or not isinstance(revision.get("to"), Mapping)
                or set(revision["from"]) != span_keys
                or set(revision["to"]) != span_keys
                or _span_identity(revision["from"]) not in deletions
                or _span_identity(revision["to"]) not in additions
                or int(revision["from"]["start"])
                >= int(revision["to"]["end"])
                or int(revision["to"]["start"])
                >= int(revision["from"]["end"])
            ):
                raise TerminologySpanError(
                    "terminology Gold boundary revision differs"
                )
    if _forbidden_key_paths(value):
        raise TerminologySpanError("terminology Gold contains task-label fields")
    return value


def _jaccard(
    left: set[tuple[int, int, str]], right: set[tuple[int, int, str]]
) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _fit_liblinear_ranker(
    rows: Sequence[Sequence[float]], labels: Sequence[int], config: Mapping[str, Any]
) -> dict[str, Any]:
    if not rows or len(rows) != len(labels) or len(set(labels)) < 2:
        raise TerminologySpanError("ranker training requires both positive and negative rows")
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise TerminologySpanError(
            "scikit-learn is required for the frozen liblinear span ranker"
        ) from exc
    ranking = config["ranking"]
    model = LogisticRegression(
        penalty=str(ranking["penalty"]),
        C=float(ranking["C"]),
        solver=str(ranking["solver"]),
        max_iter=int(ranking["max_iter"]),
        random_state=int(ranking["random_state"]),
    )
    model.fit(rows, labels)
    return {
        "policy": RANKER_POLICY_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "classes": [int(value) for value in model.classes_.tolist()],
        "coefficients": [float(value) for value in model.coef_[0].tolist()],
        "intercept": float(model.intercept_[0]),
        "parameters": {
            "penalty": ranking["penalty"],
            "C": ranking["C"],
            "solver": ranking["solver"],
            "max_iter": ranking["max_iter"],
            "random_state": ranking["random_state"],
        },
    }


def predict_ranker(model: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    if model.get("policy") != RANKER_POLICY_VERSION or model.get("feature_names") != list(FEATURE_NAMES):
        raise TerminologySpanError("span ranker model contract is invalid")
    coefficients = model.get("coefficients")
    intercept = model.get("intercept")
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != len(FEATURE_NAMES)
        or isinstance(intercept, bool)
        or not isinstance(intercept, (int, float))
    ):
        raise TerminologySpanError("span ranker parameters are invalid")
    score = float(intercept) + sum(
        float(coefficient) * float(features[name])
        for coefficient, name in zip(coefficients, FEATURE_NAMES)
    )
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exponential = math.exp(score)
    return exponential / (1.0 + exponential)


def _choose_auto_threshold(
    scored: Sequence[tuple[float, int]], *, maximum_error_rate: float, confidence: float
) -> dict[str, Any] | None:
    """Choose on A1 by point error; the locked A2 interval is the hard gate."""

    if not scored:
        return None
    candidates = sorted({float(score) for score, _ in scored}, reverse=True)
    best: dict[str, Any] | None = None
    for threshold in candidates:
        promoted = [label for score, label in scored if score >= threshold]
        if not promoted:
            continue
        errors = sum(1 for label in promoted if label == 0)
        _, error_upper = clopper_pearson_interval(
            errors, len(promoted), confidence=confidence
        )
        error_rate = errors / len(promoted)
        if error_rate <= maximum_error_rate:
            candidate = {
                "threshold": threshold,
                "promoted": len(promoted),
                "errors": errors,
                "error_rate": error_rate,
                "error_ci_upper": error_upper,
                "selection_constraint": "tune_point_error_only/v1",
            }
            if best is None or candidate["promoted"] > best["promoted"]:
                best = candidate
    return best


def _proposal_rows_for_phase(
    gold: Mapping[str, Any], *, phase: str
) -> tuple[list[dict[str, Any]], list[int], int]:
    package = gold["package"]
    cases = {str(row["case_id"]): row for row in package["cases"]}
    source_map = {str(row["case_id"]): row for row in package["source_map"]}
    proposals: list[dict[str, Any]] = []
    labels: list[int] = []
    gold_total = 0
    for case_id, mapping in source_map.items():
        if mapping.get("phase") != phase or mapping.get("occurrence") != 0:
            continue
        case = cases[case_id]
        annotation = gold["annotations"][case_id]
        truth = {_span_identity(row) for row in annotation["spans"]}
        gold_total += len(truth)
        for proposal in case["proposals"]:
            identity = _span_identity(proposal)
            # Reconstruct only public features later from the private census/Qwen
            # union; review cases intentionally do not expose provenance.
            proposals.append(
                {
                    "case_id": case_id,
                    "task_id": mapping["task_id"],
                    "record_id": mapping["record_id"],
                    "identity": identity,
                }
            )
            labels.append(int(identity in truth))
    return proposals, labels, gold_total


def _private_proposal_index(
    *,
    census: Mapping[str, Any],
    store: SpanResultStore,
    extension_gold: Sequence[Mapping[str, Any]] = (),
) -> dict[tuple[str, tuple[int, int, str]], dict[str, Any]]:
    public_by_id = {
        str(row["task_id"]): row for row in census["pilot_public"]["tasks"]
    }
    audit_by_id = {
        str(row["task_id"]): row for row in census["pilot_audit"]["tasks"]
    }
    records = {str(row["record_id"]): row for row in census["records"]}
    terminals = store.terminal_rows()
    result: dict[tuple[str, tuple[int, int, str]], dict[str, Any]] = {}
    for task_id, public in public_by_id.items():
        audit = audit_by_id[task_id]
        terminal = terminals.get(task_id)
        proposals = combine_record_proposals(
            content=str(public["content"]),
            rule_proposals=records[str(audit["record_id"])]["rule_proposals"],
            qwen_annotation=terminal["annotation"] if terminal else None,
        )
        for proposal in proposals:
            result[(task_id, _span_identity(proposal))] = proposal
    for gold in extension_gold:
        package = gold["package"]
        case_by_id = {str(row["case_id"]): row for row in package["cases"]}
        for mapping in package["source_map"]:
            if mapping.get("occurrence") != 0:
                continue
            task_id = str(mapping["task_id"])
            case = case_by_id[str(mapping["case_id"])]
            terminal = terminals.get(task_id)
            proposals = combine_record_proposals(
                content=str(case["content"]),
                rule_proposals=records[str(mapping["record_id"])]["rule_proposals"],
                qwen_annotation=terminal["annotation"] if terminal else None,
            )
            for proposal in proposals:
                result[(task_id, _span_identity(proposal))] = proposal
    return result


def calibrate_pilot_gate(
    *,
    census_dir: str | Path,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    review_package_dir: str | Path,
    annotation_path: str | Path,
    extension_reviews: Sequence[tuple[str | Path, str | Path]] = (),
    gold_output_path: str | Path | None = None,
    ranker_fit: Callable[[Sequence[Sequence[float]], Sequence[int], Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit on A1 and evaluate all hard gates on locked A2 records."""

    census = validate_census_artifact(census_dir)
    gold = validate_human_review(review_package_dir, annotation_path)
    if gold["package"]["manifest"].get("mode") != "pilot" or gold["package"]["manifest"].get("census_id") != census["census_id"]:
        raise TerminologySpanError("Gold review does not belong to this pilot")
    extension_gold: list[dict[str, Any]] = []
    extension_blocks: set[int] = set()
    for package_path, extension_annotation_path in extension_reviews:
        extra = validate_human_review(package_path, extension_annotation_path)
        manifest = extra["package"]["manifest"]
        block = manifest.get("extension_block")
        if (
            manifest.get("mode") != "pilot-extension"
            or manifest.get("census_id") != census["census_id"]
            or isinstance(block, bool)
            or not isinstance(block, int)
            or block < 1
            or block in extension_blocks
        ):
            raise TerminologySpanError("validation extension Gold is invalid or duplicated")
        extension_blocks.add(block)
        extension_gold.append(extra)
    if extension_blocks and extension_blocks != set(range(1, max(extension_blocks) + 1)):
        raise TerminologySpanError("validation extension blocks must be contiguous from one")
    store = _open_span_store(
        checkpoint_path=checkpoint_path,
        config=config,
        fit_data_sha256=str(census["metadata"]["fit_data_sha256"]),
    )
    if gold["package"]["manifest"].get("qwen_contract_sha256") != store.contract_sha256:
        raise TerminologySpanError("Gold review Qwen contract differs")
    if any(
        extra["package"]["manifest"].get("qwen_contract_sha256")
        != store.contract_sha256
        for extra in extension_gold
    ):
        raise TerminologySpanError("validation extension Qwen contract differs")
    private = _private_proposal_index(
        census=census, store=store, extension_gold=extension_gold
    )
    tune_rows, tune_labels, tune_gold_total = _proposal_rows_for_phase(
        gold, phase="tune"
    )
    validation_rows, validation_labels, validation_gold_total = _proposal_rows_for_phase(
        gold, phase="validation"
    )
    for extra in extension_gold:
        rows, labels, gold_total = _proposal_rows_for_phase(
            extra, phase="validation"
        )
        validation_rows.extend(rows)
        validation_labels.extend(labels)
        validation_gold_total += gold_total
    if (
        not tune_rows
        or not validation_rows
        or validation_gold_total == 0
        or len(set(tune_labels)) < 2
    ):
        status = "INCONCLUSIVE"
        reasons = []
        if not tune_rows or len(set(tune_labels)) < 2:
            reasons.append("insufficient_tune_ranker_classes")
        if not validation_rows or validation_gold_total == 0:
            reasons.append("insufficient_validation_gold_denominator")
        model = None
        threshold_info = None
        validation_metrics = {}
    else:
        x_tune = [
            [
                float(private[(row["task_id"], row["identity"])]["features"][name])
                for name in FEATURE_NAMES
            ]
            for row in tune_rows
        ]
        fit = _fit_liblinear_ranker if ranker_fit is None else ranker_fit
        try:
            model = dict(fit(x_tune, tune_labels, config))
        except TerminologySpanError:
            raise
        except Exception as exc:
            raise TerminologySpanError(f"span ranker fitting failed: {exc}") from exc
        tune_scores = [
            predict_ranker(model, private[(row["task_id"], row["identity"])]["features"])
            for row in tune_rows
        ]
        gates = config["gates"]
        threshold_info = _choose_auto_threshold(
            list(zip(tune_scores, tune_labels)),
            maximum_error_rate=float(gates["max_auto_error_ci_upper"]),
            confidence=float(gates["confidence_level"]),
        )
        if threshold_info is None:
            status = "FAIL"
            reasons = ["no_tune_threshold_meets_error_rate"]
            validation_metrics = {}
        else:
            threshold = float(threshold_info["threshold"])
            validation_scores = [
                predict_ranker(
                    model,
                    private[(row["task_id"], row["identity"])]["features"],
                )
                for row in validation_rows
            ]
            proposed_truth = sum(validation_labels)
            recall = proposed_truth / validation_gold_total
            recall_lower, recall_upper = clopper_pearson_interval(
                proposed_truth,
                validation_gold_total,
                confidence=float(gates["confidence_level"]),
            )
            promoted_labels = [
                label
                for score, label in zip(validation_scores, validation_labels)
                if score >= threshold
            ]
            promoted_true = sum(promoted_labels)
            promoted_errors = len(promoted_labels) - promoted_true
            _, error_upper = clopper_pearson_interval(
                promoted_errors,
                len(promoted_labels),
                confidence=float(gates["confidence_level"]),
            )
            coverage = promoted_true / validation_gold_total
            validation_metrics = {
                "gold_span_count": validation_gold_total,
                "proposal_count": len(validation_rows),
                "proposal_true_count": proposed_truth,
                "exact_recall": recall,
                "exact_recall_ci": [recall_lower, recall_upper],
                "auto_promoted_count": len(promoted_labels),
                "auto_promoted_true_count": promoted_true,
                "auto_error_count": promoted_errors,
                "auto_error_rate": promoted_errors / len(promoted_labels)
                if promoted_labels
                else None,
                "auto_error_ci_upper": error_upper,
                "auto_coverage": coverage,
            }
            hard_reasons: list[str] = []
            interval_reasons: list[str] = []
            if recall < float(gates["min_exact_recall"]):
                hard_reasons.append("exact_recall_below_threshold")
            elif recall_lower < float(gates["min_exact_recall_ci_lower"]):
                interval_reasons.append("insufficient_recall_denominator")
            auto_error_rate = (
                promoted_errors / len(promoted_labels)
                if promoted_labels
                else 1.0
            )
            if auto_error_rate > float(gates["max_auto_error_ci_upper"]):
                hard_reasons.append("auto_error_rate_above_threshold")
            elif error_upper > float(gates["max_auto_error_ci_upper"]):
                interval_reasons.append("insufficient_auto_error_denominator")
            if coverage < float(gates["min_auto_coverage"]):
                hard_reasons.append("auto_coverage_below_threshold")
            reasons = hard_reasons + interval_reasons
            status = (
                "FAIL"
                if hard_reasons
                else "INCONCLUSIVE"
                if interval_reasons
                else "PASS"
            )

    source_by_case = {
        str(row["case_id"]): row for row in gold["package"]["source_map"]
    }
    repeat_groups: dict[str, list[str]] = defaultdict(list)
    for case_id, mapping in source_by_case.items():
        if mapping.get("repeat_group"):
            repeat_groups[str(mapping["repeat_group"])].append(case_id)
    repeat_jaccards: list[float] = []
    repeat_needs: list[bool] = []
    repeat_rows: list[dict[str, Any]] = []
    for group, case_ids in sorted(repeat_groups.items()):
        if len(case_ids) != 2:
            raise TerminologySpanError("Gold hidden-repeat group is malformed")
        left, right = (gold["annotations"][case_id] for case_id in case_ids)
        left_set = {_span_identity(row) for row in left["spans"]}
        right_set = {_span_identity(row) for row in right["spans"]}
        jaccard = _jaccard(left_set, right_set)
        needs_match = left["needs_explanation"] == right["needs_explanation"]
        repeat_jaccards.append(jaccard)
        repeat_needs.append(needs_match)
        repeat_rows.append(
            {
                "repeat_group": group,
                "span_jaccard": jaccard,
                "needs_explanation_match": needs_match,
            }
        )
    repeat_metrics = {
        "pair_count": len(repeat_rows),
        "mean_span_jaccard": sum(repeat_jaccards) / len(repeat_jaccards)
        if repeat_jaccards
        else 0.0,
        "needs_explanation_agreement": sum(repeat_needs) / len(repeat_needs)
        if repeat_needs
        else 0.0,
        "pairs": repeat_rows,
    }
    gates = config["gates"]
    if repeat_metrics["mean_span_jaccard"] < float(gates["min_repeat_jaccard"]):
        reasons.append("human_repeat_jaccard_below_threshold")
    if repeat_metrics["needs_explanation_agreement"] < float(
        gates["min_repeat_needs_explanation_agreement"]
    ):
        reasons.append("human_repeat_needs_explanation_below_threshold")

    all_gold = [gold, *extension_gold]
    validation_task_ids = {
        str(mapping["task_id"])
        for reviewed in all_gold
        for mapping in reviewed["package"]["source_map"]
        if mapping.get("phase") == "validation" and mapping.get("occurrence") == 0
    }
    terminal_validation = validation_task_ids.intersection(store.terminal_rows())
    failed_validation = validation_task_ids.intersection(store.failures())
    missing_validation = validation_task_ids - terminal_validation - failed_validation
    failure_rate = (len(failed_validation) + len(missing_validation)) / len(
        validation_task_ids
    )
    provider_metrics = {
        "validation_task_count": len(validation_task_ids),
        "terminal_count": len(terminal_validation),
        "failure_count": len(failed_validation),
        "missing_count": len(missing_validation),
        "terminal_failure_rate": failure_rate,
    }
    if failure_rate > float(gates["max_terminal_failure_rate"]):
        reasons.append("provider_terminal_failure_rate_above_threshold")
    if status == "PASS" and reasons:
        status = "FAIL"
    elif status == "INCONCLUSIVE" and any(
        reason.startswith("human_repeat") or reason.startswith("provider_")
        for reason in reasons
    ):
        status = "FAIL"

    review_bindings = [
        {
            "package_id": reviewed["package_id"],
            "annotation_sha256": reviewed["annotation_sha256"],
            "mode": reviewed["package"]["manifest"]["mode"],
            "extension_block": reviewed["package"]["manifest"].get(
                "extension_block"
            ),
        }
        for reviewed in all_gold
    ]
    terminology_gold = _build_terminology_gold_document(
        census=census, reviewed=all_gold
    )
    if gold_output_path is not None:
        write_canonical_json(gold_output_path, terminology_gold)
        validate_terminology_gold(gold_output_path)
    gate = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate_kind": "span-pilot",
        "status": status,
        "reasons": sorted(set(reasons)),
        "census_id": census["census_id"],
        "config_sha256": _canonical_sha(config),
        "qwen_contract_sha256": store.contract_sha256,
        "review_package_id": gold["package_id"],
        "annotation_sha256": gold["annotation_sha256"],
        "review_bindings": review_bindings,
        "review_bindings_sha256": _canonical_sha(review_bindings),
        "terminology_gold_sha256": terminology_gold["gold_sha256"],
        "validation_extension_blocks": sorted(extension_blocks),
        "ranker": model,
        "auto_threshold": threshold_info,
        "tune": {
            "gold_span_count": tune_gold_total,
            "proposal_count": len(tune_rows),
            "positive_proposal_count": sum(tune_labels),
        },
        "validation": validation_metrics,
        "provider": provider_metrics,
        "human_repeat": repeat_metrics,
        "thresholds": dict(config["gates"]),
    }
    gate["gate_sha256"] = _canonical_sha(gate)
    return gate


def write_pilot_decision(
    *,
    gate: Mapping[str, Any],
    decision: str,
    reviewer_id: str,
    notes: str,
    output_path: str | Path,
) -> dict[str, Any]:
    normalized_decision = decision.upper()
    if normalized_decision not in {"PASS", "FAIL"}:
        raise TerminologySpanError("pilot decision must be PASS or FAIL")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise TerminologySpanError("pilot decision requires reviewer_id")
    if not isinstance(notes, str) or len(notes) > 4000:
        raise TerminologySpanError("pilot decision notes are invalid")
    if normalized_decision == "PASS" and gate.get("status") != "PASS":
        raise TerminologySpanError(
            f"cannot approve pilot whose statistical gate is {gate.get('status')}"
        )
    document = {
        "schema_version": PILOT_DECISION_SCHEMA_VERSION,
        "decision": normalized_decision,
        "reviewer_id": reviewer_id.strip(),
        "notes": notes,
        "gate": dict(gate),
        "binding": {
            key: gate.get(key)
            for key in (
                "census_id",
                "config_sha256",
                "qwen_contract_sha256",
                "review_package_id",
                "annotation_sha256",
                "review_bindings_sha256",
                "terminology_gold_sha256",
                "gate_sha256",
            )
        },
    }
    document["decision_sha256"] = _canonical_sha(document)
    write_canonical_json(output_path, document)
    return document


def validate_inconclusive_pilot_gate(
    path: str | Path,
    *,
    census: Mapping[str, Any],
    config: Mapping[str, Any],
    qwen_contract_sha256: str,
) -> dict[str, Any]:
    """Validate that another locked A2 block is the permitted next action."""

    gate = load_json(path)
    if (
        not isinstance(gate, dict)
        or gate.get("schema_version") != GATE_SCHEMA_VERSION
        or gate.get("gate_kind") != "span-pilot"
        or gate.get("status") != "INCONCLUSIVE"
        or gate.get("gate_sha256")
        != _canonical_sha(
            {key: value for key, value in gate.items() if key != "gate_sha256"}
        )
        or gate.get("census_id") != census.get("census_id")
        or gate.get("config_sha256") != _canonical_sha(config)
        or gate.get("qwen_contract_sha256") != qwen_contract_sha256
    ):
        raise TerminologySpanError(
            "validation extension requires a current INCONCLUSIVE pilot gate"
        )
    reasons = gate.get("reasons")
    extension_resolvable_reasons = {
        "insufficient_validation_gold_denominator",
        "insufficient_recall_denominator",
        "insufficient_auto_error_denominator",
    }
    if (
        not isinstance(reasons, list)
        or not reasons
        or not set(reasons).issubset(extension_resolvable_reasons)
    ):
        raise TerminologySpanError(
            "validation extension is allowed only for a locked A2 denominator"
        )
    blocks = gate.get("validation_extension_blocks")
    if (
        not isinstance(blocks, list)
        or any(
            isinstance(block, bool) or not isinstance(block, int) or block < 1
            for block in blocks
        )
        or blocks != list(range(1, len(blocks) + 1))
    ):
        raise TerminologySpanError("pilot extension history is invalid")
    return gate


def validate_pilot_decision(
    path: str | Path,
    *,
    census: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    qwen_contract_sha256: str | None = None,
    require_pass: bool = True,
) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != PILOT_DECISION_SCHEMA_VERSION:
        raise TerminologySpanError("pilot decision schema is invalid")
    declared = value.get("decision_sha256")
    unhashed = {key: inner for key, inner in value.items() if key != "decision_sha256"}
    if declared != _canonical_sha(unhashed):
        raise TerminologySpanError("pilot decision hash is invalid")
    gate = value.get("gate")
    if not isinstance(gate, Mapping) or gate.get("gate_sha256") != _canonical_sha(
        {key: inner for key, inner in gate.items() if key != "gate_sha256"}
    ):
        raise TerminologySpanError("pilot gate hash is invalid")
    if (
        gate.get("schema_version") != GATE_SCHEMA_VERSION
        or gate.get("gate_kind") != "span-pilot"
        or gate.get("status") not in {"PASS", "FAIL", "INCONCLUSIVE"}
    ):
        raise TerminologySpanError("pilot gate contract is invalid")
    if require_pass and (value.get("decision") != "PASS" or gate.get("status") != "PASS"):
        raise TerminologySpanError("full scan requires a PASS pilot decision")
    binding = value.get("binding")
    if not isinstance(binding, Mapping):
        raise TerminologySpanError("pilot decision binding is missing")
    binding_keys = {
        "census_id",
        "config_sha256",
        "qwen_contract_sha256",
        "review_package_id",
        "annotation_sha256",
        "review_bindings_sha256",
        "terminology_gold_sha256",
        "gate_sha256",
    }
    if set(binding) != binding_keys or any(
        binding.get(key) != gate.get(key) for key in binding_keys
    ):
        raise TerminologySpanError("pilot decision binding differs from its gate")
    if value.get("decision") == "PASS":
        ranker = gate.get("ranker")
        threshold = gate.get("auto_threshold")
        if (
            not isinstance(ranker, Mapping)
            or ranker.get("policy") != RANKER_POLICY_VERSION
            or ranker.get("feature_names") != list(FEATURE_NAMES)
            or not isinstance(threshold, Mapping)
            or isinstance(threshold.get("threshold"), bool)
            or not isinstance(threshold.get("threshold"), (int, float))
        ):
            raise TerminologySpanError("PASS pilot lacks a frozen ranker/threshold")
    if census is not None and binding.get("census_id") != census.get("census_id"):
        raise TerminologySpanError("pilot approval belongs to another census")
    if config is not None and binding.get("config_sha256") != _canonical_sha(config):
        raise TerminologySpanError("pilot approval was invalidated by config drift")
    if qwen_contract_sha256 is not None and binding.get("qwen_contract_sha256") != qwen_contract_sha256:
        raise TerminologySpanError("pilot approval was invalidated by Qwen contract drift")
    return value


def build_full_tasks(
    *,
    frozen: FrozenTrainInput,
    census: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct all fit-only public tasks and bind them to the A0 census."""

    if (
        len(frozen.records) != census["metadata"]["fit_record_count"]
        or frozen.train_data_sha256 != census["metadata"]["fit_data_sha256"]
        or frozen.train_ids_sha256 != census["metadata"]["fit_ids_sha256"]
    ):
        raise TerminologySpanError("full scan fit input differs from the approved census")
    census_records = {
        str(row["record_id"]): row for row in census["records"]
    }
    tasks: list[dict[str, Any]] = []
    for source in frozen.records:
        record_id = str(source["id"])
        census_row = census_records.get(record_id)
        if census_row is None:
            raise TerminologySpanError(f"full task is absent from census: {record_id}")
        source_content = _normalize_content(str(source["content"]))
        # ``char_length`` is the exact model-visible A0 prefix length.  Reusing it
        # avoids a quadratic prefix-hash search and guarantees byte-for-byte input
        # identity between the pilot and full scan.
        model_char_length = int(census_row["char_length"])
        if model_char_length < 0 or model_char_length > len(source_content):
            raise TerminologySpanError(
                f"census model-visible length is invalid: {record_id}"
            )
        content = source_content[:model_char_length]
        expected_sha = str(census_row["content_sha256"])
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != expected_sha:
            raise TerminologySpanError(
                f"full task content differs from census: {record_id}"
            )
        tasks.append(
            {
                "task_id": _task_id(record_id, expected_sha),
                "blind_alias": "Q-" + _task_id(record_id, expected_sha)[-8:],
                "content": content,
            }
        )
    tasks.sort(key=lambda row: str(row["task_id"]))
    return tasks


def run_full_qwen_scan(
    *,
    frozen: FrozenTrainInput,
    census_dir: str | Path,
    pilot_decision_path: str | Path,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    provider: QwenProviderConfig,
    exception_resolution_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    census = validate_census_artifact(census_dir)
    contract = qwen_contract(provider, config_sha256=_canonical_sha(config))
    contract_sha = _canonical_sha(contract)
    validate_pilot_decision(
        pilot_decision_path,
        census=census,
        config=config,
        qwen_contract_sha256=contract_sha,
        require_pass=True,
    )
    tasks = build_full_tasks(frozen=frozen, census=census)
    store = SpanResultStore(
        checkpoint_path,
        contract=contract,
        fit_data_sha256=str(census["metadata"]["fit_data_sha256"]),
    )
    if exception_resolution_path is not None:
        store.add_exception_resolutions(
            tasks={str(task["task_id"]): task for task in tasks},
            resolution_document=load_json(exception_resolution_path),
        )
    return run_qwen_tasks(
        tasks=tasks,
        checkpoint_path=checkpoint_path,
        provider=provider,
        config=config,
        fit_data_sha256=str(census["metadata"]["fit_data_sha256"]),
        limit=limit,
    )


def _overlap_or_nested(
    proposal: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]]
) -> bool:
    left_start, left_end = int(proposal["start"]), int(proposal["end"])
    for other in proposals:
        if other is proposal:
            continue
        right_start, right_end = int(other["start"]), int(other["end"])
        if left_start < right_end and right_start < left_end:
            return True
    return False


def materialize_full_span_frame(
    *,
    frozen: FrozenTrainInput,
    census_dir: str | Path,
    pilot_decision_path: str | Path,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    output_root: str | Path,
) -> Path:
    """Freeze A4 outputs after all 5,165 records reach a permitted terminal state."""

    census = validate_census_artifact(census_dir)
    store = _open_span_store(
        checkpoint_path=checkpoint_path,
        config=config,
        fit_data_sha256=str(census["metadata"]["fit_data_sha256"]),
    )
    decision = validate_pilot_decision(
        pilot_decision_path,
        census=census,
        config=config,
        qwen_contract_sha256=store.contract_sha256,
        require_pass=True,
    )
    tasks = build_full_tasks(frozen=frozen, census=census)
    terminals = store.terminal_rows()
    task_ids = {str(task["task_id"]) for task in tasks}
    missing = task_ids - set(terminals)
    extra = set(terminals) - task_ids
    if missing or extra:
        raise TerminologySpanError(
            f"full frame requires exact terminal coverage: missing={len(missing)} extra={len(extra)}"
        )
    if any(row["terminal_status"] not in TERMINAL_STATUSES for row in terminals.values()):
        raise TerminologySpanError("full frame contains a forbidden terminal status")
    task_by_id = {str(task["task_id"]): task for task in tasks}
    record_by_id = {
        str(row["record_id"]): row for row in census["records"]
    }
    record_id_by_task = {
        _task_id(str(row["record_id"]), str(row["content_sha256"])): str(row["record_id"])
        for row in census["records"]
    }
    ranker = decision["gate"].get("ranker")
    threshold_info = decision["gate"].get("auto_threshold")
    if not isinstance(ranker, Mapping) or not isinstance(threshold_info, Mapping):
        raise TerminologySpanError("approved pilot lacks a frozen span ranker")
    threshold = float(threshold_info["threshold"])
    record_frames: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, Any]] = {}
    occurrence_serial = 0
    for task_id in sorted(task_ids):
        task = task_by_id[task_id]
        record_id = record_id_by_task[task_id]
        terminal = terminals[task_id]
        proposals = combine_record_proposals(
            content=str(task["content"]),
            rule_proposals=record_by_id[record_id]["rule_proposals"],
            qwen_annotation=terminal["annotation"],
        )
        rendered: list[dict[str, Any]] = []
        for proposal in proposals:
            occurrence_serial += 1
            probability = predict_ranker(ranker, proposal["features"])
            decision_name = "auto_promote" if probability >= threshold else "human_required"
            occurrence = {
                "occurrence_id": f"occ-{occurrence_serial:07d}",
                "record_id": record_id,
                "surface": proposal["surface"],
                "lookup_key": proposal["lookup_key"],
                "occurrence_ordinal": proposal["occurrence_ordinal"],
                "start": proposal["start"],
                "end": proposal["end"],
                "sources": proposal["sources"],
                "model_surface": proposal["surface"]
                if "qwen" in proposal["sources"]
                else None,
                "model_occurrence_ordinal": proposal["occurrence_ordinal"]
                if "qwen" in proposal["sources"]
                else None,
                "authoritative_offset_policy": (
                    "surface-plus-occurrence-resolved-by-code/v1"
                ),
                "features": proposal["features"],
                "score": probability,
                "span_gate_decision": decision_name,
                "overlap_or_nested": _overlap_or_nested(proposal, proposals),
            }
            rendered.append(occurrence)
            term = aggregate.setdefault(
                proposal["lookup_key"],
                {
                    "lookup_key": proposal["lookup_key"],
                    "surfaces": Counter(),
                    "record_ids": set(),
                    "occurrences": [],
                    "max_score": 0.0,
                    "auto_occurrence_count": 0,
                },
            )
            term["surfaces"][proposal["surface"]] += 1
            term["record_ids"].add(record_id)
            term["occurrences"].append(occurrence)
            term["max_score"] = max(float(term["max_score"]), probability)
            if decision_name == "auto_promote":
                term["auto_occurrence_count"] += 1
        record_frames.append(
            {
                "record_id": record_id,
                "task_id": task_id,
                "content": task["content"],
                "content_sha256": hashlib.sha256(
                    str(task["content"]).encode("utf-8")
                ).hexdigest(),
                "qwen_terminal_status": terminal["terminal_status"],
                "qwen_response_model": terminal.get("response_model"),
                "proposals": rendered,
            }
        )

    candidates: list[dict[str, Any]] = []
    for lookup_key, item in aggregate.items():
        preferred_surface, _ = sorted(
            item["surfaces"].items(), key=lambda pair: (-pair[1], pair[0])
        )[0]
        occurrences = sorted(
            item["occurrences"],
            key=lambda row: (-float(row["score"]), row["record_id"], row["start"]),
        )
        candidates.append(
            {
                "candidate_id": "termcand-" + hashlib.sha256(lookup_key.encode("utf-8")).hexdigest()[:20],
                "term": preferred_surface,
                "lookup_key": lookup_key,
                "variants": sorted(item["surfaces"]),
                "document_frequency": len(item["record_ids"]),
                "occurrence_count": len(occurrences),
                "auto_occurrence_count": item["auto_occurrence_count"],
                "max_score": item["max_score"],
                "resolution_status": "ready"
                if item["auto_occurrence_count"] > 0
                else "human_span_required",
                "sample_occurrences": [
                    {
                        key: occurrence[key]
                        for key in (
                            "occurrence_id",
                            "record_id",
                            "surface",
                            "start",
                            "end",
                            "sources",
                            "score",
                            "span_gate_decision",
                        )
                    }
                    for occurrence in occurrences[:5]
                ],
            }
        )
    candidates.sort(
        key=lambda row: (
            row["resolution_status"] != "ready",
            -float(row["max_score"]),
            -int(row["document_frequency"]),
            str(row["term"]),
        )
    )
    candidates = candidates[: int(config["ranking"]["max_candidates"])]
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    identity = {
        "schema_version": SPAN_FRAME_SCHEMA_VERSION,
        "census_id": census["census_id"],
        "pilot_decision_sha256": decision["decision_sha256"],
        "qwen_contract_sha256": store.contract_sha256,
        "config_sha256": _canonical_sha(config),
        "fit_data_sha256": census["metadata"]["fit_data_sha256"],
        "records_sha256": _canonical_sha(record_frames),
        "candidates_sha256": _canonical_sha(candidates),
    }
    frame_id = "termspan-" + _canonical_sha(identity)
    metadata = {
        **identity,
        "frame_id": frame_id,
        "resource_role": RESOURCE_ROLE,
        "source_policy": SOURCE_POLICY,
        "record_count": len(record_frames),
        "terminal_status_counts": dict(
            Counter(row["qwen_terminal_status"] for row in record_frames)
        ),
        "response_model_counts": dict(
            Counter(
                str(row["qwen_response_model"])
                for row in record_frames
                if row["qwen_response_model"] is not None
            )
        ),
        "occurrence_count": occurrence_serial,
        "candidate_count": len(candidates),
        "auto_threshold": threshold,
        "task_label_contribution_count": 0,
        "dev_test_contribution_count": 0,
    }
    parent = Path(output_root).resolve()
    target = parent / frame_id
    if target.exists():
        validate_full_span_frame(target)
        return target
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".termspan-", dir=parent))
    try:
        write_canonical_json(temporary / "span_frame.meta.json", metadata)
        write_canonical_jsonl(
            temporary / "records.jsonl", record_frames, key="record_id", numeric_key=True
        )
        write_canonical_jsonl(
            temporary / "candidates.jsonl", candidates, key="rank", numeric_key=True
        )
        # A low-scoring occurrence is not a separate terminology entry when the
        # same normalized term already has an automatically accepted occurrence.
        # The human complement is therefore term-level and is restricted to the
        # frozen top-1,000 candidate set.  Every queued term is later included or
        # excluded by ``merge-human``; nothing is silently dropped.
        human_queue = [
            {
                "queue_id": "spanq-" + str(candidate["candidate_id"])[-20:],
                "candidate_id": candidate["candidate_id"],
                "rank": candidate["rank"],
                "term": candidate["term"],
                "variants": candidate["variants"],
                "sample_occurrences": candidate["sample_occurrences"],
                "reason": "no_occurrence_passed_the_frozen_span_auto_threshold",
            }
            for candidate in candidates
            if candidate["resolution_status"] == "human_span_required"
        ]
        write_canonical_jsonl(
            temporary / "human_span_queue.jsonl", human_queue, key="queue_id"
        )
        files = {
            path.name: sha256_file(path)
            for path in temporary.iterdir()
            if path.is_file()
        }
        write_canonical_json(
            temporary / "manifest.json",
            {
                "schema_version": SPAN_FRAME_SCHEMA_VERSION,
                "frame_id": frame_id,
                "files": files,
            },
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_full_span_frame(target)
    return target


def validate_full_span_frame(path: str | Path) -> dict[str, Any]:
    directory = Path(path)
    manifest = load_json(directory / "manifest.json")
    metadata = load_json(directory / "span_frame.meta.json")
    records = load_jsonl(directory / "records.jsonl")
    candidates = load_jsonl(directory / "candidates.jsonl")
    human_queue = load_jsonl(directory / "human_span_queue.jsonl")
    if manifest.get("schema_version") != SPAN_FRAME_SCHEMA_VERSION:
        raise TerminologySpanError("full span frame schema is invalid")
    frame_id = manifest.get("frame_id")
    if metadata.get("frame_id") != frame_id or directory.name != frame_id:
        raise TerminologySpanError("full span frame identity differs")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, Mapping) or set(expected_files) != {
        "span_frame.meta.json",
        "records.jsonl",
        "candidates.jsonl",
        "human_span_queue.jsonl",
    }:
        raise TerminologySpanError("full span frame file set differs")
    for name, digest in expected_files.items():
        if sha256_file(directory / name) != digest:
            raise TerminologySpanError(f"full span frame hash mismatch: {name}")
    if (
        metadata.get("records_sha256") != _canonical_sha(records)
        or metadata.get("candidates_sha256") != _canonical_sha(candidates)
    ):
        raise TerminologySpanError("full span frame payload hashes differ")
    identity_keys = (
        "schema_version",
        "census_id",
        "pilot_decision_sha256",
        "qwen_contract_sha256",
        "config_sha256",
        "fit_data_sha256",
        "records_sha256",
        "candidates_sha256",
    )
    if frame_id != "termspan-" + _canonical_sha(
        {key: metadata.get(key) for key in identity_keys}
    ):
        raise TerminologySpanError("full span frame content-addressed id is invalid")
    if len(records) != metadata.get("record_count"):
        raise TerminologySpanError("full span frame record count differs")
    if len(candidates) != metadata.get("candidate_count") or len(candidates) > 1000:
        raise TerminologySpanError("full span frame candidate count differs")
    if any(row.get("qwen_terminal_status") not in TERMINAL_STATUSES for row in records):
        raise TerminologySpanError("full span frame has a non-terminal record")
    if metadata.get("terminal_status_counts") != dict(
        Counter(str(row.get("qwen_terminal_status")) for row in records)
    ):
        raise TerminologySpanError("full span terminal-status counts differ")
    observed_model_counts = dict(
        Counter(
            str(row.get("qwen_response_model"))
            for row in records
            if row.get("qwen_response_model") is not None
        )
    )
    if metadata.get("response_model_counts") != observed_model_counts:
        raise TerminologySpanError("full span response-model counts differ")
    for row in records:
        content = row.get("content")
        if not isinstance(content, str) or hashlib.sha256(content.encode("utf-8")).hexdigest() != row.get("content_sha256"):
            raise TerminologySpanError("full span frame content hash differs")
        if (
            row.get("qwen_terminal_status") in {"success", "empty"}
            and (
                not isinstance(row.get("qwen_response_model"), str)
                or not row["qwen_response_model"]
            )
        ) or (
            row.get("qwen_terminal_status") == "exception_resolved"
            and row.get("qwen_response_model") is not None
        ):
            raise TerminologySpanError("full span response-model binding differs")
        for proposal in row.get("proposals", []):
            if content[proposal["start"] : proposal["end"]] != proposal["surface"]:
                raise TerminologySpanError("full span proposal offsets differ")
            qwen_source = "qwen" in proposal.get("sources", [])
            if (
                proposal.get("authoritative_offset_policy")
                != "surface-plus-occurrence-resolved-by-code/v1"
                or (
                    qwen_source
                    and (
                        proposal.get("model_surface") != proposal.get("surface")
                        or proposal.get("model_occurrence_ordinal")
                        != proposal.get("occurrence_ordinal")
                    )
                )
                or (
                    not qwen_source
                    and any(
                        proposal.get(key) is not None
                        for key in (
                            "model_surface",
                            "model_occurrence_ordinal",
                        )
                    )
                )
            ):
                raise TerminologySpanError(
                    "full span proposal model/authority fields differ"
                )
            if proposal.get("span_gate_decision") not in {"auto_promote", "human_required"}:
                raise TerminologySpanError("full span gate decision is invalid")
    candidate_by_id = {str(row.get("candidate_id")): row for row in candidates}
    if len(candidate_by_id) != len(candidates):
        raise TerminologySpanError("full span frame candidate ids are not unique")
    expected_human = {
        candidate_id
        for candidate_id, row in candidate_by_id.items()
        if row.get("resolution_status") == "human_span_required"
    }
    observed_human = {str(row.get("candidate_id")) for row in human_queue}
    if len(observed_human) != len(human_queue) or observed_human != expected_human:
        raise TerminologySpanError("human span queue is not the exact term complement")
    for row in human_queue:
        candidate = candidate_by_id[str(row["candidate_id"])]
        if row.get("term") != candidate.get("term") or row.get("rank") != candidate.get("rank"):
            raise TerminologySpanError("human span queue candidate binding differs")
    if any(_forbidden_key_paths(row) for row in records + candidates + human_queue):
        raise TerminologySpanError("full span artifact contains task-label fields")
    return {
        "frame_id": frame_id,
        "metadata": metadata,
        "records": records,
        "candidates": candidates,
        "human_queue": human_queue,
        "manifest": manifest,
    }


def _select_band(
    pool: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: str,
    band: str,
    selected: set[str],
    key: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    eligible = [row for row in pool if row["occurrence_id"] not in selected]
    ordered = sorted(
        eligible,
        key=lambda row: (
            key(row),
            _sampling_key(seed, "full-audit", band, row["occurrence_id"]),
        ),
    )
    chosen = ordered[:count]
    selected.update(row["occurrence_id"] for row in chosen)
    return chosen


def build_full_audit_package(
    *,
    span_frame_dir: str | Path,
    config: Mapping[str, Any],
    templates_dir: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    frame = validate_full_span_frame(span_frame_dir)
    records = {str(row["record_id"]): row for row in frame["records"]}
    pool = [
        {**proposal, "record_id": row["record_id"], "content": row["content"]}
        for row in frame["records"]
        for proposal in row["proposals"]
        if proposal["span_gate_decision"] == "auto_promote"
    ]
    if len(pool) < int(config["full_audit"]["sample_size"]):
        raise TerminologySpanError("not enough auto-promoted occurrences for full audit")
    threshold = float(frame["metadata"]["auto_threshold"])
    seed = str(config["sampling"]["seed"])
    selected_ids: set[str] = set()
    bands: list[tuple[str, list[dict[str, Any]]]] = []
    # Allocate the constrained strata first so high/near-threshold sampling
    # cannot consume the scarce Qwen-only or complex occurrences.
    rare_pool = [
        row
        for row in pool
        if row["sources"] == ["qwen"]
    ]
    bands.append(
        (
            "qwen_only_rare",
            _select_band(
                rare_pool,
                count=50,
                seed=seed,
                band="qwen_only_rare",
                selected=selected_ids,
                key=lambda row: (
                    float(row["features"]["log_doc_frequency"]),
                    float(row["score"]),
                    row["surface"],
                ),
            ),
        )
    )
    complex_pool = [
        row
        for row in pool
        if row["overlap_or_nested"] or bool(row["features"]["mixed_char_classes"])
    ]
    bands.append(
        (
            "overlap_variant_mixed",
            _select_band(
                complex_pool,
                count=50,
                seed=seed,
                band="overlap_variant_mixed",
                selected=selected_ids,
                key=lambda row: (-int(row["overlap_or_nested"]), -float(row["features"]["mixed_char_classes"])),
            ),
        )
    )
    bands.append(
        (
            "high_score",
            _select_band(
                pool,
                count=50,
                seed=seed,
                band="high_score",
                selected=selected_ids,
                key=lambda row: -float(row["score"]),
            ),
        )
    )
    bands.append(
        (
            "near_threshold",
            _select_band(
                pool,
                count=50,
                seed=seed,
                band="near_threshold",
                selected=selected_ids,
                key=lambda row: abs(float(row["score"]) - threshold),
            ),
        )
    )
    incomplete_bands = {
        band: len(rows) for band, rows in bands if len(rows) != 50
    }
    if incomplete_bands:
        raise TerminologySpanError(
            "full audit cannot satisfy the frozen 50x4 strata: "
            + json.dumps(incomplete_bands, sort_keys=True)
        )
    selected_rows = [(band, row) for band, rows in bands for row in rows]
    if len(selected_rows) != 200:
        raise TerminologySpanError("full audit could not fill 200 unique occurrences")
    selected_rows.sort(
        key=lambda pair: _sampling_key(seed, "full-audit-review-order", pair[1]["occurrence_id"])
    )
    cases: list[dict[str, Any]] = []
    source_map: list[dict[str, Any]] = []
    for index, (band, row) in enumerate(selected_rows, start=1):
        record = records[str(row["record_id"])]
        case_id = f"FSA-{index:03d}"
        cases.append(
            {
                "case_id": case_id,
                "blind_alias": f"全量抽检-{index:03d}",
                "content": row["content"],
                "focal_proposal": {
                    key: row[key]
                    for key in (
                        "surface",
                        "occurrence_ordinal",
                        "start",
                        "end",
                    )
                },
                "proposals": [
                    {
                        key: proposal[key]
                        for key in (
                            "surface",
                            "occurrence_ordinal",
                            "start",
                            "end",
                        )
                    }
                    for proposal in record["proposals"]
                ],
                "provider_terminal_status": record["qwen_terminal_status"],
            }
        )
        source_map.append(
            {
                "case_id": case_id,
                "occurrence_id": row["occurrence_id"],
                "record_id": row["record_id"],
                "audit_band": band,
                "focal_identity": [row["start"], row["end"], row["surface"]],
            }
        )
    identity = {
        "schema_version": REVIEW_PACKAGE_SCHEMA_VERSION,
        "mode": "full-audit",
        "span_frame_id": frame["frame_id"],
        "cases_sha256": _canonical_sha(cases),
        "source_map_sha256": _canonical_sha(source_map),
        "case_count": 200,
        "unique_record_count": len({row["record_id"] for row in source_map}),
        "hidden_repeat_count": 0,
    }
    package_id = "termspan-audit-" + _canonical_sha(identity)[:20]
    manifest = {**identity, "package_id": package_id, "offline": True}
    output_parent = Path(output_root).resolve()
    target = output_parent / package_id
    _write_private_source_map(target, source_map)
    if target.exists():
        existing = _load_review_package(target)
        if existing["manifest"] != manifest:
            raise TerminologySpanError("existing full-audit review package differs")
        return {
            "package_id": package_id,
            "directory": str(target),
            "archive": str(target.with_suffix(".zip")),
            "case_count": 200,
        }
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".termspan-audit-", dir=output_parent))
    try:
        _copy_review_assets(Path(templates_dir), temporary)
        write_canonical_json(temporary / "manifest.json", manifest)
        write_canonical_json(temporary / "cases.json", cases)
        write_canonical_json(
            temporary / "annotation_results.schema.json",
            _review_annotation_schema(package_id, 200),
        )
        (temporary / "cases.js").write_text(
            "window.TERMINOLOGY_SPAN_REVIEW = "
            + json.dumps(
                {"manifest": manifest, "cases": cases},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + ";\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# 全量 span 分层抽检\n\n对每条记录穷尽填写正确 span。"
            "页面会突出本次 focal proposal，但它不一定正确。\n",
            encoding="utf-8",
        )
        checksums = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(f"{sha256_file(temporary / name)}  {name}\n" for name in checksums),
            encoding="utf-8",
        )
        os.replace(temporary, target)
        archive = _write_deterministic_zip(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "package_id": package_id,
        "directory": str(target),
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "case_count": 200,
    }


def evaluate_full_audit(
    *,
    review_package_dir: str | Path,
    annotation_path: str | Path,
    config: Mapping[str, Any],
    reviewer_decision: str,
    notes: str,
    output_path: str | Path,
) -> dict[str, Any]:
    gold = validate_human_review(review_package_dir, annotation_path)
    package = gold["package"]
    if package["manifest"].get("mode") != "full-audit":
        raise TerminologySpanError("review package is not a full-audit package")
    source_map = {str(row["case_id"]): row for row in package["source_map"]}
    expected_band_counts = {
        "high_score": int(config["full_audit"]["high_score"]),
        "near_threshold": int(config["full_audit"]["near_threshold"]),
        "qwen_only_rare": int(config["full_audit"]["qwen_only_rare"]),
        "overlap_variant_mixed": int(
            config["full_audit"]["overlap_variant_mixed"]
        ),
    }
    if Counter(str(row.get("audit_band")) for row in source_map.values()) != Counter(
        expected_band_counts
    ):
        raise TerminologySpanError("full audit review strata differ from frozen 50x4")
    errors = 0
    details: list[dict[str, Any]] = []
    for case_id, annotation in gold["annotations"].items():
        focal = tuple(source_map[case_id]["focal_identity"])
        truth = {_span_identity(row) for row in annotation["spans"]}
        correct = focal in truth
        errors += int(not correct)
        details.append(
            {
                "case_id": case_id,
                "audit_band": source_map[case_id]["audit_band"],
                "focal_exact": correct,
            }
        )
    confidence = float(config["gates"]["confidence_level"])
    _, upper = clopper_pearson_interval(errors, len(details), confidence=confidence)
    statistical_status = (
        "PASS"
        if upper <= float(config["gates"]["max_auto_error_ci_upper"])
        else "FAIL"
    )
    decision = reviewer_decision.upper()
    if decision not in {"PASS", "FAIL"}:
        raise TerminologySpanError("full audit reviewer decision must be PASS or FAIL")
    if decision == "PASS" and statistical_status != "PASS":
        raise TerminologySpanError("cannot approve a failed full-span audit")
    document = {
        "schema_version": FULL_AUDIT_DECISION_SCHEMA_VERSION,
        "span_frame_id": package["manifest"]["span_frame_id"],
        "review_package_id": gold["package_id"],
        "annotation_sha256": gold["annotation_sha256"],
        "reviewer_id": gold["reviewer_id"],
        "reviewer_decision": decision,
        "statistical_status": statistical_status,
        "sample_count": len(details),
        "error_count": errors,
        "error_rate": errors / len(details),
        "error_ci_upper": upper,
        "threshold": float(config["gates"]["max_auto_error_ci_upper"]),
        "notes": notes,
        "details": details,
    }
    document["decision_sha256"] = _canonical_sha(document)
    write_canonical_json(output_path, document)
    return document


def validate_full_audit_decision(
    path: str | Path, *, span_frame_id: str, require_pass: bool = True
) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != FULL_AUDIT_DECISION_SCHEMA_VERSION:
        raise TerminologySpanError("full audit decision schema is invalid")
    declared = value.get("decision_sha256")
    unhashed = {key: inner for key, inner in value.items() if key != "decision_sha256"}
    if declared != _canonical_sha(unhashed):
        raise TerminologySpanError("full audit decision hash is invalid")
    if value.get("span_frame_id") != span_frame_id:
        raise TerminologySpanError("full audit belongs to another span frame")
    details = value.get("details")
    if (
        value.get("sample_count") != 200
        or not isinstance(details, list)
        or len(details) != 200
        or len({str(row.get("case_id")) for row in details}) != 200
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"case_id", "audit_band", "focal_exact"}
            or not isinstance(row.get("focal_exact"), bool)
            for row in details
        )
        or Counter(str(row.get("audit_band")) for row in details)
        != Counter(
            {
                "high_score": 50,
                "near_threshold": 50,
                "qwen_only_rare": 50,
                "overlap_variant_mixed": 50,
            }
        )
    ):
        raise TerminologySpanError("full audit decision sample is invalid")
    observed_errors = sum(not row["focal_exact"] for row in details)
    _, expected_upper = clopper_pearson_interval(
        observed_errors, 200, confidence=0.95
    )
    if (
        value.get("error_count") != observed_errors
        or value.get("error_rate") != observed_errors / 200
        or not math.isclose(
            float(value.get("error_ci_upper", -1.0)),
            expected_upper,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or value.get("threshold") != 0.05
        or value.get("statistical_status")
        != ("PASS" if expected_upper <= 0.05 else "FAIL")
    ):
        raise TerminologySpanError("full audit decision statistics are invalid")
    if require_pass and (
        value.get("reviewer_decision") != "PASS"
        or value.get("statistical_status") != "PASS"
    ):
        raise TerminologySpanError("resolution requires a PASS full-span audit")
    return value
