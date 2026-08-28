from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import requests
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.web_search import DisabledWebSearcher, WebSearcher

logger = logging.getLogger(__name__)

_FORMAL_DEEPSEEK_MODEL = "deepseek-v4-flash"
_FORMAL_DEEPSEEK_RETURNED_MODEL_RE = re.compile(
    r"^deepseek-v4-flash(?:-[0-9]{4}|-[0-9]{8})?\Z",
    re.IGNORECASE,
)
_FORMAL_DEEPSEEK_USAGE_FIELDS = (
    "prompt_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "completion_tokens",
    "total_tokens",
)
TERMINOLOGY_LIBRARY_ROLE = "terminology-understanding-library/v1"
LEGACY_DEROGATORY_LEXICON_ROLE = "derogatory-lexicon/v1"
TERMINOLOGY_OBJECTIVE = "terminology-value/v1"
TERMINOLOGY_SOURCE_POLICY = "content-only-no-task-labels/v1"
TERMINOLOGY_RESPONSE_KEYS = {
    "context_judge": frozenset({"supported", "confidence", "reason"}),
    "web_evidence_judge": frozenset(
        {"supported", "confidence", "reason", "evidence_ids"}
    ),
    "final_lexicon_judge": frozenset(
        {
            "include",
            "definition",
            "usage_notes",
            "ambiguity_notes",
            "variants",
            "confidence",
            "reason",
            "evidence_ids",
        }
    ),
}


DATASET_DEFAULT_PATHS = {
    "full": ["data/full/std/train.json", "data/full/std/test.json"],
    "cold": ["data/cold/std/train.json", "data/cold/std/val.json", "data/cold/std/test.json"],
    "hatexplain": [
        "data/hateXplain/std/train.json",
        "data/hateXplain/std/val.json",
        "data/hateXplain/std/test.json",
    ],
}
DATASET_DISPLAY_NAMES = {
    "full": "state/full",
    "cold": "cold",
    "hatexplain": "hatexplain",
}

ZH_CATEGORIES = ("Sexism", "Racism", "Region", "LGBTQ", "others")
DEFAULT_NEUTRAL_IDENTITY_TERMS = (
    "gay",
    "lesbian",
    "bisexual",
    "homosexual",
    "transgender",
    "同性恋",
    "男同性恋",
    "女同性恋",
    "双性恋",
    "跨性别",
    "黑人",
    "白人",
    "男性",
    "女性",
)
HATEXPLAIN_LABELS = {"hatespeech", "offensive"}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_EN_TOKEN_RE = re.compile(r"[#@]?[A-Za-z][A-Za-z0-9_'-]*")
_EN_TOKEN_GAP_RE = re.compile(r"[\s/_-]+\Z")
_SPACE_RE = re.compile(r"\s+")
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

ZH_STOP_TERMS = {
    "一个",
    "一些",
    "一种",
    "一样",
    "一下",
    "这个",
    "那个",
    "他们",
    "她们",
    "我们",
    "你们",
    "就是",
    "不是",
    "没有",
    "什么",
    "真的",
    "还是",
    "觉得",
    "因为",
    "所以",
    "如果",
    "但是",
    "可以",
    "不能",
    "不会",
    "这么",
    "那么",
    "然后",
    "现在",
    "知道",
    "看到",
    "出来",
    "时候",
    "可能",
    "应该",
    "问题",
    "喜欢",
    "的人",
    "的是",
    "不是",
}

ZH_GENERIC_GROUP_TERMS = {
    "中国",
    "中国人",
    "男人",
    "女人",
    "男性",
    "女性",
    "男的",
    "女的",
    "男生",
    "女生",
    "黑人",
    "白人",
    "非洲",
    "非洲人",
    "同性恋",
    "女权",
    "上海",
    "北京",
    "广州",
    "河南",
    "东北",
}

ZH_GENERIC_WORD_TERMS = {
    "喜欢",
    "觉得",
    "知道",
    "看到",
    "出来",
    "应该",
    "可能",
    "问题",
    "中国",
    "世界",
}

ZH_BAD_PREFIXES = {
    "的",
    "了",
    "是",
    "在",
    "就",
    "都",
    "也",
    "和",
    "有",
    "没",
    "不",
    "被",
    "把",
    "给",
    "该",
    "应",
    "真",
    "我",
    "你",
    "他",
    "她",
    "这",
    "那",
    "跟",
    "找",
    "及",
    "其",
    "它",
    "际",
}
ZH_BAD_SUFFIXES = {"的", "了", "着", "过", "吗", "呢", "吧", "啊", "呀", "是", "就", "应", "该", "这", "那", "我", "你", "个"}
ZH_INTERNAL_FRAGMENT_MARKERS = ("的", "及其", "其它", "其他", "它们", "他们", "她们")
# These grammatical particles normally require a following predicate/object.
# They are only used together with continuation evidence below; spelling alone
# must not reject homographic nouns such as ``棉被``.
ZH_DANGLING_SUFFIX_MARKERS = frozenset({"被", "把", "给"})
# Function-word boundaries that commonly appear when a character n-gram cuts
# one character too far around an otherwise complete offensive expression.
# These markers are not rejected on spelling alone: the affix filter below
# also requires a supported base candidate (or a two-character offensive
# stem), which protects ordinary words such as ``存在`` and ``自在``.
ZH_FRAGMENT_PREFIX_MARKERS = frozenset({"些"})
ZH_FRAGMENT_SUFFIX_MARKERS = frozenset({"不", "都", "自", "在"})
ZH_SENTENCE_FINAL_PARTICLES = frozenset({"吗", "呢", "吧", "啊", "呀"})
ZH_VARIANT_CHARS = set("🐒🐵💩🤢🤮4141450")
ZH_OFFENSIVE_HINT_CHARS = set("狗畜贱滚死脏蠢丑臭烂废奴鬼逼婊拳蝻蛆蛮垃")

ZH_SUBSTRING_MIN_SUPPORT = 3
ZH_SUBSTRING_DIRECTIONAL_COVERAGE = 0.8
ZH_SUBSTRING_DOMINANT_PARENT_COVERAGE = 0.8
ZH_DANGLING_SUFFIX_COVERAGE = 0.9

TRACK_PRIORITY = {
    "variant_slang": 6,
    "annotation_anchor": 5,
    "target_context": 4,
    "contrastive_phrase": 3,
    "group_reference": 2,
    "ngram_backoff": 1,
}

TRACK_SCORE_BONUS = {
    "variant_slang": 2.2,
    "annotation_anchor": 1.6,
    "target_context": 1.1,
    "contrastive_phrase": 0.8,
    "group_reference": -1.0,
    "ngram_backoff": -0.3,
}

EN_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "do",
    "eg",
    "e.g",
    "for",
    "from",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "u",
    "we",
    "with",
    "would",
    "you",
    "your",
}


@dataclass
class CandidateMention:
    term: str
    track: str
    phrase_quality: float = 0.0
    variant_signal: float = 0.0


@dataclass
class CandidateStats:
    term: str
    dataset: str
    language: str
    total_count: int = 0
    hate_count: int = 0
    non_hate_count: int = 0
    annotation_count: int = 0
    score: float = 0.0
    source_counts: Counter = field(default_factory=Counter)
    track_counts: Counter = field(default_factory=Counter)
    category_counts: Counter = field(default_factory=Counter)
    support_sample_ids: list[str] = field(default_factory=list)
    sample_contexts: list[dict[str, Any]] = field(default_factory=list)
    hate_precision: float = 0.0
    log_odds: float = 0.0
    category_purity: float = 0.0
    phrase_quality: float = 0.0
    variant_signal: float = 0.0
    nonhate_penalty: float = 0.0
    generic_group_word_penalty: float = 0.0
    bad_substring_penalty: float = 0.0
    substring_of: str = ""
    reject_hint: str = ""

    def update(
        self,
        record_id: str,
        content: str,
        is_hate: bool,
        categories: list[str],
        source: str,
        max_samples: int,
        track: str = "ngram_backoff",
        phrase_quality: float = 0.0,
        variant_signal: float = 0.0,
        include_task_labels: bool = True,
    ) -> None:
        self.source_counts[source] += 1
        self.track_counts[track] += 1
        self.phrase_quality = max(self.phrase_quality, phrase_quality)
        self.variant_signal = max(self.variant_signal, variant_signal)
        is_new_record = record_id not in self.support_sample_ids
        if is_new_record:
            self.total_count += 1
            if include_task_labels:
                if is_hate:
                    self.hate_count += 1
                else:
                    self.non_hate_count += 1
        if source in {"target", "argument", "rationale"}:
            self.annotation_count += 1
        if include_task_labels:
            for category in categories:
                self.category_counts[category] += 1
        if is_new_record:
            self.support_sample_ids.append(record_id)
        if is_new_record and len(self.sample_contexts) < max_samples:
            context = {
                "id": record_id,
                "source": source,
                "content": truncate_text(content, 280),
            }
            if include_task_labels:
                context["label"] = "hate" if is_hate else "non-hate"
                context["categories"] = categories
            self.sample_contexts.append(context)

    @property
    def primary_category(self) -> str:
        if self.category_counts:
            return self.category_counts.most_common(1)[0][0]
        return "others"

    @property
    def track(self) -> str:
        if not self.track_counts:
            return "ngram_backoff"
        return max(
            self.track_counts,
            key=lambda name: (TRACK_PRIORITY.get(name, 0), self.track_counts[name]),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_counts"] = dict(self.source_counts)
        payload["track_counts"] = dict(self.track_counts)
        payload["track"] = self.track
        payload["category_counts"] = dict(self.category_counts)
        payload["primary_category"] = self.primary_category
        return payload


@dataclass
class CorpusCandidateResult:
    candidates: list[CandidateStats]
    total_records: int
    hate_records: int
    non_hate_records: int
    input_paths: list[str]


class JudgementClient(Protocol):
    def complete_json(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class LLMAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
        retryable: bool = False,
        provider_abstention: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.retryable = retryable
        self.provider_abstention = provider_abstention

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "status_code": self.status_code,
            "retryable": self.retryable,
            "provider_abstention": self.provider_abstention,
            "response_body": truncate_text(self.response_body or "", 2000),
        }


class CandidateJudgementError(RuntimeError):
    def __init__(self, stage: str, candidate: "CandidateStats", partial: dict[str, Any], original: Exception):
        super().__init__(f"LLM judgement failed for term={candidate.term!r} at stage={stage}: {original}")
        self.stage = stage
        self.candidate = candidate
        self.partial = partial
        self.original = original

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "term": self.candidate.term,
            "error_type": self.original.__class__.__name__,
            "message": str(self.original),
            "api_error": self.original.to_payload() if isinstance(self.original, LLMAPIError) else None,
        }


class DebugRecorder:
    def __init__(self, output_dir: str | Path, settings: dict[str, Any] | bool | None = None):
        if isinstance(settings, bool):
            enabled = settings
            debug_dir = None
        else:
            settings = dict(settings or {})
            enabled = bool(settings.get("enabled", False))
            debug_dir = settings.get("debug_dir")
        self.enabled = enabled
        self.debug_dir = Path(debug_dir) if debug_dir else Path(output_dir) / "debug"
        billing_ledger_path = (
            settings.get("billing_ledger_path")
            if isinstance(settings, dict)
            else None
        )
        self.billing_ledger_path = (
            Path(str(billing_ledger_path)) if billing_ledger_path else None
        )
        if self.enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            for name in ("llm_calls.jsonl", "search_calls.jsonl"):
                (self.debug_dir / name).write_text("", encoding="utf-8")
            if self.billing_ledger_path is not None:
                self.billing_ledger_path.parent.mkdir(parents=True, exist_ok=True)
                self.billing_ledger_path.write_text("", encoding="utf-8")

    def record_llm_call(
        self,
        *,
        stage: str,
        payload: dict[str, Any],
        request_payload: dict[str, Any],
        attempt: int,
        raw_response: dict[str, Any] | None = None,
        parsed_response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        candidate = payload.get("candidate") if isinstance(payload, dict) else {}
        self._append(
            "llm_calls.jsonl",
            {
                "timestamp": time.time(),
                "stage": stage,
                "term": candidate.get("term") if isinstance(candidate, dict) else None,
                "attempt": attempt,
                "request_payload": request_payload,
                "raw_response": raw_response,
                "parsed_response": parsed_response,
                "error": error,
            },
        )
        self._append_billing_event(
            {
                "event": "llm_attempt",
                "stage": stage,
                "attempt": attempt,
                "request_model": request_payload.get("model"),
                "response_model": (
                    raw_response.get("model")
                    if isinstance(raw_response, dict)
                    else None
                ),
                "usage": self._billing_usage(raw_response),
                "failed": bool(error),
            }
        )

    def record_search_call(
        self,
        *,
        term: str,
        query: str,
        results: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._append(
            "search_calls.jsonl",
            {
                "timestamp": time.time(),
                "term": term,
                "query": query,
                "results": results or [],
                "error": error,
            },
        )
        self._append_billing_event(
            {
                "event": "web_attempt",
                "result_count": len(results or []),
                "failed": bool(error),
            }
        )

    def _append(self, filename: str, row: dict[str, Any]) -> None:
        with open(self.debug_dir / filename, "a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _billing_usage(raw_response: dict[str, Any] | None) -> dict[str, int] | None:
        usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
        if not isinstance(usage, dict):
            return None
        fields = (
            "prompt_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "completion_tokens",
            "total_tokens",
        )
        sanitized = {
            field: value
            for field in fields
            if isinstance((value := usage.get(field)), int)
            and not isinstance(value, bool)
            and value >= 0
        }
        return sanitized or None

    def _append_billing_event(self, row: dict[str, Any]) -> None:
        if self.billing_ledger_path is None:
            return
        with open(self.billing_ledger_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class DisabledJudgementClient:
    def complete_json(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        contract = payload.get("decision_contract")
        terminology_mode = (
            isinstance(contract, Mapping)
            and contract.get("resource_role") == TERMINOLOGY_LIBRARY_ROLE
        )
        if stage == "final_lexicon_judge":
            if terminology_mode:
                return {
                    "include": False,
                    "definition": "",
                    "usage_notes": "",
                    "ambiguity_notes": "",
                    "variants": [],
                    "confidence": 0,
                    "reason": "LLM backend is disabled.",
                    "evidence_ids": [],
                }
            return {
                "include": False,
                "category": "others",
                "categories": ["others"],
                "definition": "",
                "nonhateful_meaning": "",
                "variants": [],
                "confidence": 0,
                "reason": "LLM backend is disabled.",
                "evidence_ids": [],
            }
        if stage == "context_judge":
            if terminology_mode:
                return {
                    "supported": False,
                    "confidence": 0,
                    "reason": "LLM backend is disabled.",
                }
            return {
                "supported": False,
                "category": "others",
                "categories": ["others"],
                "confidence": 0,
                "reason": "LLM backend is disabled.",
            }
        return {
            "supported": False,
            "confidence": 0,
            "reason": "LLM backend is disabled.",
            "evidence_ids": [],
        }


def validate_formal_deepseek_response_metadata(
    response_data: Any,
    *,
    requested_model: Any,
    max_tokens: Any,
) -> dict[str, int]:
    if requested_model != _FORMAL_DEEPSEEK_MODEL:
        raise LLMAPIError(
            "formal DeepSeek response audit has an unauthorized request model",
            retryable=False,
        )
    if not isinstance(response_data, dict):
        raise LLMAPIError(
            "formal DeepSeek response must be a JSON object", retryable=False
        )
    returned_model = response_data.get("model")
    if (
        not isinstance(returned_model, str)
        or _FORMAL_DEEPSEEK_RETURNED_MODEL_RE.fullmatch(returned_model.strip())
        is None
    ):
        raise LLMAPIError(
            "formal DeepSeek response returned an unauthorized model",
            retryable=False,
        )
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        raise LLMAPIError(
            "formal DeepSeek response lacks token usage", retryable=False
        )
    normalized: dict[str, int] = {}
    for field in _FORMAL_DEEPSEEK_USAGE_FIELDS:
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LLMAPIError(
                f"formal DeepSeek response has invalid usage field {field}",
                retryable=False,
            )
        normalized[field] = value
    if (
        normalized["prompt_cache_hit_tokens"]
        + normalized["prompt_cache_miss_tokens"]
        != normalized["prompt_tokens"]
        or normalized["prompt_tokens"] + normalized["completion_tokens"]
        != normalized["total_tokens"]
    ):
        raise LLMAPIError(
            "formal DeepSeek response token usage totals are inconsistent",
            retryable=False,
        )
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or normalized["completion_tokens"] > max_tokens
    ):
        raise LLMAPIError(
            "formal DeepSeek response exceeds the authorized output budget",
            retryable=False,
        )
    return normalized


class OpenAICompatibleJudgementClient:
    """OpenAI-compatible chat-completions client used only for JSON judgements."""

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        debug_recorder: DebugRecorder | None = None,
        *,
        formal_checkpoint: Any | None = None,
    ):
        settings = dict(settings or {})
        self.provider = str(settings.get("provider", "openai_compatible") or "openai_compatible").lower()
        default_api_base = "https://api.deepseek.com" if self.provider == "deepseek" else None
        default_key_env = "DEEPSEEK_API_KEY" if self.provider == "deepseek" else None
        default_model = "deepseek-v4-pro" if self.provider == "deepseek" else None
        self.api_base = (
            resolve_env(settings.get("api_base"), settings.get("api_base_env"), default_api_base)
            or ""
        ).rstrip("/")
        self.api_key = resolve_env(settings.get("api_key"), settings.get("api_key_env"), None)
        if self.api_key is None and default_key_env:
            self.api_key = os.getenv(default_key_env)
        self.model = resolve_env(settings.get("model"), settings.get("model_env"), default_model)
        self.timeout = int(settings.get("timeout", 120) or 120)
        self.temperature = float(settings.get("temperature", 0) or 0)
        self.max_tokens = int(settings.get("max_tokens", 1200) or 1200)
        raw_retries = settings.get("retries", 2)
        raw_retry_sleep = settings.get("retry_sleep", 1.0)
        self.retries = int(2 if raw_retries is None else raw_retries)
        self.retry_sleep = float(1.0 if raw_retry_sleep is None else raw_retry_sleep)
        self.json_mode = bool(settings.get("json_mode", True))
        self.thinking = normalize_thinking(settings.get("thinking"), self.provider)
        self.reasoning_effort = settings.get(
            "reasoning_effort",
            "high" if self.provider == "deepseek" else None,
        )
        self.stream = bool(settings.get("stream", False))
        self.send_temperature = bool(settings.get("send_temperature", self.thinking is None))
        self.output_language = normalize_output_language(settings.get("output_language", "auto"))
        self.strict_provider_audit = bool(
            settings.get("strict_provider_audit", False)
        )
        self.debug_recorder = debug_recorder
        self.formal_checkpoint = formal_checkpoint
        if not (self.api_base and self.api_key and self.model):
            raise ValueError(
                "llm_settings requires api_base/api_key/model or api_base_env/api_key_env/model_env."
            )

    def complete_json(
        self,
        stage: str,
        payload: dict[str, Any],
        *,
        checkpoint_rank: int | None = None,
    ) -> dict[str, Any]:
        if self.formal_checkpoint is not None:
            if (
                isinstance(checkpoint_rank, bool)
                or not isinstance(checkpoint_rank, int)
                or checkpoint_rank <= 0
            ):
                raise ValueError(
                    "formal DeepSeek judgement requires a positive candidate rank"
                )
            return self._complete_json_checkpointed(
                stage, payload, checkpoint_rank=checkpoint_rank
            )
        prompt = render_stage_prompt(stage, payload, output_language=self.output_language)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request_payload: dict[str, Any] | None = None
            response_data: dict[str, Any] | None = None
            try:
                request_payload = self.build_request_payload(prompt)
                response = requests.post(
                    self.chat_completion_url(),
                    json=request_payload,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    timeout=self.timeout,
                    allow_redirects=False,
                )
                raise_for_llm_status(response)
                response_data = response.json()
                if self.strict_provider_audit:
                    validate_formal_deepseek_response_metadata(
                        response_data,
                        requested_model=self.model,
                        max_tokens=self.max_tokens,
                    )
                choice = response_data["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason in {"length", "content_filter", "insufficient_system_resource"}:
                    raise LLMAPIError(
                        f"DeepSeek/OpenAI-compatible API returned finish_reason={finish_reason}",
                        retryable=finish_reason in {"insufficient_system_resource"},
                        provider_abstention=finish_reason == "content_filter",
                    )
                message = choice.get("message") or {}
                content = str(message.get("content") or "").strip()
                if not content:
                    raise RuntimeError("LLM returned empty content while JSON mode was requested.")
                parsed = parse_json_object(content)
                if self.debug_recorder is not None:
                    self.debug_recorder.record_llm_call(
                        stage=stage,
                        payload=payload,
                        request_payload=request_payload,
                        attempt=attempt + 1,
                        raw_response=response_data,
                        parsed_response=parsed,
                    )
                return parsed
            except Exception as exc:
                last_error = exc
                if self.debug_recorder is not None and request_payload is not None:
                    self.debug_recorder.record_llm_call(
                        stage=stage,
                        payload=payload,
                        request_payload=request_payload,
                        attempt=attempt + 1,
                        raw_response=(
                            response_data
                            if isinstance(response_data, dict)
                            else exception_debug_payload(exc)
                        ),
                        error=str(exc),
                    )
                if attempt < self.retries and is_retryable_llm_error(exc):
                    logger.warning(
                        "LLM judgement retry %s/%s at stage=%s after error: %s",
                        attempt + 1,
                        self.retries,
                        stage,
                        exc,
                    )
                    time.sleep(self.retry_sleep * (attempt + 1))
                else:
                    break
        if isinstance(last_error, LLMAPIError):
            raise last_error
        raise RuntimeError(f"LLM judgement failed at stage={stage}: {last_error}") from last_error

    def _complete_json_checkpointed(
        self,
        stage: str,
        payload: dict[str, Any],
        *,
        checkpoint_rank: int,
    ) -> dict[str, Any]:
        checkpoint = self.formal_checkpoint
        prompt = render_stage_prompt(
            stage, payload, output_language=self.output_language
        )
        request_payload = self.build_request_payload(prompt)
        completed = checkpoint.get_slot_success(
            "deepseek",
            checkpoint_rank,
            stage,
            request_payload=request_payload,
        )
        if completed is not None:
            response_value = completed.response
            parsed = (
                response_value.get("parsed_response")
                if isinstance(response_value, Mapping)
                else None
            )
            if not isinstance(parsed, dict) or not parsed:
                raise RuntimeError(
                    "checkpointed DeepSeek success lacks its parsed JSON response"
                )
            _validate_formal_stage_response_before_checkpoint(
                stage,
                parsed,
                payload,
            )
            return dict(parsed)

        last_error: Exception | None = None
        while True:
            reservation = checkpoint.reserve_attempt(
                "deepseek", checkpoint_rank, stage, request_payload
            )
            response_data: dict[str, Any] | None = None
            try:
                response = requests.post(
                    self.chat_completion_url(),
                    json=request_payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout,
                    allow_redirects=False,
                )
                raise_for_llm_status(response)
                response_data = response.json()
                if self.strict_provider_audit:
                    validate_formal_deepseek_response_metadata(
                        response_data,
                        requested_model=self.model,
                        max_tokens=self.max_tokens,
                    )
                choice = response_data["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason in {
                    "length",
                    "content_filter",
                    "insufficient_system_resource",
                }:
                    raise LLMAPIError(
                        "DeepSeek/OpenAI-compatible API returned "
                        f"finish_reason={finish_reason}",
                        retryable=finish_reason
                        in {"insufficient_system_resource"},
                        provider_abstention=finish_reason == "content_filter",
                    )
                message = choice.get("message") or {}
                content = str(message.get("content") or "").strip()
                if not content:
                    raise RuntimeError(
                        "LLM returned empty content while JSON mode was requested."
                    )
                parsed = parse_json_object(content)
                _validate_formal_stage_response_before_checkpoint(
                    stage,
                    parsed,
                    payload,
                )
            except Exception as exc:
                last_error = exc
                retryable = is_retryable_llm_error(exc)
                raw_error = (
                    response_data
                    if isinstance(response_data, dict)
                    else exception_debug_payload(exc)
                )
                capture = {
                    "stage": stage,
                    "term": (
                        payload.get("candidate", {}).get("term")
                        if isinstance(payload.get("candidate"), dict)
                        else None
                    ),
                    "attempt": reservation.attempt,
                    "request_payload": request_payload,
                    "raw_response": raw_error,
                    "parsed_response": None,
                    "error": str(exc),
                }
                checkpoint.finish_attempt(
                    reservation,
                    status="retryable_failure" if retryable else "terminal_failure",
                    response=None,
                    capture=capture,
                    detail={
                        "error_type": exc.__class__.__name__,
                        "http_status": (
                            exc.status_code if isinstance(exc, LLMAPIError) else None
                        ),
                        "retryable": retryable,
                        "provider_abstention": (
                            exc.provider_abstention
                            if isinstance(exc, LLMAPIError)
                            else False
                        ),
                    },
                )
                if self.debug_recorder is not None:
                    self.debug_recorder.record_llm_call(
                        stage=stage,
                        payload=payload,
                        request_payload=request_payload,
                        attempt=reservation.attempt,
                        raw_response=raw_error,
                        error=str(exc),
                    )
                if retryable and checkpoint.can_retry(
                    "deepseek", checkpoint_rank, stage
                ):
                    logger.warning(
                        "LLM judgement retry after physical attempt %s/%s at stage=%s",
                        reservation.attempt,
                        self.retries + 1,
                        stage,
                    )
                    time.sleep(self.retry_sleep * reservation.attempt)
                    continue
                break

            capture = {
                "stage": stage,
                "term": (
                    payload.get("candidate", {}).get("term")
                    if isinstance(payload.get("candidate"), dict)
                    else None
                ),
                "attempt": reservation.attempt,
                "request_payload": request_payload,
                "raw_response": response_data,
                "parsed_response": parsed,
                "error": None,
            }
            checkpoint.finish_attempt(
                reservation,
                status="success",
                response={"parsed_response": parsed},
                capture=capture,
                detail={
                    "error_type": None,
                    "http_status": getattr(response, "status_code", None),
                    "retryable": False,
                    "provider_abstention": False,
                },
            )
            if self.debug_recorder is not None:
                self.debug_recorder.record_llm_call(
                    stage=stage,
                    payload=payload,
                    request_payload=request_payload,
                    attempt=reservation.attempt,
                    raw_response=response_data,
                    parsed_response=parsed,
                )
            return parsed

        if isinstance(last_error, LLMAPIError):
            raise last_error
        raise RuntimeError(
            f"LLM judgement failed at stage={stage}: {last_error}"
        ) from last_error

    def chat_completion_url(self) -> str:
        if self.api_base.endswith("/chat/completions"):
            return self.api_base
        return self.api_base + "/chat/completions"

    def build_request_payload(self, prompt: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are building a hate-speech research lexicon. "
                        "Return strict JSON only. Do not include markdown. "
                        "Every response must be a valid JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        if self.json_mode:
            params["response_format"] = {"type": "json_object"}
        if self.send_temperature:
            params["temperature"] = self.temperature
        if self.thinking is not None:
            params["thinking"] = self.thinking
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        return params


def raise_for_llm_status(response: requests.Response) -> None:
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and 300 <= status_code <= 399:
        raise LLMAPIError(
            f"refusing redirect status {status_code} from LLM API",
            status_code=status_code,
            response_body=response_text_for_error(response),
            retryable=False,
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = getattr(response, "status_code", None)
        response_body = response_text_for_error(response)
        retryable = status_code in {408, 409, 425, 429, 500, 502, 503, 504}
        normalized_body = response_body.casefold()
        provider_abstention = any(
            marker in normalized_body
            for marker in (
                "content exists risk",
                "content filter",
                "content_filter",
                "safety policy",
            )
        )
        message = f"{status_code or 'HTTP'} error from LLM API"
        if response_body:
            message += f": {truncate_text(response_body, 1000)}"
        raise LLMAPIError(
            message,
            status_code=status_code,
            response_body=response_body,
            retryable=retryable,
            provider_abstention=provider_abstention,
        ) from exc


def response_text_for_error(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None
    if payload is not None:
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(payload)
    return str(getattr(response, "text", "") or "")


def is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, LLMAPIError):
        return exc.retryable
    if isinstance(exc, requests.Timeout | requests.ConnectionError):
        return True
    if isinstance(exc, requests.RequestException):
        return False
    return True


def exception_debug_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, LLMAPIError):
        return {"error": exc.to_payload()}
    return {"error": {"type": exc.__class__.__name__, "message": str(exc)}}


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_progress(iterable, enabled: bool, **kwargs):
    return tqdm(iterable, **kwargs) if enabled else iterable


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, *, skip_bad_lines: bool = True) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if skip_bad_lines:
                    logger.warning("Skipping malformed JSONL row: path=%s line=%s", path, line_no)
                    continue
                raise
            if isinstance(row, dict):
                rows.append(row)
    return rows


def latest_jsonl_by_term(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        term = str(row.get("term") or "").strip()
        if term:
            rows[term] = row
    return rows


def stable_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha1_json(value: Any) -> str:
    digest = hashlib.sha1()
    digest.update(stable_dumps(value).encode("utf-8"))
    return digest.hexdigest()


def resume_config_fingerprint(config: dict[str, Any]) -> str:
    return sha1_json(
        {
            "data_paths": config.get("data_paths", {}),
            "candidate_settings": config.get("candidate_settings", {}),
            "web_settings": config.get("web_settings", {}),
            "llm_settings": config.get("llm_settings", {}),
            "inclusion": config.get("inclusion", {}),
        }
    )


def candidate_signature(candidates: list[CandidateStats]) -> str:
    return sha1_json([candidate.term for candidate in candidates])


def resolve_env(value: str | None, env_name: str | None = None, default: str | None = None) -> str | None:
    if value and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], default)
    if value and value.startswith("$") and len(value) > 1:
        return os.getenv(value[1:], default)
    if value:
        return value
    return os.getenv(env_name, default) if env_name else default


def resolve_config_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: resolve_config_env(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [resolve_config_env(item) for item in value]
    if isinstance(value, str):
        if value.startswith("${") or value.startswith("$"):
            return resolve_env(value)
        resolved = resolve_env(value)
        return resolved if resolved is not None else value
    return value


def normalize_thinking(value: Any, provider: str) -> dict[str, str] | None:
    if value is None:
        return {"type": "enabled"} if provider == "deepseek" else None
    if isinstance(value, bool):
        return {"type": "enabled" if value else "disabled"}
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"enabled", "disabled"}:
            return {"type": text}
        if text in {"true", "yes", "on", "1"}:
            return {"type": "enabled"}
        if text in {"false", "no", "off", "0"}:
            return {"type": "disabled"}
    if isinstance(value, dict):
        thinking_type = str(value.get("type", "")).strip().lower()
        if thinking_type in {"enabled", "disabled"}:
            return {"type": thinking_type}
    raise ValueError("llm_settings.thinking must be enabled/disabled, a bool, or {'type': 'enabled|disabled'}.")


def truncate_text(text: Any, max_chars: int = 240) -> str:
    text = _SPACE_RE.sub(" ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def normalize_output_language(value: Any) -> str:
    raw = str(value or "auto").strip().lower().replace("_", "-")
    if raw in {"zh", "zh-cn", "cn", "chinese", "simplified-chinese", "simplified chinese"}:
        return "zh"
    if raw in {"en", "en-us", "english"}:
        return "en"
    if raw in {"auto", "same-as-candidate", "same-as-dataset", "dataset"}:
        return "auto"
    raise ValueError("llm_settings.output_language must be one of: zh, en, auto.")


def normalize_dataset(dataset: str) -> str:
    value = str(dataset or "").strip().lower()
    aliases = {"state": "full", "toxicn": "full", "hate_xplain": "hatexplain", "hateXplain": "hatexplain"}
    return aliases.get(value, value)


def default_output_language(dataset: str) -> str:
    dataset = normalize_dataset(dataset)
    return "en" if dataset == "hatexplain" else "zh"


def default_config(dataset: str) -> dict[str, Any]:
    dataset = normalize_dataset(dataset)
    return {
        "resource_role": LEGACY_DEROGATORY_LEXICON_ROLE,
        "data_paths": {
            "input_paths": DATASET_DEFAULT_PATHS[dataset],
            "output_dir": f"data/lexicon/generated/{dataset}",
        },
        "candidate_settings": {
            "max_candidates": 300,
            "max_samples_per_candidate": 5,
            "min_count_for_llm": 1,
            "min_hate_count_for_llm": 1,
            "zh_min_ngram": 2,
            "zh_max_ngram": 4,
            "zh_token_max_ngram": 4,
            "use_jieba": True,
            "keep_all_content_ngrams": False,
            "suppressed_reject_hints": [
                "broken_fragment",
                "generic_word",
                "generic_phrase",
                "singleton_ngram",
                "substring_fragment",
            ],
            "en_max_ngram": 4 if dataset == "hatexplain" else 3,
            "max_text_chars_per_record": 800,
        },
        "web_settings": {
            "backend": "disabled",
            "max_results": 3,
            "cache_enabled": True,
            "cache_path": f"data/lexicon/generated/{dataset}/web_cache.sqlite3",
            "require_direct_term_match": False,
            "dedupe_by_url": True,
        },
        "llm_settings": {
            "backend": "deepseek",
            "api_base": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "model": "deepseek-v4-pro",
            "json_mode": True,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "stream": False,
            "temperature": 0,
            "output_language": default_output_language(dataset),
            "max_tokens": 1200,
            "timeout": 120,
        },
        "inclusion": {
            "confidence_threshold": 0.65,
            "single_mention_confidence": 0.85,
            "min_count": 2,
            "ambiguous_requires_nonhateful_meaning": True,
            "neutral_identity_terms": list(DEFAULT_NEUTRAL_IDENTITY_TERMS),
        },
        "runtime_settings": {
            "show_progress": True,
            "log_level": "INFO",
            "debug": False,
            "debug_dir": None,
            "resume": True,
            "resume_require_config_match": True,
            "llm_failure_policy": "reject_candidate",
            "max_llm_failures": 50,
            "max_consecutive_llm_failures": 8,
        },
    }


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_build_config(dataset: str, config_path: str | Path | None) -> dict[str, Any]:
    config = default_config(dataset)
    if config_path:
        override = load_json(config_path)
        config = merge_config(config, override)
    return resolve_config_env(config)


def load_dataset_records(dataset: str, input_paths: list[str], show_progress: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_progress(input_paths, show_progress, desc="Loading splits", unit="file"):
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a top-level JSON list.")
        split = Path(path).stem
        for row in payload:
            if isinstance(row, dict):
                row = dict(row)
                row.setdefault("_split", split)
                row.setdefault("_dataset", dataset)
                records.append(row)
    return records


def build_candidates(
    dataset: str,
    records: list[dict[str, Any]],
    settings: dict[str, Any] | None = None,
    input_paths: list[str] | None = None,
    show_progress: bool = False,
) -> CorpusCandidateResult:
    settings = dict(settings or {})
    terminology_mode = settings.get("objective") == TERMINOLOGY_OBJECTIVE
    max_samples = int(settings.get("max_samples_per_candidate", 5) or 5)
    max_text_chars = int(settings.get("max_text_chars_per_record", 800) or 800)
    stats: dict[str, CandidateStats] = {}
    hate_records = 0
    non_hate_records = 0

    for record in iter_progress(records, show_progress, desc="Mining candidates", unit="record"):
        content = str(record.get("content") or " ".join(record.get("tokens", []) or "")).strip()
        if max_text_chars > 0:
            content_for_candidates = content[:max_text_chars]
        else:
            content_for_candidates = content
        record_id = str(record.get("id", len(stats)))
        if terminology_mode:
            # The terminology objective is mined from utterance content only.
            # Do not even parse task labels here: this makes label independence
            # structural rather than a promise made only at prompt rendering.
            is_hate, categories = False, []
        else:
            is_hate, categories = record_label_and_categories(dataset, record)
            if is_hate:
                hate_records += 1
            else:
                non_hate_records += 1

        seen_in_record: set[tuple[str, str]] = set()

        for source, text in candidate_text_sources(
            dataset,
            record,
            content_for_candidates,
            source_policy=str(settings.get("source_policy", "")),
        ):
            from_annotation = source in {"target", "argument", "rationale"}
            for mention in extract_terms_from_text(
                text=text,
                dataset=dataset,
                source=source,
                settings=settings,
            ):
                key = normalize_term(mention.term, dataset)
                if not key:
                    continue
                # Count one source hit per record to reduce overcounting long repeated spans.
                record_source_key = (key, source)
                if record_source_key in seen_in_record:
                    continue
                seen_in_record.add(record_source_key)
                # Language follows the candidate string, not the source
                # dataset.  Mixed Chinese corpora can contain English coded
                # phrases (for example ``easy girl``), which need English
                # phrase extraction and Web queries.
                language = "zh" if contains_cjk(key) else "en"
                candidate = stats.setdefault(
                    key,
                    CandidateStats(term=key, dataset=dataset, language=language),
                )
                if from_annotation:
                    candidate.update(
                        record_id,
                        content,
                        is_hate,
                        categories,
                        source,
                        max_samples,
                        track=mention.track,
                        phrase_quality=mention.phrase_quality,
                        variant_signal=mention.variant_signal,
                        include_task_labels=not terminology_mode,
                    )
                elif (key, "__any__") not in seen_in_record:
                    candidate.update(
                        record_id,
                        content,
                        is_hate,
                        categories,
                        source,
                        max_samples,
                        track=mention.track,
                        phrase_quality=mention.phrase_quality,
                        variant_signal=mention.variant_signal,
                        include_task_labels=not terminology_mode,
                    )
                    seen_in_record.add((key, "__any__"))

    candidates = list(stats.values())
    score_candidates(
        candidates,
        hate_records,
        non_hate_records,
        objective=(TERMINOLOGY_OBJECTIVE if terminology_mode else "derogatory-value/v1"),
    )
    if terminology_mode:
        candidates.sort(
            key=lambda item: (item.score, item.total_count, item.term),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda item: (item.score, item.hate_count, item.annotation_count),
            reverse=True,
        )
    return CorpusCandidateResult(
        candidates=candidates,
        total_records=len(records),
        hate_records=hate_records,
        non_hate_records=non_hate_records,
        input_paths=list(input_paths or []),
    )


def record_label_and_categories(dataset: str, record: dict[str, Any]) -> tuple[bool, list[str]]:
    if dataset == "hatexplain":
        annotation = record.get("annotation") or {}
        label = str(annotation.get("label", "")).strip().lower()
        groups = annotation.get("target_groups") or []
        categories = [str(group).strip() for group in groups if str(group).strip()]
        return label in HATEXPLAIN_LABELS, categories or ["Other"]

    quadruples = record.get("quadruples") or record.get("gt_quadruples") or []
    if isinstance(quadruples, dict):
        quadruples = [quadruples]
    is_hate = False
    categories: list[str] = []
    for quad in quadruples if isinstance(quadruples, list) else []:
        if not isinstance(quad, dict):
            continue
        hateful = str(quad.get("hateful", "")).strip().lower().replace("_", "-")
        raw_group = quad.get("targeted_group", "")
        group_parts = split_categories(raw_group)
        canonical_group_array = isinstance(raw_group, (list, tuple))
        if hateful == "hate" or (
            not canonical_group_array and any(group != "non-hate" for group in group_parts)
        ):
            is_hate = True
        for group in group_parts:
            if group != "non-hate":
                categories.append(group)
    return is_hate, categories or ["non-hate"]


def split_categories(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(part).strip() for part in raw if str(part).strip()]
    raw = str(raw)
    parts = [part.strip() for part in re.split(r"[,;/|]", raw) if part.strip()]
    return parts or [raw.strip()]


def candidate_text_sources(
    dataset: str,
    record: dict[str, Any],
    content: str,
    *,
    source_policy: str = "",
) -> list[tuple[str, str]]:
    sources = [("content", content)]
    if source_policy == TERMINOLOGY_SOURCE_POLICY:
        return sources
    if dataset in {"full", "cold"}:
        quadruples = record.get("quadruples") or record.get("gt_quadruples") or []
        if isinstance(quadruples, dict):
            quadruples = [quadruples]
        for quad in quadruples if isinstance(quadruples, list) else []:
            if not isinstance(quad, dict):
                continue
            target = str(quad.get("target", "") or "").strip()
            argument = str(quad.get("argument", "") or "").strip()
            if target and target.upper() != "NULL":
                sources.append(("target", target))
                context = target_context_window(content, target)
                if context:
                    sources.append(("target_context", context))
            if argument:
                sources.append(("argument", argument))
    elif dataset == "hatexplain":
        annotation = record.get("annotation") or {}
        rationales = annotation.get("rationales") or []
        for rationale in rationales if isinstance(rationales, list) else []:
            if isinstance(rationale, dict):
                text = str(rationale.get("text", "") or "").strip()
            else:
                text = str(rationale or "").strip()
            if text:
                sources.append(("rationale", text))
    return sources


def target_context_window(content: str, target: str, window: int = 16) -> str:
    content = str(content or "")
    target = str(target or "").strip()
    if not content or not target:
        return ""
    index = content.find(target)
    if index < 0:
        return ""
    start = max(0, index - window)
    end = min(len(content), index + len(target) + window)
    return content[start:end]


def extract_terms_from_text(text: str, dataset: str, source: str, settings: dict[str, Any]) -> list[CandidateMention]:
    text = str(text or "").strip()
    if not text:
        return []
    mentions: dict[str, CandidateMention] = {}

    def add_mention(
        term: str,
        track: str | None = None,
        phrase_quality: float | None = None,
        variant_signal: float | None = None,
    ) -> None:
        key = normalize_term(term, dataset)
        if not is_plausible_term(key, dataset):
            return
        current_track = track or classify_candidate_track(key, dataset, source)
        current_quality = phrase_quality if phrase_quality is not None else estimate_phrase_quality(key, dataset, source)
        current_variant = variant_signal if variant_signal is not None else estimate_variant_signal(key, dataset)
        mention = CandidateMention(
            term=key,
            track=current_track,
            phrase_quality=current_quality,
            variant_signal=current_variant,
        )
        existing = mentions.get(key)
        if existing is None:
            mentions[key] = mention
            return
        existing_rank = (
            TRACK_PRIORITY.get(existing.track, 0),
            existing.phrase_quality,
            existing.variant_signal,
        )
        new_rank = (
            TRACK_PRIORITY.get(mention.track, 0),
            mention.phrase_quality,
            mention.variant_signal,
        )
        if new_rank > existing_rank:
            mentions[key] = mention

    if dataset in {"full", "cold"}:
        zh_min = int(settings.get("zh_min_ngram", 2) or 2)
        zh_max = int(settings.get("zh_max_ngram", 4) or 4)
        token_max = int(settings.get("zh_token_max_ngram", 4) or 4)
        for chunk in _CJK_RE.findall(text):
            if should_add_full_zh_chunk(chunk, source):
                add_mention(chunk, classify_candidate_track(chunk, dataset, source))
            tokens = segment_zh(chunk, settings)
            for token in tokens:
                add_mention(token, classify_candidate_track(token, dataset, source))
            for n in range(2, token_max + 1):
                if len(tokens) < n:
                    continue
                for start in range(0, len(tokens) - n + 1):
                    phrase = "".join(tokens[start : start + n])
                    if len(phrase) <= 12 and not is_sentence_like_zh(phrase):
                        add_mention(phrase, classify_candidate_track(phrase, dataset, source))
            for n in range(zh_min, zh_max + 1):
                if len(chunk) < n:
                    continue
                for start in range(0, len(chunk) - n + 1):
                    gram = chunk[start : start + n]
                    if should_keep_backoff_ngram(gram, source, settings):
                        add_mention(gram, classify_backoff_ngram_track(gram, dataset))
        en_max = int(settings.get("en_max_ngram", 3) or 3)
        for tokens in contiguous_english_token_sequences(text):
            for n in range(1, en_max + 1):
                if len(tokens) < n:
                    continue
                for start in range(0, len(tokens) - n + 1):
                    phrase = " ".join(tokens[start : start + n])
                    add_mention(
                        phrase,
                        classify_candidate_track(phrase, dataset, source),
                    )
    else:
        en_max = int(settings.get("en_max_ngram", 4) or 4)
        tokens = [normalize_english_token(token) for token in _EN_TOKEN_RE.findall(text)]
        tokens = [token for token in tokens if token]
        if source == "rationale" and 0 < len(tokens) <= en_max:
            add_mention(" ".join(tokens), "annotation_anchor")
        for n in range(1, en_max + 1):
            if len(tokens) < n:
                continue
            for start in range(0, len(tokens) - n + 1):
                gram = tokens[start : start + n]
                if n == 1 and gram[0] in EN_STOP_WORDS:
                    continue
                add_mention(
                    " ".join(gram),
                    "annotation_anchor" if source == "rationale" else "contrastive_phrase",
                )

    return list(mentions.values())


def contiguous_english_token_sequences(text: str) -> list[list[str]]:
    """Return Latin-token runs without crossing intervening CJK text.

    ``re.findall`` followed by global n-gramming would incorrectly join words
    separated by Chinese clauses.  We therefore keep only tokens whose source
    spans are separated by whitespace, ``/``, ``_`` or ``-``.
    """

    sequences: list[list[str]] = []
    current: list[str] = []
    previous_end: int | None = None
    for match in _EN_TOKEN_RE.finditer(str(text or "")):
        if previous_end is not None:
            gap = text[previous_end : match.start()]
            if _EN_TOKEN_GAP_RE.fullmatch(gap) is None:
                if current:
                    sequences.append(current)
                current = []
        normalized = normalize_english_token(match.group(0))
        if normalized:
            current.append(normalized)
        previous_end = match.end()
    if current:
        sequences.append(current)
    return sequences


def segment_zh(text: str, settings: dict[str, Any]) -> list[str]:
    if not text:
        return []
    if settings.get("use_jieba", True):
        try:
            import jieba  # type: ignore

            tokens = [token.strip() for token in jieba.cut(text) if token.strip()]
            if tokens:
                return tokens
        except Exception:
            pass
    return fallback_segment_zh(text)


def fallback_segment_zh(text: str) -> list[str]:
    return []


def should_add_full_zh_chunk(chunk: str, source: str) -> bool:
    chunk = normalize_term(chunk, "full")
    if len(chunk) <= 1 or len(chunk) > 12:
        return False
    if source == "target":
        return True
    if source == "argument":
        return len(chunk) <= 8 and not is_sentence_like_zh(chunk)
    return False


def is_sentence_like_zh(term: str) -> bool:
    if len(term) > 8:
        return True
    if len(term) >= 6 and term[-1] in ZH_SENTENCE_FINAL_PARTICLES:
        return True
    if len(term) >= 6 and term.startswith(("都", "你", "我", "他", "她", "这", "那")):
        if any(marker in term for marker in ("让", "把", "被", "给", "会", "能", "要")):
            return True
    sentence_markers = (
        "应该",
        "可以",
        "不能",
        "不会",
        "就是",
        "不是",
        "觉得",
        "喜欢",
        "这种",
        "那个",
        "这里",
        "那里",
        "我的",
        "你的",
        "他的",
        "她的",
    )
    return any(marker in term for marker in sentence_markers)


def is_plausible_zh_piece(piece: str) -> bool:
    if len(piece) < 2:
        return False
    if piece in ZH_STOP_TERMS:
        return False
    if piece[0] in ZH_BAD_PREFIXES or piece[-1] in ZH_BAD_SUFFIXES:
        return False
    if is_sentence_like_zh(piece):
        return False
    return True


def should_keep_backoff_ngram(term: str, source: str, settings: dict[str, Any]) -> bool:
    term = normalize_term(term, "full")
    if not is_plausible_zh_piece(term):
        return False
    if source in {"target", "argument", "target_context", "rationale"}:
        return True
    if settings.get("keep_all_content_ngrams", False):
        return True
    return estimate_variant_signal(term, "full") > 0 or has_offensive_hint(term)


def classify_candidate_track(term: str, dataset: str, source: str) -> str:
    if dataset == "hatexplain":
        return "annotation_anchor" if source == "rationale" else "contrastive_phrase"
    term = normalize_term(term, dataset)
    if estimate_variant_signal(term, dataset) > 0:
        return "variant_slang"
    if term in ZH_GENERIC_WORD_TERMS:
        return "ngram_backoff"
    if term in ZH_GENERIC_GROUP_TERMS:
        return "group_reference"
    if source == "target":
        return "group_reference"
    if source in {"argument", "rationale"}:
        return "annotation_anchor"
    if source == "target_context":
        return "target_context"
    return "contrastive_phrase" if len(term) >= 3 else "ngram_backoff"


def classify_backoff_ngram_track(term: str, dataset: str) -> str:
    term = normalize_term(term, dataset)
    if estimate_variant_signal(term, dataset) > 0:
        return "variant_slang"
    if term in ZH_GENERIC_WORD_TERMS:
        return "ngram_backoff"
    if term in ZH_GENERIC_GROUP_TERMS:
        return "group_reference"
    return "ngram_backoff"


def estimate_phrase_quality(term: str, dataset: str, source: str) -> float:
    term = normalize_term(term, dataset)
    if not term:
        return 0.0
    if contains_cjk(term):
        quality = min(len(term), 8) / 8.0
        if source in {"argument", "rationale", "target_context"}:
            quality += 0.2
        if term in ZH_GENERIC_WORD_TERMS:
            quality -= 0.4
        if term in ZH_GENERIC_GROUP_TERMS:
            quality -= 0.2
        if len(term) <= 3 and (term[0] in ZH_BAD_PREFIXES or term[-1] in ZH_BAD_SUFFIXES):
            quality -= 0.5
        if has_offensive_hint(term):
            quality += 0.2
        return max(0.0, min(1.3, quality))
    tokens = term.split()
    if not tokens:
        return 0.0
    quality = min(len(tokens), 4) / 4.0
    if source == "rationale":
        quality += 0.25
    return max(0.0, min(1.2, quality))


def estimate_variant_signal(term: str, dataset: str) -> float:
    term = normalize_term(term, dataset)
    if not term:
        return 0.0
    signal = 0.0
    if any(char in ZH_VARIANT_CHARS for char in term):
        signal += 0.5
    if re.search(r"[a-zA-Z0-9]", term) and contains_cjk(term):
        signal += 0.4
    if re.search(r"(.)\1{2,}", term):
        signal += 0.2
    if has_offensive_hint(term):
        signal += 0.3
    return min(signal, 1.0)


def has_offensive_hint(term: str) -> bool:
    return any(char in ZH_OFFENSIVE_HINT_CHARS for char in term)


def normalize_english_token(token: str) -> str:
    token = token.strip(" \t\r\n.,!?;:\"'()[]{}<>").lower()
    token = token.replace("_", " ")
    if len(token) < 2:
        return ""
    if token in EN_STOP_WORDS:
        return ""
    return token


def normalize_term(term: str, dataset: str) -> str:
    term = _SPACE_RE.sub(" ", str(term or "")).strip()
    term = term.strip(" \t\r\n.,!?;:\"'()[]{}<>，。！？；：“”‘’（）【】《》")
    if not term:
        return ""
    if contains_cjk(term):
        term = term.replace(" ", "")
    elif dataset == "hatexplain":
        term = term.lower()
    return term


def contains_cjk(text: str) -> bool:
    return _CJK_RE.search(text) is not None


def is_plausible_term(term: str, dataset: str) -> bool:
    term = normalize_term(term, dataset)
    if not term:
        return False
    if contains_cjk(term):
        if len(term) < 2 or len(term) > 12:
            return False
        if is_sentence_like_zh(term):
            return False
        if term in ZH_STOP_TERMS:
            return False
        if len(term) <= 3 and (term[0] in ZH_BAD_PREFIXES or term[-1] in ZH_BAD_SUFFIXES):
            return False
        if all(char == term[0] for char in term):
            return False
        return True
    tokens = term.split()
    if len(tokens) > 5:
        return False
    if len(tokens) == 1 and tokens[0] in EN_STOP_WORDS:
        return False
    if any(len(token) > 32 for token in tokens):
        return False
    return True


def score_candidates(
    candidates: list[CandidateStats],
    hate_records: int,
    non_hate_records: int,
    *,
    objective: str = "derogatory-value/v1",
) -> None:
    terminology_mode = objective == TERMINOLOGY_OBJECTIVE
    for candidate in candidates:
        if terminology_mode:
            candidate.log_odds = 0.0
            candidate.hate_precision = 0.0
            candidate.category_purity = 0.0
            candidate.nonhate_penalty = 0.0
        else:
            pos_rate = (candidate.hate_count + 0.5) / max(hate_records + 1, 1)
            neg_rate = (candidate.non_hate_count + 0.5) / max(non_hate_records + 1, 1)
            candidate.log_odds = math.log(pos_rate / neg_rate)
            candidate.hate_precision = candidate.hate_count / max(candidate.total_count, 1)
            candidate.category_purity = compute_category_purity(candidate)
            candidate.nonhate_penalty = min(
                3.0, math.log1p(candidate.non_hate_count) * 0.45
            )
        candidate.generic_group_word_penalty = generic_term_penalty(
            candidate,
            terminology_mode=terminology_mode,
        )
        candidate.bad_substring_penalty = 0.0
        candidate.substring_of = ""
        candidate.reject_hint = reject_hint_for_candidate(
            candidate, terminology_mode=terminology_mode
        )
        frequency_signal = min(
            3.5,
            math.log1p(
                candidate.total_count if terminology_mode else candidate.hate_count
            )
            * 0.85,
        )
        annotation_signal = min(2.0, math.log1p(candidate.annotation_count) * 0.55)
        track_bonus = TRACK_SCORE_BONUS.get(candidate.track, 0.0)
        if terminology_mode:
            source_diversity = min(1.5, math.log1p(len(candidate.source_counts)) * 0.7)
            candidate.score = (
                candidate.phrase_quality * 2.4
                + candidate.variant_signal * 2.0
                + frequency_signal
                + annotation_signal
                + track_bonus
                + source_diversity
                - candidate.generic_group_word_penalty
            )
        else:
            candidate.score = (
                max(candidate.log_odds, 0.0) * 3.0
                + candidate.hate_precision * 3.2
                + candidate.category_purity * 1.6
                + candidate.phrase_quality * 1.8
                + candidate.variant_signal * 2.0
                + frequency_signal
                + annotation_signal
                + track_bonus
                - candidate.nonhate_penalty
                - candidate.generic_group_word_penalty
            )
    apply_substring_suppression(candidates, terminology_mode=terminology_mode)
    for candidate in candidates:
        candidate.score -= candidate.bad_substring_penalty
        if candidate.bad_substring_penalty and not candidate.reject_hint:
            candidate.reject_hint = "substring_fragment"


def compute_category_purity(candidate: CandidateStats) -> float:
    counts = Counter(
        {
            category: count
            for category, count in candidate.category_counts.items()
            if category != "non-hate" and count > 0
        }
    )
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return counts.most_common(1)[0][1] / total


def generic_term_penalty(
    candidate: CandidateStats, *, terminology_mode: bool = False
) -> float:
    term = candidate.term
    penalty = 0.0
    if contains_cjk(term):
        if term in ZH_GENERIC_GROUP_TERMS and not terminology_mode:
            penalty += 2.4
        if term in ZH_GENERIC_WORD_TERMS:
            penalty += 2.8
        if candidate.track == "group_reference" and not terminology_mode:
            penalty += 1.0
        if len(term) == 2 and not has_offensive_hint(term):
            penalty += 0.5
    elif len(term.split()) == 1 and candidate.track != "annotation_anchor":
        penalty += 0.4
    return penalty


def reject_hint_for_candidate(
    candidate: CandidateStats, *, terminology_mode: bool = False
) -> str:
    term = candidate.term
    if contains_cjk(term):
        if term in ZH_GENERIC_WORD_TERMS:
            return "generic_word"
        if (
            any(generic in term for generic in ZH_GENERIC_WORD_TERMS)
            and not has_offensive_hint(term)
            and (
                candidate.phrase_quality < 0.75
                if terminology_mode
                else candidate.hate_precision <= 0.7
            )
        ):
            return "generic_phrase"
        if candidate.track == "ngram_backoff" and candidate.total_count <= 1:
            return "singleton_ngram"
        if candidate.track == "ngram_backoff" and any(marker in term for marker in ZH_INTERNAL_FRAGMENT_MARKERS):
            return "broken_fragment"
        if (
            not terminology_mode
            and term in ZH_GENERIC_GROUP_TERMS
            and candidate.track == "group_reference"
        ):
            return "neutral_group_reference"
        if len(term) <= 3 and (term[0] in ZH_BAD_PREFIXES or term[-1] in ZH_BAD_SUFFIXES):
            return "broken_fragment"
    return ""


def apply_substring_suppression(
    candidates: list[CandidateStats], *, terminology_mode: bool = False
) -> None:
    apply_english_phrase_suppression(
        candidates, terminology_mode=terminology_mode
    )
    apply_zh_affix_fragment_suppression(candidates)
    substring_parent: dict[str, CandidateStats] = {}
    left_extensions: dict[str, list[CandidateStats]] = {}
    right_extensions: dict[str, list[CandidateStats]] = {}
    parents = sorted(
        (
            candidate
            for candidate in candidates
            if contains_cjk(candidate.term)
            and len(candidate.term) > 2
            and not is_sentence_like_zh(candidate.term)
        ),
        key=lambda item: parent_rank_for_substring(
            item, terminology_mode=terminology_mode
        ),
        reverse=True,
    )
    for parent in parents:
        term = parent.term
        # Immediate extensions retain evidence about which boundary the
        # character n-gram miner cut.  Arbitrary containing phrases do not.
        left_extensions.setdefault(term[1:], []).append(parent)
        right_extensions.setdefault(term[:-1], []).append(parent)
        for start in range(len(term)):
            for end in range(start + 2, len(term) + 1):
                sub = term[start:end]
                if sub == term:
                    continue
                current = substring_parent.get(sub)
                if current is None or parent_rank_for_substring(
                    parent, terminology_mode=terminology_mode
                ) > parent_rank_for_substring(
                    current, terminology_mode=terminology_mode
                ):
                    substring_parent[sub] = parent

    for candidate in candidates:
        if not contains_cjk(candidate.term) or len(candidate.term) < 2:
            continue
        best_parent = substring_parent.get(candidate.term)
        if best_parent is candidate:
            continue
        if best_parent is None:
            continue
        if best_parent.total_count < candidate.total_count:
            continue
        if not terminology_mode:
            if best_parent.hate_count < max(1, candidate.hate_count * 0.25):
                continue
            if best_parent.hate_precision + 0.15 < candidate.hate_precision:
                continue
        overlap_strength = min(
            1.0,
            best_parent.total_count / max(candidate.total_count, 1)
            if terminology_mode
            else best_parent.hate_count / max(candidate.hate_count, 1),
        )
        candidate.substring_of = best_parent.term
        candidate.bad_substring_penalty = 1.5 + 3.0 * overlap_strength

    # A single-parent frequency comparison misses boundary fragments when a
    # shorter n-gram occurs in a few additional records.  It also misses spans
    # ending in a dangling particle when several different continuations cover
    # their occurrences.  Record-level directional coverage recovers both
    # cases.  Restrict this to supported backoff n-grams so direct annotations
    # and complete rare terms remain eligible.
    for candidate in candidates:
        if (
            candidate.bad_substring_penalty
            or candidate.reject_hint
            or candidate.track != "ngram_backoff"
            or not contains_cjk(candidate.term)
            or candidate.total_count < ZH_SUBSTRING_MIN_SUPPORT
        ):
            continue
        for direction, extensions in (
            ("left", left_extensions.get(candidate.term, [])),
            ("right", right_extensions.get(candidate.term, [])),
        ):
            evidence = directional_extension_evidence(
                candidate,
                extensions,
                terminology_mode=terminology_mode,
            )
            if evidence is None:
                continue
            best_parent, union_coverage, dominant_coverage = evidence
            fixed_boundary_fragment = (
                len(candidate.term) >= 3
                and union_coverage >= ZH_SUBSTRING_DIRECTIONAL_COVERAGE
                and dominant_coverage >= ZH_SUBSTRING_DOMINANT_PARENT_COVERAGE
            )
            dangling_suffix_fragment = (
                direction == "right"
                and candidate.term[-1] in ZH_DANGLING_SUFFIX_MARKERS
                and union_coverage >= ZH_DANGLING_SUFFIX_COVERAGE
            )
            if not (fixed_boundary_fragment or dangling_suffix_fragment):
                continue
            candidate.substring_of = best_parent.term
            candidate.bad_substring_penalty = 1.5 + 3.0 * max(
                union_coverage,
                dominant_coverage,
            )
            break


def _record_coverage(child: CandidateStats, parent: CandidateStats) -> float:
    child_ids = set(child.support_sample_ids)
    if not child_ids:
        return 0.0
    return len(child_ids.intersection(parent.support_sample_ids)) / len(child_ids)


def _mark_substring_fragment(
    candidate: CandidateStats,
    parent_term: str,
    strength: float,
) -> None:
    candidate.substring_of = parent_term
    candidate.bad_substring_penalty = max(
        candidate.bad_substring_penalty,
        1.5 + 3.0 * max(0.0, min(strength, 1.0)),
    )


def apply_english_phrase_suppression(
    candidates: list[CandidateStats], *, terminology_mode: bool = False
) -> None:
    """Suppress a Latin substring when its observed meaning belongs to a phrase.

    The old Chinese-only substring pass could never compare ``easy`` with
    ``easy girl`` in a mixed-language corpus.  Requiring record-level coverage
    keeps independently occurring English words eligible.
    """

    latin = [
        candidate
        for candidate in candidates
        if candidate.dataset in {"full", "cold"}
        and not contains_cjk(candidate.term)
        and candidate.term.split()
    ]
    parents = [candidate for candidate in latin if len(candidate.term.split()) >= 2]
    for candidate in latin:
        child_tokens = candidate.term.split()
        if not child_tokens:
            continue
        best: tuple[float, CandidateStats] | None = None
        for parent in parents:
            parent_tokens = parent.term.split()
            if len(parent_tokens) <= len(child_tokens):
                continue
            if not any(
                parent_tokens[start : start + len(child_tokens)] == child_tokens
                for start in range(len(parent_tokens) - len(child_tokens) + 1)
            ):
                continue
            coverage = _record_coverage(candidate, parent)
            if coverage < ZH_SUBSTRING_DIRECTIONAL_COVERAGE:
                continue
            rank = (
                coverage,
                parent.total_count,
                parent.score if terminology_mode else parent.hate_count,
                -len(parent.term),
            )
            if best is None or rank > (
                best[0],
                best[1].total_count,
                best[1].score if terminology_mode else best[1].hate_count,
                -len(best[1].term),
            ):
                best = (coverage, parent)
        if best is not None:
            _mark_substring_fragment(candidate, best[1].term, best[0])


def apply_zh_affix_fragment_suppression(candidates: list[CandidateStats]) -> None:
    """Reject supported one-character boundary spill around a complete base."""

    by_term = {candidate.term: candidate for candidate in candidates}
    for candidate in candidates:
        term = candidate.term
        if not contains_cjk(term) or len(term) < 2:
            continue
        checks = (
            (term[0] in ZH_FRAGMENT_PREFIX_MARKERS, term[1:]),
            (term[-1] in ZH_FRAGMENT_SUFFIX_MARKERS, term[:-1]),
        )
        for applicable, base_term in checks:
            if not applicable or not base_term:
                continue
            base = by_term.get(base_term)
            coverage = _record_coverage(candidate, base) if base is not None else 0.0
            supported_base = base is not None and coverage >= ZH_SUBSTRING_DIRECTIONAL_COVERAGE
            two_char_offensive_stem = len(term) == 2 and has_offensive_hint(base_term)
            if not (supported_base or two_char_offensive_stem):
                continue
            _mark_substring_fragment(
                candidate,
                base.term if base is not None else base_term,
                coverage if supported_base else 1.0,
            )
            break


def directional_extension_evidence(
    candidate: CandidateStats,
    extensions: list[CandidateStats],
    *,
    terminology_mode: bool = False,
) -> tuple[CandidateStats, float, float] | None:
    """Return the best immediate extension and its record-level coverage."""

    candidate_ids = set(candidate.support_sample_ids)
    if not candidate_ids or not extensions:
        return None
    covered_ids: set[str] = set()
    best_parent: CandidateStats | None = None
    best_overlap = 0
    for parent in extensions:
        overlap_ids = candidate_ids.intersection(parent.support_sample_ids)
        overlap = len(overlap_ids)
        if overlap <= 0:
            continue
        covered_ids.update(overlap_ids)
        if best_parent is None or (
            overlap,
            parent_rank_for_substring(
                parent, terminology_mode=terminology_mode
            ),
        ) > (
            best_overlap,
            parent_rank_for_substring(
                best_parent, terminology_mode=terminology_mode
            ),
        ):
            best_parent = parent
            best_overlap = overlap
    if best_parent is None:
        return None
    denominator = len(candidate_ids)
    return (
        best_parent,
        len(covered_ids) / denominator,
        best_overlap / denominator,
    )


def parent_rank_for_substring(
    candidate: CandidateStats, *, terminology_mode: bool = False
) -> tuple[float, float, float, int]:
    return (
        float(candidate.total_count),
        candidate.score if terminology_mode else float(candidate.hate_count),
        candidate.phrase_quality if terminology_mode else candidate.score,
        -len(candidate.term),
    )


def select_candidates(corpus: CorpusCandidateResult, settings: dict[str, Any]) -> list[CandidateStats]:
    max_candidates = int(settings.get("max_candidates", 300) or 300)
    min_count = int(settings.get("min_count_for_llm", 1) or 1)
    raw_min_hate = settings.get("min_hate_count_for_llm", 1)
    min_hate = int(1 if raw_min_hate is None else raw_min_hate)
    terminology_mode = settings.get("objective") == TERMINOLOGY_OBJECTIVE
    suppressed_hints = set(
        settings.get(
            "suppressed_reject_hints",
            ["broken_fragment", "generic_word", "generic_phrase", "substring_fragment"],
        )
        or []
    )
    selected = []
    for candidate in corpus.candidates:
        if candidate.total_count < min_count:
            continue
        if (
            not terminology_mode
            and candidate.hate_count < min_hate
            and candidate.annotation_count == 0
        ):
            continue
        if candidate.reject_hint in suppressed_hints:
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def build_search_queries(
    candidate: CandidateStats,
    *,
    resource_role: str = LEGACY_DEROGATORY_LEXICON_ROLE,
) -> list[str]:
    term = candidate.term
    if resource_role == TERMINOLOGY_LIBRARY_ROLE:
        if candidate.language == "zh":
            return [
                f'"{term}" 含义 用法',
                f'"{term}" 网络用语 语境',
                f'"{term}" 歧义 非字面含义',
            ]
        return [
            f'"{term}" meaning usage',
            f'"{term}" slang context',
            f'"{term}" ambiguity nonliteral meaning',
        ]
    if candidate.language == "zh":
        return [
            f'"{term}" 侮辱性 含义',
            f'"{term}" 仇恨言论 黑话',
            f'"{term}" 歧视 用语',
        ]
    return [
        f'"{term}" slur meaning',
        f'"{term}" hate speech',
        f'"{term}" offensive term protected group',
    ]


def output_language_instruction(
    output_language: str,
    *,
    resource_role: str = LEGACY_DEROGATORY_LEXICON_ROLE,
) -> str:
    language = normalize_output_language(output_language)
    common = (
        "Keep JSON keys, boolean values, numeric values, evidence_ids, term text, and variants "
        "unchanged."
        if resource_role == TERMINOLOGY_LIBRARY_ROLE
        else "Keep JSON keys, boolean values, numeric values, evidence_ids, term text, variants, "
        "and canonical category labels unchanged."
    )
    if language == "zh":
        if resource_role == TERMINOLOGY_LIBRARY_ROLE:
            return (
                "所有自然语言解释字段必须使用简体中文：reason、definition、usage_notes、"
                "ambiguity_notes（当这些字段存在时）。JSON key、布尔值、数值、"
                "evidence_ids、term、variants 保持不变。"
            )
        return (
            "所有自然语言解释字段必须使用简体中文：reason、definition、nonhateful_meaning"
            "（当这些字段存在时）。JSON key、布尔值、数值、evidence_ids、term、variants "
            "以及规范 category 标签保持不变。"
        )
    if language == "en":
        if resource_role == TERMINOLOGY_LIBRARY_ROLE:
            return (
                "Write all natural-language explanatory field values in English: "
                "reason, definition, usage_notes, and ambiguity_notes when those keys "
                f"are present. {common}"
            )
        return (
            "Write all natural-language explanatory field values in English: "
            "reason, definition, and nonhateful_meaning when those keys are present. "
            f"{common}"
        )
    return (
        "Write natural-language explanatory field values in the same language as the candidate "
        "and dataset examples. For Chinese candidates, use Simplified Chinese. For English "
        f"candidates, use English. {common}"
    )


def _candidate_audit_payload(
    candidate: CandidateStats,
    *,
    resource_role: str = LEGACY_DEROGATORY_LEXICON_ROLE,
) -> dict[str, Any]:
    payload = candidate.to_payload()
    if resource_role == TERMINOLOGY_LIBRARY_ROLE:
        for field in (
            "hate_count",
            "non_hate_count",
            "hate_precision",
            "log_odds",
            "category_counts",
            "category_purity",
            "nonhate_penalty",
            "primary_category",
            "annotation_count",
        ):
            payload.pop(field, None)
        for context in payload.get("sample_contexts", []):
            if isinstance(context, dict):
                context.pop("label", None)
                context.pop("categories", None)
    return payload


def _judgement_candidate_payload(
    candidate: CandidateStats,
    *,
    resource_role: str = LEGACY_DEROGATORY_LEXICON_ROLE,
) -> dict[str, Any]:
    """Return candidate facts without citation-shaped training IDs or labels."""

    payload = _candidate_audit_payload(
        candidate, resource_role=resource_role
    )
    payload.pop("support_sample_ids", None)
    contexts: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(payload.get("sample_contexts", []), start=1):
        context = dict(raw)
        context.pop("id", None)
        context["example_ordinal"] = ordinal
        contexts.append(context)
    payload["sample_contexts"] = contexts
    return payload


def _web_evidence_id_allowlist(evidence: list[dict[str, Any]]) -> list[str]:
    allowed: list[str] = []
    seen: set[str] = set()
    for row in evidence:
        evidence_id = row.get("id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            continue
        evidence_id = evidence_id.strip()
        if evidence_id not in seen:
            seen.add(evidence_id)
            allowed.append(evidence_id)
    return allowed


def _validate_model_evidence_ids(
    stage: str,
    response: dict[str, Any],
    allowed_evidence_ids: list[str],
) -> None:
    evidence_ids = response.get("evidence_ids")
    if (
        not isinstance(evidence_ids, list)
        or not all(
            isinstance(value, str) and value and value.strip() == value
            for value in evidence_ids
        )
        or len(evidence_ids) != len(set(evidence_ids))
    ):
        raise ValueError(f"{stage} returned invalid evidence_ids")
    if not set(evidence_ids).issubset(set(allowed_evidence_ids)):
        raise ValueError(f"{stage} cited an ID outside the supplied Web evidence namespace")


def _validate_formal_stage_response_before_checkpoint(
    stage: str,
    response: Mapping[str, Any],
    request_semantics: Mapping[str, Any],
) -> None:
    """Validate provider semantics before a response becomes a durable success.

    A syntactically valid JSON object is not sufficient for resume: once a
    provider slot is marked successful it is intentionally never called again.
    This validator mirrors the final formal evidence gate so an invalid model
    judgement consumes an attempt and may be retried, instead of poisoning the
    checkpoint permanently.
    """

    if stage not in {
        "context_judge",
        "web_evidence_judge",
        "final_lexicon_judge",
    }:
        raise ValueError(f"unknown formal LLM stage: {stage}")
    if not isinstance(response, Mapping) or not response:
        raise ValueError(f"{stage} response must be a non-empty object")
    confidence = response.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0 <= float(confidence) <= 1
    ):
        raise ValueError(f"{stage} response has an invalid confidence")
    if not isinstance(response.get("reason"), str):
        raise ValueError(f"{stage} response has no textual reason")

    decision_contract = request_semantics.get("decision_contract")
    terminology_mode = (
        isinstance(decision_contract, Mapping)
        and decision_contract.get("resource_role") == TERMINOLOGY_LIBRARY_ROLE
    )
    allowed_evidence_ids: list[str] | None = None
    if stage in {"web_evidence_judge", "final_lexicon_judge"}:
        raw_allowed = (
            decision_contract.get("allowed_web_evidence_ids")
            if isinstance(decision_contract, Mapping)
            else None
        )
        if (
            not isinstance(raw_allowed, list)
            or not all(
                isinstance(value, str) and value and value.strip() == value
                for value in raw_allowed
            )
            or len(raw_allowed) != len(set(raw_allowed))
        ):
            raise ValueError(
                f"{stage} request lacks a canonical Web evidence allowlist"
            )
        allowed_evidence_ids = list(raw_allowed)

    if stage in {"context_judge", "web_evidence_judge"} and not isinstance(
        response.get("supported"), bool
    ):
        raise ValueError(f"{stage} response has no boolean supported decision")

    if terminology_mode:
        expected_keys = TERMINOLOGY_RESPONSE_KEYS[stage]
        if set(response) != expected_keys:
            raise ValueError(
                f"{stage} terminology response keys must be exactly "
                f"{sorted(expected_keys)}; got {sorted(response)}"
            )
        if stage == "context_judge":
            return
        if stage == "web_evidence_judge":
            _validate_model_evidence_ids(
                stage, dict(response), allowed_evidence_ids or []
            )
            if bool(response["supported"]) and not response.get("evidence_ids"):
                raise ValueError(
                    "web_evidence_judge cannot return supported=true without citing supplied Web evidence"
                )
            return
        if not isinstance(response.get("include"), bool):
            raise ValueError(
                "final_lexicon_judge response has no boolean include decision"
            )
        for key in ("definition", "usage_notes", "ambiguity_notes", "reason"):
            value = response.get(key)
            if not isinstance(value, str) or value != value.strip():
                raise ValueError(
                    f"final_lexicon_judge response has an invalid {key}"
                )
        if bool(response["include"]) and not response["definition"]:
            raise ValueError(
                "included terminology entry must have a non-empty definition"
            )
        for key in ("variants", "evidence_ids"):
            values = response.get(key)
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise ValueError(
                    f"final_lexicon_judge response has an invalid {key}"
                )
        if any(
            not value or value != value.strip()
            for value in response["variants"]
        ):
            raise ValueError(
                "final_lexicon_judge response has non-canonical variants"
            )
        _validate_model_evidence_ids(
            stage, dict(response), allowed_evidence_ids or []
        )
        if bool(response["include"]):
            context_judge = request_semantics.get("context_judge")
            web_judge = request_semantics.get("web_evidence_judge")
            if not (
                isinstance(context_judge, Mapping)
                and context_judge.get("supported") is True
            ) and not (
                isinstance(web_judge, Mapping)
                and web_judge.get("supported") is True
            ):
                raise ValueError(
                    "final_lexicon_judge cannot include a candidate rejected by both evidence judges"
                )
        return

    canonical_categories = set(ZH_CATEGORIES)

    def validate_categories(*, negative: bool) -> None:
        category = response.get("category")
        categories = response.get("categories")
        if category not in canonical_categories:
            raise ValueError(f"{stage} response has a non-canonical category")
        if (
            not isinstance(categories, list)
            or not categories
            or any(value not in canonical_categories for value in categories)
            or len(categories) != len(set(categories))
            or category not in categories
        ):
            raise ValueError(f"{stage} response has non-canonical categories")
        if negative and (category != "others" or categories != ["others"]):
            raise ValueError(
                f"{stage} negative response must use category/categories=others"
            )

    if stage == "context_judge":
        validate_categories(negative=not bool(response["supported"]))
        return
    if stage == "web_evidence_judge":
        _validate_model_evidence_ids(
            stage, dict(response), allowed_evidence_ids or []
        )
        if bool(response["supported"]) and not response.get("evidence_ids"):
            raise ValueError(
                "web_evidence_judge cannot return supported=true without citing supplied Web evidence"
            )
        return

    if not isinstance(response.get("include"), bool):
        raise ValueError(
            "final_lexicon_judge response has no boolean include decision"
        )
    for key in ("category", "definition", "nonhateful_meaning", "reason"):
        value = response.get(key)
        if not isinstance(value, str) or value != value.strip():
            raise ValueError(
                f"final_lexicon_judge response has an invalid {key}"
            )
    for key in ("categories", "variants", "evidence_ids"):
        values = response.get(key)
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(
                f"final_lexicon_judge response has an invalid {key}"
            )
    variants = response["variants"]
    if any(not value or value != value.strip() for value in variants):
        raise ValueError(
            "final_lexicon_judge response has non-canonical variants"
        )
    validate_categories(negative=not bool(response["include"]))
    _validate_model_evidence_ids(
        stage, dict(response), allowed_evidence_ids or []
    )
    if bool(response["include"]):
        context_judge = request_semantics.get("context_judge")
        web_judge = request_semantics.get("web_evidence_judge")
        context_supported = isinstance(context_judge, Mapping) and context_judge.get(
            "supported"
        ) is True
        web_supported = isinstance(web_judge, Mapping) and web_judge.get(
            "supported"
        ) is True
        if not (context_supported or web_supported):
            raise ValueError(
                "final_lexicon_judge cannot include a candidate rejected by both evidence judges"
            )
        if context_supported and not web_supported:
            context_categories = (
                context_judge.get("categories")
                if isinstance(context_judge, Mapping)
                else None
            )
            if (
                isinstance(context_categories, list)
                and context_categories
                and not set(response["categories"]).intersection(context_categories)
            ):
                raise ValueError(
                    "final_lexicon_judge category conflicts with the only supporting evidence judge"
                )


def _decision_contract(
    *,
    resource_role: str = LEGACY_DEROGATORY_LEXICON_ROLE,
    allowed_evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    if resource_role == TERMINOLOGY_LIBRARY_ROLE:
        contract: dict[str, Any] = {
            "resource_role": TERMINOLOGY_LIBRARY_ROLE,
            "task_labels_visible": False,
            "inclusion_scope": (
                "Include a term when a concise explanation of its meaning, usage, register, "
                "nonliteral sense, or ambiguity would materially help interpret a sentence. "
                "Neutral identities, slang, coded language, pejoratives, reclaimed terms, and "
                "context-dependent expressions are all eligible; ordinary transparent words "
                "and sentence fragments are not."
            ),
            "model_visible_definition": (
                "Published entries contain no task category. Definition, usage_notes, and "
                "ambiguity_notes must describe language facts only and must not state a hate "
                "or targeted-group classification outcome."
            ),
        }
    else:
        contract = {
        "canonical_categories": list(ZH_CATEGORIES),
        "negative_category": "others",
        "candidate_attribution": (
            "Judge the exact candidate string itself. Record labels, target categories, and "
            "toxic neighbouring words describe the whole example and must not be projected "
            "onto a neutral, generic, or fragmentary candidate."
        ),
        "neutral_identity_rule": (
            "A neutral protected-identity label is not a lexicon item merely because a speaker "
            "uses it inside an insult or hateful sentence. Inclusion requires an independently "
            "established derogatory lexical sense of the candidate itself, distinct from the "
            "ordinary identity reference."
        ),
        "model_visible_definition": (
            "The definition becomes downstream model-visible lexical evidence. Describe only "
            "the term's supported meaning, usage/register, and ambiguity; do not copy a "
            "canonical task-category label or state a classification outcome as its definition."
        ),
        }
    if allowed_evidence_ids is not None:
        contract["allowed_web_evidence_ids"] = allowed_evidence_ids
        contract["citation_rule"] = (
            "evidence_ids must be a unique subset of allowed_web_evidence_ids. Training sample "
            "IDs, context ordinals, URLs, query text, and invented IDs are forbidden citations."
        )
    return contract


def render_stage_prompt(stage: str, payload: dict[str, Any], output_language: str = "auto") -> str:
    decision_contract = payload.get("decision_contract")
    resource_role = (
        decision_contract.get("resource_role")
        if isinstance(decision_contract, Mapping)
        else LEGACY_DEROGATORY_LEXICON_ROLE
    )
    terminology_mode = resource_role == TERMINOLOGY_LIBRARY_ROLE
    shared_contract = (
        "Terminology-library contract: judge whether explaining the exact candidate would "
        "materially help interpret sentence meaning or pragmatic use. Neutral identity terms, "
        "slang, coded language, pejoratives, reclaimed uses, and ambiguous expressions are "
        "eligible. Do not infer or emit task categories, targeted-group labels, or hate labels. "
        "Reject ordinary transparent words, accidental substrings, and sentence fragments."
        if terminology_mode
        else
        "Category contract: category and every value in categories must be exactly one of "
        "Sexism, Racism, Region, LGBTQ, others. Never emit none, non-hate, Other, or a new "
        "label. For a negative supported/include decision, return category=others and "
        "categories=[\"others\"]. Candidate-attribution contract: judge the exact candidate "
        "string, not the overall toxicity or protected-group topic of an example. Dataset "
        "labels and target categories apply to the complete example; nearby slurs, threats, "
        "or insults do not make a neutral candidate hateful. Reject sentence fragments, "
        "generic threats/commands, and neutral group or identity names when the candidate "
        "itself has no independently established derogatory lexical meaning. Using a neutral "
        "protected-identity label in a hateful sentence or as a contextual insult is not by "
        "itself sufficient for inclusion."
    )
    if stage == "context_judge":
        instructions = (
            "Using only the supplied unlabeled contexts, judge whether the exact candidate has "
            "a stable or context-dependent meaning/usage worth explaining in a terminology "
            "library. Return JSON with keys exactly: supported, confidence, reason."
            if terminology_mode
            else
            "Judge whether the exact candidate term itself functions as discriminatory, hateful, "
            "offensive, or coded group-reference language in the provided dataset examples. "
            "Use only those examples, not any existing lexicon. A positive example label is not "
            "evidence that every substring in that example is hateful. Return JSON with keys "
            "exactly: supported, category, categories, confidence, reason."
        )
    elif stage == "web_evidence_judge":
        instructions = (
            "Judge whether the supplied Web evidence establishes a meaning, usage, register, "
            "nonliteral sense, or ambiguity of the exact candidate. Search-query wording is not "
            "evidence. Return JSON with keys exactly: supported, confidence, reason, evidence_ids. "
            "evidence_ids must be a unique subset of "
            "payload.decision_contract.allowed_web_evidence_ids; use [] when unsupported."
            if terminology_mode
            else
            "Judge whether the supplied web evidence establishes a derogatory lexical meaning "
            "of the exact candidate itself. Search-query wording is not evidence, and a page that "
            "merely discusses a protected identity or contains another nearby slur is insufficient. "
            "Return JSON with keys exactly: supported, confidence, reason, evidence_ids. "
            "evidence_ids must be unique and must be a subset of "
            "payload.decision_contract.allowed_web_evidence_ids. Never cite training sample IDs, "
            "context ordinals, URLs, query strings, or invented IDs; use [] when no supplied Web "
            "record supports the decision."
        )
    elif stage == "final_lexicon_judge":
        instructions = (
            "Make the final inclusion decision for a category-free terminology-understanding "
            "library. Include the exact candidate when its explanation would materially help a "
            "reader interpret sentence meaning or pragmatic use, regardless of whether the term "
            "is neutral, derogatory, reclaimed, coded, or ambiguous. Return JSON with keys "
            "exactly: include, definition, usage_notes, ambiguity_notes, variants, confidence, "
            "reason, evidence_ids. Keep definition factual and concise; use usage_notes for "
            "register/context and ambiguity_notes for alternate or nonliteral senses. Never emit "
            "category, categories, targeted_group, or hateful. evidence_ids must be a unique "
            "subset of payload.decision_contract.allowed_web_evidence_ids; use [] if no supplied "
            "Web evidence supports the decision."
            if terminology_mode
            else
            "Make the final inclusion decision for a conservative research lexicon by combining "
            "dataset and Web evidence about the exact candidate. Do not turn a neutral identity "
            "label into a slur merely because the surrounding statement is hateful. Return JSON "
            "with keys exactly: include, category, categories, definition, nonhateful_meaning, "
            "variants, confidence, reason, evidence_ids. Definitions should explain an "
            "independently supported hateful/offensive lexical usage, not endorse it. The "
            "definition becomes downstream model-visible evidence: describe meaning, usage, "
            "register, and ambiguity, and do not copy a canonical task-category label or state "
            "a classification outcome as the definition. "
            "evidence_ids must be unique and must be a subset of "
            "payload.decision_contract.allowed_web_evidence_ids. Training sample IDs, context "
            "ordinals, URLs, query strings, and invented IDs are forbidden; use [] if no supplied "
            "Web evidence supports the final decision."
        )
    else:
        raise ValueError(f"Unknown LLM stage: {stage}")
    instructions = (
        instructions
        + " "
        + shared_contract
        + " "
        + output_language_instruction(
            output_language, resource_role=str(resource_role)
        )
    )
    return instructions + "\n\nPayload JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def parse_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    text = _JSON_FENCE_RE.sub("", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from LLM response.")
    return value


def create_judgement_client(
    settings: dict[str, Any],
    debug_recorder: DebugRecorder | None = None,
    *,
    formal_checkpoint: Any | None = None,
) -> JudgementClient:
    backend = str(settings.get("backend", "openai_compatible") or "openai_compatible").lower()
    if backend == "disabled":
        return DisabledJudgementClient()
    if backend == "openai_compatible":
        return OpenAICompatibleJudgementClient(
            settings,
            debug_recorder=debug_recorder,
            formal_checkpoint=formal_checkpoint,
        )
    if backend == "deepseek":
        deepseek_settings = dict(settings)
        deepseek_settings["provider"] = "deepseek"
        return OpenAICompatibleJudgementClient(
            deepseek_settings,
            debug_recorder=debug_recorder,
            formal_checkpoint=formal_checkpoint,
        )
    raise ValueError(f"Unsupported llm_settings.backend: {backend}")


def create_web_searcher(
    settings: dict[str, Any], *, formal_checkpoint: Any | None = None
) -> Any:
    backend = str(settings.get("backend", "disabled") or "disabled").lower()
    if backend == "disabled":
        return DisabledWebSearcher()
    return WebSearcher(settings, formal_checkpoint=formal_checkpoint)


def judge_candidate(
    candidate: CandidateStats,
    evidence: list[dict[str, Any]],
    judge_client: JudgementClient,
    *,
    checkpoint_rank: int | None = None,
    resource_role: str = LEGACY_DEROGATORY_LEXICON_ROLE,
) -> dict[str, Any]:
    candidate_payload = _judgement_candidate_payload(
        candidate, resource_role=resource_role
    )
    partial: dict[str, Any] = {"term": candidate.term}
    context_payload = {
        "candidate": candidate_payload,
        "decision_contract": _decision_contract(resource_role=resource_role),
    }
    try:
        context_judge = (
            judge_client.complete_json(
                "context_judge",
                context_payload,
                checkpoint_rank=checkpoint_rank,
            )
            if checkpoint_rank is not None
            else judge_client.complete_json("context_judge", context_payload)
        )
        if resource_role == TERMINOLOGY_LIBRARY_ROLE:
            _validate_formal_stage_response_before_checkpoint(
                "context_judge", context_judge, context_payload
            )
    except Exception as exc:
        raise CandidateJudgementError("context_judge", candidate, partial, exc) from exc
    partial["context_judge"] = context_judge

    allowed_evidence_ids = _web_evidence_id_allowlist(evidence)
    web_payload = {
        "candidate": candidate_payload,
        "evidence": evidence,
        "decision_contract": _decision_contract(
            resource_role=resource_role,
            allowed_evidence_ids=allowed_evidence_ids,
        ),
    }
    try:
        web_judge = (
            judge_client.complete_json(
                "web_evidence_judge",
                web_payload,
                checkpoint_rank=checkpoint_rank,
            )
            if checkpoint_rank is not None
            else judge_client.complete_json("web_evidence_judge", web_payload)
        )
        if resource_role == TERMINOLOGY_LIBRARY_ROLE:
            _validate_formal_stage_response_before_checkpoint(
                "web_evidence_judge", web_judge, web_payload
            )
        _validate_model_evidence_ids(
            "web_evidence_judge",
            web_judge,
            allowed_evidence_ids,
        )
        if web_judge.get("supported") is True and not web_judge.get("evidence_ids"):
            raise ValueError(
                "web_evidence_judge cannot return supported=true without citing supplied Web evidence"
            )
    except Exception as exc:
        raise CandidateJudgementError("web_evidence_judge", candidate, partial, exc) from exc
    partial["web_evidence_judge"] = web_judge

    final_payload = {
        "candidate": candidate_payload,
        "context_judge": context_judge,
        "web_evidence_judge": web_judge,
        "evidence": evidence,
        "decision_contract": _decision_contract(
            resource_role=resource_role,
            allowed_evidence_ids=allowed_evidence_ids,
        ),
    }
    try:
        final_judge = (
            judge_client.complete_json(
                "final_lexicon_judge",
                final_payload,
                checkpoint_rank=checkpoint_rank,
            )
            if checkpoint_rank is not None
            else judge_client.complete_json("final_lexicon_judge", final_payload)
        )
        if resource_role == TERMINOLOGY_LIBRARY_ROLE:
            _validate_formal_stage_response_before_checkpoint(
                "final_lexicon_judge", final_judge, final_payload
            )
        _validate_model_evidence_ids(
            "final_lexicon_judge",
            final_judge,
            allowed_evidence_ids,
        )
        if final_judge.get("include") is True:
            context_supported = context_judge.get("supported") is True
            web_supported = web_judge.get("supported") is True
            if not (context_supported or web_supported):
                raise ValueError(
                    "final_lexicon_judge cannot include a candidate rejected by both evidence judges"
                )
            if (
                resource_role != TERMINOLOGY_LIBRARY_ROLE
                and context_supported
                and not web_supported
            ):
                context_categories = context_judge.get("categories")
                final_categories = final_judge.get("categories")
                if (
                    isinstance(context_categories, list)
                    and isinstance(final_categories, list)
                    and context_categories
                    and not set(context_categories).intersection(final_categories)
                ):
                    raise ValueError(
                        "final_lexicon_judge category conflicts with the only supporting evidence judge"
                    )
    except Exception as exc:
        raise CandidateJudgementError("final_lexicon_judge", candidate, partial, exc) from exc
    return {
        "term": candidate.term,
        "context_judge": context_judge,
        "web_evidence_judge": web_judge,
        "final_lexicon_judge": normalize_final_judge(
            final_judge, candidate, resource_role=resource_role
        ),
    }


def failed_judgement_from_error(
    candidate: CandidateStats,
    error: CandidateJudgementError | Exception,
    *,
    resource_role: str = LEGACY_DEROGATORY_LEXICON_ROLE,
) -> dict[str, Any]:
    if isinstance(error, CandidateJudgementError):
        stage = error.stage
        partial = dict(error.partial)
        error_payload = error.to_payload()
    else:
        stage = "unknown"
        partial = {"term": candidate.term}
        error_payload = {
            "stage": stage,
            "term": candidate.term,
            "error_type": error.__class__.__name__,
            "message": str(error),
            "api_error": error.to_payload() if isinstance(error, LLMAPIError) else None,
        }
    failed_final = {
            "include": False,
            "definition": "",
            "variants": [],
            "confidence": 0,
            "reason": f"LLM judgement failed at stage={stage}; candidate rejected for safety.",
            "evidence_ids": [],
    }
    if resource_role == TERMINOLOGY_LIBRARY_ROLE:
        failed_final.update({"usage_notes": "", "ambiguity_notes": ""})
    else:
        failed_final.update(
            {
                "category": candidate.primary_category,
                "categories": [candidate.primary_category],
                "nonhateful_meaning": "",
            }
        )
    final = normalize_final_judge(
        failed_final,
        candidate,
        resource_role=resource_role,
    )
    return {
        "term": candidate.term,
        "context_judge": partial.get("context_judge"),
        "web_evidence_judge": partial.get("web_evidence_judge"),
        "final_lexicon_judge": final,
        "llm_error": error_payload,
    }


def normalize_final_judge(
    raw: dict[str, Any],
    candidate: CandidateStats,
    *,
    resource_role: str = LEGACY_DEROGATORY_LEXICON_ROLE,
) -> dict[str, Any]:
    categories = raw.get("categories") or []
    if isinstance(categories, str):
        categories = split_categories(categories)
    categories = [str(category).strip() for category in categories if str(category).strip()]
    category = str(raw.get("category") or (categories[0] if categories else candidate.primary_category)).strip()
    confidence = raw.get("confidence", 0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    variants = raw.get("variants") or []
    if isinstance(variants, str):
        variants = [item.strip() for item in variants.split(",")]
    evidence_ids = raw.get("evidence_ids") or []
    if isinstance(evidence_ids, str):
        evidence_ids = [item.strip() for item in evidence_ids.split(",")]
    common = {
        "include": bool(raw.get("include", False)),
        "definition": str(raw.get("definition", "") or "").strip(),
        "variants": [str(item).strip() for item in variants if str(item).strip()],
        "confidence": max(0.0, min(confidence, 1.0)),
        "reason": str(raw.get("reason", "") or "").strip(),
        "evidence_ids": [str(item).strip() for item in evidence_ids if str(item).strip()],
    }
    if resource_role == TERMINOLOGY_LIBRARY_ROLE:
        common.update(
            {
                "usage_notes": str(raw.get("usage_notes", "") or "").strip(),
                "ambiguity_notes": str(
                    raw.get("ambiguity_notes", "") or ""
                ).strip(),
            }
        )
    else:
        common.update(
            {
                "category": category or candidate.primary_category,
                "categories": categories
                or [category or candidate.primary_category],
                "nonhateful_meaning": str(
                    raw.get("nonhateful_meaning", "") or ""
                ).strip(),
            }
        )
    return common


def _policy_term_key(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip()).casefold()


def _configured_neutral_identity_terms(settings: dict[str, Any]) -> set[str]:
    raw = settings.get("neutral_identity_terms", DEFAULT_NEUTRAL_IDENTITY_TERMS)
    if raw is None:
        return set()
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise ValueError("inclusion.neutral_identity_terms must be an array of strings")
    if not all(isinstance(value, str) and value.strip() for value in raw):
        raise ValueError("inclusion.neutral_identity_terms must contain non-empty strings")
    return {_policy_term_key(value) for value in raw}


def should_include(candidate: CandidateStats, final: dict[str, Any], settings: dict[str, Any]) -> tuple[bool, str]:
    terminology_mode = settings.get("resource_role") == TERMINOLOGY_LIBRARY_ROLE
    confidence = float(final.get("confidence", 0) or 0)
    threshold = float(settings.get("confidence_threshold", 0.65) or 0.65)
    single_threshold = float(settings.get("single_mention_confidence", 0.85) or 0.85)
    min_count = int(settings.get("min_count", 2) or 2)
    if (
        not terminology_mode
        and _policy_term_key(candidate.term)
        in _configured_neutral_identity_terms(settings)
    ):
        return False, "candidate matches the configured neutral protected-identity safety policy."
    if not final.get("include"):
        return False, "LLM final decision excluded the term."
    if confidence < threshold:
        return False, f"confidence {confidence:.2f} < threshold {threshold:.2f}."
    if terminology_mode and settings.get("definition_required", True) and not final.get("definition"):
        return False, "terminology entry lacks a definition."
    if candidate.total_count < min_count:
        if terminology_mode:
            if confidence < single_threshold:
                return False, "single mention lacks high-confidence semantic support."
        elif not (candidate.annotation_count > 0 and confidence >= single_threshold):
            return False, "single mention lacks annotation support or high confidence."
    if terminology_mode:
        return True, "included as category-free terminology evidence"
    if candidate.non_hate_count > 0 and candidate.non_hate_count >= candidate.hate_count:
        if settings.get("ambiguous_requires_nonhateful_meaning", True) and not final.get("nonhateful_meaning"):
            return False, "ambiguous term lacks nonhateful_meaning."
        if candidate.hate_count <= candidate.non_hate_count:
            return False, "non-hate usage is not lower than hate/offensive usage."
    return True, "included"


def make_lexicon_entry(
    dataset: str,
    candidate: CandidateStats,
    final: dict[str, Any],
    *,
    resource_role: str = LEGACY_DEROGATORY_LEXICON_ROLE,
) -> dict[str, Any]:
    terminology_mode = resource_role == TERMINOLOGY_LIBRARY_ROLE
    metadata = {
        "builder": "llm_lexicon_builder",
        "dataset": dataset,
        "confidence": final["confidence"],
        "reason": final.get("reason", ""),
        "support": (
            {
                "total_count": candidate.total_count,
                "sample_ids": candidate.support_sample_ids,
            }
            if terminology_mode
            else {
                "total_count": candidate.total_count,
                "hate_count": candidate.hate_count,
                "non_hate_count": candidate.non_hate_count,
                "annotation_count": candidate.annotation_count,
                "sample_ids": candidate.support_sample_ids,
            }
        ),
        "evidence_ids": final.get("evidence_ids", []),
        "source_counts": dict(candidate.source_counts),
    }
    if terminology_mode:
        return {
            "term": candidate.term,
            "definition": final.get("definition", ""),
            "usage_notes": final.get("usage_notes", ""),
            "ambiguity_notes": final.get("ambiguity_notes", ""),
            "variants": final.get("variants", []),
            "language": candidate.language,
            "metadata": metadata,
        }
    if dataset == "hatexplain":
        return {
            "term": candidate.term,
            "variants": final.get("variants", []),
            "language": "en",
            "category": final.get("category") or candidate.primary_category,
            "categories": final.get("categories") or [final.get("category") or candidate.primary_category],
            "definition": final.get("definition", ""),
            "nonhateful_meaning": final.get("nonhateful_meaning", ""),
            "average_offensiveness": None,
            "is_unambiguous": not bool(final.get("nonhateful_meaning")),
            "metadata": metadata,
        }
    return {
        "term": candidate.term,
        "category": final.get("category") or candidate.primary_category,
        "definition": final.get("definition", ""),
        "metadata": metadata,
    }


def make_build_state(
    *,
    dataset: str,
    input_paths: list[str],
    config: dict[str, Any],
    selected: list[CandidateStats],
    status: str,
    completed_count: int = 0,
    included_count: int = 0,
    rejected_count: int = 0,
    llm_failure_count: int = 0,
    last_rank: int | None = None,
    last_term: str | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "status": status,
        "dataset": dataset,
        "input_paths": input_paths,
        "config_fingerprint": resume_config_fingerprint(config),
        "candidate_signature": candidate_signature(selected),
        "selected_count": len(selected),
        "completed_count": completed_count,
        "included_count": included_count,
        "rejected_count": rejected_count,
        "llm_failure_count": llm_failure_count,
        "last_rank": last_rank,
        "last_term": last_term,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def can_resume_from_state(
    state: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    require_config_match: bool,
    has_existing_judgements: bool,
) -> tuple[bool, str]:
    if not has_existing_judgements:
        return False, "no previous judgement rows"
    if not state:
        return True, "legacy JSONL outputs without build_state.json"
    if state.get("dataset") != expected.get("dataset"):
        return False, "dataset mismatch"
    if state.get("candidate_signature") != expected.get("candidate_signature"):
        return False, "candidate signature mismatch"
    if require_config_match and state.get("config_fingerprint") != expected.get("config_fingerprint"):
        return False, "config fingerprint mismatch"
    return True, "compatible checkpoint"


def restore_completed_outputs(
    *,
    dataset: str,
    selected: list[CandidateStats],
    judgement_by_term: dict[str, dict[str, Any]],
    rejected_by_term: dict[str, dict[str, Any]],
    evidence_by_term: dict[str, list[dict[str, Any]]],
    inclusion_settings: dict[str, Any],
    resource_role: str = LEGACY_DEROGATORY_LEXICON_ROLE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]], int, set[str]]:
    inclusion_settings = dict(inclusion_settings)
    inclusion_settings["resource_role"] = resource_role
    terms: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    restored_evidence: dict[str, list[dict[str, Any]]] = {}
    completed_terms: set[str] = set()
    llm_failure_count = 0

    for rank, candidate in enumerate(selected, start=1):
        judgement = judgement_by_term.get(candidate.term)
        if not judgement:
            continue
        final = judgement.get("final_lexicon_judge")
        if not isinstance(final, dict):
            continue
        completed_terms.add(candidate.term)
        if candidate.term in evidence_by_term:
            restored_evidence[candidate.term] = evidence_by_term[candidate.term]
        include, reason = should_include(candidate, final, inclusion_settings)
        if judgement.get("llm_error"):
            llm_failure_count += 1
            reason = f"llm_error:{judgement['llm_error'].get('stage', 'unknown')}"
        if include:
            terms.append(
                make_lexicon_entry(
                    dataset,
                    candidate,
                    final,
                    resource_role=resource_role,
                )
            )
            continue
        existing_rejected = rejected_by_term.get(candidate.term)
        if existing_rejected:
            rejected.append(existing_rejected)
        else:
            rejected_row = {
                "rank": rank,
                "term": candidate.term,
                "reject_reason": reason,
                "candidate": _candidate_audit_payload(
                    candidate, resource_role=resource_role
                ),
                "final_lexicon_judge": final,
            }
            if judgement.get("llm_error"):
                rejected_row["llm_error"] = judgement["llm_error"]
            rejected.append(rejected_row)
    return terms, rejected, restored_evidence, llm_failure_count, completed_terms


def latest_evidence_by_term(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        term = str(row.get("term") or "").strip()
        if term:
            items = row.get("evidence") or []
            evidence[term] = items if isinstance(items, list) else []
    return evidence


def build_lexicon(
    dataset: str,
    config: dict[str, Any],
    judge_client: JudgementClient | None = None,
    web_searcher: Any | None = None,
    formal_checkpoint_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = normalize_dataset(dataset)
    if dataset not in DATASET_DEFAULT_PATHS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    resource_role = str(
        config.get("resource_role", LEGACY_DEROGATORY_LEXICON_ROLE)
    ).strip()
    if resource_role not in {
        LEGACY_DEROGATORY_LEXICON_ROLE,
        TERMINOLOGY_LIBRARY_ROLE,
    }:
        raise ValueError(f"Unsupported resource_role: {resource_role}")
    candidate_settings = dict(config.get("candidate_settings", {}))
    if resource_role == TERMINOLOGY_LIBRARY_ROLE and (
        candidate_settings.get("objective") != TERMINOLOGY_OBJECTIVE
        or candidate_settings.get("source_policy")
        != TERMINOLOGY_SOURCE_POLICY
    ):
        raise ValueError(
            "terminology-understanding-library requires the content-only "
            "terminology candidate contract"
        )
    inclusion_settings = dict(config.get("inclusion", {}))
    inclusion_settings["resource_role"] = resource_role

    runtime_settings = config.get("runtime_settings", {})
    show_progress = bool(runtime_settings.get("show_progress", True))
    resume_enabled = bool(runtime_settings.get("resume", True))
    resume_require_config_match = bool(runtime_settings.get("resume_require_config_match", True))
    llm_failure_policy = str(runtime_settings.get("llm_failure_policy", "reject_candidate") or "reject_candidate").lower()
    if llm_failure_policy not in {"reject_candidate", "raise"}:
        raise ValueError("runtime_settings.llm_failure_policy must be 'reject_candidate' or 'raise'.")
    max_llm_failures = int(runtime_settings.get("max_llm_failures", 50) or 50)
    max_consecutive_llm_failures = int(runtime_settings.get("max_consecutive_llm_failures", 8) or 8)
    data_paths = config.get("data_paths", {})
    input_paths = [str(path) for path in data_paths.get("input_paths") or DATASET_DEFAULT_PATHS[dataset]]
    output_dir = Path(str(data_paths.get("output_dir") or f"data/lexicon/generated/{dataset}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_settings = runtime_settings.get("debug", False)
    if isinstance(debug_settings, dict):
        debug_config = dict(debug_settings)
    else:
        debug_config = {"enabled": bool(debug_settings)}
    if runtime_settings.get("debug_dir"):
        debug_config["debug_dir"] = runtime_settings.get("debug_dir")
    if runtime_settings.get("billing_ledger_path"):
        debug_config["billing_ledger_path"] = runtime_settings.get(
            "billing_ledger_path"
        )
    debug_recorder = DebugRecorder(output_dir, debug_config)

    logger.info(
        "Starting LLM lexicon build: requested_dataset=%s normalized_dataset=%s output_dir=%s",
        DATASET_DISPLAY_NAMES.get(dataset, dataset),
        dataset,
        output_dir,
    )
    logger.info("Input splits: %s", ", ".join(input_paths))

    records = load_dataset_records(dataset, input_paths, show_progress=show_progress)
    logger.info("Loaded %s records from %s split file(s).", len(records), len(input_paths))
    corpus = build_candidates(
        dataset=dataset,
        records=records,
        settings=candidate_settings,
        input_paths=input_paths,
        show_progress=show_progress,
    )
    logger.info(
        "Candidate mining complete: raw_candidates=%s hate_records=%s non_hate_records=%s",
        len(corpus.candidates),
        corpus.hate_records,
        corpus.non_hate_records,
    )
    selected = select_candidates(corpus, candidate_settings)
    logger.info("Selected %s candidate(s) for web evidence and LLM judging.", len(selected))
    candidate_rows = [
        {
            "rank": rank + 1,
            **_candidate_audit_payload(
                candidate, resource_role=resource_role
            ),
        }
        for rank, candidate in enumerate(selected)
    ]
    write_jsonl(
        candidate_rows,
        output_dir / "candidates.jsonl",
    )
    logger.info("Wrote candidate audit file: %s", output_dir / "candidates.jsonl")

    formal_checkpoint = None
    checkpoint_summary: dict[str, Any] | None = None
    if formal_checkpoint_context is not None:
        from build_lex.formal_checkpoint import (
            CheckpointSpec,
            FormalCheckpointError,
            FormalLexiconCheckpoint,
        )

        if set(formal_checkpoint_context) != {
            "checkpoint_root",
            "scope_anchor_path",
            "scope_anchor",
            "intent",
            "forbidden_values",
        }:
            raise FormalCheckpointError(
                "formal checkpoint context fields are not canonical"
            )
        checkpoint_root = Path(str(formal_checkpoint_context["checkpoint_root"]))
        scope_anchor_path = Path(
            str(formal_checkpoint_context["scope_anchor_path"])
        )
        scope_anchor = formal_checkpoint_context.get("scope_anchor")
        intent = formal_checkpoint_context.get("intent")
        if not isinstance(intent, Mapping) or not isinstance(scope_anchor, Mapping):
            raise FormalCheckpointError("formal checkpoint context lacks its intent")
        from build_lex.train_only import (
            _activate_formal_checkpoint_scope,
            _validate_formal_checkpoint_scope_anchor,
        )

        anchor_state = _validate_formal_checkpoint_scope_anchor(
            checkpoint_root=checkpoint_root,
            anchor_path=scope_anchor_path,
            expected_anchor=scope_anchor,
        )
        web_settings = config.get("web_settings", {})
        web_budget = web_settings.get("physical_attempt_budget")
        if not isinstance(web_budget, Mapping):
            raise FormalCheckpointError("formal checkpoint lacks its Tavily budget")
        max_llm_http_attempts = runtime_settings.get("max_llm_http_attempts")
        if (
            runtime_settings.get("resume") is not True
            or runtime_settings.get("resume_require_config_match") is not True
            or runtime_settings.get("formal_checkpoint_policy")
            != "provider-slot-checkpoint/v1"
            or runtime_settings.get("ambiguous_attempt_policy")
            != "count-and-retry-within-budget/v1"
            or isinstance(max_llm_http_attempts, bool)
            or not isinstance(max_llm_http_attempts, int)
            or max_llm_http_attempts <= 0
        ):
            raise FormalCheckpointError(
                "formal runtime checkpoint policy is invalid"
            )
        llm_retries = config.get("llm_settings", {}).get("retries")
        transport_policy = web_settings.get("transport_retry_policy")
        if (
            isinstance(llm_retries, bool)
            or not isinstance(llm_retries, int)
            or llm_retries < 0
            or not isinstance(transport_policy, Mapping)
            or transport_policy.get("retries") != llm_retries
        ):
            raise FormalCheckpointError(
                "formal provider slot-attempt policies disagree"
            )
        spec = CheckpointSpec.build(
            intent=intent,
            provider_caps={
                "tavily": {
                    "cap": web_budget.get("cap"),
                    "scope_id": web_budget.get("scope_id"),
                },
                "deepseek": {
                    "cap": max_llm_http_attempts,
                    "scope_id": "stage1-p0-wp3-formal-full-deepseek/v1",
                },
            },
            candidate_frame=candidate_rows,
            max_slot_attempts=llm_retries + 1,
            active_provider_slots={
                "tavily": (
                    ("query_1", "query_2", "query_3")
                    if str(web_settings.get("backend", "")).lower()
                    == "search_api"
                    else ()
                ),
                "deepseek": (
                    "context_judge",
                    "web_evidence_judge",
                    "final_lexicon_judge",
                ),
            },
        )
        forbidden_values = tuple(
            str(value)
            for value in formal_checkpoint_context.get("forbidden_values", ())
            if isinstance(value, str) and value
        )
        if anchor_state == "active":
            formal_checkpoint = FormalLexiconCheckpoint.resume(
                checkpoint_root,
                spec,
                forbidden_values=forbidden_values,
            )
        else:
            formal_checkpoint = FormalLexiconCheckpoint.open_or_create(
                checkpoint_root,
                spec,
                forbidden_values=forbidden_values,
            )
        try:
            _activate_formal_checkpoint_scope(
                checkpoint_root=checkpoint_root,
                anchor_path=scope_anchor_path,
                expected_anchor=scope_anchor,
            )
        except BaseException:
            formal_checkpoint.close()
            raise
        logger.info(
            "Formal checkpoint ready: id=%s committed_prefix=%s/%s",
            spec.checkpoint_id,
            formal_checkpoint.committed_prefix,
            len(selected),
        )

    if judge_client is None:
        logger.info("Creating LLM judgement backend: %s", config.get("llm_settings", {}).get("backend", "openai_compatible"))
        judge_client = create_judgement_client(
            config.get("llm_settings", {}),
            debug_recorder=debug_recorder,
            formal_checkpoint=formal_checkpoint,
        )
    if web_searcher is None:
        logger.info("Creating web search backend: %s", config.get("web_settings", {}).get("backend", "disabled"))
        web_searcher = create_web_searcher(
            config.get("web_settings", {}),
            formal_checkpoint=formal_checkpoint,
        )

    evidence_path = output_dir / "web_evidence.jsonl"
    judgement_path = output_dir / "llm_judgements.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    state_path = output_dir / "build_state.json"
    expected_state = make_build_state(
        dataset=dataset,
        input_paths=input_paths,
        config=config,
        selected=selected,
        status="running",
    )

    if formal_checkpoint is not None:
        previous_evidence_by_term: dict[str, list[dict[str, Any]]] = {}
        for path in (evidence_path, judgement_path, rejected_path):
            path.write_text("", encoding="utf-8")
        terms: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        evidence_by_term: dict[str, list[dict[str, Any]]] = {}
        completed_terms: set[str] = set()
        llm_failure_count = 0
        restored_bundles = formal_checkpoint.completed_candidate_rows()
        for rank, bundle in enumerate(restored_bundles, start=1):
            candidate = selected[rank - 1]
            candidate_row = bundle.get("candidate")
            web_row = bundle.get("web_evidence")
            judgement_row = bundle.get("llm_judgement")
            term_row = bundle.get("term")
            rejected_row = bundle.get("rejected")
            if (
                candidate_row != candidate_rows[rank - 1]
                or not isinstance(web_row, Mapping)
                or web_row.get("rank") != rank
                or web_row.get("term") != candidate.term
                or not isinstance(web_row.get("evidence"), list)
                or not isinstance(judgement_row, Mapping)
                or judgement_row.get("rank") != rank
                or judgement_row.get("term") != candidate.term
                or judgement_row.get("llm_error")
                or (term_row is None) == (rejected_row is None)
                or (term_row is not None and not isinstance(term_row, Mapping))
                or (
                    rejected_row is not None
                    and (
                        not isinstance(rejected_row, Mapping)
                        or rejected_row.get("rank") != rank
                        or rejected_row.get("term") != candidate.term
                    )
                )
            ):
                raise RuntimeError(
                    "formal checkpoint candidate bundle is inconsistent with the frozen frame"
                )
            evidence_by_term[candidate.term] = [
                dict(item) for item in web_row["evidence"]
            ]
            if term_row is not None:
                terms.append(dict(term_row))
            else:
                rejected.append(dict(rejected_row))
            completed_terms.add(candidate.term)
        resume_active = bool(restored_bundles)
        resume_reason = "durable provider-slot checkpoint"
        logger.info(
            "Formal resume restored %s/%s committed candidate(s).",
            len(completed_terms),
            len(selected),
        )
    else:
        existing_state: dict[str, Any] | None = None
        if state_path.exists():
            try:
                loaded_state = load_json(state_path)
                existing_state = loaded_state if isinstance(loaded_state, dict) else None
            except Exception as exc:
                logger.warning("Could not read build checkpoint %s: %s", state_path, exc)
        existing_judgements = latest_jsonl_by_term(judgement_path)
        resume_active, resume_reason = can_resume_from_state(
            existing_state,
            expected_state,
            require_config_match=resume_require_config_match,
            has_existing_judgements=bool(existing_judgements),
        )
        resume_active = resume_enabled and resume_active
        if not resume_enabled:
            resume_reason = "resume disabled by runtime_settings.resume=false"
        if resume_active:
            previous_evidence_by_term = latest_evidence_by_term(evidence_path)
            previous_rejected_by_term = latest_jsonl_by_term(rejected_path)
            terms, rejected, evidence_by_term, llm_failure_count, completed_terms = restore_completed_outputs(
                dataset=dataset,
                selected=selected,
                judgement_by_term=existing_judgements,
                rejected_by_term=previous_rejected_by_term,
                evidence_by_term=previous_evidence_by_term,
                inclusion_settings=inclusion_settings,
                resource_role=resource_role,
            )
            logger.info(
                "Resume enabled: restored %s/%s completed candidate(s) from %s.",
                len(completed_terms),
                len(selected),
                resume_reason,
            )
        else:
            previous_evidence_by_term = {}
            for path in (evidence_path, judgement_path, rejected_path):
                path.write_text("", encoding="utf-8")
            terms = []
            rejected = []
            evidence_by_term = {}
            llm_failure_count = 0
            completed_terms = set()
            logger.info("Starting fresh judgement run: %s.", resume_reason)
    dump_json(
        {
            **expected_state,
            "completed_count": len(completed_terms),
            "included_count": len(terms),
            "rejected_count": len(rejected),
            "llm_failure_count": llm_failure_count,
        },
        state_path,
    )
    consecutive_llm_failures = 0

    judging_iter = enumerate(selected, start=1)
    progress_bar = (
        tqdm(total=len(selected), initial=len(completed_terms), desc="Web + LLM judging", unit="term")
        if show_progress
        else None
    )
    try:
        for rank, candidate in judging_iter:
            if candidate.term in completed_terms:
                continue

            evidence = evidence_by_term.get(candidate.term)
            if evidence is None:
                existing_evidence = previous_evidence_by_term.get(candidate.term) if resume_active else None
                if existing_evidence is not None:
                    evidence = existing_evidence
                else:
                    evidence = collect_web_evidence(
                        candidate,
                        web_searcher,
                        debug_recorder=debug_recorder,
                        resource_role=resource_role,
                        checkpoint_rank=(
                            rank if formal_checkpoint is not None else None
                        ),
                    )
            web_evidence_row = {
                "rank": rank,
                "term": candidate.term,
                "queries": build_search_queries(
                    candidate, resource_role=resource_role
                ),
                "evidence": evidence,
            }
            append_jsonl(web_evidence_row, evidence_path)
            evidence_by_term[candidate.term] = evidence

            try:
                judgement = judge_candidate(
                    candidate,
                    evidence,
                    judge_client,
                    checkpoint_rank=(rank if formal_checkpoint is not None else None),
                    resource_role=resource_role,
                )
                consecutive_llm_failures = 0
            except Exception as exc:
                if llm_failure_policy == "raise":
                    raise
                llm_failure_count += 1
                consecutive_llm_failures += 1
                error = exc if isinstance(exc, CandidateJudgementError) else CandidateJudgementError("unknown", candidate, {"term": candidate.term}, exc)
                judgement = failed_judgement_from_error(
                    candidate, error, resource_role=resource_role
                )
                logger.warning(
                    "LLM judgement failed; rejecting candidate and continuing: rank=%s term=%s failure=%s/%s consecutive=%s/%s error=%s",
                    rank,
                    candidate.term,
                    llm_failure_count,
                    max_llm_failures,
                    consecutive_llm_failures,
                    max_consecutive_llm_failures,
                    error,
                )
            judgement_row = {"rank": rank, **judgement}
            append_jsonl(judgement_row, judgement_path)
            final = judgement["final_lexicon_judge"]
            include, reason = should_include(
                candidate, final, inclusion_settings
            )
            if judgement.get("llm_error"):
                reason = f"llm_error:{judgement['llm_error'].get('stage', 'unknown')}"
            term_entry: dict[str, Any] | None = None
            rejected_row: dict[str, Any] | None = None
            if include:
                term_entry = make_lexicon_entry(
                    dataset,
                    candidate,
                    final,
                    resource_role=resource_role,
                )
                logger.debug("Included term: %s confidence=%.2f", candidate.term, final.get("confidence", 0))
            else:
                rejected_row = {
                    "rank": rank,
                    "term": candidate.term,
                    "reject_reason": reason,
                    "candidate": _candidate_audit_payload(
                        candidate, resource_role=resource_role
                    ),
                    "final_lexicon_judge": final,
                }
                if judgement.get("llm_error"):
                    rejected_row["llm_error"] = judgement["llm_error"]
                logger.debug("Rejected term: %s reason=%s", candidate.term, reason)
            if formal_checkpoint is not None:
                formal_checkpoint.commit_candidate(
                    rank,
                    {
                        "candidate": candidate_rows[rank - 1],
                        "web_evidence": web_evidence_row,
                        "llm_judgement": judgement_row,
                        "term": term_entry,
                        "rejected": rejected_row,
                    },
                )
            if term_entry is not None:
                terms.append(term_entry)
            elif rejected_row is not None:
                rejected.append(rejected_row)
                append_jsonl(rejected_row, rejected_path)
            completed_terms.add(candidate.term)
            dump_json(
                make_build_state(
                    dataset=dataset,
                    input_paths=input_paths,
                    config=config,
                    selected=selected,
                    status="running",
                    completed_count=len(completed_terms),
                    included_count=len(terms),
                    rejected_count=len(rejected),
                    llm_failure_count=llm_failure_count,
                    last_rank=rank,
                    last_term=candidate.term,
                ),
                state_path,
            )
            if (
                llm_failure_count >= max_llm_failures
                or consecutive_llm_failures >= max_consecutive_llm_failures
            ):
                raise RuntimeError(
                    "Too many LLM judgement failures "
                    f"(total={llm_failure_count}/{max_llm_failures}, "
                    f"consecutive={consecutive_llm_failures}/{max_consecutive_llm_failures}). "
                    "Check llm_settings, model capability, content filtering, or enable --debug."
                )
            if progress_bar is not None:
                progress_bar.set_postfix(
                    included=len(terms),
                    rejected=len(rejected),
                    llm_errors=llm_failure_count,
                    refresh=False,
                )
                progress_bar.update(1)
    except BaseException:
        if formal_checkpoint is not None:
            formal_checkpoint.close()
        if hasattr(web_searcher, "close"):
            web_searcher.close()
        raise
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if formal_checkpoint is not None:
        try:
            checkpoint_summary = formal_checkpoint.materialize(output_dir)
        except BaseException:
            formal_checkpoint.close()
            raise
        logger.info(
            "Materialized durable formal checkpoint: committed_prefix=%s provider_attempts=%s",
            checkpoint_summary["committed_prefix"],
            checkpoint_summary["provider_attempt_counts"],
        )

    logger.info(
        "LLM judging complete: included=%s rejected=%s llm_errors=%s evidence_file=%s judgement_file=%s",
        len(terms),
        len(rejected),
        llm_failure_count,
        evidence_path,
        judgement_path,
    )

    lexicon_payload = make_lexicon_payload(dataset, terms, input_paths, config)
    dump_json(lexicon_payload, output_dir / "lexicon.json")
    report = make_report(dataset, corpus, selected, terms, rejected, input_paths, output_dir, config)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    logger.info("Wrote lexicon: %s", output_dir / "lexicon.json")
    logger.info("Wrote report: %s", output_dir / "report.md")
    dump_json(
        make_build_state(
            dataset=dataset,
            input_paths=input_paths,
            config=config,
            selected=selected,
            status="complete",
            completed_count=len(completed_terms),
            included_count=len(terms),
            rejected_count=len(rejected),
            llm_failure_count=llm_failure_count,
            last_rank=len(selected) if selected else None,
            last_term=selected[-1].term if selected else None,
        ),
        state_path,
    )
    logger.info("Wrote build checkpoint: %s", state_path)
    if debug_recorder.enabled:
        logger.info("Debug traces written under: %s", debug_recorder.debug_dir)

    if hasattr(web_searcher, "close"):
        web_searcher.close()
    if formal_checkpoint is not None:
        formal_checkpoint.close()

    return {
        "dataset": dataset,
        "output_dir": str(output_dir),
        "lexicon_path": str(output_dir / "lexicon.json"),
        "total_records": corpus.total_records,
        "candidate_count": len(selected),
        "included_count": len(terms),
        "rejected_count": len(rejected),
        "llm_failure_count": llm_failure_count,
        "formal_checkpoint_summary": checkpoint_summary,
        "terms": terms,
        "evidence_by_term": evidence_by_term,
    }


def collect_web_evidence(
    candidate: CandidateStats,
    web_searcher: Any,
    debug_recorder: DebugRecorder | None = None,
    *,
    checkpoint_rank: int | None = None,
    resource_role: str = LEGACY_DEROGATORY_LEXICON_ROLE,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for slot_number, query in enumerate(
        build_search_queries(candidate, resource_role=resource_role), start=1
    ):
        try:
            query_results = (
                web_searcher.search(
                    query,
                    checkpoint_rank=checkpoint_rank,
                    checkpoint_slot=slot_number,
                    checkpoint_term=candidate.term,
                )
                if checkpoint_rank is not None
                else web_searcher.search(query)
            )
        except Exception as exc:
            if debug_recorder is not None:
                debug_recorder.record_search_call(term=candidate.term, query=query, error=str(exc))
            raise
        if debug_recorder is not None:
            debug_recorder.record_search_call(term=candidate.term, query=query, results=query_results)
        for item in query_results:
            evidence_id = str(item.get("id") or sha1_json(item)[:16])
            if evidence_id in seen_ids:
                continue
            seen_ids.add(evidence_id)
            item = dict(item)
            item["id"] = evidence_id
            evidence.append(item)
    return evidence


def make_lexicon_payload(
    dataset: str,
    terms: list[dict[str, Any]],
    input_paths: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    resource_role = str(
        config.get("resource_role", LEGACY_DEROGATORY_LEXICON_ROLE)
    )
    common = {
        "source": "llm_lexicon_builder",
        "dataset": dataset,
        "input_paths": input_paths,
        "build_config_sha1": sha1_json(config),
        "total_terms": len(terms),
        "terms": terms,
        "resource_role": resource_role,
    }
    if resource_role == TERMINOLOGY_LIBRARY_ROLE:
        display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        return {
            **common,
            "title": f"{display_name} 无类别术语理解库",
            "description": (
                "从无任务标签的 fit 文本候选与外部词义证据构建；条目只描述术语含义、"
                "用法、语域与歧义，不包含仇恨或目标群体类别。"
            ),
            "language": "en" if dataset == "hatexplain" else "zh",
        }
    if dataset == "hatexplain":
        return {
            **common,
            "title": "LLM-generated HateXplain lexicon",
            "description": "Automatically constructed from all configured HateXplain splits without using existing lexicons.",
            "language": "en",
        }
    categories = sorted({term.get("category", "others") for term in terms} | set(ZH_CATEGORIES))
    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    return {
        **common,
        "title": f"LLM 自动构建{display_name}仇恨/冒犯词典",
        "description": "从目标数据集全量 split 自动挖掘候选，并经联网证据与 LLM 判定后生成；未使用已有词典。",
        "language": "zh",
        "categories": categories,
    }


def make_report(
    dataset: str,
    corpus: CorpusCandidateResult,
    selected: list[CandidateStats],
    terms: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    input_paths: list[str],
    output_dir: Path,
    config: dict[str, Any],
) -> str:
    top_terms = ", ".join(term["term"] for term in terms[:20]) or "None"
    llm_error_count = sum(1 for item in rejected if item.get("llm_error"))
    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    terminology_mode = (
        config.get("resource_role") == TERMINOLOGY_LIBRARY_ROLE
    )
    lines = [
        f"# LLM Lexicon Build Report: {display_name}",
        "",
        f"- Normalized dataset key: `{dataset}`",
        "",
        "## Inputs",
        "",
        *[f"- `{path}`" for path in input_paths],
        "",
        "## Summary",
        "",
        f"- Records: {corpus.total_records}",
        *(
            ["- Task labels/categories: not used for candidate sources or LLM evidence"]
            if terminology_mode
            else [
                f"- Hate/offensive records: {corpus.hate_records}",
                f"- Non-hate/normal records: {corpus.non_hate_records}",
            ]
        ),
        f"- Raw candidates: {len(corpus.candidates)}",
        f"- Judged candidates: {len(selected)}",
        f"- Included terms: {len(terms)}",
        f"- Rejected terms: {len(rejected)}",
        f"- LLM judgement errors: {llm_error_count}",
        f"- Output: `{output_dir / 'lexicon.json'}`",
        "",
        "## Top Included Terms",
        "",
        top_terms,
        "",
        "## Settings",
        "",
        "```json",
        json.dumps(
            {
                "candidate_settings": config.get("candidate_settings", {}),
                "web_backend": config.get("web_settings", {}).get("backend", "disabled"),
                "llm_backend": config.get("llm_settings", {}).get("backend", "openai_compatible"),
                "inclusion": config.get("inclusion", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a dataset-derived lexicon with web evidence and LLM judgements. "
            "`full`, `state`, and `toxicn` all point to the local state/full dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["full", "cold", "hatexplain", "state", "toxicn"],
        help="Dataset key. full/state/toxicn are aliases for data/full.",
    )
    parser.add_argument("--config", default=None, help="Path to config/lexicon/<dataset>_llm.json.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write every LLM request/response and search query/output under <output_dir>/debug.",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional explicit debug output directory. Implies --debug.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume and start a fresh judgement run.",
    )
    parser.add_argument(
        "--data-ref",
        default=None,
        help="Frozen Stage-1 data locator; enables formal train-only publication.",
    )
    parser.add_argument(
        "--train-partition-ref",
        default=None,
        help=(
            "Frozen fit/calibration partition locator; required with --data-ref "
            "for formal Stage-1 publication."
        ),
    )
    parser.add_argument(
        "--train-records",
        default=None,
        help="Engineering-only normalized train JSON input (mutually exclusive with --data-ref).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train"],
        help="Stage-1 lexicon source split; v1 permits train only.",
    )
    parser.add_argument(
        "--target-root",
        default="exps/causal_context/stage1_p0/lexicons",
        help="Content-addressed Stage-1 lexicon target root.",
    )
    parser.add_argument(
        "--write-ref",
        default=None,
        help="Write a Stage-1 lexicon locator ref after validation.",
    )
    parser.add_argument(
        "--env-file",
        default=str(SRC_ROOT.parent / ".env"),
        help=(
            "Allowlisted credential file used by the formal Stage-1 preflight. "
            "Values are never expanded or included in the public report."
        ),
    )
    parser.add_argument(
        "--workspace-root",
        default=str(SRC_ROOT.parent),
        help="Workspace root used to resolve portable formal dependencies.",
    )
    return parser.parse_args()


def _canonical_formal_builder() -> Any:
    """Resolve the exact module object frozen by the formal authorization.

    ``python -m build_lex.llm_lexicon_builder`` executes this file under the
    temporary ``__main__`` module name.  Passing that duplicate module's
    ``build_lexicon`` object would correctly fail the formal identity gate even
    though its source bytes are identical.  Importing the canonical module here
    preserves the strict object/module/name checks without weakening them.
    """

    from build_lex import llm_lexicon_builder as frozen_builder_module

    return frozen_builder_module.build_lexicon


def main() -> None:
    args = parse_args()
    dataset = normalize_dataset(args.dataset)
    preflight = None
    if args.data_ref:
        # This gate intentionally runs before config merging, target-directory
        # creation, candidate mining, cache construction, or any HTTP client.
        from build_lex.stage1_preflight import preflight_formal_lexicon

        preflight = preflight_formal_lexicon(
            dataset=dataset,
            config_path=args.config,
            data_ref=args.data_ref,
            train_partition_ref=args.train_partition_ref,
            env_file=args.env_file,
            repository_root=args.workspace_root,
        )
        if not preflight.passed:
            print(
                json.dumps(preflight.to_public_dict(), ensure_ascii=False, indent=2, sort_keys=True),
                file=sys.stderr,
            )
            raise SystemExit(2)
        if args.debug or args.debug_dir:
            raise SystemExit(
                "formal Stage-1 mode forbids CLI debug overrides; use the frozen runtime policy"
            )
        if args.no_resume:
            raise SystemExit(
                "formal Stage-1 mode requires the frozen provider-slot checkpoint; "
                "do not pass --no-resume"
            )
        # Activation is process-local and happens only after every read-only
        # gate has passed; it never overwrites a different existing value.
        preflight.activate_credentials()
        # Use the exact in-memory object that passed preflight.  Re-reading the
        # path here would create a config drift/TOCTOU gap.
        # Formal config is a complete frozen protocol, not an override.  Do not
        # merge implicit defaults or perform generic $VAR interpolation here;
        # credentials are available only through the allowlisted activation.
        config = preflight.frozen_config()
    else:
        if args.train_partition_ref:
            raise SystemExit(
                "--train-partition-ref requires formal --data-ref mode"
            )
        config = load_build_config(dataset, args.config)
    if args.debug or args.debug_dir:
        config.setdefault("runtime_settings", {})["debug"] = True
    if args.debug_dir:
        config.setdefault("runtime_settings", {})["debug_dir"] = args.debug_dir
    if args.no_resume:
        config.setdefault("runtime_settings", {})["resume"] = False
    log_level = str(config.get("runtime_settings", {}).get("log_level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)-8s %(asctime)s - %(name)s - %(message)s",
    )
    if args.dataset != dataset:
        logger.info("Dataset alias resolved: %s -> %s", args.dataset, dataset)
    if args.data_ref or args.train_records or args.write_ref:
        if bool(args.data_ref) == bool(args.train_records):
            raise SystemExit("Stage-1 mode requires exactly one of --data-ref or --train-records")
        from build_lex.train_only import build_train_only_lexicon

        records = load_json(args.train_records) if args.train_records else None
        result = build_train_only_lexicon(
            dataset,
            config,
            builder=(
                _canonical_formal_builder()
                if args.data_ref
                else build_lexicon
            ),
            data_ref=args.data_ref,
            train_partition_ref=args.train_partition_ref,
            train_records=records,
            formal=bool(args.data_ref),
            target_root=args.target_root,
            write_ref=args.write_ref,
            build_authorization=(preflight.build_authorization if preflight is not None else None),
            workspace_root=args.workspace_root,
        )
    else:
        result = build_lexicon(dataset, config)
    print(json.dumps({key: value for key, value in result.items() if key != "terms"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
