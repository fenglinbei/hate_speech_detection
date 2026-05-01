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
from typing import Any, Protocol

import requests
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.web_search import DisabledWebSearcher, WebSearcher

logger = logging.getLogger(__name__)


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
HATEXPLAIN_LABELS = {"hatespeech", "offensive"}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_EN_TOKEN_RE = re.compile(r"[#@]?[A-Za-z][A-Za-z0-9_'-]*")
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
ZH_VARIANT_CHARS = set("🐒🐵💩🤢🤮4141450")
ZH_OFFENSIVE_HINT_CHARS = set("狗畜贱滚死脏蠢丑臭烂废奴鬼逼婊拳蝻蛆蛮垃")

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
    ) -> None:
        self.source_counts[source] += 1
        self.track_counts[track] += 1
        self.phrase_quality = max(self.phrase_quality, phrase_quality)
        self.variant_signal = max(self.variant_signal, variant_signal)
        is_new_record = record_id not in self.support_sample_ids
        if is_new_record:
            self.total_count += 1
            if is_hate:
                self.hate_count += 1
            else:
                self.non_hate_count += 1
        if source in {"target", "argument", "rationale"}:
            self.annotation_count += 1
        for category in categories:
            self.category_counts[category] += 1
        if is_new_record:
            self.support_sample_ids.append(record_id)
        if is_new_record and len(self.sample_contexts) < max_samples:
            self.sample_contexts.append(
                {
                    "id": record_id,
                    "label": "hate" if is_hate else "non-hate",
                    "categories": categories,
                    "source": source,
                    "content": truncate_text(content, 280),
                }
            )

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
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.retryable = retryable

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "status_code": self.status_code,
            "retryable": self.retryable,
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
        if self.enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            for name in ("llm_calls.jsonl", "search_calls.jsonl"):
                (self.debug_dir / name).write_text("", encoding="utf-8")

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

    def _append(self, filename: str, row: dict[str, Any]) -> None:
        with open(self.debug_dir / filename, "a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


class DisabledJudgementClient:
    def complete_json(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        if stage == "final_lexicon_judge":
            return {
                "include": False,
                "category": payload.get("candidate", {}).get("primary_category", "others"),
                "categories": [],
                "definition": "",
                "nonhateful_meaning": "",
                "variants": [],
                "confidence": 0,
                "reason": "LLM backend is disabled.",
                "evidence_ids": [],
            }
        return {"supported": False, "confidence": 0, "reason": "LLM backend is disabled."}


class OpenAICompatibleJudgementClient:
    """OpenAI-compatible chat-completions client used only for JSON judgements."""

    def __init__(self, settings: dict[str, Any] | None = None, debug_recorder: DebugRecorder | None = None):
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
        self.retries = int(settings.get("retries", 2) or 2)
        self.retry_sleep = float(settings.get("retry_sleep", 1.0) or 1.0)
        self.json_mode = bool(settings.get("json_mode", True))
        self.thinking = normalize_thinking(settings.get("thinking"), self.provider)
        self.reasoning_effort = settings.get(
            "reasoning_effort",
            "high" if self.provider == "deepseek" else None,
        )
        self.stream = bool(settings.get("stream", False))
        self.send_temperature = bool(settings.get("send_temperature", self.thinking is None))
        self.debug_recorder = debug_recorder
        if not (self.api_base and self.api_key and self.model):
            raise ValueError(
                "llm_settings requires api_base/api_key/model or api_base_env/api_key_env/model_env."
            )

    def complete_json(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = render_stage_prompt(stage, payload)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request_payload: dict[str, Any] | None = None
            try:
                request_payload = self.build_request_payload(prompt)
                response = requests.post(
                    self.chat_completion_url(),
                    json=request_payload,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                raise_for_llm_status(response)
                response_data = response.json()
                choice = response_data["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason in {"length", "content_filter", "insufficient_system_resource"}:
                    raise LLMAPIError(
                        f"DeepSeek/OpenAI-compatible API returned finish_reason={finish_reason}",
                        retryable=finish_reason in {"insufficient_system_resource"},
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
                        raw_response=exception_debug_payload(exc),
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
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = getattr(response, "status_code", None)
        response_body = response_text_for_error(response)
        retryable = status_code in {408, 409, 425, 429, 500, 502, 503, 504}
        message = f"{status_code or 'HTTP'} error from LLM API"
        if response_body:
            message += f": {truncate_text(response_body, 1000)}"
        raise LLMAPIError(
            message,
            status_code=status_code,
            response_body=response_body,
            retryable=retryable,
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


def normalize_dataset(dataset: str) -> str:
    value = str(dataset or "").strip().lower()
    aliases = {"state": "full", "toxicn": "full", "hate_xplain": "hatexplain", "hateXplain": "hatexplain"}
    return aliases.get(value, value)


def default_config(dataset: str) -> dict[str, Any]:
    dataset = normalize_dataset(dataset)
    return {
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
            "max_tokens": 1200,
            "timeout": 120,
        },
        "inclusion": {
            "confidence_threshold": 0.65,
            "single_mention_confidence": 0.85,
            "min_count": 2,
            "ambiguous_requires_nonhateful_meaning": True,
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
        is_hate, categories = record_label_and_categories(dataset, record)
        if is_hate:
            hate_records += 1
        else:
            non_hate_records += 1

        seen_in_record: set[tuple[str, str]] = set()

        for source, text in candidate_text_sources(dataset, record, content_for_candidates):
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
                language = "zh" if contains_cjk(key) or dataset in {"full", "cold"} else "en"
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
                    )
                    seen_in_record.add((key, "__any__"))

    candidates = list(stats.values())
    score_candidates(candidates, hate_records, non_hate_records)
    candidates.sort(key=lambda item: (item.score, item.hate_count, item.annotation_count), reverse=True)
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
        raw_group = str(quad.get("targeted_group", "")).strip()
        group_parts = split_categories(raw_group)
        if hateful == "hate" or any(group != "non-hate" for group in group_parts):
            is_hate = True
        for group in group_parts:
            if group != "non-hate":
                categories.append(group)
    return is_hate, categories or ["non-hate"]


def split_categories(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [part.strip() for part in re.split(r"[,;/|]", raw) if part.strip()]
    return parts or [raw.strip()]


def candidate_text_sources(dataset: str, record: dict[str, Any], content: str) -> list[tuple[str, str]]:
    sources = [("content", content)]
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
        for token in _EN_TOKEN_RE.findall(text):
            normalized = normalize_english_token(token)
            if normalized:
                add_mention(normalized, classify_candidate_track(normalized, dataset, source))
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


def score_candidates(candidates: list[CandidateStats], hate_records: int, non_hate_records: int) -> None:
    for candidate in candidates:
        pos_rate = (candidate.hate_count + 0.5) / max(hate_records + 1, 1)
        neg_rate = (candidate.non_hate_count + 0.5) / max(non_hate_records + 1, 1)
        candidate.log_odds = math.log(pos_rate / neg_rate)
        candidate.hate_precision = candidate.hate_count / max(candidate.total_count, 1)
        candidate.category_purity = compute_category_purity(candidate)
        candidate.nonhate_penalty = min(3.0, math.log1p(candidate.non_hate_count) * 0.45)
        candidate.generic_group_word_penalty = generic_term_penalty(candidate)
        candidate.bad_substring_penalty = 0.0
        candidate.substring_of = ""
        candidate.reject_hint = reject_hint_for_candidate(candidate)
        frequency_signal = min(3.5, math.log1p(candidate.hate_count) * 0.85)
        annotation_signal = min(2.0, math.log1p(candidate.annotation_count) * 0.55)
        track_bonus = TRACK_SCORE_BONUS.get(candidate.track, 0.0)
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
    apply_substring_suppression(candidates)
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


def generic_term_penalty(candidate: CandidateStats) -> float:
    term = candidate.term
    penalty = 0.0
    if contains_cjk(term):
        if term in ZH_GENERIC_GROUP_TERMS:
            penalty += 2.4
        if term in ZH_GENERIC_WORD_TERMS:
            penalty += 2.8
        if candidate.track == "group_reference":
            penalty += 1.0
        if len(term) == 2 and not has_offensive_hint(term):
            penalty += 0.5
    elif len(term.split()) == 1 and candidate.track != "annotation_anchor":
        penalty += 0.4
    return penalty


def reject_hint_for_candidate(candidate: CandidateStats) -> str:
    term = candidate.term
    if contains_cjk(term):
        if term in ZH_GENERIC_WORD_TERMS:
            return "generic_word"
        if (
            any(generic in term for generic in ZH_GENERIC_WORD_TERMS)
            and not has_offensive_hint(term)
            and candidate.hate_precision <= 0.7
        ):
            return "generic_phrase"
        if candidate.track == "ngram_backoff" and candidate.total_count <= 1:
            return "singleton_ngram"
        if candidate.track == "ngram_backoff" and any(marker in term for marker in ZH_INTERNAL_FRAGMENT_MARKERS):
            return "broken_fragment"
        if term in ZH_GENERIC_GROUP_TERMS and candidate.track == "group_reference":
            return "neutral_group_reference"
        if len(term) <= 3 and (term[0] in ZH_BAD_PREFIXES or term[-1] in ZH_BAD_SUFFIXES):
            return "broken_fragment"
    return ""


def apply_substring_suppression(candidates: list[CandidateStats]) -> None:
    substring_parent: dict[str, CandidateStats] = {}
    parents = sorted(
        (
            candidate
            for candidate in candidates
            if contains_cjk(candidate.term)
            and len(candidate.term) > 2
            and not is_sentence_like_zh(candidate.term)
        ),
        key=lambda item: (item.total_count, item.hate_count, item.score, -len(item.term)),
        reverse=True,
    )
    for parent in parents:
        term = parent.term
        for start in range(len(term)):
            for end in range(start + 2, len(term) + 1):
                sub = term[start:end]
                if sub == term:
                    continue
                current = substring_parent.get(sub)
                if current is None or parent_rank_for_substring(parent) > parent_rank_for_substring(current):
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
        if best_parent.hate_count < max(1, candidate.hate_count * 0.25):
            continue
        if best_parent.hate_precision + 0.15 < candidate.hate_precision:
            continue
        overlap_strength = min(1.0, best_parent.hate_count / max(candidate.hate_count, 1))
        candidate.substring_of = best_parent.term
        candidate.bad_substring_penalty = 1.5 + 3.0 * overlap_strength


def parent_rank_for_substring(candidate: CandidateStats) -> tuple[float, int, int, int]:
    return (
        candidate.total_count,
        candidate.hate_count,
        candidate.score,
        -len(candidate.term),
    )


def select_candidates(corpus: CorpusCandidateResult, settings: dict[str, Any]) -> list[CandidateStats]:
    max_candidates = int(settings.get("max_candidates", 300) or 300)
    min_count = int(settings.get("min_count_for_llm", 1) or 1)
    min_hate = int(settings.get("min_hate_count_for_llm", 1) or 1)
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
        if candidate.hate_count < min_hate and candidate.annotation_count == 0:
            continue
        if candidate.reject_hint in suppressed_hints:
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def build_search_queries(candidate: CandidateStats) -> list[str]:
    term = candidate.term
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


def render_stage_prompt(stage: str, payload: dict[str, Any]) -> str:
    if stage == "context_judge":
        instructions = (
            "Judge whether the candidate term functions as discriminatory, hateful, offensive, "
            "or coded group-reference language inside the provided dataset examples. "
            "Use only these examples, not any existing lexicon. Return JSON with keys: "
            "supported, category, categories, confidence, reason."
        )
    elif stage == "web_evidence_judge":
        instructions = (
            "Judge whether the web evidence supports this candidate as a hate/offensive/coded "
            "group-reference lexicon term. Return JSON with keys: supported, confidence, "
            "reason, evidence_ids."
        )
    elif stage == "final_lexicon_judge":
        instructions = (
            "Make the final inclusion decision for a research lexicon. Combine dataset evidence "
            "and web evidence. Return JSON with keys exactly: include, category, categories, "
            "definition, nonhateful_meaning, variants, confidence, reason, evidence_ids. "
            "Definitions should explain the hateful/offensive usage, not endorse it."
        )
    else:
        raise ValueError(f"Unknown LLM stage: {stage}")
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
) -> JudgementClient:
    backend = str(settings.get("backend", "openai_compatible") or "openai_compatible").lower()
    if backend == "disabled":
        return DisabledJudgementClient()
    if backend == "openai_compatible":
        return OpenAICompatibleJudgementClient(settings, debug_recorder=debug_recorder)
    if backend == "deepseek":
        deepseek_settings = dict(settings)
        deepseek_settings["provider"] = "deepseek"
        return OpenAICompatibleJudgementClient(deepseek_settings, debug_recorder=debug_recorder)
    raise ValueError(f"Unsupported llm_settings.backend: {backend}")


def create_web_searcher(settings: dict[str, Any]) -> Any:
    backend = str(settings.get("backend", "disabled") or "disabled").lower()
    if backend == "disabled":
        return DisabledWebSearcher()
    return WebSearcher(settings)


def judge_candidate(
    candidate: CandidateStats,
    evidence: list[dict[str, Any]],
    judge_client: JudgementClient,
) -> dict[str, Any]:
    candidate_payload = candidate.to_payload()
    partial: dict[str, Any] = {"term": candidate.term}
    context_payload = {"candidate": candidate_payload}
    try:
        context_judge = judge_client.complete_json("context_judge", context_payload)
    except Exception as exc:
        raise CandidateJudgementError("context_judge", candidate, partial, exc) from exc
    partial["context_judge"] = context_judge

    web_payload = {"candidate": candidate_payload, "evidence": evidence}
    try:
        web_judge = judge_client.complete_json("web_evidence_judge", web_payload)
    except Exception as exc:
        raise CandidateJudgementError("web_evidence_judge", candidate, partial, exc) from exc
    partial["web_evidence_judge"] = web_judge

    final_payload = {
        "candidate": candidate_payload,
        "context_judge": context_judge,
        "web_evidence_judge": web_judge,
        "evidence": evidence,
    }
    try:
        final_judge = judge_client.complete_json("final_lexicon_judge", final_payload)
    except Exception as exc:
        raise CandidateJudgementError("final_lexicon_judge", candidate, partial, exc) from exc
    return {
        "term": candidate.term,
        "context_judge": context_judge,
        "web_evidence_judge": web_judge,
        "final_lexicon_judge": normalize_final_judge(final_judge, candidate),
    }


def failed_judgement_from_error(candidate: CandidateStats, error: CandidateJudgementError | Exception) -> dict[str, Any]:
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
    final = normalize_final_judge(
        {
            "include": False,
            "category": candidate.primary_category,
            "categories": [candidate.primary_category],
            "definition": "",
            "nonhateful_meaning": "",
            "variants": [],
            "confidence": 0,
            "reason": f"LLM judgement failed at stage={stage}; candidate rejected for safety.",
            "evidence_ids": [],
        },
        candidate,
    )
    return {
        "term": candidate.term,
        "context_judge": partial.get("context_judge"),
        "web_evidence_judge": partial.get("web_evidence_judge"),
        "final_lexicon_judge": final,
        "llm_error": error_payload,
    }


def normalize_final_judge(raw: dict[str, Any], candidate: CandidateStats) -> dict[str, Any]:
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
    return {
        "include": bool(raw.get("include", False)),
        "category": category or candidate.primary_category,
        "categories": categories or [category or candidate.primary_category],
        "definition": str(raw.get("definition", "") or "").strip(),
        "nonhateful_meaning": str(raw.get("nonhateful_meaning", "") or "").strip(),
        "variants": [str(item).strip() for item in variants if str(item).strip()],
        "confidence": max(0.0, min(confidence, 1.0)),
        "reason": str(raw.get("reason", "") or "").strip(),
        "evidence_ids": [str(item).strip() for item in evidence_ids if str(item).strip()],
    }


def should_include(candidate: CandidateStats, final: dict[str, Any], settings: dict[str, Any]) -> tuple[bool, str]:
    confidence = float(final.get("confidence", 0) or 0)
    threshold = float(settings.get("confidence_threshold", 0.65) or 0.65)
    single_threshold = float(settings.get("single_mention_confidence", 0.85) or 0.85)
    min_count = int(settings.get("min_count", 2) or 2)
    if not final.get("include"):
        return False, "LLM final decision excluded the term."
    if confidence < threshold:
        return False, f"confidence {confidence:.2f} < threshold {threshold:.2f}."
    if candidate.total_count < min_count:
        if not (candidate.annotation_count > 0 and confidence >= single_threshold):
            return False, "single mention lacks annotation support or high confidence."
    if candidate.non_hate_count > 0 and candidate.non_hate_count >= candidate.hate_count:
        if settings.get("ambiguous_requires_nonhateful_meaning", True) and not final.get("nonhateful_meaning"):
            return False, "ambiguous term lacks nonhateful_meaning."
        if candidate.hate_count <= candidate.non_hate_count:
            return False, "non-hate usage is not lower than hate/offensive usage."
    return True, "included"


def make_lexicon_entry(dataset: str, candidate: CandidateStats, final: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "builder": "llm_lexicon_builder",
        "dataset": dataset,
        "confidence": final["confidence"],
        "reason": final.get("reason", ""),
        "support": {
            "total_count": candidate.total_count,
            "hate_count": candidate.hate_count,
            "non_hate_count": candidate.non_hate_count,
            "annotation_count": candidate.annotation_count,
            "sample_ids": candidate.support_sample_ids,
        },
        "evidence_ids": final.get("evidence_ids", []),
        "source_counts": dict(candidate.source_counts),
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]], int, set[str]]:
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
            terms.append(make_lexicon_entry(dataset, candidate, final))
            continue
        existing_rejected = rejected_by_term.get(candidate.term)
        if existing_rejected:
            rejected.append(existing_rejected)
        else:
            rejected_row = {
                "rank": rank,
                "term": candidate.term,
                "reject_reason": reason,
                "candidate": candidate.to_payload(),
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
) -> dict[str, Any]:
    dataset = normalize_dataset(dataset)
    if dataset not in DATASET_DEFAULT_PATHS:
        raise ValueError(f"Unsupported dataset: {dataset}")

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
        settings=config.get("candidate_settings", {}),
        input_paths=input_paths,
        show_progress=show_progress,
    )
    logger.info(
        "Candidate mining complete: raw_candidates=%s hate_records=%s non_hate_records=%s",
        len(corpus.candidates),
        corpus.hate_records,
        corpus.non_hate_records,
    )
    selected = select_candidates(corpus, config.get("candidate_settings", {}))
    logger.info("Selected %s candidate(s) for web evidence and LLM judging.", len(selected))
    write_jsonl(
        [
            {"rank": rank + 1, **candidate.to_payload()}
            for rank, candidate in enumerate(selected)
        ],
        output_dir / "candidates.jsonl",
    )
    logger.info("Wrote candidate audit file: %s", output_dir / "candidates.jsonl")

    if judge_client is None:
        logger.info("Creating LLM judgement backend: %s", config.get("llm_settings", {}).get("backend", "openai_compatible"))
        judge_client = create_judgement_client(config.get("llm_settings", {}), debug_recorder=debug_recorder)
    if web_searcher is None:
        logger.info("Creating web search backend: %s", config.get("web_settings", {}).get("backend", "disabled"))
        web_searcher = create_web_searcher(config.get("web_settings", {}))

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
            inclusion_settings=config.get("inclusion", {}),
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
                    evidence = collect_web_evidence(candidate, web_searcher, debug_recorder=debug_recorder)
                    append_jsonl(
                        {
                            "rank": rank,
                            "term": candidate.term,
                            "queries": build_search_queries(candidate),
                            "evidence": evidence,
                        },
                        evidence_path,
                    )
            evidence_by_term[candidate.term] = evidence

            try:
                judgement = judge_candidate(candidate, evidence, judge_client)
                consecutive_llm_failures = 0
            except Exception as exc:
                if llm_failure_policy == "raise":
                    raise
                llm_failure_count += 1
                consecutive_llm_failures += 1
                error = exc if isinstance(exc, CandidateJudgementError) else CandidateJudgementError("unknown", candidate, {"term": candidate.term}, exc)
                judgement = failed_judgement_from_error(candidate, error)
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
            append_jsonl({"rank": rank, **judgement}, judgement_path)
            final = judgement["final_lexicon_judge"]
            include, reason = should_include(candidate, final, config.get("inclusion", {}))
            if judgement.get("llm_error"):
                reason = f"llm_error:{judgement['llm_error'].get('stage', 'unknown')}"
            if include:
                terms.append(make_lexicon_entry(dataset, candidate, final))
                logger.debug("Included term: %s confidence=%.2f", candidate.term, final.get("confidence", 0))
            else:
                rejected_row = {
                    "rank": rank,
                    "term": candidate.term,
                    "reject_reason": reason,
                    "candidate": candidate.to_payload(),
                    "final_lexicon_judge": final,
                }
                if judgement.get("llm_error"):
                    rejected_row["llm_error"] = judgement["llm_error"]
                rejected.append(rejected_row)
                append_jsonl(rejected_row, rejected_path)
                logger.debug("Rejected term: %s reason=%s", candidate.term, reason)
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
    finally:
        if progress_bar is not None:
            progress_bar.close()

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

    return {
        "dataset": dataset,
        "output_dir": str(output_dir),
        "lexicon_path": str(output_dir / "lexicon.json"),
        "total_records": corpus.total_records,
        "candidate_count": len(selected),
        "included_count": len(terms),
        "rejected_count": len(rejected),
        "llm_failure_count": llm_failure_count,
        "terms": terms,
        "evidence_by_term": evidence_by_term,
    }


def collect_web_evidence(
    candidate: CandidateStats,
    web_searcher: Any,
    debug_recorder: DebugRecorder | None = None,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for query in build_search_queries(candidate):
        try:
            query_results = web_searcher.search(query)
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
    common = {
        "source": "llm_lexicon_builder",
        "dataset": dataset,
        "input_paths": input_paths,
        "build_config_sha1": sha1_json(config),
        "total_terms": len(terms),
        "terms": terms,
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
        f"- Hate/offensive records: {corpus.hate_records}",
        f"- Non-hate/normal records: {corpus.non_hate_records}",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = normalize_dataset(args.dataset)
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
    result = build_lexicon(dataset, config)
    print(json.dumps({key: value for key, value in result.items() if key != "terms"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
