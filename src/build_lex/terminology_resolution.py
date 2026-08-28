"""Conditional explanation/Web/human resolution for terminology candidates.

This module consumes an approved full span frame.  It never performs live Web
access during downstream formal inference: all provider calls happen while an
offline, versioned terminology library is being built.
"""

from __future__ import annotations

import hashlib
import html.parser
import ipaddress
import json
import math
import os
import re
import shutil
import socket
import sqlite3
import tempfile
import time
import ctypes
import ctypes.util
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit

from build_lex.terminology_span_pipeline import (
    GATE_SCHEMA_VERSION,
    RESOURCE_ROLE,
    TASK_FIELD_KEYS,
    TerminologySpanError,
    _canonical_sha,
    _forbidden_key_paths,
    _sampling_key,
    clopper_pearson_interval,
    validate_full_audit_decision,
    validate_full_span_frame,
    validate_pilot_decision,
)
from build_lex.web_search import WebSearcher
from data.training_artifacts import (
    build_payload_manifest,
    load_json,
    load_jsonl,
    sha256_file,
    validate_payload_manifest,
    write_canonical_json,
    write_canonical_jsonl,
)
from rag.types import stable_term_evidence_id


RESOLUTION_SCHEMA_VERSION = "terminology-resolution/v1"
RESOLUTION_CHECKPOINT_VERSION = "terminology-resolution-checkpoint/v1"
RESOLUTION_EXECUTION_CONTRACT_VERSION = (
    "terminology-resolution-execution-contract/v1"
)
RESOLUTION_GATE_KIND = "terminology-resolution"
RESOLUTION_GOLD_VERSION = "terminology-resolution-gold/v1"
HUMAN_RESOLUTION_VERSION = "human-terminology-resolution/v1"
MERGED_RESOLUTION_VERSION = "merged-terminology-resolution/v1"
LIBRARY_SCHEMA_VERSION = "terminology-understanding-library/v1"
LIBRARY_MANIFEST_VERSION = "terminology-understanding-library-manifest/v1"
STAGE1_PUBLICATION_MANIFEST_VERSION = (
    "stage1-terminology-lifecycle-lexicon-manifest/v1"
)
STAGE1_PUBLICATION_PROVENANCE_VERSION = (
    "stage1-terminology-lifecycle-lexicon-provenance/v1"
)
STAGE1_PUBLICATION_POLICY = "pilot-gated-terminology-lifecycle/v1"
STAGE1_LEXICON_SCHEMA_VERSION = "stage1-train-only-terminology-library/v1"
STAGE1_LOCATOR_VERSION = "stage1-locator-ref/v1"

ENTRY_KEYS = frozenset(
    {"term", "definition", "usage_notes", "ambiguity_notes", "variants"}
)
REVIEW_KEYS = frozenset(
    {
        "supported",
        "context_consistent",
        "material_conflict",
        "independent_definition",
        "reason",
    }
)
WEB_REVIEW_KEYS = frozenset(
    {"supported", "material_conflict", "evidence_ids", "reason"}
)


class TerminologyResolutionError(TerminologySpanError):
    """Raised when offline terminology resolution cannot safely continue."""


class JsonProvider(Protocol):
    def complete_json(self, stage: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        ...


class SearchProvider(Protocol):
    def search(self, query: str) -> list[dict[str, str]]:
        ...


class PageProvider(Protocol):
    def fetch(self, url: str) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    model: str
    api_base: str
    api_key: str
    timeout: int = 180
    max_attempts: int = 3
    max_tokens: int = 1024

    def public_dict(self) -> dict[str, Any]:
        parsed = urlsplit(self.api_base)
        port = f":{parsed.port}" if parsed.port else ""
        return {
            "provider": self.provider,
            "model": self.model,
            "api_origin": f"{parsed.scheme}://{parsed.hostname}{port}",
            "timeout": self.timeout,
            "max_attempts": self.max_attempts,
            "max_tokens": self.max_tokens,
        }


_STAGE_INSTRUCTIONS = {
    "qwen_self_explanation": """根据给出的术语及 fit-only 原文上下文，生成无类别术语解释。不要判断仇恨、目标群体或任务类别。只输出 JSON，键必须恰为 term、definition、usage_notes、ambiguity_notes、variants。term 必须与输入术语完全一致；其余文字使用中性中文。""",
    "deepseek_self_review": """独立检查候选术语解释是否与给出的原文上下文一致。不要依据任务类别，不要输出类别。只输出 JSON，键必须恰为 supported、context_consistent、material_conflict、independent_definition、reason。前三项为布尔值；独立给出简短词义用于语义一致性比较。""",
    "qwen_web_rewrite": """仅根据给出的冻结网页证据，为术语重写无类别解释。不要使用证据之外的断言，不要输出任务类别。只输出 JSON，键必须恰为 term、definition、usage_notes、ambiguity_notes、variants。""",
    "deepseek_web_review": """检查解释是否被给出的网页证据直接支持。只输出 JSON，键必须恰为 supported、material_conflict、evidence_ids、reason；evidence_ids 只能引用输入中的证据 ID。不要输出任务类别。""",
}


def _implementation_name(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _public_provider_contract(value: Any) -> dict[str, Any]:
    settings = getattr(value, "settings", None)
    if isinstance(settings, ProviderSettings):
        return settings.public_dict()
    return {"implementation": _implementation_name(value)}


def _sanitize_public_settings(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, inner in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in {"api_key", "token", "password", "secret"} or lowered.endswith(
                ("_token", "_password", "_secret")
            ):
                continue
            result[key] = _sanitize_public_settings(inner)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_public_settings(inner) for inner in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def build_resolution_execution_contract(
    *,
    config: Mapping[str, Any],
    qwen: Any,
    deepseek: Any,
    searcher: Any,
    similarity: Any,
    web_settings: Mapping[str, Any] | None = None,
    attempt_budgets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze every model/prompt/search/BGE input used by resolution."""

    resolution = config["resolution"]
    _frozen_libpsl(str(resolution["public_suffix_sha1"]))
    contract = {
        "schema_version": RESOLUTION_EXECUTION_CONTRACT_VERSION,
        "pipeline_config_sha256": _canonical_sha(config),
        "qwen": _public_provider_contract(qwen),
        "deepseek": _public_provider_contract(deepseek),
        "search": {
            "implementation": _implementation_name(searcher),
            "settings": _sanitize_public_settings(web_settings or {}),
        },
        "similarity": {
            "implementation": "build_lex.terminology_resolution.BgeSimilarity",
            "logical_model_path": resolution["bge_model_path"],
            "model_file_tree_sha256": resolution[
                "bge_model_file_tree_sha256"
            ],
            "minimum_similarity": resolution["minimum_bge_similarity"],
            "feature_only": resolution["bge_similarity_is_feature_only"],
        },
        "public_suffix": {
            "policy": resolution["public_suffix_policy"],
            "sha1": resolution["public_suffix_sha1"],
        },
        "attempt_budgets": dict(
            attempt_budgets
            or {
                "search": 3100,
                "deepseek": 9000,
            }
        ),
        "stage_instruction_sha256s": {
            name: hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            for name, instruction in sorted(_STAGE_INSTRUCTIONS.items())
        },
        "adapter_code_sha256s": {
            "terminology_resolution.py": sha256_file(Path(__file__)),
            "web_search.py": sha256_file(Path(__file__).with_name("web_search.py")),
        },
    }
    if _forbidden_key_paths(contract):
        raise TerminologyResolutionError(
            "resolution execution contract contains task-category fields"
        )
    contract["contract_sha256"] = _canonical_sha(contract)
    return contract


def validate_resolution_execution_contract(
    value: Mapping[str, Any], *, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    expected_instruction_hashes = {
        name: hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        for name, instruction in sorted(_STAGE_INSTRUCTIONS.items())
    }
    expected_code_hashes = {
        "terminology_resolution.py": sha256_file(Path(__file__)),
        "web_search.py": sha256_file(Path(__file__).with_name("web_search.py")),
    }
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema_version",
            "pipeline_config_sha256",
            "qwen",
            "deepseek",
            "search",
            "similarity",
            "public_suffix",
            "attempt_budgets",
            "stage_instruction_sha256s",
            "adapter_code_sha256s",
            "contract_sha256",
        }
        or value.get("schema_version") != RESOLUTION_EXECUTION_CONTRACT_VERSION
        or value.get("contract_sha256")
        != _canonical_sha(
            {key: inner for key, inner in value.items() if key != "contract_sha256"}
        )
        or value.get("stage_instruction_sha256s") != expected_instruction_hashes
        or value.get("adapter_code_sha256s") != expected_code_hashes
        or not isinstance(value.get("qwen"), Mapping)
        or not isinstance(value.get("deepseek"), Mapping)
        or not isinstance(value.get("search"), Mapping)
        or not isinstance(value.get("attempt_budgets"), Mapping)
        or set(value.get("attempt_budgets", {})) != {"search", "deepseek"}
        or any(
            isinstance(cap, bool) or not isinstance(cap, int) or cap < 1
            for cap in value.get("attempt_budgets", {}).values()
        )
        or _forbidden_key_paths(value)
    ):
        raise TerminologyResolutionError("resolution execution contract is invalid")
    similarity = value.get("similarity")
    suffix = value.get("public_suffix")
    if (
        not isinstance(similarity, Mapping)
        or similarity.get("feature_only") is not True
        or not isinstance(similarity.get("model_file_tree_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(similarity.get("model_file_tree_sha256"))
        )
        is None
        or not isinstance(suffix, Mapping)
        or re.fullmatch(r"[0-9a-f]{40}", str(suffix.get("sha1"))) is None
    ):
        raise TerminologyResolutionError(
            "resolution execution resources differ from config"
        )
    if config is not None and (
        value.get("pipeline_config_sha256") != _canonical_sha(config)
        or similarity.get("logical_model_path")
        != config["resolution"]["bge_model_path"]
        or similarity.get("model_file_tree_sha256")
        != config["resolution"]["bge_model_file_tree_sha256"]
        or similarity.get("minimum_similarity")
        != config["resolution"]["minimum_bge_similarity"]
        or suffix
        != {
            "policy": config["resolution"]["public_suffix_policy"],
            "sha1": config["resolution"]["public_suffix_sha1"],
        }
    ):
        raise TerminologyResolutionError(
            "resolution execution contract differs from config"
        )
    return dict(value)


class OpenAIJsonProvider:
    """Small OpenAI-compatible JSON client for local Qwen or DeepSeek."""

    def __init__(self, settings: ProviderSettings) -> None:
        parsed = urlsplit(settings.api_base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise TerminologyResolutionError("provider api_base is malformed")
        if settings.max_attempts < 1 or settings.max_attempts > 3:
            raise TerminologyResolutionError("provider max attempts must be in [1,3]")
        self.settings = settings

    def _url(self) -> str:
        return self.settings.api_base.rstrip("/") + "/chat/completions"

    def complete_json(self, stage: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        instruction = _STAGE_INSTRUCTIONS.get(stage)
        if instruction is None:
            raise TerminologyResolutionError(f"unknown resolution stage: {stage}")
        if _forbidden_key_paths(payload):
            raise TerminologyResolutionError("resolution provider payload contains task fields")
        request = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ],
            "temperature": 0,
            "stream": False,
            "max_tokens": self.settings.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.settings.provider == "qwen":
            request["chat_template_kwargs"] = {
                "enable_thinking": False,
                "preserve_thinking": False,
            }
        elif self.settings.provider == "deepseek":
            request["thinking"] = {"type": "disabled"}
        else:
            raise TerminologyResolutionError("unsupported resolution provider")
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_attempts + 1):
            try:
                import requests

                response = requests.post(
                    self._url(),
                    json=request,
                    headers={
                        "Authorization": f"Bearer {self.settings.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.settings.timeout,
                    allow_redirects=False,
                )
                if response.status_code != 200:
                    raise TerminologyResolutionError(
                        f"provider HTTP {response.status_code}"
                    )
                wire = response.json()
                choices = wire.get("choices") if isinstance(wire, Mapping) else None
                message = choices[0].get("message") if isinstance(choices, list) and choices else None
                content = message.get("content") if isinstance(message, Mapping) else None
                if not isinstance(content, str) or not content.strip():
                    raise TerminologyResolutionError("provider returned empty JSON content")
                value = content.strip()
                if value.startswith("```json"):
                    value = value[7:]
                elif value.startswith("```"):
                    value = value[3:]
                if value.endswith("```"):
                    value = value[:-3]
                try:
                    parsed = json.loads(value.strip())
                except json.JSONDecodeError:
                    start, end = value.find("{"), value.rfind("}")
                    parsed = json.loads(value[start : end + 1])
                if not isinstance(parsed, dict):
                    raise TerminologyResolutionError("provider JSON is not an object")
                return parsed
            except Exception as exc:
                last_error = exc
                if attempt < self.settings.max_attempts:
                    time.sleep(min(4.0, float(attempt)))
        raise TerminologyResolutionError(
            f"resolution provider failed at {stage}: {last_error}"
        ) from last_error


class _TextExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth and data.strip():
            self.parts.append(data.strip())


def _public_host(hostname: str) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except OSError:
        return False
    if not addresses:
        return False
    for address in addresses:
        value = ipaddress.ip_address(address[4][0])
        if (
            value.is_private
            or value.is_loopback
            or value.is_link_local
            or value.is_multicast
            or value.is_reserved
            or value.is_unspecified
        ):
            return False
    return True


class SafePageFetcher:
    """Bounded, no-redirect page fetcher that rejects private network targets."""

    def __init__(self, *, timeout: int = 20, maximum_bytes: int = 250_000) -> None:
        self.timeout = timeout
        self.maximum_bytes = maximum_bytes

    def fetch(self, url: str) -> dict[str, Any]:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 80, 443}
            or not _public_host(parsed.hostname)
        ):
            raise TerminologyResolutionError("refusing unsafe evidence URL")
        import requests

        response = requests.get(
            url,
            headers={"User-Agent": "WP3TerminologyResearch/1.0"},
            timeout=self.timeout,
            allow_redirects=False,
            stream=True,
        )
        if response.status_code != 200:
            raise TerminologyResolutionError(
                f"evidence page HTTP {response.status_code}"
            )
        content_type = response.headers.get("content-type", "").lower()
        if not (content_type.startswith("text/html") or content_type.startswith("text/plain")):
            raise TerminologyResolutionError("evidence page is not text")
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=16_384):
            size += len(chunk)
            if size > self.maximum_bytes:
                raise TerminologyResolutionError("evidence page exceeds byte limit")
            chunks.append(chunk)
        payload = b"".join(chunks)
        encoding = response.encoding or "utf-8"
        text = payload.decode(encoding, errors="replace")
        if content_type.startswith("text/html"):
            extractor = _TextExtractor()
            extractor.feed(text)
            text = "\n".join(extractor.parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return {
            "url": url,
            "content_sha256": hashlib.sha256(payload).hexdigest(),
            "text_excerpt": text[:4000],
            "content_type": content_type.split(";", 1)[0],
            "byte_count": len(payload),
        }


_LIBPSL_CACHE: dict[str, tuple[Any, int]] = {}


def _frozen_libpsl(expected_sha1: str) -> tuple[Any, int]:
    cached = _LIBPSL_CACHE.get(expected_sha1)
    if cached is not None:
        return cached
    library_name = ctypes.util.find_library("psl")
    if not library_name:
        raise TerminologyResolutionError(
            "libpsl is required for frozen eTLD+1 resolution"
        )
    library = ctypes.CDLL(library_name)
    library.psl_builtin.restype = ctypes.c_void_p
    library.psl_builtin_sha1sum.restype = ctypes.c_char_p
    library.psl_registrable_domain.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    library.psl_registrable_domain.restype = ctypes.c_char_p
    observed_raw = library.psl_builtin_sha1sum()
    observed = observed_raw.decode("ascii") if observed_raw else ""
    if observed != expected_sha1:
        raise TerminologyResolutionError(
            f"frozen public-suffix list differs: {observed or 'missing'}"
        )
    context = int(library.psl_builtin() or 0)
    if not context:
        raise TerminologyResolutionError("libpsl has no built-in suffix context")
    result = (library, context)
    _LIBPSL_CACHE[expected_sha1] = result
    return result


def _registrable_domain(url: str, *, expected_psl_sha1: str) -> str:
    host = (urlsplit(url).hostname or "").lower().strip(".")
    if not host:
        return ""
    try:
        ascii_host = host.encode("idna")
    except UnicodeError as exc:
        raise TerminologyResolutionError("evidence hostname is invalid") from exc
    library, context = _frozen_libpsl(expected_psl_sha1)
    value = library.psl_registrable_domain(context, ascii_host)
    return value.decode("ascii").lower() if value else host


class ResolutionCheckpoint:
    """Branch-aware attempt ledger with request-hash replay protection."""

    def __init__(
        self,
        path: str | Path,
        *,
        binding: Mapping[str, Any],
        attempt_budgets: Mapping[str, int],
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.binding = dict(binding)
        self.binding_sha256 = _canonical_sha(binding)
        self.attempt_budgets = {
            str(key): int(value) for key, value in attempt_budgets.items()
        }
        if set(self.attempt_budgets) != {"search", "deepseek"} or any(
            value < 1 for value in self.attempt_budgets.values()
        ):
            raise TerminologyResolutionError(
                "resolution provider attempt budgets are invalid"
            )
        old_umask = os.umask(0o077)
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS attempts(
                      candidate_id TEXT NOT NULL, stage TEXT NOT NULL, attempt INTEGER NOT NULL,
                      request_sha256 TEXT NOT NULL, status TEXT NOT NULL, response_json TEXT, error TEXT,
                      PRIMARY KEY(candidate_id,stage,attempt));
                    CREATE TABLE IF NOT EXISTS successes(
                      candidate_id TEXT NOT NULL, stage TEXT NOT NULL, request_sha256 TEXT NOT NULL,
                      response_json TEXT NOT NULL, PRIMARY KEY(candidate_id,stage));
                    """
                )
                expected = {
                    "schema_version": RESOLUTION_CHECKPOINT_VERSION,
                    "binding": json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    "binding_sha256": self.binding_sha256,
                    "attempt_budgets": json.dumps(
                        self.attempt_budgets,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
                for key, value in expected.items():
                    row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
                    if row is None:
                        connection.execute("INSERT INTO meta(key,value) VALUES(?,?)", (key, value))
                    elif row["value"] != value:
                        raise TerminologyResolutionError(f"resolution checkpoint drift: {key}")
            os.chmod(self.path, 0o600)
        finally:
            os.umask(old_umask)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def get(self, candidate_id: str, stage: str, request: Mapping[str, Any]) -> Any | None:
        request_sha = _canonical_sha(request)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT request_sha256,response_json FROM successes WHERE candidate_id=? AND stage=?",
                (candidate_id, stage),
            ).fetchone()
        if row is None:
            return None
        if row["request_sha256"] != request_sha:
            raise TerminologyResolutionError("cached resolution request changed")
        return json.loads(row["response_json"])

    def execute(
        self,
        candidate_id: str,
        stage: str,
        request: Mapping[str, Any],
        operation: Callable[[], Any],
        *,
        max_attempts: int = 3,
    ) -> Any:
        cached = self.get(candidate_id, stage, request)
        if cached is not None:
            return cached
        request_sha = _canonical_sha(request)
        last_error: Exception | None = None
        while True:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                budget_name = (
                    "search"
                    if stage.startswith("search_")
                    else "deepseek"
                    if stage.startswith("deepseek_")
                    else None
                )
                if budget_name is not None:
                    used = int(
                        connection.execute(
                            "SELECT COUNT(*) AS count FROM attempts WHERE stage LIKE ?",
                            ("search_%" if budget_name == "search" else "deepseek_%",),
                        ).fetchone()["count"]
                    )
                    if used >= self.attempt_budgets[budget_name]:
                        connection.rollback()
                        raise TerminologyResolutionError(
                            f"resolution {budget_name} physical-attempt cap exhausted"
                        )
                rows = connection.execute(
                    "SELECT attempt,request_sha256 FROM attempts WHERE candidate_id=? AND stage=? ORDER BY attempt",
                    (candidate_id, stage),
                ).fetchall()
                if any(row["request_sha256"] != request_sha for row in rows):
                    connection.rollback()
                    raise TerminologyResolutionError("resolution request changed across attempts")
                attempt = len(rows) + 1
                if attempt > max_attempts:
                    connection.rollback()
                    raise TerminologyResolutionError(
                        f"resolution attempt budget exhausted: {candidate_id}/{stage}"
                    )
                connection.execute(
                    "INSERT INTO attempts(candidate_id,stage,attempt,request_sha256,status) VALUES(?,?,?,?,'reserved')",
                    (candidate_id, stage, attempt, request_sha),
                )
                connection.commit()
            try:
                response = operation()
            except Exception as exc:
                last_error = exc
                with self._connect() as connection:
                    connection.execute(
                        "UPDATE attempts SET status='failure',error=? WHERE candidate_id=? AND stage=? AND attempt=?",
                        (str(exc)[:1000], candidate_id, stage, attempt),
                    )
                if attempt >= max_attempts:
                    raise TerminologyResolutionError(
                        f"resolution stage failed: {candidate_id}/{stage}: {last_error}"
                    ) from last_error
                continue
            wire = json.dumps(response, ensure_ascii=False, sort_keys=True)
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE attempts SET status='success',response_json=? WHERE candidate_id=? AND stage=? AND attempt=?",
                    (wire, candidate_id, stage, attempt),
                )
                connection.execute(
                    "INSERT INTO successes(candidate_id,stage,request_sha256,response_json) VALUES(?,?,?,?)",
                    (candidate_id, stage, request_sha, wire),
                )
                connection.commit()
            return response


def normalize_entry(value: Any, *, expected_term: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != ENTRY_KEYS:
        raise TerminologyResolutionError("terminology entry fields are not canonical")
    if _forbidden_key_paths(value):
        raise TerminologyResolutionError("terminology entry contains task-category fields")
    term = value.get("term")
    definition = value.get("definition")
    usage = value.get("usage_notes")
    ambiguity = value.get("ambiguity_notes")
    variants = value.get("variants")
    if term != expected_term or not isinstance(definition, str) or not definition.strip():
        raise TerminologyResolutionError("terminology entry term/definition is invalid")
    if (
        not isinstance(usage, str)
        or not isinstance(ambiguity, str)
        or not isinstance(variants, list)
        or any(not isinstance(item, str) or not item.strip() for item in variants)
    ):
        raise TerminologyResolutionError("terminology entry notes/variants are invalid")
    normalized_variants = [item.strip() for item in variants if item.strip() != term]
    if len(normalized_variants) != len(set(normalized_variants)):
        raise TerminologyResolutionError("terminology entry variants are duplicated")
    return {
        "term": term,
        "definition": definition.strip(),
        "usage_notes": usage.strip(),
        "ambiguity_notes": ambiguity.strip(),
        "variants": normalized_variants,
    }


def normalize_self_review(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != REVIEW_KEYS or _forbidden_key_paths(value):
        raise TerminologyResolutionError("self-review response is invalid")
    if any(not isinstance(value.get(key), bool) for key in ("supported", "context_consistent", "material_conflict")):
        raise TerminologyResolutionError("self-review booleans are invalid")
    if not isinstance(value.get("independent_definition"), str) or not value["independent_definition"].strip() or not isinstance(value.get("reason"), str):
        raise TerminologyResolutionError("self-review text is invalid")
    return dict(value)


def normalize_web_review(value: Any, *, allowed_ids: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != WEB_REVIEW_KEYS or _forbidden_key_paths(value):
        raise TerminologyResolutionError("Web-review response is invalid")
    if not isinstance(value.get("supported"), bool) or not isinstance(value.get("material_conflict"), bool):
        raise TerminologyResolutionError("Web-review booleans are invalid")
    evidence_ids = value.get("evidence_ids")
    if (
        not isinstance(evidence_ids, list)
        or any(item not in allowed_ids for item in evidence_ids)
        or len(evidence_ids) != len(set(evidence_ids))
        or (value["supported"] and not evidence_ids)
        or not isinstance(value.get("reason"), str)
    ):
        raise TerminologyResolutionError("Web-review citations are invalid")
    return dict(value)


def _bigram_similarity(left: str, right: str) -> float:
    def grams(value: str) -> Counter[str]:
        normalized = re.sub(r"\s+", "", value.casefold())
        return Counter(normalized[index : index + 2] for index in range(max(0, len(normalized) - 1)))

    a, b = grams(left), grams(right)
    if not a and not b:
        return 1.0
    overlap = sum((a & b).values())
    return 2.0 * overlap / (sum(a.values()) + sum(b.values())) if a and b else 0.0


def bge_similarity(left: str, right: str, *, model_path: str | Path) -> float:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise TerminologyResolutionError("sentence-transformers is required for BGE similarity") from exc
    model = SentenceTransformer(str(model_path), local_files_only=True)
    embeddings = model.encode([left, right], normalize_embeddings=True)
    return float(embeddings[0] @ embeddings[1])


class BgeSimilarity:
    """Load the frozen local BGE model once and expose a pair scorer."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        expected_file_tree_sha256: str | None = None,
    ) -> None:
        if expected_file_tree_sha256 is not None:
            from data.build_context_manifest import (
                embedding_model_file_tree_sha256,
            )

            observed = embedding_model_file_tree_sha256(model_path)
            if observed != expected_file_tree_sha256:
                raise TerminologyResolutionError(
                    "frozen BGE model file tree differs"
                )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise TerminologyResolutionError(
                "sentence-transformers is required for BGE similarity"
            ) from exc
        self.model = SentenceTransformer(str(model_path), local_files_only=True)

    def __call__(self, left: str, right: str) -> float:
        embeddings = self.model.encode([left, right], normalize_embeddings=True)
        return float(embeddings[0] @ embeddings[1])


def calibrate_resolution_gate(
    *,
    gold_path: str | Path,
    execution_contract_path: str | Path,
    span_frame_id: str,
    config: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Calibrate self/Web automatic branches from a single-reviewer Gold file."""

    gold = load_json(gold_path)
    execution_contract = validate_resolution_execution_contract(
        load_json(execution_contract_path), config=config
    )
    if (
        not isinstance(gold, Mapping)
        or set(gold)
        != {
            "schema_version",
            "span_frame_id",
            "execution_contract_sha256",
            "reviewer_id",
            "cases",
        }
        or gold.get("schema_version") != RESOLUTION_GOLD_VERSION
        or gold.get("span_frame_id") != span_frame_id
        or gold.get("execution_contract_sha256")
        != execution_contract["contract_sha256"]
        or not isinstance(gold.get("reviewer_id"), str)
        or not gold["reviewer_id"].strip()
        or not isinstance(gold.get("cases"), list)
        or _forbidden_key_paths(gold)
    ):
        raise TerminologyResolutionError("resolution Gold document is invalid")
    allowed_branches = {"self_explanation", "web_evidence"}
    case_ids: set[str] = set()
    for row in gold["cases"]:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"case_id", "branch", "auto_accept", "human_acceptable", "notes"}
            or not isinstance(row.get("case_id"), str)
            or not row["case_id"].strip()
            or row["case_id"] in case_ids
            or row.get("branch") not in allowed_branches
            or not isinstance(row.get("auto_accept"), bool)
            or not isinstance(row.get("human_acceptable"), bool)
            or not isinstance(row.get("notes"), str)
        ):
            raise TerminologyResolutionError("resolution Gold case is invalid")
        case_ids.add(row["case_id"])
    branches: dict[str, Any] = {}
    confidence = float(config["gates"]["confidence_level"])
    maximum = float(config["gates"]["max_auto_error_ci_upper"])
    for branch in ("self_explanation", "web_evidence"):
        rows = [row for row in gold["cases"] if isinstance(row, Mapping) and row.get("branch") == branch]
        promoted = [row for row in rows if row["auto_accept"]]
        errors = sum(not row["human_acceptable"] for row in promoted)
        _, upper = clopper_pearson_interval(errors, len(promoted), confidence=confidence)
        branches[branch] = {
            "enabled": bool(promoted) and upper <= maximum,
            "reviewed_case_count": len(rows),
            "auto_promoted_count": len(promoted),
            "error_count": errors,
            "error_rate": errors / len(promoted) if promoted else None,
            "error_ci_upper": upper,
            "maximum_error_ci_upper": maximum,
        }
    gate = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate_kind": RESOLUTION_GATE_KIND,
        "span_frame_id": span_frame_id,
        "config_sha256": _canonical_sha(config),
        "execution_contract_sha256": execution_contract["contract_sha256"],
        "gold_sha256": _canonical_sha(gold),
        "reviewer_id": gold["reviewer_id"],
        "minimum_bge_similarity": float(config["resolution"].get("minimum_bge_similarity", 0.70)),
        "branches": branches,
    }
    gate["gate_sha256"] = _canonical_sha(gate)
    write_canonical_json(output_path, gate)
    return gate


def validate_resolution_gate(
    path: str | Path,
    *,
    span_frame_id: str,
    config: Mapping[str, Any],
    execution_contract_sha256: str | None = None,
) -> dict[str, Any]:
    value = load_json(path)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "gate_kind",
            "span_frame_id",
            "config_sha256",
            "execution_contract_sha256",
            "gold_sha256",
            "reviewer_id",
            "minimum_bge_similarity",
            "branches",
            "gate_sha256",
        }
        or value.get("schema_version") != GATE_SCHEMA_VERSION
        or value.get("gate_kind") != RESOLUTION_GATE_KIND
        or value.get("span_frame_id") != span_frame_id
        or value.get("config_sha256") != _canonical_sha(config)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("execution_contract_sha256"))
        )
        is None
        or (
            execution_contract_sha256 is not None
            and value.get("execution_contract_sha256")
            != execution_contract_sha256
        )
        or _forbidden_key_paths(value)
    ):
        raise TerminologyResolutionError("resolution gate binding is invalid")
    declared = value.get("gate_sha256")
    unhashed = {key: inner for key, inner in value.items() if key != "gate_sha256"}
    if declared != _canonical_sha(unhashed):
        raise TerminologyResolutionError("resolution gate hash is invalid")
    branches = value.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != {"self_explanation", "web_evidence"}:
        raise TerminologyResolutionError("resolution gate branches are invalid")
    for branch in branches.values():
        if (
            not isinstance(branch, Mapping)
            or set(branch)
            != {
                "enabled",
                "reviewed_case_count",
                "auto_promoted_count",
                "error_count",
                "error_rate",
                "error_ci_upper",
                "maximum_error_ci_upper",
            }
            or not isinstance(branch.get("enabled"), bool)
            or any(
                isinstance(branch.get(key), bool)
                or not isinstance(branch.get(key), int)
                or branch[key] < 0
                for key in ("reviewed_case_count", "auto_promoted_count", "error_count")
            )
            or branch["auto_promoted_count"] > branch["reviewed_case_count"]
            or branch["error_count"] > branch["auto_promoted_count"]
            or branch["maximum_error_ci_upper"]
            != float(config["gates"]["max_auto_error_ci_upper"])
        ):
            raise TerminologyResolutionError("resolution gate branch is invalid")
        promoted = int(branch["auto_promoted_count"])
        errors = int(branch["error_count"])
        expected_rate = errors / promoted if promoted else None
        _, expected_upper = clopper_pearson_interval(
            errors,
            promoted,
            confidence=float(config["gates"]["confidence_level"]),
        )
        expected_enabled = bool(promoted) and expected_upper <= float(
            branch["maximum_error_ci_upper"]
        )
        if (
            branch.get("error_rate") != expected_rate
            or branch.get("error_ci_upper") != expected_upper
            or branch.get("enabled") != expected_enabled
        ):
            raise TerminologyResolutionError("resolution gate metrics differ")
        if branch["enabled"] and float(branch["error_ci_upper"]) > float(branch["maximum_error_ci_upper"]):
            raise TerminologyResolutionError("unsafe resolution branch was enabled")
    if value.get("minimum_bge_similarity") != float(
        config["resolution"].get("minimum_bge_similarity", 0.70)
    ):
        raise TerminologyResolutionError("resolution semantic threshold differs")
    return value


def _search_queries(term: str) -> list[str]:
    return [term, f'"{term}" 意思', f'"{term}" 网络用语']


def _web_evidence(
    *,
    candidate_id: str,
    term: str,
    searcher: SearchProvider,
    fetcher: PageProvider,
    checkpoint: ResolutionCheckpoint,
    maximum_queries: int,
    maximum_pages: int,
    expected_psl_sha1: str,
) -> list[dict[str, Any]]:
    search_rows: list[dict[str, Any]] = []
    for index, query in enumerate(_search_queries(term)[:maximum_queries], start=1):
        request = {"query": query, "slot": index, "term": term}
        results = checkpoint.execute(
            candidate_id,
            f"search_{index}",
            request,
            lambda query=query: searcher.search(query),
        )
        if isinstance(results, list):
            search_rows.extend(item for item in results if isinstance(item, Mapping))
    by_url: dict[str, dict[str, Any]] = {}
    for row in search_rows:
        url = str(row.get("url") or "").strip()
        if url and url not in by_url:
            by_url[url] = dict(row)
    domain_rows: list[tuple[str, dict[str, Any], str]] = []
    for url, row in by_url.items():
        domain = _registrable_domain(url, expected_psl_sha1=expected_psl_sha1)
        if domain:
            domain_rows.append((url, row, domain))
    selected_rows: list[tuple[str, dict[str, Any], str]] = []
    selected_urls: set[str] = set()
    selected_domains: set[str] = set()
    for row in domain_rows:
        if row[2] in selected_domains:
            continue
        selected_rows.append(row)
        selected_urls.add(row[0])
        selected_domains.add(row[2])
        if len(selected_rows) == maximum_pages:
            break
    if len(selected_rows) < maximum_pages:
        for row in domain_rows:
            if row[0] in selected_urls:
                continue
            selected_rows.append(row)
            selected_urls.add(row[0])
            if len(selected_rows) == maximum_pages:
                break
    evidence: list[dict[str, Any]] = []
    for index, (url, search_row, domain) in enumerate(selected_rows, start=1):
        request = {"url": url, "slot": index}
        try:
            page = checkpoint.execute(
                candidate_id,
                f"page_{index}",
                request,
                lambda url=url: fetcher.fetch(url),
            )
        except TerminologyResolutionError:
            continue
        evidence_id = "web-" + hashlib.sha256(
            f"{candidate_id}:{url}:{page['content_sha256']}".encode("utf-8")
        ).hexdigest()[:20]
        evidence.append(
            {
                "evidence_id": evidence_id,
                "url": url,
                "domain": domain,
                "title": str(search_row.get("title") or "")[:180],
                "search_snippet": str(search_row.get("snippet") or "")[:320],
                "content_sha256": page["content_sha256"],
                "text_excerpt": page["text_excerpt"],
            }
        )
    return evidence


def _sources_sufficient(evidence: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[bool, str]:
    registry = {
        str(item.get("domain", "")).lower()
        for item in config["resolution"].get("trusted_source_registry", [])
        if isinstance(item, Mapping) and item.get("domain")
    }
    domains = {str(item.get("domain", "")).lower() for item in evidence if item.get("domain")}
    if registry.intersection(domains):
        return True, "one_frozen_trusted_source"
    minimum = int(config["resolution"].get("minimum_independent_sources", 2))
    return (len(domains) >= minimum, "independent_domains" if len(domains) >= minimum else "insufficient_sources")


def _candidate_contexts(candidate: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for occurrence in candidate.get("sample_occurrences", [])[:5]:
        record = records.get(str(occurrence.get("record_id")))
        if record is None:
            continue
        contexts.append(
            {
                "content": record["content"],
                "surface": occurrence["surface"],
                "start": occurrence["start"],
                "end": occurrence["end"],
            }
        )
    return contexts


def run_resolution(
    *,
    span_frame_dir: str | Path,
    full_audit_decision_path: str | Path,
    resolution_gate_path: str | Path,
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    output_root: str | Path,
    qwen: JsonProvider,
    deepseek: JsonProvider,
    searcher: SearchProvider,
    fetcher: PageProvider,
    similarity: Callable[[str, str], float] | None = None,
    execution_contract: Mapping[str, Any] | None = None,
    limit: int | None = None,
) -> Path:
    """Run conditional self/Web resolution and freeze its exact human complement."""

    if similarity is None:
        raise TerminologyResolutionError(
            "the frozen BGE semantic feature is required for resolution"
        )
    frame = validate_full_span_frame(span_frame_dir)
    audit = validate_full_audit_decision(
        full_audit_decision_path,
        span_frame_id=frame["frame_id"],
        require_pass=True,
    )
    resolved_execution_contract = build_resolution_execution_contract(
        config=config,
        qwen=qwen,
        deepseek=deepseek,
        searcher=searcher,
        similarity=similarity,
    ) if execution_contract is None else dict(execution_contract)
    resolved_execution_contract = validate_resolution_execution_contract(
        resolved_execution_contract, config=config
    )
    gate = validate_resolution_gate(
        resolution_gate_path,
        span_frame_id=frame["frame_id"],
        config=config,
        execution_contract_sha256=resolved_execution_contract[
            "contract_sha256"
        ],
    )
    binding = {
        "schema_version": RESOLUTION_CHECKPOINT_VERSION,
        "span_frame_id": frame["frame_id"],
        "full_audit_decision_sha256": audit["decision_sha256"],
        "resolution_gate_sha256": gate["gate_sha256"],
        "config_sha256": _canonical_sha(config),
        "execution_contract_sha256": resolved_execution_contract[
            "contract_sha256"
        ],
    }
    checkpoint = ResolutionCheckpoint(
        checkpoint_path,
        binding=binding,
        attempt_budgets=resolved_execution_contract["attempt_budgets"],
    )
    records = {str(row["record_id"]): row for row in frame["records"]}
    candidates = list(frame["candidates"])
    ready_candidates = [
        row for row in candidates if row.get("resolution_status") == "ready"
    ]
    if limit is not None and limit < 1:
        raise TerminologyResolutionError("resolution limit must be positive")
    process_ids = {
        str(row["candidate_id"])
        for row in (ready_candidates if limit is None else ready_candidates[:limit])
    }
    formal_complete = limit is None
    rows: list[dict[str, Any]] = []
    self_enabled = bool(gate["branches"]["self_explanation"]["enabled"])
    web_enabled = bool(gate["branches"]["web_evidence"]["enabled"])
    minimum_similarity = float(gate["minimum_bge_similarity"])

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        term = str(candidate["term"])
        contexts = _candidate_contexts(candidate, records)
        diagnostic: dict[str, Any] = {}
        entry: dict[str, Any] | None = None
        route = "human_required"

        # Terms for which no occurrence passed the calibrated span threshold are
        # preserved as the exact human complement.  A human may include them with
        # a corrected explanation or exclude them in the final merge.
        if candidate.get("resolution_status") == "human_span_required":
            diagnostic["span_gate"] = {
                "status": "human_required",
                "reason": "no occurrence passed the frozen span auto threshold",
            }
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "rank": candidate["rank"],
                    "term": term,
                    "route": route,
                    "entry": None,
                    "diagnostic": diagnostic,
                }
            )
            continue

        # ``limit`` exists only for a diagnostic smoke run.  The artifact remains
        # explicitly incomplete and cannot be human-merged or finalized.
        if candidate_id not in process_ids:
            diagnostic["execution"] = {
                "status": "not_run",
                "reason": "diagnostic resolution limit",
            }
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "rank": candidate["rank"],
                    "term": term,
                    "route": route,
                    "entry": None,
                    "diagnostic": diagnostic,
                }
            )
            continue

        self_hard_pass = False
        try:
            request = {"term": term, "contexts": contexts}
            self_entry = checkpoint.execute(
                candidate_id,
                "qwen_self_explanation",
                request,
                lambda request=request, term=term: normalize_entry(
                    qwen.complete_json("qwen_self_explanation", request),
                    expected_term=term,
                ),
            )
            review_request = {"term": term, "contexts": contexts, "candidate_explanation": self_entry}
            review = checkpoint.execute(
                candidate_id,
                "deepseek_self_review",
                review_request,
                lambda request=review_request: normalize_self_review(
                    deepseek.complete_json("deepseek_self_review", request)
                ),
            )
            similarity_value = (
                float(similarity(self_entry["definition"], review["independent_definition"]))
                if similarity is not None
                else _bigram_similarity(self_entry["definition"], review["independent_definition"])
            )
            diagnostic["self_explanation"] = {
                "entry": self_entry,
                "review": review,
                "semantic_similarity": similarity_value,
                "bge_feature_used": similarity is not None,
            }
            self_hard_pass = (
                review["supported"]
                and review["context_consistent"]
                and not review["material_conflict"]
                and similarity_value >= minimum_similarity
            )
            if self_enabled and self_hard_pass:
                entry = self_entry
                route = "auto_self"
        except Exception as exc:
            # Invalid structure, provider failure, contextual inconsistency and
            # model disagreement are all reasons to distrust self-explanation;
            # they route to evidence resolution instead of ending the pipeline.
            diagnostic["self_explanation"] = {
                "hard_pass": False,
                "error": f"{exc.__class__.__name__}: {str(exc)[:1000]}",
            }

        if route != "auto_self":
            try:
                evidence = _web_evidence(
                    candidate_id=candidate_id,
                    term=term,
                    searcher=searcher,
                    fetcher=fetcher,
                    checkpoint=checkpoint,
                    maximum_queries=int(config["resolution"]["max_queries_per_term"]),
                    maximum_pages=int(config["resolution"]["max_pages_per_term"]),
                    expected_psl_sha1=str(
                        config["resolution"]["public_suffix_sha1"]
                    ),
                )
                sufficient, source_policy = _sources_sufficient(evidence, config)
                diagnostic["web_evidence"] = {
                    "source_sufficient": sufficient,
                    "source_policy_result": source_policy,
                    "evidence": evidence,
                }
                if sufficient:
                    rewrite_request = {"term": term, "contexts": contexts, "evidence": evidence}
                    web_entry = checkpoint.execute(
                        candidate_id,
                        "qwen_web_rewrite",
                        rewrite_request,
                        lambda request=rewrite_request, term=term: normalize_entry(
                            qwen.complete_json("qwen_web_rewrite", request),
                            expected_term=term,
                        ),
                    )
                    web_review_request = {
                        "term": term,
                        "candidate_explanation": web_entry,
                        "evidence": evidence,
                        "allowed_evidence_ids": [row["evidence_id"] for row in evidence],
                    }
                    web_review = checkpoint.execute(
                        candidate_id,
                        "deepseek_web_review",
                        web_review_request,
                        lambda request=web_review_request, evidence=evidence: normalize_web_review(
                            deepseek.complete_json("deepseek_web_review", request),
                            allowed_ids={row["evidence_id"] for row in evidence},
                        ),
                    )
                    diagnostic["web_evidence"]["entry"] = web_entry
                    diagnostic["web_evidence"]["review"] = web_review
                    if web_enabled and web_review["supported"] and not web_review["material_conflict"]:
                        entry = web_entry
                        route = "auto_web"
            except Exception as exc:
                diagnostic.setdefault("web_evidence", {})["error"] = (
                    f"{exc.__class__.__name__}: {str(exc)[:1000]}"
                )
                route = "human_required"
                entry = None
        rows.append(
            {
                "candidate_id": candidate_id,
                "rank": candidate["rank"],
                "term": term,
                "route": route,
                "entry": entry,
                "diagnostic": diagnostic,
            }
        )

    auto_rows = [row for row in rows if row["route"] in {"auto_self", "auto_web"}]
    qc_count = math.ceil(len(auto_rows) * float(config["resolution"]["auto_qc_fraction"]))
    seed = str(config["sampling"]["seed"])
    qc_sample = sorted(
        auto_rows,
        key=lambda row: _sampling_key(seed, "resolution-qc", row["candidate_id"]),
    )[:qc_count]
    qc_sample.sort(key=lambda row: int(row["rank"]))
    human_queue = [
        {
            "candidate_id": row["candidate_id"],
            "rank": row["rank"],
            "term": row["term"],
            "reason": (
                row["diagnostic"].get("span_gate", {}).get("reason")
                or row["diagnostic"].get("execution", {}).get("reason")
                or row["diagnostic"].get("web_evidence", {}).get("error")
                or "automatic explanation gates did not pass"
            ),
            "diagnostic": row["diagnostic"],
        }
        for row in rows
        if row["route"] == "human_required"
    ]
    identity = {
        "schema_version": RESOLUTION_SCHEMA_VERSION,
        "span_frame_id": frame["frame_id"],
        "full_audit_decision_sha256": audit["decision_sha256"],
        "resolution_gate_sha256": gate["gate_sha256"],
        "execution_contract_sha256": resolved_execution_contract[
            "contract_sha256"
        ],
        "checkpoint_binding_sha256": checkpoint.binding_sha256,
        "formal_complete": formal_complete,
        "expected_candidate_count": len(candidates),
        "rows_sha256": _canonical_sha(rows),
        "human_queue_sha256": _canonical_sha(human_queue),
        "qc_sample_sha256": _canonical_sha(qc_sample),
    }
    resolution_id = "termres-" + _canonical_sha(identity)
    metadata = {
        **identity,
        "resolution_id": resolution_id,
        "resource_role": RESOURCE_ROLE,
        "candidate_count": len(rows),
        "expected_candidate_count": len(candidates),
        "formal_complete": formal_complete,
        "model_processed_candidate_count": len(process_ids),
        "diagnostic_limit": limit,
        "route_counts": dict(Counter(row["route"] for row in rows)),
        "human_queue_count": len(human_queue),
        "qc_sample_count": len(qc_sample),
        "live_web_during_formal_inference": False,
    }
    parent = Path(output_root).resolve()
    target = parent / resolution_id
    if target.exists():
        validate_resolution_artifact(target)
        return target
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".termres-", dir=parent))
    try:
        write_canonical_json(temporary / "resolution.meta.json", metadata)
        write_canonical_json(
            temporary / "execution_contract.json", resolved_execution_contract
        )
        write_canonical_jsonl(
            temporary / "rows.jsonl", rows, key="rank", numeric_key=True
        )
        write_canonical_jsonl(
            temporary / "human_terminology_queue.jsonl",
            human_queue,
            key="rank",
            numeric_key=True,
        )
        write_canonical_jsonl(
            temporary / "auto_qc_sample.jsonl",
            qc_sample,
            key="rank",
            numeric_key=True,
        )
        files = {
            path.name: sha256_file(path)
            for path in temporary.iterdir()
            if path.is_file()
        }
        write_canonical_json(
            temporary / "manifest.json",
            {
                "schema_version": RESOLUTION_SCHEMA_VERSION,
                "resolution_id": resolution_id,
                "files": files,
            },
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_resolution_artifact(target)
    return target


def validate_resolution_artifact(path: str | Path) -> dict[str, Any]:
    directory = Path(path)
    manifest = load_json(directory / "manifest.json")
    metadata = load_json(directory / "resolution.meta.json")
    execution_contract = load_json(directory / "execution_contract.json")
    rows = load_jsonl(directory / "rows.jsonl")
    queue = load_jsonl(directory / "human_terminology_queue.jsonl")
    qc = load_jsonl(directory / "auto_qc_sample.jsonl")
    if manifest.get("schema_version") != RESOLUTION_SCHEMA_VERSION:
        raise TerminologyResolutionError("resolution artifact schema is invalid")
    resolution_id = manifest.get("resolution_id")
    if metadata.get("resolution_id") != resolution_id or directory.name != resolution_id:
        raise TerminologyResolutionError("resolution artifact identity differs")
    expected = manifest.get("files")
    if not isinstance(expected, Mapping) or set(expected) != {
        "resolution.meta.json",
        "execution_contract.json",
        "rows.jsonl",
        "human_terminology_queue.jsonl",
        "auto_qc_sample.jsonl",
    }:
        raise TerminologyResolutionError("resolution artifact file set differs")
    for name, digest in expected.items():
        if sha256_file(directory / name) != digest:
            raise TerminologyResolutionError(f"resolution artifact hash mismatch: {name}")
    computed = {
        "execution_contract_sha256": execution_contract.get("contract_sha256")
        if isinstance(execution_contract, Mapping)
        else None,
        "rows_sha256": _canonical_sha(rows),
        "human_queue_sha256": _canonical_sha(queue),
        "qc_sample_sha256": _canonical_sha(qc),
    }
    if any(metadata.get(key) != digest for key, digest in computed.items()):
        raise TerminologyResolutionError("resolution payload hashes differ")
    execution_contract = validate_resolution_execution_contract(
        execution_contract
    )
    identity_keys = (
        "schema_version",
        "span_frame_id",
        "full_audit_decision_sha256",
        "resolution_gate_sha256",
        "execution_contract_sha256",
        "checkpoint_binding_sha256",
        "formal_complete",
        "expected_candidate_count",
        "rows_sha256",
        "human_queue_sha256",
        "qc_sample_sha256",
    )
    if resolution_id != "termres-" + _canonical_sha(
        {key: metadata.get(key) for key in identity_keys}
    ):
        raise TerminologyResolutionError("resolution content-addressed id is invalid")
    if len(rows) != metadata.get("candidate_count") or len(queue) != metadata.get("human_queue_count") or len(qc) != metadata.get("qc_sample_count"):
        raise TerminologyResolutionError("resolution artifact counts differ")
    formal_complete = metadata.get("formal_complete")
    expected_count = metadata.get("expected_candidate_count")
    if not isinstance(formal_complete, bool) or not isinstance(expected_count, int):
        raise TerminologyResolutionError("resolution completeness metadata is invalid")
    if expected_count != len(rows):
        raise TerminologyResolutionError("resolution does not cover the frozen candidate set")
    if formal_complete and metadata.get("diagnostic_limit") is not None:
        raise TerminologyResolutionError("formal resolution has a diagnostic limit")
    if not formal_complete and (
        not isinstance(metadata.get("diagnostic_limit"), int)
        or metadata["diagnostic_limit"] < 1
    ):
        raise TerminologyResolutionError("incomplete resolution lacks its diagnostic limit")
    if any(_forbidden_key_paths(row) for row in rows + queue + qc):
        raise TerminologyResolutionError("resolution artifact contains task-category fields")
    row_ids = [str(row.get("candidate_id")) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise TerminologyResolutionError("resolution candidate ids are not unique")
    for row in rows:
        if row.get("route") not in {"auto_self", "auto_web", "human_required"}:
            raise TerminologyResolutionError("resolution route is invalid")
        if row["route"].startswith("auto_"):
            normalize_entry(row.get("entry"), expected_term=str(row["term"]))
        elif row.get("entry") is not None:
            raise TerminologyResolutionError("human-required row has an automatic entry")
    queue_ids = {row["candidate_id"] for row in queue}
    expected_queue = {row["candidate_id"] for row in rows if row["route"] == "human_required"}
    if len(queue_ids) != len(queue) or queue_ids != expected_queue:
        raise TerminologyResolutionError("human terminology queue is not the exact complement")
    qc_ids = {row["candidate_id"] for row in qc}
    auto_ids = {
        row["candidate_id"]
        for row in rows
        if row["route"] in {"auto_self", "auto_web"}
    }
    if len(qc_ids) != len(qc) or not qc_ids.issubset(auto_ids):
        raise TerminologyResolutionError("automatic QC sample is invalid")
    return {
        "resolution_id": resolution_id,
        "metadata": metadata,
        "execution_contract": execution_contract,
        "rows": rows,
        "queue": queue,
        "qc": qc,
    }


def _build_merged_resolution(
    resolution: Mapping[str, Any], human: Any
) -> dict[str, Any]:
    """Validate the exact human complement and derive the canonical merge."""

    if not resolution["metadata"]["formal_complete"]:
        raise TerminologyResolutionError(
            "diagnostic resolution artifacts cannot be human-merged"
        )
    if (
        not isinstance(human, Mapping)
        or human.get("schema_version") != HUMAN_RESOLUTION_VERSION
        or human.get("resolution_id") != resolution["resolution_id"]
        or not isinstance(human.get("reviewer_id"), str)
        or not human["reviewer_id"].strip()
        or not isinstance(human.get("decisions"), list)
        or not isinstance(human.get("qc_reviews"), list)
    ):
        raise TerminologyResolutionError("human terminology resolution is invalid")
    row_by_id = {row["candidate_id"]: row for row in resolution["rows"]}
    decisions: dict[str, dict[str, Any]] = {}
    for row in human["decisions"]:
        if not isinstance(row, Mapping) or set(row) != {"candidate_id", "action", "entry", "notes"}:
            raise TerminologyResolutionError("human terminology decision fields are invalid")
        candidate_id = str(row["candidate_id"])
        source = row_by_id.get(candidate_id)
        if source is None or candidate_id in decisions or row["action"] not in {"include", "exclude"} or not isinstance(row["notes"], str):
            raise TerminologyResolutionError("human terminology decision is invalid")
        entry = None
        if row["action"] == "include":
            entry = normalize_entry(row["entry"], expected_term=str(source["term"]))
        elif row["entry"] is not None:
            raise TerminologyResolutionError("excluded terminology decision must have entry=null")
        decisions[candidate_id] = {"action": row["action"], "entry": entry, "notes": row["notes"]}
    qc_expected = {row["candidate_id"] for row in resolution["qc"]}
    qc_reviews: dict[str, bool] = {}
    for row in human["qc_reviews"]:
        if not isinstance(row, Mapping) or set(row) != {"candidate_id", "material_error", "notes"}:
            raise TerminologyResolutionError("QC review fields are invalid")
        candidate_id = str(row["candidate_id"])
        if candidate_id not in qc_expected or candidate_id in qc_reviews or not isinstance(row["material_error"], bool) or not isinstance(row["notes"], str):
            raise TerminologyResolutionError("QC review is invalid")
        qc_reviews[candidate_id] = row["material_error"]
    if set(qc_reviews) != qc_expected:
        raise TerminologyResolutionError("automatic QC review is incomplete")
    qc_failed = any(qc_reviews.values())
    disabled_auto_routes = sorted(
        {
            str(row_by_id[candidate_id]["route"])
            for candidate_id, material_error in qc_reviews.items()
            if material_error
        }
    )
    required = {row["candidate_id"] for row in resolution["queue"]}
    if disabled_auto_routes:
        required |= {
            row["candidate_id"]
            for row in resolution["rows"]
            if row["route"] in disabled_auto_routes
        }
    if set(decisions) != required:
        raise TerminologyResolutionError(
            "human decisions are not the exact required set after QC policy"
        )
    final_entries: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in resolution["rows"]:
        candidate_id = row["candidate_id"]
        if candidate_id in decisions:
            decision = decisions[candidate_id]
            if decision["action"] == "include":
                final_entries.append({"candidate_id": candidate_id, "entry": decision["entry"], "source": "human"})
            else:
                excluded.append({"candidate_id": candidate_id, "term": row["term"], "reason": decision["notes"]})
        elif row["route"] in {"auto_self", "auto_web"}:
            final_entries.append({"candidate_id": candidate_id, "entry": row["entry"], "source": row["route"]})
    document = {
        "schema_version": MERGED_RESOLUTION_VERSION,
        "resolution_id": resolution["resolution_id"],
        "human_source_sha256": _canonical_sha(human),
        "reviewer_id": human["reviewer_id"],
        "qc_failed": qc_failed,
        "disabled_auto_routes": disabled_auto_routes,
        "entries": final_entries,
        "excluded": excluded,
        "unresolved_count": 0,
    }
    document["merged_sha256"] = _canonical_sha(document)
    return document


def merge_human_resolution(
    *,
    resolution_dir: str | Path,
    human_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    resolution = validate_resolution_artifact(resolution_dir)
    human = load_json(human_path)
    document = _build_merged_resolution(resolution, human)
    write_canonical_json(output_path, document)
    return document


def _lightweight_lex_v2_catalog(
    terms: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate lex:v2 identity/render compatibility without NumPy/sklearn."""

    catalog: list[dict[str, Any]] = []
    rendered_hashes: set[str] = set()
    for row in terms:
        entry = normalize_entry(
            {key: row.get(key) for key in ENTRY_KEYS},
            expected_term=str(row.get("term", "")),
        )
        expected_id = stable_term_evidence_id(
            entry["term"],
            entry["definition"],
            entry["variants"],
            entry["usage_notes"],
            entry["ambiguity_notes"],
        )
        if (
            (row.get("lexicon_id") is not None and row.get("lexicon_id") != expected_id)
            or not expected_id.startswith("lex:v2:")
        ):
            raise TerminologyResolutionError("terminology entry has a noncanonical lex:v2 id")
        rendered = f"术语：{entry['term']}\n词义说明：{entry['definition']}"
        if entry["usage_notes"]:
            rendered += f"\n用法提示：{entry['usage_notes']}"
        if entry["ambiguity_notes"]:
            rendered += f"\n歧义提示：{entry['ambiguity_notes']}"
        if entry["variants"]:
            rendered += "\n词形变体：" + "、".join(entry["variants"])
        rendered_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        if rendered_sha in rendered_hashes:
            raise TerminologyResolutionError(
                "terminology library has duplicate model-visible evidence"
            )
        rendered_hashes.add(rendered_sha)
        catalog.append(
            {
                "lexicon_id": expected_id,
                "rendered_block": rendered,
                "rendered_block_sha256": rendered_sha,
            }
        )
    if len({row["lexicon_id"] for row in catalog}) != len(catalog):
        raise TerminologyResolutionError("terminology lex:v2 ids are duplicated")
    return catalog


def finalize_library(
    *,
    resolution_dir: str | Path,
    merged_path: str | Path,
    output_root: str | Path,
) -> Path:
    resolution = validate_resolution_artifact(resolution_dir)
    if not resolution["metadata"]["formal_complete"]:
        raise TerminologyResolutionError(
            "diagnostic resolution artifacts cannot be finalized"
        )
    merged = load_json(merged_path)
    if (
        not isinstance(merged, Mapping)
        or merged.get("schema_version") != MERGED_RESOLUTION_VERSION
        or merged.get("resolution_id") != resolution["resolution_id"]
        or merged.get("unresolved_count") != 0
    ):
        raise TerminologyResolutionError("merged terminology resolution is invalid")
    declared = merged.get("merged_sha256")
    if declared != _canonical_sha({key: value for key, value in merged.items() if key != "merged_sha256"}):
        raise TerminologyResolutionError("merged terminology resolution hash is invalid")
    terms: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    provenance: list[dict[str, Any]] = []
    for row in merged.get("entries", []):
        if not isinstance(row, Mapping) or set(row) != {"candidate_id", "entry", "source"}:
            raise TerminologyResolutionError("merged terminology entry row is invalid")
        entry = normalize_entry(row["entry"], expected_term=str(row["entry"].get("term", "")))
        lookup_key = entry["term"].casefold().strip()
        if lookup_key in seen_keys:
            raise TerminologyResolutionError("final terminology library has duplicate terms")
        seen_keys.add(lookup_key)
        lexicon_id = stable_term_evidence_id(
            entry["term"],
            entry["definition"],
            entry["variants"],
            entry["usage_notes"],
            entry["ambiguity_notes"],
        )
        # The public terminology entry is deliberately limited to the five
        # understanding fields.  lex:v2 identity and resolution lineage stay in
        # the separate provenance plane and are recomputed by retrieval.
        terms.append(entry)
        provenance.append(
            {
                "lexicon_id": lexicon_id,
                "candidate_id": row["candidate_id"],
                "resolution_source": row["source"],
            }
        )
    terms.sort(key=lambda row: (str(row["term"]).casefold(), str(row["term"])))
    provenance.sort(key=lambda row: row["lexicon_id"])
    _lightweight_lex_v2_catalog(terms)
    identity = {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "resource_role": RESOURCE_ROLE,
        "resolution_id": resolution["resolution_id"],
        "merged_sha256": merged["merged_sha256"],
        "terms_sha256": _canonical_sha(terms),
        "provenance_sha256": _canonical_sha(provenance),
    }
    library_id = "termlib-" + _canonical_sha(identity)
    metadata = {
        **identity,
        "library_id": library_id,
        "entry_count": len(terms),
        "model_visible_fields": [
            "term", "definition", "usage_notes", "ambiguity_notes", "variants"
        ],
        "task_label_visibility": "absent",
        "live_web_during_formal_inference": False,
    }
    parent = Path(output_root).resolve()
    target = parent / library_id
    if target.exists():
        validate_library(target)
        return target
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".termlib-", dir=parent))
    try:
        write_canonical_json(temporary / "library.meta.json", metadata)
        write_canonical_jsonl(temporary / "terms.jsonl", terms, key="term")
        write_canonical_jsonl(temporary / "provenance.jsonl", provenance, key="lexicon_id")
        write_canonical_json(
            temporary / "library.json",
            {
                "schema_version": LIBRARY_SCHEMA_VERSION,
                "resource_role": RESOURCE_ROLE,
                "terms": terms,
            },
        )
        files = {
            path.name: sha256_file(path)
            for path in temporary.iterdir()
            if path.is_file()
        }
        write_canonical_json(
            temporary / "manifest.json",
            {
                "schema_version": LIBRARY_MANIFEST_VERSION,
                "library_id": library_id,
                "files": files,
            },
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_library(target)
    return target


def validate_library(path: str | Path) -> dict[str, Any]:
    directory = Path(path)
    manifest = load_json(directory / "manifest.json")
    metadata = load_json(directory / "library.meta.json")
    library = load_json(directory / "library.json")
    terms = load_jsonl(directory / "terms.jsonl")
    provenance = load_jsonl(directory / "provenance.jsonl")
    if manifest.get("schema_version") != LIBRARY_MANIFEST_VERSION:
        raise TerminologyResolutionError("terminology library manifest schema is invalid")
    library_id = manifest.get("library_id")
    if metadata.get("library_id") != library_id or directory.name != library_id:
        raise TerminologyResolutionError("terminology library identity differs")
    expected = manifest.get("files")
    if not isinstance(expected, Mapping) or set(expected) != {
        "library.meta.json", "library.json", "terms.jsonl", "provenance.jsonl"
    }:
        raise TerminologyResolutionError("terminology library file set differs")
    for name, digest in expected.items():
        if sha256_file(directory / name) != digest:
            raise TerminologyResolutionError(f"terminology library hash mismatch: {name}")
    if library != {"schema_version": LIBRARY_SCHEMA_VERSION, "resource_role": RESOURCE_ROLE, "terms": terms}:
        raise TerminologyResolutionError("terminology library JSON/JSONL differs")
    if len(terms) != metadata.get("entry_count") or len(provenance) != len(terms):
        raise TerminologyResolutionError("terminology library counts differ")
    if (
        metadata.get("terms_sha256") != _canonical_sha(terms)
        or metadata.get("provenance_sha256") != _canonical_sha(provenance)
    ):
        raise TerminologyResolutionError("terminology library payload hashes differ")
    identity_keys = (
        "schema_version",
        "resource_role",
        "resolution_id",
        "merged_sha256",
        "terms_sha256",
        "provenance_sha256",
    )
    if library_id != "termlib-" + _canonical_sha(
        {key: metadata.get(key) for key in identity_keys}
    ):
        raise TerminologyResolutionError("terminology library content-addressed id is invalid")
    if any(_forbidden_key_paths(row) for row in terms + provenance):
        raise TerminologyResolutionError("terminology library contains task-category fields")
    if any(set(row) != ENTRY_KEYS for row in terms):
        raise TerminologyResolutionError(
            "public terminology entries must expose exactly five understanding fields"
        )
    catalog = _lightweight_lex_v2_catalog(terms)
    return {
        "library_id": library_id,
        "entry_count": len(terms),
        "terms": terms,
        "catalog": catalog,
        "metadata": metadata,
    }


def _lifecycle_dependency(
    path: str | Path, *, workspace_root: str | Path, kind: str, artifact_id: str
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    target = Path(path).resolve()
    try:
        logical = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise TerminologyResolutionError(
            f"{kind} lifecycle dependency must be inside the workspace"
        ) from exc
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise TerminologyResolutionError(f"{kind} lifecycle manifest is missing")
    return {
        "schema_version": "terminology-lifecycle-dependency/v1",
        "artifact_kind": kind,
        "artifact_id": artifact_id,
        "logical_repo_path": logical,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _resolve_lifecycle_dependency(
    dependency: Mapping[str, Any], *, workspace_root: str | Path
) -> tuple[Path, dict[str, Any]]:
    if (
        not isinstance(dependency, Mapping)
        or set(dependency)
        != {
            "schema_version",
            "artifact_kind",
            "artifact_id",
            "logical_repo_path",
            "manifest_sha256",
        }
        or dependency.get("schema_version")
        != "terminology-lifecycle-dependency/v1"
        or dependency.get("artifact_kind")
        not in {"terminology-span-frame", "terminology-resolution", "terminology-library"}
        or not isinstance(dependency.get("logical_repo_path"), str)
        or Path(str(dependency["logical_repo_path"])).is_absolute()
        or ".." in Path(str(dependency["logical_repo_path"])).parts
    ):
        raise TerminologyResolutionError("terminology lifecycle dependency is invalid")
    root = Path(workspace_root).resolve()
    target = (root / str(dependency["logical_repo_path"])).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise TerminologyResolutionError(
            "terminology lifecycle dependency escapes workspace"
        ) from exc
    if sha256_file(target / "manifest.json") != dependency.get("manifest_sha256"):
        raise TerminologyResolutionError("terminology lifecycle manifest hash differs")
    kind = dependency["artifact_kind"]
    report = (
        validate_full_span_frame(target)
        if kind == "terminology-span-frame"
        else validate_resolution_artifact(target)
        if kind == "terminology-resolution"
        else validate_library(target)
    )
    observed_id = (
        report["frame_id"]
        if kind == "terminology-span-frame"
        else report["resolution_id"]
        if kind == "terminology-resolution"
        else report["library_id"]
    )
    if observed_id != dependency["artifact_id"]:
        raise TerminologyResolutionError("terminology lifecycle dependency id differs")
    return target, report


def _publication_file_set() -> set[str]:
    return {
        "data_ref.json",
        "train_partition_ref.json",
        "lexicon.json",
        "manifest.json",
        "provenance.json",
        "lifecycle.json",
        "pipeline_config.json",
        "span_pilot_decision.json",
        "full_span_audit_decision.json",
        "resolution_gate.json",
        "merged_resolution.json",
        "human_resolution.json",
        "payload_manifest.json",
    }


def _validate_frame_against_frozen_fit(
    frame: Mapping[str, Any],
    frozen: Any,
    config: Mapping[str, Any],
) -> None:
    record_by_id = {
        str(row.get("record_id")): row for row in frame.get("records", [])
    }
    expected_ids = {str(row.get("id")) for row in frozen.records}
    if len(record_by_id) != len(frame.get("records", [])) or set(record_by_id) != expected_ids:
        raise TerminologyResolutionError("span frame record ids differ from frozen fit")
    maximum = int(config["extraction"]["max_text_chars_per_record"])
    for source in frozen.records:
        record_id = str(source["id"])
        content = str(source["content"]).replace("\r\n", "\n").replace("\r", "\n")
        if len(content) > maximum:
            raise TerminologyResolutionError(
                f"frozen fit content exceeds the full-text limit: {record_id}"
            )
        frame_row = record_by_id[record_id]
        if (
            frame_row.get("content") != content
            or frame_row.get("content_sha256")
            != hashlib.sha256(content.encode("utf-8")).hexdigest()
        ):
            raise TerminologyResolutionError(
                f"span frame content differs from frozen fit: {record_id}"
            )


def publish_stage1_library(
    *,
    library_dir: str | Path,
    span_frame_dir: str | Path,
    pilot_decision_path: str | Path,
    full_audit_decision_path: str | Path,
    resolution_gate_path: str | Path,
    resolution_dir: str | Path,
    merged_path: str | Path,
    human_resolution_path: str | Path,
    config: Mapping[str, Any],
    data_ref: str | Path,
    train_partition_ref: str | Path,
    workspace_root: str | Path,
    target_root: str | Path,
    write_ref: str | Path | None = None,
) -> dict[str, Any]:
    """Publish the frozen library as the main experiment's lexicon dependency."""

    root = Path(workspace_root).resolve()
    from build_lex.train_only import resolve_train_input

    frozen = resolve_train_input(
        data_ref=data_ref,
        train_partition_ref=train_partition_ref,
        workspace_root=root,
        formal=True,
    )
    frame = validate_full_span_frame(span_frame_dir)
    pilot = validate_pilot_decision(
        pilot_decision_path,
        config=config,
        qwen_contract_sha256=str(frame["metadata"]["qwen_contract_sha256"]),
        require_pass=True,
    )
    if pilot["decision_sha256"] != frame["metadata"]["pilot_decision_sha256"]:
        raise TerminologyResolutionError("span frame and pilot decision differ")
    audit = validate_full_audit_decision(
        full_audit_decision_path,
        span_frame_id=frame["frame_id"],
        require_pass=True,
    )
    gate = validate_resolution_gate(
        resolution_gate_path, span_frame_id=frame["frame_id"], config=config
    )
    resolution = validate_resolution_artifact(resolution_dir)
    validate_resolution_execution_contract(
        resolution["execution_contract"], config=config
    )
    if (
        not resolution["metadata"]["formal_complete"]
        or resolution["metadata"]["span_frame_id"] != frame["frame_id"]
        or resolution["metadata"]["full_audit_decision_sha256"]
        != audit["decision_sha256"]
        or resolution["metadata"]["resolution_gate_sha256"]
        != gate["gate_sha256"]
        or gate["execution_contract_sha256"]
        != resolution["execution_contract"]["contract_sha256"]
    ):
        raise TerminologyResolutionError("resolution lifecycle binding differs")
    merged = load_json(merged_path)
    human = load_json(human_resolution_path)
    if merged != _build_merged_resolution(resolution, human):
        raise TerminologyResolutionError("merged/human resolution binding differs")
    library = validate_library(library_dir)
    if (
        library["metadata"]["resolution_id"] != resolution["resolution_id"]
        or library["metadata"]["merged_sha256"] != merged["merged_sha256"]
        or frame["metadata"]["fit_data_sha256"] != frozen.train_data_sha256
        or frame["metadata"]["record_count"] != len(frozen.records)
    ):
        raise TerminologyResolutionError("library or fit lineage differs")
    _validate_frame_against_frozen_fit(frame, frozen, config)

    dependencies = {
        "span_frame": _lifecycle_dependency(
            span_frame_dir,
            workspace_root=root,
            kind="terminology-span-frame",
            artifact_id=frame["frame_id"],
        ),
        "resolution": _lifecycle_dependency(
            resolution_dir,
            workspace_root=root,
            kind="terminology-resolution",
            artifact_id=resolution["resolution_id"],
        ),
        "library": _lifecycle_dependency(
            library_dir,
            workspace_root=root,
            kind="terminology-library",
            artifact_id=library["library_id"],
        ),
    }
    id_inputs = {
        "schema_version": STAGE1_PUBLICATION_MANIFEST_VERSION,
        "publication_policy": STAGE1_PUBLICATION_POLICY,
        "resource_role": RESOURCE_ROLE,
        "dataset": "full",
        "data_build_id": frozen.data_build_id,
        "train_data_sha256": frozen.train_data_sha256,
        "train_ids_sha256": frozen.train_ids_sha256,
        "source_train_data_sha256": frozen.source_train_data_sha256,
        "source_train_ids_sha256": frozen.source_train_ids_sha256,
        "train_partition_dependency": frozen.train_partition_dependency,
        "fit_record_count": len(frozen.records),
        "config_sha256": _canonical_sha(config),
        "lifecycle_dependencies": dependencies,
        "pilot_decision_sha256": pilot["decision_sha256"],
        "full_audit_decision_sha256": audit["decision_sha256"],
        "resolution_gate_sha256": gate["gate_sha256"],
        "merged_sha256": merged["merged_sha256"],
        "human_resolution_sha256": _canonical_sha(human),
        "terms_sha256": _canonical_sha(library["terms"]),
        "publisher_code_sha256": sha256_file(Path(__file__)),
    }
    build_id = "lex-" + _canonical_sha(id_inputs)
    lexicon = {
        "schema_version": STAGE1_LEXICON_SCHEMA_VERSION,
        "resource_role": RESOURCE_ROLE,
        "lexicon_build_id": build_id,
        "dataset": "full",
        "source": "pilot-gated-terminology-lifecycle",
        "description": "Category-free fit-only terminology understanding library.",
        "total_terms": len(library["terms"]),
        "terms": library["terms"],
    }
    catalog = _lightweight_lex_v2_catalog(library["terms"])
    manifest = {
        "schema_version": STAGE1_PUBLICATION_MANIFEST_VERSION,
        "resource_role": RESOURCE_ROLE,
        "publication_policy": STAGE1_PUBLICATION_POLICY,
        "lexicon_build_id": build_id,
        "data_build_id": frozen.data_build_id,
        "source_split": "train",
        "source_partition": "fit",
        "source_mode": "data_ref+train_partition",
        "scientific_eligible": True,
        "train_record_count": len(frozen.records),
        "train_data_sha256": frozen.train_data_sha256,
        "train_ids_sha256": frozen.train_ids_sha256,
        "source_train_data_sha256": frozen.source_train_data_sha256,
        "source_train_ids_sha256": frozen.source_train_ids_sha256,
        "train_partition_dependency": frozen.train_partition_dependency,
        "fit_only_verified": True,
        "calibration_contribution_count": 0,
        "terms_sha256": _canonical_sha(library["terms"]),
        "lexicon_ids": [row["lexicon_id"] for row in catalog],
        "train_only_verified": True,
        "lifecycle_dependencies": dependencies,
        "id_inputs": id_inputs,
    }
    provenance = {
        "schema_version": STAGE1_PUBLICATION_PROVENANCE_VERSION,
        "resource_role": RESOURCE_ROLE,
        "publication_policy": STAGE1_PUBLICATION_POLICY,
        "lexicon_build_id": build_id,
        "lexicon_source_partition": "fit",
        "calibration_contribution_count": 0,
        "lifecycle_dependencies": dependencies,
        "train_only_verified": True,
        "model_visible_fields": sorted(ENTRY_KEYS),
        "live_web_during_formal_inference": False,
    }
    lifecycle = {
        "schema_version": "stage1-terminology-lifecycle-publication/v1",
        "lexicon_build_id": build_id,
        "lifecycle_dependencies": dependencies,
        "pilot_decision_sha256": pilot["decision_sha256"],
        "full_audit_decision_sha256": audit["decision_sha256"],
        "resolution_gate_sha256": gate["gate_sha256"],
        "merged_sha256": merged["merged_sha256"],
        "human_resolution_sha256": _canonical_sha(human),
        "unresolved_count": 0,
    }
    parent = Path(target_root).resolve()
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise TerminologyResolutionError("Stage1 publication root escapes workspace") from exc
    target = parent / build_id
    if not target.exists():
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".termpub-", dir=parent))
        try:
            for name, value in (
                ("data_ref.json", frozen.data_ref),
                ("train_partition_ref.json", frozen.train_partition_ref),
                ("lexicon.json", lexicon),
                ("manifest.json", manifest),
                ("provenance.json", provenance),
                ("lifecycle.json", lifecycle),
                ("pipeline_config.json", config),
                ("span_pilot_decision.json", pilot),
                ("full_span_audit_decision.json", audit),
                ("resolution_gate.json", gate),
                ("merged_resolution.json", merged),
                ("human_resolution.json", human),
            ):
                write_canonical_json(temporary / name, value)
            write_canonical_json(
                temporary / "payload_manifest.json", build_payload_manifest(temporary)
            )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    validate_stage1_published_library(target, workspace_root=root)
    locator = {
        "schema_version": STAGE1_LOCATOR_VERSION,
        "artifact_kind": "lexicon",
        "artifact_id": build_id,
        "target_path": str(target),
        "payload_manifest_sha256": sha256_file(target / "payload_manifest.json"),
    }
    if write_ref is not None:
        write_canonical_json(write_ref, locator)
    return locator


def validate_stage1_published_library(
    path: str | Path, *, workspace_root: str | Path
) -> dict[str, Any]:
    target = Path(path).resolve()
    actual = {
        item.relative_to(target).as_posix()
        for item in target.rglob("*")
        if item.is_file()
    }
    if actual != _publication_file_set():
        raise TerminologyResolutionError("Stage1 terminology publication file set differs")
    validate_payload_manifest(target)
    lexicon = load_json(target / "lexicon.json")
    manifest = load_json(target / "manifest.json")
    provenance = load_json(target / "provenance.json")
    lifecycle = load_json(target / "lifecycle.json")
    config = load_json(target / "pipeline_config.json")
    if (
        manifest.get("schema_version") != STAGE1_PUBLICATION_MANIFEST_VERSION
        or provenance.get("schema_version") != STAGE1_PUBLICATION_PROVENANCE_VERSION
        or lexicon.get("schema_version") != STAGE1_LEXICON_SCHEMA_VERSION
        or manifest.get("publication_policy") != STAGE1_PUBLICATION_POLICY
        or manifest.get("resource_role") != RESOURCE_ROLE
        or provenance.get("resource_role") != RESOURCE_ROLE
        or lexicon.get("resource_role") != RESOURCE_ROLE
    ):
        raise TerminologyResolutionError("Stage1 terminology publication schema differs")
    id_inputs = manifest.get("id_inputs")
    expected_id_input_keys = {
        "schema_version",
        "publication_policy",
        "resource_role",
        "dataset",
        "data_build_id",
        "train_data_sha256",
        "train_ids_sha256",
        "source_train_data_sha256",
        "source_train_ids_sha256",
        "train_partition_dependency",
        "fit_record_count",
        "config_sha256",
        "lifecycle_dependencies",
        "pilot_decision_sha256",
        "full_audit_decision_sha256",
        "resolution_gate_sha256",
        "merged_sha256",
        "human_resolution_sha256",
        "terms_sha256",
        "publisher_code_sha256",
    }
    if not isinstance(id_inputs, Mapping) or set(id_inputs) != expected_id_input_keys:
        raise TerminologyResolutionError("Stage1 terminology ID inputs differ")
    build_id = manifest.get("lexicon_build_id")
    if (
        not isinstance(build_id, str)
        or build_id != "lex-" + _canonical_sha(id_inputs)
        or target.name != build_id
        or lexicon.get("lexicon_build_id") != build_id
        or provenance.get("lexicon_build_id") != build_id
        or lifecycle.get("lexicon_build_id") != build_id
    ):
        raise TerminologyResolutionError("Stage1 terminology publication id differs")
    terms = lexicon.get("terms")
    if (
        set(lexicon)
        != {
            "schema_version",
            "resource_role",
            "lexicon_build_id",
            "dataset",
            "source",
            "description",
            "total_terms",
            "terms",
        }
        or lexicon.get("dataset") != "full"
        or lexicon.get("source") != "pilot-gated-terminology-lifecycle"
        or not isinstance(terms, list)
        or len(terms) != lexicon.get("total_terms")
        or any(not isinstance(row, Mapping) or set(row) != ENTRY_KEYS for row in terms)
        or manifest.get("terms_sha256") != _canonical_sha(terms)
        or manifest.get("id_inputs", {}).get("terms_sha256") != _canonical_sha(terms)
    ):
        raise TerminologyResolutionError("Stage1 terminology public terms differ")
    catalog = _lightweight_lex_v2_catalog(terms)
    if manifest.get("lexicon_ids") != [row["lexicon_id"] for row in catalog]:
        raise TerminologyResolutionError("Stage1 terminology lex:v2 ids differ")
    if _forbidden_key_paths(lexicon):
        raise TerminologyResolutionError("Stage1 terminology publication contains task fields")

    dependencies = manifest.get("lifecycle_dependencies")
    if (
        not isinstance(dependencies, Mapping)
        or set(dependencies) != {"span_frame", "resolution", "library"}
        or provenance.get("lifecycle_dependencies") != dependencies
        or lifecycle.get("lifecycle_dependencies") != dependencies
        or manifest.get("id_inputs", {}).get("lifecycle_dependencies") != dependencies
    ):
        raise TerminologyResolutionError("Stage1 lifecycle dependencies differ")
    _, frame = _resolve_lifecycle_dependency(
        dependencies["span_frame"], workspace_root=workspace_root
    )
    _, resolution = _resolve_lifecycle_dependency(
        dependencies["resolution"], workspace_root=workspace_root
    )
    validate_resolution_execution_contract(
        resolution["execution_contract"], config=config
    )
    _, library = _resolve_lifecycle_dependency(
        dependencies["library"], workspace_root=workspace_root
    )
    pilot = validate_pilot_decision(
        target / "span_pilot_decision.json",
        config=config,
        qwen_contract_sha256=str(frame["metadata"]["qwen_contract_sha256"]),
        require_pass=True,
    )
    audit = validate_full_audit_decision(
        target / "full_span_audit_decision.json",
        span_frame_id=frame["frame_id"],
        require_pass=True,
    )
    gate = validate_resolution_gate(
        target / "resolution_gate.json",
        span_frame_id=frame["frame_id"],
        config=config,
    )
    merged = load_json(target / "merged_resolution.json")
    human = load_json(target / "human_resolution.json")
    canonical_merged = _build_merged_resolution(resolution, human)
    manifest_id_bindings = {
        "schema_version": STAGE1_PUBLICATION_MANIFEST_VERSION,
        "publication_policy": STAGE1_PUBLICATION_POLICY,
        "resource_role": RESOURCE_ROLE,
        "dataset": "full",
        "data_build_id": manifest.get("data_build_id"),
        "train_data_sha256": manifest.get("train_data_sha256"),
        "train_ids_sha256": manifest.get("train_ids_sha256"),
        "source_train_data_sha256": manifest.get("source_train_data_sha256"),
        "source_train_ids_sha256": manifest.get("source_train_ids_sha256"),
        "train_partition_dependency": manifest.get("train_partition_dependency"),
        "fit_record_count": manifest.get("train_record_count"),
        "config_sha256": _canonical_sha(config),
        "lifecycle_dependencies": dependencies,
        "pilot_decision_sha256": pilot["decision_sha256"],
        "full_audit_decision_sha256": audit["decision_sha256"],
        "resolution_gate_sha256": gate["gate_sha256"],
        "merged_sha256": merged.get("merged_sha256") if isinstance(merged, Mapping) else None,
        "human_resolution_sha256": _canonical_sha(human),
        "terms_sha256": _canonical_sha(terms),
    }
    if (
        any(id_inputs.get(key) != value for key, value in manifest_id_bindings.items())
        or not isinstance(id_inputs.get("publisher_code_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(id_inputs.get("publisher_code_sha256")))
        is None
        or pilot["decision_sha256"]
        != frame["metadata"]["pilot_decision_sha256"]
        or resolution["metadata"]["full_audit_decision_sha256"]
        != audit["decision_sha256"]
        or resolution["metadata"]["resolution_gate_sha256"] != gate["gate_sha256"]
        or gate["execution_contract_sha256"]
        != resolution["execution_contract"]["contract_sha256"]
        or resolution["metadata"]["span_frame_id"] != frame["frame_id"]
        or library["metadata"]["resolution_id"] != resolution["resolution_id"]
        or library["terms"] != terms
        or merged != canonical_merged
        or library["metadata"]["merged_sha256"] != merged["merged_sha256"]
    ):
        raise TerminologyResolutionError("Stage1 terminology lifecycle chain differs")

    expected_lifecycle = {
        "schema_version": "stage1-terminology-lifecycle-publication/v1",
        "lexicon_build_id": build_id,
        "lifecycle_dependencies": dependencies,
        "pilot_decision_sha256": pilot["decision_sha256"],
        "full_audit_decision_sha256": audit["decision_sha256"],
        "resolution_gate_sha256": gate["gate_sha256"],
        "merged_sha256": merged["merged_sha256"],
        "human_resolution_sha256": _canonical_sha(human),
        "unresolved_count": 0,
    }
    expected_provenance = {
        "schema_version": STAGE1_PUBLICATION_PROVENANCE_VERSION,
        "resource_role": RESOURCE_ROLE,
        "publication_policy": STAGE1_PUBLICATION_POLICY,
        "lexicon_build_id": build_id,
        "lexicon_source_partition": "fit",
        "calibration_contribution_count": 0,
        "lifecycle_dependencies": dependencies,
        "train_only_verified": True,
        "model_visible_fields": sorted(ENTRY_KEYS),
        "live_web_during_formal_inference": False,
    }
    if lifecycle != expected_lifecycle or provenance != expected_provenance:
        raise TerminologyResolutionError("Stage1 lifecycle/provenance payload differs")

    from build_lex.train_only import _resolve_portable_train_input

    frozen = _resolve_portable_train_input(
        data_dependency=load_json(target / "data_ref.json"),
        train_partition_dependency=load_json(target / "train_partition_ref.json"),
        workspace_root=workspace_root,
    )
    _validate_frame_against_frozen_fit(frame, frozen, config)
    if (
        frozen.source_mode != "data_ref+train_partition"
        or len(frozen.records) != manifest.get("train_record_count")
        or frozen.data_build_id != manifest.get("data_build_id")
        or frozen.train_data_sha256 != manifest.get("train_data_sha256")
        or frozen.train_ids_sha256 != manifest.get("train_ids_sha256")
        or frozen.source_train_data_sha256
        != manifest.get("source_train_data_sha256")
        or frozen.source_train_ids_sha256
        != manifest.get("source_train_ids_sha256")
        or frozen.train_partition_dependency
        != manifest.get("train_partition_dependency")
        or load_json(target / "data_ref.json") != frozen.data_ref
        or load_json(target / "train_partition_ref.json")
        != frozen.train_partition_ref
        or frame["metadata"]["fit_data_sha256"] != frozen.train_data_sha256
        or manifest.get("source_split") != "train"
        or manifest.get("source_partition") != "fit"
        or manifest.get("source_mode") != "data_ref+train_partition"
        or manifest.get("fit_only_verified") is not True
        or manifest.get("scientific_eligible") is not True
        or manifest.get("calibration_contribution_count") != 0
        or manifest.get("train_only_verified") is not True
        or provenance.get("train_only_verified") is not True
    ):
        raise TerminologyResolutionError("Stage1 terminology fit lineage differs")
    return dict(manifest)
