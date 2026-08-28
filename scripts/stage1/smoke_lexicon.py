#!/usr/bin/env python3
"""Run a bounded, non-publishing WP3 lexicon engineering smoke.

The live mode is deliberately separate from the formal builder.  It evaluates
eight rank-bound candidates, writes only sanitized diagnostics, and cannot
create a formal lexicon target or locator ref.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import requests


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.llm_lexicon_builder import (  # noqa: E402
    CandidateJudgementError,
    CandidateStats,
    build_candidates,
    build_search_queries,
    collect_web_evidence,
    create_judgement_client,
    create_web_searcher,
    judge_candidate,
    select_candidates,
    should_include,
)
from build_lex.stage1_preflight import preflight_formal_lexicon  # noqa: E402
from build_lex.train_only import (  # noqa: E402
    _validate_stage_response,
    resolve_train_input,
)


SCHEMA_VERSION = "stage1-wp3-lexicon-engineering-smoke/v2"
RUN_CLASS = "engineering-smoke"
SMOKE_SELECTION: tuple[tuple[int, str], ...] = (
    (1, "基佬"),
    (5, "偷井盖"),
    (7, "仙女"),
    (11, "gay"),
    (100, "弄死"),
    (160, "德绑架"),
    (250, "天被"),
    (300, "别打拳"),
)
WEB_REQUEST_LIMIT = 24
LLM_REQUEST_LIMIT = 72
LLM_STAGE_LIMIT = 24
CANONICAL_CATEGORIES = {"Sexism", "Racism", "Region", "LGBTQ", "others"}
EXPECTED_OFFLINE_FILTERED_TERMS = frozenset({"德绑架", "天被"})
DIAGNOSTICS_ROOT = (
    REPOSITORY_ROOT
    / "exps"
    / "causal_context"
    / "stage1_p0"
    / "diagnostics"
    / "wp3_lexicon_smoke"
)
DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "stage1" / "lexicon_train_only.json"
DEFAULT_DATA_REF = REPOSITORY_ROOT / "exps" / "causal_context" / "stage1_p0" / "refs" / "data_ref.json"
DEFAULT_PARTITION_REF = (
    REPOSITORY_ROOT
    / "exps"
    / "causal_context"
    / "stage1_p0"
    / "refs"
    / "train_partition_ref.json"
)
PROTOCOL_SOURCE_PATHS = (
    Path("scripts/stage1/smoke_lexicon.py"),
    Path("src/build_lex/llm_lexicon_builder.py"),
    Path("src/build_lex/train_only.py"),
    Path("src/build_lex/web_search.py"),
    Path("src/build_lex/stage1_preflight.py"),
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_\-]{2,}")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


class SmokeLexiconError(RuntimeError):
    """A fail-closed smoke gate or execution error."""


@dataclass(frozen=True)
class PreparedCase:
    """One fixed first-round case bound to its current offline status."""

    baseline_rank: int
    candidate: CandidateStats
    eligible_rank: int | None
    offline_filtered: bool


def _case_binding(case: PreparedCase) -> dict[str, Any]:
    return {
        "baseline_rank": case.baseline_rank,
        "rank": case.baseline_rank,
        "term": case.candidate.term,
        "eligible_rank": case.eligible_rank,
        "offline_filtered": case.offline_filtered,
        "filter_reason": case.candidate.reject_hint or None,
        "regression_probe": True,
    }


def _run_flags() -> dict[str, Any]:
    return {
        "run_class": RUN_CLASS,
        "smoke_only": True,
        "scientific_eligible": False,
        "formal_publication_eligible": False,
        "published_ref": None,
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _redact_text(value: Any, limit: int) -> str:
    text = _truncate(value, limit * 2)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _HANDLE_RE.sub("@[USER]", text)
    text = _URL_RE.sub("[URL]", text)
    return _truncate(text, limit)


def _safe_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return ""
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return ""
    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path, "", ""))


def _sample_ref(value: Any) -> str:
    return hashlib.sha256(f"wp3-smoke:{value}".encode("utf-8")).hexdigest()[:12]


def _public_candidate(case: PreparedCase) -> dict[str, Any]:
    candidate = case.candidate
    contexts = []
    for row in candidate.sample_contexts[:2]:
        contexts.append(
            {
                "sample_ref": _sample_ref(row.get("id")),
                "label": str(row.get("label") or ""),
                "categories": [str(item) for item in (row.get("categories") or [])],
                "source": str(row.get("source") or ""),
                "content": _redact_text(row.get("content"), 160),
            }
        )
    return {
        **_case_binding(case),
        "term": candidate.term,
        "language": candidate.language,
        "score": round(float(candidate.score), 6),
        "total_count": candidate.total_count,
        "hate_count": candidate.hate_count,
        "non_hate_count": candidate.non_hate_count,
        "annotation_count": candidate.annotation_count,
        "track": candidate.track,
        "primary_category": candidate.primary_category,
        "category_counts": dict(sorted(candidate.category_counts.items())),
        "reject_hint": candidate.reject_hint,
        "contexts": contexts,
    }


def _public_evidence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for row in rows:
        safe_url = _safe_url(row.get("url"))
        domain = urlsplit(safe_url).netloc if safe_url else ""
        evidence.append(
            {
                "id": str(row.get("id") or ""),
                "query": _truncate(row.get("query"), 180),
                "title": _redact_text(row.get("title"), 180),
                "snippet": _redact_text(row.get("snippet"), 320),
                "url": safe_url,
                "domain": domain,
                "source": _truncate(row.get("source"), 80),
            }
        )
    return evidence


def _public_judgement(stage: str, value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    common = {
        "confidence": value.get("confidence"),
        "reason": _redact_text(value.get("reason"), 600),
    }
    if stage == "context_judge":
        return {
            **common,
            "supported": value.get("supported"),
            "category": value.get("category"),
            "categories": value.get("categories"),
        }
    if stage == "web_evidence_judge":
        return {
            **common,
            "supported": value.get("supported"),
            "evidence_ids": value.get("evidence_ids"),
        }
    return {
        **common,
        "include": value.get("include"),
        "category": value.get("category"),
        "categories": value.get("categories"),
        "definition": _redact_text(value.get("definition"), 500),
        "nonhateful_meaning": _redact_text(value.get("nonhateful_meaning"), 500),
        "variants": value.get("variants"),
        "evidence_ids": value.get("evidence_ids"),
    }


def _validate_citations(judgement: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> None:
    available = {str(row.get("id") or "") for row in evidence}
    for stage in ("web_evidence_judge", "final_lexicon_judge"):
        stage_value = judgement.get(stage)
        if not isinstance(stage_value, Mapping):
            raise SmokeLexiconError(f"missing validated stage output: {stage}")
        cited = stage_value.get("evidence_ids") or []
        if not isinstance(cited, list) or not all(isinstance(item, str) for item in cited):
            raise SmokeLexiconError(f"{stage} returned invalid evidence ID types")
        unknown = sorted({str(item) for item in cited} - available)
        if unknown:
            raise SmokeLexiconError(f"{stage} cited evidence IDs that do not exist")


def _validate_categories(judgement: Mapping[str, Any]) -> None:
    for stage in ("context_judge", "final_lexicon_judge"):
        value = judgement.get(stage)
        if not isinstance(value, Mapping):
            raise SmokeLexiconError(f"missing validated stage output: {stage}")
        category = value.get("category")
        categories = value.get("categories")
        if category not in CANONICAL_CATEGORIES:
            raise SmokeLexiconError(f"{stage} returned a non-canonical category")
        if (
            not isinstance(categories, list)
            or not all(isinstance(item, str) and item in CANONICAL_CATEGORIES for item in categories)
        ):
            raise SmokeLexiconError(f"{stage} returned non-canonical categories")


def _effective_smoke_policy(
    case: PreparedCase,
    final: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    include, reason = should_include(case.candidate, final, settings)
    if case.offline_filtered:
        return {
            "include": False,
            "reason": (
                "offline candidate gate excluded this regression probe as "
                f"{case.candidate.reject_hint}."
            ),
            "post_judgement_include": include,
            "post_judgement_reason": reason,
        }
    return {"include": include, "reason": reason}


def _validate_smoke_config(config: Mapping[str, Any]) -> dict[str, int]:
    candidate = config.get("candidate_settings")
    web = config.get("web_settings")
    llm = config.get("llm_settings")
    runtime = config.get("runtime_settings")
    if not all(isinstance(item, Mapping) for item in (candidate, web, llm, runtime)):
        raise SmokeLexiconError("smoke requires candidate/web/llm/runtime settings")
    if int(candidate.get("max_candidates", 0) or 0) < max(rank for rank, _ in SMOKE_SELECTION):
        raise SmokeLexiconError("candidate budget does not reach the bound rank 300")
    web_base = urlsplit(str(web.get("api_base") or ""))
    if (
        str(web.get("backend") or "").lower() != "search_api"
        or str(web.get("api_method") or "").upper() != "POST"
        or web_base.hostname != "api.tavily.com"
        or bool(web.get("cache_enabled"))
        or web.get("cache_path") is not None
        or web.get("require_direct_term_match") is not True
        or web.get("dedupe_by_url") is not True
        or int(web.get("max_results", 0) or 0) != 3
        or int((web.get("api_extra_params") or {}).get("max_results", 0) or 0) != 3
    ):
        raise SmokeLexiconError("smoke requires the frozen Tavily POST/no-cache/3-result policy")
    retries = llm.get("retries")
    llm_base = urlsplit(str(llm.get("api_base") or ""))
    if (
        str(llm.get("backend") or "").lower() != "deepseek"
        or str(llm.get("provider") or "").lower() != "deepseek"
        or llm_base.hostname != "api.deepseek.com"
        or isinstance(retries, bool)
        or not isinstance(retries, int)
        or retries != 2
    ):
        raise SmokeLexiconError("smoke requires the frozen DeepSeek two-retry policy")
    if bool(runtime.get("debug")) or bool(runtime.get("resume")):
        raise SmokeLexiconError("smoke requires debug=false and resume=false")
    neutral_terms = config.get("inclusion", {}).get("neutral_identity_terms")
    if not isinstance(neutral_terms, list) or "gay" not in {
        str(value).casefold() for value in neutral_terms
    }:
        raise SmokeLexiconError("smoke requires the frozen neutral-identity safety policy")
    web_requests = len(SMOKE_SELECTION) * 3
    llm_stage_calls = len(SMOKE_SELECTION) * 3
    llm_attempts = llm_stage_calls * (retries + 1)
    if web_requests != WEB_REQUEST_LIMIT or llm_attempts > LLM_REQUEST_LIMIT:
        raise SmokeLexiconError("derived smoke request budget exceeds its hard limits")
    return {
        "candidate_count": len(SMOKE_SELECTION),
        "queries_per_candidate": 3,
        "web_request_limit": WEB_REQUEST_LIMIT,
        "llm_stages_per_candidate": 3,
        "llm_attempts_per_stage": retries + 1,
        "llm_request_limit": LLM_REQUEST_LIMIT,
        "derived_llm_request_limit": llm_attempts,
    }


def _select_regression_cases(
    corpus_candidates: Sequence[CandidateStats],
    selected: Sequence[CandidateStats],
) -> list[PreparedCase]:
    """Rebind the original eight terms without hiding changed offline outcomes."""

    by_term = {candidate.term: candidate for candidate in corpus_candidates}
    if len(by_term) != len(corpus_candidates):
        raise SmokeLexiconError("candidate corpus contains duplicate canonical terms")
    eligible_ranks = {candidate.term: rank for rank, candidate in enumerate(selected, start=1)}
    cases: list[PreparedCase] = []
    observed_filtered: set[str] = set()
    for baseline_rank, expected_term in SMOKE_SELECTION:
        candidate = by_term.get(expected_term)
        if candidate is None:
            raise SmokeLexiconError(f"regression term disappeared from corpus: {expected_term!r}")
        eligible_rank = eligible_ranks.get(expected_term)
        offline_filtered = eligible_rank is None
        if offline_filtered:
            observed_filtered.add(expected_term)
        if len(build_search_queries(candidate)) != 3:
            raise SmokeLexiconError(
                f"regression term at baseline rank {baseline_rank} does not produce three queries"
            )
        cases.append(
            PreparedCase(
                baseline_rank=baseline_rank,
                candidate=candidate,
                eligible_rank=eligible_rank,
                offline_filtered=offline_filtered,
            )
        )
    if observed_filtered != EXPECTED_OFFLINE_FILTERED_TERMS:
        raise SmokeLexiconError(
            "offline regression status changed: expected only the two known fragments to be filtered"
        )
    for case in cases:
        if case.offline_filtered and case.candidate.reject_hint != "substring_fragment":
            raise SmokeLexiconError("an expected fragment lacks the substring_fragment gate reason")
    return cases


@dataclass(frozen=True)
class PreparedSmoke:
    config: dict[str, Any]
    preflight_public: dict[str, Any]
    cases: tuple[PreparedCase, ...]
    corpus_candidate_count: int
    eligible_candidate_count: int
    selection_sha256: str
    request_budget: dict[str, int]


def prepare_smoke(
    *,
    config_path: Path,
    data_ref: Path,
    train_partition_ref: Path,
    env_file: Path,
) -> PreparedSmoke:
    """Run all deterministic gates.  This function performs no HTTP or writes."""

    preflight = preflight_formal_lexicon(
        dataset="full",
        config_path=config_path,
        data_ref=data_ref,
        train_partition_ref=train_partition_ref,
        env_file=env_file,
        repository_root=REPOSITORY_ROOT,
    )
    if not preflight.passed:
        codes = ",".join(item.code for item in preflight.blockers) or "UNKNOWN"
        raise SmokeLexiconError(f"formal input preflight is blocked: {codes}")
    config = preflight.frozen_config()
    request_budget = _validate_smoke_config(config)
    frozen = resolve_train_input(
        data_ref=data_ref,
        train_partition_ref=train_partition_ref,
        formal=True,
        workspace_root=REPOSITORY_ROOT,
    )
    if len(frozen.records) != preflight.train_record_count:
        raise SmokeLexiconError("fit record count changed after preflight")
    records = [dict(row) for row in frozen.records]
    corpus = build_candidates(
        "full",
        records,
        settings=dict(config["candidate_settings"]),
        input_paths=[],
        show_progress=False,
    )
    selected = select_candidates(corpus, dict(config["candidate_settings"]))
    cases = tuple(_select_regression_cases(corpus.candidates, selected))
    binding = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": preflight.config_sha256,
        "data_build_id": preflight.data_build_id,
        "train_partition_id": preflight.train_partition_id,
        "train_data_sha256": preflight.train_data_sha256,
        "train_ids_sha256": preflight.train_ids_sha256,
        "selection": [_case_binding(case) for case in cases],
        "protocol_sources": {
            str(path): _file_sha256(REPOSITORY_ROOT / path) for path in PROTOCOL_SOURCE_PATHS
        },
    }
    return PreparedSmoke(
        config=config,
        preflight_public=preflight.to_public_dict(),
        cases=cases,
        corpus_candidate_count=len(corpus.candidates),
        eligible_candidate_count=len(selected),
        selection_sha256=_sha256(binding),
        request_budget=request_budget,
    )


class SafeAttemptRecorder:
    """Count attempts while discarding prompts, headers, and provider bodies."""

    def __init__(self, limit: int = LLM_REQUEST_LIMIT):
        self.limit = limit
        self.llm_attempt_count = 0
        self.search_call_count = 0
        self.llm_events: list[dict[str, Any]] = []
        self.search_events: list[dict[str, Any]] = []

    def record_llm_call(
        self,
        *,
        stage: str,
        payload: Mapping[str, Any],
        request_payload: Mapping[str, Any],
        attempt: int,
        raw_response: Mapping[str, Any] | None = None,
        parsed_response: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        del request_payload, raw_response, parsed_response
        self.llm_attempt_count += 1
        if self.llm_attempt_count > self.limit:
            raise SmokeLexiconError("DeepSeek attempt recorder exceeded its hard limit")
        candidate = payload.get("candidate") if isinstance(payload, Mapping) else None
        term = candidate.get("term") if isinstance(candidate, Mapping) else None
        self.llm_events.append(
            {
                "stage": str(stage),
                "term": str(term or ""),
                "attempt": int(attempt),
                "outcome": "error" if error else "success",
            }
        )

    def record_search_call(
        self,
        *,
        term: str,
        query: str,
        results: Sequence[Mapping[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        self.search_call_count += 1
        self.search_events.append(
            {
                "term": str(term),
                "query": _truncate(query, 180),
                "result_count": len(results or []),
                "outcome": "error" if error else "success",
            }
        )


class StrictStageClient:
    """Validate raw stage JSON before normalization and cap stage invocations."""

    def __init__(self, client: Any, limit: int = LLM_STAGE_LIMIT):
        self.client = client
        self.limit = limit
        self.stage_call_count = 0

    def complete_json(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.stage_call_count >= self.limit:
            raise SmokeLexiconError("DeepSeek stage-call hard limit reached")
        self.stage_call_count += 1
        response = self.client.complete_json(stage, payload)
        _validate_stage_response(stage, response)
        return response


class HTTPRequestBudget:
    """Intercept project-level requests calls and refuse over-budget or unknown hosts."""

    def __init__(self, web_url: str, llm_url: str, before_request: Any | None = None):
        self.web_url = web_url.rstrip("/")
        self.llm_url = llm_url.rstrip("/")
        self.before_request = before_request
        self.web_count = 0
        self.llm_count = 0
        self.events: list[dict[str, Any]] = []
        self._original_post: Any = None
        self._original_get: Any = None

    def _classify(self, url: Any) -> str:
        normalized = str(url or "").rstrip("/")
        if normalized == self.web_url:
            return "tavily"
        if normalized == self.llm_url:
            return "deepseek"
        raise SmokeLexiconError("smoke blocked an unexpected outbound HTTP target")

    def _call(self, method: str, original: Any, url: Any, *args: Any, **kwargs: Any) -> Any:
        provider = self._classify(url)
        # A provider redirect would be an unclassified extra network hop and
        # could leave the two-host authorization boundary.  Refuse to follow it.
        kwargs["allow_redirects"] = False
        if provider == "tavily":
            if self.web_count >= WEB_REQUEST_LIMIT:
                raise SmokeLexiconError("Tavily request hard limit reached")
            ordinal = self.web_count + 1
        else:
            if self.llm_count >= LLM_REQUEST_LIMIT:
                raise SmokeLexiconError("DeepSeek request hard limit reached")
            ordinal = self.llm_count + 1
        event = {"provider": provider, "ordinal": ordinal, "method": method, "outcome": "pending"}
        if self.before_request is not None:
            # Persist the conservative count before dispatch.  A crash between
            # this write and the socket call may over-count, but can never hide
            # a consumed request.
            self.before_request(provider=provider, ordinal=ordinal, method=method)
        if provider == "tavily":
            self.web_count = ordinal
        else:
            self.llm_count = ordinal
        self.events.append(event)
        try:
            response = original(url, *args, **kwargs)
        except Exception as exc:
            event["outcome"] = "error"
            event["error_type"] = exc.__class__.__name__
            raise
        status = getattr(response, "status_code", None)
        event["outcome"] = "response"
        event["status_class"] = f"{int(status) // 100}xx" if isinstance(status, int) else "unknown"
        return response

    def __enter__(self) -> "HTTPRequestBudget":
        self._original_post = requests.post
        self._original_get = requests.get

        def guarded_post(url: Any, *args: Any, **kwargs: Any) -> Any:
            return self._call("POST", self._original_post, url, *args, **kwargs)

        def guarded_get(url: Any, *args: Any, **kwargs: Any) -> Any:
            return self._call("GET", self._original_get, url, *args, **kwargs)

        requests.post = guarded_post
        requests.get = guarded_get
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        requests.post = self._original_post
        requests.get = self._original_get


class SensitiveLogSilencer:
    """Suppress retry warnings that can contain provider response bodies."""

    LOGGER_NAMES = ("build_lex.llm_lexicon_builder",)

    def __init__(self):
        self._states: dict[str, bool] = {}

    def __enter__(self) -> "SensitiveLogSilencer":
        for name in self.LOGGER_NAMES:
            logger = logging.getLogger(name)
            self._states[name] = logger.disabled
            logger.disabled = True
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        for name, disabled in self._states.items():
            logging.getLogger(name).disabled = disabled


def _case_failure(exc: Exception) -> dict[str, str]:
    if isinstance(exc, CandidateJudgementError):
        return {"error_type": exc.__class__.__name__, "failed_stage": exc.stage}
    if isinstance(exc, SmokeLexiconError):
        return {"error_type": exc.__class__.__name__, "message": _truncate(str(exc), 240)}
    return {"error_type": exc.__class__.__name__}


def _diagnostics_output_root(path: Path) -> Path:
    root = path.resolve()
    required = DIAGNOSTICS_ROOT.resolve()
    if root != required and required not in root.parents:
        raise SmokeLexiconError("live output must stay under the WP3 diagnostics root")
    if any(part.lower() in {"refs", "lexicons"} for part in root.parts):
        raise SmokeLexiconError("smoke diagnostics cannot be written under refs or lexicons")
    return root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class RunLedger:
    """Durable conservative accounting for exactly one authorized smoke run."""

    def __init__(self, run_dir: Path, payload: dict[str, Any]):
        self.run_dir = run_dir
        self.path = run_dir / "ledger.json"
        self.payload = payload

    def _persist(self) -> None:
        text = json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _secret_scan((text,))
        _atomic_write_text(self.path, text)

    def before_request(self, *, provider: str, ordinal: int, method: str) -> None:
        key = "tavily" if provider == "tavily" else "deepseek"
        current = int(self.payload["request_counts"][key])
        if ordinal != current + 1:
            raise SmokeLexiconError("durable request ledger lost ordinal continuity")
        limit = WEB_REQUEST_LIMIT if key == "tavily" else LLM_REQUEST_LIMIT
        if ordinal > limit:
            raise SmokeLexiconError("durable request ledger reached its hard limit")
        self.payload["state"] = "running"
        self.payload["updated_at"] = _utc_now()
        self.payload["request_counts"][key] = ordinal
        self.payload["last_dispatch"] = {
            "provider": provider,
            "ordinal": ordinal,
            "method": method,
        }
        self._persist()

    def case_finished(self, *, rank: int, status: str) -> None:
        self.payload["updated_at"] = _utc_now()
        self.payload["finished_cases"].append({"rank": rank, "status": status})
        self._persist()

    def finish(self, *, outcome: str) -> None:
        self.payload["state"] = "finished"
        self.payload["outcome"] = outcome
        self.payload["updated_at"] = _utc_now()
        self._persist()

    def fail(self, exc: BaseException) -> None:
        self.payload["state"] = "failed"
        self.payload["outcome"] = "runner-failure"
        self.payload["error_type"] = exc.__class__.__name__
        self.payload["updated_at"] = _utc_now()
        self._persist()


def _start_run_ledger(output_root: Path, prepared: PreparedSmoke) -> RunLedger:
    root = _diagnostics_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for child in sorted(root.iterdir()):
        if child.is_symlink():
            raise SmokeLexiconError("smoke diagnostics root contains a symlink")
        ledger_path = child / "ledger.json"
        if not ledger_path.is_file():
            continue
        try:
            prior = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SmokeLexiconError("existing smoke request ledger is unreadable") from exc
        if not isinstance(prior, Mapping):
            raise SmokeLexiconError("existing smoke request ledger is invalid")
        if prior.get("selection_sha256") == prepared.selection_sha256:
            raise SmokeLexiconError(
                "this selection already has a durable live ledger; rerun requires new authorization"
            )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / f"smoke-{timestamp}-{prepared.selection_sha256[:12]}"
    run_dir.mkdir(mode=0o700)
    payload = {
        "schema_version": SCHEMA_VERSION,
        **_run_flags(),
        "state": "initialized",
        "outcome": None,
        "selection_sha256": prepared.selection_sha256,
        "selected_cases": [_case_binding(case) for case in prepared.cases],
        "request_limits": {"tavily": WEB_REQUEST_LIMIT, "deepseek": LLM_REQUEST_LIMIT},
        "request_counts": {"tavily": 0, "deepseek": 0},
        "finished_cases": [],
        "last_dispatch": None,
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    ledger = RunLedger(run_dir, payload)
    ledger._persist()
    return ledger


def _secret_scan(payloads: Sequence[str]) -> None:
    combined = "\n".join(payloads)
    if "Authorization" in combined or "Bearer " in combined:
        raise SmokeLexiconError("credential-like HTTP material reached diagnostics")
    for name in ("TAVILY_API_KEY", "DEEPSEEK_API_KEY"):
        value = os.environ.get(name, "")
        if len(value) >= 8 and value in combined:
            raise SmokeLexiconError("credential value reached diagnostics")


def _render_report(summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# WP3 lexicon engineering smoke",
        "",
        "> Engineering diagnostics only. Not scientific evidence and not eligible for formal publication.",
        "",
        f"- Selection binding: `{summary['selection_sha256']}`",
        f"- Tavily requests: {summary['requests']['tavily']} / {WEB_REQUEST_LIMIT}",
        f"- DeepSeek attempts: {summary['requests']['deepseek']} / {LLM_REQUEST_LIMIT}",
        f"- Completed cases: {summary['completed_case_count']} / {len(cases)}",
        "",
        "| baseline rank | current eligible rank | term | offline filtered | status | evidence | context | web | model | policy | confidence |",
        "|---:|---:|---|---|---|---:|---|---|---|---|---:|",
    ]
    for case in cases:
        candidate = case["candidate"]
        final = (case.get("judgements") or {}).get("final_lexicon_judge") or {}
        context = (case.get("judgements") or {}).get("context_judge") or {}
        web = (case.get("judgements") or {}).get("web_evidence_judge") or {}
        term = str(candidate["term"]).replace("|", "\\|")
        lines.append(
            "| {rank} | {eligible_rank} | {term} | {offline_filtered} | {status} | {evidence} | {context} | {web} | {model} | {policy} | {confidence} |".format(
                rank=candidate["rank"],
                eligible_rank=candidate.get("eligible_rank") or "—",
                term=term,
                offline_filtered=candidate.get("offline_filtered", False),
                status=case["status"],
                evidence=len(case.get("evidence") or []),
                context=context.get("supported", "-"),
                web=web.get("supported", "-"),
                model=final.get("include", "-"),
                policy=(case.get("policy") or {}).get("include", "-"),
                confidence=final.get("confidence", "-"),
            )
        )
    lines.extend(
        [
            "",
            "The cases require human review for web relevance, contextual correctness, non-stigmatizing definitions, ambiguity handling, and citation support.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_diagnostics(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> Path:
    root = _diagnostics_output_root(run_dir.parent)
    if run_dir.resolve().parent != root or not (run_dir / "ledger.json").is_file():
        raise SmokeLexiconError("diagnostics require a reserved live run ledger")
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    cases_text = "".join(
        json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases
    )
    report_text = _render_report(summary, cases)
    _secret_scan((summary_text, cases_text, report_text))
    for name, text in (
        ("summary.json", summary_text),
        ("cases.jsonl", cases_text),
        ("report.md", report_text),
    ):
        _atomic_write_text(run_dir / name, text)
    return run_dir


def run_live(prepared: PreparedSmoke, *, preflight_args: Mapping[str, Path], output_root: Path) -> dict[str, Any]:
    # Validate the diagnostics boundary before credentials or clients exist.
    _diagnostics_output_root(output_root)
    activation = preflight_formal_lexicon(
        dataset="full",
        config_path=preflight_args["config_path"],
        data_ref=preflight_args["data_ref"],
        train_partition_ref=preflight_args["train_partition_ref"],
        env_file=preflight_args["env_file"],
        repository_root=REPOSITORY_ROOT,
    )
    identity_fields = (
        "config_sha256",
        "data_build_id",
        "train_partition_id",
        "train_partition_payload_sha256",
        "train_record_count",
        "train_data_sha256",
        "train_ids_sha256",
    )
    if not activation.passed or any(
        activation.to_public_dict().get(field) != prepared.preflight_public.get(field)
        for field in identity_fields
    ):
        raise SmokeLexiconError("preflight changed before credential activation")
    ledger = _start_run_ledger(output_root, prepared)
    try:
        # Activate only the two allowlisted, preflighted credential values.
        activation.activate_credentials()
        recorder = SafeAttemptRecorder()
        web_searcher = create_web_searcher(dict(prepared.config["web_settings"]))
        base_judge = create_judgement_client(
            dict(prepared.config["llm_settings"]),
            debug_recorder=recorder,
        )
        judge_client = StrictStageClient(base_judge)
        web_url = str(prepared.config["web_settings"]["api_base"])
        llm_base = str(prepared.config["llm_settings"]["api_base"]).rstrip("/")
        llm_url = (
            llm_base if llm_base.endswith("/chat/completions") else llm_base + "/chat/completions"
        )
        cases: list[dict[str, Any]] = []

        try:
            with SensitiveLogSilencer(), HTTPRequestBudget(
                web_url,
                llm_url,
                before_request=ledger.before_request,
            ) as http_budget:
                for prepared_case in prepared.cases:
                    rank = prepared_case.baseline_rank
                    candidate = prepared_case.candidate
                    web_before = http_budget.web_count
                    llm_before = http_budget.llm_count
                    stage_before = judge_client.stage_call_count
                    evidence: list[dict[str, Any]] = []
                    judgement: dict[str, Any] | None = None
                    policy: dict[str, Any] | None = None
                    error: dict[str, str] | None = None
                    try:
                        evidence = collect_web_evidence(
                            candidate,
                            web_searcher,
                            debug_recorder=recorder,
                        )
                        judgement = judge_candidate(candidate, evidence, judge_client)
                        _validate_citations(judgement, evidence)
                        _validate_categories(judgement)
                        policy = _effective_smoke_policy(
                            prepared_case,
                            judgement["final_lexicon_judge"],
                            dict(prepared.config["inclusion"]),
                        )
                        status = "completed"
                    except Exception as exc:
                        status = "failed"
                        error = _case_failure(exc)
                        if isinstance(exc, CandidateJudgementError):
                            judgement = dict(exc.partial)
                    cases.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            **_run_flags(),
                            "status": status,
                            "candidate": _public_candidate(prepared_case),
                            "queries": build_search_queries(candidate),
                            "evidence": _public_evidence(evidence),
                            "judgements": {
                                "context_judge": _public_judgement(
                                    "context_judge",
                                    judgement.get("context_judge") if judgement else None,
                                ),
                                "web_evidence_judge": _public_judgement(
                                    "web_evidence_judge",
                                    judgement.get("web_evidence_judge") if judgement else None,
                                ),
                                "final_lexicon_judge": _public_judgement(
                                    "final_lexicon_judge",
                                    judgement.get("final_lexicon_judge") if judgement else None,
                                ),
                            },
                            "policy": policy,
                            "requests": {
                                "tavily": http_budget.web_count - web_before,
                                "deepseek": http_budget.llm_count - llm_before,
                                "llm_stage_calls": judge_client.stage_call_count - stage_before,
                            },
                            "error": error,
                        }
                    )
                    ledger.case_finished(rank=rank, status=status)
                request_counts = {
                    "tavily": http_budget.web_count,
                    "deepseek": http_budget.llm_count,
                    "llm_stage_calls": judge_client.stage_call_count,
                    "recorded_llm_attempts": recorder.llm_attempt_count,
                    "recorded_search_calls": recorder.search_call_count,
                }
                http_events = list(http_budget.events)
        finally:
            if hasattr(web_searcher, "close"):
                web_searcher.close()

        if request_counts["deepseek"] != request_counts["recorded_llm_attempts"]:
            raise SmokeLexiconError("DeepSeek HTTP and sanitized recorder counts disagree")
        if request_counts["tavily"] != request_counts["recorded_search_calls"]:
            raise SmokeLexiconError("Tavily HTTP and sanitized recorder counts disagree")
        summary = {
            "schema_version": SCHEMA_VERSION,
            **_run_flags(),
            "status": "complete" if all(case["status"] == "completed" for case in cases) else "partial",
            "selection_sha256": prepared.selection_sha256,
            "preflight": prepared.preflight_public,
            "corpus_candidate_count": prepared.corpus_candidate_count,
            "eligible_candidate_count": prepared.eligible_candidate_count,
            "selected_cases": [_case_binding(case) for case in prepared.cases],
            "offline_filtered_case_count": sum(
                case.offline_filtered for case in prepared.cases
            ),
            "request_budget": prepared.request_budget,
            "requests": request_counts,
            "completed_case_count": sum(case["status"] == "completed" for case in cases),
            "failed_case_count": sum(case["status"] != "completed" for case in cases),
            "http_events": http_events,
            "llm_events": recorder.llm_events,
            "search_events": recorder.search_events,
        }
        run_dir = _write_diagnostics(run_dir=ledger.run_dir, summary=summary, cases=cases)
        ledger.finish(outcome=summary["status"])
        return {
            **_run_flags(),
            "status": summary["status"],
            "output_dir": str(run_dir),
            "selection_sha256": prepared.selection_sha256,
            "completed_case_count": summary["completed_case_count"],
            "failed_case_count": summary["failed_case_count"],
            "requests": request_counts,
        }
    except BaseException as exc:
        ledger.fail(exc)
        raise


def _dry_run_public(prepared: PreparedSmoke) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        **_run_flags(),
        "status": "dry-run-pass",
        "http_performed": False,
        "writes_performed": False,
        "selection_sha256": prepared.selection_sha256,
        "fit_record_count": prepared.preflight_public["train_record_count"],
        "corpus_candidate_count": prepared.corpus_candidate_count,
        "eligible_candidate_count": prepared.eligible_candidate_count,
        "selected_cases": [_case_binding(case) for case in prepared.cases],
        "offline_filtered_case_count": sum(case.offline_filtered for case in prepared.cases),
        "request_budget": prepared.request_budget,
        "actual_requests": {"tavily": 0, "deepseek": 0},
        "live_output_root": str(DIAGNOSTICS_ROOT),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "live"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-ref", type=Path, default=DEFAULT_DATA_REF)
    parser.add_argument("--train-partition-ref", type=Path, default=DEFAULT_PARTITION_REF)
    parser.add_argument("--env-file", type=Path, default=REPOSITORY_ROOT / ".env")
    parser.add_argument("--output-root", type=Path, default=DIAGNOSTICS_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepared = prepare_smoke(
            config_path=args.config,
            data_ref=args.data_ref,
            train_partition_ref=args.train_partition_ref,
            env_file=args.env_file,
        )
        if args.mode == "dry-run":
            result = _dry_run_public(prepared)
        else:
            result = run_live(
                prepared,
                preflight_args={
                    "config_path": args.config,
                    "data_ref": args.data_ref,
                    "train_partition_ref": args.train_partition_ref,
                    "env_file": args.env_file,
                },
                output_root=args.output_root,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] in {"dry-run-pass", "complete"} else 3
    except SmokeLexiconError as exc:
        print(
            json.dumps(
                {"status": "blocked", "error_type": exc.__class__.__name__, "message": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": exc.__class__.__name__},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
