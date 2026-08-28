from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

import requests

from utils.sqlite_kv_cache import SQLiteKVCache

logger = logging.getLogger(__name__)

_QUOTED_QUERY_TERM_RE = re.compile(r'["“”]([^"“”]+)["“”]')
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_TRACKING_QUERY_KEYS = frozenset(
    {
        "dclid",
        "fbclid",
        "from",
        "gclid",
        "msclkid",
        "ref",
        "source",
        "tracking",
    }
)
DEFAULT_EVIDENCE_TEXT_LIMITS = {
    "max_title_chars": 180,
    "max_snippet_chars": 320,
    "max_url_chars": 2048,
    "max_source_chars": 80,
}


@dataclass
class SearchEvidence:
    id: str
    query: str
    title: str
    snippet: str
    url: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _stable_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha1_text(text: str) -> str:
    digest = hashlib.sha1()
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _canonical_result_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return ""
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return ""
    path = quote(unquote(parts.path), safe="/%:@-._~!$&'()*+,;=")
    path = path.rstrip("/") or "/"
    semantic_query = urlencode(
        [
            (key, item_value)
            for key, item_value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_")
            and key.casefold() not in _TRACKING_QUERY_KEYS
        ],
        doseq=True,
    )
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, semantic_query, "")
    )


def _query_candidate_term(query: str) -> str:
    match = _QUOTED_QUERY_TERM_RE.search(str(query or ""))
    if match:
        return match.group(1).strip()
    return str(query or "").strip().split(maxsplit=1)[0].strip("\"'“”")


def _normalized_match_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


def _bounded_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _positive_limit(settings: dict[str, Any], key: str) -> int:
    raw = settings.get(key, DEFAULT_EVIDENCE_TEXT_LIMITS[key])
    if isinstance(raw, bool):
        raise ValueError(f"web_settings.{key} must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"web_settings.{key} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"web_settings.{key} must be a positive integer")
    return value


def _direct_term_match(term: str, *values: Any) -> bool:
    needle = _normalized_match_text(term).strip()
    if not needle:
        return False
    haystack = "\n".join(_normalized_match_text(value) for value in values)
    if _CJK_RE.search(needle):
        return needle in haystack
    pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(needle)}(?![a-z0-9_])")
    return pattern.search(haystack) is not None


def _resolve_env(value: str | None, env_name: str | None = None, default: str | None = None) -> str | None:
    if value and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], default)
    if value and value.startswith("$") and len(value) > 1:
        return os.getenv(value[1:], default)
    if value:
        return value
    return os.getenv(env_name, default) if env_name else default


def _resolve_nested_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_nested_env(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_resolve_nested_env(item) for item in value]
    if isinstance(value, str) and (value.startswith("${") or value.startswith("$")):
        return _resolve_env(value)
    return value


def _is_retryable_search_transport_error(exc: Exception) -> bool:
    """Return whether a Tavily/search-api failure is safe to retry narrowly.

    JSON/schema failures are deliberately terminal: the retry allowance is for
    transient transport/provider availability only, not for masking a changed
    provider contract.
    """

    if isinstance(exc, requests.exceptions.JSONDecodeError):
        return False
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        status_code = getattr(response, "status_code", None)
        return status_code in {408, 425, 429} or (
            isinstance(status_code, int) and 500 <= status_code <= 599
        )
    return isinstance(
        exc,
        (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
    )


def _raise_for_search_status(response: requests.Response) -> None:
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and 300 <= status_code <= 399:
        raise requests.exceptions.HTTPError(
            f"refusing search API redirect status {status_code}",
            response=response,
        )
    response.raise_for_status()


class WebSearcher:
    """Small cached web-evidence adapter for lexicon construction.

    The module intentionally does not know about any existing project lexicon.
    It only takes a query and returns short evidence records from one configured
    external backend, or an empty list when disabled.
    """

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        cache_path: str | Path | None = None,
        *,
        formal_checkpoint: Any | None = None,
    ):
        settings = dict(settings or {})
        self.backend = str(settings.get("backend", settings.get("web_backend", "disabled")) or "disabled").lower()
        self.max_results = int(settings.get("max_results", 3) or 3)
        self.timeout = int(settings.get("timeout", 30) or 30)
        transport_retry_policy = dict(
            settings.get("transport_retry_policy", {}) or {}
        )
        raw_transport_retries = transport_retry_policy.get("retries", 0)
        raw_transport_sleep = transport_retry_policy.get("base_sleep_seconds", 1.0)
        if (
            isinstance(raw_transport_retries, bool)
            or not isinstance(raw_transport_retries, int)
            or raw_transport_retries < 0
        ):
            raise ValueError(
                "web_settings.transport_retry_policy.retries must be a non-negative integer"
            )
        if (
            isinstance(raw_transport_sleep, bool)
            or not isinstance(raw_transport_sleep, (int, float))
            or raw_transport_sleep < 0
        ):
            raise ValueError(
                "web_settings.transport_retry_policy.base_sleep_seconds must be non-negative"
            )
        self.transport_retry_policy_id = str(
            transport_retry_policy.get("id", "none/v1") or "none/v1"
        )
        self.transport_retries = raw_transport_retries
        self.transport_retry_sleep = float(raw_transport_sleep)
        self.cache_enabled = bool(settings.get("cache_enabled", True))
        self.require_direct_term_match = bool(settings.get("require_direct_term_match", False))
        self.dedupe_by_url = bool(settings.get("dedupe_by_url", True))
        self.max_title_chars = _positive_limit(settings, "max_title_chars")
        self.max_snippet_chars = _positive_limit(settings, "max_snippet_chars")
        self.max_url_chars = _positive_limit(settings, "max_url_chars")
        self.max_source_chars = _positive_limit(settings, "max_source_chars")
        resolved_cache_path = cache_path or settings.get("cache_path")
        self.cache = SQLiteKVCache(str(resolved_cache_path), enabled=self.cache_enabled) if resolved_cache_path else None

        self.api_key_env_name = settings.get("api_key_env")
        self.api_base = _resolve_env(settings.get("api_base"), settings.get("api_base_env"))
        self.api_key = _resolve_env(settings.get("api_key"), self.api_key_env_name)
        self.api_method = str(settings.get("api_method", "GET") or "GET").upper()
        self.api_query_param = str(settings.get("api_query_param", "q") or "q")
        self.api_key_header = settings.get("api_key_header")
        self.api_key_header_prefix = str(settings.get("api_key_header_prefix", "") or "")
        self.api_key_param = settings.get("api_key_param")
        self.api_extra_params = _resolve_nested_env(settings.get("api_extra_params", {}) or {})
        self.api_results_path = str(settings.get("api_results_path", "") or "")
        self.formal_checkpoint = formal_checkpoint

        self.llm_api_base = (_resolve_env(settings.get("llm_api_base"), settings.get("llm_api_base_env")) or "").rstrip("/")
        self.llm_api_key = _resolve_env(settings.get("llm_api_key"), settings.get("llm_api_key_env"))
        self.llm_model = _resolve_env(settings.get("llm_model"), settings.get("llm_model_env"))

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()

    def search(
        self,
        query: str,
        *,
        checkpoint_rank: int | None = None,
        checkpoint_slot: int | None = None,
        checkpoint_term: str | None = None,
    ) -> list[dict[str, str]]:
        query = str(query or "").strip()
        if not query or self.backend == "disabled":
            return []

        cached = self._get_cache(query)
        if cached is not None:
            logger.debug("Web search cache hit: %s", query)
            return cached

        if self.backend == "search_api":
            logger.debug("Web search via search_api: %s", query)
            evidence = self._search_api(
                query,
                checkpoint_rank=checkpoint_rank,
                checkpoint_slot=checkpoint_slot,
                checkpoint_term=checkpoint_term,
            )
        elif self.backend == "llm_web":
            logger.debug("Web search via llm_web: %s", query)
            evidence = self._llm_web_search(query)
        else:
            raise ValueError(f"Unsupported web search backend: {self.backend}")

        evidence = evidence[: self.max_results]
        self._set_cache(query, evidence)
        return evidence

    def _cache_key(self, query: str) -> str:
        payload = {
            "backend": self.backend,
            "query": query,
            "max_results": self.max_results,
            "require_direct_term_match": self.require_direct_term_match,
            "dedupe_by_url": self.dedupe_by_url,
            "max_title_chars": self.max_title_chars,
            "max_snippet_chars": self.max_snippet_chars,
            "max_url_chars": self.max_url_chars,
            "max_source_chars": self.max_source_chars,
            "api_base": self.api_base,
            "llm_api_base": self.llm_api_base,
            "llm_model": self.llm_model,
        }
        return _sha1_text(_stable_dumps(payload))

    def _get_cache(self, query: str) -> list[dict[str, str]] | None:
        if self.cache is None:
            return None
        raw = self.cache.get("web_search", self._cache_key(query))
        if raw is None:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if isinstance(value, list) else None

    def _set_cache(self, query: str, evidence: list[dict[str, str]]) -> None:
        if self.cache is None:
            return
        self.cache.set("web_search", self._cache_key(query), _stable_dumps(evidence).encode("utf-8"))

    def _search_api(
        self,
        query: str,
        *,
        checkpoint_rank: int | None = None,
        checkpoint_slot: int | None = None,
        checkpoint_term: str | None = None,
    ) -> list[dict[str, str]]:
        if not self.api_base:
            raise ValueError("web_settings.api_base or api_base_env is required for search_api backend.")
        if (self.api_key_header or self.api_key_param) and not self.api_key:
            key_source = self.api_key_env_name or "web_settings.api_key"
            raise ValueError(
                "Search API authentication is configured but no API key was resolved. "
                f"Set {key_source} in the environment or provide web_settings.api_key."
            )

        headers: dict[str, str] = {"accept": "application/json"}
        params: dict[str, Any] = {
            key: value for key, value in dict(self.api_extra_params).items() if value is not None
        }
        params[self.api_query_param] = query
        if self.api_key and self.api_key_header:
            headers[str(self.api_key_header)] = f"{self.api_key_header_prefix}{self.api_key}"
        if self.api_key and self.api_key_param:
            params[str(self.api_key_param)] = self.api_key

        if self.formal_checkpoint is not None:
            if (
                isinstance(checkpoint_rank, bool)
                or not isinstance(checkpoint_rank, int)
                or isinstance(checkpoint_slot, bool)
                or not isinstance(checkpoint_slot, int)
                or not isinstance(checkpoint_term, str)
                or not checkpoint_term
            ):
                raise ValueError(
                    "formal Web search requires candidate rank, slot, and term context"
                )
            return self._checkpointed_search_api(
                query=query,
                term=checkpoint_term,
                rank=checkpoint_rank,
                slot_number=checkpoint_slot,
                params=params,
                headers=headers,
            )

        last_error: Exception | None = None
        for attempt_index in range(self.transport_retries + 1):
            try:
                if self.api_method == "POST":
                    response = requests.post(
                        self.api_base,
                        json=params,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=False,
                    )
                else:
                    response = requests.get(
                        self.api_base,
                        params=params,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=False,
                    )
                _raise_for_search_status(response)
                payload = response.json()
                return self._normalize_search_results(payload, query)
            except Exception as exc:
                last_error = exc
                if (
                    attempt_index >= self.transport_retries
                    or not _is_retryable_search_transport_error(exc)
                ):
                    raise
                logger.warning(
                    "Web search transport retry %s/%s after %s",
                    attempt_index + 1,
                    self.transport_retries,
                    exc.__class__.__name__,
                )
                time.sleep(self.transport_retry_sleep * (attempt_index + 1))
        assert last_error is not None
        raise last_error

    def _checkpointed_search_api(
        self,
        *,
        query: str,
        term: str,
        rank: int,
        slot_number: int,
        params: dict[str, Any],
        headers: dict[str, str],
    ) -> list[dict[str, str]]:
        checkpoint = self.formal_checkpoint
        slot = f"query_{slot_number}"
        private_params = {
            key: value
            for key, value in params.items()
            if self.api_key_param is None or key != str(self.api_key_param)
        }
        request_payload = {
            "method": self.api_method,
            "url": self.api_base,
            "params": private_params,
            "timeout": self.timeout,
        }
        completed = checkpoint.get_slot_success(
            "tavily", rank, slot, request_payload=request_payload
        )
        if completed is not None:
            response_value = completed.response
            results = (
                response_value.get("results")
                if isinstance(response_value, Mapping)
                else None
            )
            if not isinstance(results, list) or not all(
                isinstance(item, dict) for item in results
            ):
                raise ValueError("checkpointed Tavily success has invalid normalized results")
            return [dict(item) for item in results]

        while True:
            reservation = checkpoint.reserve_attempt(
                "tavily", rank, slot, request_payload
            )
            response: Any | None = None
            try:
                if self.api_method == "POST":
                    response = requests.post(
                        self.api_base,
                        json=params,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=False,
                    )
                else:
                    response = requests.get(
                        self.api_base,
                        params=params,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=False,
                    )
                _raise_for_search_status(response)
                payload = response.json()
                normalized = self._normalize_search_results(payload, query)
            except Exception as exc:
                retryable = _is_retryable_search_transport_error(exc)
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                checkpoint.finish_attempt(
                    reservation,
                    status="retryable_failure" if retryable else "terminal_failure",
                    detail={
                        "error_type": exc.__class__.__name__,
                        "http_status": status_code,
                        "retryable": retryable,
                    },
                )
                if retryable and checkpoint.can_retry("tavily", rank, slot):
                    logger.warning(
                        "Tavily transport retry after physical attempt %s/%s",
                        reservation.attempt,
                        self.transport_retries + 1,
                    )
                    time.sleep(self.transport_retry_sleep * reservation.attempt)
                    continue
                raise

            logical_capture = {
                "term": term,
                "query": query,
                "results": normalized,
                "error": None,
            }
            checkpoint.finish_attempt(
                reservation,
                status="success",
                response={"results": normalized},
                capture=logical_capture,
                detail={
                    "error_type": None,
                    "http_status": getattr(response, "status_code", None),
                    "retryable": False,
                },
            )
            return normalized

    def _llm_web_search(self, query: str) -> list[dict[str, str]]:
        if not (self.llm_api_base and self.llm_api_key and self.llm_model):
            raise ValueError(
                "llm_web backend requires llm_api_base/llm_api_key/llm_model or their *_env settings."
            )

        url = self.llm_api_base.rstrip("/") + "/chat/completions"
        prompt = (
            "Search the web for evidence about this possible hate-speech lexicon term. "
            "Return only JSON with a top-level 'results' array. Each result must have "
            "title, snippet, url, and source. Query: "
            f"{query}"
        )
        response = requests.post(
            url,
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 800,
            },
            headers={"Authorization": f"Bearer {self.llm_api_key}", "Content-Type": "application/json"},
            timeout=self.timeout,
            allow_redirects=False,
        )
        _raise_for_search_status(response)
        data = response.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _parse_json_object(text)
        return self._normalize_search_results(parsed, query)

    def _normalize_search_results(self, payload: Any, query: str) -> list[dict[str, str]]:
        results = _dig_results(payload, self.api_results_path)
        normalized: list[dict[str, str]] = []
        seen_keys: set[str] = set()
        candidate_term = _query_candidate_term(query)
        for index, item in enumerate(results):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("name") or item.get("headline") or "").strip()
            snippet = str(
                item.get("snippet")
                or item.get("summary")
                or item.get("description")
                or item.get("content")
                or ""
            ).strip()
            url = str(item.get("url") or item.get("link") or item.get("href") or "").strip()
            source = str(
                item.get("source")
                or item.get("site")
                or item.get("provider")
                or item.get("displayLink")
                or self.backend
            ).strip()
            if not (title or snippet or url):
                continue
            canonical_url = _canonical_result_url(url)
            title = _bounded_text(title, self.max_title_chars)
            snippet = _bounded_text(snippet, self.max_snippet_chars)
            bounded_url = canonical_url
            if len(bounded_url) > self.max_url_chars:
                bounded_url = ""
            source = _bounded_text(source, self.max_source_chars)
            if not (title or snippet or bounded_url):
                continue
            if self.require_direct_term_match and not _direct_term_match(
                candidate_term,
                title,
                snippet,
                unquote(bounded_url),
            ):
                continue
            identity = bounded_url or _stable_dumps(
                {
                    "title": _normalized_match_text(title).strip(),
                    "snippet": _normalized_match_text(snippet).strip(),
                    "source": _normalized_match_text(source).strip(),
                }
            )
            dedupe_key = identity if self.dedupe_by_url else f"{query}\n{index}\n{identity}"
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            evidence_id = _sha1_text(dedupe_key)[:16]
            normalized.append(
                SearchEvidence(
                    id=evidence_id,
                    query=query,
                    title=title,
                    snippet=snippet,
                    url=bounded_url,
                    source=source,
                ).to_dict()
            )
        return normalized[: self.max_results]


class DisabledWebSearcher:
    def search(self, _query: str, **_context: Any) -> list[dict[str, str]]:
        return []

    def close(self) -> None:
        return None


def _dig_results(payload: Any, dotted_path: str = "") -> Iterable[Any]:
    if dotted_path:
        current = payload
        for part in dotted_path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = None
                break
        if isinstance(current, list):
            return current

    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("results", "items", "organic_results", "webPages", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if key == "webPages" and isinstance(value, dict) and isinstance(value.get("value"), list):
            return value["value"]
    return []


def _parse_json_object(text: str) -> Any:
    text = str(text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
