from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from utils.sqlite_kv_cache import SQLiteKVCache

logger = logging.getLogger(__name__)


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


class WebSearcher:
    """Small cached web-evidence adapter for lexicon construction.

    The module intentionally does not know about any existing project lexicon.
    It only takes a query and returns short evidence records from one configured
    external backend, or an empty list when disabled.
    """

    def __init__(self, settings: dict[str, Any] | None = None, cache_path: str | Path | None = None):
        settings = dict(settings or {})
        self.backend = str(settings.get("backend", settings.get("web_backend", "disabled")) or "disabled").lower()
        self.max_results = int(settings.get("max_results", 3) or 3)
        self.timeout = int(settings.get("timeout", 30) or 30)
        self.cache_enabled = bool(settings.get("cache_enabled", True))
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

        self.llm_api_base = (_resolve_env(settings.get("llm_api_base"), settings.get("llm_api_base_env")) or "").rstrip("/")
        self.llm_api_key = _resolve_env(settings.get("llm_api_key"), settings.get("llm_api_key_env"))
        self.llm_model = _resolve_env(settings.get("llm_model"), settings.get("llm_model_env"))

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()

    def search(self, query: str) -> list[dict[str, str]]:
        query = str(query or "").strip()
        if not query or self.backend == "disabled":
            return []

        cached = self._get_cache(query)
        if cached is not None:
            logger.debug("Web search cache hit: %s", query)
            return cached

        if self.backend == "search_api":
            logger.debug("Web search via search_api: %s", query)
            evidence = self._search_api(query)
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

    def _search_api(self, query: str) -> list[dict[str, str]]:
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

        if self.api_method == "POST":
            response = requests.post(self.api_base, json=params, headers=headers, timeout=self.timeout)
        else:
            response = requests.get(self.api_base, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        return self._normalize_search_results(payload, query)

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
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _parse_json_object(text)
        return self._normalize_search_results(parsed, query)

    def _normalize_search_results(self, payload: Any, query: str) -> list[dict[str, str]]:
        results = _dig_results(payload, self.api_results_path)
        normalized: list[dict[str, str]] = []
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
            evidence_id = _sha1_text(f"{query}\n{index}\n{title}\n{snippet}\n{url}")[:16]
            normalized.append(
                SearchEvidence(
                    id=evidence_id,
                    query=query,
                    title=title,
                    snippet=snippet,
                    url=url,
                    source=source,
                ).to_dict()
            )
        return normalized[: self.max_results]


class DisabledWebSearcher:
    def search(self, _query: str) -> list[dict[str, str]]:
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
