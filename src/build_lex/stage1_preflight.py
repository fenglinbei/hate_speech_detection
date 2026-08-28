"""Read-only, fail-closed preflight for formal Stage-1 lexicon builds.

The preflight deliberately does not construct HTTP clients, open SQLite caches,
mine candidates, create directories, or write reports.  Its public result only
contains credential *presence* and source metadata; credential values are kept
out of ``repr`` and serialized output.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit

from build_lex.train_only import (
    TrainOnlyLexiconError,
    _mint_formal_build_authorization,
    resolve_train_input,
)


PREFLIGHT_SCHEMA_VERSION = "stage1-formal-lexicon-preflight/v3"
FORMAL_CONFIG_SCHEMA_VERSION = "stage1-train-only-terminology-library-config/v1"
FORMAL_RESOURCE_ROLE = "terminology-understanding-library/v1"
FORMAL_CANDIDATE_OBJECTIVE = "terminology-value/v1"
FORMAL_CANDIDATE_SOURCE_POLICY = "content-only-no-task-labels/v1"
FORMAL_TAVILY_TRANSPORT_RETRY_POLICY = {
    "id": "tavily-transient/v1",
    "retries": 2,
    "base_sleep_seconds": 1.0,
}
FORMAL_TAVILY_PHYSICAL_ATTEMPT_BUDGET = {
    "scope_id": "stage1-p0-wp3-formal-full-tavily-key2/v1",
    "cap": 3100,
}
FORMAL_CHECKPOINT_POLICY = "provider-slot-checkpoint/v1"
FORMAL_AMBIGUOUS_ATTEMPT_POLICY = "count-and-retry-within-budget/v1"
FORMAL_MAX_LLM_HTTP_ATTEMPTS = 9000
ALLOWED_CREDENTIAL_ENV_NAMES = frozenset({"TAVILY_API_KEY", "DEEPSEEK_API_KEY"})

_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# Formal configs are protocol documents, not templates.  Reject every dollar
# marker rather than trying to maintain a second parser for all forms accepted
# by provider clients (including unusual environment names such as ``$9``).
_CONFIG_ENV_PLACEHOLDER_RE = re.compile(r"\$")
_TOP_LEVEL_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "resource_role",
        "candidate_settings",
        "web_settings",
        "llm_settings",
        "inclusion",
        "runtime_settings",
        "partition_policy",
    }
)
_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "llm_api_key",
        "password",
        "token",
        "access_token",
        "secret",
    }
)
_SECTION_ALLOWED_KEYS = {
    "candidate_settings": frozenset(
        {
            "max_candidates",
            "objective",
            "source_policy",
            "max_samples_per_candidate",
            "min_count_for_llm",
            "min_hate_count_for_llm",
            "zh_min_ngram",
            "zh_max_ngram",
            "zh_token_max_ngram",
            "use_jieba",
            "keep_all_content_ngrams",
            "suppressed_reject_hints",
            "en_max_ngram",
            "max_text_chars_per_record",
        }
    ),
    "web_settings": frozenset(
        {
            "backend",
            "api_base",
            "api_method",
            "api_query_param",
            "api_key_env",
            "api_key_header",
            "api_key_header_prefix",
            "api_extra_params",
            "api_results_path",
            "cache_enabled",
            "cache_path",
            "max_results",
            "timeout",
            "require_direct_term_match",
            "dedupe_by_url",
            "max_title_chars",
            "max_snippet_chars",
            "max_url_chars",
            "max_source_chars",
            "transport_retry_policy",
            "physical_attempt_budget",
        }
    ),
    "llm_settings": frozenset(
        {
            "backend",
            "provider",
            "api_base",
            "api_key_env",
            "model",
            "json_mode",
            "thinking",
            "reasoning_effort",
            "stream",
            "temperature",
            "send_temperature",
            "output_language",
            "max_tokens",
            "timeout",
            "retries",
            "retry_sleep",
        }
    ),
    "inclusion": frozenset(
        {
            "confidence_threshold",
            "single_mention_confidence",
            "min_count",
            "definition_required",
            "ambiguity_notes_required_when_ambiguous",
        }
    ),
    "runtime_settings": frozenset(
        {
            "show_progress",
            "log_level",
            "debug",
            "resume",
            "resume_require_config_match",
            "llm_failure_policy",
            "max_llm_failures",
            "max_consecutive_llm_failures",
            "formal_checkpoint_policy",
            "ambiguous_attempt_policy",
            "max_llm_http_attempts",
        }
    ),
    "partition_policy": frozenset(
        {
            "required_artifact_kind",
            "lexicon_source_partition",
            "excluded_partition",
            "calibration_contribution_count",
            "dependency_binding",
        }
    ),
}
_SECTION_REQUIRED_KEYS = {
    name: frozenset(keys)
    for name, keys in {
        "candidate_settings": {
            "objective",
            "source_policy",
            "max_candidates",
            "max_samples_per_candidate",
            "min_count_for_llm",
            "min_hate_count_for_llm",
            "zh_min_ngram",
            "zh_max_ngram",
            "zh_token_max_ngram",
            "use_jieba",
            "keep_all_content_ngrams",
            "suppressed_reject_hints",
            "en_max_ngram",
            "max_text_chars_per_record",
        },
        "web_settings": {
            "backend",
            "cache_enabled",
            "cache_path",
            "require_direct_term_match",
            "dedupe_by_url",
            "transport_retry_policy",
            "physical_attempt_budget",
        },
        "llm_settings": {
            "backend",
            "provider",
            "api_base",
            "api_key_env",
            "model",
            "json_mode",
            "thinking",
            "reasoning_effort",
            "stream",
            "temperature",
            "send_temperature",
            "output_language",
            "max_tokens",
            "timeout",
            "retries",
            "retry_sleep",
        },
        "inclusion": {
            "confidence_threshold",
            "single_mention_confidence",
            "min_count",
            "definition_required",
            "ambiguity_notes_required_when_ambiguous",
        },
        "runtime_settings": {
            "show_progress",
            "log_level",
            "debug",
            "resume",
            "resume_require_config_match",
            "llm_failure_policy",
            "max_llm_failures",
            "max_consecutive_llm_failures",
            "formal_checkpoint_policy",
            "ambiguous_attempt_policy",
            "max_llm_http_attempts",
        },
        "partition_policy": {
            "required_artifact_kind",
            "lexicon_source_partition",
            "excluded_partition",
            "calibration_contribution_count",
            "dependency_binding",
        },
    }.items()
}


class FormalLexiconPreflightError(ValueError):
    """Raised when the preflight itself cannot be evaluated safely."""


@dataclass(frozen=True)
class PreflightFinding:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class FormalLexiconPreflightResult:
    status: str
    dataset: str
    config_sha256: str | None
    data_build_id: str | None
    train_partition_id: str | None
    train_partition_payload_sha256: str | None
    train_record_count: int | None
    train_data_sha256: str | None
    train_ids_sha256: str | None
    web_backend: str | None
    llm_backend: str | None
    cache_policy: str
    request_budget: Mapping[str, int]
    credential_sources: Mapping[str, str]
    blockers: tuple[PreflightFinding, ...]
    _credential_values: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    _raw_config: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    _build_authorization: object | None = field(default=None, repr=False, compare=False)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report that can never contain credential values."""

        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": self.status,
            "dataset": self.dataset,
            "config_sha256": self.config_sha256,
            "data_build_id": self.data_build_id,
            "train_partition_id": self.train_partition_id,
            "train_partition_payload_sha256": self.train_partition_payload_sha256,
            "train_record_count": self.train_record_count,
            "train_data_sha256": self.train_data_sha256,
            "train_ids_sha256": self.train_ids_sha256,
            "web_backend": self.web_backend,
            "llm_backend": self.llm_backend,
            "cache_policy": self.cache_policy,
            "request_budget": dict(sorted(self.request_budget.items())),
            "credential_sources": dict(sorted(self.credential_sources.items())),
            "blockers": [finding.to_dict() for finding in self.blockers],
        }

    def activate_credentials(self, environ: MutableMapping[str, str] | None = None) -> None:
        """Install preflighted values in the current process immediately before build.

        This is intentionally separate from :func:`preflight_formal_lexicon`, so
        the preflight itself stays side-effect free.  Only allowlisted variables
        may be installed and an existing, different value is never overwritten.
        """

        if not self.passed:
            raise FormalLexiconPreflightError("cannot activate credentials after a blocked preflight")
        target = os.environ if environ is None else environ
        for name, value in self._credential_values.items():
            if name not in ALLOWED_CREDENTIAL_ENV_NAMES:
                raise FormalLexiconPreflightError("preflight contains a non-allowlisted credential name")
            existing = target.get(name)
            if existing and existing != value:
                raise FormalLexiconPreflightError(
                    f"refusing to overwrite an existing credential variable: {name}"
                )
            if not existing:
                target[name] = value

    def frozen_config(self) -> dict[str, Any]:
        """Return the exact config object checked by this successful preflight."""

        if not self.passed or self._raw_config is None:
            raise FormalLexiconPreflightError("cannot use config from a blocked preflight")
        return copy.deepcopy(dict(self._raw_config))

    @property
    def build_authorization(self) -> object:
        """Return the opaque one-use capability minted by a passing preflight."""

        if not self.passed or self._build_authorization is None:
            raise FormalLexiconPreflightError(
                "cannot authorize a formal build after a blocked preflight"
            )
        return self._build_authorization


class _DuplicateJSONKey(ValueError):
    pass


class _NonFiniteJSONNumber(ValueError):
    pass


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJSONKey(key)
        value[key] = item
    return value


def _reject_nonfinite_number(value: str) -> Any:
    raise _NonFiniteJSONNumber(value)


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise FormalLexiconPreflightError("formal lexicon config must not be a symlink")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_number,
            )
    except FileNotFoundError as exc:
        raise FormalLexiconPreflightError("formal lexicon config is missing") from exc
    except _DuplicateJSONKey as exc:
        raise FormalLexiconPreflightError(f"formal lexicon config has duplicate key: {exc}") from exc
    except _NonFiniteJSONNumber as exc:
        raise FormalLexiconPreflightError(
            f"formal lexicon config contains a non-finite JSON number: {exc}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalLexiconPreflightError("formal lexicon config is not readable canonical JSON") from exc
    if not isinstance(value, dict):
        raise FormalLexiconPreflightError("formal lexicon config must be a JSON object")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_env_value(raw: str, *, name: str, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {'"', "'"}:
        quote = value[0]
        end = value.find(quote, 1)
        if end < 0:
            raise FormalLexiconPreflightError(
                f"env file has an unterminated quoted value for {name} at line {line_number}"
            )
        tail = value[end + 1 :].strip()
        if tail and not tail.startswith("#"):
            raise FormalLexiconPreflightError(
                f"env file has trailing content for {name} at line {line_number}"
            )
        # Keep the quoted text literal.  In particular, do not expand $NAME,
        # ${NAME}, command substitutions, or backslash escapes.
        return value[1:end]
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    if any(character.isspace() for character in value):
        raise FormalLexiconPreflightError(
            f"env file has an unquoted whitespace value for {name} at line {line_number}"
        )
    return value


def parse_allowlisted_env_file(
    path: str | Path | None,
    *,
    allowlist: Sequence[str] = tuple(sorted(ALLOWED_CREDENTIAL_ENV_NAMES)),
) -> dict[str, str]:
    """Read only allowlisted assignments from ``.env`` without shell expansion."""

    allowed = frozenset(str(name) for name in allowlist)
    if not allowed.issubset(ALLOWED_CREDENTIAL_ENV_NAMES):
        raise FormalLexiconPreflightError("env allowlist contains an unsupported credential name")
    if path is None:
        return {}
    env_path = Path(path)
    if not env_path.exists():
        return {}
    if env_path.is_symlink() or not env_path.is_file():
        raise FormalLexiconPreflightError("env file must be a regular non-symlink file")

    result: dict[str, str] = {}
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise FormalLexiconPreflightError("env file is not readable UTF-8 text") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.lstrip("\ufeff") if line_number == 1 else raw_line
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            continue
        raw_name, raw_value = stripped.split("=", 1)
        name = raw_name.strip()
        if not _ENV_NAME_RE.fullmatch(name) or name not in allowed:
            continue
        if name in result:
            raise FormalLexiconPreflightError(f"env file assigns {name} more than once")
        result[name] = _parse_env_value(raw_value, name=name, line_number=line_number)
    return result


def _mapping(value: Any, section: str, blockers: list[PreflightFinding]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        blockers.append(PreflightFinding("CONFIG_SECTION_INVALID", f"{section} must be an object"))
        return {}
    return value


def _has_inline_secrets(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, inner in value.items():
            key_text = str(key)
            normalized = key_text.lower()
            next_path = (*path, key_text)
            if normalized in _SECRET_FIELD_NAMES or normalized.endswith(("_password", "_secret")):
                if inner is not None and inner != "":
                    findings.append(".".join(next_path))
            findings.extend(_has_inline_secrets(inner, next_path))
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            findings.extend(_has_inline_secrets(inner, (*path, str(index))))
    return findings


def _config_placeholder_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, inner in value.items():
            key_text = str(key)
            next_path = (*path, key_text)
            if _CONFIG_ENV_PLACEHOLDER_RE.search(key_text):
                findings.append(".".join(next_path) + " (key)")
            findings.extend(_config_placeholder_paths(inner, next_path))
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            findings.extend(_config_placeholder_paths(inner, (*path, str(index))))
    elif isinstance(value, str) and _CONFIG_ENV_PLACEHOLDER_RE.search(value):
        findings.append(".".join(path) or "<root>")
    return findings


def _is_positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _is_nonnegative_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= 0


def _validate_https_endpoint(
    value: Any,
    *,
    field_name: str,
    allowed_host: str,
    allowed_path: str,
    blockers: list[PreflightFinding],
) -> None:
    if not isinstance(value, str) or not value:
        blockers.append(PreflightFinding("ENDPOINT_MISSING", f"{field_name} must be explicit"))
        return
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != allowed_host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != allowed_path.rstrip("/")
    ):
        blockers.append(
            PreflightFinding(
                "ENDPOINT_NOT_ALLOWLISTED",
                f"{field_name} must use the frozen HTTPS endpoint on {allowed_host}",
            )
        )


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _git_tracks_path(repository_root: Path, path: Path) -> bool:
    if not (repository_root / ".git").exists() or not _path_is_relative_to(path, repository_root):
        return False
    relative = path.relative_to(repository_root).as_posix()
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "ls-files", "--error-unmatch", "--", relative],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        # If Git state cannot be inspected, a configured cache cannot be proven
        # untracked.  The caller separately blocks every enabled formal cache.
        return False
    return completed.returncode == 0


def _append_cache_path_findings(
    resolved: Path,
    *,
    repository_root: Path,
    blockers: list[PreflightFinding],
) -> None:
    legacy_root = (repository_root / "data" / "lexicon" / "generated").resolve()
    if _path_is_relative_to(resolved, legacy_root):
        blockers.append(
            PreflightFinding(
                "LEGACY_CACHE_PATH",
                "formal WP3 must not use a cache under data/lexicon/generated",
            )
        )
    if _git_tracks_path(repository_root, resolved):
        blockers.append(
            PreflightFinding("TRACKED_CACHE_PATH", "formal WP3 must not use a Git-tracked cache file")
        )
    if resolved.exists():
        blockers.append(
            PreflightFinding("SHARED_CACHE_PATH", "formal WP3 must not open an existing/shared cache file")
        )


def _validate_cache_policy(
    web: Mapping[str, Any],
    *,
    repository_root: Path,
    blockers: list[PreflightFinding],
) -> str:
    if "cache_enabled" not in web or not isinstance(web.get("cache_enabled"), bool):
        blockers.append(
            PreflightFinding(
                "CACHE_POLICY_UNRESOLVED",
                "formal cache policy must explicitly set web_settings.cache_enabled=false; no implicit default is allowed",
            )
        )
        cache_enabled = None
        # The legacy builder's implicit default is known and unsafe.  Surface
        # its concrete risk instead of reporting only that the field is absent.
        inherited = (repository_root / "data" / "lexicon" / "generated" / "full" / "web_cache.sqlite3").resolve()
        _append_cache_path_findings(inherited, repository_root=repository_root, blockers=blockers)
    else:
        cache_enabled = bool(web["cache_enabled"])
        if cache_enabled:
            blockers.append(
                PreflightFinding(
                    "FORMAL_CACHE_ENABLED",
                    "formal WP3 cache reuse is not approved; choose and freeze an isolated versioned policy or set cache_enabled=false",
                )
            )

    raw_cache_path = web.get("cache_path")
    if raw_cache_path is not None and raw_cache_path != "":
        if not isinstance(raw_cache_path, str):
            blockers.append(PreflightFinding("CACHE_PATH_INVALID", "web_settings.cache_path must be a string"))
            return "blocked"
        cache_path = Path(raw_cache_path)
        resolved = cache_path.resolve() if cache_path.is_absolute() else (repository_root / cache_path).resolve()
        _append_cache_path_findings(resolved, repository_root=repository_root, blockers=blockers)
        if cache_enabled is False:
            blockers.append(
                PreflightFinding(
                    "DISABLED_CACHE_HAS_PATH",
                    "remove web_settings.cache_path when formal caching is disabled",
                )
            )
    return "disabled" if cache_enabled is False and (raw_cache_path is None or raw_cache_path == "") else "blocked"


def _credential_requirements(
    web: Mapping[str, Any],
    llm: Mapping[str, Any],
    blockers: list[PreflightFinding],
) -> tuple[str, ...]:
    required: list[str] = []
    web_backend = str(web.get("backend", "") or "").lower()
    for key in ("require_direct_term_match", "dedupe_by_url"):
        if not isinstance(web.get(key), bool):
            blockers.append(
                PreflightFinding(
                    "WEB_EVIDENCE_POLICY_INVALID",
                    f"web_settings.{key} must be an explicit boolean",
                )
            )
    if web_backend == "search_api":
        if any(web.get(key) is not True for key in ("require_direct_term_match", "dedupe_by_url")):
            blockers.append(
                PreflightFinding(
                    "WEB_EVIDENCE_POLICY_DISABLED",
                    "formal Web evidence requires direct candidate matching and canonical URL deduplication",
                )
            )
        env_name = web.get("api_key_env")
        if env_name != "TAVILY_API_KEY":
            blockers.append(
                PreflightFinding(
                    "WEB_CREDENTIAL_BINDING_INVALID",
                    "search_api must bind credentials through TAVILY_API_KEY",
                )
            )
        else:
            required.append(env_name)
    elif web_backend != "disabled":
        blockers.append(
            PreflightFinding(
                "WEB_BACKEND_UNSUPPORTED",
                "formal WP3 supports only search_api or an explicitly frozen disabled backend",
            )
        )

    llm_backend = str(llm.get("backend", "") or "").lower()
    if llm_backend != "deepseek" or str(llm.get("provider", "") or "").lower() != "deepseek":
        blockers.append(
            PreflightFinding(
                "LLM_BACKEND_UNSUPPORTED",
                "formal WP3 currently requires the frozen DeepSeek judgement backend",
            )
        )
    env_name = llm.get("api_key_env")
    if env_name != "DEEPSEEK_API_KEY":
        blockers.append(
            PreflightFinding(
                "LLM_CREDENTIAL_BINDING_INVALID",
                "DeepSeek must bind credentials through DEEPSEEK_API_KEY",
            )
        )
    else:
        required.append(env_name)
    return tuple(dict.fromkeys(required))


def _validate_config(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    blockers: list[PreflightFinding],
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    if config.get("schema_version") != FORMAL_CONFIG_SCHEMA_VERSION:
        blockers.append(
            PreflightFinding("CONFIG_SCHEMA_INVALID", f"schema_version must be {FORMAL_CONFIG_SCHEMA_VERSION}")
        )
    if config.get("resource_role") != FORMAL_RESOURCE_ROLE:
        blockers.append(
            PreflightFinding(
                "RESOURCE_ROLE_INVALID",
                f"resource_role must be {FORMAL_RESOURCE_ROLE}",
            )
        )
    keys = frozenset(config)
    missing = sorted(_TOP_LEVEL_CONFIG_KEYS - keys)
    unexpected = sorted(keys - _TOP_LEVEL_CONFIG_KEYS)
    if missing:
        blockers.append(PreflightFinding("CONFIG_KEYS_MISSING", "missing top-level fields: " + ", ".join(missing)))
    if unexpected:
        blockers.append(
            PreflightFinding("CONFIG_KEYS_UNEXPECTED", "unexpected top-level fields: " + ", ".join(unexpected))
        )
    inline_secrets = _has_inline_secrets(config)
    if inline_secrets:
        blockers.append(
            PreflightFinding(
                "INLINE_SECRET_FORBIDDEN",
                "literal secrets are forbidden in config fields: " + ", ".join(sorted(inline_secrets)),
            )
        )
    placeholder_paths = _config_placeholder_paths(config)
    if placeholder_paths:
        blockers.append(
            PreflightFinding(
                "CONFIG_ENV_PLACEHOLDER_FORBIDDEN",
                "formal config must not contain environment placeholders: "
                + ", ".join(sorted(placeholder_paths)),
            )
        )

    sections: dict[str, Mapping[str, Any]] = {}
    for name in _SECTION_ALLOWED_KEYS:
        section = _mapping(config.get(name), name, blockers)
        sections[name] = section
        unexpected_section_keys = sorted(set(section) - _SECTION_ALLOWED_KEYS[name])
        missing_section_keys = sorted(_SECTION_REQUIRED_KEYS[name] - set(section))
        if unexpected_section_keys:
            blockers.append(
                PreflightFinding(
                    "CONFIG_SECTION_KEYS_UNEXPECTED",
                    f"{name} has unexpected fields: " + ", ".join(unexpected_section_keys),
                )
            )
        if missing_section_keys:
            blockers.append(
                PreflightFinding(
                    "CONFIG_SECTION_KEYS_MISSING",
                    f"{name} is missing fields: " + ", ".join(missing_section_keys),
                )
            )
    web = sections["web_settings"]
    llm = sections["llm_settings"]
    runtime = sections["runtime_settings"]
    candidate = sections["candidate_settings"]
    inclusion = sections["inclusion"]
    partition_policy = sections["partition_policy"]
    expected_partition_policy = {
        "required_artifact_kind": "train-partition",
        "lexicon_source_partition": "fit",
        "excluded_partition": "calibration",
        "calibration_contribution_count": 0,
        "dependency_binding": "exact-portable-dependency/v1",
    }
    if dict(partition_policy) != expected_partition_policy:
        blockers.append(
            PreflightFinding(
                "PARTITION_POLICY_INVALID",
                "partition_policy must freeze fit-only terminology contribution and exact dependency binding",
            )
        )

    for key in (
        "max_candidates",
        "max_samples_per_candidate",
        "min_count_for_llm",
        "zh_min_ngram",
        "zh_max_ngram",
        "zh_token_max_ngram",
        "en_max_ngram",
        "max_text_chars_per_record",
    ):
        if not _is_positive_int(candidate.get(key)):
            blockers.append(
                PreflightFinding(
                    "CANDIDATE_SETTING_INVALID",
                    f"candidate_settings.{key} must be a positive integer",
                )
            )
    if candidate.get("min_hate_count_for_llm") != 0:
        blockers.append(
            PreflightFinding(
                "CANDIDATE_SETTING_INVALID",
                "candidate_settings.min_hate_count_for_llm must be 0 for the category-free terminology objective",
            )
        )
    if candidate.get("objective") != FORMAL_CANDIDATE_OBJECTIVE:
        blockers.append(
            PreflightFinding(
                "CANDIDATE_SETTING_INVALID",
                f"candidate_settings.objective must be {FORMAL_CANDIDATE_OBJECTIVE}",
            )
        )
    if candidate.get("source_policy") != FORMAL_CANDIDATE_SOURCE_POLICY:
        blockers.append(
            PreflightFinding(
                "CANDIDATE_SETTING_INVALID",
                f"candidate_settings.source_policy must be {FORMAL_CANDIDATE_SOURCE_POLICY}",
            )
        )
    if all(
        _is_positive_int(candidate.get(key))
        for key in ("zh_min_ngram", "zh_max_ngram", "zh_token_max_ngram")
    ) and not (
        candidate["zh_min_ngram"]
        <= candidate["zh_max_ngram"]
        <= candidate["zh_token_max_ngram"]
    ):
        blockers.append(
            PreflightFinding(
                "CANDIDATE_NGRAM_RANGE_INVALID",
                "candidate_settings Chinese n-gram bounds must be monotonically ordered",
            )
        )
    for key in ("use_jieba", "keep_all_content_ngrams"):
        if not isinstance(candidate.get(key), bool):
            blockers.append(
                PreflightFinding(
                    "CANDIDATE_SETTING_INVALID",
                    f"candidate_settings.{key} must be boolean",
                )
            )
    if candidate.get("use_jieba") is not False:
        blockers.append(
            PreflightFinding(
                "UNFROZEN_OPTIONAL_SEGMENTER",
                "formal candidate mining requires use_jieba=false unless a versioned segmenter dependency is added to the protocol",
            )
        )
    if candidate.get("keep_all_content_ngrams") is not True:
        blockers.append(
            PreflightFinding(
                "NARROW_TERMINOLOGY_CANDIDATE_FRAME",
                "formal terminology mining must keep all plausible content n-grams before utility judging",
            )
        )
    suppressed = candidate.get("suppressed_reject_hints")
    if (
        not isinstance(suppressed, list)
        or not all(isinstance(value, str) and value.strip() == value and value for value in suppressed)
        or len(suppressed) != len(set(suppressed))
    ):
        blockers.append(
            PreflightFinding(
                "CANDIDATE_SETTING_INVALID",
                "candidate_settings.suppressed_reject_hints must be a unique canonical string array",
            )
        )
    elif "singleton_ngram" in suppressed:
        blockers.append(
            PreflightFinding(
                "NARROW_TERMINOLOGY_CANDIDATE_FRAME",
                "singleton n-grams must reach the terminology judge; high-confidence singleton inclusion is governed downstream",
            )
        )

    for key in ("confidence_threshold", "single_mention_confidence"):
        value = inclusion.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            blockers.append(
                PreflightFinding(
                    "INCLUSION_SETTING_INVALID",
                    f"inclusion.{key} must be a number from 0 through 1",
                )
            )
    if not _is_positive_int(inclusion.get("min_count")):
        blockers.append(
            PreflightFinding("INCLUSION_SETTING_INVALID", "inclusion.min_count must be positive")
        )
    for key in (
        "definition_required",
        "ambiguity_notes_required_when_ambiguous",
    ):
        if inclusion.get(key) is not True:
            blockers.append(
                PreflightFinding(
                    "INCLUSION_SETTING_INVALID",
                    f"inclusion.{key} must be true",
                )
            )

    web_backend = str(web.get("backend", "") or "").lower()
    transport_retry_policy = web.get("transport_retry_policy")
    if (
        not isinstance(transport_retry_policy, Mapping)
        or set(transport_retry_policy) != set(FORMAL_TAVILY_TRANSPORT_RETRY_POLICY)
        or transport_retry_policy.get("id")
        != FORMAL_TAVILY_TRANSPORT_RETRY_POLICY["id"]
        or isinstance(transport_retry_policy.get("retries"), bool)
        or not isinstance(transport_retry_policy.get("retries"), int)
        or transport_retry_policy.get("retries")
        != FORMAL_TAVILY_TRANSPORT_RETRY_POLICY["retries"]
        or isinstance(transport_retry_policy.get("base_sleep_seconds"), bool)
        or not isinstance(
            transport_retry_policy.get("base_sleep_seconds"), (int, float)
        )
        or transport_retry_policy.get("base_sleep_seconds")
        != FORMAL_TAVILY_TRANSPORT_RETRY_POLICY["base_sleep_seconds"]
    ):
        blockers.append(
            PreflightFinding(
                "WEB_TRANSPORT_RETRY_POLICY_INVALID",
                "formal WP3 requires the exact tavily-transient/v1 two-retry transport policy",
            )
        )
    physical_attempt_budget = web.get("physical_attempt_budget")
    if (
        not isinstance(physical_attempt_budget, Mapping)
        or set(physical_attempt_budget)
        != set(FORMAL_TAVILY_PHYSICAL_ATTEMPT_BUDGET)
        or physical_attempt_budget.get("scope_id")
        != FORMAL_TAVILY_PHYSICAL_ATTEMPT_BUDGET["scope_id"]
        or isinstance(physical_attempt_budget.get("cap"), bool)
        or not isinstance(physical_attempt_budget.get("cap"), int)
        or physical_attempt_budget.get("cap")
        != FORMAL_TAVILY_PHYSICAL_ATTEMPT_BUDGET["cap"]
    ):
        blockers.append(
            PreflightFinding(
                "WEB_PHYSICAL_ATTEMPT_BUDGET_INVALID",
                "formal WP3 requires the exact key2 Tavily physical-attempt scope and cap=3100",
            )
        )
    if web_backend == "search_api":
        evidence_text_limits = {
            "max_title_chars": 180,
            "max_snippet_chars": 320,
            "max_url_chars": 2048,
            "max_source_chars": 80,
        }
        for key, maximum in evidence_text_limits.items():
            value = web.get(key)
            if not _is_positive_int(value) or value > maximum:
                blockers.append(
                    PreflightFinding(
                        "WEB_EVIDENCE_TEXT_LIMIT_INVALID",
                        f"web_settings.{key} must be an integer from 1 through {maximum}",
                    )
                )
        web_extra = web.get("api_extra_params")
        if not isinstance(web_extra, Mapping) or set(web_extra) != {"max_results", "search_depth"}:
            blockers.append(
                PreflightFinding(
                    "WEB_EXTRA_PARAMS_INVALID",
                    "web_settings.api_extra_params must contain exactly max_results and search_depth",
                )
            )
        else:
            if not _is_positive_int(web_extra.get("max_results")) or web_extra.get("max_results") > 10:
                blockers.append(
                    PreflightFinding(
                        "WEB_MAX_RESULTS_INVALID",
                        "web_settings.api_extra_params.max_results must be an integer from 1 through 10",
                    )
                )
            if web_extra.get("search_depth") not in {"basic", "advanced"}:
                blockers.append(
                    PreflightFinding(
                        "WEB_SEARCH_DEPTH_INVALID",
                        "web_settings.api_extra_params.search_depth must be basic or advanced",
                    )
                )
            outer_max_results = web.get("max_results")
            if (
                not _is_positive_int(outer_max_results)
                or outer_max_results != web_extra.get("max_results")
            ):
                blockers.append(
                    PreflightFinding(
                        "WEB_RESULT_LIMIT_MISMATCH",
                        "web max_results must equal api_extra_params.max_results when explicitly supplied",
                    )
                )
        if not _is_positive_int(web.get("timeout")) or web.get("timeout") > 300:
            blockers.append(
                PreflightFinding(
                    "WEB_TIMEOUT_INVALID",
                    "web_settings.timeout must be an integer from 1 through 300",
                )
            )
        _validate_https_endpoint(
            web.get("api_base"),
            field_name="web_settings.api_base",
            allowed_host="api.tavily.com",
            allowed_path="/search",
            blockers=blockers,
        )
        if web.get("api_method") != "POST":
            blockers.append(PreflightFinding("WEB_METHOD_INVALID", "Tavily endpoint requires POST"))
        if web.get("api_query_param") != "query" or web.get("api_results_path") != "results":
            blockers.append(
                PreflightFinding("WEB_PROTOCOL_INVALID", "Tavily query/results field mapping is not frozen correctly")
            )
        if web.get("api_key_header") != "Authorization" or web.get("api_key_header_prefix") != "Bearer ":
            blockers.append(
                PreflightFinding("WEB_AUTH_PROTOCOL_INVALID", "Tavily Authorization bearer binding is invalid")
            )
    elif web_backend == "disabled":
        stale_endpoint_fields = sorted(
            key
            for key in (
                "api_base",
                "api_key_env",
                "api_method",
                "api_query_param",
                "api_key_header",
                "api_key_header_prefix",
                "api_extra_params",
                "api_results_path",
                "max_results",
                "timeout",
            )
            if web.get(key) is not None and web.get(key) != ""
        )
        if stale_endpoint_fields:
            blockers.append(
                PreflightFinding(
                    "DISABLED_WEB_HAS_ENDPOINT",
                    "disabled web backend must remove endpoint/auth fields: " + ", ".join(stale_endpoint_fields),
                )
            )

    _validate_https_endpoint(
        llm.get("api_base"),
        field_name="llm_settings.api_base",
        allowed_host="api.deepseek.com",
        allowed_path="",
        blockers=blockers,
    )
    model = llm.get("model")
    if model != "deepseek-v4-flash":
        blockers.append(
            PreflightFinding(
                "LLM_MODEL_INVALID",
                "formal WP3 requires llm_settings.model=deepseek-v4-flash",
            )
        )
    if llm.get("json_mode") is not True or llm.get("stream") is not False:
        blockers.append(
            PreflightFinding("LLM_PROTOCOL_INVALID", "formal judgements require json_mode=true and stream=false")
        )
    temperature = llm.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or temperature != 0
        or llm.get("send_temperature") is not True
    ):
        blockers.append(
            PreflightFinding("LLM_DETERMINISM_INVALID", "formal judgements require an explicit temperature=0 request")
        )
    thinking = llm.get("thinking")
    if (
        not isinstance(thinking, Mapping)
        or set(thinking) != {"type"}
        or thinking.get("type") != "disabled"
    ):
        blockers.append(
            PreflightFinding(
                "LLM_THINKING_INVALID",
                "formal WP3 requires llm_settings.thinking={'type':'disabled'}",
            )
        )
    elif llm.get("reasoning_effort") is not None:
        blockers.append(
            PreflightFinding(
                "LLM_REASONING_INVALID",
                "reasoning_effort must be null when thinking is disabled",
            )
        )
    retries = llm.get("retries")
    retry_sleep = llm.get("retry_sleep")
    if retries != 2:
        blockers.append(
            PreflightFinding(
                "LLM_RETRIES_INVALID",
                "formal WP3 requires llm_settings.retries=2",
            )
        )
    if (
        isinstance(retry_sleep, bool)
        or not isinstance(retry_sleep, (int, float))
        or retry_sleep != 1.0
    ):
        blockers.append(
            PreflightFinding(
                "LLM_RETRY_SLEEP_INVALID",
                "formal WP3 requires llm_settings.retry_sleep=1.0",
            )
        )
    if llm.get("max_tokens") != 1024:
        blockers.append(
            PreflightFinding(
                "LLM_MAX_TOKENS_INVALID",
                "formal WP3 requires llm_settings.max_tokens=1024",
            )
        )
    if not _is_positive_int(llm.get("timeout")) or llm.get("timeout") > 600:
        blockers.append(
            PreflightFinding("LLM_TIMEOUT_INVALID", "llm_settings.timeout must be an integer from 1 through 600")
        )
    if llm.get("output_language") != "zh":
        blockers.append(
            PreflightFinding(
                "LLM_OUTPUT_LANGUAGE_INVALID",
                "formal WP3 requires llm_settings.output_language=zh",
            )
        )

    if runtime.get("resume") is not True:
        blockers.append(
            PreflightFinding(
                "RESUME_REQUIRED",
                "formal WP3 requires runtime_settings.resume=true for provider-slot checkpointing",
            )
        )
    if runtime.get("resume_require_config_match") is not True:
        blockers.append(
            PreflightFinding("RESUME_GUARD_INVALID", "runtime_settings.resume_require_config_match must be true")
        )
    if runtime.get("formal_checkpoint_policy") != FORMAL_CHECKPOINT_POLICY:
        blockers.append(
            PreflightFinding(
                "FORMAL_CHECKPOINT_POLICY_INVALID",
                f"formal WP3 requires runtime_settings.formal_checkpoint_policy={FORMAL_CHECKPOINT_POLICY}",
            )
        )
    if runtime.get("ambiguous_attempt_policy") != FORMAL_AMBIGUOUS_ATTEMPT_POLICY:
        blockers.append(
            PreflightFinding(
                "AMBIGUOUS_ATTEMPT_POLICY_INVALID",
                "formal WP3 requires count-and-retry-within-budget handling for ambiguous provider attempts",
            )
        )
    max_llm_http_attempts = runtime.get("max_llm_http_attempts")
    if (
        isinstance(max_llm_http_attempts, bool)
        or not isinstance(max_llm_http_attempts, int)
        or max_llm_http_attempts != FORMAL_MAX_LLM_HTTP_ATTEMPTS
    ):
        blockers.append(
            PreflightFinding(
                "MAX_LLM_HTTP_ATTEMPTS_INVALID",
                f"formal WP3 requires runtime_settings.max_llm_http_attempts={FORMAL_MAX_LLM_HTTP_ATTEMPTS}",
            )
        )
    if runtime.get("llm_failure_policy") != "raise":
        blockers.append(PreflightFinding("LLM_FAILURE_POLICY_INVALID", "formal WP3 requires llm_failure_policy=raise"))
    if runtime.get("debug") is not False:
        blockers.append(
            PreflightFinding(
                "DEBUG_POLICY_INVALID",
                "formal config must set debug=false; the wrapper owns the immutable capture directory",
            )
        )
    if not isinstance(runtime.get("show_progress"), bool):
        blockers.append(
            PreflightFinding("RUNTIME_SETTING_INVALID", "runtime_settings.show_progress must be boolean")
        )
    if runtime.get("log_level") not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        blockers.append(
            PreflightFinding("RUNTIME_SETTING_INVALID", "runtime_settings.log_level is not canonical")
        )
    for key in ("max_llm_failures", "max_consecutive_llm_failures"):
        if not _is_positive_int(runtime.get(key)):
            blockers.append(
                PreflightFinding(
                    "RUNTIME_SETTING_INVALID",
                    f"runtime_settings.{key} must be a positive integer",
                )
            )

    cache_policy = _validate_cache_policy(web, repository_root=repository_root, blockers=blockers)
    return web, llm, cache_policy


def preflight_formal_lexicon(
    *,
    dataset: str = "full",
    config_path: str | Path | None,
    data_ref: str | Path | None,
    env_file: str | Path | None,
    repository_root: str | Path,
    train_partition_ref: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> FormalLexiconPreflightResult:
    """Evaluate all formal WP3 gates without writing or constructing clients."""

    blockers: list[PreflightFinding] = []
    config_sha256: str | None = None
    data_build_id: str | None = None
    train_partition_id: str | None = None
    train_partition_payload_sha256: str | None = None
    train_record_count: int | None = None
    train_data_sha256: str | None = None
    train_ids_sha256: str | None = None
    web: Mapping[str, Any] = {}
    llm: Mapping[str, Any] = {}
    cache_policy = "blocked"
    config: dict[str, Any] | None = None
    request_budget: dict[str, int] = {}
    repository = Path(repository_root).resolve()
    normalized_dataset = str(dataset or "").strip().lower()
    if normalized_dataset != "full" or dataset != normalized_dataset:
        blockers.append(
            PreflightFinding(
                "DATASET_UNSUPPORTED",
                "formal Stage-1 WP3 requires the normalized dataset name full",
            )
        )

    if config_path is None:
        blockers.append(
            PreflightFinding("CONFIG_MISSING", "formal WP3 requires an explicit frozen config file")
        )
    else:
        try:
            config = _load_json_object(Path(config_path))
            config_sha256 = _canonical_sha256(config)
            web, llm, cache_policy = _validate_config(
                config,
                repository_root=repository,
                blockers=blockers,
            )
        except FormalLexiconPreflightError as exc:
            blockers.append(PreflightFinding("CONFIG_UNREADABLE", str(exc)))

    required_credentials: tuple[str, ...] = ()
    if config is not None:
        required_credentials = _credential_requirements(web, llm, blockers)
        candidate_settings = config.get("candidate_settings")
        max_candidates = (
            candidate_settings.get("max_candidates")
            if isinstance(candidate_settings, Mapping)
            else None
        )
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or not 1 <= max_candidates <= 1000
        ):
            blockers.append(
                PreflightFinding(
                    "CANDIDATE_BUDGET_INVALID",
                    "candidate_settings.max_candidates must be an integer from 1 through 1000",
                )
            )
        else:
            web_calls = 3 if str(web.get("backend", "") or "").lower() == "search_api" else 0
            transport_policy = web.get("transport_retry_policy")
            transport_retries = (
                transport_policy.get("retries")
                if isinstance(transport_policy, Mapping)
                else None
            )
            max_transport_attempts = (
                transport_retries + 1
                if isinstance(transport_retries, int)
                and not isinstance(transport_retries, bool)
                and transport_retries >= 0
                else 0
            )
            physical_budget = web.get("physical_attempt_budget")
            physical_cap = (
                physical_budget.get("cap")
                if isinstance(physical_budget, Mapping)
                and isinstance(physical_budget.get("cap"), int)
                and not isinstance(physical_budget.get("cap"), bool)
                and physical_budget.get("cap") >= 0
                else 0
            )
            logical_web_requests = max_candidates * web_calls
            runtime_settings = config.get("runtime_settings")
            max_llm_http_attempts = (
                runtime_settings.get("max_llm_http_attempts")
                if isinstance(runtime_settings, Mapping)
                and isinstance(runtime_settings.get("max_llm_http_attempts"), int)
                and not isinstance(runtime_settings.get("max_llm_http_attempts"), bool)
                and runtime_settings.get("max_llm_http_attempts") >= 0
                else 0
            )
            request_budget = {
                "logical_web_requests": logical_web_requests,
                "max_transport_attempts_per_logical": max_transport_attempts,
                "theoretical_web_physical_attempts": (
                    logical_web_requests * max_transport_attempts
                ),
                "global_web_physical_attempt_cap": physical_cap,
                "web_retry_reserve": max(0, physical_cap - logical_web_requests),
                "max_llm_requests": max_llm_http_attempts,
            }

    try:
        file_credentials = parse_allowlisted_env_file(env_file)
    except FormalLexiconPreflightError as exc:
        blockers.append(PreflightFinding("ENV_FILE_INVALID", str(exc)))
        file_credentials = {}
    process_env = os.environ if environ is None else environ
    credential_values: dict[str, str] = {}
    credential_sources: dict[str, str] = {}
    for name in required_credentials:
        process_value = process_env.get(name)
        file_value = file_credentials.get(name)
        process_present = isinstance(process_value, str) and bool(process_value.strip())
        file_present = isinstance(file_value, str) and bool(file_value.strip())
        if process_present and file_present and process_value != file_value:
            credential_sources[name] = "conflict"
            blockers.append(
                PreflightFinding(
                    "CREDENTIAL_SOURCE_CONFLICT",
                    f"process environment and env file disagree for: {name}",
                )
            )
        elif process_present:
            credential_values[name] = process_value
            credential_sources[name] = "process-environment"
        elif file_present:
            credential_values[name] = file_value
            credential_sources[name] = "env-file"
        else:
            credential_sources[name] = "missing"
            blockers.append(
                PreflightFinding("CREDENTIAL_MISSING", f"required credential is absent: {name}")
            )

    frozen = None
    data_ref_valid = False
    if data_ref is None:
        blockers.append(PreflightFinding("DATA_REF_MISSING", "formal WP3 requires a frozen data_ref"))
    else:
        try:
            resolve_train_input(data_ref=data_ref, formal=False)
            data_ref_valid = True
        except (OSError, ValueError, TrainOnlyLexiconError) as exc:
            blockers.append(
                PreflightFinding(
                    "DATA_REF_INVALID",
                    f"frozen data_ref validation failed: {exc}",
                )
            )
    if train_partition_ref is None:
        blockers.append(
            PreflightFinding(
                "TRAIN_PARTITION_REF_MISSING",
                "formal WP3 requires a frozen train_partition_ref",
            )
        )
    if data_ref_valid and train_partition_ref is not None:
        try:
            frozen = resolve_train_input(
                data_ref=data_ref,
                train_partition_ref=train_partition_ref,
                formal=True,
                workspace_root=repository,
            )
            data_build_id = frozen.data_build_id
            train_record_count = len(frozen.records)
            train_data_sha256 = frozen.train_data_sha256
            train_ids_sha256 = frozen.train_ids_sha256
            if frozen.train_partition_dependency is None:
                raise TrainOnlyLexiconError(
                    "resolved formal input lacks partition dependency"
                )
            train_partition_id = str(
                frozen.train_partition_dependency["artifact_id"]
            )
            train_partition_payload_sha256 = str(
                frozen.train_partition_dependency["payload_manifest_sha256"]
            )
        except (OSError, ValueError, TrainOnlyLexiconError) as exc:
            blockers.append(
                PreflightFinding(
                    "TRAIN_PARTITION_REF_INVALID",
                    f"frozen data/partition validation failed: {exc}",
                )
            )

    build_authorization: object | None = None
    if (
        not blockers
        and config is not None
        and frozen is not None
        and data_ref is not None
        and train_partition_ref is not None
    ):
        try:
            build_authorization = _mint_formal_build_authorization(
                dataset=normalized_dataset,
                config=config,
                data_ref_path=data_ref,
                train_partition_ref_path=train_partition_ref,
                workspace_root=repository,
                frozen_train_input=frozen,
                credential_values=credential_values,
            )
        except (OSError, TypeError, ValueError, TrainOnlyLexiconError) as exc:
            blockers.append(
                PreflightFinding(
                    "BUILD_AUTHORIZATION_FAILED",
                    f"could not bind formal config/code/data authorization: {exc}",
                )
            )

    # Stable ordering makes reports and tests reproducible without turning the
    # report into an artifact or writing it to disk.
    unique: dict[tuple[str, str], PreflightFinding] = {}
    for finding in blockers:
        unique[(finding.code, finding.message)] = finding
    ordered = tuple(unique[key] for key in sorted(unique))
    return FormalLexiconPreflightResult(
        status="PASS" if not ordered else "BLOCKED",
        dataset=normalized_dataset,
        config_sha256=config_sha256,
        data_build_id=data_build_id,
        train_partition_id=train_partition_id,
        train_partition_payload_sha256=train_partition_payload_sha256,
        train_record_count=train_record_count,
        train_data_sha256=train_data_sha256,
        train_ids_sha256=train_ids_sha256,
        web_backend=str(web.get("backend", "") or "").lower() or None,
        llm_backend=str(llm.get("backend", "") or "").lower() or None,
        cache_policy=cache_policy,
        request_budget=request_budget,
        credential_sources=credential_sources,
        blockers=ordered,
        _credential_values=credential_values,
        _raw_config=config,
        _build_authorization=build_authorization,
    )


__all__ = [
    "ALLOWED_CREDENTIAL_ENV_NAMES",
    "FORMAL_CONFIG_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "FormalLexiconPreflightError",
    "FormalLexiconPreflightResult",
    "PreflightFinding",
    "parse_allowlisted_env_file",
    "preflight_formal_lexicon",
]
