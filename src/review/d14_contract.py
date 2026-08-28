"""Canonical Stage-1 D14 reviewer identity and policy contract.

This module contains no credentials and performs no network access.  It is the
single code-level authority for the two reviewer identities frozen by D14.
Callers may use a synthetic transport only for an explicitly non-scientific CF
proposal; such a run is permanently marked as engineering synthetic and is not
allowed to satisfy this live reviewer-resolution contract.
"""

from __future__ import annotations

import copy
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit


D14_POLICY_SCHEMA = "stage1-blind-review-policy/v1"
D14_LIVE_EXECUTION_MODE = "live-d14"
D14_SYNTHETIC_EXECUTION_MODE = "engineering-synthetic"
D14_EXECUTION_MODES = (
    D14_LIVE_EXECUTION_MODE,
    D14_SYNTHETIC_EXECUTION_MODE,
)

D14_REVIEWER_CONFIG = (
    {
        "reviewer_id": "glm_high_parameter",
        "backend": "openai_compatible",
        "api_base_env": "STAGE1_GLM_API_BASE",
        "default_api_base": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "GLM_API_KEY",
        "model_env": "STAGE1_GLM_MODEL",
        "default_model": "glm-5.3",
    },
    {
        "reviewer_id": "deepseek_flash",
        "backend": "openai_compatible",
        "api_base_env": "STAGE1_DEEPSEEK_API_BASE",
        "default_api_base": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "STAGE1_DEEPSEEK_MODEL",
        "default_model": "deepseek-v4-flash",
    },
)
D14_REVIEWER_IDS = tuple(
    sorted(str(row["reviewer_id"]) for row in D14_REVIEWER_CONFIG)
)
D14_REVIEWER_RESOLUTION = tuple(
    {
        "reviewer_id": str(row["reviewer_id"]),
        "api_base": str(row["default_api_base"]),
        "model": str(row["default_model"]),
    }
    for row in D14_REVIEWER_CONFIG
)
D14_REQUEST_OPTIONS = {
    "glm_high_parameter": {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    },
    "deepseek_flash": {
        "thinking": {"type": "disabled"},
        "reasoning_effort": "medium",
    },
}
D14_BLINDNESS = {
    "reviewers_see_each_other": False,
    "show_model_predictions": False,
    "show_experimental_condition": False,
    "show_source_split": False,
    "stable_shuffle_seed": 42,
}
D14_CONSENSUS = {
    "auto_accept_only_exact_structured_agreement": True,
    "low_confidence_threshold": 0.8,
    "agreement_qc_fraction": 0.1,
    "agreement_qc_assignment": "sha256-review-item-id-v1",
    "human_queue_reasons": [
        "model_disagreement",
        "parse_failure",
        "low_confidence",
        "agreement_qc_sample",
    ],
}
D14_PROVENANCE = {
    "retain_raw_request_without_secrets": True,
    "retain_raw_response": True,
    "record_prompt_sha256": True,
    "record_model_identifier": True,
    "record_provider_request_id_when_available": True,
}

_D14_POLICY_DOCUMENT = {
    "schema_version": D14_POLICY_SCHEMA,
    "temperature": 0,
    "top_p": 1,
    "max_attempts": 1,
    "request_timeout_seconds": 120,
    "max_tokens": 4096,
    "max_concurrency_per_reviewer": 1,
    "reviewer_request_options": D14_REQUEST_OPTIONS,
    "reviewers": list(D14_REVIEWER_CONFIG),
    "blindness": D14_BLINDNESS,
    "consensus": D14_CONSENSUS,
    "provenance": D14_PROVENANCE,
}


def d14_policy_document() -> dict[str, Any]:
    """Return a mutable copy of the exact credential-free D14 policy."""

    return copy.deepcopy(_D14_POLICY_DOCUMENT)


def validate_d14_policy_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete policy, including order-sensitive reviewer fields."""

    policy = dict(document)
    if policy != _D14_POLICY_DOCUMENT:
        raise ValueError("blind-review policy differs from the canonical D14 contract")
    return copy.deepcopy(policy)


def normalize_model_identifier(value: Any) -> str:
    """Normalize a provider model identifier for an exact identity comparison."""

    if not isinstance(value, str):
        raise ValueError("model identifier must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    if not normalized or any(character.isspace() for character in normalized):
        raise ValueError("model identifier must be a non-empty token")
    return normalized


def normalize_provider_api_base(value: Any) -> str:
    """Normalize a credential-free HTTP(S) provider base for exact comparison."""

    if not isinstance(value, str):
        raise ValueError("provider API base must be a non-empty string")
    rendered = unicodedata.normalize("NFKC", value).strip().rstrip("/")
    parsed = urlsplit(rendered)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("provider API base must be a credential-free HTTP(S) origin/path")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("provider API base contains an invalid port") from exc
    hostname = parsed.hostname.casefold()
    netloc = hostname if port is None else f"{hostname}:{port}"
    path = parsed.path.rstrip("/")
    if any(part in {".", ".."} for part in path.split("/")):
        raise ValueError("provider API base path cannot contain dot segments")
    return urlunsplit((parsed.scheme.casefold(), netloc, path, "", ""))


def validate_reviewer_resolution(
    reviewers: Sequence[Mapping[str, Any]],
    *,
    require_live_d14: bool,
) -> list[dict[str, str]]:
    """Validate two independent resolutions and, for live runs, exact D14 IDs."""

    if len(reviewers) != 2:
        raise ValueError("reviewer resolution must contain exactly two reviewers")
    normalized: list[dict[str, str]] = []
    for raw in reviewers:
        if set(raw) != {"reviewer_id", "api_base", "model"}:
            raise ValueError("reviewer resolution row is non-canonical")
        reviewer_id = raw.get("reviewer_id")
        api_base = raw.get("api_base")
        model = raw.get("model")
        if not all(isinstance(item, str) and item for item in (reviewer_id, api_base, model)):
            raise ValueError("reviewer resolution fields must be non-empty strings")
        normalized.append(
            {
                "reviewer_id": str(reviewer_id),
                "api_base": normalize_provider_api_base(api_base),
                "model": normalize_model_identifier(model),
            }
        )
    if tuple(sorted(row["reviewer_id"] for row in normalized)) != D14_REVIEWER_IDS:
        raise ValueError("reviewer resolution is not the frozen GLM/DeepSeek pair")
    if len({(row["api_base"], row["model"]) for row in normalized}) != 2:
        raise ValueError("reviewer resolutions are not independent")
    if require_live_d14:
        expected = {
            row["reviewer_id"]: {
                "api_base": normalize_provider_api_base(row["api_base"]),
                "model": normalize_model_identifier(row["model"]),
            }
            for row in D14_REVIEWER_RESOLUTION
        }
        for row in normalized:
            if {
                "api_base": row["api_base"],
                "model": row["model"],
            } != expected[row["reviewer_id"]]:
                raise ValueError(
                    "live reviewer provider/model differs from the canonical D14 resolution"
                )
    return normalized


def validate_requested_returned_model(requested: Any, returned: Any) -> str:
    """Require non-empty requested/returned IDs with exact normalized equality."""

    requested_normalized = normalize_model_identifier(requested)
    returned_normalized = normalize_model_identifier(returned)
    if requested_normalized != returned_normalized:
        raise ValueError("returned model does not match the requested model")
    return requested_normalized


__all__ = [
    "D14_BLINDNESS",
    "D14_CONSENSUS",
    "D14_EXECUTION_MODES",
    "D14_LIVE_EXECUTION_MODE",
    "D14_POLICY_SCHEMA",
    "D14_PROVENANCE",
    "D14_REQUEST_OPTIONS",
    "D14_REVIEWER_CONFIG",
    "D14_REVIEWER_IDS",
    "D14_REVIEWER_RESOLUTION",
    "D14_SYNTHETIC_EXECUTION_MODE",
    "d14_policy_document",
    "normalize_model_identifier",
    "normalize_provider_api_base",
    "validate_d14_policy_document",
    "validate_requested_returned_model",
    "validate_reviewer_resolution",
]
