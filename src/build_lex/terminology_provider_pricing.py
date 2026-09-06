"""Offline, evidence-replaying pricing verification for the WP3 successor run.

The successor runner reserves budget before each provider request.  A URL and a
claimed digest are not sufficient evidence for the rates used by that budget
gate: this module requires local UTF-8 snapshots of both providers' official
pricing pages, replays exact model and rate evidence against those bytes, and
derives the small pricing projection consumed by the run plan.

Importing and calling this module never performs network access.  Snapshot
capture is deliberately outside this lifecycle so a failed or drifting source
cannot silently become an executable price.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from data.training_artifacts import (
    TrainingArtifactError,
    canonical_sha256,
    validate_json_schema,
    write_canonical_json,
)


EVIDENCE_SCHEMA_VERSION = "wp3-provider-pricing-evidence/v1"
EVIDENCE_SCHEMA_PATH = "schemas/wp3_provider_pricing_evidence_v1.schema.json"
PROJECTION_SCHEMA_VERSION = "wp3-provider-pricing-snapshot/v1"
VERIFICATION_SCHEMA_VERSION = "wp3-provider-pricing-verification/v1"
BUNDLE_SCHEMA_VERSION = "wp3-provider-pricing-verification-bundle/v1"
BUNDLE_SCHEMA_PATH = (
    "schemas/wp3_provider_pricing_verification_bundle_v1.schema.json"
)
RATE_POLICY_ID = "wp3-conservative-peak-cache-miss/v1"
APPLICABILITY_SCOPE = "requested-model-direct-token-tariff-only/v1"
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024
PROVIDER_IDS = ("glm_flash", "deepseek_flash")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
CAPTURED_AT_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]{0,8})(?:\.[0-9]{1,9})?$")

PROVIDER_POLICY = {
    "glm_flash": {
        "requested_model": "glm-5.3-flash",
        "official_hosts": frozenset({"bigmodel.cn"}),
        "official_source_urls": frozenset({"https://bigmodel.cn/pricing"}),
        "input_basis": "maximum-published-applicable-nondiscounted-input/v1",
        "output_basis": "maximum-published-applicable-nondiscounted-output/v1",
    },
    "deepseek_flash": {
        "requested_model": "deepseek-v4-flash",
        "official_hosts": frozenset({"api-docs.deepseek.com"}),
        "input_basis": "peak-cache-miss-input/v1",
        "output_basis": "peak-output/v1",
    },
}

DEVELOPMENT_MARKING = {
    "scope": "development-only",
    "scientific_eligible": False,
    "sealed": False,
}


class ProviderPricingError(RuntimeError):
    """Raised when local pricing evidence cannot be independently replayed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(inner) for inner in value]
    return copy.deepcopy(value)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderPricingError(f"{label} must be an object")
    return dict(value)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProviderPricingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ProviderPricingError(f"non-finite JSON number: {value}")


def _load_strict_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except ProviderPricingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderPricingError(f"cannot read pricing JSON {source}: {exc}") from exc
    return _object(value, "pricing JSON")


def _load_mapping(value_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value_or_path, Mapping):
        return copy.deepcopy(dict(value_or_path))
    return _load_strict_json(value_or_path)


def _validate_with_schema(
    value: Mapping[str, Any], *, workspace_root: Path, schema_path: str
) -> None:
    try:
        validate_json_schema(value, workspace_root / schema_path)
    except TrainingArtifactError as exc:
        raise ProviderPricingError(str(exc)) from exc


def _validate_captured_at(value: Any, label: str) -> str:
    if not isinstance(value, str) or CAPTURED_AT_RE.fullmatch(value) is None:
        raise ProviderPricingError(f"{label} must be a whole-second UTC timestamp")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProviderPricingError(f"{label} is not a valid timestamp") from exc
    return value


def _decimal_text(value: Any, label: str) -> tuple[str, Decimal]:
    if not isinstance(value, str) or DECIMAL_RE.fullmatch(value) is None:
        raise ProviderPricingError(f"{label} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ProviderPricingError(f"{label} is not a decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ProviderPricingError(f"{label} must be positive")
    normalized = format(parsed.normalize(), "f")
    if normalized.startswith("."):
        normalized = "0" + normalized
    if value != normalized:
        raise ProviderPricingError(f"{label} is not in canonical form")
    return value, parsed


def _safe_relative_snapshot_path(value: Any) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
    ):
        raise ProviderPricingError("snapshot_path must be a safe workspace-relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ProviderPricingError("snapshot_path must be a safe workspace-relative path")
    return relative


def _read_snapshot(
    workspace_root: Path,
    relative: PurePosixPath,
    *,
    expected_sha256: str,
    expected_size: int,
) -> str:
    current = workspace_root
    final_metadata: os.stat_result | None = None
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ProviderPricingError(f"pricing snapshot is unavailable: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ProviderPricingError("pricing snapshot path cannot contain symlinks")
        if index < len(relative.parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ProviderPricingError("pricing snapshot parent is not a directory")
        else:
            final_metadata = metadata
    if final_metadata is None or not stat.S_ISREG(final_metadata.st_mode):
        raise ProviderPricingError("pricing snapshot must be a regular file")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or expected_size > MAX_SNAPSHOT_BYTES
        or final_metadata.st_size != expected_size
    ):
        raise ProviderPricingError("pricing snapshot size does not match the receipt")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(current, flags)
    except OSError as exc:
        raise ProviderPricingError("pricing snapshot cannot be safely opened") from exc
    try:
        opened_metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or opened_metadata.st_dev != final_metadata.st_dev
            or opened_metadata.st_ino != final_metadata.st_ino
            or opened_metadata.st_size != expected_size
        ):
            raise ProviderPricingError("pricing snapshot changed while it was opened")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            payload = handle.read(MAX_SNAPSHOT_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) != expected_size or len(payload) > MAX_SNAPSHOT_BYTES:
        raise ProviderPricingError("pricing snapshot size changed while it was read")
    digest = hashlib.sha256(payload).hexdigest()
    if HASH_RE.fullmatch(expected_sha256) is None or digest != expected_sha256:
        raise ProviderPricingError("pricing snapshot SHA-256 does not match the receipt")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProviderPricingError("pricing snapshot is not strict UTF-8") from exc
    if "\x00" in text:
        raise ProviderPricingError("pricing snapshot contains a NUL character")
    return text


def _official_source_url(provider_id: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ProviderPricingError("pricing source URL is missing")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ProviderPricingError("pricing source URL is malformed") from exc
    policy = PROVIDER_POLICY[provider_id]
    if (
        parsed.scheme != "https"
        or parsed.hostname not in policy["official_hosts"]
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderPricingError(
            f"{provider_id} pricing source is not an allowlisted official HTTPS page"
        )
    exact_urls = policy.get("official_source_urls")
    if exact_urls is not None and value not in exact_urls:
        raise ProviderPricingError(
            f"{provider_id} pricing source is not an allowlisted official HTTPS page"
        )
    return value


def _replay_exact_text(
    snapshot: str,
    evidence: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, int, int, int]:
    row = _object(evidence, label)
    if set(row) not in (
        {"exact_text", "occurrence_ordinal"},
        {
            "exact_text",
            "occurrence_ordinal",
            "numeric_text",
            "semantic_label",
        },
    ):
        raise ProviderPricingError(f"{label} evidence fields differ")
    exact_text = row.get("exact_text")
    ordinal = row.get("occurrence_ordinal")
    if (
        not isinstance(exact_text, str)
        or not exact_text
        or len(exact_text) > 4096
        or "\x00" in exact_text
        or isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 1
    ):
        raise ProviderPricingError(f"{label} exact evidence is invalid")
    starts: list[int] = []
    cursor = 0
    while True:
        start = snapshot.find(exact_text, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + 1
    if ordinal > len(starts):
        raise ProviderPricingError(f"{label} exact evidence is absent from the snapshot")
    start = starts[ordinal - 1]
    return exact_text, ordinal, start, start + len(exact_text)


def _model_replay(
    snapshot: str, evidence: Mapping[str, Any], *, requested_model: str
) -> dict[str, Any]:
    exact_text, ordinal, start, end = _replay_exact_text(
        snapshot, evidence, label="model"
    )
    if requested_model.casefold() not in exact_text.casefold():
        raise ProviderPricingError("model evidence does not identify the requested model")
    return {
        "exact_text_sha256": hashlib.sha256(exact_text.encode("utf-8")).hexdigest(),
        "occurrence_ordinal": ordinal,
        "start_character": start,
        "end_character": end,
    }


def _rate_replay(
    snapshot: str,
    evidence: Mapping[str, Any],
    *,
    semantic_label: str,
    selected_rate: str,
) -> dict[str, Any]:
    row = _object(evidence, semantic_label)
    exact_text, ordinal, start, end = _replay_exact_text(
        snapshot, row, label=semantic_label
    )
    if row.get("semantic_label") != semantic_label:
        raise ProviderPricingError(f"{semantic_label} evidence label differs")
    numeric_text, numeric_value = _decimal_text(
        row.get("numeric_text"), f"{semantic_label} evidence numeric_text"
    )
    selected_text, selected_value = _decimal_text(
        selected_rate, f"selected {semantic_label} rate"
    )
    if numeric_value != selected_value:
        raise ProviderPricingError(
            f"selected {semantic_label} rate does not match its exact evidence"
        )
    numeric_matches = list(
        re.finditer(rf"(?<![0-9.]){re.escape(numeric_text)}(?![0-9.])", exact_text)
    )
    if len(numeric_matches) != 1:
        raise ProviderPricingError(
            f"{semantic_label} numeric evidence must occur exactly once in its quote"
        )
    return {
        "exact_text_sha256": hashlib.sha256(exact_text.encode("utf-8")).hexdigest(),
        "occurrence_ordinal": ordinal,
        "start_character": start,
        "end_character": end,
        "numeric_text": numeric_text,
        "normalized_rmb_per_million": selected_text,
    }


def _validate_evidence_shape(
    evidence: Mapping[str, Any], *, workspace_root: Path
) -> dict[str, Any]:
    value = copy.deepcopy(dict(evidence))
    _validate_with_schema(
        value, workspace_root=workspace_root, schema_path=EVIDENCE_SCHEMA_PATH
    )
    required = {
        "schema_version",
        "scope",
        "scientific_eligible",
        "sealed",
        "currency",
        "captured_at",
        "rate_policy",
        "source_receipts",
        "provider_rates",
    }
    if set(value) != required:
        raise ProviderPricingError("pricing evidence fields differ")
    if (
        value.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or value.get("scope") != DEVELOPMENT_MARKING["scope"]
        or value.get("scientific_eligible") is not False
        or value.get("sealed") is not False
        or value.get("currency") != "RMB"
        or value.get("rate_policy") != RATE_POLICY_ID
    ):
        raise ProviderPricingError("pricing evidence scope or policy differs")
    _validate_captured_at(value.get("captured_at"), "captured_at")
    receipts = value.get("source_receipts")
    if not isinstance(receipts, list) or len(receipts) != len(PROVIDER_IDS):
        raise ProviderPricingError("pricing evidence requires exactly two receipts")
    provider_rates = _object(value.get("provider_rates"), "provider_rates")
    if set(provider_rates) != set(PROVIDER_IDS):
        raise ProviderPricingError("provider_rates must cover exactly both providers")
    return value


def verify_provider_pricing_evidence(
    value_or_path: Mapping[str, Any] | str | Path,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Replay snapshots and return a canonical projection plus verification receipt.

    The returned ``pricing_projection`` has the exact five-field shape consumed
    by the successor run plan.  Callers must not accept a projection supplied
    independently of this function (or ``validate_provider_pricing_bundle``).
    """

    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ProviderPricingError("workspace root is not a directory")
    evidence = _validate_evidence_shape(
        _load_mapping(value_or_path), workspace_root=root
    )
    captured_at = evidence["captured_at"]
    rates = _object(evidence["provider_rates"], "provider_rates")
    receipts_by_provider: dict[str, dict[str, Any]] = {}
    for raw_receipt in evidence["source_receipts"]:
        receipt = _object(raw_receipt, "source receipt")
        provider_id = receipt.get("provider_id")
        if provider_id not in PROVIDER_POLICY or provider_id in receipts_by_provider:
            raise ProviderPricingError("source receipts must cover each provider once")
        receipts_by_provider[provider_id] = receipt
    if set(receipts_by_provider) != set(PROVIDER_IDS):
        raise ProviderPricingError("source receipts must cover each provider once")

    projection_receipts: list[dict[str, Any]] = []
    projection_rates: dict[str, dict[str, str]] = {}
    source_replays: list[dict[str, Any]] = []
    for provider_id in PROVIDER_IDS:
        policy = PROVIDER_POLICY[provider_id]
        receipt = receipts_by_provider[provider_id]
        requested_model = receipt.get("requested_model")
        if requested_model != policy["requested_model"]:
            raise ProviderPricingError(f"{provider_id} requested model differs")
        if receipt.get("captured_at") != captured_at:
            raise ProviderPricingError("source receipt capture time differs")
        source_url = _official_source_url(provider_id, receipt.get("source_url"))
        relative = _safe_relative_snapshot_path(receipt.get("snapshot_path"))
        snapshot_sha256 = receipt.get("snapshot_sha256")
        if not isinstance(snapshot_sha256, str) or HASH_RE.fullmatch(snapshot_sha256) is None:
            raise ProviderPricingError("source receipt snapshot SHA-256 is invalid")
        snapshot = _read_snapshot(
            root,
            relative,
            expected_sha256=snapshot_sha256,
            expected_size=receipt.get("snapshot_size_bytes"),
        )
        basis = _object(receipt.get("rate_policy_application"), "rate policy application")
        if basis != {
            "applicability_scope": APPLICABILITY_SCOPE,
            "input_basis": policy["input_basis"],
            "output_basis": policy["output_basis"],
        }:
            raise ProviderPricingError(f"{provider_id} conservative rate basis differs")
        selected = _object(rates[provider_id], f"{provider_id} provider rate")
        if set(selected) != {
            "input_rmb_per_million",
            "output_rmb_per_million",
        }:
            raise ProviderPricingError(f"{provider_id} provider rate fields differ")
        input_text, _ = _decimal_text(
            selected.get("input_rmb_per_million"),
            f"{provider_id} input_rmb_per_million",
        )
        output_text, _ = _decimal_text(
            selected.get("output_rmb_per_million"),
            f"{provider_id} output_rmb_per_million",
        )
        evidence_rows = _object(receipt.get("evidence"), "source receipt evidence")
        if set(evidence_rows) != {"model", "input_rate", "output_rate"}:
            raise ProviderPricingError("source receipt evidence fields differ")
        model_replay = _model_replay(
            snapshot, evidence_rows["model"], requested_model=requested_model
        )
        input_replay = _rate_replay(
            snapshot,
            evidence_rows["input_rate"],
            semantic_label="input_rmb_per_million",
            selected_rate=input_text,
        )
        output_replay = _rate_replay(
            snapshot,
            evidence_rows["output_rate"],
            semantic_label="output_rmb_per_million",
            selected_rate=output_text,
        )
        projection_receipts.append(
            {
                "provider_id": provider_id,
                "source_url": source_url,
                "snapshot_sha256": snapshot_sha256,
            }
        )
        projection_rates[provider_id] = {
            "input_rmb_per_million": input_text,
            "output_rmb_per_million": output_text,
        }
        source_replays.append(
            {
                "provider_id": provider_id,
                "requested_model": requested_model,
                "source_url": source_url,
                "snapshot_path": relative.as_posix(),
                "snapshot_sha256": snapshot_sha256,
                "snapshot_size_bytes": receipt["snapshot_size_bytes"],
                "captured_at": captured_at,
                "rate_policy_application": copy.deepcopy(basis),
                "evidence_replay": {
                    "model": model_replay,
                    "input_rate": input_replay,
                    "output_rate": output_replay,
                },
            }
        )

    projection = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "currency": "RMB",
        "captured_at": captured_at,
        "source_receipts": projection_receipts,
        "provider_rates": projection_rates,
    }
    receipt_identity = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        **DEVELOPMENT_MARKING,
        "rate_policy": RATE_POLICY_ID,
        "pricing_evidence_sha256": canonical_sha256(evidence),
        "pricing_projection_sha256": canonical_sha256(projection),
        "source_replays": source_replays,
        "network_access_performed": False,
        "verified": True,
    }
    receipt = {
        **receipt_identity,
        "verification_id": "wp3pricingverify-" + canonical_sha256(receipt_identity),
    }
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        **DEVELOPMENT_MARKING,
        "pricing_projection": projection,
        "verification_receipt": receipt,
        "verification_receipt_sha256": canonical_sha256(receipt),
    }
    _validate_with_schema(
        bundle, workspace_root=root, schema_path=BUNDLE_SCHEMA_PATH
    )
    return bundle


def validate_provider_pricing_bundle(
    bundle_or_path: Mapping[str, Any] | str | Path,
    *,
    evidence: Mapping[str, Any] | str | Path,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Rebuild a verified bundle from local evidence and require byte identity."""

    root = Path(workspace_root).resolve()
    supplied = _load_mapping(bundle_or_path)
    _validate_with_schema(
        supplied, workspace_root=root, schema_path=BUNDLE_SCHEMA_PATH
    )
    expected = verify_provider_pricing_evidence(evidence, workspace_root=root)
    if supplied != expected:
        raise ProviderPricingError(
            "pricing verification bundle differs from replayed source evidence"
        )
    return expected


def pricing_projection_for_successor_plan(
    *,
    evidence: Mapping[str, Any] | str | Path,
    workspace_root: str | Path,
    verified_bundle: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Return the only pricing projection safe to pass to a successor plan."""

    if verified_bundle is None:
        bundle = verify_provider_pricing_evidence(
            evidence, workspace_root=workspace_root
        )
    else:
        bundle = validate_provider_pricing_bundle(
            verified_bundle,
            evidence=evidence,
            workspace_root=workspace_root,
        )
    return copy.deepcopy(bundle["pricing_projection"])


def write_verified_pricing_bundle(
    path: str | Path, bundle: Mapping[str, Any]
) -> None:
    """Write a bundle already produced by the replaying verifier."""

    value = _object(bundle, "pricing verification bundle")
    if value.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ProviderPricingError("pricing verification bundle schema differs")
    write_canonical_json(path, _plain(value))


__all__ = [
    "APPLICABILITY_SCOPE",
    "BUNDLE_SCHEMA_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "MAX_SNAPSHOT_BYTES",
    "PROVIDER_POLICY",
    "ProviderPricingError",
    "RATE_POLICY_ID",
    "pricing_projection_for_successor_plan",
    "validate_provider_pricing_bundle",
    "verify_provider_pricing_evidence",
    "write_verified_pricing_bundle",
]
