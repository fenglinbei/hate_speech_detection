"""Immutable, content-clustered Stage 1 fit/calibration partition.

The scientific development split is never used for checkpoint selection.  A
fixed subset of the finalized training split is instead held out for that
purpose.  Partitioning by query ID alone would still allow a duplicate text to
appear as fit evidence for a calibration query, so this lifecycle first groups
the complete 5,781-record train frame by normalized-content SHA-256 and assigns
each whole cluster from its smallest canonical numeric query ID.

Only locator refs contain absolute paths.  The target embeds a portable data
dependency, a complete row for every train query, the resolved policy, and all
inputs needed to reproduce its content-addressed lifecycle ID.
"""

from __future__ import annotations

import copy
import hashlib
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.stage1_data import Stage1DataError, validate_data_target
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    canonical_sha256,
    ensure_exact_file_set,
    finalize_target_atomic,
    load_json,
    load_jsonl,
    new_staging_directory,
    portable_dependency,
    resolve_dependency_target,
    resolve_locator_ref,
    sha256_file,
    validate_dependency_ref,
    validate_json_schema,
    validate_payload_manifest,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "config/stage1/train_calibration_partition.json"
DEFAULT_SCHEMA = REPOSITORY_ROOT / "schemas/stage1_train_partition_v1.schema.json"

ARTIFACT_KIND = "train-partition"
SCHEMA_VERSION = "stage1-train-partition/v1"
POLICY_SCHEMA_VERSION = "stage1-train-partition-policy/v1"
CONFIG_SCHEMA_VERSION = "stage1-train-calibration-partition-config/v1"
VALIDATION_SCHEMA_VERSION = "stage1-train-partition-validation-report/v1"
EXPECTED_TRAIN_COUNT = 5781
ID_PREFIX = "tpart-"
CANONICAL_NUMERIC_ID_RE = re.compile(r"^[1-9][0-9]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FROZEN_POLICY: dict[str, Any] = {
    "schema_version": POLICY_SCHEMA_VERSION,
    "input_split": "train",
    "content_normalization": "crlf-to-lf/v1",
    "content_hash": "sha256-utf8-normalized-content/v1",
    "cluster_policy": "exact-normalized-content-sha256/v1",
    "cluster_representative": "minimum-canonical-numeric-query-id/v1",
    "assignment": {
        "assignment": "sha256-query-id-v1",
        "salt": "stage1-train-calibration-v1",
        "fraction": 0.1,
        "hash_modulus": 10000,
        "hash_threshold_exclusive": 1000,
        "id_fields": ["id", "query_id"],
        "nominal_fraction": 0.1,
        "fraction_semantics": "nominal-hash-threshold-not-exact-count/v1",
        "id_input": "cluster_representative_query_id",
    },
    "partitions": {
        "fit": "bucket-greater-than-or-equal-to-threshold",
        "calibration": "bucket-strictly-less-than-threshold",
    },
    "information_isolation": "full-information-isolated",
    "fit_evidence_scope": "fit-only-demo-and-lexicon",
    "calibration_presentation": "fixed-calibration-presentation",
}


class TrainPartitionError(TrainingArtifactError):
    """Raised when the immutable train partition contract is violated."""


@dataclass(frozen=True)
class TrainPartitionBundle:
    """Deep-validated partition plus its materialized finalized data records."""

    locator: dict[str, Any]
    target: Path
    partition_dependency: dict[str, Any]
    data_dependency: dict[str, Any]
    data_target: Path
    meta: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    train_records: tuple[dict[str, Any], ...]
    fit_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    fit_records: tuple[dict[str, Any], ...]
    calibration_records: tuple[dict[str, Any], ...]


def normalize_content(content: str) -> str:
    """Apply the sole frozen normalization: every CRLF pair becomes LF."""

    if not isinstance(content, str) or not content:
        raise TrainPartitionError("train content must be non-empty text")
    return content.replace("\r\n", "\n")


def normalized_content_sha256(content: str) -> str:
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


def _canonical_numeric_query_id(value: Any) -> str:
    if not isinstance(value, str) or CANONICAL_NUMERIC_ID_RE.fullmatch(value) is None:
        raise TrainPartitionError(
            f"train query ID must be a canonical positive decimal string: {value!r}"
        )
    return value


def calibration_bucket(query_id: str, assignment: Mapping[str, Any]) -> int:
    """Replay the repository's existing ``sha256-query-id-v1`` wire exactly."""

    query_id = _canonical_numeric_query_id(query_id)
    if assignment.get("assignment") != "sha256-query-id-v1":
        raise TrainPartitionError("unsupported calibration assignment policy")
    modulus = assignment.get("hash_modulus")
    salt = assignment.get("salt")
    if (
        isinstance(modulus, bool)
        or not isinstance(modulus, int)
        or modulus <= 1
        or not isinstance(salt, str)
        or not salt
    ):
        raise TrainPartitionError("invalid calibration hash assignment")
    identity = {
        "assignment": "sha256-query-id-v1",
        "query_id": query_id,
        "salt": salt,
    }
    return int(hashlib.sha256(canonical_json_bytes(identity)).hexdigest(), 16) % modulus


def _load_and_validate_config(path: str | Path) -> dict[str, Any]:
    config = load_json(path)
    required = {"schema_version", "artifact_root", "expected_train_count", "policy"}
    if not isinstance(config, dict) or set(config) != required:
        raise TrainPartitionError("train partition config fields are not canonical")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise TrainPartitionError("unsupported train partition config schema")
    artifact_root = config.get("artifact_root")
    if not isinstance(artifact_root, str) or not artifact_root:
        raise TrainPartitionError("train partition artifact_root must be non-empty text")
    artifact_path = Path(artifact_root)
    if (
        artifact_path.is_absolute()
        or ".." in artifact_path.parts
        or artifact_root != artifact_path.as_posix()
    ):
        raise TrainPartitionError("train partition artifact_root must be workspace-relative")
    count = config.get("expected_train_count")
    if isinstance(count, bool) or count != EXPECTED_TRAIN_COUNT:
        raise TrainPartitionError(
            f"formal train partition requires exactly {EXPECTED_TRAIN_COUNT} records"
        )
    if config.get("policy") != FROZEN_POLICY:
        raise TrainPartitionError("train partition policy differs from the frozen contract")
    return copy.deepcopy(config)


def _dependency_identity(dependency: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": dependency.get("schema_version"),
        "artifact_kind": dependency.get("artifact_kind"),
        "artifact_id": dependency.get("artifact_id"),
        "payload_manifest_sha256": dependency.get("payload_manifest_sha256"),
    }


def _require_expected_dependency(
    actual: Mapping[str, Any], expected: Mapping[str, Any] | None
) -> None:
    if expected is None:
        return
    compact_keys = {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "payload_manifest_sha256",
    }
    portable_keys = compact_keys | {"logical_repo_path"}
    if set(expected) == compact_keys:
        if _dependency_identity(actual) != dict(expected):
            raise TrainPartitionError(
                "train partition data dependency identity differs from expected"
            )
        return
    if set(expected) == portable_keys:
        try:
            frozen = validate_dependency_ref(expected, expected_kind="data")
        except TrainingArtifactError as exc:
            raise TrainPartitionError(f"invalid expected data dependency: {exc}") from exc
        if dict(actual) != frozen:
            raise TrainPartitionError(
                "train partition portable data dependency differs from expected"
            )
        return
    raise TrainPartitionError(
        "expected data dependency must be a compact four-field or portable five-field ref"
    )


def _deep_data_records(
    dependency: Mapping[str, Any], workspace_root: Path
) -> tuple[Path, list[dict[str, Any]]]:
    try:
        frozen = validate_dependency_ref(dependency, expected_kind="data")
        target = resolve_dependency_target(frozen, workspace_root)
        report = validate_data_target(target)
    except (TrainingArtifactError, Stage1DataError, OSError, ValueError) as exc:
        raise TrainPartitionError(f"formal data dependency failed deep validation: {exc}") from exc
    if (
        report.get("data_build_id") != frozen["artifact_id"]
        or report.get("payload_manifest_sha256")
        != frozen["payload_manifest_sha256"]
        or report.get("split_counts", {}).get("train") != EXPECTED_TRAIN_COUNT
    ):
        raise TrainPartitionError("formal data dependency identity or train frame is invalid")
    train = load_json(target / "train.json")
    if validate_payload_manifest(target) != frozen["payload_manifest_sha256"]:
        raise TrainPartitionError("formal data dependency changed while it was read")
    if not isinstance(train, list) or len(train) != EXPECTED_TRAIN_COUNT:
        raise TrainPartitionError(
            f"formal data train frame must contain exactly {EXPECTED_TRAIN_COUNT} records"
        )
    if any(not isinstance(record, dict) for record in train):
        raise TrainPartitionError("formal data train frame contains a non-object record")
    return target, [dict(record) for record in train]


def _partition_rows(
    records: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if len(records) != EXPECTED_TRAIN_COUNT:
        raise TrainPartitionError(
            f"partition input must cover exactly {EXPECTED_TRAIN_COUNT} train records"
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    normalized_by_hash: dict[str, str] = {}
    cluster_ids: dict[str, list[str]] = {}
    for record in records:
        query_id = _canonical_numeric_query_id(record.get("id"))
        if query_id in by_id:
            raise TrainPartitionError(f"duplicate train query ID: {query_id}")
        content = record.get("content")
        normalized = normalize_content(content)
        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        prior = normalized_by_hash.setdefault(content_hash, normalized)
        if prior != normalized:
            raise TrainPartitionError("SHA-256 collision in normalized train content")
        by_id[query_id] = record
        cluster_ids.setdefault(content_hash, []).append(query_id)

    assignment = policy.get("assignment")
    if not isinstance(assignment, Mapping):
        raise TrainPartitionError("partition policy lacks its assignment")
    threshold = assignment.get("hash_threshold_exclusive")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int)
        or not 0 < threshold < int(assignment.get("hash_modulus", 0))
    ):
        raise TrainPartitionError("partition policy has an invalid calibration threshold")

    cluster_assignment: dict[str, tuple[str, int, str]] = {}
    for content_hash, query_ids in cluster_ids.items():
        representative = min(query_ids, key=int)
        bucket = calibration_bucket(representative, assignment)
        partition = "calibration" if bucket < threshold else "fit"
        cluster_assignment[content_hash] = (representative, bucket, partition)

    rows: list[dict[str, Any]] = []
    for query_id, record in sorted(by_id.items(), key=lambda item: int(item[0])):
        content_hash = normalized_content_sha256(str(record["content"]))
        representative, bucket, partition = cluster_assignment[content_hash]
        rows.append(
            {
                "query_id": query_id,
                "content_sha256": content_hash,
                "cluster_representative_query_id": representative,
                "bucket": bucket,
                "partition": partition,
            }
        )
    partitions = {row["partition"] for row in rows}
    if partitions != {"fit", "calibration"}:
        raise TrainPartitionError("fit and calibration partitions must both be non-empty")
    return rows


def _row_partitions(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    fit = [str(row["query_id"]) for row in rows if row.get("partition") == "fit"]
    calibration = [
        str(row["query_id"])
        for row in rows
        if row.get("partition") == "calibration"
    ]
    return fit, calibration


def _meta_document(
    *,
    data_dependency: Mapping[str, Any],
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    rows_sha256: str,
    builder_code_sha256: str,
) -> dict[str, Any]:
    fit_ids, calibration_ids = _row_partitions(rows)
    ordered_ids = [str(row["query_id"]) for row in rows]
    config_sha256 = canonical_sha256(config)
    policy = copy.deepcopy(dict(config["policy"]))
    id_inputs = {
        "schema_version": SCHEMA_VERSION,
        "data_dependency": copy.deepcopy(dict(data_dependency)),
        "policy": policy,
        "config_sha256": config_sha256,
        "builder_code_sha256": builder_code_sha256,
        "partition_rows_sha256": rows_sha256,
        "ordered_query_ids_sha256": canonical_sha256(ordered_ids),
        "fit_ids_sha256": canonical_sha256(fit_ids),
        "calibration_ids_sha256": canonical_sha256(calibration_ids),
        "train_count": len(rows),
        "fit_count": len(fit_ids),
        "calibration_count": len(calibration_ids),
    }
    artifact_id = ID_PREFIX + canonical_sha256(id_inputs)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "train_partition_id": artifact_id,
        "data_dependency": copy.deepcopy(dict(data_dependency)),
        "policy": policy,
        "config_sha256": config_sha256,
        "builder_code_sha256": builder_code_sha256,
        "partition_rows_sha256": rows_sha256,
        "ordered_query_ids_sha256": canonical_sha256(ordered_ids),
        "fit_ids_sha256": canonical_sha256(fit_ids),
        "calibration_ids_sha256": canonical_sha256(calibration_ids),
        "train_count": len(rows),
        "fit_count": len(fit_ids),
        "calibration_count": len(calibration_ids),
        "id_inputs": id_inputs,
    }


def _canonical_document(path: Path, document: Mapping[str, Any], *, label: str) -> None:
    try:
        actual = path.read_bytes()
    except OSError as exc:
        raise TrainPartitionError(f"cannot read {label}: {exc}") from exc
    if actual != canonical_json_bytes(document) + b"\n":
        raise TrainPartitionError(f"{label} is not canonical JSON")


def _validate_target(
    target: Path,
    *,
    workspace_root: Path,
    expected_data_dependency: Mapping[str, Any] | None,
    schema_path: Path,
    require_directory_name: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path]:
    if not target.is_dir() or target.is_symlink():
        raise TrainPartitionError(f"train partition target is not a real directory: {target}")
    ensure_exact_file_set(
        target,
        {
            "partition.jsonl",
            "partition.meta.json",
            "data_ref.json",
            "config.resolved.json",
            "payload_manifest.json",
        },
    )
    payload_hash = validate_payload_manifest(target)
    meta = load_json(target / "partition.meta.json")
    config = _load_and_validate_config(target / "config.resolved.json")
    data_dependency = load_json(target / "data_ref.json")
    rows = load_jsonl(target / "partition.jsonl")
    if not isinstance(meta, dict) or not isinstance(data_dependency, dict):
        raise TrainPartitionError("partition metadata or data dependency is malformed")
    validate_json_schema(meta, schema_path)
    _canonical_document(target / "partition.meta.json", meta, label="partition metadata")
    _canonical_document(target / "config.resolved.json", config, label="resolved config")
    _canonical_document(target / "data_ref.json", data_dependency, label="data dependency")

    try:
        frozen_dependency = validate_dependency_ref(
            data_dependency, expected_kind="data"
        )
    except TrainingArtifactError as exc:
        raise TrainPartitionError(f"invalid embedded data dependency: {exc}") from exc
    if meta.get("data_dependency") != frozen_dependency or meta.get(
        "id_inputs", {}
    ).get("data_dependency") != frozen_dependency:
        raise TrainPartitionError("metadata and embedded data dependency disagree")
    _require_expected_dependency(frozen_dependency, expected_data_dependency)
    data_target, train_records = _deep_data_records(frozen_dependency, workspace_root)

    if len(rows) != EXPECTED_TRAIN_COUNT:
        raise TrainPartitionError(
            f"partition rows must cover exactly {EXPECTED_TRAIN_COUNT} train queries"
        )
    row_keys = {
        "query_id",
        "content_sha256",
        "cluster_representative_query_id",
        "bucket",
        "partition",
    }
    for ordinal, row in enumerate(rows):
        if set(row) != row_keys:
            raise TrainPartitionError(f"partition row {ordinal} fields are not canonical")
        _canonical_numeric_query_id(row.get("query_id"))
        _canonical_numeric_query_id(row.get("cluster_representative_query_id"))
        if not isinstance(row.get("content_sha256"), str) or SHA256_RE.fullmatch(
            str(row.get("content_sha256"))
        ) is None:
            raise TrainPartitionError(f"partition row {ordinal} content hash is invalid")
        bucket = row.get("bucket")
        if isinstance(bucket, bool) or not isinstance(bucket, int) or not 0 <= bucket < 10000:
            raise TrainPartitionError(f"partition row {ordinal} bucket is invalid")
        if row.get("partition") not in {"fit", "calibration"}:
            raise TrainPartitionError(f"partition row {ordinal} label is invalid")
    query_ids = [str(row["query_id"]) for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise TrainPartitionError("partition query IDs are not unique")
    if query_ids != sorted(query_ids, key=int):
        raise TrainPartitionError("partition rows are not sorted by numeric query ID")
    expected_wire = canonical_jsonl_bytes(rows, key="query_id", numeric_key=True)
    try:
        actual_wire = (target / "partition.jsonl").read_bytes()
    except OSError as exc:
        raise TrainPartitionError(f"cannot read partition rows: {exc}") from exc
    if actual_wire != expected_wire:
        raise TrainPartitionError("partition rows are not canonical JSONL")
    fit_ids, calibration_ids = _row_partitions(rows)
    if not fit_ids or not calibration_ids:
        raise TrainPartitionError("fit and calibration partitions must both be non-empty")
    if set(fit_ids).intersection(calibration_ids) or set(fit_ids) | set(
        calibration_ids
    ) != set(query_ids):
        raise TrainPartitionError("fit/calibration IDs are not unique, disjoint, and exhaustive")
    fit_content = {
        str(row["content_sha256"]) for row in rows if row["partition"] == "fit"
    }
    calibration_content = {
        str(row["content_sha256"])
        for row in rows
        if row["partition"] == "calibration"
    }
    if fit_content.intersection(calibration_content):
        raise TrainPartitionError("normalized content overlaps fit and calibration pools")

    expected_rows = _partition_rows(train_records, config["policy"])
    if rows != expected_rows:
        raise TrainPartitionError(
            "partition row replay mismatch (content cluster, representative, bucket, or label)"
        )
    rows_hash = hashlib.sha256(expected_wire).hexdigest()
    builder_hash = sha256_file(__file__)
    expected_meta = _meta_document(
        data_dependency=frozen_dependency,
        config=config,
        rows=rows,
        rows_sha256=rows_hash,
        builder_code_sha256=builder_hash,
    )
    if meta != expected_meta:
        raise TrainPartitionError("partition metadata cannot be replayed from its inputs")
    if validate_payload_manifest(target) != payload_hash:
        raise TrainPartitionError("train partition target changed during validation")
    artifact_id = expected_meta["train_partition_id"]
    if require_directory_name and target.name != artifact_id:
        raise TrainPartitionError("train partition directory name differs from its artifact ID")
    return meta, rows, train_records, data_target


def _partition_dependency(target: Path, meta: Mapping[str, Any], workspace_root: Path) -> dict[str, Any]:
    locator_identity = {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_id": meta["train_partition_id"],
        "payload_manifest_sha256": validate_payload_manifest(target),
    }
    return portable_dependency(locator_identity, target, workspace_root)


def validate_train_partition_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    expected_data_dependency: Mapping[str, Any] | None = None,
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Deep-validate a partition target and replay it from finalized data."""

    root = Path(workspace_root).resolve()
    target = Path(target_dir).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise TrainPartitionError("train partition target must live below workspace_root") from exc
    try:
        meta, rows, _, _ = _validate_target(
            target,
            workspace_root=root,
            expected_data_dependency=expected_data_dependency,
            schema_path=Path(schema_path),
            require_directory_name=True,
        )
    except TrainPartitionError:
        raise
    except (TrainingArtifactError, Stage1DataError, OSError, ValueError) as exc:
        raise TrainPartitionError(f"train partition target validation failed: {exc}") from exc
    fit_ids, calibration_ids = _row_partitions(rows)
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "valid": True,
        "artifact_kind": ARTIFACT_KIND,
        "train_partition_id": meta["train_partition_id"],
        "payload_manifest_sha256": validate_payload_manifest(target),
        "partition_dependency": _partition_dependency(target, meta, root),
        "data_dependency": copy.deepcopy(meta["data_dependency"]),
        "train_count": meta["train_count"],
        "fit_count": meta["fit_count"],
        "calibration_count": meta["calibration_count"],
        "ordered_query_ids_sha256": meta["ordered_query_ids_sha256"],
        "fit_ids_sha256": meta["fit_ids_sha256"],
        "calibration_ids_sha256": meta["calibration_ids_sha256"],
        "fit_ids": fit_ids,
        "calibration_ids": calibration_ids,
    }


def validate_train_partition(
    partition_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    expected_data_dependency: Mapping[str, Any] | None = None,
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Validate both a locator ref and the complete partition target."""

    try:
        locator, target = resolve_locator_ref(partition_ref, expected_kind=ARTIFACT_KIND)
    except TrainingArtifactError as exc:
        raise TrainPartitionError(f"train partition locator validation failed: {exc}") from exc
    report = validate_train_partition_target(
        target,
        workspace_root=workspace_root,
        expected_data_dependency=expected_data_dependency,
        schema_path=schema_path,
    )
    if (
        report["train_partition_id"] != locator["artifact_id"]
        or report["payload_manifest_sha256"]
        != locator["payload_manifest_sha256"]
    ):
        raise TrainPartitionError("partition locator differs from its validated target")
    return report


def load_train_partition(
    partition_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    expected_data_dependency: Mapping[str, Any] | None = None,
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> TrainPartitionBundle:
    """Return the validated row and record views needed by downstream builders."""

    root = Path(workspace_root).resolve()
    try:
        locator, target = resolve_locator_ref(partition_ref, expected_kind=ARTIFACT_KIND)
    except TrainingArtifactError as exc:
        raise TrainPartitionError(f"train partition locator validation failed: {exc}") from exc
    report = validate_train_partition_target(
        target,
        workspace_root=root,
        expected_data_dependency=expected_data_dependency,
        schema_path=schema_path,
    )
    if (
        report["train_partition_id"] != locator["artifact_id"]
        or report["payload_manifest_sha256"]
        != locator["payload_manifest_sha256"]
    ):
        raise TrainPartitionError("partition locator differs from its validated target")
    meta = load_json(target / "partition.meta.json")
    rows = load_jsonl(target / "partition.jsonl")
    data_dependency = dict(meta["data_dependency"])
    data_target, train_records = _deep_data_records(data_dependency, root)
    by_id = {str(record["id"]): dict(record) for record in train_records}
    fit_ids = tuple(report["fit_ids"])
    calibration_ids = tuple(report["calibration_ids"])
    return TrainPartitionBundle(
        locator=dict(locator),
        target=target,
        partition_dependency=dict(report["partition_dependency"]),
        data_dependency=data_dependency,
        data_target=data_target,
        meta=dict(meta),
        rows=tuple(dict(row) for row in rows),
        train_records=tuple(dict(record) for record in train_records),
        fit_ids=fit_ids,
        calibration_ids=calibration_ids,
        fit_records=tuple(by_id[query_id] for query_id in fit_ids),
        calibration_records=tuple(by_id[query_id] for query_id in calibration_ids),
    )


def build_train_partition(
    *,
    data_ref: str | Path,
    write_ref: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    workspace_root: str | Path = REPOSITORY_ROOT,
    target_root: str | Path | None = None,
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Build and publish the immutable full-information-isolated partition."""

    root = Path(workspace_root).resolve()
    config = _load_and_validate_config(config_path)
    try:
        data_locator, data_target = resolve_locator_ref(data_ref, expected_kind="data")
        data_report = validate_data_target(data_target)
    except (TrainingArtifactError, Stage1DataError, OSError, ValueError) as exc:
        raise TrainPartitionError(f"input data_ref failed deep validation: {exc}") from exc
    if (
        data_report.get("data_build_id") != data_locator["artifact_id"]
        or data_report.get("payload_manifest_sha256")
        != data_locator["payload_manifest_sha256"]
        or data_report.get("split_counts", {}).get("train") != EXPECTED_TRAIN_COUNT
    ):
        raise TrainPartitionError("input data_ref is not the formal 5,781-record data target")
    data_dependency = portable_dependency(data_locator, data_target, root)
    _, train_records = _deep_data_records(data_dependency, root)
    rows = _partition_rows(train_records, config["policy"])
    rows_wire = canonical_jsonl_bytes(rows, key="query_id", numeric_key=True)
    builder_hash = sha256_file(__file__)
    meta = _meta_document(
        data_dependency=data_dependency,
        config=config,
        rows=rows,
        rows_sha256=hashlib.sha256(rows_wire).hexdigest(),
        builder_code_sha256=builder_hash,
    )
    artifact_id = meta["train_partition_id"]
    parent = (
        Path(target_root).resolve()
        if target_root is not None
        else (root / str(config["artifact_root"]) / "train_partitions").resolve()
    )
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise TrainPartitionError("train partition target_root must live below workspace_root") from exc
    target = parent / artifact_id
    staging = new_staging_directory(parent, artifact_id)
    try:
        write_canonical_json(staging / "data_ref.json", data_dependency)
        write_canonical_json(staging / "config.resolved.json", config)
        write_canonical_jsonl(
            staging / "partition.jsonl", rows, key="query_id", numeric_key=True
        )
        write_canonical_json(staging / "partition.meta.json", meta)
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda path: _validate_target(
                path,
                workspace_root=root,
                expected_data_dependency=data_dependency,
                schema_path=Path(schema_path),
                require_directory_name=False,
            ),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    report = validate_train_partition_target(
        target,
        workspace_root=root,
        expected_data_dependency=data_dependency,
        schema_path=schema_path,
    )
    if report["payload_manifest_sha256"] != payload_hash:
        raise TrainPartitionError("final partition payload changed after publication")
    return write_locator_ref(
        write_ref,
        artifact_kind=ARTIFACT_KIND,
        artifact_id=artifact_id,
        target=target,
        payload_manifest_sha256=payload_hash,
    )


__all__ = [
    "ARTIFACT_KIND",
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_CONFIG",
    "DEFAULT_SCHEMA",
    "EXPECTED_TRAIN_COUNT",
    "FROZEN_POLICY",
    "SCHEMA_VERSION",
    "TrainPartitionBundle",
    "TrainPartitionError",
    "build_train_partition",
    "calibration_bucket",
    "load_train_partition",
    "normalize_content",
    "normalized_content_sha256",
    "validate_train_partition",
    "validate_train_partition_target",
]
