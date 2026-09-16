"""Stage 1 data audit, adjudication, and immutable data artifacts.

This module intentionally treats ``data/full/std/train.json`` and
``data/full/std/test.json`` as the only source inputs for Stage 1.  It does not
reuse the legacy raw-to-std converter: that converter drops partially populated
quadruples and therefore has a different source contract.

The public functions are used by ``scripts/stage1/build_contexts.py``.  All
artifact payloads are deterministic and content addressed; external locator
refs are the only files that may contain absolute paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUDIT_SCHEMA_VERSION = "stage1-data-audit/v1"
AUDIT_ISSUE_SCHEMA_VERSION = "stage1-data-audit-issue/v1"
ADJUDICATION_ROW_SCHEMA_VERSION = "stage1-data-adjudication-row/v1"
DECLARATION_SCHEMA_VERSION = "stage1-data-reviewer-declaration/v1"
RUBRIC_META_SCHEMA_VERSION = "stage1-data-adjudication-rubric-meta/v1"
DATA_SCHEMA_VERSION = "stage1-data-manifest/v1"
SPLIT_SCHEMA_VERSION = "stage1-split/v1"
LOCATOR_REF_SCHEMA_VERSION = "stage1-locator-ref/v1"
DEPENDENCY_REF_SCHEMA_VERSION = "stage1-dependency-ref/v1"
PAYLOAD_MANIFEST_SCHEMA_VERSION = "stage1-payload-manifest/v1"

AUDIT_RULE_VERSION = "stage1-std-data-audit/v1"
ISSUE_RULE_VERSION = "stage1-data-issues/v1"
WARNING_RULE_VERSION = "stage1-data-warnings/v1"
NORMALIZATION_SCHEMA_VERSION = "stage1-source-normalization/v1"
OUTPUT_SCHEMA_VERSION = "stage1-normalized-record/v1"
SPLIT_POLICY_VERSION = "prefix-5781-643/v1"
ID_POLICY_VERSION = "canonical-decimal-string/v1"
CANONICAL_JSONL_VERSION = "canonical-jsonl/v1"
ADJUDICATION_FRAME_SCHEMA_VERSION = "stage1-data-adjudication-frame/v1"
HUMAN_QUEUE_ROW_SCHEMA_VERSION = "stage1-data-human-adjudication-queue-row/v1"
BLIND_PANEL_REVIEWER_ID = "dual-blind-panel-v1"
BLIND_REVIEW_ISSUE_KIND = "group-hate"
HUMAN_ONLY_ISSUE_KINDS = frozenset({"field-type"})

EXPECTED_SOURCE_PATHS = {
    "std-train": "data/full/std/train.json",
    "std-test": "data/full/std/test.json",
}
GROUP_ORDER = ("Racism", "Region", "LGBTQ", "Sexism", "others", "non-hate")
GROUP_SET = frozenset(GROUP_ORDER)
CANONICAL_ID_RE = re.compile(r"^[1-9][0-9]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Stage1DataError(ValueError):
    """Raised when a Stage 1 data contract is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's canonical JSON representation."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Stage1DataError(f"Value is not canonical-JSON serializable: {exc}") from exc
    return rendered.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _ordered_json_bytes(value: Any) -> bytes:
    """Encode fixed-order record/quad wires without alphabetizing their keys."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Stage1DataError(f"Value is not ordered-JSON serializable: {exc}") from exc
    return rendered.encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]], *, key: str) -> bytes:
    materialized = [dict(row) for row in rows]
    try:
        materialized.sort(key=lambda row: str(row[key]))
    except KeyError as exc:
        raise Stage1DataError(f"JSONL row lacks primary key {key!r}") from exc
    if len({str(row[key]) for row in materialized}) != len(materialized):
        raise Stage1DataError(f"JSONL primary key {key!r} is not unique")
    return b"".join(canonical_json_bytes(row) + b"\n" for row in materialized)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage1DataError(f"Cannot read JSON from {source}: {exc}") from exc


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Stage1DataError(
                        f"Blank JSONL line is forbidden: {source}:{line_number}"
                    )
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise Stage1DataError(
                        f"JSONL row must be an object: {source}:{line_number}"
                    )
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage1DataError(f"Cannot read JSONL from {source}: {exc}") from exc
    return rows


def _atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: str | Path, value: Any) -> None:
    _atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def _write_ordered_json(path: str | Path, value: Any) -> None:
    _atomic_write_bytes(path, _ordered_json_bytes(value) + b"\n")


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]], *, key: str) -> None:
    _atomic_write_bytes(path, canonical_jsonl_bytes(rows, key=key))


def _workspace_root(config_path: str | Path) -> Path:
    config = Path(config_path).resolve()
    for candidate in (config.parent, *config.parents):
        if all((candidate / relative).is_file() for relative in EXPECTED_SOURCE_PATHS.values()):
            return candidate
    raise Stage1DataError(
        f"Cannot locate workspace root containing the frozen std sources from {config}"
    )


def _load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    config = _load_json(config_path)
    if not isinstance(config, dict):
        raise Stage1DataError("Stage 1 config must be a JSON object")
    if config.get("schema_version") != "stage1-context-factorial-config/v1":
        raise Stage1DataError("Unsupported Stage 1 config schema_version")
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise Stage1DataError("Stage 1 config must contain a data object")
    sources = data_config.get("sources")
    if sources != {"train": EXPECTED_SOURCE_PATHS["std-train"], "test": EXPECTED_SOURCE_PATHS["std-test"]}:
        raise Stage1DataError(
            "Stage 1 source-of-truth must be exactly data/full/std/train.json and "
            "data/full/std/test.json"
        )
    if data_config.get("source_of_truth") != "data/full/std/v1":
        raise Stage1DataError("data.source_of_truth must be data/full/std/v1")
    if data_config.get("id_policy") != ID_POLICY_VERSION:
        raise Stage1DataError(f"data.id_policy must be {ID_POLICY_VERSION}")
    split = data_config.get("split")
    expected_split = {
        "policy": SPLIT_POLICY_VERSION,
        "train_count": 5781,
        "dev_count": 643,
        "test_count": 1605,
    }
    if split != expected_split:
        raise Stage1DataError(f"data.split must equal {expected_split}")
    artifact_root = config.get("artifact_root")
    if not isinstance(artifact_root, str) or not artifact_root or Path(artifact_root).is_absolute():
        raise Stage1DataError("artifact_root must be a non-empty workspace-relative path")
    return config, _workspace_root(config_path)


def _source_paths(config: Mapping[str, Any], workspace_root: Path) -> dict[str, Path]:
    sources = config["data"]["sources"]
    resolved = {
        "std-train": (workspace_root / sources["train"]).resolve(),
        "std-test": (workspace_root / sources["test"]).resolve(),
    }
    for source_key, expected_relative in EXPECTED_SOURCE_PATHS.items():
        expected = (workspace_root / expected_relative).resolve()
        if resolved[source_key] != expected:
            raise Stage1DataError(f"Unexpected source path for {source_key}")
        if not expected.is_file():
            raise Stage1DataError(f"Missing frozen source file: {expected}")
    return resolved


def canonical_record_id(raw_id: Any, *, source_input: bool = False) -> str:
    """Convert a positive integer record ID to canonical decimal text.

    The frozen std sources must contain JSON integers.  Corrected in-memory rows
    may already contain canonical decimal strings (for duplicate-ID fixtures).
    """

    if isinstance(raw_id, bool):
        raise Stage1DataError("Boolean is not a valid record ID")
    if isinstance(raw_id, int):
        if raw_id <= 0:
            raise Stage1DataError("Record ID must be positive")
        return str(raw_id)
    if not source_input and isinstance(raw_id, str) and CANONICAL_ID_RE.fullmatch(raw_id):
        return raw_id
    expected = "a positive JSON integer" if source_input else "a positive integer or canonical decimal string"
    raise Stage1DataError(f"Record ID must be {expected}; got {raw_id!r}")


def _logical_split(source_key: str, source_ordinal: int) -> str:
    if source_key == "std-test":
        return "test"
    return "train" if source_ordinal < 5781 else "dev"


def _parse_group_atoms(raw_group: Any) -> list[str]:
    if isinstance(raw_group, str):
        atoms = [atom.strip() for atom in raw_group.split(",")]
    elif isinstance(raw_group, (list, tuple)):
        atoms = [str(atom).strip() for atom in raw_group]
    else:
        raise Stage1DataError(f"targeted_group must be a string or array; got {raw_group!r}")
    if not atoms or any(not atom for atom in atoms):
        raise Stage1DataError(f"targeted_group contains an empty atom: {raw_group!r}")
    unknown = [atom for atom in atoms if atom not in GROUP_SET]
    if unknown:
        raise Stage1DataError(f"Unknown targeted_group atom(s): {unknown}")
    if len(atoms) != len(set(atoms)):
        raise Stage1DataError(f"Duplicate targeted_group atom: {raw_group!r}")
    if "non-hate" in atoms and len(atoms) != 1:
        raise Stage1DataError("non-hate cannot coexist with another targeted_group atom")
    return atoms


def _validate_source_records(source_key: str, records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise Stage1DataError(f"{source_key} must be a JSON array")
    expected_count = 6424 if source_key == "std-train" else 1605
    if len(records) != expected_count:
        raise Stage1DataError(
            f"{source_key} has {len(records)} records; frozen contract requires {expected_count}"
        )
    validated: list[dict[str, Any]] = []
    for source_ordinal, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"id", "content", "quadruples"}:
            raise Stage1DataError(
                f"{source_key}[{source_ordinal}] must have exactly id/content/quadruples"
            )
        canonical_record_id(record["id"], source_input=True)
        if not isinstance(record["content"], str) or not record["content"]:
            raise Stage1DataError(f"{source_key}[{source_ordinal}].content must be non-empty text")
        quadruples = record["quadruples"]
        if not isinstance(quadruples, list) or not quadruples:
            raise Stage1DataError(
                f"{source_key}[{source_ordinal}].quadruples must be a non-empty array"
            )
        for tuple_index, quad in enumerate(quadruples):
            if not isinstance(quad, dict) or set(quad) != {
                "target",
                "argument",
                "targeted_group",
                "hateful",
            }:
                raise Stage1DataError(
                    f"{source_key}[{source_ordinal}].quadruples[{tuple_index}] has invalid keys"
                )
            _parse_group_atoms(quad["targeted_group"])
            if not isinstance(quad["hateful"], str):
                raise Stage1DataError(
                    f"{source_key}[{source_ordinal}].quadruples[{tuple_index}].hateful "
                    "must be a string"
                )
            if quad["hateful"] not in {"hate", "non-hate", "NULL"}:
                raise Stage1DataError(
                    f"Unknown hateful label at {source_key}[{source_ordinal}] tuple {tuple_index}"
                )
        validated.append(record)
    return validated


def _load_sources(
    config: Mapping[str, Any], workspace_root: Path, *, copy_records: bool = False
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    paths = _source_paths(config, workspace_root)
    records: dict[str, list[dict[str, Any]]] = {}
    inventory_entries: list[dict[str, Any]] = []
    inventory_by_key: dict[str, dict[str, Any]] = {}
    for source_key in ("std-train", "std-test"):
        path = paths[source_key]
        loaded = _validate_source_records(source_key, _load_json(path))
        records[source_key] = copy.deepcopy(loaded) if copy_records else loaded
        stat = path.stat()
        entry = {
            "source_key": source_key,
            "repo_relative_path": EXPECTED_SOURCE_PATHS[source_key],
            "size": stat.st_size,
            "sha256": sha256_file(path),
            "record_count": len(loaded),
            "tuple_count": sum(len(record["quadruples"]) for record in loaded),
        }
        inventory_entries.append(entry)
        inventory_by_key[source_key] = entry
    inventory = {
        "schema_version": "stage1-source-inventory/v1",
        "sources": inventory_entries,
    }
    return records, {"document": inventory, "by_key": inventory_by_key}


def _id_list_sha256(ids: Sequence[str]) -> str:
    return canonical_json_sha256(list(ids))


def _build_split_manifest(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    inventory: Mapping[str, Any],
    *,
    require_unique: bool = True,
) -> dict[str, Any]:
    train_source = records["std-train"]
    test_source = records["std-test"]
    train_ids = [canonical_record_id(record["id"]) for record in train_source[:5781]]
    dev_ids = [canonical_record_id(record["id"]) for record in train_source[5781:]]
    test_ids = [canonical_record_id(record["id"]) for record in test_source]
    all_ids = train_ids + dev_ids + test_ids
    if len(all_ids) != 8029:
        raise Stage1DataError("Frozen split must contain exactly 8029 IDs")
    if require_unique and len(set(all_ids)) != len(all_ids):
        raise Stage1DataError("Frozen split IDs are not globally unique")
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "source_sha256": {
            source_key: inventory["by_key"][source_key]["sha256"]
            for source_key in ("std-train", "std-test")
        },
        "policy": SPLIT_POLICY_VERSION,
        "id_policy": ID_POLICY_VERSION,
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "test_ids": test_ids,
        "train_ids_sha256": _id_list_sha256(train_ids),
        "dev_ids_sha256": _id_list_sha256(dev_ids),
        "test_ids_sha256": _id_list_sha256(test_ids),
    }


def _location(
    *,
    source_key: str,
    source_ordinal: int,
    tuple_index: int | None,
    json_pointer: str,
    observed_value: Any,
    source_inventory: Mapping[str, Any],
    source_record_id: str,
) -> dict[str, Any]:
    location = {
        "source_key": source_key,
        "source_file_sha256": source_inventory["by_key"][source_key]["sha256"],
        "source_ordinal": source_ordinal,
        "split": _logical_split(source_key, source_ordinal),
        "source_record_id": source_record_id,
        "json_pointer": json_pointer,
        "observed_value_sha256": canonical_json_sha256(observed_value),
    }
    if tuple_index is not None:
        location["tuple_index"] = tuple_index
    return location


def _issue_id(issue_code: str, locations: Sequence[Mapping[str, Any]]) -> str:
    stable_locations = [
        {
            key: location[key]
            for key in ("source_key", "source_ordinal", "tuple_index", "json_pointer")
            if key in location
        }
        for location in locations
    ]
    stable_locations.sort(
        key=lambda item: (
            item["source_key"],
            item["source_ordinal"],
            item.get("tuple_index", -1),
            item["json_pointer"],
        )
    )
    digest = canonical_json_sha256(
        {
            "issue_code": issue_code,
            "issue_rule_version": ISSUE_RULE_VERSION,
            "locations": stable_locations,
        }
    )
    return f"dissue:v1:{digest}"


def _warning_id(warning_code: str, location: Mapping[str, Any]) -> str:
    stable_location = {
        key: location[key]
        for key in ("source_key", "source_ordinal", "tuple_index", "json_pointer")
        if key in location
    }
    return "dwarn:v1:" + canonical_json_sha256(
        {
            "warning_code": warning_code,
            "warning_rule_version": WARNING_RULE_VERSION,
            "location": stable_location,
        }
    )


def _group_hate_issue_spec(group_atoms: Sequence[str], hateful: str) -> dict[str, Any] | None:
    expected = "non-hate" if list(group_atoms) == ["non-hate"] else "hate"
    if hateful == expected:
        return None
    if hateful == "NULL":
        return {
            "issue_code": "hateful-null-sentinel",
            "issue_subtype": "missing-hateful-label",
            "accept_allowed": False,
            "editable_fields": ("hateful",),
        }
    if list(group_atoms) == ["non-hate"] and hateful == "hate":
        return {
            "issue_code": "group-hate-conflict",
            "issue_subtype": "non-hate-group-with-hate-label",
            "accept_allowed": False,
            "editable_fields": ("targeted_group", "hateful"),
        }
    if list(group_atoms) != ["non-hate"] and hateful == "non-hate":
        return {
            "issue_code": "group-hate-atypical",
            "issue_subtype": "specific-group-with-non-hate-label",
            "accept_allowed": True,
            "editable_fields": ("targeted_group", "hateful"),
        }
    raise Stage1DataError(f"Unhandled group/hateful combination: {group_atoms!r}, {hateful!r}")


def _audit_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    inventory: Mapping[str, Any],
    *,
    data_audit_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    normalization_observations: Counter[str] = Counter()
    id_locations: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for source_key in ("std-train", "std-test"):
        for source_ordinal, record in enumerate(records[source_key]):
            source_record_id = canonical_record_id(record["id"], source_input=True)
            id_locations[source_record_id].append(
                _location(
                    source_key=source_key,
                    source_ordinal=source_ordinal,
                    tuple_index=None,
                    json_pointer="/id",
                    observed_value=record["id"],
                    source_inventory=inventory,
                    source_record_id=source_record_id,
                )
            )
            if record["content"] != record["content"].strip():
                normalization_observations["content_edge_whitespace_preserved"] += 1
            if record["content"] != unicodedata.normalize("NFC", record["content"]):
                normalization_observations["content_non_nfc_preserved"] += 1

            for tuple_index, quad in enumerate(record["quadruples"]):
                group_atoms = _parse_group_atoms(quad["targeted_group"])
                tuple_pointer = f"/quadruples/{tuple_index}"
                spec = _group_hate_issue_spec(group_atoms, quad["hateful"])
                if spec is not None:
                    location = _location(
                        source_key=source_key,
                        source_ordinal=source_ordinal,
                        tuple_index=tuple_index,
                        json_pointer=tuple_pointer,
                        observed_value=quad,
                        source_inventory=inventory,
                        source_record_id=source_record_id,
                    )
                    issue_code = spec["issue_code"]
                    issue = {
                        "schema_version": AUDIT_ISSUE_SCHEMA_VERSION,
                        "data_audit_id": data_audit_id,
                        "issue_id": _issue_id(issue_code, [location]),
                        "issue_kind": "group-hate",
                        "issue_code": issue_code,
                        "issue_subtype": spec["issue_subtype"],
                        "issue_rule_version": ISSUE_RULE_VERSION,
                        "severity": "blocking",
                        "locations": [location],
                        "accept_allowed": spec["accept_allowed"],
                        "allowed_edit_paths": [
                            f"{tuple_pointer}/{field}" for field in spec["editable_fields"]
                        ],
                        "review_context": {
                            "content": record["content"],
                            "tuple_before": quad,
                            "expected_hateful_from_legacy_coupling": (
                                "non-hate" if group_atoms == ["non-hate"] else "hate"
                            ),
                        },
                    }
                    issues.append(issue)

                for field in ("target", "argument"):
                    value = quad[field]
                    pointer = f"{tuple_pointer}/{field}"
                    if value == "NULL":
                        normalization_observations[f"legacy_null_{field}_to_json_null"] += 1
                    elif value is None:
                        normalization_observations[f"json_null_{field}_preserved"] += 1
                    elif not isinstance(value, str):
                        location = _location(
                            source_key=source_key,
                            source_ordinal=source_ordinal,
                            tuple_index=tuple_index,
                            json_pointer=pointer,
                            observed_value=value,
                            source_inventory=inventory,
                            source_record_id=source_record_id,
                        )
                        issue_code = "non-string-quad-field"
                        issues.append(
                            {
                                "schema_version": AUDIT_ISSUE_SCHEMA_VERSION,
                                "data_audit_id": data_audit_id,
                                "issue_id": _issue_id(issue_code, [location]),
                                "issue_kind": "field-type",
                                "issue_code": issue_code,
                                "issue_subtype": f"numeric-{field}",
                                "issue_rule_version": ISSUE_RULE_VERSION,
                                "severity": "blocking",
                                "locations": [location],
                                "accept_allowed": False,
                                "allowed_edit_paths": [pointer],
                                "review_context": {
                                    "content": record["content"],
                                    "tuple_before": quad,
                                    "field": field,
                                    "observed_type": type(value).__name__,
                                    "allowed_correction_types": ["string", "null"],
                                },
                            }
                        )
                    else:
                        if value != value.strip():
                            normalization_observations[f"{field}_edge_whitespace_trimmed"] += 1
                        if value != unicodedata.normalize("NFC", value):
                            normalization_observations[f"{field}_unicode_nfc_normalized"] += 1
                        if value not in record["content"]:
                            location = _location(
                                source_key=source_key,
                                source_ordinal=source_ordinal,
                                tuple_index=tuple_index,
                                json_pointer=pointer,
                                observed_value=value,
                                source_inventory=inventory,
                                source_record_id=source_record_id,
                            )
                            warning_code = "field-not-exact-content-substring"
                            warnings.append(
                                {
                                    "schema_version": "stage1-data-audit-warning/v1",
                                    "warning_id": _warning_id(warning_code, location),
                                    "warning_code": warning_code,
                                    "warning_rule_version": WARNING_RULE_VERSION,
                                    "severity": "warning",
                                    "location": location,
                                    "field": field,
                                    "message": (
                                        "Gold field is not an exact content substring; preserve gold and "
                                        "do not auto-correct."
                                    ),
                                }
                            )

                canonical_group_surface = ", ".join(group_atoms)
                if quad["targeted_group"] != canonical_group_surface:
                    normalization_observations["targeted_group_surface_normalized"] += 1

    for record_id, locations in sorted(id_locations.items(), key=lambda item: int(item[0])):
        if len(locations) < 2:
            continue
        locations.sort(
            key=lambda item: (item["source_key"], item["source_ordinal"], item["json_pointer"])
        )
        issue_code = "duplicate-record-id"
        issues.append(
            {
                "schema_version": AUDIT_ISSUE_SCHEMA_VERSION,
                "data_audit_id": data_audit_id,
                "issue_id": _issue_id(issue_code, locations),
                "issue_kind": "record-id",
                "issue_code": issue_code,
                "issue_subtype": "global-duplicate-id",
                "issue_rule_version": ISSUE_RULE_VERSION,
                "severity": "blocking",
                "locations": locations,
                "accept_allowed": False,
                "allowed_edit_paths": ["/id"],
                "review_context": {
                    "duplicate_record_id": record_id,
                    "location_count": len(locations),
                },
            }
        )

    if len({issue["issue_id"] for issue in issues}) != len(issues):
        raise Stage1DataError("Generated blocking issue IDs are not unique")
    if len({warning["warning_id"] for warning in warnings}) != len(warnings):
        raise Stage1DataError("Generated warning IDs are not unique")
    issues.sort(key=lambda issue: issue["issue_id"])
    warnings.sort(key=lambda warning: warning["warning_id"])
    return issues, warnings, dict(sorted(normalization_observations.items()))


def _rubric_meta(rubric_body_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": RUBRIC_META_SCHEMA_VERSION,
        "rubric_version": "stage1-data-adjudication-rubric/v1",
        "rubric_body_sha256": rubric_body_sha256,
        "declaration_schema_version": DECLARATION_SCHEMA_VERSION,
        "decision_codes": ["accepted", "corrected"],
        "reason_codes": {
            "group-hate-atypical": {
                "accepted": ["valid-independent-label-combination"],
                "corrected": ["correct-source-label", "resolve-group-hate-conflict"],
            },
            "hateful-null-sentinel": {
                "accepted": [],
                "corrected": ["replace-legacy-null", "correct-source-label"],
            },
            "group-hate-conflict": {
                "accepted": [],
                "corrected": ["resolve-group-hate-conflict", "correct-source-label"],
            },
            "non-string-quad-field": {
                "accepted": [],
                "corrected": ["coerce-numeric-annotation", "set-explicit-null"],
            },
            "duplicate-record-id": {
                "accepted": [],
                "corrected": ["correct-source-record-id"],
            },
        },
        "edit_contract": {
            "op": "set",
            "field_type_corrections": ["string", "null"],
            "canonical_order": ["location_index", "json_pointer"],
        },
    }


def _payload_manifest(directory: Path) -> dict[str, Any]:
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "payload_manifest.json":
            continue
        relative = path.relative_to(directory).as_posix()
        files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": PAYLOAD_MANIFEST_SCHEMA_VERSION, "files": files}


def _verify_payload_manifest(directory: Path) -> str:
    manifest_path = directory / "payload_manifest.json"
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PAYLOAD_MANIFEST_SCHEMA_VERSION:
        raise Stage1DataError(f"Invalid payload manifest in {directory}")
    expected = _payload_manifest(directory)
    if manifest != expected:
        raise Stage1DataError(f"Payload manifest mismatch in {directory}")
    return sha256_file(manifest_path)


def _finalize_target(temporary_directory: Path, target_directory: Path) -> str:
    _write_json(temporary_directory / "payload_manifest.json", _payload_manifest(temporary_directory))
    payload_manifest_sha256 = _verify_payload_manifest(temporary_directory)
    if target_directory.exists():
        existing_hash = _verify_payload_manifest(target_directory)
        if existing_hash != payload_manifest_sha256:
            raise Stage1DataError(
                f"Lifecycle ID collision: {target_directory.name} already has a different payload"
            )
        shutil.rmtree(temporary_directory)
        return existing_hash
    target_directory.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_directory, target_directory)
    return payload_manifest_sha256


def _write_locator_ref(
    ref_path: str | Path,
    *,
    artifact_kind: str,
    artifact_id: str,
    target_directory: Path,
    payload_manifest_sha256: str,
) -> dict[str, Any]:
    ref = {
        "schema_version": LOCATOR_REF_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "artifact_id": artifact_id,
        "target_path": str(target_directory.resolve()),
        "payload_manifest_sha256": payload_manifest_sha256,
    }
    _write_json(ref_path, ref)
    return ref


def _resolve_locator_ref(ref_path: str | Path, *, artifact_kind: str) -> tuple[dict[str, Any], Path]:
    ref = _load_json(ref_path)
    if not isinstance(ref, dict) or ref.get("schema_version") != LOCATOR_REF_SCHEMA_VERSION:
        raise Stage1DataError(f"Invalid locator ref: {ref_path}")
    if ref.get("artifact_kind") != artifact_kind:
        raise Stage1DataError(
            f"Expected {artifact_kind} ref, got {ref.get('artifact_kind')!r}"
        )
    artifact_id = ref.get("artifact_id")
    target_path = ref.get("target_path")
    payload_hash = ref.get("payload_manifest_sha256")
    if not isinstance(artifact_id, str) or not isinstance(target_path, str):
        raise Stage1DataError(f"Malformed locator ref: {ref_path}")
    if not isinstance(payload_hash, str) or not SHA256_RE.fullmatch(payload_hash):
        raise Stage1DataError(f"Malformed payload hash in locator ref: {ref_path}")
    target = Path(target_path)
    if not target.is_absolute() or not target.is_dir():
        raise Stage1DataError(f"Locator target does not exist: {target}")
    if target.name != artifact_id:
        raise Stage1DataError("Locator artifact ID does not match target directory")
    actual_hash = _verify_payload_manifest(target)
    if actual_hash != payload_hash:
        raise Stage1DataError("Locator payload hash does not match target")
    return ref, target


def _portable_dependency(locator: Mapping[str, Any], target: Path, workspace_root: Path) -> dict[str, Any]:
    try:
        logical_path = target.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError as exc:
        raise Stage1DataError("Artifact target must live below the workspace root") from exc
    return {
        "schema_version": DEPENDENCY_REF_SCHEMA_VERSION,
        "artifact_kind": locator["artifact_kind"],
        "artifact_id": locator["artifact_id"],
        "payload_manifest_sha256": locator["payload_manifest_sha256"],
        "logical_repo_path": logical_path,
    }


def _artifact_root(config: Mapping[str, Any], workspace_root: Path) -> Path:
    return (workspace_root / config["artifact_root"]).resolve()


def _new_target_temp(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{prefix}.", dir=str(parent)))


def _check_expected_audit_counts(config: Mapping[str, Any], issues: Sequence[Mapping[str, Any]], warnings: Sequence[Mapping[str, Any]]) -> None:
    expected = config["data"].get("expected_counts", {})
    actual = {
        "group_hate_blocking": sum(issue["issue_kind"] == "group-hate" for issue in issues),
        "field_type_blocking": sum(issue["issue_kind"] == "field-type" for issue in issues),
        "blocking_total": len(issues),
        "substring_warnings": sum(
            warning["warning_code"] == "field-not-exact-content-substring"
            for warning in warnings
        ),
    }
    for key, expected_value in expected.items():
        if expected_value is not None and actual.get(key) != expected_value:
            raise Stage1DataError(
                f"Audit count {key}={actual.get(key)} does not match frozen expectation {expected_value}"
            )


def audit_data(
    *,
    config_path: str | Path,
    review_rubric_path: str | Path,
    write_ref: str | Path,
) -> dict[str, Any]:
    """Create an immutable Stage 1 data-audit target and locator ref."""

    config, workspace_root = _load_config(config_path)
    records, inventory = _load_sources(config, workspace_root)
    split_manifest = _build_split_manifest(records, inventory, require_unique=False)

    rubric_path = Path(review_rubric_path)
    expected_rubric = workspace_root / "config/stage1/data_adjudication_rubric.md"
    if rubric_path.resolve() != expected_rubric.resolve():
        raise Stage1DataError(
            "The canonical rubric must be config/stage1/data_adjudication_rubric.md"
        )
    rubric_bytes = rubric_path.read_bytes().replace(b"\r\n", b"\n")
    try:
        rubric_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Stage1DataError("Data adjudication rubric must be UTF-8") from exc
    rubric_body_sha256 = hashlib.sha256(rubric_bytes).hexdigest()
    rubric_meta = _rubric_meta(rubric_body_sha256)
    rubric_meta_sha256 = canonical_json_sha256(rubric_meta)

    module_code_sha256 = sha256_file(__file__)
    data_config = config["data"]
    audit_id_inputs = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source_inventory_sha256": canonical_json_sha256(inventory["document"]),
        "split_policy_sha256": canonical_json_sha256(data_config["split"]),
        "audit_config_sha256": canonical_json_sha256(config),
        "audit_rule_version": AUDIT_RULE_VERSION,
        "normalization_schema_version": NORMALIZATION_SCHEMA_VERSION,
        "rubric_body_sha256": rubric_body_sha256,
        "rubric_meta_sha256": rubric_meta_sha256,
        "audit_code_sha256": module_code_sha256,
    }
    data_audit_id = "daudit-" + canonical_json_sha256(audit_id_inputs)
    issues, warnings, normalization_observations = _audit_records(
        records, inventory, data_audit_id=data_audit_id
    )
    _check_expected_audit_counts(config, issues, warnings)

    ordered_issue_ids = [issue["issue_id"] for issue in issues]
    ordered_issue_ids_sha256 = canonical_json_sha256(ordered_issue_ids)
    issue_counts_by_kind = Counter(issue["issue_kind"] for issue in issues)
    issue_counts_by_code = Counter(issue["issue_code"] for issue in issues)
    issue_counts_by_split: Counter[str] = Counter()
    for issue in issues:
        splits = {location["split"] for location in issue["locations"]}
        issue_counts_by_split[next(iter(splits)) if len(splits) == 1 else "multi-split"] += 1
    warning_counts_by_field = Counter(warning["field"] for warning in warnings)

    audit_meta = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "data_audit_id": data_audit_id,
        "ordered_issue_ids_sha256": ordered_issue_ids_sha256,
        "blocking_issue_count": len(issues),
        "blocking_issue_counts_by_kind": dict(sorted(issue_counts_by_kind.items())),
        "blocking_issue_counts_by_code": dict(sorted(issue_counts_by_code.items())),
        "blocking_issue_counts_by_split": dict(sorted(issue_counts_by_split.items())),
        "warning_count": len(warnings),
        "audit_id_inputs": audit_id_inputs,
    }
    audit_report = {
        "schema_version": "stage1-data-audit-report/v1",
        "data_audit_id": data_audit_id,
        "source_summary": {
            entry["source_key"]: {
                "record_count": entry["record_count"],
                "tuple_count": entry["tuple_count"],
            }
            for entry in inventory["document"]["sources"]
        },
        "blocking": {
            "count": len(issues),
            "by_kind": dict(sorted(issue_counts_by_kind.items())),
            "by_code": dict(sorted(issue_counts_by_code.items())),
        },
        "warnings": {
            "count": len(warnings),
            "by_field": dict(sorted(warning_counts_by_field.items())),
            "items": warnings,
        },
        "normalization_observations": normalization_observations,
    }
    templates = [
        {
            "schema_version": ADJUDICATION_ROW_SCHEMA_VERSION,
            "data_audit_id": data_audit_id,
            "issue_id": issue["issue_id"],
            "issue_kind": issue["issue_kind"],
            "decision": "",
            "edits": [],
            "reason_code": "",
            "reason": "",
            "reviewer_id": "",
            "reviewed_at": "",
        }
        for issue in issues
    ]
    provenance = {
        "schema_version": "stage1-data-audit-provenance/v1",
        "data_audit_id": data_audit_id,
        "source_inventory_sha256": audit_id_inputs["source_inventory_sha256"],
        "config_sha256": canonical_json_sha256(config),
        "rubric_body_sha256": rubric_body_sha256,
        "rubric_meta_sha256": rubric_meta_sha256,
        "audit_code_sha256": module_code_sha256,
    }

    target_parent = _artifact_root(config, workspace_root) / "data_audits"
    target = target_parent / data_audit_id
    temporary = _new_target_temp(target_parent, data_audit_id)
    try:
        _write_json(temporary / "config.resolved.json", config)
        _write_json(temporary / "source_inventory.json", inventory["document"])
        _write_json(temporary / "split_manifest.proposed.json", split_manifest)
        _write_json(temporary / "audit.meta.json", audit_meta)
        _write_jsonl(temporary / "issues.jsonl", issues, key="issue_id")
        _atomic_write_bytes(temporary / "adjudication_rubric.md", rubric_bytes)
        _write_json(temporary / "adjudication_rubric.meta.json", rubric_meta)
        _write_jsonl(
            temporary / "adjudication_template.jsonl", templates, key="issue_id"
        )
        _write_json(temporary / "audit_report.json", audit_report)
        _write_json(temporary / "provenance.json", provenance)
        payload_hash = _finalize_target(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return _write_locator_ref(
        write_ref,
        artifact_kind="data-audit",
        artifact_id=data_audit_id,
        target_directory=target,
        payload_manifest_sha256=payload_hash,
    )


def export_data_adjudication(
    *, audit_ref: str | Path, output_path: str | Path
) -> Path:
    """Export the immutable empty adjudication template to a human workspace."""

    _, audit_target = _resolve_locator_ref(audit_ref, artifact_kind="data-audit")
    source = audit_target / "adjudication_template.jsonl"
    if not source.is_file():
        raise Stage1DataError("Audit target lacks adjudication_template.jsonl")
    _atomic_write_bytes(output_path, source.read_bytes())
    return Path(output_path)


def export_data_review_subset(
    *,
    audit_ref: str | Path,
    issue_kinds: Sequence[str],
    reviewer_id: str,
    output_path: str | Path,
    packet_output_path: str | Path,
) -> dict[str, Any]:
    """Export a model-output-free human workspace for selected issue kinds."""

    requested = {str(value) for value in issue_kinds if str(value)}
    if not requested:
        raise Stage1DataError("At least one issue_kind is required")
    if not reviewer_id.strip():
        raise Stage1DataError("reviewer_id must be non-empty")
    _, audit_target = _resolve_locator_ref(audit_ref, artifact_kind="data-audit")
    issues = _read_jsonl(audit_target / "issues.jsonl")
    templates = _read_jsonl(audit_target / "adjudication_template.jsonl")
    available = {str(issue.get("issue_kind")) for issue in issues}
    unknown = sorted(requested - available)
    if unknown:
        raise Stage1DataError(f"Unknown audit issue_kind values: {unknown}")
    selected_issues = [issue for issue in issues if issue.get("issue_kind") in requested]
    selected_ids = {issue["issue_id"] for issue in selected_issues}
    selected_templates = []
    for row in templates:
        if row.get("issue_id") not in selected_ids:
            continue
        completed = dict(row)
        completed["reviewer_id"] = reviewer_id
        selected_templates.append(completed)
    if len(selected_templates) != len(selected_issues):
        raise Stage1DataError("Audit issue/template subset is inconsistent")
    packets = [
        {
            "schema_version": "stage1-local-human-review-packet/v1",
            "issue_id": issue["issue_id"],
            "issue_kind": issue["issue_kind"],
            "issue_code": issue["issue_code"],
            "accept_allowed": issue["accept_allowed"],
            "allowed_edit_paths": issue["allowed_edit_paths"],
            "locations": issue["locations"],
            "review_context": issue["review_context"],
            "model_votes_hidden": True,
            "saw_condition_outputs": False,
            "saw_model_scores": False,
        }
        for issue in selected_issues
    ]
    _write_jsonl(output_path, selected_templates, key="issue_id")
    _write_jsonl(packet_output_path, packets, key="issue_id")
    return {
        "schema_version": "stage1-data-review-subset-export/v1",
        "issue_kinds": sorted(requested),
        "row_count": len(selected_templates),
        "output": str(Path(output_path).resolve()),
        "packet_output": str(Path(packet_output_path).resolve()),
    }


def merge_completed_adjudications(
    *,
    audit_ref: str | Path,
    adjudication_files: Sequence[str | Path],
    reviewer_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate and merge disjoint completed review streams into the exact audit frame."""

    if not adjudication_files:
        raise Stage1DataError("At least one adjudication input file is required")
    _, audit_target = _resolve_locator_ref(audit_ref, artifact_kind="data-audit")
    issues = _read_jsonl(audit_target / "issues.jsonl")
    rubric_meta = _load_json(audit_target / "adjudication_rubric.meta.json")
    rows: list[dict[str, Any]] = []
    for source in adjudication_files:
        rows.extend(_read_jsonl(source))
    completed = _validate_completed_rows(
        rows,
        issues,
        rubric_meta,
        reviewer_id=reviewer_id,
    )
    _write_jsonl(output_path, completed, key="issue_id")
    return {
        "schema_version": "stage1-data-adjudication-merge/v1",
        "row_count": len(completed),
        "completed_rows_sha256": hashlib.sha256(
            canonical_jsonl_bytes(completed, key="issue_id")
        ).hexdigest(),
        "output": str(Path(output_path).resolve()),
    }


def _rows_by_issue(
    rows: Sequence[Mapping[str, Any]], issues: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    issue_map = {str(issue["issue_id"]): dict(issue) for issue in issues}
    row_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        issue_id = row.get("issue_id")
        if not isinstance(issue_id, str):
            raise Stage1DataError("Every adjudication row must contain a string issue_id")
        if issue_id in row_map:
            raise Stage1DataError(f"Duplicate adjudication row for {issue_id}")
        row_map[issue_id] = dict(row)
    missing = sorted(set(issue_map) - set(row_map))
    extra = sorted(set(row_map) - set(issue_map))
    if missing or extra:
        raise Stage1DataError(
            f"Adjudication issue set mismatch: missing={missing}, extra={extra}"
        )
    return row_map, issue_map


def _validate_completed_rows(
    rows: Sequence[Mapping[str, Any]],
    issues: Sequence[Mapping[str, Any]],
    rubric_meta: Mapping[str, Any],
    *,
    reviewer_id: str | None = None,
) -> list[dict[str, Any]]:
    row_map, issue_map = _rows_by_issue(rows, issues)
    completed: list[dict[str, Any]] = []
    for issue_id in sorted(issue_map):
        issue = issue_map[issue_id]
        row = row_map[issue_id]
        required_keys = {
            "schema_version",
            "data_audit_id",
            "issue_id",
            "issue_kind",
            "decision",
            "edits",
            "reason_code",
            "reason",
            "reviewer_id",
            "reviewed_at",
        }
        if set(row) != required_keys:
            raise Stage1DataError(f"Adjudication row {issue_id} has unexpected keys")
        if row["schema_version"] != ADJUDICATION_ROW_SCHEMA_VERSION:
            raise Stage1DataError(f"Adjudication row {issue_id} has wrong schema_version")
        if row["data_audit_id"] != issue["data_audit_id"]:
            raise Stage1DataError(f"Adjudication row {issue_id} targets another audit")
        if row["issue_kind"] != issue["issue_kind"]:
            raise Stage1DataError(f"Adjudication row {issue_id} changed issue_kind")
        decision = row["decision"]
        edits = row["edits"]
        if decision not in {"accepted", "corrected"}:
            raise Stage1DataError(f"Adjudication row {issue_id} is incomplete")
        if not isinstance(edits, list):
            raise Stage1DataError(f"Adjudication row {issue_id}.edits must be an array")
        if decision == "accepted":
            if not issue["accept_allowed"] or edits:
                raise Stage1DataError(f"Issue {issue_id} cannot be accepted with these edits")
        elif not edits:
            raise Stage1DataError(f"Corrected issue {issue_id} requires at least one edit")
        canonical_edit_keys = []
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) != {
                "location_index",
                "op",
                "json_pointer",
                "value",
            }:
                raise Stage1DataError(f"Malformed edit in issue {issue_id}")
            location_index = edit["location_index"]
            pointer = edit["json_pointer"]
            if not isinstance(location_index, int) or isinstance(location_index, bool):
                raise Stage1DataError(f"Invalid edit location_index in issue {issue_id}")
            if location_index < 0 or location_index >= len(issue["locations"]):
                raise Stage1DataError(f"Edit location_index out of range in issue {issue_id}")
            if edit["op"] != "set" or pointer not in issue["allowed_edit_paths"]:
                raise Stage1DataError(f"Unauthorized edit path in issue {issue_id}: {pointer!r}")
            if issue["issue_kind"] == "field-type" and not (
                edit["value"] is None or isinstance(edit["value"], str)
            ):
                raise Stage1DataError(
                    f"Field-type correction for {issue_id} must be an explicit string or null"
                )
            canonical_edit_keys.append((location_index, pointer))
        if canonical_edit_keys != sorted(canonical_edit_keys) or len(set(canonical_edit_keys)) != len(
            canonical_edit_keys
        ):
            raise Stage1DataError(f"Edits for {issue_id} are not unique canonical order")
        reason_codes = rubric_meta["reason_codes"].get(issue["issue_code"], {}).get(
            decision, []
        )
        if row["reason_code"] not in reason_codes:
            raise Stage1DataError(
                f"Reason code {row['reason_code']!r} is invalid for {issue_id}/{decision}"
            )
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise Stage1DataError(f"Adjudication row {issue_id} requires a reason")
        if not isinstance(row["reviewer_id"], str) or not row["reviewer_id"].strip():
            raise Stage1DataError(f"Adjudication row {issue_id} requires reviewer_id")
        if reviewer_id is not None and row["reviewer_id"] != reviewer_id:
            raise Stage1DataError(f"Reviewer mismatch in adjudication row {issue_id}")
        if not isinstance(row["reviewed_at"], str) or not row["reviewed_at"].strip():
            raise Stage1DataError(f"Adjudication row {issue_id} requires reviewed_at")
        completed.append(dict(row))
    return completed


def prepare_data_declaration(
    *,
    audit_ref: str | Path,
    adjudication_file: str | Path,
    reviewer_id: str,
    write_template: str | Path,
) -> dict[str, Any]:
    """Prepare a hash-bound declaration that the reviewer must explicitly sign."""

    if not reviewer_id.strip():
        raise Stage1DataError("reviewer_id must be non-empty")
    _, audit_target = _resolve_locator_ref(audit_ref, artifact_kind="data-audit")
    audit_meta = _load_json(audit_target / "audit.meta.json")
    issues = _read_jsonl(audit_target / "issues.jsonl")
    rubric_meta = _load_json(audit_target / "adjudication_rubric.meta.json")
    rows = _read_jsonl(adjudication_file)
    completed = _validate_completed_rows(
        rows, issues, rubric_meta, reviewer_id=reviewer_id
    )
    completed_bytes = canonical_jsonl_bytes(completed, key="issue_id")
    declaration = {
        "schema_version": DECLARATION_SCHEMA_VERSION,
        "data_audit_id": audit_meta["data_audit_id"],
        "reviewer_id": reviewer_id,
        "rubric_body_sha256": rubric_meta["rubric_body_sha256"],
        "rubric_meta_sha256": canonical_json_sha256(rubric_meta),
        "ordered_issue_ids_sha256": audit_meta["ordered_issue_ids_sha256"],
        "completed_rows_sha256": hashlib.sha256(completed_bytes).hexdigest(),
        "saw_condition_outputs": False,
        "saw_model_scores": False,
        "attestation_confirmed": False,
    }
    _write_json(write_template, declaration)
    return declaration


def _load_audit_for_finalize(
    audit_ref: str | Path,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], dict[str, Any]]:
    locator, target = _resolve_locator_ref(audit_ref, artifact_kind="data-audit")
    issues = _read_jsonl(target / "issues.jsonl")
    rubric_meta = _load_json(target / "adjudication_rubric.meta.json")
    return locator, target, issues, rubric_meta


def _set_pointer(record: dict[str, Any], pointer: str, value: Any) -> Any:
    if pointer == "/id":
        old = record["id"]
        record["id"] = value
        return old
    match = re.fullmatch(r"/quadruples/(\d+)/(target|argument|targeted_group|hateful)", pointer)
    if not match:
        raise Stage1DataError(f"Unsupported adjudication JSON pointer: {pointer}")
    tuple_index = int(match.group(1))
    field = match.group(2)
    try:
        quad = record["quadruples"][tuple_index]
    except (KeyError, IndexError, TypeError) as exc:
        raise Stage1DataError(f"Adjudication pointer does not exist: {pointer}") from exc
    old = quad[field]
    quad[field] = value
    return old


def _apply_adjudication(
    records: dict[str, list[dict[str, Any]]],
    issues: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    issue_map = {issue["issue_id"]: issue for issue in issues}
    occupied: dict[tuple[str, int, str], Any] = {}
    log_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["issue_id"]):
        issue = issue_map[row["issue_id"]]
        applied_edits = []
        for edit in row["edits"]:
            location = issue["locations"][edit["location_index"]]
            coordinate = (
                location["source_key"],
                location["source_ordinal"],
                edit["json_pointer"],
            )
            if coordinate in occupied:
                raise Stage1DataError(
                    f"Multiple adjudication edits target the same source location: {coordinate}"
                )
            occupied[coordinate] = edit["value"]
            source_record = records[location["source_key"]][location["source_ordinal"]]
            old = _set_pointer(source_record, edit["json_pointer"], edit["value"])
            applied_edits.append(
                {
                    "location_index": edit["location_index"],
                    "json_pointer": edit["json_pointer"],
                    "old_value": old,
                    "new_value": edit["value"],
                }
            )
        log_rows.append(
            {
                "schema_version": "stage1-data-adjudication-log-row/v1",
                "issue_id": row["issue_id"],
                "issue_kind": issue["issue_kind"],
                "decision": row["decision"],
                "applied_edits": applied_edits,
                "reason_code": row["reason_code"],
                "reason": row["reason"],
                "reviewer_id": row["reviewer_id"],
                "reviewed_at": row["reviewed_at"],
            }
        )
    return log_rows


def _adapt_source_quad(raw: Mapping[str, Any]) -> Any:
    # Imported lazily so audit/export can run while WP2 lands in parallel.  The
    # interface itself is frozen by the phase-1 design contract.
    try:
        from utils.quadruple import adapt_source_quad
    except ImportError as exc:  # pragma: no cover - only possible during parallel landing
        raise Stage1DataError(
            "WP2 source adapter utils.quadruple.adapt_source_quad is required to finalize data"
        ) from exc
    return adapt_source_quad(raw)


def _quad_wire(quad: Any) -> dict[str, Any]:
    if isinstance(quad, Mapping):
        target = quad["target"]
        argument = quad["argument"]
        groups = quad["targeted_group"]
        hateful = quad["hateful"]
    else:
        target = quad.target
        argument = quad.argument
        groups = quad.targeted_group
        hateful = quad.hateful
    return {
        "target": target,
        "argument": argument,
        "targeted_group": list(groups),
        "hateful": hateful,
    }


def _normalize_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    normalized_sources: dict[str, list[dict[str, Any]]] = {}
    all_ids: list[str] = []
    for source_key in ("std-train", "std-test"):
        normalized = []
        for source_ordinal, record in enumerate(records[source_key]):
            record_id = canonical_record_id(record["id"])
            try:
                quads = [_quad_wire(_adapt_source_quad(quad)) for quad in record["quadruples"]]
            except Exception as exc:
                raise Stage1DataError(
                    f"Source adapter rejected {source_key}[{source_ordinal}] after adjudication: {exc}"
                ) from exc
            normalized.append(
                {"id": record_id, "content": record["content"], "quadruples": quads}
            )
            all_ids.append(record_id)
        normalized_sources[source_key] = normalized
    if len(all_ids) != len(set(all_ids)):
        raise Stage1DataError("Adjudicated record IDs are not globally unique")
    return normalized_sources


def _validate_normalized_quad(quad: Any, *, coordinate: str) -> None:
    if not isinstance(quad, dict) or list(quad) != [
        "target",
        "argument",
        "targeted_group",
        "hateful",
    ]:
        raise Stage1DataError(f"{coordinate} does not use canonical quad key order")
    for field in ("target", "argument"):
        value = quad[field]
        if value is not None and (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or value != unicodedata.normalize("NFC", value)
        ):
            raise Stage1DataError(f"{coordinate}.{field} is not canonical string|null")
    groups = quad["targeted_group"]
    if not isinstance(groups, list) or not groups or any(group not in GROUP_SET for group in groups):
        raise Stage1DataError(f"{coordinate}.targeted_group is invalid")
    if len(groups) != len(set(groups)):
        raise Stage1DataError(f"{coordinate}.targeted_group contains duplicates")
    expected_order = [group for group in GROUP_ORDER if group in groups]
    if groups != expected_order:
        raise Stage1DataError(f"{coordinate}.targeted_group is not canonical order")
    if "non-hate" in groups and len(groups) != 1:
        raise Stage1DataError(f"{coordinate}.targeted_group mixes non-hate")
    if quad["hateful"] not in {"hate", "non-hate"}:
        raise Stage1DataError(f"{coordinate}.hateful is invalid")


def _validate_normalized_records(splits: Mapping[str, Any]) -> None:
    expected_counts = {"train": 5781, "dev": 643, "test": 1605}
    all_ids: list[str] = []
    for split, count in expected_counts.items():
        records = splits.get(split)
        if not isinstance(records, list) or len(records) != count:
            raise Stage1DataError(f"Normalized {split} count must be {count}")
        for ordinal, record in enumerate(records):
            if not isinstance(record, dict) or list(record) != ["id", "content", "quadruples"]:
                raise Stage1DataError(f"Normalized {split}[{ordinal}] has invalid record keys/order")
            if not isinstance(record["id"], str) or not CANONICAL_ID_RE.fullmatch(record["id"]):
                raise Stage1DataError(f"Normalized {split}[{ordinal}] has invalid canonical ID")
            if not isinstance(record["content"], str) or not record["content"]:
                raise Stage1DataError(f"Normalized {split}[{ordinal}] has invalid content")
            if not isinstance(record["quadruples"], list) or not record["quadruples"]:
                raise Stage1DataError(f"Normalized {split}[{ordinal}] has no quadruples")
            for tuple_index, quad in enumerate(record["quadruples"]):
                _validate_normalized_quad(
                    quad, coordinate=f"{split}[{ordinal}].quadruples[{tuple_index}]"
                )
            all_ids.append(record["id"])
    if len(all_ids) != len(set(all_ids)):
        raise Stage1DataError("Normalized split IDs are not globally unique")


def _post_adjudication_group_hate_check(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    issues: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    decisions = {row["issue_id"]: row["decision"] for row in rows}
    issue_by_location: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for issue in issues:
        if issue["issue_kind"] != "group-hate":
            continue
        location = issue["locations"][0]
        issue_by_location[
            (location["source_key"], location["source_ordinal"], location["tuple_index"])
        ] = issue
    for source_key in ("std-train", "std-test"):
        for source_ordinal, record in enumerate(records[source_key]):
            for tuple_index, quad in enumerate(record["quadruples"]):
                spec = _group_hate_issue_spec(
                    _parse_group_atoms(quad["targeted_group"]), quad["hateful"]
                )
                coordinate = (source_key, source_ordinal, tuple_index)
                original = issue_by_location.get(coordinate)
                if spec is None:
                    continue
                if original is None or decisions[original["issue_id"]] != "accepted":
                    raise Stage1DataError(
                        f"Unresolved or newly introduced group-hate issue at {coordinate}"
                    )


def _validate_portable_dependency(
    value: Any,
    *,
    artifact_kind: str,
    artifact_id_pattern: str,
    label: str,
) -> dict[str, Any]:
    """Validate and normalize the five-field portable dependency wire."""

    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "artifact_kind",
            "artifact_id",
            "payload_manifest_sha256",
            "logical_repo_path",
        }
        or value.get("schema_version") != DEPENDENCY_REF_SCHEMA_VERSION
        or value.get("artifact_kind") != artifact_kind
        or not isinstance(value.get("artifact_id"), str)
        or not re.fullmatch(artifact_id_pattern, value["artifact_id"])
        or not isinstance(value.get("payload_manifest_sha256"), str)
        or not SHA256_RE.fullmatch(value["payload_manifest_sha256"])
    ):
        raise Stage1DataError(f"Embedded {label} ref is not a portable dependency ref")
    logical_path = value.get("logical_repo_path")
    if (
        not isinstance(logical_path, str)
        or not logical_path
        or Path(logical_path).is_absolute()
        or ".." in Path(logical_path).parts
        or Path(logical_path).as_posix() != logical_path
    ):
        raise Stage1DataError(f"Embedded {label} dependency path is not portable")
    return dict(value)


def _adjudication_composition(
    *,
    audit_dependency: Mapping[str, Any],
    blind_review_dependency: Mapping[str, Any],
    blind_review_target: Path,
    issues: Sequence[Mapping[str, Any]],
    rubric_meta: Mapping[str, Any],
    completed_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the frozen blind/human composition and return its queue/frame.

    The sealed blind-review artifact is authoritative for the complete
    ``group-hate`` scope.  Rows outside that scope are a separate, deterministic
    human-only queue derived from the immutable audit.  This function never
    manufactures a completed decision.
    """

    review_audit_dependency = _validate_portable_dependency(
        _load_json(blind_review_target / "data_audit_ref.json"),
        artifact_kind="data-audit",
        artifact_id_pattern=r"daudit-[0-9a-f]{64}",
        label="blind-review audit",
    )
    if review_audit_dependency != dict(audit_dependency):
        raise Stage1DataError("Data blind review targets a different data audit")

    issue_map = {str(issue["issue_id"]): dict(issue) for issue in issues}
    final_map = {str(row["issue_id"]): dict(row) for row in completed_rows}
    blind_issue_ids = {
        issue_id
        for issue_id, issue in issue_map.items()
        if issue.get("issue_kind") == BLIND_REVIEW_ISSUE_KIND
    }
    human_only_issue_ids = {
        issue_id
        for issue_id, issue in issue_map.items()
        if issue.get("issue_kind") in HUMAN_ONLY_ISSUE_KINDS
    }
    unsupported = sorted(
        issue_id
        for issue_id, issue in issue_map.items()
        if issue.get("issue_kind")
        not in ({BLIND_REVIEW_ISSUE_KIND} | HUMAN_ONLY_ISSUE_KINDS)
    )
    if unsupported:
        raise Stage1DataError(
            "Audit contains issue kinds with no frozen adjudication source: "
            f"{unsupported}"
        )

    run_meta = _load_json(blind_review_target / "run.meta.json")
    scope = run_meta.get("issue_scope") if isinstance(run_meta, dict) else None
    retry_scope = (
        isinstance(run_meta, dict)
        and isinstance(run_meta.get("parent_review_run_id"), str)
        and run_meta.get("retry_policy") == "failed-reviewer-items-only/v1"
        and isinstance(run_meta.get("retry_issue_ids_sha256_by_reviewer"), dict)
    )
    initial_scope = (
        isinstance(scope, dict)
        and scope.get("issue_kinds") == [BLIND_REVIEW_ISSUE_KIND]
        and scope.get("issue_count") == len(blind_issue_ids)
    )
    if not initial_scope and not retry_scope:
        raise Stage1DataError(
            "Data blind review must freeze the complete group-hate audit scope"
        )

    auto_rows = _read_jsonl(blind_review_target / "auto_adjudication.jsonl")
    queue_packets = _read_jsonl(blind_review_target / "human_queue.jsonl")
    templates = _read_jsonl(
        blind_review_target / "human_adjudication_template.jsonl"
    )
    auto_map = {str(row.get("issue_id")): dict(row) for row in auto_rows}
    queue_map = {str(row.get("issue_id")): dict(row) for row in queue_packets}
    template_map = {str(row.get("issue_id")): dict(row) for row in templates}
    if (
        len(auto_map) != len(auto_rows)
        or len(queue_map) != len(queue_packets)
        or len(template_map) != len(templates)
    ):
        raise Stage1DataError("Data blind review contains duplicate issue IDs")
    if set(auto_map) & set(queue_map):
        raise Stage1DataError("Blind auto and human queues overlap")
    if set(auto_map) | set(queue_map) != blind_issue_ids:
        raise Stage1DataError(
            "Blind auto/human rows do not exactly cover the group-hate audit scope"
        )
    if set(template_map) != set(queue_map):
        raise Stage1DataError("Blind human queue/template issue frames differ")

    for issue_id, auto_row in auto_map.items():
        if final_map.get(issue_id) != auto_row:
            raise Stage1DataError(
                f"Final adjudication changed or omitted blind auto row {issue_id}"
            )

    blind_human_rows = [final_map[issue_id] for issue_id in sorted(queue_map)]
    human_only_rows = [final_map[issue_id] for issue_id in sorted(human_only_issue_ids)]
    _validate_completed_rows(
        blind_human_rows,
        [issue_map[issue_id] for issue_id in sorted(queue_map)],
        rubric_meta,
        reviewer_id=BLIND_PANEL_REVIEWER_ID,
    )
    _validate_completed_rows(
        human_only_rows,
        [issue_map[issue_id] for issue_id in sorted(human_only_issue_ids)],
        rubric_meta,
        reviewer_id=BLIND_PANEL_REVIEWER_ID,
    )
    immutable_template_fields = (
        "schema_version",
        "data_audit_id",
        "issue_id",
        "issue_kind",
        "reviewer_id",
    )
    for issue_id, template in template_map.items():
        final_row = final_map[issue_id]
        if any(final_row.get(key) != template.get(key) for key in immutable_template_fields):
            raise Stage1DataError(
                f"Completed blind human row changed its frozen template: {issue_id}"
            )

    queue_rows = [
        {
            "schema_version": HUMAN_QUEUE_ROW_SCHEMA_VERSION,
            "data_audit_id": issue_map[issue_id]["data_audit_id"],
            "issue_id": issue_id,
            "issue_kind": issue_map[issue_id]["issue_kind"],
            "queue_source": "data-blind-review",
            "source_packet_sha256": canonical_json_sha256(queue_map[issue_id]),
        }
        for issue_id in sorted(queue_map)
    ] + [
        {
            "schema_version": HUMAN_QUEUE_ROW_SCHEMA_VERSION,
            "data_audit_id": issue_map[issue_id]["data_audit_id"],
            "issue_id": issue_id,
            "issue_kind": issue_map[issue_id]["issue_kind"],
            "queue_source": "audit-human-only",
            "source_packet_sha256": canonical_json_sha256(issue_map[issue_id]),
        }
        for issue_id in sorted(human_only_issue_ids)
    ]
    queue_rows.sort(key=lambda row: row["issue_id"])
    queue_sha256 = hashlib.sha256(
        canonical_jsonl_bytes(queue_rows, key="issue_id")
    ).hexdigest()
    final_rows_sha256 = hashlib.sha256(
        canonical_jsonl_bytes(completed_rows, key="issue_id")
    ).hexdigest()
    frame = {
        "schema_version": ADJUDICATION_FRAME_SCHEMA_VERSION,
        "data_audit_dependency": dict(audit_dependency),
        "data_blind_review_dependency": dict(blind_review_dependency),
        "blind_scope": {
            "issue_kinds": [BLIND_REVIEW_ISSUE_KIND],
            "issue_ids_sha256": canonical_json_sha256(sorted(blind_issue_ids)),
            "auto_issue_ids_sha256": canonical_json_sha256(sorted(auto_map)),
            "human_issue_ids_sha256": canonical_json_sha256(sorted(queue_map)),
        },
        "audit_human_only_scope": {
            "issue_kinds": sorted(HUMAN_ONLY_ISSUE_KINDS),
            "issue_ids_sha256": canonical_json_sha256(sorted(human_only_issue_ids)),
        },
        "blind_auto_adjudication_sha256": hashlib.sha256(
            canonical_jsonl_bytes(auto_rows, key="issue_id")
        ).hexdigest(),
        "blind_human_completed_sha256": hashlib.sha256(
            canonical_jsonl_bytes(blind_human_rows, key="issue_id")
        ).hexdigest(),
        "audit_human_only_completed_sha256": hashlib.sha256(
            canonical_jsonl_bytes(human_only_rows, key="issue_id")
        ).hexdigest(),
        "human_adjudication_queue_sha256": queue_sha256,
        "final_adjudication_rows_sha256": final_rows_sha256,
    }
    return queue_rows, frame


def _split_normalized_sources(
    normalized_sources: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    train_source = normalized_sources["std-train"]
    return {
        "train": [dict(record) for record in train_source[:5781]],
        "dev": [dict(record) for record in train_source[5781:]],
        "test": [dict(record) for record in normalized_sources["std-test"]],
    }


def finalize_data(
    *,
    config_path: str | Path,
    audit_ref: str | Path,
    data_blind_review_ref: str | Path,
    adjudication_file: str | Path,
    reviewer_declaration: str | Path,
    write_ref: str | Path,
) -> dict[str, Any]:
    """Apply a complete signed adjudication and create normalized split target."""

    config, workspace_root = _load_config(config_path)
    audit_locator, audit_target, issues, rubric_meta = _load_audit_for_finalize(audit_ref)
    frozen_config = _load_json(audit_target / "config.resolved.json")
    if canonical_json_bytes(config) != canonical_json_bytes(frozen_config):
        raise Stage1DataError("Finalize config differs from the immutable audit config")
    records, inventory = _load_sources(config, workspace_root, copy_records=True)
    frozen_inventory = _load_json(audit_target / "source_inventory.json")
    if inventory["document"] != frozen_inventory:
        raise Stage1DataError("Frozen std source inventory changed after audit")

    rows_raw = _read_jsonl(adjudication_file)
    declaration = _load_json(reviewer_declaration)
    if not isinstance(declaration, dict) or declaration.get("schema_version") != DECLARATION_SCHEMA_VERSION:
        raise Stage1DataError("Invalid data reviewer declaration")
    audit_meta = _load_json(audit_target / "audit.meta.json")
    if declaration.get("data_audit_id") != audit_meta["data_audit_id"]:
        raise Stage1DataError("Reviewer declaration targets another data audit")
    reviewer_id = declaration.get("reviewer_id")
    if not isinstance(reviewer_id, str) or not reviewer_id:
        raise Stage1DataError("Reviewer declaration lacks reviewer_id")
    completed_rows = _validate_completed_rows(
        rows_raw, issues, rubric_meta, reviewer_id=reviewer_id
    )
    completed_bytes = canonical_jsonl_bytes(completed_rows, key="issue_id")
    expected_declaration = {
        "rubric_body_sha256": rubric_meta["rubric_body_sha256"],
        "rubric_meta_sha256": canonical_json_sha256(rubric_meta),
        "ordered_issue_ids_sha256": audit_meta["ordered_issue_ids_sha256"],
        "completed_rows_sha256": hashlib.sha256(completed_bytes).hexdigest(),
    }
    for key, expected_value in expected_declaration.items():
        if declaration.get(key) != expected_value:
            raise Stage1DataError(f"Reviewer declaration {key} hash mismatch")
    if declaration.get("saw_condition_outputs") is not False:
        raise Stage1DataError("Reviewer declaration must assert saw_condition_outputs=false")
    if declaration.get("saw_model_scores") is not False:
        raise Stage1DataError("Reviewer declaration must assert saw_model_scores=false")
    if declaration.get("attestation_confirmed") is not True:
        raise Stage1DataError("Reviewer must set attestation_confirmed=true before finalize")

    audit_dependency = _portable_dependency(audit_locator, audit_target, workspace_root)
    try:
        # Lazy import avoids the intentional reverse import used by the sealed
        # review validator for adjudication-row validation.
        from review.data_review_artifact import resolve_data_review_ref

        blind_locator, blind_target, blind_report = resolve_data_review_ref(
            data_blind_review_ref, workspace_root=workspace_root
        )
    except Exception as exc:
        raise Stage1DataError(
            f"Data blind-review dependency failed deep validation: {exc}"
        ) from exc
    if blind_report.get("data_audit_id") != audit_meta["data_audit_id"]:
        raise Stage1DataError("Data blind review targets another data audit")
    blind_dependency = _portable_dependency(
        blind_locator, blind_target, workspace_root
    )
    human_queue_rows, adjudication_frame = _adjudication_composition(
        audit_dependency=audit_dependency,
        blind_review_dependency=blind_dependency,
        blind_review_target=blind_target,
        issues=issues,
        rubric_meta=rubric_meta,
        completed_rows=completed_rows,
    )

    adjudication_log = _apply_adjudication(records, issues, completed_rows)
    _post_adjudication_group_hate_check(records, issues, completed_rows)
    normalized_sources = _normalize_records(records)
    splits = _split_normalized_sources(normalized_sources)
    _validate_normalized_records(splits)
    split_manifest = _build_split_manifest(normalized_sources, inventory)
    if split_manifest["train_ids"] != [record["id"] for record in splits["train"]]:
        raise Stage1DataError("Final train split manifest mismatch")
    if split_manifest["dev_ids"] != [record["id"] for record in splits["dev"]]:
        raise Stage1DataError("Final dev split manifest mismatch")
    if split_manifest["test_ids"] != [record["id"] for record in splits["test"]]:
        raise Stage1DataError("Final test split manifest mismatch")

    completed_rows_sha256 = hashlib.sha256(completed_bytes).hexdigest()
    declaration_sha256 = canonical_json_sha256(declaration)
    queue_bytes = canonical_jsonl_bytes(human_queue_rows, key="issue_id")
    queue_sha256 = hashlib.sha256(queue_bytes).hexdigest()
    adjudication_frame_sha256 = canonical_json_sha256(adjudication_frame)
    adjudication_log_bytes = canonical_jsonl_bytes(adjudication_log, key="issue_id")
    adjudication_log_sha256 = hashlib.sha256(adjudication_log_bytes).hexdigest()
    split_payloads = {
        split: _ordered_json_bytes(splits[split]) + b"\n"
        for split in ("train", "dev", "test")
    }
    split_content_sha256 = {
        split: hashlib.sha256(payload).hexdigest()
        for split, payload in split_payloads.items()
    }
    split_manifest_sha256 = canonical_json_sha256(split_manifest)
    source_inventory_sha256 = canonical_json_sha256(inventory["document"])
    substring_warning_count = _load_json(audit_target / "audit_report.json")[
        "warnings"
    ]["count"]
    data_review_id = "dreview-" + canonical_json_sha256(
        {
            "data_audit_id": audit_meta["data_audit_id"],
            "data_blind_review_dependency": blind_dependency,
            "adjudication_rows_sha256": completed_rows_sha256,
            "human_adjudication_queue_sha256": queue_sha256,
            "adjudication_frame_sha256": adjudication_frame_sha256,
            "declaration_sha256": declaration_sha256,
        }
    )
    finalizer_code_sha256 = sha256_file(__file__)
    data_id_inputs = {
        "schema_version": DATA_SCHEMA_VERSION,
        "data_audit_dependency": audit_dependency,
        "data_blind_review_dependency": blind_dependency,
        "data_review_id": data_review_id,
        "adjudication_rows_sha256": completed_rows_sha256,
        "adjudication_log_sha256": adjudication_log_sha256,
        "human_adjudication_queue_sha256": queue_sha256,
        "adjudication_frame_sha256": adjudication_frame_sha256,
        "reviewer_declaration_sha256": declaration_sha256,
        "adjudication_input_sha256": sha256_file(adjudication_file),
        "declaration_input_sha256": sha256_file(reviewer_declaration),
        "split_manifest_sha256": split_manifest_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "substring_warning_count": substring_warning_count,
        "train_content_sha256": split_content_sha256["train"],
        "dev_content_sha256": split_content_sha256["dev"],
        "test_content_sha256": split_content_sha256["test"],
        "split_policy_version": SPLIT_POLICY_VERSION,
        "normalization_policy_version": NORMALIZATION_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "finalizer_code_sha256": finalizer_code_sha256,
    }
    data_build_id = "data-" + canonical_json_sha256(data_id_inputs)
    decision_counts = Counter(row["decision"] for row in completed_rows)
    final_report = {
        "schema_version": "stage1-data-finalization-report/v1",
        "data_build_id": data_build_id,
        "data_review_id": data_review_id,
        "blocking_issue_count": len(issues),
        "resolved_issue_count": len(completed_rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "accepted_group_hate_warning_count": sum(
            issue["issue_kind"] == "group-hate"
            and decisions_row["decision"] == "accepted"
            for issue, decisions_row in (
                (issue, next(row for row in completed_rows if row["issue_id"] == issue["issue_id"]))
                for issue in issues
            )
        ),
        "substring_warning_count": substring_warning_count,
        "post_finalize_schema_valid": True,
    }
    provenance = {
        "schema_version": "stage1-data-provenance/v1",
        "data_build_id": data_build_id,
        "data_id_inputs": data_id_inputs,
        "source_inventory_sha256": source_inventory_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "finalizer_code_sha256": finalizer_code_sha256,
    }

    target_parent = _artifact_root(config, workspace_root) / "data"
    target = target_parent / data_build_id
    temporary = _new_target_temp(target_parent, data_build_id)
    try:
        _write_json(temporary / "audit_ref.json", audit_dependency)
        _write_json(temporary / "data_blind_review_ref.json", blind_dependency)
        for split, payload in split_payloads.items():
            _atomic_write_bytes(temporary / f"{split}.json", payload)
        _write_json(temporary / "split_manifest.json", split_manifest)
        _atomic_write_bytes(
            temporary / "adjudication_rows.jsonl", completed_bytes
        )
        _atomic_write_bytes(
            temporary / "adjudication_log.jsonl", adjudication_log_bytes
        )
        _atomic_write_bytes(
            temporary / "human_adjudication_queue.jsonl", queue_bytes
        )
        _write_json(temporary / "adjudication_frame.json", adjudication_frame)
        _write_json(temporary / "reviewer_declaration.json", declaration)
        _write_json(temporary / "audit_report.json", final_report)
        _write_json(temporary / "provenance.json", provenance)
        payload_hash = _finalize_target(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return _write_locator_ref(
        write_ref,
        artifact_kind="data",
        artifact_id=data_build_id,
        target_directory=target,
        payload_manifest_sha256=payload_hash,
    )


def validate_data_target(
    target_dir: str | Path,
    *,
    require_directory_name: bool = True,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Deep, read-only validation of a finalized Stage 1 data target.

    This target-level entry point is intentionally independent of a locator so
    downstream artifacts can validate portable dependency refs after resolving
    them inside a workspace.  Locator validation remains the responsibility of
    :func:`validate_data`.
    """

    target = Path(target_dir)
    if not target.is_dir() or target.is_symlink():
        raise Stage1DataError(f"Final data target is not a real directory: {target}")
    payload_manifest_sha256 = _verify_payload_manifest(target)
    required = {
        "audit_ref.json",
        "data_blind_review_ref.json",
        "train.json",
        "dev.json",
        "test.json",
        "split_manifest.json",
        "adjudication_rows.jsonl",
        "adjudication_log.jsonl",
        "human_adjudication_queue.jsonl",
        "adjudication_frame.json",
        "reviewer_declaration.json",
        "audit_report.json",
        "provenance.json",
        "payload_manifest.json",
    }
    for path in target.rglob("*"):
        if path.is_symlink():
            raise Stage1DataError("Final data target cannot contain symlinks")
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    }
    if actual != required:
        raise Stage1DataError(
            f"Final data target file set mismatch: missing={sorted(required-actual)}, "
            f"extra={sorted(actual-required)}"
        )
    splits = {
        split: _load_json(target / f"{split}.json")
        for split in ("train", "dev", "test")
    }
    _validate_normalized_records(splits)
    split_content_sha256: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        expected_payload = _ordered_json_bytes(splits[split]) + b"\n"
        actual_payload = (target / f"{split}.json").read_bytes()
        if actual_payload != expected_payload:
            raise Stage1DataError(f"Final {split}.json is not canonical ordered JSON")
        split_content_sha256[split] = hashlib.sha256(actual_payload).hexdigest()
    split_manifest = _load_json(target / "split_manifest.json")
    if (target / "split_manifest.json").read_bytes() != canonical_json_bytes(
        split_manifest
    ) + b"\n":
        raise Stage1DataError("Final split manifest is not canonical JSON")
    if split_manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise Stage1DataError("Final split manifest has wrong schema_version")
    if (
        split_manifest.get("policy") != SPLIT_POLICY_VERSION
        or split_manifest.get("id_policy") != ID_POLICY_VERSION
    ):
        raise Stage1DataError("Final split manifest policy lineage is invalid")
    for split in ("train", "dev", "test"):
        ids = [record["id"] for record in splits[split]]
        if split_manifest.get(f"{split}_ids") != ids:
            raise Stage1DataError(f"Final split manifest {split}_ids mismatch")
        if split_manifest.get(f"{split}_ids_sha256") != _id_list_sha256(ids):
            raise Stage1DataError(f"Final split manifest {split}_ids_sha256 mismatch")
    declaration = _load_json(target / "reviewer_declaration.json")
    if (target / "reviewer_declaration.json").read_bytes() != canonical_json_bytes(
        declaration
    ) + b"\n":
        raise Stage1DataError("Stored reviewer declaration is not canonical JSON")
    if (
        declaration.get("schema_version") != DECLARATION_SCHEMA_VERSION
        or declaration.get("attestation_confirmed") is not True
        or declaration.get("saw_condition_outputs") is not False
        or declaration.get("saw_model_scores") is not False
    ):
        raise Stage1DataError("Stored reviewer declaration is not signed")
    completed_rows = _read_jsonl(target / "adjudication_rows.jsonl")
    completed_bytes = canonical_jsonl_bytes(completed_rows, key="issue_id")
    if (target / "adjudication_rows.jsonl").read_bytes() != completed_bytes:
        raise Stage1DataError("Final adjudication rows are not canonical JSONL")
    completed_rows_sha256 = hashlib.sha256(completed_bytes).hexdigest()
    if declaration.get("completed_rows_sha256") != completed_rows_sha256:
        raise Stage1DataError("Reviewer declaration does not bind stored adjudication rows")
    reviewer_id = declaration.get("reviewer_id")
    if reviewer_id != BLIND_PANEL_REVIEWER_ID or any(
        row.get("reviewer_id") != reviewer_id for row in completed_rows
    ):
        raise Stage1DataError("Final adjudication rows do not use the frozen blind panel ID")

    adjudication_log = _read_jsonl(target / "adjudication_log.jsonl")
    adjudication_log_bytes = canonical_jsonl_bytes(adjudication_log, key="issue_id")
    if (target / "adjudication_log.jsonl").read_bytes() != adjudication_log_bytes:
        raise Stage1DataError("Final adjudication log is not canonical JSONL")
    if [row.get("issue_id") for row in adjudication_log] != [
        row.get("issue_id") for row in completed_rows
    ]:
        raise Stage1DataError("Adjudication row/log issue frames differ")
    completed_map = {str(row["issue_id"]): row for row in completed_rows}
    for log_row in adjudication_log:
        row = completed_map[str(log_row["issue_id"])]
        for field in (
            "issue_kind",
            "decision",
            "reason_code",
            "reason",
            "reviewer_id",
            "reviewed_at",
        ):
            if log_row.get(field) != row.get(field):
                raise Stage1DataError(
                    f"Adjudication log changed {field} for {row['issue_id']}"
                )
        if not isinstance(log_row.get("applied_edits"), list):
            raise Stage1DataError(
                f"Adjudication log has malformed edits for {row['issue_id']}"
            )

    human_queue = _read_jsonl(target / "human_adjudication_queue.jsonl")
    human_queue_bytes = canonical_jsonl_bytes(human_queue, key="issue_id")
    if (target / "human_adjudication_queue.jsonl").read_bytes() != human_queue_bytes:
        raise Stage1DataError("Human adjudication queue is not canonical JSONL")
    queue_map: dict[str, dict[str, Any]] = {}
    for queue_row in human_queue:
        issue_id = queue_row.get("issue_id")
        if (
            set(queue_row)
            != {
                "schema_version",
                "data_audit_id",
                "issue_id",
                "issue_kind",
                "queue_source",
                "source_packet_sha256",
            }
            or queue_row.get("schema_version") != HUMAN_QUEUE_ROW_SCHEMA_VERSION
            or not isinstance(issue_id, str)
            or issue_id not in completed_map
            or queue_row.get("data_audit_id")
            != completed_map[issue_id].get("data_audit_id")
            or queue_row.get("issue_kind")
            != completed_map[issue_id].get("issue_kind")
            or queue_row.get("queue_source")
            not in {"data-blind-review", "audit-human-only"}
            or not isinstance(queue_row.get("source_packet_sha256"), str)
            or not SHA256_RE.fullmatch(queue_row["source_packet_sha256"])
        ):
            raise Stage1DataError("Human adjudication queue contains a malformed row")
        queue_map[issue_id] = queue_row

    provenance = _load_json(target / "provenance.json")
    if (target / "provenance.json").read_bytes() != canonical_json_bytes(
        provenance
    ) + b"\n":
        raise Stage1DataError("Final data provenance is not canonical JSON")
    if set(provenance) != {
        "schema_version",
        "data_build_id",
        "data_id_inputs",
        "source_inventory_sha256",
        "split_manifest_sha256",
        "finalizer_code_sha256",
    } or provenance.get("schema_version") != "stage1-data-provenance/v1":
        raise Stage1DataError("Final data provenance has wrong schema_version")
    data_id_inputs = provenance.get("data_id_inputs")
    if (
        not isinstance(data_id_inputs, dict)
        or set(data_id_inputs)
        != {
            "schema_version",
            "data_audit_dependency",
            "data_blind_review_dependency",
            "data_review_id",
            "adjudication_rows_sha256",
            "adjudication_log_sha256",
            "human_adjudication_queue_sha256",
            "adjudication_frame_sha256",
            "reviewer_declaration_sha256",
            "adjudication_input_sha256",
            "declaration_input_sha256",
            "split_manifest_sha256",
            "source_inventory_sha256",
            "substring_warning_count",
            "train_content_sha256",
            "dev_content_sha256",
            "test_content_sha256",
            "split_policy_version",
            "normalization_policy_version",
            "output_schema_version",
            "finalizer_code_sha256",
        }
        or data_id_inputs.get("schema_version") != DATA_SCHEMA_VERSION
        or data_id_inputs.get("split_policy_version") != SPLIT_POLICY_VERSION
        or data_id_inputs.get("normalization_policy_version")
        != NORMALIZATION_SCHEMA_VERSION
        or data_id_inputs.get("output_schema_version") != OUTPUT_SCHEMA_VERSION
        or not isinstance(data_id_inputs.get("source_inventory_sha256"), str)
        or not SHA256_RE.fullmatch(data_id_inputs["source_inventory_sha256"])
        or isinstance(data_id_inputs.get("substring_warning_count"), bool)
        or not isinstance(data_id_inputs.get("substring_warning_count"), int)
        or data_id_inputs["substring_warning_count"] < 0
    ):
        raise Stage1DataError("Data provenance lacks data_id_inputs")
    recomputed_id = "data-" + canonical_json_sha256(data_id_inputs)
    if provenance.get("data_build_id") != recomputed_id:
        raise Stage1DataError("Data build ID is not reproducible from provenance")
    if require_directory_name and target.name != recomputed_id:
        raise Stage1DataError("Data target directory name does not match build ID")
    dependency = _validate_portable_dependency(
        _load_json(target / "audit_ref.json"),
        artifact_kind="data-audit",
        artifact_id_pattern=r"daudit-[0-9a-f]{64}",
        label="audit",
    )
    blind_dependency = _validate_portable_dependency(
        _load_json(target / "data_blind_review_ref.json"),
        artifact_kind="data-blind-review",
        artifact_id_pattern=r"dreview-[0-9a-f]{64}",
        label="data blind-review",
    )
    if (target / "audit_ref.json").read_bytes() != canonical_json_bytes(
        dependency
    ) + b"\n":
        raise Stage1DataError("Embedded audit dependency is not canonical JSON")
    if (
        (target / "data_blind_review_ref.json").read_bytes()
        != canonical_json_bytes(blind_dependency) + b"\n"
    ):
        raise Stage1DataError(
            "Embedded data blind-review dependency is not canonical JSON"
        )
    if data_id_inputs.get("data_audit_dependency") != dependency:
        raise Stage1DataError("Data ID inputs disagree with embedded audit dependency")
    if data_id_inputs.get("data_blind_review_dependency") != blind_dependency:
        raise Stage1DataError(
            "Data ID inputs disagree with embedded data blind-review dependency"
        )
    if data_id_inputs.get("reviewer_declaration_sha256") != canonical_json_sha256(
        declaration
    ):
        raise Stage1DataError("Data ID inputs disagree with reviewer declaration")
    if provenance.get("split_manifest_sha256") != canonical_json_sha256(
        split_manifest
    ) or data_id_inputs.get("split_manifest_sha256") != canonical_json_sha256(
        split_manifest
    ):
        raise Stage1DataError("Data provenance split manifest hash mismatch")
    if provenance.get("source_inventory_sha256") != data_id_inputs.get(
        "source_inventory_sha256"
    ):
        raise Stage1DataError("Data source-inventory lineage is inconsistent")
    for split in ("train", "dev", "test"):
        if data_id_inputs.get(f"{split}_content_sha256") != split_content_sha256[split]:
            raise Stage1DataError(
                f"Data ID inputs disagree with final {split}.json content"
            )
    if data_id_inputs.get("adjudication_rows_sha256") != completed_rows_sha256:
        raise Stage1DataError("Data ID inputs disagree with final adjudication rows")
    adjudication_log_sha256 = hashlib.sha256(adjudication_log_bytes).hexdigest()
    if data_id_inputs.get("adjudication_log_sha256") != adjudication_log_sha256:
        raise Stage1DataError("Data ID inputs disagree with final adjudication log")
    queue_sha256 = hashlib.sha256(human_queue_bytes).hexdigest()
    if data_id_inputs.get("human_adjudication_queue_sha256") != queue_sha256:
        raise Stage1DataError("Data ID inputs disagree with frozen human queue")

    frame = _load_json(target / "adjudication_frame.json")
    if (target / "adjudication_frame.json").read_bytes() != canonical_json_bytes(
        frame
    ) + b"\n":
        raise Stage1DataError("Stored adjudication composition is not canonical JSON")
    expected_frame_keys = {
        "schema_version",
        "data_audit_dependency",
        "data_blind_review_dependency",
        "blind_scope",
        "audit_human_only_scope",
        "blind_auto_adjudication_sha256",
        "blind_human_completed_sha256",
        "audit_human_only_completed_sha256",
        "human_adjudication_queue_sha256",
        "final_adjudication_rows_sha256",
    }
    if (
        not isinstance(frame, dict)
        or set(frame) != expected_frame_keys
        or frame.get("schema_version") != ADJUDICATION_FRAME_SCHEMA_VERSION
        or frame.get("data_audit_dependency") != dependency
        or frame.get("data_blind_review_dependency") != blind_dependency
        or frame.get("human_adjudication_queue_sha256") != queue_sha256
        or frame.get("final_adjudication_rows_sha256") != completed_rows_sha256
    ):
        raise Stage1DataError("Stored adjudication composition frame is malformed")
    blind_rows = [
        row
        for row in completed_rows
        if row.get("issue_kind") == BLIND_REVIEW_ISSUE_KIND
    ]
    human_only_rows = [
        row for row in completed_rows if row.get("issue_kind") in HUMAN_ONLY_ISSUE_KINDS
    ]
    if len(blind_rows) + len(human_only_rows) != len(completed_rows):
        raise Stage1DataError("Final adjudication contains an unsupported issue kind")
    blind_queue_ids = sorted(
        issue_id
        for issue_id, row in queue_map.items()
        if row["queue_source"] == "data-blind-review"
    )
    human_only_queue_ids = sorted(
        issue_id
        for issue_id, row in queue_map.items()
        if row["queue_source"] == "audit-human-only"
    )
    blind_ids = sorted(str(row["issue_id"]) for row in blind_rows)
    human_only_ids = sorted(str(row["issue_id"]) for row in human_only_rows)
    auto_ids = sorted(set(blind_ids) - set(blind_queue_ids))
    if (
        set(blind_queue_ids) - set(blind_ids)
        or human_only_queue_ids != human_only_ids
        or any(queue_map[value]["issue_kind"] != BLIND_REVIEW_ISSUE_KIND for value in blind_queue_ids)
        or any(queue_map[value]["issue_kind"] not in HUMAN_ONLY_ISSUE_KINDS for value in human_only_queue_ids)
    ):
        raise Stage1DataError("Human queue does not match blind/human-only scopes")
    expected_blind_scope = {
        "issue_kinds": [BLIND_REVIEW_ISSUE_KIND],
        "issue_ids_sha256": canonical_json_sha256(blind_ids),
        "auto_issue_ids_sha256": canonical_json_sha256(auto_ids),
        "human_issue_ids_sha256": canonical_json_sha256(blind_queue_ids),
    }
    expected_human_only_scope = {
        "issue_kinds": sorted(HUMAN_ONLY_ISSUE_KINDS),
        "issue_ids_sha256": canonical_json_sha256(human_only_ids),
    }
    auto_rows = [completed_map[issue_id] for issue_id in auto_ids]
    blind_human_rows = [completed_map[issue_id] for issue_id in blind_queue_ids]
    if (
        frame.get("blind_scope") != expected_blind_scope
        or frame.get("audit_human_only_scope") != expected_human_only_scope
        or frame.get("blind_auto_adjudication_sha256")
        != hashlib.sha256(canonical_jsonl_bytes(auto_rows, key="issue_id")).hexdigest()
        or frame.get("blind_human_completed_sha256")
        != hashlib.sha256(
            canonical_jsonl_bytes(blind_human_rows, key="issue_id")
        ).hexdigest()
        or frame.get("audit_human_only_completed_sha256")
        != hashlib.sha256(
            canonical_jsonl_bytes(human_only_rows, key="issue_id")
        ).hexdigest()
    ):
        raise Stage1DataError("Stored adjudication composition cannot be reproduced")
    frame_sha256 = canonical_json_sha256(frame)
    if data_id_inputs.get("adjudication_frame_sha256") != frame_sha256:
        raise Stage1DataError("Data ID inputs disagree with adjudication composition")
    expected_data_review_id = "dreview-" + canonical_json_sha256(
        {
            "data_audit_id": dependency["artifact_id"],
            "data_blind_review_dependency": blind_dependency,
            "adjudication_rows_sha256": completed_rows_sha256,
            "human_adjudication_queue_sha256": queue_sha256,
            "adjudication_frame_sha256": frame_sha256,
            "declaration_sha256": canonical_json_sha256(declaration),
        }
    )
    if data_id_inputs.get("data_review_id") != expected_data_review_id:
        raise Stage1DataError("Final data review ID cannot be reproduced")
    if provenance.get("finalizer_code_sha256") != data_id_inputs.get(
        "finalizer_code_sha256"
    ):
        raise Stage1DataError("Data finalizer code lineage is inconsistent")
    final_report = _load_json(target / "audit_report.json")
    if (target / "audit_report.json").read_bytes() != canonical_json_bytes(
        final_report
    ) + b"\n":
        raise Stage1DataError("Final data report is not canonical JSON")
    expected_report_keys = {
        "schema_version",
        "data_build_id",
        "data_review_id",
        "blocking_issue_count",
        "resolved_issue_count",
        "decision_counts",
        "accepted_group_hate_warning_count",
        "substring_warning_count",
        "post_finalize_schema_valid",
    }
    if (
        set(final_report) != expected_report_keys
        or
        final_report.get("schema_version")
        != "stage1-data-finalization-report/v1"
        or final_report.get("data_build_id") != recomputed_id
        or final_report.get("data_review_id") != data_id_inputs.get("data_review_id")
        or final_report.get("blocking_issue_count") != len(completed_rows)
        or final_report.get("resolved_issue_count") != len(completed_rows)
        or final_report.get("decision_counts")
        != dict(sorted(Counter(row["decision"] for row in completed_rows).items()))
        or final_report.get("accepted_group_hate_warning_count")
        != sum(
            row.get("issue_kind") == BLIND_REVIEW_ISSUE_KIND
            and row.get("decision") == "accepted"
            for row in completed_rows
        )
        or final_report.get("substring_warning_count")
        != data_id_inputs.get("substring_warning_count")
        or final_report.get("post_finalize_schema_valid") is not True
    ):
        raise Stage1DataError("Final data report is inconsistent with provenance")
    manifest_document = _load_json(target / "payload_manifest.json")
    if (target / "payload_manifest.json").read_bytes() != canonical_json_bytes(
        manifest_document
    ) + b"\n":
        raise Stage1DataError("Final payload manifest is not canonical JSON")
    upstream_replay = False
    source_replay = False
    candidate_root: Path | None = Path(workspace_root).resolve() if workspace_root else None
    if candidate_root is None:
        for candidate in target.resolve().parents:
            try:
                audit_candidate = (candidate / dependency["logical_repo_path"]).resolve()
                blind_candidate = (
                    candidate / blind_dependency["logical_repo_path"]
                ).resolve()
                audit_candidate.relative_to(candidate)
                blind_candidate.relative_to(candidate)
            except ValueError:
                continue
            if (
                audit_candidate.is_dir()
                and blind_candidate.is_dir()
                and (candidate / "config/stage1/blind_review.json").is_file()
            ):
                candidate_root = candidate
                break
    if candidate_root is not None:
        try:
            audit_target = (candidate_root / dependency["logical_repo_path"]).resolve()
            blind_target = (
                candidate_root / blind_dependency["logical_repo_path"]
            ).resolve()
            audit_target.relative_to(candidate_root)
            blind_target.relative_to(candidate_root)
            if (
                audit_target.name != dependency["artifact_id"]
                or _verify_payload_manifest(audit_target)
                != dependency["payload_manifest_sha256"]
                or blind_target.name != blind_dependency["artifact_id"]
            ):
                raise Stage1DataError("Embedded upstream dependency identity mismatch")
            from review.data_review_artifact import validate_data_review_target

            blind_report = validate_data_review_target(
                blind_target,
                workspace_root=candidate_root,
                audit_target=audit_target,
            )
            if (
                blind_report.get("data_blind_review_id")
                != blind_dependency["artifact_id"]
                or blind_report.get("payload_manifest_sha256")
                != blind_dependency["payload_manifest_sha256"]
            ):
                raise Stage1DataError("Embedded blind-review dependency failed identity replay")
            issues = _read_jsonl(audit_target / "issues.jsonl")
            rubric_meta = _load_json(audit_target / "adjudication_rubric.meta.json")
            validated_rows = _validate_completed_rows(
                completed_rows,
                issues,
                rubric_meta,
                reviewer_id=BLIND_PANEL_REVIEWER_ID,
            )
            replay_queue, replay_frame = _adjudication_composition(
                audit_dependency=dependency,
                blind_review_dependency=blind_dependency,
                blind_review_target=blind_target,
                issues=issues,
                rubric_meta=rubric_meta,
                completed_rows=validated_rows,
            )
            if replay_queue != human_queue or replay_frame != frame:
                raise Stage1DataError("Audit/blind-review adjudication replay differs from target")
            frozen_inventory = _load_json(audit_target / "source_inventory.json")
            if canonical_json_sha256(frozen_inventory) != data_id_inputs.get(
                "source_inventory_sha256"
            ):
                raise Stage1DataError(
                    "Data ID inputs disagree with frozen audit source inventory"
                )
            frozen_audit_report = _load_json(audit_target / "audit_report.json")
            if (
                frozen_audit_report.get("warnings", {}).get("count")
                != data_id_inputs.get("substring_warning_count")
            ):
                raise Stage1DataError(
                    "Data ID inputs disagree with frozen audit warning count"
                )
            upstream_replay = True

            frozen_config = _load_json(audit_target / "config.resolved.json")
            source_paths = [
                candidate_root / relative for relative in EXPECTED_SOURCE_PATHS.values()
            ]
            if all(path.is_file() for path in source_paths):
                source_records, source_inventory = _load_sources(
                    frozen_config, candidate_root, copy_records=True
                )
                if source_inventory["document"] != frozen_inventory:
                    raise Stage1DataError("Source inventory differs from immutable audit")
                if canonical_json_sha256(
                    source_inventory["document"]
                ) != data_id_inputs.get("source_inventory_sha256"):
                    raise Stage1DataError(
                        "Data ID inputs disagree with replayed source inventory"
                    )
                replay_log = _apply_adjudication(
                    source_records, issues, validated_rows
                )
                _post_adjudication_group_hate_check(
                    source_records, issues, validated_rows
                )
                replay_sources = _normalize_records(source_records)
                replay_splits = _split_normalized_sources(replay_sources)
                replay_manifest = _build_split_manifest(
                    replay_sources, source_inventory
                )
                if (
                    replay_splits != splits
                    or replay_manifest != split_manifest
                    or replay_log != adjudication_log
                ):
                    raise Stage1DataError(
                        "Source plus adjudication replay differs from finalized outputs"
                    )
                source_replay = True
        except Stage1DataError:
            raise
        except Exception as exc:
            raise Stage1DataError(f"Cannot replay finalized data lineage: {exc}") from exc

    return {
        "schema_version": "stage1-data-validation-report/v1",
        "valid": True,
        "data_build_id": recomputed_id,
        "payload_manifest_sha256": payload_manifest_sha256,
        "split_counts": {split: len(records) for split, records in splits.items()},
        "split_id_hashes": {
            split: split_manifest[f"{split}_ids_sha256"]
            for split in ("train", "dev", "test")
        },
        "upstream_adjudication_replayed": upstream_replay,
        "source_adjudication_replayed": source_replay,
    }


def validate_data(
    *, data_ref: str | Path, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    """Read-only validation of a finalized Stage 1 data locator and target."""

    locator, target = _resolve_locator_ref(data_ref, artifact_kind="data")
    report = validate_data_target(target, workspace_root=workspace_root)
    if report["data_build_id"] != locator["artifact_id"]:
        raise Stage1DataError("Data locator ID does not match validated target")
    if report["payload_manifest_sha256"] != locator["payload_manifest_sha256"]:
        raise Stage1DataError("Data locator payload hash does not match validated target")
    return report


__all__ = [
    "Stage1DataError",
    "audit_data",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_record_id",
    "export_data_adjudication",
    "export_data_review_subset",
    "finalize_data",
    "merge_completed_adjudications",
    "prepare_data_declaration",
    "validate_data",
    "validate_data_target",
]
