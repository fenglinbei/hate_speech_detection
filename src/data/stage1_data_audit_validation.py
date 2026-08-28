"""Independent deep replay validator for immutable Stage 1 data audits.

This module deliberately lives outside :mod:`data.stage1_data`: the audit ID
binds the complete builder module bytes, so adding validation-only code to that
module would invalidate an already published audit and its completed blind
review.  Validation nevertheless reuses the builder's deterministic private
primitives and reconstructs every payload document from the frozen std inputs.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from data import stage1_data as lifecycle


class Stage1DataAuditValidationError(ValueError):
    """Raised when an audit cannot be reproduced from its source inputs."""


EXPECTED_FILES = frozenset(
    {
        "adjudication_rubric.md",
        "adjudication_rubric.meta.json",
        "adjudication_template.jsonl",
        "audit.meta.json",
        "audit_report.json",
        "config.resolved.json",
        "issues.jsonl",
        "payload_manifest.json",
        "provenance.json",
        "source_inventory.json",
        "split_manifest.proposed.json",
    }
)


def _fail(message: str) -> None:
    raise Stage1DataAuditValidationError(message)


def validate_data_audit_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Deep-replay a data audit from the selected source files and builder.

    Unlike a structural hash-chain check, this proves that source inventory,
    proposed split, every issue/warning, empty adjudication frame, rubric and
    provenance are exactly what the current frozen audit implementation emits.
    """

    target = Path(target_dir).resolve()
    root = Path(workspace_root).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise Stage1DataAuditValidationError(
            "data audit target is outside workspace_root"
        ) from exc
    if target.is_symlink() or not target.is_dir():
        _fail("data audit target must be a real directory")
    actual_files = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    }
    if actual_files != EXPECTED_FILES:
        _fail("data audit exact file set mismatch")
    if any(path.is_symlink() for path in target.rglob("*")):
        _fail("data audit payload cannot contain symlinks")

    try:
        payload_hash = lifecycle._verify_payload_manifest(target)
        config, discovered_root = lifecycle._load_config(
            target / "config.resolved.json"
        )
    except lifecycle.Stage1DataError as exc:
        raise Stage1DataAuditValidationError(str(exc)) from exc
    if discovered_root.resolve() != root:
        _fail("data audit resolved a different workspace root")
    canonical_config_path = root / "config/stage1/context_factorial.json"
    if not canonical_config_path.is_file():
        _fail("canonical Stage 1 context config is missing")
    canonical_config = lifecycle._load_json(canonical_config_path)
    if (
        not isinstance(canonical_config, dict)
        or config.get("schema_version") != canonical_config.get("schema_version")
        or config.get("artifact_root") != canonical_config.get("artifact_root")
        or config.get("data") != canonical_config.get("data")
    ):
        _fail("data audit data policy differs from the selected canonical config")

    source_paths = lifecycle._source_paths(config, root)
    source_guard_before = {
        key: lifecycle.sha256_file(path) for key, path in source_paths.items()
    }
    try:
        records, inventory = lifecycle._load_sources(config, root)
        split_manifest = lifecycle._build_split_manifest(
            records, inventory, require_unique=False
        )
    except lifecycle.Stage1DataError as exc:
        raise Stage1DataAuditValidationError(str(exc)) from exc

    canonical_rubric_path = root / "config/stage1/data_adjudication_rubric.md"
    if not canonical_rubric_path.is_file():
        _fail("canonical adjudication rubric is missing")
    rubric_bytes = canonical_rubric_path.read_bytes().replace(b"\r\n", b"\n")
    try:
        rubric_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Stage1DataAuditValidationError(
            "canonical adjudication rubric is not UTF-8"
        ) from exc
    rubric_body_sha256 = hashlib.sha256(rubric_bytes).hexdigest()
    rubric_meta = lifecycle._rubric_meta(rubric_body_sha256)
    rubric_meta_sha256 = lifecycle.canonical_json_sha256(rubric_meta)
    stored_meta = lifecycle._load_json(target / "audit.meta.json")
    stored_id_inputs = (
        stored_meta.get("audit_id_inputs")
        if isinstance(stored_meta, dict)
        else None
    )
    audit_code_sha256 = (
        stored_id_inputs.get("audit_code_sha256")
        if isinstance(stored_id_inputs, dict)
        else None
    )
    if (
        not isinstance(audit_code_sha256, str)
        or not lifecycle.SHA256_RE.fullmatch(audit_code_sha256)
    ):
        _fail("data audit does not bind a valid builder code hash")
    audit_id_inputs = {
        "schema_version": lifecycle.AUDIT_SCHEMA_VERSION,
        "source_inventory_sha256": lifecycle.canonical_json_sha256(
            inventory["document"]
        ),
        "split_policy_sha256": lifecycle.canonical_json_sha256(
            config["data"]["split"]
        ),
        "audit_config_sha256": lifecycle.canonical_json_sha256(config),
        "audit_rule_version": lifecycle.AUDIT_RULE_VERSION,
        "normalization_schema_version": lifecycle.NORMALIZATION_SCHEMA_VERSION,
        "rubric_body_sha256": rubric_body_sha256,
        "rubric_meta_sha256": rubric_meta_sha256,
        "audit_code_sha256": audit_code_sha256,
    }
    audit_id = "daudit-" + lifecycle.canonical_json_sha256(audit_id_inputs)
    if target.name != audit_id:
        _fail("data audit directory/ID differs from exact replay")
    try:
        issues, warnings, normalization_observations = lifecycle._audit_records(
            records, inventory, data_audit_id=audit_id
        )
        lifecycle._check_expected_audit_counts(config, issues, warnings)
    except lifecycle.Stage1DataError as exc:
        raise Stage1DataAuditValidationError(str(exc)) from exc

    ordered_issue_ids = [issue["issue_id"] for issue in issues]
    issue_counts_by_kind = Counter(issue["issue_kind"] for issue in issues)
    issue_counts_by_code = Counter(issue["issue_code"] for issue in issues)
    issue_counts_by_split: Counter[str] = Counter()
    for issue in issues:
        splits = {location["split"] for location in issue["locations"]}
        issue_counts_by_split[
            next(iter(splits)) if len(splits) == 1 else "multi-split"
        ] += 1
    warning_counts_by_field = Counter(warning["field"] for warning in warnings)

    expected_meta = {
        "schema_version": lifecycle.AUDIT_SCHEMA_VERSION,
        "data_audit_id": audit_id,
        "ordered_issue_ids_sha256": lifecycle.canonical_json_sha256(
            ordered_issue_ids
        ),
        "blocking_issue_count": len(issues),
        "blocking_issue_counts_by_kind": dict(
            sorted(issue_counts_by_kind.items())
        ),
        "blocking_issue_counts_by_code": dict(
            sorted(issue_counts_by_code.items())
        ),
        "blocking_issue_counts_by_split": dict(
            sorted(issue_counts_by_split.items())
        ),
        "warning_count": len(warnings),
        "audit_id_inputs": audit_id_inputs,
    }
    expected_report = {
        "schema_version": "stage1-data-audit-report/v1",
        "data_audit_id": audit_id,
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
    expected_templates = [
        {
            "schema_version": lifecycle.ADJUDICATION_ROW_SCHEMA_VERSION,
            "data_audit_id": audit_id,
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
    expected_provenance = {
        "schema_version": "stage1-data-audit-provenance/v1",
        "data_audit_id": audit_id,
        "source_inventory_sha256": audit_id_inputs[
            "source_inventory_sha256"
        ],
        "config_sha256": lifecycle.canonical_json_sha256(config),
        "rubric_body_sha256": rubric_body_sha256,
        "rubric_meta_sha256": rubric_meta_sha256,
        "audit_code_sha256": audit_code_sha256,
    }

    comparisons = (
        ("config", lifecycle._load_json(target / "config.resolved.json"), config),
        (
            "source inventory",
            lifecycle._load_json(target / "source_inventory.json"),
            inventory["document"],
        ),
        (
            "split manifest",
            lifecycle._load_json(target / "split_manifest.proposed.json"),
            split_manifest,
        ),
        ("audit meta", lifecycle._load_json(target / "audit.meta.json"), expected_meta),
        ("issues", lifecycle._read_jsonl(target / "issues.jsonl"), issues),
        (
            "rubric meta",
            lifecycle._load_json(target / "adjudication_rubric.meta.json"),
            rubric_meta,
        ),
        (
            "adjudication template",
            lifecycle._read_jsonl(target / "adjudication_template.jsonl"),
            expected_templates,
        ),
        (
            "audit report",
            lifecycle._load_json(target / "audit_report.json"),
            expected_report,
        ),
        (
            "provenance",
            lifecycle._load_json(target / "provenance.json"),
            expected_provenance,
        ),
    )
    for label, actual, expected in comparisons:
        if actual != expected:
            _fail(f"data audit {label} differs from exact source replay")
    if (target / "adjudication_rubric.md").read_bytes() != rubric_bytes:
        _fail("data audit rubric bytes differ from the canonical rubric")

    source_guard_after = {
        key: lifecycle.sha256_file(path) for key, path in source_paths.items()
    }
    if source_guard_after != source_guard_before:
        _fail("frozen std source changed during data audit validation")
    if lifecycle._verify_payload_manifest(target) != payload_hash:
        _fail("data audit payload changed during validation")
    return {
        "schema_version": "stage1-data-audit-validation-report/v1",
        "valid": True,
        "data_audit_id": audit_id,
        "payload_manifest_sha256": payload_hash,
        "blocking_issue_count": len(issues),
        "warning_count": len(warnings),
        "source_inventory_sha256": audit_id_inputs["source_inventory_sha256"],
    }


__all__ = [
    "Stage1DataAuditValidationError",
    "validate_data_audit_target",
]
