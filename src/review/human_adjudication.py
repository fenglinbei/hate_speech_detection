"""Safe local helper for Stage 1 human data adjudication.

The helper is deliberately narrower than the lifecycle builders.  It renders
only an allowlisted projection of the frozen audit, never opens model-review
streams, and never signs a reviewer declaration.  Every committed row is
validated against the immutable audit and rubric before the editable JSONL is
atomically replaced.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from data.stage1_data import (
    ADJUDICATION_ROW_SCHEMA_VERSION,
    GROUP_ORDER,
    Stage1DataError,
    _group_hate_issue_spec,
    _parse_group_atoms,
    _validate_completed_rows,
    canonical_record_id,
)
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_json_bytes,
    load_json,
    load_jsonl,
    resolve_locator_ref,
    sha256_file,
    validate_payload_manifest,
)


STATUS_SCHEMA_VERSION = "stage1-human-adjudication-status/v1"
QUEUE_PACKET_SCHEMA_VERSION = "stage1-human-review-queue/v1"
LOCAL_PACKET_SCHEMA_VERSION = "stage1-local-human-review-packet/v1"
FROZEN_REVIEWER_ID = "dual-blind-panel-v1"

ROW_KEYS = frozenset(
    {
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
)
QUEUE_PACKET_KEYS = frozenset(
    {
        "schema_version",
        "issue_id",
        "issue_code",
        "issue_kind",
        "accept_allowed",
        "allowed_edit_paths",
        "review_context",
        "queue_reasons",
        "model_votes_hidden",
    }
)
LOCAL_PACKET_KEYS = frozenset(
    {
        "schema_version",
        "issue_id",
        "issue_code",
        "issue_kind",
        "accept_allowed",
        "allowed_edit_paths",
        "locations",
        "review_context",
        "model_votes_hidden",
        "saw_condition_outputs",
        "saw_model_scores",
    }
)
AUDIT_BOUND_PACKET_FIELDS = (
    "issue_id",
    "issue_code",
    "issue_kind",
    "accept_allowed",
    "allowed_edit_paths",
    "review_context",
)
ALLOWED_QUEUE_REASONS = frozenset(
    {
        "parse_failure",
        "low_confidence",
        "model_disagreement",
        "agreement_qc_sample",
    }
)

# These are the only audit context fields that the terminal UI may render.
# In particular, group-hate queue reasons and the legacy-coupling hint remain
# hidden from the second-round reviewer.
DISPLAY_CONTEXT_FIELDS = {
    "group-hate": ("content", "tuple_before"),
    "field-type": (
        "content",
        "tuple_before",
        "field",
        "observed_type",
        "allowed_correction_types",
    ),
    "duplicate-id": ("content",),
}


class HumanAdjudicationError(RuntimeError):
    """Raised when a human-review workspace violates its frozen contract."""


class HumanAdjudicationIncomplete(HumanAdjudicationError):
    """Raised when strict validation is requested before all rows are done."""


@dataclass
class HumanReviewWorkspace:
    audit_ref: Path
    audit_target: Path
    packet_file: Path
    adjudication_file: Path
    review_target: Path | None
    locator: dict[str, Any]
    issues_by_id: dict[str, dict[str, Any]]
    packets_by_id: dict[str, dict[str, Any]]
    rows: list[dict[str, Any]]
    rubric_meta: dict[str, Any]
    packet_schema_version: str
    reviewer_id: str
    dependency_hashes: dict[Path, str]
    adjudication_sha256: str


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HumanAdjudicationError(f"{label} is not readable: {path}: {exc}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise HumanAdjudicationError(f"{label} must be a regular non-symlink file: {path}")


def _unique_map(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        issue_id = row.get("issue_id")
        if not isinstance(issue_id, str) or not issue_id:
            raise HumanAdjudicationError(f"{label} contains a missing or invalid issue_id")
        if issue_id in result:
            raise HumanAdjudicationError(f"{label} contains duplicate issue_id {issue_id}")
        result[issue_id] = dict(row)
    return result


def _is_pristine(row: Mapping[str, Any]) -> bool:
    return (
        row.get("decision") == ""
        and row.get("edits") == []
        and row.get("reason_code") == ""
        and row.get("reason") == ""
        and row.get("reviewed_at") == ""
    )


def _validate_static_row(
    row: Mapping[str, Any], issue: Mapping[str, Any], *, reviewer_id: str
) -> None:
    issue_id = str(issue["issue_id"])
    if set(row) != ROW_KEYS:
        raise HumanAdjudicationError(f"adjudication row {issue_id} changed its field set")
    expected = {
        "schema_version": ADJUDICATION_ROW_SCHEMA_VERSION,
        "data_audit_id": issue["data_audit_id"],
        "issue_id": issue_id,
        "issue_kind": issue["issue_kind"],
        "reviewer_id": reviewer_id,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise HumanAdjudicationError(
                f"adjudication row {issue_id} changed immutable field {key}"
            )


def _validate_one_completed(
    row: Mapping[str, Any],
    issue: Mapping[str, Any],
    rubric_meta: Mapping[str, Any],
    *,
    reviewer_id: str,
) -> dict[str, Any]:
    try:
        completed = _validate_completed_rows(
            [row], [issue], rubric_meta, reviewer_id=reviewer_id
        )[0]
    except Stage1DataError as exc:
        raise HumanAdjudicationError(str(exc)) from exc
    _validate_semantic_row(completed, issue)
    return completed


def _validate_reviewed_at(value: Any, *, issue_id: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+00:00",
        value,
    ):
        raise HumanAdjudicationError(f"adjudication row {issue_id} has invalid reviewed_at")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HumanAdjudicationError(
            f"adjudication row {issue_id} reviewed_at is not RFC3339"
        ) from exc
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise HumanAdjudicationError(
            f"adjudication row {issue_id} reviewed_at must include a timezone"
        )


def _value_before_edit(issue: Mapping[str, Any], pointer: str) -> Any:
    context = issue.get("review_context")
    if not isinstance(context, dict):
        return object()
    if pointer == "/id":
        return context.get("record_id", object())
    tuple_before = context.get("tuple_before")
    if not isinstance(tuple_before, dict):
        return object()
    return tuple_before.get(pointer.rsplit("/", 1)[-1], object())


def _validate_semantic_row(row: Mapping[str, Any], issue: Mapping[str, Any]) -> None:
    """Apply UI-level constraints that make a correction immediately usable."""

    if row.get("decision") != "corrected":
        _validate_reviewed_at(row.get("reviewed_at"), issue_id=str(issue["issue_id"]))
        return
    for edit in row.get("edits", []):
        if not isinstance(edit, dict):
            continue  # The authoritative shape validator reports this below.
        pointer = edit.get("json_pointer")
        if isinstance(pointer, str):
            _validate_edit_value(pointer, edit.get("value"))
            before = _value_before_edit(issue, pointer)
            if before == edit.get("value"):
                raise HumanAdjudicationError(
                    f"correction for {issue['issue_id']} contains a no-op edit"
                )
    if issue.get("issue_kind") == "field-type":
        values = [edit.get("value") for edit in row.get("edits", []) if isinstance(edit, dict)]
        if row.get("reason_code") == "set-explicit-null" and any(
            value is not None for value in values
        ):
            raise HumanAdjudicationError(
                "set-explicit-null requires every correction value to be JSON null"
            )
        if row.get("reason_code") == "coerce-numeric-annotation" and any(
            not isinstance(value, str) for value in values
        ):
            raise HumanAdjudicationError(
                "coerce-numeric-annotation requires every correction value to be a string"
            )
    if issue.get("issue_kind") == "group-hate":
        context = issue.get("review_context")
        tuple_after = dict(context.get("tuple_before", {})) if isinstance(context, dict) else {}
        for edit in row.get("edits", []):
            if isinstance(edit, dict) and isinstance(edit.get("json_pointer"), str):
                tuple_after[edit["json_pointer"].rsplit("/", 1)[-1]] = edit.get("value")
        try:
            atoms = _parse_group_atoms(tuple_after.get("targeted_group"))
            remaining = _group_hate_issue_spec(atoms, tuple_after.get("hateful"))
        except Stage1DataError as exc:
            raise HumanAdjudicationError(str(exc)) from exc
        if remaining is not None:
            raise HumanAdjudicationError(
                f"correction for {issue['issue_id']} leaves a group–hate blocking issue"
            )
    _validate_reviewed_at(row.get("reviewed_at"), issue_id=str(issue["issue_id"]))


def _row_state(
    row: Mapping[str, Any],
    issue: Mapping[str, Any],
    rubric_meta: Mapping[str, Any],
    *,
    reviewer_id: str,
) -> str:
    _validate_static_row(row, issue, reviewer_id=reviewer_id)
    if _is_pristine(row):
        return "incomplete"
    if row.get("decision") not in {"accepted", "corrected"}:
        raise HumanAdjudicationError(
            f"adjudication row {issue['issue_id']} is partially edited; restore it or complete it"
        )
    _validate_one_completed(row, issue, rubric_meta, reviewer_id=reviewer_id)
    return "complete"


def _validate_packet(
    packet: Mapping[str, Any], issue: Mapping[str, Any], *, schema_version: str
) -> None:
    issue_id = str(issue["issue_id"])
    expected_keys = (
        QUEUE_PACKET_KEYS
        if schema_version == QUEUE_PACKET_SCHEMA_VERSION
        else LOCAL_PACKET_KEYS
    )
    if set(packet) != expected_keys:
        raise HumanAdjudicationError(f"review packet {issue_id} changed its field set")
    if packet.get("schema_version") != schema_version:
        raise HumanAdjudicationError(f"review packet {issue_id} changed schema_version")
    if packet.get("model_votes_hidden") is not True:
        raise HumanAdjudicationError(f"review packet {issue_id} does not hide model votes")
    for field in AUDIT_BOUND_PACKET_FIELDS:
        if packet.get(field) != issue.get(field):
            raise HumanAdjudicationError(
                f"review packet {issue_id} does not match frozen audit field {field}"
            )
    if schema_version == QUEUE_PACKET_SCHEMA_VERSION:
        reasons = packet.get("queue_reasons")
        if (
            not isinstance(reasons, list)
            or not reasons
            or len(reasons) != len(set(reasons))
            or any(reason not in ALLOWED_QUEUE_REASONS for reason in reasons)
        ):
            raise HumanAdjudicationError(f"review packet {issue_id} has invalid queue metadata")
    else:
        if packet.get("locations") != issue.get("locations"):
            raise HumanAdjudicationError(
                f"review packet {issue_id} locations do not match the frozen audit"
            )
        if packet.get("saw_condition_outputs") is not False:
            raise HumanAdjudicationError(
                f"review packet {issue_id} is not blind to condition outputs"
            )
        if packet.get("saw_model_scores") is not False:
            raise HumanAdjudicationError(
                f"review packet {issue_id} is not blind to model scores"
            )


def _load_review_target(
    review_target: Path,
    packets_by_id: Mapping[str, Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    audit_locator: Mapping[str, Any],
    reviewer_id: str,
    expected_review_run_id: str | None = None,
) -> list[Path]:
    if review_target.is_symlink() or not review_target.is_dir():
        raise HumanAdjudicationError(
            f"review target must be a regular directory: {review_target}"
        )
    names = (
        "run.meta.json",
        "human_queue.jsonl",
        "human_adjudication_template.jsonl",
        "auto_adjudication.jsonl",
        "consensus.jsonl",
    )
    paths = [review_target / name for name in names]
    for path in paths:
        _require_regular_file(path, label="review-target input")
    meta = load_json(paths[0])
    expected_run_id = expected_review_run_id or review_target.name
    if (
        not isinstance(meta, dict)
        or meta.get("review_run_id") != expected_run_id
        or meta.get("audit_artifact_id") != audit_locator.get("artifact_id")
        or meta.get("audit_payload_manifest_sha256")
        != audit_locator.get("payload_manifest_sha256")
    ):
        raise HumanAdjudicationError("review target identity does not match its directory")
    frozen_packets = _unique_map(load_jsonl(paths[1]), label="frozen human queue")
    if frozen_packets != {key: dict(value) for key, value in packets_by_id.items()}:
        raise HumanAdjudicationError("packet file does not equal the frozen human queue")
    templates = _unique_map(load_jsonl(paths[2]), label="frozen human template")
    if set(templates) != set(packets_by_id):
        raise HumanAdjudicationError("frozen human queue/template frames differ")
    for issue_id, template in templates.items():
        if set(template) != ROW_KEYS or not _is_pristine(template):
            raise HumanAdjudicationError(f"frozen template {issue_id} is not pristine")
        if template.get("reviewer_id") != reviewer_id:
            raise HumanAdjudicationError(f"frozen template {issue_id} changed reviewer identity")
        for field in (
            "schema_version",
            "data_audit_id",
            "issue_id",
            "issue_kind",
            "reviewer_id",
        ):
            if rows_by_id[issue_id].get(field) != template.get(field):
                raise HumanAdjudicationError(
                    f"adjudication row {issue_id} differs from frozen template field {field}"
                )
    auto_rows = _unique_map(load_jsonl(paths[3]), label="frozen auto adjudication")
    consensus_rows = _unique_map(load_jsonl(paths[4]), label="frozen consensus")
    human_ids = set(packets_by_id)
    auto_ids = set(auto_rows)
    if human_ids.intersection(auto_ids) or human_ids.union(auto_ids) != set(consensus_rows):
        raise HumanAdjudicationError("review target human/auto/consensus frames differ")
    return paths


def load_human_review_workspace(
    *,
    audit_ref: str | Path,
    packet_file: str | Path,
    adjudication_file: str | Path,
    review_target: str | Path | None = None,
    review_ref: str | Path | None = None,
    workspace_root: str | Path | None = None,
    reviewer_id: str = FROZEN_REVIEWER_ID,
) -> HumanReviewWorkspace:
    """Load and fully validate an editable human-review workspace."""

    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise HumanAdjudicationError("reviewer_id must be non-empty")
    audit_ref_path = Path(audit_ref)
    packet_path = Path(packet_file)
    adjudication_path = Path(adjudication_file)
    for path, label in (
        (audit_ref_path, "audit ref"),
        (packet_path, "packet file"),
        (adjudication_path, "adjudication file"),
    ):
        _require_regular_file(path, label=label)

    try:
        locator, audit_target = resolve_locator_ref(
            audit_ref_path, expected_kind="data-audit"
        )
        issues = load_jsonl(audit_target / "issues.jsonl")
        rubric_meta = load_json(audit_target / "adjudication_rubric.meta.json")
        packets = load_jsonl(packet_path)
        rows = load_jsonl(adjudication_path)
    except (TrainingArtifactError, OSError, json.JSONDecodeError) as exc:
        raise HumanAdjudicationError(str(exc)) from exc
    if not isinstance(rubric_meta, dict):
        raise HumanAdjudicationError("audit rubric metadata must be an object")
    issues_by_id = _unique_map(issues, label="audit issues")
    packets_by_id = _unique_map(packets, label="review packets")
    rows_by_id = _unique_map(rows, label="adjudication rows")
    if not packets_by_id:
        raise HumanAdjudicationError("review packet frame is empty")
    if set(rows_by_id) != set(packets_by_id):
        raise HumanAdjudicationError("adjudication and packet issue frames differ")
    extra = sorted(set(packets_by_id) - set(issues_by_id))
    if extra:
        raise HumanAdjudicationError(f"packet frame contains unknown audit issues: {extra}")
    schemas = {packet.get("schema_version") for packet in packets}
    if len(schemas) != 1:
        raise HumanAdjudicationError("one packet file must contain exactly one packet schema")
    packet_schema = next(iter(schemas))
    if packet_schema not in {QUEUE_PACKET_SCHEMA_VERSION, LOCAL_PACKET_SCHEMA_VERSION}:
        raise HumanAdjudicationError(f"unsupported review packet schema {packet_schema!r}")
    if packet_schema == LOCAL_PACKET_SCHEMA_VERSION:
        local_kinds = {str(packet.get("issue_kind")) for packet in packets}
        if len(local_kinds) != 1:
            raise HumanAdjudicationError(
                "one local packet file must contain exactly one audit issue_kind"
            )
        local_kind = next(iter(local_kinds))
        expected_local_ids = {
            issue_id
            for issue_id, issue in issues_by_id.items()
            if issue.get("issue_kind") == local_kind
        }
        if set(packets_by_id) != expected_local_ids:
            raise HumanAdjudicationError(
                "local packet does not cover the complete frozen audit issue-kind frame"
            )

    for issue_id, packet in packets_by_id.items():
        issue = issues_by_id[issue_id]
        _validate_packet(packet, issue, schema_version=str(packet_schema))
        _row_state(
            rows_by_id[issue_id], issue, rubric_meta, reviewer_id=reviewer_id
        )

    if review_target is not None and review_ref is not None:
        raise HumanAdjudicationError(
            "provide a sealed --review-ref or a legacy --review-target, not both"
        )

    resolved_review_target: Path | None = None
    review_ref_path: Path | None = None
    expected_review_run_id: str | None = None
    sealed_review_paths: list[Path] = []
    review_paths: list[Path] = []
    if packet_schema == QUEUE_PACKET_SCHEMA_VERSION:
        if review_ref is not None:
            if workspace_root is None:
                raise HumanAdjudicationError(
                    "sealed review refs require an explicit workspace_root"
                )
            review_ref_path = Path(review_ref)
            _require_regular_file(review_ref_path, label="data blind-review ref")
            try:
                from review.data_review_artifact import resolve_data_review_ref

                _, resolved_review_target, review_report = resolve_data_review_ref(
                    review_ref_path, workspace_root=workspace_root
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise HumanAdjudicationError(str(exc)) from exc
            if review_report.get("data_audit_id") != locator.get("artifact_id"):
                raise HumanAdjudicationError(
                    "sealed review ref does not bind the supplied data audit"
                )
            expected_review_run_id = str(review_report["source_review_run_id"])
            sealed_review_paths = sorted(
                path
                for path in resolved_review_target.rglob("*")
                if path.is_file()
            )
        elif review_target is not None:
            resolved_review_target = Path(review_target)
        else:
            raise HumanAdjudicationError(
                "model-queue packets require a sealed --review-ref "
                "or legacy --review-target"
            )
        review_paths = _load_review_target(
            resolved_review_target,
            packets_by_id,
            rows_by_id,
            audit_locator=locator,
            reviewer_id=reviewer_id,
            expected_review_run_id=expected_review_run_id,
        )
    elif review_target is not None or review_ref is not None:
        raise HumanAdjudicationError(
            "local audit packets must not be paired with a model review ref/target"
        )

    dependency_paths = [
        audit_ref_path,
        packet_path,
        *([review_ref_path] if review_ref_path is not None else []),
        *(sealed_review_paths or review_paths),
    ]
    dependency_hashes = {path: sha256_file(path) for path in dependency_paths}
    return HumanReviewWorkspace(
        audit_ref=audit_ref_path,
        audit_target=audit_target,
        packet_file=packet_path,
        adjudication_file=adjudication_path,
        review_target=resolved_review_target,
        locator=locator,
        issues_by_id={key: issues_by_id[key] for key in packets_by_id},
        packets_by_id=packets_by_id,
        rows=[dict(row) for row in rows],
        rubric_meta=dict(rubric_meta),
        packet_schema_version=str(packet_schema),
        reviewer_id=reviewer_id,
        dependency_hashes=dependency_hashes,
        adjudication_sha256=sha256_file(adjudication_path),
    )


def workspace_status(workspace: HumanReviewWorkspace) -> dict[str, Any]:
    states: list[str] = []
    decisions: Counter[str] = Counter()
    for row in workspace.rows:
        issue = workspace.issues_by_id[str(row["issue_id"])]
        state = _row_state(
            row,
            issue,
            workspace.rubric_meta,
            reviewer_id=workspace.reviewer_id,
        )
        states.append(state)
        if state == "complete":
            decisions[str(row["decision"])] += 1
    occupied: set[tuple[str, int, str]] = set()
    for row in workspace.rows:
        if row.get("decision") != "corrected":
            continue
        issue = workspace.issues_by_id[str(row["issue_id"])]
        for edit in row["edits"]:
            location = issue["locations"][edit["location_index"]]
            coordinate = (
                str(location["source_key"]),
                int(location["source_ordinal"]),
                str(edit["json_pointer"]),
            )
            if coordinate in occupied:
                raise HumanAdjudicationError(
                    "multiple completed rows edit the same frozen source coordinate"
                )
            occupied.add(coordinate)
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "data_audit_id": workspace.locator["artifact_id"],
        "packet_schema_version": workspace.packet_schema_version,
        "reviewer_id": workspace.reviewer_id,
        "row_count": len(workspace.rows),
        "complete_count": states.count("complete"),
        "incomplete_count": states.count("incomplete"),
        "decision_counts": dict(sorted(decisions.items())),
        "all_complete": all(state == "complete" for state in states),
    }


def validate_human_review_workspace(workspace: HumanReviewWorkspace) -> dict[str, Any]:
    status_report = workspace_status(workspace)
    if not status_report["all_complete"]:
        raise HumanAdjudicationIncomplete(
            "human adjudication is incomplete: "
            f"{status_report['complete_count']}/{status_report['row_count']} rows complete"
        )
    return status_report


def render_issue_for_human(
    issue: Mapping[str, Any], *, ordinal: int, total: int, blind_alias: str
) -> str:
    """Return an allowlisted, model-output-free terminal projection."""

    issue_kind = str(issue.get("issue_kind"))
    allowed_fields = DISPLAY_CONTEXT_FIELDS.get(issue_kind)
    if allowed_fields is None:
        raise HumanAdjudicationError(
            f"no safe display projection is defined for issue kind {issue_kind!r}"
        )
    context = issue.get("review_context")
    if not isinstance(context, dict):
        raise HumanAdjudicationError(f"issue {issue.get('issue_id')} lacks review_context")
    lines = [
        f"=== {ordinal}/{total} ===",
        f"盲化编号: {blind_alias}",
        f"issue_kind: {issue_kind}",
        f"issue_code: {issue['issue_code']}",
    ]
    labels = {
        "content": "原文",
        "tuple_before": "当前四元组",
        "field": "异常字段",
        "observed_type": "当前类型",
        "allowed_correction_types": "允许修正类型",
    }
    for field in allowed_fields:
        if field not in context:
            continue
        value = context[field]
        rendered = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, sort_keys=True
        )
        rendered = _terminal_safe(rendered)
        lines.append(f"{labels[field]}: {rendered}")
    lines.append(
        "可接受原标注: " + ("是" if issue.get("accept_allowed") is True else "否")
    )
    lines.append(
        "允许编辑路径: "
        + json.dumps(issue.get("allowed_edit_paths"), ensure_ascii=False)
    )
    return "\n".join(lines)


def _terminal_safe(value: str) -> str:
    """Escape terminal controls and Unicode directional overrides."""

    directional = {
        0x061C,
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2066,
        0x2067,
        0x2068,
        0x2069,
    }
    rendered: list[str] = []
    for character in value:
        codepoint = ord(character)
        if codepoint < 0x20 or 0x7F <= codepoint <= 0x9F or codepoint in directional:
            rendered.append(f"\\u{codepoint:04x}")
        else:
            rendered.append(character)
    return "".join(rendered)


def _blind_alias(data_audit_id: str, issue_id: str) -> str:
    digest = hashlib.sha256(
        f"stage1-human-alias/v1:{data_audit_id}:{issue_id}".encode("utf-8")
    ).hexdigest()
    return "H-" + digest[:12]


def _blind_order_key(data_audit_id: str, issue_id: str) -> str:
    return hashlib.sha256(
        f"stage1-human-order/v1:{data_audit_id}:{issue_id}".encode("utf-8")
    ).hexdigest()


def _prompt_choice(
    prompt: str,
    *,
    allowed: set[str],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    while True:
        answer = input_fn(prompt).strip().lower()
        if answer in allowed:
            return answer
        output_fn("输入无效，请重试。")


def _prompt_index(
    label: str,
    values: Sequence[str],
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> int:
    if not values:
        raise HumanAdjudicationError(f"{label} has no allowed values")
    if len(values) == 1:
        output_fn(f"{label}: {values[0]}")
        return 0
    for index, value in enumerate(values, 1):
        output_fn(f"  [{index}] {value}")
    while True:
        raw = input_fn(f"选择{label}编号: ").strip()
        try:
            selected = int(raw) - 1
        except ValueError:
            selected = -1
        if 0 <= selected < len(values):
            return selected
        output_fn("编号无效，请重试。")


def _prompt_paths(
    paths: Sequence[str],
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> list[str]:
    if not paths:
        raise HumanAdjudicationError("issue has no allowed edit paths")
    if len(paths) == 1:
        output_fn(f"编辑路径: {paths[0]}")
        return [str(paths[0])]
    for index, path in enumerate(paths, 1):
        output_fn(f"  [{index}] {path}")
    while True:
        raw = input_fn("选择一个或多个编辑路径编号（逗号分隔）: ").strip()
        try:
            indices = [int(value.strip()) - 1 for value in raw.split(",")]
        except ValueError:
            indices = []
        if (
            indices
            and len(indices) == len(set(indices))
            and all(0 <= index < len(paths) for index in indices)
        ):
            return [str(paths[index]) for index in sorted(indices)]
        output_fn("路径编号无效或重复，请重试。")


def _validate_edit_value(pointer: str, value: Any) -> None:
    field = pointer.rsplit("/", 1)[-1]
    if field == "hateful" and value not in {"hate", "non-hate"}:
        raise HumanAdjudicationError(
            "hateful 修正值必须是 JSON 字符串 \"hate\" 或 \"non-hate\""
        )
    if field in {"target", "argument"} and not (
        value is None or isinstance(value, str)
    ):
        raise HumanAdjudicationError(f"{field} 修正值必须是 JSON 字符串或 null")
    if field in {"target", "argument"} and isinstance(value, str):
        if (
            not value
            or value != value.strip()
            or value != unicodedata.normalize("NFC", value)
            or value == "NULL"
        ):
            raise HumanAdjudicationError(
                f"{field} 字符串必须非空、无首尾空白、为 NFC，且不能是 \"NULL\""
            )
    if field == "targeted_group":
        try:
            _parse_group_atoms(value)
        except Stage1DataError as exc:
            raise HumanAdjudicationError(str(exc)) from exc
    if field == "id":
        try:
            canonical_record_id(value)
        except Stage1DataError as exc:
            raise HumanAdjudicationError(str(exc)) from exc
    try:
        canonical_json_bytes(value)
    except TrainingArtifactError as exc:
        raise HumanAdjudicationError(str(exc)) from exc


def _prompt_json_value(
    pointer: str,
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> Any:
    while True:
        raw = input_fn(
            f"输入 {pointer} 的 JSON 值（字符串必须带双引号，空值写 null）: "
        )
        try:
            value = json.loads(raw)
            _validate_edit_value(pointer, value)
            return value
        except (json.JSONDecodeError, HumanAdjudicationError, TrainingArtifactError) as exc:
            output_fn(f"JSON 值无效: {exc}")


def _prompt_edit_value(
    pointer: str,
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> Any:
    field = pointer.rsplit("/", 1)[-1]
    if field == "hateful":
        values = ["hate", "non-hate"]
        return values[
            _prompt_index("hateful", values, input_fn=input_fn, output_fn=output_fn)
        ]
    if field == "targeted_group":
        for index, group in enumerate(GROUP_ORDER, 1):
            output_fn(f"  [{index}] {group}")
        while True:
            raw = input_fn(
                "选择一个或多个 targeted_group 编号（逗号分隔）: "
            ).strip()
            try:
                indices = [int(value.strip()) - 1 for value in raw.split(",")]
            except ValueError:
                indices = []
            if (
                indices
                and len(indices) == len(set(indices))
                and all(0 <= index < len(GROUP_ORDER) for index in indices)
            ):
                selected = [group for group in GROUP_ORDER if GROUP_ORDER.index(group) in indices]
                if "non-hate" not in selected or len(selected) == 1:
                    return ", ".join(selected)
            output_fn("targeted_group 选择无效；non-hate 不能与其他组并存。")
    return _prompt_json_value(pointer, input_fn=input_fn, output_fn=output_fn)


def _build_candidate_row(
    pristine_row: Mapping[str, Any],
    issue: Mapping[str, Any],
    rubric_meta: Mapping[str, Any],
    *,
    action: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> dict[str, Any]:
    decision = "accepted" if action == "a" else "corrected"
    edits: list[dict[str, Any]] = []
    if decision == "corrected":
        paths = _prompt_paths(
            list(issue["allowed_edit_paths"]), input_fn=input_fn, output_fn=output_fn
        )
        locations = issue.get("locations")
        if not isinstance(locations, list) or not locations:
            raise HumanAdjudicationError(f"issue {issue['issue_id']} has no edit locations")
        for pointer in paths:
            if len(locations) == 1:
                location_index = 0
            else:
                safe_locations = [
                    "tuple_index="
                    + str(location.get("tuple_index"))
                    + ", detected_pointer="
                    + str(location.get("json_pointer"))
                    for location in locations
                ]
                location_index = _prompt_index(
                    "位置", safe_locations, input_fn=input_fn, output_fn=output_fn
                )
            edits.append(
                {
                    "location_index": location_index,
                    "op": "set",
                    "json_pointer": pointer,
                    "value": _prompt_edit_value(
                        pointer, input_fn=input_fn, output_fn=output_fn
                    ),
                }
            )
        edits.sort(key=lambda edit: (edit["location_index"], edit["json_pointer"]))
    codes = (
        rubric_meta.get("reason_codes", {})
        .get(issue.get("issue_code"), {})
        .get(decision, [])
    )
    if not isinstance(codes, list) or not codes or not all(
        isinstance(code, str) and code for code in codes
    ):
        raise HumanAdjudicationError(
            f"rubric has no valid {decision} reason codes for {issue['issue_id']}"
        )
    reason_code = codes[
        _prompt_index("reason_code", codes, input_fn=input_fn, output_fn=output_fn)
    ]
    while True:
        reason = input_fn("填写裁决理由: ").strip()
        if reason:
            break
        output_fn("裁决理由不能为空。")
    candidate = dict(pristine_row)
    candidate.update(
        {
            "decision": decision,
            "edits": edits,
            "reason_code": reason_code,
            "reason": reason,
            "reviewed_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
        }
    )
    return candidate


def _assert_dependencies_unchanged(workspace: HumanReviewWorkspace) -> None:
    try:
        if validate_payload_manifest(workspace.audit_target) != workspace.locator.get(
            "payload_manifest_sha256"
        ):
            raise HumanAdjudicationError("audit payload changed during review")
        for path, expected in workspace.dependency_hashes.items():
            _require_regular_file(path, label="review dependency")
            if sha256_file(path) != expected:
                raise HumanAdjudicationError(
                    f"review dependency changed during session: {path}"
                )
        _require_regular_file(workspace.adjudication_file, label="adjudication file")
        if sha256_file(workspace.adjudication_file) != workspace.adjudication_sha256:
            raise HumanAdjudicationError(
                "adjudication file changed concurrently; reload before continuing"
            )
    except TrainingArtifactError as exc:
        raise HumanAdjudicationError(str(exc)) from exc


def _write_rows_atomic(workspace: HumanReviewWorkspace) -> None:
    destination = workspace.adjudication_file
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise HumanAdjudicationError("adjudication parent must be a regular directory")
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in workspace.rows)
    lock_flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    lock_descriptor = os.open(destination, lock_flags)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HumanAdjudicationError(
                "another reviewer currently holds the adjudication file lock"
            ) from exc
        _assert_dependencies_unchanged(workspace)
        mode = stat.S_IMODE(os.fstat(lock_descriptor).st_mode)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=str(parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                os.fchmod(handle.fileno(), mode)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            directory_descriptor = os.open(
                parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if temporary.exists():
                temporary.unlink()
    finally:
        os.close(lock_descriptor)
    workspace.adjudication_sha256 = sha256_file(destination)


def review_interactively(
    workspace: HumanReviewWorkspace,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Review pristine rows, committing one validated decision at a time."""

    audit_id = str(workspace.locator["artifact_id"])
    aliases = {
        str(row["issue_id"]): _blind_alias(audit_id, str(row["issue_id"]))
        for row in workspace.rows
    }
    if len(set(aliases.values())) != len(aliases):
        raise HumanAdjudicationError("blind alias collision in review frame")
    ordered_indices = sorted(
        range(len(workspace.rows)),
        key=lambda index: _blind_order_key(
            audit_id, str(workspace.rows[index]["issue_id"])
        ),
    )
    total = len(workspace.rows)
    for ordinal, row_index in enumerate(ordered_indices, 1):
        row = workspace.rows[row_index]
        issue = workspace.issues_by_id[str(row["issue_id"])]
        state = _row_state(
            row,
            issue,
            workspace.rubric_meta,
            reviewer_id=workspace.reviewer_id,
        )
        if state == "complete":
            continue
        alias = aliases[str(row["issue_id"])]
        output_fn(
            render_issue_for_human(
                issue, ordinal=ordinal, total=total, blind_alias=alias
            )
        )
        allowed = {"c", "s", "q"}
        prompt = "操作：[c] 修正  [s] 跳过  [q] 退出"
        if issue.get("accept_allowed") is True:
            allowed.add("a")
            prompt = "操作：[a] 接受原标注  [c] 修正  [s] 跳过  [q] 退出"
        while True:
            try:
                action = _prompt_choice(
                    prompt + ": ",
                    allowed=allowed,
                    input_fn=input_fn,
                    output_fn=output_fn,
                )
                if action == "q":
                    return workspace_status(workspace)
                if action == "s":
                    break
                try:
                    candidate = _build_candidate_row(
                        row,
                        issue,
                        workspace.rubric_meta,
                        action=action,
                        input_fn=input_fn,
                        output_fn=output_fn,
                    )
                    _validate_one_completed(
                        candidate,
                        issue,
                        workspace.rubric_meta,
                        reviewer_id=workspace.reviewer_id,
                    )
                except HumanAdjudicationError:
                    output_fn(
                        f"{alias} 的裁决未通过冻结规则；本条未写入，请重新选择。"
                    )
                    continue
                summary = {
                    "decision": candidate["decision"],
                    "edits": candidate["edits"],
                    "reason_code": candidate["reason_code"],
                    "reason": candidate["reason"],
                }
                output_fn(
                    "待提交裁决: "
                    + _terminal_safe(
                        json.dumps(summary, ensure_ascii=False, sort_keys=True)
                    )
                )
                confirmed = _prompt_choice(
                    "确认并原子写入？[y/n]: ",
                    allowed={"y", "n"},
                    input_fn=input_fn,
                    output_fn=output_fn,
                )
                if confirmed == "n":
                    output_fn("本条未写入，请重新选择或跳过。")
                    continue
                previous = workspace.rows[row_index]
                workspace.rows[row_index] = candidate
                try:
                    workspace_status(workspace)
                    _write_rows_atomic(workspace)
                except Exception:
                    workspace.rows[row_index] = previous
                    raise
                output_fn(f"已提交 {alias}。")
                break
            except (EOFError, KeyboardInterrupt):
                output_fn("\n审查已中止；已确认的行仍保留，当前行未写入。")
                return workspace_status(workspace)
    return workspace_status(workspace)


__all__ = [
    "FROZEN_REVIEWER_ID",
    "HumanAdjudicationError",
    "HumanAdjudicationIncomplete",
    "HumanReviewWorkspace",
    "load_human_review_workspace",
    "render_issue_for_human",
    "review_interactively",
    "validate_human_review_workspace",
    "workspace_status",
]
