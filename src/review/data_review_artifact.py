"""Seal legacy Stage-1 data blind-review runs as immutable artifacts.

The original data-review runner writes a complete, auditable directory but it
predates the shared immutable-artifact lifecycle.  This module is a narrow
bridge: it validates that directory against its data-audit dependency, copies
all model/raw provenance into a canonical snapshot, removes the source-only
absolute ``summary.target_path``, and publishes a content-addressed locator.

No function in this module resolves credentials or performs network calls.
Public validation reports deliberately contain counts and hashes only; callers
must explicitly open the sealed target if they are authorized to inspect votes.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from data.stage1_data import Stage1DataError, _validate_completed_rows
from data.training_artifacts import (
    SHA256_RE,
    TrainingArtifactError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    canonical_sha256,
    finalize_target_atomic,
    load_json,
    load_jsonl,
    new_staging_directory,
    portable_dependency,
    resolve_dependency_target,
    resolve_locator_ref,
    sha256_file,
    validate_dependency_ref,
    validate_payload_manifest,
    write_bytes_atomic,
    write_canonical_json,
    write_locator_ref,
)
from review.blind_review import (
    BlindReviewError,
    SYSTEM_PROMPT,
    _chat_url,
    _extract_json_object,
    _item_prompt,
    _normalize_judgement_shape,
    _validate_judgement,
    build_consensus,
)
from review.d14_contract import (
    D14_BLINDNESS,
    D14_CONSENSUS,
    D14_POLICY_SCHEMA,
    D14_PROVENANCE,
    D14_REQUEST_OPTIONS,
    D14_REVIEWER_CONFIG,
    D14_REVIEWER_IDS,
    D14_REVIEWER_RESOLUTION,
    validate_d14_policy_document,
    validate_requested_returned_model,
    validate_reviewer_resolution,
)


ARTIFACT_KIND = "data-blind-review"
ARTIFACT_PREFIX = "dreview-"
SOURCE_RUN_ID_RE = re.compile(r"^dblind-[0-9a-f]{64}$")
SOURCE_RUN_SCHEMA = "stage1-dual-blind-review-run/v1"
SOURCE_SUMMARY_SCHEMA = "stage1-dual-blind-review-summary/v1"
MODEL_REVIEW_SCHEMA = "stage1-model-review/v1"
CONSENSUS_SCHEMA = "stage1-review-consensus/v1"
QUEUE_SCHEMA = "stage1-human-review-queue/v1"
FINAL_ROW_SCHEMA = "stage1-data-adjudication-row/v1"
POLICY_SCHEMA = D14_POLICY_SCHEMA
PROVENANCE_SCHEMA = "stage1-data-blind-review-provenance/v1"
ID_INPUT_SCHEMA = "stage1-data-blind-review-id-input/v1"
VALIDATION_SCHEMA = "stage1-data-blind-review-validation/v1"
PANEL_REVIEWER_ID = "dual-blind-panel-v1"
REVIEWER_IDS = D14_REVIEWER_IDS
APPROVED_REVIEWER_RESOLUTION = D14_REVIEWER_RESOLUTION
APPROVED_REVIEWER_CONFIG = D14_REVIEWER_CONFIG
# The first live DeepSeek pass in this immutable parent run inherited the GLM
# request-option profile.  Its system/user prompts, endpoint, requested model,
# returned model, raw response, and judgements remain exactly replayable.  The
# compatibility branch below is restricted to this content-addressed retry
# lineage; all later runs must use the reviewer-specific D14 option profile.
LEGACY_OPTION_ROUTING_RUN = (
    "dblind-705414ce7e750434417387c6e646ea92bd0a300849e42785baa9ad094997b54d"
)
LEGACY_OPTION_ROUTING_PARENT = (
    "dblind-f27df1a78d56e1b314219bcb015db796c5a90b9d31ecdd5887f246471f6fad87"
)

JSON_FILES = (
    "run.meta.json",
    "policy.resolved.json",
    "model_resolution.json",
    "summary.json",
)
JSONL_FILES = (
    "consensus.jsonl",
    "human_queue.jsonl",
    "auto_adjudication.jsonl",
    "human_adjudication_template.jsonl",
)
GENERATED_FILES = ("data_audit_ref.json", "provenance.json", "payload_manifest.json")


class DataReviewArtifactError(RuntimeError):
    """Raised when a source run or sealed data-review artifact is invalid."""


def _source_file_set() -> set[str]:
    return {
        *JSON_FILES,
        *JSONL_FILES,
        *(f"model_reviews/{reviewer_id}.jsonl" for reviewer_id in REVIEWER_IDS),
    }


def _sealed_file_set() -> set[str]:
    return _source_file_set() | set(GENERATED_FILES)


def _inspect_regular_tree(target: Path, expected_files: set[str]) -> dict[str, str]:
    """Reject links, special nodes, unexpected directories/files; return hashes."""

    if target.is_symlink() or not target.is_dir():
        raise DataReviewArtifactError(f"review target is not a regular directory: {target}")
    allowed_directories = {"", "model_reviews"}
    actual_files: set[str] = set()
    actual_directories = {""}
    for directory, dirnames, filenames in os.walk(target, topdown=True, followlinks=False):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(target).as_posix()
        if relative_directory == ".":
            relative_directory = ""
        if relative_directory not in allowed_directories:
            raise DataReviewArtifactError(
                f"unexpected review directory: {relative_directory}"
            )
        for name in sorted(dirnames):
            path = directory_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise DataReviewArtifactError(f"review target contains symlink: {path}")
            if not stat.S_ISDIR(mode):
                raise DataReviewArtifactError(f"review target contains special node: {path}")
            relative = path.relative_to(target).as_posix()
            actual_directories.add(relative)
            if relative not in allowed_directories:
                raise DataReviewArtifactError(f"unexpected review directory: {relative}")
        for name in sorted(filenames):
            path = directory_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise DataReviewArtifactError(f"review target contains symlink: {path}")
            if not stat.S_ISREG(mode):
                raise DataReviewArtifactError(f"review target contains special node: {path}")
            actual_files.add(path.relative_to(target).as_posix())
    if actual_directories != allowed_directories:
        raise DataReviewArtifactError(
            "review target directory frame mismatch: "
            f"missing={sorted(allowed_directories-actual_directories)}, "
            f"extra={sorted(actual_directories-allowed_directories)}"
        )
    if actual_files != expected_files:
        raise DataReviewArtifactError(
            "review target file frame mismatch: "
            f"missing={sorted(expected_files-actual_files)}, "
            f"extra={sorted(actual_files-expected_files)}"
        )
    return {relative: sha256_file(target / relative) for relative in sorted(actual_files)}


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise DataReviewArtifactError(f"{field} must be a lowercase SHA-256")
    return value


def _reviewer_projection(document: Any) -> list[dict[str, str]]:
    reviewers = document.get("reviewers") if isinstance(document, Mapping) else None
    if not isinstance(reviewers, list) or len(reviewers) != 2:
        raise DataReviewArtifactError("model resolution must contain exactly two reviewers")
    normalized: list[dict[str, str]] = []
    for row in reviewers:
        if not isinstance(row, Mapping) or set(row) != {"reviewer_id", "api_base", "model"}:
            raise DataReviewArtifactError("model-resolution reviewer is non-canonical")
        if any(not isinstance(row.get(key), str) or not row[key] for key in row):
            raise DataReviewArtifactError("model-resolution reviewer fields must be non-empty")
        normalized.append({key: str(row[key]) for key in ("reviewer_id", "api_base", "model")})
    try:
        validate_reviewer_resolution(normalized, require_live_d14=True)
    except ValueError as exc:
        raise DataReviewArtifactError(str(exc)) from exc
    return normalized


def _validate_policy(policy: Any, reviewer_ids: set[str]) -> dict[str, Any]:
    if not isinstance(policy, Mapping) or policy.get("schema_version") != POLICY_SCHEMA:
        raise DataReviewArtifactError("unsupported data blind-review policy")
    try:
        canonical = validate_d14_policy_document(policy)
    except ValueError as exc:
        raise DataReviewArtifactError(str(exc)) from exc
    policy_ids = {row["reviewer_id"] for row in canonical["reviewers"]}
    if policy_ids != reviewer_ids:
        raise DataReviewArtifactError("policy/model-resolution reviewer frames differ")
    return canonical


def _load_audit(
    dependency: Mapping[str, Any], audit_target: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        audit_target.name != dependency["artifact_id"]
        or validate_payload_manifest(audit_target) != dependency["payload_manifest_sha256"]
    ):
        raise DataReviewArtifactError("data-audit dependency does not match its target")
    issues = load_jsonl(audit_target / "issues.jsonl")
    rubric_meta = load_json(audit_target / "adjudication_rubric.meta.json")
    if not isinstance(rubric_meta, Mapping):
        raise DataReviewArtifactError("data-audit rubric metadata is malformed")
    issue_ids = [row.get("issue_id") for row in issues]
    if (
        not issues
        or any(not isinstance(value, str) or not value for value in issue_ids)
        or len(issue_ids) != len(set(issue_ids))
        or any(row.get("data_audit_id") != dependency["artifact_id"] for row in issues)
    ):
        raise DataReviewArtifactError("data-audit issue frame is malformed")
    return issues, dict(rubric_meta)


def _load_frozen_policy(
    workspace_root: str | Path,
) -> tuple[dict[str, Any], str, list[dict[str, str]]]:
    """Load the repository policy that authorized the provider calls."""

    policy_path = Path(workspace_root).resolve() / "config" / "stage1" / "blind_review.json"
    try:
        mode = policy_path.lstat().st_mode
    except OSError as exc:
        raise DataReviewArtifactError("frozen blind-review policy is unavailable") from exc
    if policy_path.is_symlink() or not stat.S_ISREG(mode):
        raise DataReviewArtifactError("frozen blind-review policy must be a regular file")
    policy = load_json(policy_path)
    if not isinstance(policy, Mapping):
        raise DataReviewArtifactError("frozen blind-review policy must be an object")
    try:
        policy = validate_d14_policy_document(policy)
    except ValueError as exc:
        raise DataReviewArtifactError(str(exc)) from exc
    reviewers = policy.get("reviewers")
    if not isinstance(reviewers, list) or len(reviewers) != 2:
        raise DataReviewArtifactError("frozen blind-review policy has no reviewer pair")
    expected_resolution: list[dict[str, str]] = []
    for reviewer in reviewers:
        if not isinstance(reviewer, Mapping):
            raise DataReviewArtifactError("frozen blind-review reviewer is malformed")
        reviewer_id = reviewer.get("reviewer_id")
        api_base = reviewer.get("default_api_base")
        model = reviewer.get("default_model")
        if any(not isinstance(value, str) or not value for value in (reviewer_id, api_base, model)):
            raise DataReviewArtifactError("frozen reviewer defaults are incomplete")
        expected_resolution.append(
            {"reviewer_id": reviewer_id, "api_base": api_base, "model": model}
        )
    if tuple(sorted(row["reviewer_id"] for row in expected_resolution)) != REVIEWER_IDS:
        raise DataReviewArtifactError("frozen policy is not the approved GLM/DeepSeek pair")
    if expected_resolution != list(APPROVED_REVIEWER_RESOLUTION):
        raise DataReviewArtifactError(
            "frozen reviewer endpoint/model defaults differ from the approved decision"
        )
    if reviewers != list(APPROVED_REVIEWER_CONFIG):
        raise DataReviewArtifactError(
            "frozen reviewer credential/backend bindings differ from the approved decision"
        )
    if (
        policy["blindness"] != D14_BLINDNESS
        or policy["consensus"] != D14_CONSENSUS
        or policy["reviewer_request_options"] != D14_REQUEST_OPTIONS
        or policy["provenance"] != D14_PROVENANCE
    ):  # Defensive assertion; validate_d14_policy_document already checks this.
        raise DataReviewArtifactError("frozen blind-review controls differ from D14")
    return dict(policy), sha256_file(policy_path), expected_resolution


def _read_snapshot(target: Path, *, source: bool) -> dict[str, Any]:
    documents = {name: load_json(target / name) for name in JSON_FILES}
    rows = {name: load_jsonl(target / name) for name in JSONL_FILES}
    model_rows = {
        reviewer_id: load_jsonl(target / "model_reviews" / f"{reviewer_id}.jsonl")
        for reviewer_id in REVIEWER_IDS
    }
    summary = documents["summary.json"]
    if not isinstance(summary, Mapping):
        raise DataReviewArtifactError("review summary must be an object")
    summary = dict(summary)
    if source:
        source_path = summary.pop("target_path", None)
        if not isinstance(source_path, str) or not Path(source_path).is_absolute():
            raise DataReviewArtifactError("source review summary lacks an absolute target_path")
        if Path(source_path).resolve() != target.resolve():
            raise DataReviewArtifactError("source review summary target_path does not match source")
    elif "target_path" in summary:
        raise DataReviewArtifactError("sealed review summary contains source target_path")
    documents["summary.json"] = summary
    return {"documents": documents, "rows": rows, "model_rows": model_rows}


def _blind_map(issues: Sequence[Mapping[str, Any]], seed: int) -> dict[str, str]:
    ordered = sorted(
        issues,
        key=lambda issue: hashlib.sha256(
            f"{seed}:{issue['issue_id']}".encode("utf-8")
        ).hexdigest(),
    )
    return {str(issue["issue_id"]): f"B{index:04d}" for index, issue in enumerate(ordered, 1)}


def _contains_sensitive_key(value: Any) -> bool:
    forbidden = {"authorization", "api_key", "apikey", "credentials", "password", "headers"}
    if isinstance(value, Mapping):
        return any(str(key).lower() in forbidden or _contains_sensitive_key(nested) for key, nested in value.items())
    if isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _validate_model_row(
    row: Mapping[str, Any],
    *,
    issue: Mapping[str, Any],
    reviewer: Mapping[str, str],
    blind_item_id: str,
    rubric_meta: Mapping[str, Any],
    policy: Mapping[str, Any],
    review_run_id: str,
    retry_parent: str | None,
) -> None:
    base_keys = {
        "schema_version",
        "issue_id",
        "blind_item_id",
        "reviewer_id",
        "provider_api_base",
        "requested_model",
        "returned_model",
        "provider_request_id",
        "prompt_sha256",
        "request_without_secrets",
        "raw_response",
        "judgement",
        "normalizations",
        "error",
    }
    allowed_keys = base_keys | ({"retry_lineage"} if retry_parent is not None else set())
    if set(row) not in (base_keys, allowed_keys) or row.get("schema_version") != MODEL_REVIEW_SCHEMA:
        raise DataReviewArtifactError("model-review row has non-canonical fields/schema")
    if (
        row.get("issue_id") != issue["issue_id"]
        or row.get("blind_item_id") != blind_item_id
        or row.get("reviewer_id") != reviewer["reviewer_id"]
        or row.get("provider_api_base") != reviewer["api_base"]
        or row.get("requested_model") != reviewer["model"]
    ):
        raise DataReviewArtifactError("model-review identity/frame mismatch")
    requested_model = row.get("requested_model")
    returned_model = row.get("returned_model")
    try:
        validate_requested_returned_model(requested_model, returned_model)
    except ValueError as exc:
        raise DataReviewArtifactError(str(exc)) from exc
    prompt = _item_prompt(blind_item_id, issue, rubric_meta)
    prompt_sha256 = hashlib.sha256(
        (SYSTEM_PROMPT + "\n" + prompt).encode("utf-8")
    ).hexdigest()
    if row.get("prompt_sha256") != prompt_sha256:
        raise DataReviewArtifactError("model-review prompt hash cannot be reproduced")
    request = row.get("request_without_secrets")
    expected_payload: dict[str, Any] = {
        "model": reviewer["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": policy["temperature"],
        "top_p": policy["top_p"],
        "stream": False,
        "max_tokens": policy["max_tokens"],
        "response_format": {"type": "json_object"},
        **dict(D14_REQUEST_OPTIONS[reviewer["reviewer_id"]]),
    }
    legacy_payload = dict(expected_payload)
    legacy_payload.update(D14_REQUEST_OPTIONS["glm_high_parameter"])
    legacy_option_routing = (
        review_run_id == LEGACY_OPTION_ROUTING_RUN
        and retry_parent == LEGACY_OPTION_ROUTING_PARENT
        and reviewer["reviewer_id"] == "deepseek_flash"
        and row.get("retry_lineage") is None
        and isinstance(request, Mapping)
        and request.get("payload") == legacy_payload
    )
    if (
        not isinstance(request, Mapping)
        or set(request) != {"url", "payload", "prompt_sha256"}
        or request.get("prompt_sha256") != prompt_sha256
        or request.get("url") != _chat_url(reviewer["api_base"])
        or (
            request.get("payload") != expected_payload
            and not legacy_option_routing
        )
        or _contains_sensitive_key(request)
    ):
        raise DataReviewArtifactError(
            "model-review prompt/request provenance is malformed or non-canonical"
        )
    raw = row.get("raw_response")
    if not isinstance(raw, Mapping):
        raise DataReviewArtifactError(
            "D14 model review must retain a raw response with provider model identity"
        )
    if returned_model != raw.get("model"):
        raise DataReviewArtifactError("returned model does not match raw response")
    if row.get("provider_request_id") != raw.get("id"):
        raise DataReviewArtifactError("provider request ID does not match raw response")
    if not isinstance(row.get("normalizations"), list):
        raise DataReviewArtifactError("model-review normalizations must be an array")
    judgement = row.get("judgement")
    error = row.get("error")
    if error is None:
        if not isinstance(judgement, Mapping):
            raise DataReviewArtifactError("successful model review lacks judgement")
        try:
            choices = raw.get("choices")
            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], Mapping)
                or choices[0].get("finish_reason") != "stop"
                or not isinstance(choices[0].get("message"), Mapping)
                or not isinstance(choices[0]["message"].get("content"), str)
            ):
                raise DataReviewArtifactError(
                    "successful raw provider response has a non-canonical choice"
                )
            parsed, normalizations = _normalize_judgement_shape(
                _extract_json_object(choices[0]["message"]["content"])
            )
            if (
                _validate_judgement(parsed, issue, rubric_meta) != judgement
                or normalizations != row.get("normalizations")
            ):
                raise DataReviewArtifactError("model judgement is not canonical")
        except (BlindReviewError, ValueError, KeyError, TypeError) as exc:
            raise DataReviewArtifactError(f"invalid model judgement: {exc}") from exc
    elif judgement is not None or not isinstance(error, Mapping):
        raise DataReviewArtifactError("failed model review has inconsistent error/judgement")
    lineage = row.get("retry_lineage")
    if lineage is not None:
        if (
            retry_parent is None
            or not isinstance(lineage, Mapping)
            or set(lineage) != {"parent_review_run_id", "previous_error", "attempt"}
            or lineage.get("parent_review_run_id") != retry_parent
            or not isinstance(lineage.get("previous_error"), Mapping)
            or lineage.get("attempt") != 2
        ):
            raise DataReviewArtifactError("model-review retry lineage is malformed")


def _validate_run_meta(
    run_meta: Any,
    *,
    dependency: Mapping[str, Any],
    reviewers: list[dict[str, str]],
    policy_sha256: str,
) -> str | None:
    if not isinstance(run_meta, Mapping):
        raise DataReviewArtifactError("run metadata must be an object")
    initial_keys = {
        "schema_version", "audit_artifact_id", "audit_payload_manifest_sha256",
        "policy_sha256", "system_prompt_sha256", "reviewers", "issue_scope",
        "ordered_issue_ids_sha256", "review_run_id",
    }
    retry_keys = {
        "schema_version", "audit_artifact_id", "audit_payload_manifest_sha256",
        "parent_review_run_id", "retry_policy", "retry_issue_ids_sha256_by_reviewer",
        "policy_sha256", "system_prompt_sha256", "reviewers",
        "ordered_issue_ids_sha256", "review_run_id",
    }
    if set(run_meta) not in (initial_keys, retry_keys):
        raise DataReviewArtifactError("run metadata has an unsupported initial/retry shape")
    run_id = run_meta.get("review_run_id")
    if (
        run_meta.get("schema_version") != SOURCE_RUN_SCHEMA
        or not isinstance(run_id, str)
        or not SOURCE_RUN_ID_RE.fullmatch(run_id)
        or run_meta.get("audit_artifact_id") != dependency["artifact_id"]
        or run_meta.get("audit_payload_manifest_sha256") != dependency["payload_manifest_sha256"]
        or run_meta.get("reviewers") != reviewers
        or run_meta.get("policy_sha256") != policy_sha256
        or run_meta.get("system_prompt_sha256")
        != hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    ):
        raise DataReviewArtifactError("run metadata identity/dependency frame is invalid")
    for field in ("policy_sha256", "system_prompt_sha256", "ordered_issue_ids_sha256"):
        _require_sha256(run_meta.get(field), f"run metadata {field}")
    run_inputs = dict(run_meta)
    run_inputs.pop("review_run_id")
    if run_id != "dblind-" + canonical_sha256(run_inputs):
        raise DataReviewArtifactError("source review run ID cannot be reproduced")
    if set(run_meta) == initial_keys:
        return None
    parent = run_meta.get("parent_review_run_id")
    retry_hashes = run_meta.get("retry_issue_ids_sha256_by_reviewer")
    if (
        not isinstance(parent, str)
        or not SOURCE_RUN_ID_RE.fullmatch(parent)
        or run_meta.get("retry_policy") != "failed-reviewer-items-only/v1"
        or not isinstance(retry_hashes, Mapping)
        or set(retry_hashes) != set(REVIEWER_IDS)
    ):
        raise DataReviewArtifactError("retry run lineage is malformed")
    for reviewer_id, value in retry_hashes.items():
        _require_sha256(value, f"retry issue hash for {reviewer_id}")
    return parent


def _validate_snapshot(
    snapshot: Mapping[str, Any],
    *,
    dependency: Mapping[str, Any],
    audit_issues: Sequence[Mapping[str, Any]],
    rubric_meta: Mapping[str, Any],
    frozen_policy: Mapping[str, Any],
    frozen_policy_sha256: str,
    expected_reviewers: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    documents = snapshot["documents"]
    rows = snapshot["rows"]
    model_rows = snapshot["model_rows"]
    reviewers = _reviewer_projection(documents["model_resolution.json"])
    if reviewers != list(expected_reviewers):
        raise DataReviewArtifactError(
            "resolved reviewer models differ from the approved frozen defaults"
        )
    reviewer_map = {row["reviewer_id"]: row for row in reviewers}
    if documents["policy.resolved.json"] != frozen_policy:
        raise DataReviewArtifactError(
            "review run policy differs from config/stage1/blind_review.json"
        )
    policy = _validate_policy(documents["policy.resolved.json"], set(reviewer_map))
    run_meta = documents["run.meta.json"]
    retry_parent = _validate_run_meta(
        run_meta,
        dependency=dependency,
        reviewers=reviewers,
        policy_sha256=frozen_policy_sha256,
    )
    issue_map = {str(issue["issue_id"]): issue for issue in audit_issues}
    consensus = rows["consensus.jsonl"]
    consensus_ids = [row.get("issue_id") for row in consensus]
    if (
        not consensus_ids
        or any(not isinstance(value, str) or value not in issue_map for value in consensus_ids)
        or len(consensus_ids) != len(set(consensus_ids))
    ):
        raise DataReviewArtifactError("consensus issue frame is empty, duplicate, or unknown")
    selected_ids = set(consensus_ids)
    if retry_parent is None:
        scope = run_meta.get("issue_scope")
        if not isinstance(scope, Mapping):
            raise DataReviewArtifactError("initial review run lacks issue_scope")
        kinds = scope.get("issue_kinds")
        if (
            not isinstance(kinds, list)
            or not kinds
            or len(kinds) != len(set(kinds))
            or any(not isinstance(kind, str) or not kind for kind in kinds)
            or scope.get("issue_count") != len(selected_ids)
            or scope.get("scope_policy") not in {"explicit-audit-subset/v1", "complete-audit/v1"}
        ):
            raise DataReviewArtifactError("initial review issue_scope is malformed")
    else:
        kinds = sorted({str(issue_map[issue_id]["issue_kind"]) for issue_id in selected_ids})
    expected_ids = {
        issue_id for issue_id, issue in issue_map.items() if issue.get("issue_kind") in set(kinds)
    }
    if selected_ids != expected_ids:
        raise DataReviewArtifactError("review run omits/adds issues within its frozen audit scope")
    selected_issues = [issue_map[issue_id] for issue_id in sorted(selected_ids)]
    if retry_parent is None:
        expected_order_hash = canonical_sha256(
            [issue_id for issue_id, _ in sorted(
                ((str(issue["issue_id"]), hashlib.sha256(
                    f"{policy['blindness']['stable_shuffle_seed']}:{issue['issue_id']}".encode("utf-8")
                ).hexdigest()) for issue in selected_issues),
                key=lambda item: item[1],
            )]
        )
    else:
        expected_order_hash = canonical_sha256(sorted(selected_ids))
    if run_meta.get("ordered_issue_ids_sha256") != expected_order_hash:
        raise DataReviewArtifactError("ordered issue frame does not match audit/policy")
    blind_map = _blind_map(selected_issues, int(policy["blindness"]["stable_shuffle_seed"]))
    for reviewer_id in REVIEWER_IDS:
        stream = model_rows[reviewer_id]
        stream_ids = [row.get("issue_id") for row in stream]
        if len(stream_ids) != len(set(stream_ids)) or set(stream_ids) != selected_ids:
            raise DataReviewArtifactError("model-review stream is incomplete or duplicate")
        for row in stream:
            issue_id = str(row["issue_id"])
            _validate_model_row(
                row,
                issue=issue_map[issue_id],
                reviewer=reviewer_map[reviewer_id],
                blind_item_id=blind_map[issue_id],
                rubric_meta=rubric_meta,
                policy=policy,
                review_run_id=str(run_meta["review_run_id"]),
                retry_parent=retry_parent,
            )
    expected_consensus, expected_queue, expected_auto = build_consensus(
        selected_issues,
        model_rows,
        low_confidence_threshold=float(policy["consensus"]["low_confidence_threshold"]),
        agreement_qc_fraction=float(policy["consensus"]["agreement_qc_fraction"]),
    )
    consensus_sorted = sorted(consensus, key=lambda row: str(row.get("issue_id")))
    if consensus_sorted != expected_consensus or any(
        row.get("schema_version") != CONSENSUS_SCHEMA for row in consensus_sorted
    ):
        raise DataReviewArtifactError("stored consensus cannot be replayed")
    queue = sorted(rows["human_queue.jsonl"], key=lambda row: str(row.get("issue_id")))
    if queue != expected_queue or any(
        row.get("schema_version") != QUEUE_SCHEMA or row.get("model_votes_hidden") is not True
        for row in queue
    ):
        raise DataReviewArtifactError("stored human queue cannot be replayed safely")
    auto_rows = sorted(rows["auto_adjudication.jsonl"], key=lambda row: str(row.get("issue_id")))
    if len(auto_rows) != len(expected_auto):
        raise DataReviewArtifactError("automatic adjudication frame has the wrong size")
    expected_auto_map = {row["issue_id"]: row for row in expected_auto}
    for row in auto_rows:
        expected = expected_auto_map.get(row.get("issue_id"))
        if expected is None or {key: value for key, value in row.items() if key != "reviewed_at"} != {
            key: value for key, value in expected.items() if key != "reviewed_at"
        }:
            raise DataReviewArtifactError("automatic adjudication cannot be replayed")
        reviewed_at = row.get("reviewed_at")
        try:
            parsed_time = datetime.fromisoformat(str(reviewed_at))
        except ValueError as exc:
            raise DataReviewArtifactError("automatic adjudication timestamp is invalid") from exc
        if parsed_time.tzinfo is None:
            raise DataReviewArtifactError("automatic adjudication timestamp lacks timezone")
    try:
        _validate_completed_rows(
            auto_rows,
            [issue_map[str(row["issue_id"])] for row in auto_rows],
            rubric_meta,
            reviewer_id=PANEL_REVIEWER_ID,
        )
    except Stage1DataError as exc:
        raise DataReviewArtifactError(f"automatic adjudication is invalid: {exc}") from exc
    expected_templates = sorted(
        [
            {
                "schema_version": FINAL_ROW_SCHEMA,
                "data_audit_id": issue_map[packet["issue_id"]]["data_audit_id"],
                "issue_id": packet["issue_id"],
                "issue_kind": packet["issue_kind"],
                "decision": "",
                "edits": [],
                "reason_code": "",
                "reason": "",
                "reviewer_id": PANEL_REVIEWER_ID,
                "reviewed_at": "",
            }
            for packet in queue
        ],
        key=lambda row: row["issue_id"],
    )
    templates = sorted(
        rows["human_adjudication_template.jsonl"], key=lambda row: str(row.get("issue_id"))
    )
    if templates != expected_templates:
        raise DataReviewArtifactError("human-adjudication template cannot be replayed")
    summary = documents["summary.json"]
    retry_summary_fields = (
        {"parent_review_run_id", "retry_counts_by_reviewer", "remaining_errors_by_reviewer"}
        if retry_parent is not None else set()
    )
    expected_summary_keys = {
        "schema_version", "review_run_id", "issue_count", "auto_adjudication_count",
        "human_queue_count", "human_queue_reasons",
    } | retry_summary_fields
    if not isinstance(summary, Mapping) or set(summary) != expected_summary_keys:
        raise DataReviewArtifactError("normalized review summary has a non-canonical shape")
    queue_reasons = policy["consensus"]["human_queue_reasons"]
    expected_reason_counts = {
        reason: sum(reason in packet["queue_reasons"] for packet in queue)
        for reason in queue_reasons
    }
    if (
        summary.get("schema_version") != SOURCE_SUMMARY_SCHEMA
        or summary.get("review_run_id") != run_meta["review_run_id"]
        or summary.get("issue_count") != len(selected_ids)
        or summary.get("auto_adjudication_count") != len(auto_rows)
        or summary.get("human_queue_count") != len(queue)
        or summary.get("human_queue_reasons") != expected_reason_counts
    ):
        raise DataReviewArtifactError("review summary counts/frame cannot be reproduced")
    if retry_parent is not None:
        retry_counts = summary.get("retry_counts_by_reviewer")
        remaining = summary.get("remaining_errors_by_reviewer")
        if (
            summary.get("parent_review_run_id") != retry_parent
            or not isinstance(retry_counts, Mapping)
            or set(retry_counts) != set(REVIEWER_IDS)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in retry_counts.values())
            or remaining != {
                reviewer_id: sum(row.get("error") is not None for row in model_rows[reviewer_id])
                for reviewer_id in REVIEWER_IDS
            }
        ):
            raise DataReviewArtifactError("retry summary lineage/counts are invalid")
    return {
        "source_review_run_id": run_meta["review_run_id"],
        "issue_count": len(selected_ids),
        "auto_adjudication_count": len(auto_rows),
        "human_queue_count": len(queue),
        "reviewer_count": len(REVIEWER_IDS),
    }


def _canonical_source_payloads(snapshot: Mapping[str, Any]) -> dict[str, bytes]:
    payloads = {
        name: canonical_json_bytes(snapshot["documents"][name]) + b"\n"
        for name in JSON_FILES
    }
    payloads.update(
        {
            name: canonical_jsonl_bytes(snapshot["rows"][name], key="issue_id")
            for name in JSONL_FILES
        }
    )
    payloads.update(
        {
            f"model_reviews/{reviewer_id}.jsonl": canonical_jsonl_bytes(
                snapshot["model_rows"][reviewer_id], key="issue_id"
            )
            for reviewer_id in REVIEWER_IDS
        }
    )
    return dict(sorted(payloads.items()))


def _payload_hashes(payloads: Mapping[str, bytes]) -> dict[str, str]:
    return {
        relative: hashlib.sha256(payload).hexdigest()
        for relative, payload in sorted(payloads.items())
    }


def _id_inputs(
    *, source_review_run_id: str, dependency: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> dict[str, Any]:
    return {
        "schema_version": ID_INPUT_SCHEMA,
        "source_review_run_id": source_review_run_id,
        "data_audit_dependency": dict(dependency),
        "canonical_source_files_sha256": _payload_hashes(payloads),
    }


def _provenance(
    *, artifact_id: str, id_inputs: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "data_blind_review_id": artifact_id,
        **dict(id_inputs),
        "source_summary_normalization": {
            "removed_fields": ["target_path"],
            "policy": "remove-source-absolute-target-path/v1",
        },
        "model_reviews_and_raw_provenance_retained": True,
        "model_votes_excluded_from_validation_report": True,
    }


def _assert_canonical_files(target: Path, payloads: Mapping[str, bytes]) -> None:
    for relative, expected in payloads.items():
        if (target / relative).read_bytes() != expected:
            raise DataReviewArtifactError(f"sealed review file is not canonical: {relative}")


def seal_data_review(
    *,
    audit_ref: str | Path,
    review_target: str | Path,
    output_root: str | Path,
    write_ref: str | Path,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Validate and immutably seal one existing data dual-blind review run."""

    source = Path(review_target)
    try:
        locator, audit_target = resolve_locator_ref(audit_ref, expected_kind="data-audit")
        dependency = portable_dependency(locator, audit_target, workspace_root)
    except TrainingArtifactError as exc:
        raise DataReviewArtifactError(f"cannot resolve data-audit dependency: {exc}") from exc
    output = Path(output_root)
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    ref_resolved = Path(write_ref).resolve()
    if source_resolved == output_resolved or source_resolved in output_resolved.parents:
        raise DataReviewArtifactError("output root cannot be inside the source review target")
    if source_resolved == ref_resolved or source_resolved in ref_resolved.parents:
        raise DataReviewArtifactError("locator ref cannot be written inside the source review target")
    before = _inspect_regular_tree(source, _source_file_set())
    try:
        snapshot = _read_snapshot(source, source=True)
        audit_issues, rubric_meta = _load_audit(dependency, audit_target)
        frozen_policy, frozen_policy_sha256, expected_reviewers = _load_frozen_policy(
            workspace_root
        )
        report = _validate_snapshot(
            snapshot,
            dependency=dependency,
            audit_issues=audit_issues,
            rubric_meta=rubric_meta,
            frozen_policy=frozen_policy,
            frozen_policy_sha256=frozen_policy_sha256,
            expected_reviewers=expected_reviewers,
        )
    except (TrainingArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        raise DataReviewArtifactError(f"cannot validate source review: {exc}") from exc
    if source.name != report["source_review_run_id"]:
        raise DataReviewArtifactError("source review directory name differs from review_run_id")
    after = _inspect_regular_tree(source, _source_file_set())
    if before != after:
        raise DataReviewArtifactError("source review changed while it was being validated")
    payloads = _canonical_source_payloads(snapshot)
    id_inputs = _id_inputs(
        source_review_run_id=report["source_review_run_id"],
        dependency=dependency,
        payloads=payloads,
    )
    artifact_id = ARTIFACT_PREFIX + canonical_sha256(id_inputs)
    provenance = _provenance(artifact_id=artifact_id, id_inputs=id_inputs)
    target = output / artifact_id
    if Path(write_ref).is_symlink():
        raise DataReviewArtifactError("locator ref cannot be a symlink")
    staging = new_staging_directory(output, artifact_id)
    try:
        write_canonical_json(staging / "data_audit_ref.json", dependency)
        for relative, payload in payloads.items():
            write_bytes_atomic(staging / relative, payload)
        write_canonical_json(staging / "provenance.json", provenance)
        if before != _inspect_regular_tree(source, _source_file_set()):
            raise DataReviewArtifactError("source review changed before publication")
        payload_manifest_sha256 = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda candidate: validate_data_review_target(
                candidate,
                workspace_root=workspace_root,
                audit_target=audit_target,
                require_directory_name=False,
            ),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    try:
        locator_result = write_locator_ref(
            write_ref,
            artifact_kind=ARTIFACT_KIND,
            artifact_id=artifact_id,
            target=target,
            payload_manifest_sha256=payload_manifest_sha256,
        )
    except TrainingArtifactError as exc:
        raise DataReviewArtifactError(f"cannot publish data blind-review locator: {exc}") from exc
    return locator_result


def validate_data_review_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path,
    audit_target: str | Path | None = None,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    """Replay a sealed data-review target without returning model votes."""

    target = Path(target_dir)
    try:
        payload_manifest_sha256 = validate_payload_manifest(target)
        _inspect_regular_tree(target, _sealed_file_set())
        dependency = validate_dependency_ref(
            load_json(target / "data_audit_ref.json"), expected_kind="data-audit"
        )
        resolved_audit = (
            Path(audit_target).resolve()
            if audit_target is not None
            else resolve_dependency_target(dependency, workspace_root)
        )
        snapshot = _read_snapshot(target, source=False)
        audit_issues, rubric_meta = _load_audit(dependency, resolved_audit)
        frozen_policy, frozen_policy_sha256, expected_reviewers = _load_frozen_policy(
            workspace_root
        )
        frame_report = _validate_snapshot(
            snapshot,
            dependency=dependency,
            audit_issues=audit_issues,
            rubric_meta=rubric_meta,
            frozen_policy=frozen_policy,
            frozen_policy_sha256=frozen_policy_sha256,
            expected_reviewers=expected_reviewers,
        )
        payloads = _canonical_source_payloads(snapshot)
        _assert_canonical_files(target, payloads)
        if (target / "data_audit_ref.json").read_bytes() != canonical_json_bytes(dependency) + b"\n":
            raise DataReviewArtifactError("sealed data-audit dependency is not canonical")
        id_inputs = _id_inputs(
            source_review_run_id=frame_report["source_review_run_id"],
            dependency=dependency,
            payloads=payloads,
        )
        artifact_id = ARTIFACT_PREFIX + canonical_sha256(id_inputs)
        provenance = load_json(target / "provenance.json")
        expected_provenance = _provenance(artifact_id=artifact_id, id_inputs=id_inputs)
        if provenance != expected_provenance:
            raise DataReviewArtifactError("sealed review provenance cannot be reproduced")
        if (target / "provenance.json").read_bytes() != canonical_json_bytes(provenance) + b"\n":
            raise DataReviewArtifactError("sealed review provenance is not canonical")
        if require_directory_name and target.name != artifact_id:
            raise DataReviewArtifactError("sealed review directory differs from content ID")
    except DataReviewArtifactError:
        raise
    except (TrainingArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        raise DataReviewArtifactError(f"cannot validate sealed data review: {exc}") from exc
    return {
        "schema_version": VALIDATION_SCHEMA,
        "valid": True,
        "data_blind_review_id": artifact_id,
        "source_review_run_id": frame_report["source_review_run_id"],
        "data_audit_id": dependency["artifact_id"],
        "issue_count": frame_report["issue_count"],
        "reviewer_count": frame_report["reviewer_count"],
        "auto_adjudication_count": frame_report["auto_adjudication_count"],
        "human_queue_count": frame_report["human_queue_count"],
        "payload_manifest_sha256": payload_manifest_sha256,
    }


def resolve_data_review_ref(
    review_ref: str | Path, *, workspace_root: str | Path
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Resolve and validate a locator, returning no review/vote documents."""

    try:
        locator, target = resolve_locator_ref(review_ref, expected_kind=ARTIFACT_KIND)
    except TrainingArtifactError as exc:
        raise DataReviewArtifactError(f"invalid data blind-review locator: {exc}") from exc
    report = validate_data_review_target(target, workspace_root=workspace_root)
    if (
        report["data_blind_review_id"] != locator["artifact_id"]
        or report["payload_manifest_sha256"] != locator["payload_manifest_sha256"]
    ):
        raise DataReviewArtifactError("locator differs from sealed data-review target")
    return locator, target, report


def validate_data_review_ref(
    review_ref: str | Path, *, workspace_root: str | Path
) -> dict[str, Any]:
    """Validate a data blind-review locator and return its vote-free report."""

    return resolve_data_review_ref(review_ref, workspace_root=workspace_root)[2]


__all__ = [
    "ARTIFACT_KIND",
    "DataReviewArtifactError",
    "resolve_data_review_ref",
    "seal_data_review",
    "validate_data_review_ref",
    "validate_data_review_target",
]
