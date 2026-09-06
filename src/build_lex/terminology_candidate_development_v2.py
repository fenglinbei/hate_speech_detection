"""S2.1b development artifacts layered beside the frozen legacy review.

This module intentionally does not edit or reinterpret the immutable S2.1
legacy frame.  Its first lifecycle freezes the already locked Phase-A mention
annotations as development-only raw gold without requiring decisions for the
superseded 6,306-proposal legacy queue.

No provider execution, network access, S2.2 materialization, or lexicon
publication is implemented here.
"""

from __future__ import annotations

import copy
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from build_lex.terminology_candidate_generators import (
    CandidateGeneratorError,
    resolve_exact_span,
)
from build_lex.terminology_candidate_review import (
    EXPECTED_UNIQUE_RECORDS,
    FRAME_ARTIFACT_KIND,
    PROVISIONAL_ROUTES,
    REASON_CODES,
    REVIEW_POLICY,
    SESSION_SCHEMA_VERSION,
    SHA256_RE,
    CandidateReviewError,
    _canonical_sha,
    _content_sha256,
    _forbidden_key_paths,
    _read_session,
    _target_name_matches_artifact_id,
    validate_development_frame,
)
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_sha256,
    ensure_exact_file_set,
    finalize_target_atomic,
    load_json,
    new_staging_directory,
    sha256_file,
    validate_json_schema,
    validate_payload_manifest,
    write_canonical_json,
    write_locator_ref,
)


PROTOCOL_SCHEMA_VERSION = "wp3-s21b-protocol/v1"
PROTOCOL_REVISION = "wp3-s21b-current-generator-diagnostic/v1"
RAW_GOLD_SCHEMA_VERSION = "wp3-s21-locked-raw-gold/v1"
RAW_GOLD_SCHEMA_ID = (
    "https://local.invalid/schemas/wp3_s21_locked_raw_gold_v1.schema.json"
)
RAW_GOLD_SCHEMA_SHA256 = (
    "cab8287f2510d14c38484ceec218268fc5428e1889ea76957be84b4ae3426bb8"
)
RAW_GOLD_ARTIFACT_KIND = "wp3-s21-locked-raw-gold"
RAW_GOLD_DECLARATION_SCHEMA_VERSION = "wp3-s21-locked-raw-declaration/v1"
LEGACY_PHASE_B_STATUS = "superseded-not-completion-gate"
LEGACY_PROPOSAL_COUNT = 6306
RAW_GOLD_ID_PREFIX = "wp3rawgold-"


class CandidateDevelopmentV2Error(CandidateReviewError):
    """Raised when an S2.1b protocol or artifact fails closed."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateDevelopmentV2Error(f"{label} must be an object")
    return dict(value)


def _nonempty_text(value: Any, label: str, *, maximum: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise CandidateDevelopmentV2Error(f"{label} must be non-empty text")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise CandidateDevelopmentV2Error(f"{label} must be a SHA-256 digest")
    return value


def _is_exact_int(value: Any, expected: int | None = None) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and (expected is None or value == expected)
    )


def load_s21b_protocol(path: str | Path) -> dict[str, Any]:
    """Load the accepted S2.1b protocol and enforce its safety boundaries."""

    try:
        value = load_json(path)
    except TrainingArtifactError as exc:
        raise CandidateDevelopmentV2Error(str(exc)) from exc
    protocol = _object(value, "S2.1b protocol")
    expected_fields = {
        "schema_version",
        "protocol_revision",
        "accepted_date",
        "scope",
        "scientific_eligible",
        "sealed",
        "legacy_frame_dependency",
        "legacy_phase_b",
        "locked_raw_gold",
        "current_generator_pilot",
        "phase_b_v2",
        "execution",
        "s22",
    }
    if set(protocol) != expected_fields:
        raise CandidateDevelopmentV2Error("S2.1b protocol fields are not canonical")
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or protocol.get("protocol_revision") != PROTOCOL_REVISION
        or protocol.get("scope") != "development-only"
        or protocol.get("scientific_eligible") is not False
        or protocol.get("sealed") is not False
    ):
        raise CandidateDevelopmentV2Error("S2.1b protocol identity or scope differs")
    if protocol.get("accepted_date") != "2026-08-30":
        raise CandidateDevelopmentV2Error("S2.1b accepted date differs")

    dependency = _object(
        protocol.get("legacy_frame_dependency"), "legacy_frame_dependency"
    )
    if set(dependency) != {
        "artifact_kind",
        "artifact_id",
        "payload_manifest_sha256",
    }:
        raise CandidateDevelopmentV2Error("legacy frame dependency fields differ")
    if dependency.get("artifact_kind") != FRAME_ARTIFACT_KIND:
        raise CandidateDevelopmentV2Error("legacy frame artifact kind differs")
    frame_id = _nonempty_text(dependency.get("artifact_id"), "legacy frame ID")
    if not frame_id.startswith("wp3devframe-"):
        raise CandidateDevelopmentV2Error("legacy frame ID is invalid")
    _sha(dependency.get("payload_manifest_sha256"), "legacy frame payload hash")

    legacy = _object(protocol.get("legacy_phase_b"), "legacy_phase_b")
    if (
        set(legacy)
        != {
            "status",
            "frozen_proposal_count",
            "required_decision_count",
            "preserve_frame_and_session",
            "permitted_use",
        }
        or legacy.get("status") != LEGACY_PHASE_B_STATUS
        or not _is_exact_int(
            legacy.get("frozen_proposal_count"), LEGACY_PROPOSAL_COUNT
        )
        or not _is_exact_int(legacy.get("required_decision_count"), 0)
        or legacy.get("preserve_frame_and_session") is not True
        or legacy.get("permitted_use")
        != "archival-metrics-and-stratified-error-sampling-only"
    ):
        raise CandidateDevelopmentV2Error("legacy Phase-B supersession differs")

    raw = _object(protocol.get("locked_raw_gold"), "locked_raw_gold")
    if (
        set(raw)
        != {
            "artifact_kind",
            "schema_version",
            "annotation_schema_id",
            "annotation_schema_sha256",
            "required_case_count",
            "gold_scope",
            "provisional_routes_are_formal_tiers",
            "allowed_post_reveal_raw_amendment_count",
            "write_back_to_legacy_session",
        }
        or raw.get("artifact_kind") != RAW_GOLD_ARTIFACT_KIND
        or raw.get("schema_version") != RAW_GOLD_SCHEMA_VERSION
        or raw.get("annotation_schema_id") != RAW_GOLD_SCHEMA_ID
        or raw.get("annotation_schema_sha256") != RAW_GOLD_SCHEMA_SHA256
        or not _is_exact_int(
            raw.get("required_case_count"), EXPECTED_UNIQUE_RECORDS
        )
        or raw.get("gold_scope") != "mention-occurrence-only"
        or raw.get("provisional_routes_are_formal_tiers") is not False
        or not _is_exact_int(
            raw.get("allowed_post_reveal_raw_amendment_count"), 0
        )
        or raw.get("write_back_to_legacy_session") is not False
    ):
        raise CandidateDevelopmentV2Error("locked raw-gold protocol differs")

    pilot = _object(protocol.get("current_generator_pilot"), "current_generator_pilot")
    if (
        set(pilot)
        != {
            "case_count",
            "input_policy",
            "g1",
            "g2",
            "g3",
            "total_logical_model_task_count",
            "full_fit_execution_authorized",
            "run_artifact_status",
        }
        or not _is_exact_int(pilot.get("case_count"), EXPECTED_UNIQUE_RECORDS)
        or pilot.get("input_policy")
        != "formal-fit-content-only-no-gold-fields/v1"
        or not _is_exact_int(pilot.get("total_logical_model_task_count"), 1696)
        or pilot.get("full_fit_execution_authorized") is not False
        or pilot.get("run_artifact_status") != "not-implemented"
    ):
        raise CandidateDevelopmentV2Error("current-generator pilot boundary differs")
    g1 = _object(pilot.get("g1"), "current_generator_pilot.g1")
    g2 = _object(pilot.get("g2"), "current_generator_pilot.g2")
    g3 = _object(pilot.get("g3"), "current_generator_pilot.g3")
    if (
        set(g1) != {"passes", "providers", "logical_task_count"}
        or g1.get("passes") != ["surface_decode", "lexical_pragmatic"]
        or g1.get("providers") != ["qwen:Qwen3.8-27B"]
        or not _is_exact_int(g1.get("logical_task_count"), 848)
        or set(g2) != {"providers", "aggregation", "logical_task_count"}
        or g2.get("providers")
        != [
            "qwen:Qwen3.8-27B",
            "deepseek:deepseek-v4-flash",
        ]
        or g2.get("aggregation") != "provider-union-no-vote/v1"
        or not _is_exact_int(g2.get("logical_task_count"), 848)
        or set(g3) != {"execution", "logical_model_task_count"}
        or g3.get("execution") != "offline-deterministic"
        or not _is_exact_int(g3.get("logical_model_task_count"), 0)
    ):
        raise CandidateDevelopmentV2Error("current-generator provider plan differs")

    phase_b = _object(protocol.get("phase_b_v2"), "phase_b_v2")
    if (
        set(phase_b)
        != {
            "status",
            "queue_policy",
            "required_census",
            "sampled_population",
            "forbidden_completion_sources",
            "human_visible_provenance_fields",
            "required_final_defer_count",
        }
        or phase_b.get("status")
        != "blocked-until-current-generator-run-artifact"
        or phase_b.get("queue_policy")
        != "gold-relation-census-plus-seeded-stratified-fp/v1"
        or phase_b.get("required_census")
        != [
            "missed_gold",
            "boundary_overlap_errors",
            "generator_unique_exact_true_positives",
        ]
        or phase_b.get("sampled_population")
        != "disjoint-no-overlap-false-positives"
        or phase_b.get("forbidden_completion_sources")
        != ["historical_a1", "historical_dual_model"]
        or phase_b.get("human_visible_provenance_fields") != []
        or not _is_exact_int(phase_b.get("required_final_defer_count"), 0)
    ):
        raise CandidateDevelopmentV2Error("Phase-B-v2 review boundary differs")

    execution = _object(protocol.get("execution"), "execution")
    if set(execution) != {
        "model_calls_authorized",
        "network_calls_authorized",
        "paid_calls_authorized",
        "formal_artifact_publication_authorized",
    } or any(value is not False for value in execution.values()):
        raise CandidateDevelopmentV2Error("S2.1b execution must remain unauthorized")
    s22 = _object(protocol.get("s22"), "s22")
    if (
        set(s22)
        != {
            "status",
            "must_not_overlap_s21",
            "requires_frozen_current_generators",
            "model_call_authorization_required",
        }
        or s22.get("status") != "deferred-not-implemented"
        or s22.get("must_not_overlap_s21") is not True
        or s22.get("requires_frozen_current_generators") is not True
        or s22.get("model_call_authorization_required") is not True
    ):
        raise CandidateDevelopmentV2Error("S2.2 boundary differs")
    if _forbidden_key_paths(protocol):
        raise CandidateDevelopmentV2Error("S2.1b protocol contains a forbidden task field")
    return protocol


def _validate_annotation_schema_binding(
    protocol: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    raw = _object(protocol.get("locked_raw_gold"), "locked_raw_gold")
    if (
        schema.get("$id") != RAW_GOLD_SCHEMA_ID
        or raw.get("annotation_schema_id") != RAW_GOLD_SCHEMA_ID
        or canonical_sha256(schema) != RAW_GOLD_SCHEMA_SHA256
        or raw.get("annotation_schema_sha256") != canonical_sha256(schema)
    ):
        raise CandidateDevelopmentV2Error(
            "locked raw-gold annotation schema is not protocol-bound"
        )


def _validate_frame_protocol_binding(
    frame: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    expected = {
        "artifact_kind": FRAME_ARTIFACT_KIND,
        "artifact_id": frame["frame_id"],
        "payload_manifest_sha256": frame["payload_manifest_sha256"],
    }
    if protocol.get("legacy_frame_dependency") != expected:
        raise CandidateDevelopmentV2Error(
            "S2.1b protocol is not bound to the selected legacy frame"
        )


def _validate_raw_annotations(
    frame: Mapping[str, Any], raw_annotations: Any
) -> dict[str, int]:
    raw = _object(raw_annotations, "raw_annotations")
    case_by_id = {str(row["case_id"]): row for row in frame["cases"]}
    if set(raw) != set(case_by_id) or len(raw) != EXPECTED_UNIQUE_RECORDS:
        raise CandidateDevelopmentV2Error("raw annotation coverage differs")
    mention_count = 0
    empty_count = 0
    for case_id, value in raw.items():
        annotation = _object(value, f"raw annotation {case_id}")
        if set(annotation) != {
            "status",
            "content_sha256",
            "needs_explanation",
            "mentions",
            "notes",
        }:
            raise CandidateDevelopmentV2Error("raw annotation fields are not canonical")
        content = str(case_by_id[case_id]["content"])
        mentions = annotation.get("mentions")
        if (
            annotation.get("status") != "confirmed"
            or annotation.get("content_sha256") != _content_sha256(content)
            or not isinstance(annotation.get("needs_explanation"), bool)
            or not isinstance(mentions, list)
            or annotation.get("needs_explanation") != bool(mentions)
            or not isinstance(annotation.get("notes"), str)
            or len(annotation["notes"]) > 4000
        ):
            raise CandidateDevelopmentV2Error("raw annotation state or content differs")
        if not mentions:
            empty_count += 1
        seen: set[tuple[int, int, str]] = set()
        normalized_order: list[tuple[int, int, str]] = []
        for value in mentions:
            mention = _object(value, f"raw mention in {case_id}")
            if set(mention) != {
                "mention_id",
                "surface",
                "occurrence_ordinal",
                "start",
                "end",
                "provisional_route",
                "reason_codes",
                "notes",
            }:
                raise CandidateDevelopmentV2Error("raw mention fields are not canonical")
            surface = _nonempty_text(mention.get("surface"), "mention surface", maximum=80)
            if surface != surface.strip():
                raise CandidateDevelopmentV2Error(
                    "mention surface cannot have leading or trailing whitespace"
                )
            ordinal = mention.get("occurrence_ordinal")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
                raise CandidateDevelopmentV2Error("mention occurrence ordinal is invalid")
            try:
                start, end = resolve_exact_span(content, surface, ordinal)
            except CandidateGeneratorError as exc:
                raise CandidateDevelopmentV2Error("raw mention cannot replay") from exc
            identity = (start, end, surface)
            mention_identity = {
                "surface": surface,
                "occurrence_ordinal": ordinal,
                "start": start,
                "end": end,
            }
            reasons = mention.get("reason_codes")
            notes = mention.get("notes")
            if (
                (mention.get("start"), mention.get("end")) != (start, end)
                or identity in seen
                or mention.get("mention_id")
                != "s21mention-" + _canonical_sha(mention_identity)[:32]
                or mention.get("provisional_route") not in PROVISIONAL_ROUTES
                or not isinstance(reasons, list)
                or not reasons
                or len(reasons) != len(set(reasons))
                or any(reason not in REASON_CODES for reason in reasons)
                or not isinstance(notes, str)
                or len(notes) > 2000
                or ("other" in reasons and not notes.strip())
            ):
                raise CandidateDevelopmentV2Error("raw mention identity or route differs")
            seen.add(identity)
            normalized_order.append(identity)
        if normalized_order != sorted(normalized_order):
            raise CandidateDevelopmentV2Error("raw mentions are not canonically ordered")
        mention_count += len(mentions)
    if _forbidden_key_paths(raw):
        raise CandidateDevelopmentV2Error("locked raw gold contains a forbidden task field")
    return {
        "case_count": len(raw),
        "mention_count": mention_count,
        "empty_record_count": empty_count,
    }


def _validate_raw_lock(raw_lock: Any, raw_annotations: Mapping[str, Any]) -> None:
    lock = _object(raw_lock, "raw_lock")
    expected_fields = {"policy", "raw_annotations_sha256", "locked_at"}
    if (
        set(lock) != expected_fields
        or lock.get("policy") != REVIEW_POLICY
        or lock.get("raw_annotations_sha256") != _canonical_sha(raw_annotations)
        or not isinstance(lock.get("locked_at"), str)
        or not lock["locked_at"]
    ):
        raise CandidateDevelopmentV2Error("raw lock does not bind the annotations")


def _validate_raw_amendments(
    frame: Mapping[str, Any], amendments: Any, raw_lock: Mapping[str, Any]
) -> dict[str, int]:
    if not isinstance(amendments, list):
        raise CandidateDevelopmentV2Error("raw amendments must be an array")
    case_ids = {str(row["case_id"]) for row in frame["cases"]}
    post_reveal_count = 0
    for value in amendments:
        amendment = _object(value, "raw amendment")
        if set(amendment) != {
            "scope",
            "case_id",
            "proposal_id",
            "reason",
            "post_reveal_raw_amendment",
            "reopened_at",
            "prior_revision",
        }:
            raise CandidateDevelopmentV2Error("raw amendment fields are not canonical")
        if (
            amendment.get("scope") != "raw"
            or amendment.get("case_id") not in case_ids
            or amendment.get("proposal_id") is not None
            or amendment.get("post_reveal_raw_amendment") is not False
            or not isinstance(amendment.get("reopened_at"), str)
            or not amendment["reopened_at"]
        ):
            raise CandidateDevelopmentV2Error("raw amendment identity differs")
        _nonempty_text(amendment.get("reason"), "raw amendment reason", maximum=1000)
        _sha(amendment.get("prior_revision"), "raw amendment prior revision")
        post_reveal_count += int(amendment["post_reveal_raw_amendment"])
    if set(raw_lock) != {"policy", "raw_annotations_sha256", "locked_at"}:
        raise CandidateDevelopmentV2Error("amended raw locks cannot be frozen as v1")
    if post_reveal_count:
        raise CandidateDevelopmentV2Error(
            "post-reveal raw amendments cannot be frozen as v1"
        )
    return {
        "raw_amendment_count": len(amendments),
        "post_reveal_raw_amendment_count": post_reveal_count,
    }


def _legacy_proposal_keys(frame: Mapping[str, Any]) -> set[str]:
    proposal_rows = frame.get("proposal_rows")
    if not isinstance(proposal_rows, list):
        raise CandidateDevelopmentV2Error("legacy proposal rows are unavailable")
    keys = {
        f"{row['case_id']}:{proposal['proposal_id']}"
        for row in proposal_rows
        for proposal in row["proposals"]
    }
    if len(keys) != LEGACY_PROPOSAL_COUNT:
        raise CandidateDevelopmentV2Error("legacy proposal count differs")
    return keys


def _project_and_validate_raw_amendments(
    frame: Mapping[str, Any], amendments: Any
) -> list[dict[str, Any]]:
    if not isinstance(amendments, list):
        raise CandidateDevelopmentV2Error("session amendments must be an array")
    case_ids = {str(row["case_id"]) for row in frame["cases"]}
    proposal_keys = _legacy_proposal_keys(frame)
    raw_amendments: list[dict[str, Any]] = []
    expected_fields = {
        "scope",
        "case_id",
        "proposal_id",
        "reason",
        "post_reveal_raw_amendment",
        "reopened_at",
        "prior_revision",
    }
    for value in amendments:
        amendment = _object(value, "session amendment")
        if set(amendment) != expected_fields:
            raise CandidateDevelopmentV2Error(
                "session amendment fields are not canonical"
            )
        scope = amendment.get("scope")
        case_id = amendment.get("case_id")
        proposal_id = amendment.get("proposal_id")
        if (
            scope not in {"raw", "diagnostic"}
            or case_id not in case_ids
            or amendment.get("post_reveal_raw_amendment") is not False
            or not isinstance(amendment.get("reopened_at"), str)
            or not amendment["reopened_at"]
        ):
            raise CandidateDevelopmentV2Error("session amendment identity differs")
        _nonempty_text(amendment.get("reason"), "amendment reason", maximum=1000)
        _sha(amendment.get("prior_revision"), "amendment prior revision")
        if scope == "raw":
            if proposal_id is not None:
                raise CandidateDevelopmentV2Error(
                    "raw amendment cannot name a proposal"
                )
            raw_amendments.append(copy.deepcopy(amendment))
        elif (
            not isinstance(proposal_id, str)
            or f"{case_id}:{proposal_id}" not in proposal_keys
        ):
            raise CandidateDevelopmentV2Error(
                "diagnostic amendment proposal identity differs"
            )
    return raw_amendments


def validate_locked_raw_session(
    frame: Mapping[str, Any],
    session: Mapping[str, Any],
    *,
    legacy_decision_count_override: int | None = None,
) -> dict[str, int]:
    """Validate only the locked Phase-A state, never legacy proposal decisions."""

    if (
        session.get("schema_version") != SESSION_SCHEMA_VERSION
        or session.get("frame_id") != frame["frame_id"]
        or session.get("frame_payload_manifest_sha256")
        != frame["payload_manifest_sha256"]
        or session.get("phase") != "diagnostic"
        or not isinstance(session.get("raw_lock"), Mapping)
        or session.get("finalized_gold_id") is not None
    ):
        raise CandidateDevelopmentV2Error(
            "session is not an unfinalized locked legacy review"
        )
    _nonempty_text(session.get("reviewer_id"), "reviewer_id", maximum=100)
    _sha(session.get("revision"), "source session revision")
    _nonempty_text(session.get("updated_at"), "source session updated_at", maximum=100)
    raw_annotations = _object(session.get("raw_annotations"), "raw_annotations")
    raw_summary = _validate_raw_annotations(frame, raw_annotations)
    _validate_raw_lock(session.get("raw_lock"), raw_annotations)
    raw_amendments = _project_and_validate_raw_amendments(
        frame, session.get("amendments")
    )
    amendment_summary = _validate_raw_amendments(
        frame, raw_amendments, session["raw_lock"]
    )
    if legacy_decision_count_override is None:
        decisions = session.get("diagnostic_decisions")
        if not isinstance(decisions, Mapping):
            raise CandidateDevelopmentV2Error(
                "legacy diagnostic decisions must be an object"
            )
        proposal_keys = _legacy_proposal_keys(frame)
        if (
            len(decisions) > LEGACY_PROPOSAL_COUNT
            or any(not isinstance(key, str) for key in decisions)
            or not set(decisions).issubset(proposal_keys)
            or any(not isinstance(row, Mapping) for row in decisions.values())
        ):
            raise CandidateDevelopmentV2Error(
                "legacy diagnostic decision identities differ"
            )
        legacy_decision_count = len(decisions)
    else:
        if "diagnostic_decisions" in session:
            raise CandidateDevelopmentV2Error(
                "legacy diagnostic decision projection includes decision bodies"
            )
        if (
            isinstance(legacy_decision_count_override, bool)
            or not isinstance(legacy_decision_count_override, int)
            or not 0 <= legacy_decision_count_override <= LEGACY_PROPOSAL_COUNT
        ):
            raise CandidateDevelopmentV2Error(
                "legacy diagnostic decision count differs"
            )
        legacy_decision_count = legacy_decision_count_override
    return raw_summary | amendment_summary | {
        "legacy_diagnostic_decision_count": legacy_decision_count
    }


def _artifact_annotations(
    *, frame: Mapping[str, Any], session: Mapping[str, Any]
) -> dict[str, Any]:
    raw_amendments = _project_and_validate_raw_amendments(
        frame, session["amendments"]
    )
    return {
        "schema_version": RAW_GOLD_SCHEMA_VERSION,
        "protocol_revision": PROTOCOL_REVISION,
        "frame_id": frame["frame_id"],
        "reviewer_id": session["reviewer_id"],
        "scope": "development-only",
        "scientific_eligible": False,
        "sealed": False,
        "raw_annotations": copy.deepcopy(session["raw_annotations"]),
        "raw_lock": copy.deepcopy(session["raw_lock"]),
        "raw_amendments": raw_amendments,
        "source_session": {
            "schema_version": SESSION_SCHEMA_VERSION,
            "frame_payload_manifest_sha256": frame["payload_manifest_sha256"],
            "revision": session["revision"],
            "phase": session["phase"],
            "updated_at": session["updated_at"],
            "legacy_proposals_revealed": True,
            "legacy_diagnostic_decision_count": len(
                session["diagnostic_decisions"]
            ),
            "legacy_finalized_gold_id": session["finalized_gold_id"],
        },
    }


def _artifact_declaration(
    annotations: Mapping[str, Any], *, post_reveal_raw_amendment_count: int
) -> dict[str, Any]:
    return {
        "schema_version": RAW_GOLD_DECLARATION_SCHEMA_VERSION,
        "protocol_revision": PROTOCOL_REVISION,
        "frame_id": annotations["frame_id"],
        "reviewer_id": annotations["reviewer_id"],
        "source_session_revision": annotations["source_session"]["revision"],
        "attestations": {
            "raw_annotations_completed_before_legacy_proposal_reveal": True,
            "task_annotations_were_not_used": True,
            "provisional_routes_are_not_formal_tiers": True,
            "legacy_proposals_are_not_current_generator_evidence": True,
            "legacy_phase_b_is_not_a_completion_gate": True,
            "current_generator_diagnostic_is_pending": True,
            "development_only_nonsealed_nonscientific": True,
        },
        "post_reveal_raw_amendment_count": post_reveal_raw_amendment_count,
        "annotation_sha256": canonical_sha256(annotations),
        "frozen_from_session_updated_at": annotations["source_session"][
            "updated_at"
        ],
    }


def _harden_artifact_modes(target: Path) -> None:
    os.chmod(target, 0o700)
    for path in target.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o700)
        elif path.is_file():
            os.chmod(path, 0o600)


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _validate_publication_paths(
    *,
    frame_dir: str | Path,
    session_file: Path,
    target: Path,
    write_ref: str | Path | None,
) -> None:
    frame_target = Path(frame_dir).resolve()
    session_target = session_file.resolve()
    target_resolved = target.resolve()
    if target.is_symlink():
        raise CandidateDevelopmentV2Error("raw-gold target cannot be a symlink")
    if (
        target_resolved == session_target
        or _path_is_within(target_resolved, frame_target)
        or _path_is_within(frame_target, target_resolved)
    ):
        raise CandidateDevelopmentV2Error(
            "raw-gold output would modify a protected legacy path"
        )
    if write_ref is None:
        return
    ref_path = Path(write_ref)
    ref_target = ref_path.resolve()
    if ref_path.is_symlink():
        raise CandidateDevelopmentV2Error("raw-gold ref cannot be a symlink")
    if (
        ref_target == session_target
        or _path_is_within(ref_target, frame_target)
        or _path_is_within(ref_target, target_resolved)
    ):
        raise CandidateDevelopmentV2Error(
            "raw-gold ref would modify a protected session or artifact path"
        )
    if ref_path.exists():
        try:
            existing_ref = _object(load_json(ref_path), "existing locator ref")
        except TrainingArtifactError as exc:
            raise CandidateDevelopmentV2Error(str(exc)) from exc
        if existing_ref.get("artifact_kind") != RAW_GOLD_ARTIFACT_KIND:
            raise CandidateDevelopmentV2Error(
                "raw-gold ref cannot replace a different artifact locator"
            )


def _validate_locked_raw_payload(
    target: Path,
    *,
    frame: Mapping[str, Any],
    expected_protocol: Mapping[str, Any] | None,
    require_current_validator: bool,
) -> dict[str, Any]:
    try:
        manifest = _object(load_json(target / "manifest.json"), "raw-gold manifest")
        annotations = _object(
            load_json(target / "annotations.json"), "raw-gold annotations"
        )
        declaration = _object(
            load_json(target / "reviewer_declaration.json"),
            "raw-gold reviewer declaration",
        )
        protocol = _object(load_json(target / "protocol.json"), "embedded protocol")
        schema = _object(load_json(target / "schema.json"), "embedded schema")
    except TrainingArtifactError as exc:
        raise CandidateDevelopmentV2Error(str(exc)) from exc
    load_s21b_protocol(target / "protocol.json")
    _validate_annotation_schema_binding(protocol, schema)
    try:
        validate_json_schema(annotations, target / "schema.json")
    except TrainingArtifactError as exc:
        raise CandidateDevelopmentV2Error(str(exc)) from exc
    if expected_protocol is not None and protocol != expected_protocol:
        raise CandidateDevelopmentV2Error("embedded S2.1b protocol differs")
    _validate_frame_protocol_binding(frame, protocol)

    manifest_fields = {
        "schema_version",
        "artifact_kind",
        "protocol_revision",
        "protocol_sha256",
        "annotation_schema_sha256",
        "frame_dependency",
        "source_session_revision",
        "raw_annotations_sha256",
        "annotations_sha256",
        "declaration_sha256",
        "validator_implementation_sha256",
        "case_count",
        "mention_count",
        "empty_record_count",
        "raw_amendment_count",
        "post_reveal_raw_amendment_count",
        "legacy_diagnostic_decision_count_at_freeze",
        "legacy_phase_b_status",
        "scope",
        "scientific_eligible",
        "sealed",
        "raw_gold_id",
    }
    declaration_fields = {
        "schema_version",
        "protocol_revision",
        "frame_id",
        "reviewer_id",
        "source_session_revision",
        "attestations",
        "post_reveal_raw_amendment_count",
        "annotation_sha256",
        "frozen_from_session_updated_at",
    }
    if set(manifest) != manifest_fields or set(declaration) != declaration_fields:
        raise CandidateDevelopmentV2Error("raw-gold artifact fields are not canonical")
    count_values = {
        key: manifest.get(key)
        for key in {
            "case_count",
            "mention_count",
            "empty_record_count",
            "raw_amendment_count",
            "post_reveal_raw_amendment_count",
            "legacy_diagnostic_decision_count_at_freeze",
        }
    }
    if (
        not _is_exact_int(count_values["case_count"], EXPECTED_UNIQUE_RECORDS)
        or not _is_exact_int(count_values["mention_count"])
        or count_values["mention_count"] < 0
        or not _is_exact_int(count_values["empty_record_count"])
        or not 0
        <= count_values["empty_record_count"]
        <= EXPECTED_UNIQUE_RECORDS
        or not _is_exact_int(count_values["raw_amendment_count"])
        or count_values["raw_amendment_count"] < 0
        or not _is_exact_int(count_values["post_reveal_raw_amendment_count"], 0)
        or not _is_exact_int(
            count_values["legacy_diagnostic_decision_count_at_freeze"]
        )
        or not 0
        <= count_values["legacy_diagnostic_decision_count_at_freeze"]
        <= LEGACY_PROPOSAL_COUNT
        or not _is_exact_int(declaration.get("post_reveal_raw_amendment_count"), 0)
    ):
        raise CandidateDevelopmentV2Error(
            "raw-gold manifest or declaration counts are not canonical"
        )
    if (
        manifest.get("schema_version") != RAW_GOLD_SCHEMA_VERSION
        or manifest.get("artifact_kind") != RAW_GOLD_ARTIFACT_KIND
        or manifest.get("protocol_revision") != PROTOCOL_REVISION
        or manifest.get("protocol_sha256") != canonical_sha256(protocol)
        or manifest.get("annotation_schema_sha256") != canonical_sha256(schema)
        or manifest.get("source_session_revision")
        != annotations["source_session"]["revision"]
        or manifest.get("raw_annotations_sha256")
        != _canonical_sha(annotations["raw_annotations"])
        or manifest.get("annotations_sha256") != canonical_sha256(annotations)
        or manifest.get("declaration_sha256") != canonical_sha256(declaration)
        or declaration.get("schema_version")
        != RAW_GOLD_DECLARATION_SCHEMA_VERSION
        or declaration.get("protocol_revision") != PROTOCOL_REVISION
        or declaration.get("frame_id") != annotations.get("frame_id")
        or declaration.get("reviewer_id") != annotations.get("reviewer_id")
        or declaration.get("source_session_revision")
        != annotations["source_session"]["revision"]
        or declaration.get("annotation_sha256") != canonical_sha256(annotations)
        or declaration.get("frozen_from_session_updated_at")
        != annotations["source_session"]["updated_at"]
        or manifest.get("legacy_phase_b_status") != LEGACY_PHASE_B_STATUS
        or manifest.get("scope") != "development-only"
        or manifest.get("scientific_eligible") is not False
        or manifest.get("sealed") is not False
    ):
        raise CandidateDevelopmentV2Error("raw-gold artifact bindings differ")
    if require_current_validator and manifest.get(
        "validator_implementation_sha256"
    ) != sha256_file(Path(__file__)):
        raise CandidateDevelopmentV2Error("raw-gold validator implementation drifted")
    _sha(
        manifest.get("validator_implementation_sha256"),
        "validator implementation hash",
    )
    frame_dependency = {
        "artifact_kind": FRAME_ARTIFACT_KIND,
        "artifact_id": frame["frame_id"],
        "payload_manifest_sha256": frame["payload_manifest_sha256"],
    }
    if manifest.get("frame_dependency") != frame_dependency:
        raise CandidateDevelopmentV2Error("raw-gold frame dependency differs")

    session_projection = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "frame_id": annotations["frame_id"],
        "frame_payload_manifest_sha256": annotations["source_session"][
            "frame_payload_manifest_sha256"
        ],
        "reviewer_id": annotations["reviewer_id"],
        "phase": annotations["source_session"]["phase"],
        "raw_annotations": annotations["raw_annotations"],
        "raw_lock": annotations["raw_lock"],
        "amendments": annotations["raw_amendments"],
        "finalized_gold_id": annotations["source_session"][
            "legacy_finalized_gold_id"
        ],
        "updated_at": annotations["source_session"]["updated_at"],
        "revision": annotations["source_session"]["revision"],
    }
    summary = validate_locked_raw_session(
        frame,
        session_projection,
        legacy_decision_count_override=annotations["source_session"][
            "legacy_diagnostic_decision_count"
        ],
    )
    expected_summary = {
        "case_count": manifest.get("case_count"),
        "mention_count": manifest.get("mention_count"),
        "empty_record_count": manifest.get("empty_record_count"),
        "raw_amendment_count": manifest.get("raw_amendment_count"),
        "post_reveal_raw_amendment_count": manifest.get(
            "post_reveal_raw_amendment_count"
        ),
        "legacy_diagnostic_decision_count": manifest.get(
            "legacy_diagnostic_decision_count_at_freeze"
        ),
    }
    if summary != expected_summary:
        raise CandidateDevelopmentV2Error("raw-gold manifest counts differ")
    if declaration.get("post_reveal_raw_amendment_count") != summary[
        "post_reveal_raw_amendment_count"
    ]:
        raise CandidateDevelopmentV2Error("raw-gold declaration amendment count differs")
    expected_attestations = {
        "raw_annotations_completed_before_legacy_proposal_reveal",
        "task_annotations_were_not_used",
        "provisional_routes_are_not_formal_tiers",
        "legacy_proposals_are_not_current_generator_evidence",
        "legacy_phase_b_is_not_a_completion_gate",
        "current_generator_diagnostic_is_pending",
        "development_only_nonsealed_nonscientific",
    }
    attestations = declaration.get("attestations")
    if (
        not isinstance(attestations, Mapping)
        or set(attestations) != expected_attestations
        or any(attestations.get(key) is not True for key in expected_attestations)
    ):
        raise CandidateDevelopmentV2Error("raw-gold declaration is incomplete")
    identity = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "raw_gold_id"
    }
    raw_gold_id = RAW_GOLD_ID_PREFIX + canonical_sha256(identity)
    if manifest.get("raw_gold_id") != raw_gold_id or not _target_name_matches_artifact_id(
        target, raw_gold_id
    ):
        raise CandidateDevelopmentV2Error("raw-gold content-addressed ID differs")
    return {
        "raw_gold_id": raw_gold_id,
        "manifest": manifest,
        "annotations": annotations,
        "declaration": declaration,
        "protocol": protocol,
        "summary": summary,
    }


def build_locked_raw_gold(
    *,
    frame_dir: str | Path,
    session_path: str | Path,
    workspace_root: str | Path,
    legacy_generator_config_path: str | Path,
    protocol_path: str | Path,
    annotation_schema_path: str | Path,
    output_root: str | Path,
    reviewer_id: str,
    write_ref: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze the locked Phase-A annotations without touching the old session."""

    frame = validate_development_frame(
        frame_dir,
        workspace_root=workspace_root,
        generator_config_path=legacy_generator_config_path,
        require_current_sources=True,
    )
    protocol = load_s21b_protocol(protocol_path)
    _validate_frame_protocol_binding(frame, protocol)
    session_file = Path(session_path)
    session_before = session_file.read_bytes()
    session = _read_session(session_file)
    reviewer = _nonempty_text(reviewer_id, "reviewer_id", maximum=100)
    if session.get("reviewer_id") != reviewer:
        raise CandidateDevelopmentV2Error("reviewer does not own the locked session")
    summary = validate_locked_raw_session(frame, session)

    try:
        schema_value = load_json(annotation_schema_path)
        schema = _object(schema_value, "locked raw-gold schema")
    except TrainingArtifactError as exc:
        raise CandidateDevelopmentV2Error(str(exc)) from exc
    _validate_annotation_schema_binding(protocol, schema)
    annotations = _artifact_annotations(frame=frame, session=session)
    try:
        validate_json_schema(annotations, annotation_schema_path)
    except TrainingArtifactError as exc:
        raise CandidateDevelopmentV2Error(str(exc)) from exc
    declaration = _artifact_declaration(
        annotations,
        post_reveal_raw_amendment_count=summary[
            "post_reveal_raw_amendment_count"
        ],
    )
    identity = {
        "schema_version": RAW_GOLD_SCHEMA_VERSION,
        "artifact_kind": RAW_GOLD_ARTIFACT_KIND,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_sha256": canonical_sha256(protocol),
        "annotation_schema_sha256": canonical_sha256(schema),
        "frame_dependency": {
            "artifact_kind": FRAME_ARTIFACT_KIND,
            "artifact_id": frame["frame_id"],
            "payload_manifest_sha256": frame["payload_manifest_sha256"],
        },
        "source_session_revision": session["revision"],
        "raw_annotations_sha256": _canonical_sha(session["raw_annotations"]),
        "annotations_sha256": canonical_sha256(annotations),
        "declaration_sha256": canonical_sha256(declaration),
        "validator_implementation_sha256": sha256_file(Path(__file__)),
        "case_count": summary["case_count"],
        "mention_count": summary["mention_count"],
        "empty_record_count": summary["empty_record_count"],
        "raw_amendment_count": summary["raw_amendment_count"],
        "post_reveal_raw_amendment_count": summary[
            "post_reveal_raw_amendment_count"
        ],
        "legacy_diagnostic_decision_count_at_freeze": summary[
            "legacy_diagnostic_decision_count"
        ],
        "legacy_phase_b_status": LEGACY_PHASE_B_STATUS,
        "scope": "development-only",
        "scientific_eligible": False,
        "sealed": False,
    }
    raw_gold_id = RAW_GOLD_ID_PREFIX + canonical_sha256(identity)
    manifest = {**identity, "raw_gold_id": raw_gold_id}
    output_parent = Path(output_root).resolve()
    target = output_parent / raw_gold_id
    _validate_publication_paths(
        frame_dir=frame_dir,
        session_file=session_file,
        target=target,
        write_ref=write_ref,
    )
    try:
        output_parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            staging = new_staging_directory(output_parent, raw_gold_id)
            try:
                write_canonical_json(staging / "manifest.json", manifest)
                write_canonical_json(staging / "annotations.json", annotations)
                write_canonical_json(
                    staging / "reviewer_declaration.json", declaration
                )
                write_canonical_json(staging / "protocol.json", protocol)
                write_canonical_json(staging / "schema.json", schema)
                payload_hash = finalize_target_atomic(
                    staging,
                    target,
                    validate_staging=lambda staged: _validate_locked_raw_payload(
                        staged,
                        frame=frame,
                        expected_protocol=protocol,
                        require_current_validator=True,
                    ),
                )
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        else:
            payload_hash = validate_payload_manifest(target)
        _harden_artifact_modes(target)
        result = validate_locked_raw_gold(
            target,
            frame_dir=frame_dir,
            workspace_root=workspace_root,
            legacy_generator_config_path=legacy_generator_config_path,
            protocol_path=protocol_path,
            require_current_sources=True,
            require_current_validator=True,
        )
        if result["payload_manifest_sha256"] != payload_hash:
            raise CandidateDevelopmentV2Error(
                "published raw-gold payload hash drifted"
            )
        if write_ref is not None:
            if session_file.read_bytes() != session_before:
                raise CandidateDevelopmentV2Error(
                    "source session changed before raw-gold ref publication"
                )
            write_locator_ref(
                Path(write_ref).resolve(),
                artifact_kind=RAW_GOLD_ARTIFACT_KIND,
                artifact_id=raw_gold_id,
                target=target,
                payload_manifest_sha256=payload_hash,
            )
        return result
    finally:
        if session_file.read_bytes() != session_before:
            raise CandidateDevelopmentV2Error(
                "raw-gold freeze modified the legacy session"
            )


def validate_locked_raw_gold(
    raw_gold_dir: str | Path,
    *,
    frame_dir: str | Path,
    workspace_root: str | Path,
    legacy_generator_config_path: str | Path,
    protocol_path: str | Path | None = None,
    require_current_sources: bool = False,
    require_current_validator: bool = False,
) -> dict[str, Any]:
    """Independently validate one S2.1b locked raw-gold artifact."""

    target = Path(raw_gold_dir).resolve()
    try:
        payload_hash = validate_payload_manifest(target)
        ensure_exact_file_set(
            target,
            {
                "manifest.json",
                "annotations.json",
                "reviewer_declaration.json",
                "protocol.json",
                "schema.json",
                "payload_manifest.json",
            },
        )
    except TrainingArtifactError as exc:
        raise CandidateDevelopmentV2Error(str(exc)) from exc
    frame = validate_development_frame(
        frame_dir,
        workspace_root=workspace_root,
        generator_config_path=legacy_generator_config_path,
        require_current_sources=require_current_sources,
    )
    expected_protocol = (
        load_s21b_protocol(protocol_path) if protocol_path is not None else None
    )
    result = _validate_locked_raw_payload(
        target,
        frame=frame,
        expected_protocol=expected_protocol,
        require_current_validator=require_current_validator,
    )
    result["payload_manifest_sha256"] = payload_hash
    result["target"] = str(target)
    return result


__all__ = [
    "CandidateDevelopmentV2Error",
    "LEGACY_PHASE_B_STATUS",
    "PROTOCOL_REVISION",
    "RAW_GOLD_ARTIFACT_KIND",
    "RAW_GOLD_SCHEMA_VERSION",
    "build_locked_raw_gold",
    "load_s21b_protocol",
    "validate_locked_raw_gold",
    "validate_locked_raw_session",
]
