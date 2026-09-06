"""Offline WP3 S2.1 development-review lifecycle.

The lifecycle deliberately stops before any new model execution and before the
S2.2 sealed benchmark.  It rebuilds the selected A1/dual-model records from the
formal fit partition, presents a raw-only first pass, and only reveals
unattributed historical/G3 proposals after every raw annotation is confirmed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_lex.terminology_candidate_generators import (
    CandidateGeneratorError,
    generate_g3_observations,
    load_generator_config,
    resolve_exact_span,
)
from build_lex.train_only import FrozenTrainInput, resolve_train_input
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_json_bytes,
    canonical_sha256,
    ensure_exact_file_set,
    finalize_target_atomic,
    load_json,
    new_staging_directory,
    portable_dependency,
    resolve_dependency_target,
    resolve_locator_ref,
    sha256_file,
    validate_payload_manifest,
    write_canonical_json,
    write_locator_ref,
)


FRAME_SCHEMA_VERSION = "wp3-s21-development-frame/v1"
PRIVATE_MAP_SCHEMA_VERSION = "wp3-s21-private-source-map/v1"
SESSION_SCHEMA_VERSION = "wp3-s21-review-session/v1"
GOLD_SCHEMA_VERSION = "wp3-s21-development-gold/v1"
DECLARATION_SCHEMA_VERSION = "wp3-s21-reviewer-declaration/v1"
REPORT_SCHEMA_VERSION = "wp3-s21-development-report/v1"
FRAME_ARTIFACT_KIND = "wp3-s21-development-frame"
GOLD_ARTIFACT_KIND = "wp3-s21-development-gold"
REPORT_ARTIFACT_KIND = "wp3-s21-development-report"
SOURCE_POLICY = "fit-content-only-no-task-fields/v1"
SELECTION_POLICY = "historical-a1-200-plus-dm-primary-240-dedup/v1"
LEGACY_DM_PROJECTION = "legacy-surface-to-all-exact-occurrences/v1"
REVIEW_POLICY = "raw-lock-before-unattributed-proposal-reveal/v1"
EXPECTED_A1_RECORDS = 200
EXPECTED_DM_RECORDS = 240
EXPECTED_INTERSECTION = 16
EXPECTED_UNIQUE_RECORDS = 424
SELECTION_SEED = "wp3-s21-development-frame-424/v1"
ANNOTATION_SCHEMA_PATHS = (
    "schemas/wp3_s21_raw_annotation_v1.schema.json",
    "schemas/wp3_s21_diagnostic_decision_v1.schema.json",
)

PROVISIONAL_ROUTES = frozenset(
    {"A_candidate", "B_candidate", "C_candidate"}
)
PROPOSAL_ACTIONS = frozenset(
    {"accept", "trim", "expand", "split", "reject", "defer"}
)
REASON_CODES = frozenset(
    {
        "stable_core_candidate",
        "context_required",
        "evidence_required",
        "transparent",
        "ordinary_identity_or_name",
        "generic_insult",
        "fragment",
        "function_word_attached",
        "sentence_level",
        "wrong_boundary",
        "substring_projection",
        "unsupported_sense",
        "one_off_creation",
        "ambiguous_surface",
        "context_polysemy",
        "quoted_or_reclaimed",
        "evidence_conflict",
        "variant_unresolved",
        "non_contiguous_unresolved",
        "label_derived",
        "not_fit_attested",
        "duplicate",
        "no_neutral_gloss",
        "other",
    }
)
TASK_FIELD_KEYS = frozenset(
    {
        "annotation_count",
        "argument",
        "categories",
        "category",
        "category_counts",
        "category_purity",
        "hate_count",
        "hate_precision",
        "hateful",
        "label",
        "labels",
        "log_odds",
        "non_hate_count",
        "nonhate_penalty",
        "primary_category",
        "quadruples",
        "target",
        "targeted_group",
        "task_prediction",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CandidateReviewError(RuntimeError):
    """Raised when an S2.1 review contract or artifact is invalid."""


class CandidateReviewConflict(CandidateReviewError):
    """Raised when a mutable review session changed concurrently."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_sha(value: Any) -> str:
    return canonical_sha256(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_text(value: Any, name: str, *, maximum: int = 2000) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
    ):
        raise CandidateReviewError(f"{name} must be non-empty trimmed text")
    return value


def _optional_text(value: Any, name: str, *, maximum: int = 4000) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise CandidateReviewError(f"{name} must be text no longer than {maximum}")
    return value


def _forbidden_key_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child_value in value.items():
            child = (*path, str(key))
            if str(key).casefold() in TASK_FIELD_KEYS:
                found.append(".".join(child))
            found.extend(_forbidden_key_paths(child_value, child))
    elif isinstance(value, list):
        for index, child_value in enumerate(value):
            found.extend(_forbidden_key_paths(child_value, (*path, str(index))))
    return found


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _tree_sha256(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise CandidateReviewError(f"source package is not a regular directory: {root}")
    rows: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise CandidateReviewError(f"source package contains a symlink: {item}")
        if item.is_file():
            rows.append(
                {
                    "path": item.relative_to(root).as_posix(),
                    "size": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    return _canonical_sha(rows)


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise CandidateReviewError(f"{label} must be a JSON object")
    return value


def _load_json_array(path: str | Path, label: str) -> list[dict[str, Any]]:
    value = load_json(path)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise CandidateReviewError(f"{label} must be an array of objects")
    return [dict(row) for row in value]


def _load_js_assignment(path: str | Path, label: str) -> dict[str, Any]:
    try:
        wire = Path(path).read_text(encoding="utf-8")
        start = wire.index("{")
        end = wire.rindex("}") + 1
        value = json.loads(wire[start:end])
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise CandidateReviewError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateReviewError(f"{label} must contain an object")
    return value


def _private_source_map_path(frame_dir: str | Path) -> Path:
    target = Path(frame_dir).resolve()
    return target.parent / ".private" / f"{target.name}.source_map.json"


def _target_name_matches_artifact_id(target: Path, artifact_id: str) -> bool:
    """Accept the final directory and the lifecycle helper's private staging name."""

    return target.name == artifact_id or target.name.startswith(f".{artifact_id}.")


def _write_private_source_map(frame_dir: Path, source_map: Mapping[str, Any]) -> Path:
    path = _private_source_map_path(frame_dir)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists():
        if load_json(path) != source_map:
            raise CandidateReviewError("existing private source map differs")
    else:
        write_canonical_json(path, source_map)
    os.chmod(path, 0o600)
    return path


def _load_private_source_map(frame_dir: str | Path) -> dict[str, Any]:
    path = _private_source_map_path(frame_dir)
    if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o077:
        raise CandidateReviewError("private source map is missing or not owner-only")
    return _load_json_object(path, "private source map")


def _source_projection(
    *,
    path: Path,
    package_id: str,
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "package_id": package_id,
        "tree_sha256": _tree_sha256(path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _source_case_map(
    *,
    a1_package: Path,
    a1_source_map_path: Path,
    dm_package: Path,
) -> tuple[
    dict[str, set[str]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    a1_manifest = _load_json_object(a1_package / "manifest.json", "A1 manifest")
    if (
        a1_manifest.get("schema_version") != "terminology-span-review-package/v1"
        or a1_manifest.get("case_count") != EXPECTED_A1_RECORDS
        or a1_manifest.get("mode") != "pilot-tune"
    ):
        raise CandidateReviewError("A1 package is not the frozen 200-case tune package")
    a1_cases = _load_json_array(a1_package / "cases.json", "A1 cases")
    a1_map = _load_json_array(a1_source_map_path, "A1 private source map")
    a1_case_by_id = {str(row.get("case_id")): row for row in a1_cases}
    a1_record_ids: set[str] = set()
    a1_proposals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_tags: dict[str, set[str]] = defaultdict(set)
    for mapping in a1_map:
        case_id = str(mapping.get("case_id", ""))
        record_id = str(mapping.get("record_id", ""))
        case = a1_case_by_id.get(case_id)
        if not case_id or not record_id or case is None:
            raise CandidateReviewError("A1 case/source mapping is incomplete")
        a1_record_ids.add(record_id)
        source_tags[record_id].add("historical_a1_200")
        proposals = case.get("proposals")
        if not isinstance(proposals, list):
            raise CandidateReviewError("A1 proposals are malformed")
        for proposal in proposals:
            if not isinstance(proposal, Mapping):
                raise CandidateReviewError("A1 proposal is malformed")
            a1_proposals[record_id].append(dict(proposal))
    if len(a1_record_ids) != EXPECTED_A1_RECORDS:
        raise CandidateReviewError("A1 unique record count differs")

    dm_manifest = _load_json_object(dm_package / "manifest.json", "dual-model manifest")
    if (
        dm_manifest.get("schema_version") != "dual-model-span-review-package/v1"
        or dm_manifest.get("unique_record_count") != EXPECTED_DM_RECORDS
        or dm_manifest.get("case_count") != 288
        or dm_manifest.get("hidden_repeat_count") != 48
    ):
        raise CandidateReviewError("dual-model package counts differ")
    dm_payload = _load_js_assignment(dm_package / "cases.js", "dual-model cases")
    dm_cases = dm_payload.get("cases")
    if not isinstance(dm_cases, list) or len(dm_cases) != 288:
        raise CandidateReviewError("dual-model cases are malformed")
    dm_case_by_id = {
        str(row.get("case_id")): row
        for row in dm_cases
        if isinstance(row, Mapping)
    }
    dm_map_payload = _load_json_object(dm_package / "audit/source_map.json", "dual-model source map")
    dm_map = dm_map_payload.get("cases")
    if not isinstance(dm_map, list) or len(dm_map) != 288:
        raise CandidateReviewError("dual-model source map is malformed")
    dm_record_ids: set[str] = set()
    dm_surfaces: dict[str, set[str]] = defaultdict(set)
    for mapping in dm_map:
        if not isinstance(mapping, Mapping):
            raise CandidateReviewError("dual-model source mapping row is malformed")
        case_id = str(mapping.get("case_id", ""))
        record_id = str(mapping.get("source_record_id", ""))
        occurrence = mapping.get("occurrence")
        case = dm_case_by_id.get(case_id)
        if not case_id or not record_id or case is None:
            raise CandidateReviewError("dual-model case/source mapping is incomplete")
        source_tags[record_id].add("historical_dual_model_240")
        if occurrence != 0:
            continue
        dm_record_ids.add(record_id)
        for model_key in ("model_a", "model_b"):
            model = case.get(model_key)
            if not isinstance(model, Mapping):
                raise CandidateReviewError("dual-model normalized output is malformed")
            surfaces = model.get("exact_surfaces")
            if not isinstance(surfaces, list) or any(
                not isinstance(surface, str) or not surface for surface in surfaces
            ):
                raise CandidateReviewError("dual-model exact surfaces are malformed")
            dm_surfaces[record_id].update(surfaces)
    if len(dm_record_ids) != EXPECTED_DM_RECORDS:
        raise CandidateReviewError("dual-model primary unique record count differs")
    if len(a1_record_ids & dm_record_ids) != EXPECTED_INTERSECTION:
        raise CandidateReviewError("historical development source intersection differs")
    if len(a1_record_ids | dm_record_ids) != EXPECTED_UNIQUE_RECORDS:
        raise CandidateReviewError("historical development source union differs")
    source_identity = {
        "a1": _source_projection(
            path=a1_package,
            package_id=str(a1_manifest["package_id"]),
            manifest_path=a1_package / "manifest.json",
        )
        | {"private_source_map_sha256": sha256_file(a1_source_map_path)},
        "dual_model": _source_projection(
            path=dm_package,
            package_id=str(dm_manifest["package_id"]),
            manifest_path=dm_package / "manifest.json",
        ),
        "selection_counts": {
            "a1_unique": len(a1_record_ids),
            "dual_model_unique": len(dm_record_ids),
            "intersection": len(a1_record_ids & dm_record_ids),
            "union": len(a1_record_ids | dm_record_ids),
            "dual_model_repeat_pages_excluded": 48,
        },
        "dual_model_legacy_selection_notice": (
            "Historical dual-model records were category-stratified before the current "
            "content-only protocol; category values are not imported and this frame is "
            "development-only."
        ),
    }
    dm_proposals = {
        record_id: [{"surface": surface} for surface in sorted(surfaces)]
        for record_id, surfaces in dm_surfaces.items()
    }
    return source_tags, dict(a1_proposals), dm_proposals, source_identity


def _find_all_occurrences(content: str, surface: str) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    cursor = 0
    while surface and cursor <= len(content) - len(surface):
        start = content.find(surface, cursor)
        if start < 0:
            break
        positions.append((start, start + len(surface)))
        cursor = start + 1
    return positions


def _proposal_key(value: Mapping[str, Any]) -> tuple[int, int, str]:
    return int(value["start"]), int(value["end"]), str(value["surface"])


def _build_proposals_for_record(
    *,
    record_id: str,
    content: str,
    a1_rows: Sequence[Mapping[str, Any]],
    dm_rows: Sequence[Mapping[str, Any]],
    g3_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    grouped: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for row in a1_rows:
        surface = str(row.get("surface", ""))
        ordinal = row.get("occurrence_ordinal")
        if not surface or isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise CandidateReviewError("A1 proposal lacks an exact occurrence")
        try:
            start, end = resolve_exact_span(content, surface, ordinal)
        except CandidateGeneratorError as exc:
            raise CandidateReviewError(f"A1 proposal cannot replay: {exc}") from exc
        if row.get("start") != start or row.get("end") != end:
            raise CandidateReviewError("A1 proposal offsets differ from formal fit content")
        grouped[(start, end, surface)].add("historical_a1")
    for row in dm_rows:
        surface = str(row.get("surface", ""))
        positions = _find_all_occurrences(content, surface)
        if not positions:
            raise CandidateReviewError("dual-model surface is absent from formal fit content")
        for start, end in positions:
            grouped[(start, end, surface)].add("historical_dual_model")
    g3 = g3_config.get("g3")
    if not isinstance(g3, Mapping):
        raise CandidateReviewError("candidate config lacks G3 settings")
    try:
        g3_rows = generate_g3_observations(
            record_id=record_id,
            content=content,
            reference_rows=(),
            romanizer=None,
            rule_version=str(g3["rule_version"]),
            max_surface_chars=int(g3["max_surface_chars"]),
            mixed_script_max_chars=int(g3["mixed_script_max_chars"]),
            phonetic_max_distance=int(g3["phonetic_max_distance"]),
        )
    except (CandidateGeneratorError, KeyError, TypeError, ValueError) as exc:
        raise CandidateReviewError(
            f"G3 development proposals failed for {record_id}: {exc}"
        ) from exc
    for row in g3_rows:
        grouped[_proposal_key(row)].add("g3_form_rule")

    proposals: list[dict[str, Any]] = []
    sources_by_id: dict[str, list[str]] = {}
    for (start, end, surface), sources in sorted(grouped.items()):
        ordinal = _find_all_occurrences(content, surface).index((start, end)) + 1
        identity = {
            "record_id": record_id,
            "content_sha256": _content_sha256(content),
            "surface": surface,
            "start": start,
            "end": end,
            "occurrence_ordinal": ordinal,
        }
        proposal_id = "s21prop-" + _canonical_sha(identity)[:32]
        proposals.append(
            {
                "proposal_id": proposal_id,
                "surface": surface,
                "occurrence_ordinal": ordinal,
                "start": start,
                "end": end,
            }
        )
        sources_by_id[proposal_id] = sorted(sources)
    return proposals, sources_by_id


def build_development_frame(
    *,
    workspace_root: str | Path,
    data_ref: str | Path,
    train_partition_ref: str | Path,
    generator_config_path: str | Path,
    a1_package: str | Path,
    a1_source_map: str | Path,
    dual_model_package: str | Path,
    output_root: str | Path,
    write_ref: str | Path | None = None,
) -> dict[str, Any]:
    """Build the immutable 424-record S2.1 development frame."""

    root = Path(workspace_root).resolve()
    generator_config_file = Path(generator_config_path).resolve()
    config = load_generator_config(generator_config_file, workspace_root=root)
    execution = config.get("execution")
    if not isinstance(execution, Mapping) or any(
        execution.get(key) is not False
        for key in (
            "model_calls_authorized",
            "network_calls_authorized",
            "paid_calls_authorized",
            "formal_artifact_publication_authorized",
        )
    ):
        raise CandidateReviewError(
            "S2.1 requires model/network/publication authorization to remain disabled"
        )
    handbook_path = root / str(config["handbook"]["path"])
    implementation_path = Path(__file__).resolve()
    candidate_implementation = root / "src/build_lex/terminology_candidate_generators.py"
    frozen = resolve_train_input(
        data_ref=data_ref,
        train_partition_ref=train_partition_ref,
        formal=True,
        workspace_root=root,
    )
    if len(frozen.records) != int(config["expected_fit_count"]):
        raise CandidateReviewError("formal fit partition count differs from candidate config")
    record_by_id = {str(row.get("id")): row for row in frozen.records}
    if len(record_by_id) != len(frozen.records):
        raise CandidateReviewError("formal fit partition record IDs are not unique")

    a1_dir = Path(a1_package).resolve()
    dm_dir = Path(dual_model_package).resolve()
    a1_map_path = Path(a1_source_map).resolve()
    source_tags, a1_proposals, dm_proposals, source_identity = _source_case_map(
        a1_package=a1_dir,
        a1_source_map_path=a1_map_path,
        dm_package=dm_dir,
    )
    selected_ids = set(source_tags)
    if selected_ids - set(record_by_id):
        raise CandidateReviewError("historical development source is outside formal fit")
    selected_rows: list[dict[str, Any]] = []
    for record_id in selected_ids:
        row = record_by_id[record_id]
        content = row.get("content")
        if not isinstance(content, str) or not content:
            raise CandidateReviewError("formal fit record has invalid content")
        content_sha = _content_sha256(content)
        if (
            record_id in frozen.forbidden_ids
            or content_sha in frozen.forbidden_hashes
            or content in frozen.forbidden_contents
        ):
            raise CandidateReviewError("development record intersects calibration/dev/test")
        selected_rows.append(
            {
                "record_id": record_id,
                "content": content,
                "content_sha256": content_sha,
            }
        )
    selected_rows.sort(
        key=lambda row: _canonical_sha(
            {
                "seed": SELECTION_SEED,
                "record_id": row["record_id"],
                "content_sha256": row["content_sha256"],
            }
        )
    )

    cases: list[dict[str, Any]] = []
    proposals_document: list[dict[str, Any]] = []
    private_cases: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        case_id = f"S21-{index:03d}"
        blind_alias = f"术语开发-{index:03d}"
        proposals, proposal_sources = _build_proposals_for_record(
            record_id=row["record_id"],
            content=row["content"],
            a1_rows=a1_proposals.get(row["record_id"], []),
            dm_rows=dm_proposals.get(row["record_id"], []),
            g3_config=config,
        )
        cases.append(
            {
                "case_id": case_id,
                "blind_alias": blind_alias,
                "content": row["content"],
            }
        )
        proposals_document.append({"case_id": case_id, "proposals": proposals})
        private_cases.append(
            {
                "case_id": case_id,
                "record_id": row["record_id"],
                "content_sha256": row["content_sha256"],
                "historical_sources": sorted(source_tags[row["record_id"]]),
                "proposal_sources": proposal_sources,
            }
        )
    if len(cases) != EXPECTED_UNIQUE_RECORDS:
        raise CandidateReviewError("development frame does not contain 424 cases")

    data_locator, data_target = resolve_locator_ref(data_ref, "data")
    partition_locator, partition_target = resolve_locator_ref(
        train_partition_ref, "train-partition"
    )
    dependencies = {
        "data": portable_dependency(data_locator, data_target, root),
        "train_partition": portable_dependency(
            partition_locator, partition_target, root
        ),
    }
    private_map = {
        "schema_version": PRIVATE_MAP_SCHEMA_VERSION,
        "selection_policy": SELECTION_POLICY,
        "case_count": len(private_cases),
        "cases": private_cases,
    }
    identity = {
        "schema_version": FRAME_SCHEMA_VERSION,
        "artifact_kind": FRAME_ARTIFACT_KIND,
        "source_policy": SOURCE_POLICY,
        "selection_policy": SELECTION_POLICY,
        "review_policy": REVIEW_POLICY,
        "dependencies": dependencies,
        "fit_data_sha256": frozen.train_data_sha256,
        "fit_ids_sha256": frozen.train_ids_sha256,
        "generator_config_sha256": sha256_file(generator_config_file),
        "generator_implementation_sha256": sha256_file(candidate_implementation),
        "review_implementation_sha256": sha256_file(implementation_path),
        "annotation_schema_sha256": {
            path: sha256_file(root / path) for path in ANNOTATION_SCHEMA_PATHS
        },
        "handbook_version": config["handbook"]["version"],
        "handbook_sha256": sha256_file(handbook_path),
        "historical_sources": source_identity,
        "cases_sha256": _canonical_sha(cases),
        "diagnostic_proposals_sha256": _canonical_sha(proposals_document),
        "private_source_map_sha256": _canonical_sha(private_map),
        "case_count": len(cases),
        "task_label_contribution_count": 0,
        "calibration_dev_test_intersection_count": 0,
        "scientific_eligible": False,
        "sealed": False,
    }
    frame_id = "wp3devframe-" + _canonical_sha(identity)
    manifest = {
        **identity,
        "frame_id": frame_id,
        "human_visible_phase_a_fields": ["case_id", "blind_alias", "content"],
        "proposal_reveal_policy": REVIEW_POLICY,
        "legacy_proposals_are_diagnostic_only": True,
    }
    output_parent = Path(output_root).resolve()
    target = output_parent / frame_id
    output_parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        result = validate_development_frame(
            target,
            workspace_root=root,
            generator_config_path=generator_config_file,
            require_current_sources=True,
        )
        if write_ref is not None:
            write_locator_ref(
                write_ref,
                artifact_kind=FRAME_ARTIFACT_KIND,
                artifact_id=frame_id,
                target=target,
                payload_manifest_sha256=result["payload_manifest_sha256"],
            )
        return result
    staging = new_staging_directory(output_parent, frame_id)
    try:
        write_canonical_json(staging / "manifest.json", manifest)
        write_canonical_json(staging / "cases.json", cases)
        write_canonical_json(
            staging / "diagnostic_proposals.json", proposals_document
        )
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda staged: _validate_frame_payload(
                staged,
                private_map=private_map,
                workspace_root=root,
                generator_config_path=generator_config_file,
                require_current_sources=True,
            ),
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    _write_private_source_map(target, private_map)
    result = validate_development_frame(
        target,
        workspace_root=root,
        generator_config_path=generator_config_file,
        require_current_sources=True,
    )
    if result["payload_manifest_sha256"] != payload_hash:
        raise CandidateReviewError("published development frame hash drifted")
    if write_ref is not None:
        write_locator_ref(
            write_ref,
            artifact_kind=FRAME_ARTIFACT_KIND,
            artifact_id=frame_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    return result


def _validate_frame_payload(
    target: Path,
    *,
    private_map: Mapping[str, Any],
    workspace_root: Path,
    generator_config_path: Path,
    require_current_sources: bool,
) -> dict[str, Any]:
    manifest = _load_json_object(target / "manifest.json", "frame manifest")
    cases = _load_json_array(target / "cases.json", "frame cases")
    proposal_rows = _load_json_array(
        target / "diagnostic_proposals.json", "frame proposals"
    )
    if manifest.get("schema_version") != FRAME_SCHEMA_VERSION:
        raise CandidateReviewError("development frame schema is invalid")
    identity = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key
        not in {
            "frame_id",
            "human_visible_phase_a_fields",
            "proposal_reveal_policy",
            "legacy_proposals_are_diagnostic_only",
        }
    }
    expected_id = "wp3devframe-" + _canonical_sha(identity)
    if manifest.get("frame_id") != expected_id or not _target_name_matches_artifact_id(
        target, expected_id
    ):
        raise CandidateReviewError("development frame identity is invalid")
    if (
        len(cases) != EXPECTED_UNIQUE_RECORDS
        or manifest.get("case_count") != EXPECTED_UNIQUE_RECORDS
        or len(proposal_rows) != EXPECTED_UNIQUE_RECORDS
        or _canonical_sha(cases) != manifest.get("cases_sha256")
        or _canonical_sha(proposal_rows)
        != manifest.get("diagnostic_proposals_sha256")
        or _canonical_sha(private_map) != manifest.get("private_source_map_sha256")
    ):
        raise CandidateReviewError("development frame payload counts or hashes differ")
    if _forbidden_key_paths(cases) or _forbidden_key_paths(proposal_rows):
        raise CandidateReviewError("human-visible frame contains a forbidden task field")
    case_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        if set(case) != {"case_id", "blind_alias", "content"}:
            raise CandidateReviewError("phase-A case fields are not canonical")
        case_id = _require_text(case["case_id"], "case_id", maximum=40)
        _require_text(case["blind_alias"], "blind_alias", maximum=80)
        _require_text(case["content"], "content", maximum=10000)
        if case_id in case_by_id:
            raise CandidateReviewError("development frame case IDs are duplicated")
        case_by_id[case_id] = case
    proposals_by_case = {str(row.get("case_id")): row for row in proposal_rows}
    if set(proposals_by_case) != set(case_by_id):
        raise CandidateReviewError("proposal/case coverage differs")
    for case_id, row in proposals_by_case.items():
        if set(row) != {"case_id", "proposals"} or not isinstance(
            row["proposals"], list
        ):
            raise CandidateReviewError("proposal row fields are not canonical")
        seen: set[str] = set()
        content = str(case_by_id[case_id]["content"])
        for proposal in row["proposals"]:
            if not isinstance(proposal, Mapping) or set(proposal) != {
                "proposal_id",
                "surface",
                "occurrence_ordinal",
                "start",
                "end",
            }:
                raise CandidateReviewError("diagnostic proposal fields are not canonical")
            proposal_id = _require_text(
                proposal["proposal_id"], "proposal_id", maximum=80
            )
            if proposal_id in seen:
                raise CandidateReviewError("diagnostic proposal IDs are duplicated")
            seen.add(proposal_id)
            surface = _require_text(proposal["surface"], "surface", maximum=80)
            ordinal = proposal["occurrence_ordinal"]
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
                raise CandidateReviewError("proposal occurrence ordinal is invalid")
            try:
                expected_start, expected_end = resolve_exact_span(
                    content, surface, ordinal
                )
            except CandidateGeneratorError as exc:
                raise CandidateReviewError(f"proposal cannot replay: {exc}") from exc
            if (proposal["start"], proposal["end"]) != (
                expected_start,
                expected_end,
            ):
                raise CandidateReviewError("proposal offsets do not replay")
    if manifest.get("task_label_contribution_count") != 0 or manifest.get(
        "calibration_dev_test_intersection_count"
    ) != 0:
        raise CandidateReviewError("development frame leakage counters are nonzero")
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise CandidateReviewError("development frame dependencies are missing")
    resolve_dependency_target(dependencies["data"], workspace_root)
    resolve_dependency_target(dependencies["train_partition"], workspace_root)
    if require_current_sources:
        config = load_generator_config(
            generator_config_path, workspace_root=workspace_root
        )
        if (
            sha256_file(generator_config_path)
            != manifest.get("generator_config_sha256")
            or sha256_file(
                workspace_root / "src/build_lex/terminology_candidate_generators.py"
            )
            != manifest.get("generator_implementation_sha256")
            or sha256_file(Path(__file__))
            != manifest.get("review_implementation_sha256")
            or manifest.get("annotation_schema_sha256")
            != {
                path: sha256_file(workspace_root / path)
                for path in ANNOTATION_SCHEMA_PATHS
            }
            or sha256_file(workspace_root / str(config["handbook"]["path"]))
            != manifest.get("handbook_sha256")
        ):
            raise CandidateReviewError("current S2.1 source/config/handbook drifted")
    return {
        "frame_id": manifest["frame_id"],
        "case_count": len(cases),
        "manifest": manifest,
        "cases": cases,
        "proposal_rows": proposal_rows,
    }


def validate_development_frame(
    frame_dir: str | Path,
    *,
    workspace_root: str | Path,
    generator_config_path: str | Path,
    require_current_sources: bool = True,
) -> dict[str, Any]:
    target = Path(frame_dir).resolve()
    try:
        payload_hash = validate_payload_manifest(target)
        ensure_exact_file_set(
            target,
            {
                "manifest.json",
                "cases.json",
                "diagnostic_proposals.json",
                "payload_manifest.json",
            },
        )
    except TrainingArtifactError as exc:
        raise CandidateReviewError(str(exc)) from exc
    private_map = _load_private_source_map(target)
    result = _validate_frame_payload(
        target,
        private_map=private_map,
        workspace_root=Path(workspace_root).resolve(),
        generator_config_path=Path(generator_config_path).resolve(),
        require_current_sources=require_current_sources,
    )
    private_cases = private_map.get("cases")
    if (
        private_map.get("schema_version") != PRIVATE_MAP_SCHEMA_VERSION
        or private_map.get("case_count") != EXPECTED_UNIQUE_RECORDS
        or not isinstance(private_cases, list)
        or len(private_cases) != EXPECTED_UNIQUE_RECORDS
    ):
        raise CandidateReviewError("private source map schema or count differs")
    private_by_id = {
        str(row.get("case_id")): row
        for row in private_cases
        if isinstance(row, Mapping)
    }
    if set(private_by_id) != {str(row["case_id"]) for row in result["cases"]}:
        raise CandidateReviewError("private/public case coverage differs")
    selected_ids: set[str] = set()
    public_by_id = {str(row["case_id"]): row for row in result["cases"]}
    a1_ids: set[str] = set()
    dual_model_ids: set[str] = set()
    for row in private_by_id.values():
        record_id = _require_text(row.get("record_id"), "record_id", maximum=256)
        content_sha = str(row.get("content_sha256", ""))
        if not SHA256_RE.fullmatch(content_sha) or record_id in selected_ids:
            raise CandidateReviewError("private record identity is invalid or duplicated")
        selected_ids.add(record_id)
        historical_sources = row.get("historical_sources")
        if (
            not isinstance(historical_sources, list)
            or not historical_sources
            or len(historical_sources) != len(set(historical_sources))
            or any(
                source
                not in {
                    "historical_a1_200",
                    "historical_dual_model_240",
                }
                for source in historical_sources
            )
        ):
            raise CandidateReviewError("private historical source tags are malformed")
        if "historical_a1_200" in historical_sources:
            a1_ids.add(record_id)
        if "historical_dual_model_240" in historical_sources:
            dual_model_ids.add(record_id)
        proposal_sources = row.get("proposal_sources")
        if not isinstance(proposal_sources, Mapping):
            raise CandidateReviewError("private proposal sources are malformed")
        public_proposals = next(
            proposal_row["proposals"]
            for proposal_row in result["proposal_rows"]
            if proposal_row["case_id"] == row["case_id"]
        )
        if set(proposal_sources) != {
            str(proposal["proposal_id"]) for proposal in public_proposals
        }:
            raise CandidateReviewError("private/public proposal coverage differs")
    if (
        len(a1_ids) != EXPECTED_A1_RECORDS
        or len(dual_model_ids) != EXPECTED_DM_RECORDS
        or len(a1_ids & dual_model_ids) != EXPECTED_INTERSECTION
        or len(a1_ids | dual_model_ids) != EXPECTED_UNIQUE_RECORDS
    ):
        raise CandidateReviewError("private historical source counts differ")

    dependencies = result["manifest"]["dependencies"]
    data_target = resolve_dependency_target(dependencies["data"], workspace_root)
    partition_target = resolve_dependency_target(
        dependencies["train_partition"], workspace_root
    )
    with tempfile.TemporaryDirectory(prefix="wp3-s21-validate-") as temporary:
        temporary_root = Path(temporary)
        data_locator = {
            "schema_version": "stage1-locator-ref/v1",
            "artifact_kind": dependencies["data"]["artifact_kind"],
            "artifact_id": dependencies["data"]["artifact_id"],
            "target_path": str(data_target),
            "payload_manifest_sha256": dependencies["data"][
                "payload_manifest_sha256"
            ],
        }
        partition_locator = {
            "schema_version": "stage1-locator-ref/v1",
            "artifact_kind": dependencies["train_partition"]["artifact_kind"],
            "artifact_id": dependencies["train_partition"]["artifact_id"],
            "target_path": str(partition_target),
            "payload_manifest_sha256": dependencies["train_partition"][
                "payload_manifest_sha256"
            ],
        }
        data_ref = temporary_root / "data_ref.json"
        partition_ref = temporary_root / "train_partition_ref.json"
        write_canonical_json(data_ref, data_locator)
        write_canonical_json(partition_ref, partition_locator)
        frozen = resolve_train_input(
            data_ref=data_ref,
            train_partition_ref=partition_ref,
            formal=True,
            workspace_root=workspace_root,
        )
    if (
        frozen.train_data_sha256 != result["manifest"]["fit_data_sha256"]
        or frozen.train_ids_sha256 != result["manifest"]["fit_ids_sha256"]
    ):
        raise CandidateReviewError("formal fit projection hashes differ")
    fit_by_id = {str(row["id"]): row for row in frozen.records}
    for case_id, private_row in private_by_id.items():
        record_id = str(private_row["record_id"])
        fit_row = fit_by_id.get(record_id)
        content = fit_row.get("content") if isinstance(fit_row, Mapping) else None
        if (
            not isinstance(content, str)
            or public_by_id[case_id]["content"] != content
            or private_row["content_sha256"] != _content_sha256(content)
            or record_id in frozen.forbidden_ids
            or content in frozen.forbidden_contents
            or _content_sha256(content) in frozen.forbidden_hashes
        ):
            raise CandidateReviewError(
                "development case no longer replays from the formal fit partition"
            )
    result["private_map"] = private_map
    result["payload_manifest_sha256"] = payload_hash
    result["target"] = str(target)
    return result


def _session_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(item) for key, item in value.items() if key != "revision"}


def _with_session_revision(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _session_payload(value)
    result["revision"] = _canonical_sha(result)
    return result


def _read_session(path: str | Path) -> dict[str, Any]:
    session = _load_json_object(path, "review session")
    if session.get("revision") != _canonical_sha(_session_payload(session)):
        raise CandidateReviewError("review session revision hash is invalid")
    return session


def _write_session_cas(
    path: str | Path,
    session: Mapping[str, Any],
    *,
    expected_revision: str | None,
) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        current = _read_session(destination)
        if expected_revision is not None and current["revision"] != expected_revision:
            raise CandidateReviewConflict("review session changed concurrently")
    elif expected_revision is not None:
        raise CandidateReviewConflict("review session does not exist")
    result = _with_session_revision(session)
    write_canonical_json(destination, result)
    return result


def create_review_session(
    *,
    frame_dir: str | Path,
    session_path: str | Path,
    reviewer_id: str,
    workspace_root: str | Path,
    generator_config_path: str | Path,
) -> dict[str, Any]:
    frame = validate_development_frame(
        frame_dir,
        workspace_root=workspace_root,
        generator_config_path=generator_config_path,
    )
    reviewer = _require_text(reviewer_id, "reviewer_id", maximum=100)
    destination = Path(session_path)
    if destination.exists():
        existing = _read_session(destination)
        if (
            existing.get("frame_id") != frame["frame_id"]
            or existing.get("reviewer_id") != reviewer
        ):
            raise CandidateReviewError("existing session belongs to another frame/reviewer")
        return existing
    annotations = {
        str(case["case_id"]): {
            "status": "draft",
            "content_sha256": _content_sha256(str(case["content"])),
            "needs_explanation": False,
            "mentions": [],
            "notes": "",
        }
        for case in frame["cases"]
    }
    session = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "frame_id": frame["frame_id"],
        "frame_payload_manifest_sha256": frame["payload_manifest_sha256"],
        "reviewer_id": reviewer,
        "phase": "raw",
        "raw_annotations": annotations,
        "raw_lock": None,
        "diagnostic_decisions": {},
        "amendments": [],
        "finalized_gold_id": None,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    return _write_session_cas(destination, session, expected_revision=None)


def _normalize_mentions(
    content: str,
    mentions: Any,
    *,
    require_route: bool,
) -> list[dict[str, Any]]:
    if not isinstance(mentions, list):
        raise CandidateReviewError("mentions must be an array")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for index, row in enumerate(mentions):
        if not isinstance(row, Mapping):
            raise CandidateReviewError(f"mention {index} must be an object")
        allowed = {
            "surface",
            "occurrence_ordinal",
            "provisional_route",
            "reason_codes",
            "notes",
        }
        if set(row) != allowed:
            raise CandidateReviewError("mention fields are not canonical")
        surface = _require_text(row["surface"], "mention.surface", maximum=80)
        ordinal = row["occurrence_ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
            raise CandidateReviewError("mention occurrence ordinal is invalid")
        route = row["provisional_route"]
        if require_route and route not in PROVISIONAL_ROUTES:
            raise CandidateReviewError("mention provisional route is invalid")
        reasons = row["reason_codes"]
        if (
            not isinstance(reasons, list)
            or not reasons
            or len(reasons) != len(set(reasons))
            or any(reason not in REASON_CODES for reason in reasons)
        ):
            raise CandidateReviewError("mention reason codes are invalid")
        notes = _optional_text(row["notes"], "mention.notes", maximum=2000)
        try:
            start, end = resolve_exact_span(content, surface, ordinal)
        except CandidateGeneratorError as exc:
            raise CandidateReviewError(f"mention cannot replay: {exc}") from exc
        identity = (start, end, surface)
        if identity in seen:
            raise CandidateReviewError("mention occurrence is duplicated")
        seen.add(identity)
        mention_identity = {
            "surface": surface,
            "occurrence_ordinal": ordinal,
            "start": start,
            "end": end,
        }
        normalized.append(
            {
                "mention_id": "s21mention-" + _canonical_sha(mention_identity)[:32],
                **mention_identity,
                "provisional_route": route,
                "reason_codes": sorted(reasons),
                "notes": notes,
            }
        )
        if "other" in reasons and not notes:
            raise CandidateReviewError("mention reason 'other' requires notes")
    normalized.sort(key=lambda row: (row["start"], row["end"], row["surface"]))
    return normalized


def save_raw_annotation(
    *,
    frame_dir: str | Path,
    session_path: str | Path,
    case_id: str,
    annotation: Mapping[str, Any],
    confirm: bool,
    expected_revision: str,
) -> dict[str, Any]:
    frame_target = Path(frame_dir).resolve()
    frame_cases = _load_json_array(frame_target / "cases.json", "frame cases")
    case_by_id = {str(row["case_id"]): row for row in frame_cases}
    if case_id not in case_by_id:
        raise CandidateReviewError("unknown development case")
    session = _read_session(session_path)
    if session.get("finalized_gold_id") is not None:
        raise CandidateReviewError("finalized review session is immutable")
    if session.get("phase") not in {"raw", "diagnostic"}:
        raise CandidateReviewError("review session phase is invalid")
    if not isinstance(annotation, Mapping) or set(annotation) != {
        "needs_explanation",
        "mentions",
        "notes",
    }:
        raise CandidateReviewError("raw annotation fields are not canonical")
    needs = annotation["needs_explanation"]
    if not isinstance(needs, bool):
        raise CandidateReviewError("needs_explanation must be boolean")
    mentions = _normalize_mentions(
        str(case_by_id[case_id]["content"]),
        annotation["mentions"],
        require_route=True,
    )
    if needs != bool(mentions):
        raise CandidateReviewError("needs_explanation must agree with mentions")
    notes = _optional_text(annotation["notes"], "case.notes", maximum=4000)
    current = session["raw_annotations"].get(case_id)
    if not isinstance(current, Mapping):
        raise CandidateReviewError("session case state is missing")
    if current.get("status") == "confirmed":
        raise CandidateReviewError("confirmed raw annotation must be explicitly reopened")
    updated = copy.deepcopy(session)
    updated["raw_annotations"][case_id] = {
        "status": "confirmed" if confirm else "draft",
        "content_sha256": _content_sha256(str(case_by_id[case_id]["content"])),
        "needs_explanation": needs,
        "mentions": mentions,
        "notes": notes,
    }
    if updated.get("raw_lock") is not None and confirm:
        updated["raw_lock"] = {
            "policy": REVIEW_POLICY,
            "raw_annotations_sha256": _canonical_sha(updated["raw_annotations"]),
            "locked_at": updated["raw_lock"]["locked_at"],
            "amended_at": _utc_now(),
            "amendment_count": sum(
                row.get("scope") == "raw" for row in updated["amendments"]
            ),
        }
    updated["updated_at"] = _utc_now()
    return _write_session_cas(
        session_path, updated, expected_revision=expected_revision
    )


def lock_raw_phase(
    *, session_path: str | Path, expected_revision: str
) -> dict[str, Any]:
    session = _read_session(session_path)
    if session.get("phase") != "raw" or session.get("raw_lock") is not None:
        raise CandidateReviewError("raw phase is already locked or unavailable")
    annotations = session.get("raw_annotations")
    if not isinstance(annotations, Mapping) or len(annotations) != EXPECTED_UNIQUE_RECORDS:
        raise CandidateReviewError("raw annotation coverage differs")
    incomplete = sorted(
        case_id
        for case_id, row in annotations.items()
        if not isinstance(row, Mapping) or row.get("status") != "confirmed"
    )
    if incomplete:
        raise CandidateReviewError(
            f"raw phase is incomplete: {len(incomplete)} case(s) remain"
        )
    raw_sha = _canonical_sha(annotations)
    updated = copy.deepcopy(session)
    updated["phase"] = "diagnostic"
    updated["raw_lock"] = {
        "policy": REVIEW_POLICY,
        "raw_annotations_sha256": raw_sha,
        "locked_at": _utc_now(),
    }
    updated["updated_at"] = _utc_now()
    return _write_session_cas(
        session_path, updated, expected_revision=expected_revision
    )


def _normalize_result_mentions(content: str, value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise CandidateReviewError("result_mentions must be an array")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
            "surface",
            "occurrence_ordinal",
        }:
            raise CandidateReviewError("result mention fields are not canonical")
        surface = _require_text(row["surface"], "result surface", maximum=80)
        ordinal = row["occurrence_ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
            raise CandidateReviewError("result occurrence ordinal is invalid")
        try:
            start, end = resolve_exact_span(content, surface, ordinal)
        except CandidateGeneratorError as exc:
            raise CandidateReviewError(f"result mention cannot replay: {exc}") from exc
        identity = (start, end, surface)
        if identity in seen:
            raise CandidateReviewError("result mention is duplicated")
        seen.add(identity)
        normalized.append(
            {
                "surface": surface,
                "occurrence_ordinal": ordinal,
                "start": start,
                "end": end,
            }
        )
    normalized.sort(key=lambda row: (row["start"], row["end"], row["surface"]))
    return normalized


def save_diagnostic_decision(
    *,
    frame_dir: str | Path,
    session_path: str | Path,
    case_id: str,
    proposal_id: str,
    decision: Mapping[str, Any],
    confirm: bool,
    expected_revision: str,
) -> dict[str, Any]:
    session = _read_session(session_path)
    if session.get("phase") != "diagnostic" or session.get("raw_lock") is None:
        raise CandidateReviewError("proposals cannot be reviewed before the raw lock")
    if session.get("finalized_gold_id") is not None:
        raise CandidateReviewError("finalized review session is immutable")
    cases = _load_json_array(Path(frame_dir) / "cases.json", "frame cases")
    proposal_rows = _load_json_array(
        Path(frame_dir) / "diagnostic_proposals.json", "frame proposals"
    )
    case_by_id = {str(row["case_id"]): row for row in cases}
    proposals_by_case = {
        str(row["case_id"]): {
            str(proposal["proposal_id"]): proposal
            for proposal in row["proposals"]
        }
        for row in proposal_rows
    }
    if case_id not in case_by_id or proposal_id not in proposals_by_case.get(case_id, {}):
        raise CandidateReviewError("unknown diagnostic proposal")
    if not isinstance(decision, Mapping) or set(decision) != {
        "action",
        "result_mentions",
        "reason_codes",
        "notes",
    }:
        raise CandidateReviewError("diagnostic decision fields are not canonical")
    action = decision["action"]
    if action not in PROPOSAL_ACTIONS:
        raise CandidateReviewError("diagnostic action is invalid")
    result_mentions = _normalize_result_mentions(
        str(case_by_id[case_id]["content"]), decision["result_mentions"]
    )
    proposal = proposals_by_case[case_id][proposal_id]
    if action in {"reject", "defer"} and result_mentions:
        raise CandidateReviewError(f"{action} cannot retain result mentions")
    if action in {"accept", "trim", "expand"} and len(result_mentions) != 1:
        raise CandidateReviewError(f"{action} requires exactly one result mention")
    if action == "split" and len(result_mentions) < 2:
        raise CandidateReviewError("split requires at least two result mentions")
    if action == "accept" and (
        result_mentions[0]["surface"],
        result_mentions[0]["occurrence_ordinal"],
    ) != (proposal["surface"], proposal["occurrence_ordinal"]):
        raise CandidateReviewError("accept must retain the exact proposed occurrence")
    if action == "trim" and not (
        result_mentions[0]["start"] >= proposal["start"]
        and result_mentions[0]["end"] <= proposal["end"]
        and (
            result_mentions[0]["start"],
            result_mentions[0]["end"],
        )
        != (proposal["start"], proposal["end"])
    ):
        raise CandidateReviewError("trim must select a proper contained occurrence")
    if action == "expand" and not (
        result_mentions[0]["start"] <= proposal["start"]
        and result_mentions[0]["end"] >= proposal["end"]
        and (
            result_mentions[0]["start"],
            result_mentions[0]["end"],
        )
        != (proposal["start"], proposal["end"])
    ):
        raise CandidateReviewError("expand must select a proper containing occurrence")
    if action == "split" and any(
        mention["start"] < proposal["start"] or mention["end"] > proposal["end"]
        for mention in result_mentions
    ):
        raise CandidateReviewError("split results must remain inside the proposal")
    reasons = decision["reason_codes"]
    if (
        not isinstance(reasons, list)
        or not reasons
        or len(reasons) != len(set(reasons))
        or any(reason not in REASON_CODES for reason in reasons)
    ):
        raise CandidateReviewError("diagnostic reason codes are invalid")
    notes = _optional_text(decision["notes"], "diagnostic notes", maximum=2000)
    if (action == "defer" or "other" in reasons) and not notes:
        raise CandidateReviewError("defer/other decisions require notes")
    key = f"{case_id}:{proposal_id}"
    current = session["diagnostic_decisions"].get(key)
    if isinstance(current, Mapping) and current.get("status") == "confirmed":
        raise CandidateReviewError("confirmed diagnostic decision must be reopened")
    updated = copy.deepcopy(session)
    updated["diagnostic_decisions"][key] = {
        "case_id": case_id,
        "proposal_id": proposal_id,
        "status": "confirmed" if confirm else "draft",
        "action": action,
        "result_mentions": result_mentions,
        "reason_codes": sorted(reasons),
        "notes": notes,
    }
    updated["updated_at"] = _utc_now()
    return _write_session_cas(
        session_path, updated, expected_revision=expected_revision
    )


def reopen_review_item(
    *,
    session_path: str | Path,
    scope: str,
    case_id: str,
    proposal_id: str | None,
    reason: str,
    expected_revision: str,
) -> dict[str, Any]:
    session = _read_session(session_path)
    if session.get("finalized_gold_id") is not None:
        raise CandidateReviewError("finalized review session is immutable")
    amendment_reason = _require_text(reason, "amendment reason", maximum=1000)
    updated = copy.deepcopy(session)
    if scope == "raw":
        row = updated["raw_annotations"].get(case_id)
        if not isinstance(row, Mapping) or row.get("status") != "confirmed":
            raise CandidateReviewError("raw case is not confirmed")
        updated["raw_annotations"][case_id]["status"] = "draft"
        for key in list(updated["diagnostic_decisions"]):
            if key.startswith(f"{case_id}:"):
                del updated["diagnostic_decisions"][key]
        post_reveal = updated.get("raw_lock") is not None
    elif scope == "diagnostic":
        if not proposal_id:
            raise CandidateReviewError("diagnostic reopen requires proposal_id")
        key = f"{case_id}:{proposal_id}"
        row = updated["diagnostic_decisions"].get(key)
        if not isinstance(row, Mapping) or row.get("status") != "confirmed":
            raise CandidateReviewError("diagnostic decision is not confirmed")
        updated["diagnostic_decisions"][key]["status"] = "draft"
        post_reveal = True
    else:
        raise CandidateReviewError("reopen scope is invalid")
    updated["amendments"].append(
        {
            "scope": scope,
            "case_id": case_id,
            "proposal_id": proposal_id,
            "reason": amendment_reason,
            "post_reveal_raw_amendment": scope == "raw" and post_reveal,
            "reopened_at": _utc_now(),
            "prior_revision": session["revision"],
        }
    )
    updated["updated_at"] = _utc_now()
    return _write_session_cas(
        session_path, updated, expected_revision=expected_revision
    )


def review_session_status(
    *, frame_dir: str | Path, session_path: str | Path
) -> dict[str, Any]:
    session = _read_session(session_path)
    proposal_rows = _load_json_array(
        Path(frame_dir) / "diagnostic_proposals.json", "frame proposals"
    )
    proposal_count = sum(len(row["proposals"]) for row in proposal_rows)
    raw_confirmed = sum(
        isinstance(row, Mapping) and row.get("status") == "confirmed"
        for row in session["raw_annotations"].values()
    )
    diagnostic_confirmed = sum(
        isinstance(row, Mapping) and row.get("status") == "confirmed"
        for row in session["diagnostic_decisions"].values()
    )
    return {
        "frame_id": session["frame_id"],
        "reviewer_id": session["reviewer_id"],
        "phase": session["phase"],
        "revision": session["revision"],
        "raw": {"confirmed": raw_confirmed, "total": EXPECTED_UNIQUE_RECORDS},
        "diagnostic": {
            "confirmed": diagnostic_confirmed,
            "total": proposal_count,
        },
        "amendment_count": len(session["amendments"]),
        "finalized_gold_id": session["finalized_gold_id"],
    }


def _validate_session_complete(
    frame: Mapping[str, Any], session: Mapping[str, Any]
) -> None:
    if (
        session.get("schema_version") != SESSION_SCHEMA_VERSION
        or session.get("frame_id") != frame["frame_id"]
        or session.get("frame_payload_manifest_sha256")
        != frame["payload_manifest_sha256"]
        or session.get("phase") != "diagnostic"
        or not isinstance(session.get("raw_lock"), Mapping)
    ):
        raise CandidateReviewError("review session is not bound to a locked frame")
    raw = session.get("raw_annotations")
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {str(row["case_id"]) for row in frame["cases"]}
        or any(
            not isinstance(row, Mapping) or row.get("status") != "confirmed"
            for row in raw.values()
        )
    ):
        raise CandidateReviewError("raw annotations are incomplete")
    if session["raw_lock"].get("raw_annotations_sha256") != _canonical_sha(raw):
        raise CandidateReviewError("raw annotations do not match the current raw lock")
    raw_lock = session["raw_lock"]
    allowed_raw_lock_fields = {
        "policy",
        "raw_annotations_sha256",
        "locked_at",
    }
    if "amended_at" in raw_lock or "amendment_count" in raw_lock:
        allowed_raw_lock_fields |= {"amended_at", "amendment_count"}
    if (
        set(raw_lock) != allowed_raw_lock_fields
        or raw_lock.get("policy") != REVIEW_POLICY
        or not SHA256_RE.fullmatch(str(raw_lock.get("raw_annotations_sha256", "")))
        or not isinstance(raw_lock.get("locked_at"), str)
        or not raw_lock["locked_at"]
        or (
            "amendment_count" in raw_lock
            and (
                isinstance(raw_lock["amendment_count"], bool)
                or not isinstance(raw_lock["amendment_count"], int)
                or raw_lock["amendment_count"] < 1
                or not isinstance(raw_lock.get("amended_at"), str)
                or not raw_lock["amended_at"]
            )
        )
    ):
        raise CandidateReviewError("raw lock fields are invalid")
    case_by_id = {str(row["case_id"]): row for row in frame["cases"]}
    for case_id, annotation in raw.items():
        content = str(case_by_id[case_id]["content"])
        if set(annotation) != {
            "status",
            "content_sha256",
            "needs_explanation",
            "mentions",
            "notes",
        }:
            raise CandidateReviewError("raw annotation fields are not canonical")
        if (
            annotation.get("content_sha256") != _content_sha256(content)
            or not isinstance(annotation.get("mentions"), list)
        ):
            raise CandidateReviewError("raw annotation content hash differs")
        if annotation.get("needs_explanation") != bool(annotation.get("mentions")):
            raise CandidateReviewError("raw mention/explanation state differs")
        _optional_text(annotation.get("notes"), "case.notes", maximum=4000)
        seen_mentions: set[tuple[int, int, str]] = set()
        for mention in annotation.get("mentions", []):
            if not isinstance(mention, Mapping) or set(mention) != {
                "mention_id",
                "surface",
                "occurrence_ordinal",
                "start",
                "end",
                "provisional_route",
                "reason_codes",
                "notes",
            }:
                raise CandidateReviewError("raw mention is malformed")
            try:
                start, end = resolve_exact_span(
                    content,
                    str(mention["surface"]),
                    int(mention["occurrence_ordinal"]),
                )
            except (CandidateGeneratorError, KeyError, TypeError, ValueError) as exc:
                raise CandidateReviewError("raw mention cannot replay") from exc
            identity = (start, end, str(mention["surface"]))
            mention_identity = {
                "surface": str(mention["surface"]),
                "occurrence_ordinal": int(mention["occurrence_ordinal"]),
                "start": start,
                "end": end,
            }
            reasons = mention.get("reason_codes")
            notes = _optional_text(
                mention.get("notes"), "mention.notes", maximum=2000
            )
            if (
                (mention.get("start"), mention.get("end")) != (start, end)
                or identity in seen_mentions
                or mention.get("provisional_route") not in PROVISIONAL_ROUTES
                or mention.get("mention_id")
                != "s21mention-" + _canonical_sha(mention_identity)[:32]
                or not isinstance(reasons, list)
                or not reasons
                or len(reasons) != len(set(reasons))
                or any(reason not in REASON_CODES for reason in reasons)
                or ("other" in reasons and not notes)
            ):
                raise CandidateReviewError("raw mention offsets/routes differ")
            seen_mentions.add(identity)
    expected_proposals = {
        f"{row['case_id']}:{proposal['proposal_id']}"
        for row in frame["proposal_rows"]
        for proposal in row["proposals"]
    }
    decisions = session.get("diagnostic_decisions")
    if (
        not isinstance(decisions, Mapping)
        or set(decisions) != expected_proposals
        or any(
            not isinstance(row, Mapping) or row.get("status") != "confirmed"
            for row in decisions.values()
        )
    ):
        raise CandidateReviewError("diagnostic proposal decisions are incomplete")
    if any(row.get("action") == "defer" for row in decisions.values()):
        raise CandidateReviewError("deferred proposals must be resolved before finalize")
    proposal_by_key = {
        f"{row['case_id']}:{proposal['proposal_id']}": proposal
        for row in frame["proposal_rows"]
        for proposal in row["proposals"]
    }
    for key, decision in decisions.items():
        case_id, proposal_id = key.split(":", 1)
        if not isinstance(decision, Mapping) or set(decision) != {
            "case_id",
            "proposal_id",
            "status",
            "action",
            "result_mentions",
            "reason_codes",
            "notes",
        }:
            raise CandidateReviewError("diagnostic decision fields are not canonical")
        action = decision.get("action")
        reasons = decision.get("reason_codes")
        notes = _optional_text(
            decision.get("notes"), "diagnostic notes", maximum=2000
        )
        if (
            decision.get("case_id") != case_id
            or decision.get("proposal_id") != proposal_id
            or action not in PROPOSAL_ACTIONS
            or not isinstance(reasons, list)
            or not reasons
            or len(reasons) != len(set(reasons))
            or any(reason not in REASON_CODES for reason in reasons)
            or ("other" in reasons and not notes)
            or not isinstance(decision.get("result_mentions"), list)
        ):
            raise CandidateReviewError("diagnostic decision identity differs")
        content = str(case_by_id[case_id]["content"])
        normalized_results: list[dict[str, Any]] = []
        for mention in decision.get("result_mentions", []):
            if not isinstance(mention, Mapping) or set(mention) != {
                "surface",
                "occurrence_ordinal",
                "start",
                "end",
            }:
                raise CandidateReviewError("diagnostic result fields are not canonical")
            try:
                start, end = resolve_exact_span(
                    content,
                    str(mention["surface"]),
                    int(mention["occurrence_ordinal"]),
                )
            except (CandidateGeneratorError, KeyError, TypeError, ValueError) as exc:
                raise CandidateReviewError("diagnostic result cannot replay") from exc
            if (mention.get("start"), mention.get("end")) != (start, end):
                raise CandidateReviewError("diagnostic result offsets differ")
            normalized_results.append(dict(mention))
        proposal = proposal_by_key[key]
        if action in {"reject", "defer"} and normalized_results:
            raise CandidateReviewError(f"{action} cannot retain result mentions")
        if action in {"accept", "trim", "expand"} and len(normalized_results) != 1:
            raise CandidateReviewError(f"{action} requires one result mention")
        if action == "split" and len(normalized_results) < 2:
            raise CandidateReviewError("split requires at least two result mentions")
        if action == "accept" and (
            normalized_results[0]["surface"],
            normalized_results[0]["occurrence_ordinal"],
        ) != (proposal["surface"], proposal["occurrence_ordinal"]):
            raise CandidateReviewError("accept does not retain the proposed occurrence")
        if action == "trim" and not (
            normalized_results[0]["start"] >= proposal["start"]
            and normalized_results[0]["end"] <= proposal["end"]
            and (
                normalized_results[0]["start"],
                normalized_results[0]["end"],
            )
            != (proposal["start"], proposal["end"])
        ):
            raise CandidateReviewError("final trim relation is invalid")
        if action == "expand" and not (
            normalized_results[0]["start"] <= proposal["start"]
            and normalized_results[0]["end"] >= proposal["end"]
            and (
                normalized_results[0]["start"],
                normalized_results[0]["end"],
            )
            != (proposal["start"], proposal["end"])
        ):
            raise CandidateReviewError("final expand relation is invalid")
        if action == "split" and any(
            mention["start"] < proposal["start"]
            or mention["end"] > proposal["end"]
            for mention in normalized_results
        ):
            raise CandidateReviewError("final split relation is invalid")
    amendments = session.get("amendments")
    if not isinstance(amendments, list):
        raise CandidateReviewError("amendments must be an array")
    for amendment in amendments:
        if not isinstance(amendment, Mapping) or set(amendment) != {
            "scope",
            "case_id",
            "proposal_id",
            "reason",
            "post_reveal_raw_amendment",
            "reopened_at",
            "prior_revision",
        }:
            raise CandidateReviewError("amendment fields are not canonical")
        if (
            amendment.get("scope") not in {"raw", "diagnostic"}
            or amendment.get("case_id") not in case_by_id
            or not isinstance(amendment.get("post_reveal_raw_amendment"), bool)
            or not SHA256_RE.fullmatch(str(amendment.get("prior_revision", "")))
            or not isinstance(amendment.get("reopened_at"), str)
            or not amendment["reopened_at"]
        ):
            raise CandidateReviewError("amendment identity is invalid")
        _require_text(amendment.get("reason"), "amendment reason", maximum=1000)
        if amendment["scope"] == "diagnostic":
            proposal_id = amendment.get("proposal_id")
            if f"{amendment['case_id']}:{proposal_id}" not in proposal_by_key:
                raise CandidateReviewError("amendment proposal identity is invalid")
        elif amendment.get("proposal_id") is not None:
            raise CandidateReviewError("raw amendment cannot name a proposal")
    if "amendment_count" in raw_lock and raw_lock["amendment_count"] != sum(
        row.get("scope") == "raw" for row in amendments
    ):
        raise CandidateReviewError("raw lock amendment count differs")
    if _forbidden_key_paths(
        {
            "raw_annotations": raw,
            "diagnostic_decisions": decisions,
            "amendments": amendments,
        }
    ):
        raise CandidateReviewError("development gold contains a forbidden task field")


def finalize_development_gold(
    *,
    frame_dir: str | Path,
    session_path: str | Path,
    workspace_root: str | Path,
    generator_config_path: str | Path,
    output_root: str | Path,
    reviewer_id: str,
    write_ref: str | Path | None = None,
) -> dict[str, Any]:
    frame = validate_development_frame(
        frame_dir,
        workspace_root=workspace_root,
        generator_config_path=generator_config_path,
    )
    session = _read_session(session_path)
    reviewer = _require_text(reviewer_id, "reviewer_id", maximum=100)
    if session.get("reviewer_id") != reviewer:
        raise CandidateReviewError("finalize reviewer does not own the review session")
    if session.get("finalized_gold_id") is not None:
        raise CandidateReviewError("review session was already finalized")
    _validate_session_complete(frame, session)
    annotations = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "frame_id": frame["frame_id"],
        "reviewer_id": session["reviewer_id"],
        "raw_annotations": session["raw_annotations"],
        "raw_lock": session["raw_lock"],
        "diagnostic_decisions": session["diagnostic_decisions"],
        "amendments": session["amendments"],
        "review_session_revision": session["revision"],
    }
    declaration = {
        "schema_version": DECLARATION_SCHEMA_VERSION,
        "frame_id": frame["frame_id"],
        "reviewer_id": session["reviewer_id"],
        "review_policy": REVIEW_POLICY,
        "attestations": {
            "raw_annotations_completed_before_proposal_reveal": True,
            "task_annotations_were_not_used": True,
            "provisional_routes_are_not_formal_tiers": True,
            "all_diagnostic_proposals_were_resolved": True,
            "development_only_nonsealed_nonscientific": True,
        },
        "annotation_sha256": _canonical_sha(annotations),
        "finalized_at": _utc_now(),
    }
    identity = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "artifact_kind": GOLD_ARTIFACT_KIND,
        "frame_dependency": {
            "artifact_id": frame["frame_id"],
            "artifact_kind": FRAME_ARTIFACT_KIND,
            "payload_manifest_sha256": frame["payload_manifest_sha256"],
        },
        "annotation_sha256": _canonical_sha(annotations),
        "declaration_sha256": _canonical_sha(declaration),
        "review_implementation_sha256": sha256_file(Path(__file__)),
        "case_count": EXPECTED_UNIQUE_RECORDS,
        "scientific_eligible": False,
        "sealed": False,
    }
    gold_id = "wp3devgold-" + _canonical_sha(identity)
    manifest = {**identity, "gold_id": gold_id}
    output_parent = Path(output_root).resolve()
    target = output_parent / gold_id
    output_parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        staging = new_staging_directory(output_parent, gold_id)
        try:
            write_canonical_json(staging / "manifest.json", manifest)
            write_canonical_json(staging / "annotations.json", annotations)
            write_canonical_json(staging / "reviewer_declaration.json", declaration)
            payload_hash = finalize_target_atomic(
                staging,
                target,
                validate_staging=lambda staged: _validate_gold_payload(
                    staged, frame=frame
                ),
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    else:
        payload_hash = validate_payload_manifest(target)
    result = validate_development_gold(target, frame_dir=frame_dir)
    if result["payload_manifest_sha256"] != payload_hash:
        raise CandidateReviewError("published development gold hash drifted")
    updated = copy.deepcopy(session)
    updated["finalized_gold_id"] = gold_id
    updated["updated_at"] = _utc_now()
    _write_session_cas(
        session_path, updated, expected_revision=session["revision"]
    )
    if write_ref is not None:
        write_locator_ref(
            write_ref,
            artifact_kind=GOLD_ARTIFACT_KIND,
            artifact_id=gold_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    return result


def _validate_gold_payload(
    target: Path, *, frame: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = _load_json_object(target / "manifest.json", "gold manifest")
    annotations = _load_json_object(target / "annotations.json", "gold annotations")
    declaration = _load_json_object(
        target / "reviewer_declaration.json", "reviewer declaration"
    )
    expected_manifest_fields = {
        "schema_version",
        "artifact_kind",
        "frame_dependency",
        "annotation_sha256",
        "declaration_sha256",
        "review_implementation_sha256",
        "case_count",
        "scientific_eligible",
        "sealed",
        "gold_id",
    }
    expected_annotation_fields = {
        "schema_version",
        "frame_id",
        "reviewer_id",
        "raw_annotations",
        "raw_lock",
        "diagnostic_decisions",
        "amendments",
        "review_session_revision",
    }
    expected_declaration_fields = {
        "schema_version",
        "frame_id",
        "reviewer_id",
        "review_policy",
        "attestations",
        "annotation_sha256",
        "finalized_at",
    }
    if (
        set(manifest) != expected_manifest_fields
        or set(annotations) != expected_annotation_fields
        or set(declaration) != expected_declaration_fields
        or manifest.get("schema_version") != GOLD_SCHEMA_VERSION
        or annotations.get("schema_version") != GOLD_SCHEMA_VERSION
        or declaration.get("schema_version") != DECLARATION_SCHEMA_VERSION
        or annotations.get("frame_id") != frame["frame_id"]
        or declaration.get("frame_id") != frame["frame_id"]
        or _canonical_sha(annotations) != manifest.get("annotation_sha256")
        or _canonical_sha(declaration) != manifest.get("declaration_sha256")
        or declaration.get("annotation_sha256") != _canonical_sha(annotations)
        or annotations.get("reviewer_id") != declaration.get("reviewer_id")
        or manifest.get("artifact_kind") != GOLD_ARTIFACT_KIND
        or manifest.get("case_count") != EXPECTED_UNIQUE_RECORDS
        or manifest.get("sealed") is not False
        or manifest.get("scientific_eligible") is not False
        or declaration.get("review_policy") != REVIEW_POLICY
        or not SHA256_RE.fullmatch(
            str(annotations.get("review_session_revision", ""))
        )
    ):
        raise CandidateReviewError("development gold bindings differ")
    frame_dependency = manifest.get("frame_dependency")
    if frame_dependency != {
        "artifact_id": frame["frame_id"],
        "artifact_kind": FRAME_ARTIFACT_KIND,
        "payload_manifest_sha256": frame["payload_manifest_sha256"],
    }:
        raise CandidateReviewError("development gold frame dependency differs")
    identity = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "gold_id"
    }
    gold_id = "wp3devgold-" + _canonical_sha(identity)
    if manifest.get("gold_id") != gold_id or not _target_name_matches_artifact_id(
        target, gold_id
    ):
        raise CandidateReviewError("development gold identity is invalid")
    session_projection = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "frame_id": annotations["frame_id"],
        "frame_payload_manifest_sha256": frame["payload_manifest_sha256"],
        "reviewer_id": annotations["reviewer_id"],
        "phase": "diagnostic",
        "raw_annotations": annotations["raw_annotations"],
        "raw_lock": annotations["raw_lock"],
        "diagnostic_decisions": annotations["diagnostic_decisions"],
        "amendments": annotations["amendments"],
        "finalized_gold_id": None,
        "created_at": "not-part-of-gold-validation",
        "updated_at": "not-part-of-gold-validation",
        "revision": annotations["review_session_revision"],
    }
    # Validate completeness without trusting mutable timestamps/revision payload.
    _validate_session_complete(frame, session_projection)
    attestations = declaration.get("attestations")
    expected_attestations = {
        "raw_annotations_completed_before_proposal_reveal",
        "task_annotations_were_not_used",
        "provisional_routes_are_not_formal_tiers",
        "all_diagnostic_proposals_were_resolved",
        "development_only_nonsealed_nonscientific",
    }
    if (
        not isinstance(attestations, Mapping)
        or set(attestations) != expected_attestations
        or not all(
        attestations.get(key) is True
        for key in expected_attestations
        )
    ):
        raise CandidateReviewError("reviewer declaration is incomplete")
    return {
        "gold_id": gold_id,
        "manifest": manifest,
        "annotations": annotations,
        "declaration": declaration,
    }


def validate_development_gold(
    gold_dir: str | Path, *, frame_dir: str | Path
) -> dict[str, Any]:
    target = Path(gold_dir).resolve()
    try:
        payload_hash = validate_payload_manifest(target)
        ensure_exact_file_set(
            target,
            {
                "manifest.json",
                "annotations.json",
                "reviewer_declaration.json",
                "payload_manifest.json",
            },
        )
    except TrainingArtifactError as exc:
        raise CandidateReviewError(str(exc)) from exc
    frame_manifest = _load_json_object(
        Path(frame_dir) / "manifest.json", "frame manifest"
    )
    frame = {
        "frame_id": frame_manifest["frame_id"],
        "payload_manifest_sha256": validate_payload_manifest(frame_dir),
        "cases": _load_json_array(Path(frame_dir) / "cases.json", "frame cases"),
        "proposal_rows": _load_json_array(
            Path(frame_dir) / "diagnostic_proposals.json", "frame proposals"
        ),
    }
    result = _validate_gold_payload(target, frame=frame)
    result["payload_manifest_sha256"] = payload_hash
    result["target"] = str(target)
    return result


def _metric(predicted: set[tuple[str, int, int, str]], gold: set[tuple[str, int, int, str]]) -> dict[str, Any]:
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = true_positive / len(gold) if gold else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": true_positive,
        "predicted": len(predicted),
        "gold": len(gold),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "exact_boundary_f1": round(f1, 8),
    }


def build_development_report(
    *,
    frame_dir: str | Path,
    gold_dir: str | Path,
    output_root: str | Path,
    write_ref: str | Path | None = None,
) -> dict[str, Any]:
    frame_target = Path(frame_dir).resolve()
    frame_manifest = _load_json_object(frame_target / "manifest.json", "frame manifest")
    frame = {
        "frame_id": frame_manifest["frame_id"],
        "payload_manifest_sha256": validate_payload_manifest(frame_target),
        "cases": _load_json_array(frame_target / "cases.json", "frame cases"),
        "proposal_rows": _load_json_array(
            frame_target / "diagnostic_proposals.json", "frame proposals"
        ),
        "private_map": _load_private_source_map(frame_target),
    }
    gold = validate_development_gold(gold_dir, frame_dir=frame_dir)
    annotations = gold["annotations"]
    private_by_case = {
        str(row["case_id"]): row for row in frame["private_map"]["cases"]
    }
    gold_set: set[tuple[str, int, int, str]] = set()
    route_counts: Counter[str] = Counter()
    occurrence_counts: Counter[int] = Counter()
    empty_records = 0
    for case_id, row in annotations["raw_annotations"].items():
        mentions = row["mentions"]
        if not mentions:
            empty_records += 1
        record_id = str(private_by_case[case_id]["record_id"])
        for mention in mentions:
            gold_set.add(
                (
                    record_id,
                    int(mention["start"]),
                    int(mention["end"]),
                    str(mention["surface"]),
                )
            )
            route_counts[str(mention["provisional_route"])] += 1
            occurrence_counts[int(mention["occurrence_ordinal"])] += 1
    predicted_by_source: dict[str, set[tuple[str, int, int, str]]] = defaultdict(set)
    union: set[tuple[str, int, int, str]] = set()
    proposal_by_id: dict[str, dict[str, Any]] = {}
    for row in frame["proposal_rows"]:
        case_id = str(row["case_id"])
        record_id = str(private_by_case[case_id]["record_id"])
        source_map = private_by_case[case_id]["proposal_sources"]
        for proposal in row["proposals"]:
            proposal_by_id[str(proposal["proposal_id"])] = dict(proposal)
            identity = (
                record_id,
                int(proposal["start"]),
                int(proposal["end"]),
                str(proposal["surface"]),
            )
            union.add(identity)
            for source in source_map[str(proposal["proposal_id"])]:
                predicted_by_source[str(source)].add(identity)
    metrics = {source: _metric(values, gold_set) for source, values in sorted(predicted_by_source.items())}
    metrics["development_union"] = _metric(union, gold_set)
    unique_true_positives: dict[str, int] = {}
    for source in sorted(predicted_by_source):
        others = set().union(
            *(
                values
                for other, values in predicted_by_source.items()
                if other != source
            )
        ) if len(predicted_by_source) > 1 else set()
        unique_true_positives[source] = len(
            (predicted_by_source[source] & gold_set) - others
        )
    ablation_recall: dict[str, float] = {}
    for source in sorted(predicted_by_source):
        ablated = set().union(
            *(
                values
                for other, values in predicted_by_source.items()
                if other != source
            )
        ) if len(predicted_by_source) > 1 else set()
        ablation_recall[source] = _metric(ablated, gold_set)["recall"]
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for decision in annotations["diagnostic_decisions"].values():
        action_counts[str(decision["action"])] += 1
        reason_counts.update(str(value) for value in decision["reason_codes"])
    error_counts: Counter[str] = Counter(
        {
            "too-wide": 0,
            "too-narrow": 0,
            "fragment": 0,
            "substring": 0,
            "wrong-occurrence": 0,
            "non-gold-proposal": 0,
        }
    )
    for identity in union - gold_set:
        record_id, start, end, surface = identity
        same_record_gold = [row for row in gold_set if row[0] == record_id]
        if any(row[3] == surface for row in same_record_gold):
            error_counts["wrong-occurrence"] += 1
        elif any(
            start > row[1] and end < row[2] for row in same_record_gold
        ):
            error_counts["substring"] += 1
        elif any(
            start >= row[1] and end <= row[2] for row in same_record_gold
        ):
            error_counts["too-narrow"] += 1
        elif any(
            start <= row[1] and end >= row[2] for row in same_record_gold
        ):
            error_counts["too-wide"] += 1
        elif any(max(start, row[1]) < min(end, row[2]) for row in same_record_gold):
            error_counts["fragment"] += 1
        else:
            error_counts["non-gold-proposal"] += 1
    report_payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "frame_id": frame["frame_id"],
        "gold_id": gold["gold_id"],
        "scope": "development-only",
        "scientific_eligible": False,
        "sealed": False,
        "case_count": EXPECTED_UNIQUE_RECORDS,
        "empty_record_count": empty_records,
        "empty_record_rate": round(empty_records / EXPECTED_UNIQUE_RECORDS, 8),
        "gold_mention_count": len(gold_set),
        "occurrence_ordinal_counts": dict(sorted(occurrence_counts.items())),
        "provisional_route_counts": dict(sorted(route_counts.items())),
        "diagnostic_action_counts": dict(sorted(action_counts.items())),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "proposal_metrics": metrics,
        "unique_true_positives": unique_true_positives,
        "ablation_recall": ablation_recall,
        "boundary_error_counts": dict(sorted(error_counts.items())),
        "amendment_count": len(annotations["amendments"]),
        "post_reveal_raw_amendment_count": sum(
            bool(row.get("post_reveal_raw_amendment"))
            for row in annotations["amendments"]
        ),
        "raw_offset_replay_rate": 1.0,
        "task_label_contribution_count": 0,
        "calibration_dev_test_intersection_count": 0,
        "unresolved_proposal_decision_count": 0,
        "forbidden_field_scan": {
            "frame": "pass",
            "diagnostic_proposals": "pass",
            "development_gold": "pass",
        },
        "artifact_hash_validation": "pass",
        "legacy_metric_warning": (
            "Historical A1/dual-model proposals used retired prompts and are diagnostic only; "
            "only G3 is the current offline generator implementation."
        ),
    }
    identity = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "frame_id": frame["frame_id"],
        "frame_payload_manifest_sha256": frame["payload_manifest_sha256"],
        "gold_id": gold["gold_id"],
        "gold_payload_manifest_sha256": gold["payload_manifest_sha256"],
        "report_sha256": _canonical_sha(report_payload),
        "review_implementation_sha256": sha256_file(Path(__file__)),
    }
    report_id = "wp3devreport-" + _canonical_sha(identity)
    manifest = {**identity, "report_id": report_id}
    markdown = _render_report_markdown(report_payload)
    output_parent = Path(output_root).resolve()
    target = output_parent / report_id
    output_parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        staging = new_staging_directory(output_parent, report_id)
        try:
            write_canonical_json(staging / "manifest.json", manifest)
            write_canonical_json(staging / "report.json", report_payload)
            (staging / "report.md").write_text(markdown, encoding="utf-8")
            payload_hash = finalize_target_atomic(
                staging,
                target,
                validate_staging=lambda staged: _validate_report_payload(
                    staged, frame=frame, gold=gold
                ),
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    else:
        payload_hash = validate_payload_manifest(target)
    result = validate_development_report(
        target, frame_dir=frame_dir, gold_dir=gold_dir
    )
    if result["payload_manifest_sha256"] != payload_hash:
        raise CandidateReviewError("published development report hash drifted")
    if write_ref is not None:
        write_locator_ref(
            write_ref,
            artifact_kind=REPORT_ARTIFACT_KIND,
            artifact_id=report_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    return result


def _validate_report_payload(
    target: Path,
    *,
    frame: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _load_json_object(target / "manifest.json", "report manifest")
    report = _load_json_object(target / "report.json", "development report")
    markdown = (target / "report.md").read_text(encoding="utf-8")
    if (
        manifest.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or manifest.get("frame_id") != frame["frame_id"]
        or manifest.get("frame_payload_manifest_sha256")
        != frame["payload_manifest_sha256"]
        or manifest.get("gold_id") != gold["gold_id"]
        or manifest.get("gold_payload_manifest_sha256")
        != gold["payload_manifest_sha256"]
        or manifest.get("report_sha256") != _canonical_sha(report)
    ):
        raise CandidateReviewError("development report bindings differ")
    identity = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "report_id"
    }
    report_id = "wp3devreport-" + _canonical_sha(identity)
    if manifest.get("report_id") != report_id or not _target_name_matches_artifact_id(
        target, report_id
    ):
        raise CandidateReviewError("development report identity is invalid")
    if (
        report.get("scope") != "development-only"
        or report.get("sealed") is not False
        or report.get("scientific_eligible") is not False
        or report.get("case_count") != EXPECTED_UNIQUE_RECORDS
        or report.get("raw_offset_replay_rate") != 1.0
        or report.get("task_label_contribution_count") != 0
        or report.get("calibration_dev_test_intersection_count") != 0
        or report.get("unresolved_proposal_decision_count") != 0
        or "DEVELOPMENT ONLY / NON-SEALED / NON-SCIENTIFIC" not in markdown
        or _forbidden_key_paths(report)
    ):
        raise CandidateReviewError("development report completion/safety gate differs")
    return {"report_id": report_id, "manifest": manifest, "report": report}


def validate_development_report(
    report_dir: str | Path,
    *,
    frame_dir: str | Path,
    gold_dir: str | Path,
) -> dict[str, Any]:
    target = Path(report_dir).resolve()
    try:
        payload_hash = validate_payload_manifest(target)
        ensure_exact_file_set(
            target,
            {"manifest.json", "report.json", "report.md", "payload_manifest.json"},
        )
    except TrainingArtifactError as exc:
        raise CandidateReviewError(str(exc)) from exc
    frame_target = Path(frame_dir).resolve()
    frame_manifest = _load_json_object(frame_target / "manifest.json", "frame manifest")
    frame = {
        "frame_id": frame_manifest["frame_id"],
        "payload_manifest_sha256": validate_payload_manifest(frame_target),
    }
    gold = validate_development_gold(gold_dir, frame_dir=frame_dir)
    result = _validate_report_payload(target, frame=frame, gold=gold)
    result["payload_manifest_sha256"] = payload_hash
    result["target"] = str(target)
    return result


def _render_report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# WP3 S2.1 development report",
        "",
        "> DEVELOPMENT ONLY / NON-SEALED / NON-SCIENTIFIC",
        "",
        f"- cases: {report['case_count']}",
        f"- gold mentions: {report['gold_mention_count']}",
        f"- empty records: {report['empty_record_count']}",
        f"- amendments: {report['amendment_count']}",
        "",
        "## Proposal metrics",
        "",
        "| source | precision | recall | exact-boundary F1 | unique TP |",
        "|---|---:|---:|---:|---:|",
    ]
    for source, metric in report["proposal_metrics"].items():
        unique = report["unique_true_positives"].get(source, "-")
        lines.append(
            f"| {source} | {metric['precision']:.4f} | {metric['recall']:.4f} | "
            f"{metric['exact_boundary_f1']:.4f} | {unique} |"
        )
    lines.extend(
        [
            "",
            "## Distributions",
            "",
            f"- occurrence ordinals: `{json.dumps(report['occurrence_ordinal_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- provisional routes: `{json.dumps(report['provisional_route_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- diagnostic actions: `{json.dumps(report['diagnostic_action_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- reason codes: `{json.dumps(report['reason_code_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- boundary errors: `{json.dumps(report['boundary_error_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- source ablation recall: `{json.dumps(report['ablation_recall'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "## Safety",
            "",
            f"- raw offset replay rate: {report['raw_offset_replay_rate']}",
            f"- task-label contribution: {report['task_label_contribution_count']}",
            f"- calibration/dev/test intersection: {report['calibration_dev_test_intersection_count']}",
            f"- unresolved proposal decisions: {report['unresolved_proposal_decision_count']}",
            "",
            f"> {report['legacy_metric_warning']}",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "CandidateReviewConflict",
    "CandidateReviewError",
    "FRAME_ARTIFACT_KIND",
    "GOLD_ARTIFACT_KIND",
    "PROPOSAL_ACTIONS",
    "PROVISIONAL_ROUTES",
    "REASON_CODES",
    "REPORT_ARTIFACT_KIND",
    "build_development_frame",
    "build_development_report",
    "create_review_session",
    "finalize_development_gold",
    "lock_raw_phase",
    "reopen_review_item",
    "review_session_status",
    "save_diagnostic_decision",
    "save_raw_annotation",
    "validate_development_frame",
    "validate_development_gold",
    "validate_development_report",
]
