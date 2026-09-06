"""Immutable, public-frame-only execution lifecycle for WP3 full G3.

The lifecycle deliberately accepts only three data-bearing inputs: a finalized
form-reference artifact, the frozen full-v2 G3 profile, and the public 424-case
S2.1 frame.  It never resolves the frame's private source map, fit projection,
or any raw-gold artifact.  Completion is recomputed by the validator from a
fresh deterministic G3 replay; callers cannot supply a completion boolean.
"""

from __future__ import annotations

import copy
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from build_lex.terminology_candidate_generators_v2 import (
    G3_FAMILIES,
    CandidateGeneratorError,
    build_pypinyin_romanizer,
    generate_g3_observations,
    pinyin_initials,
    resolve_exact_span,
    validate_form_reference_document,
    validate_g3_profile,
    validate_observation,
)
from build_lex.terminology_g3_form_reference import (
    FRAME_ARTIFACT_KIND as FORM_FRAME_ARTIFACT_KIND,
    REFERENCE_ARTIFACT_KIND,
    SOURCE_BUNDLE_ARTIFACT_KIND,
    G3FormReferenceError,
    validate_form_reference,
)
from data.training_artifacts import (
    TrainingArtifactError,
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


RUN_SCHEMA_VERSION = "wp3-s21-g3-development-run/v1"
RECEIPT_SCHEMA_VERSION = "wp3-s21-g3-completion-receipt/v1"
RUN_ARTIFACT_KIND = "wp3-s21-g3-development-run"
PUBLIC_FRAME_ARTIFACT_KIND = "wp3-s21-development-frame"
PUBLIC_FRAME_SCHEMA_VERSION = "wp3-s21-development-frame/v1"
RUN_ID_PREFIX = "wp3g3run-"
EXPECTED_CASE_COUNT = 424
SCOPE = "development-only"
PROFILE_ID = "wp3-g3-profile/full-v2"
REFERENCE_ROLE = "form-only-label-free-non-lexicon"
OBSERVATION_SCHEMA_PATH = "schemas/wp3_candidate_observation_v1.schema.json"
REFERENCE_SCHEMA_PATH = "schemas/wp3_g3_form_reference_v1.schema.json"

# This ordering is part of the downstream runner contract.  Do not derive it
# from the frozenset exported by the generator module.
FAMILY_ORDER = (
    "mixed_script",
    "unicode_nfkc",
    "emoji",
    "known_variant",
    "pinyin_initials",
    "phonetic_variant",
    "separator_insertion",
    "orthographic_variant",
)

FORBIDDEN_PUBLIC_INPUT_KEYS = frozenset(
    {
        "annotation",
        "annotations",
        "gold",
        "raw_gold",
        "raw-gold",
        "fit",
        "record_id",
        "label",
        "labels",
        "task",
        "task_label",
        "verdict",
        "abc",
    }
)


class G3DevelopmentRunError(RuntimeError):
    """Raised when the public-only G3 lifecycle fails closed."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise G3DevelopmentRunError(f"{label} must be an object")
    return dict(value)


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise G3DevelopmentRunError(f"{label} must be an array")
    return list(value)


def _trimmed_text(value: Any, label: str, *, maximum: int = 10000) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise G3DevelopmentRunError(f"{label} must be non-empty trimmed text")
    return value


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        return _object(load_json(path), label)
    except TrainingArtifactError as exc:
        raise G3DevelopmentRunError(str(exc)) from exc


def _load_array(path: str | Path, label: str) -> list[Any]:
    try:
        return _array(load_json(path), label)
    except TrainingArtifactError as exc:
        raise G3DevelopmentRunError(str(exc)) from exc


def _target_matches(target: Path, artifact_id: str) -> bool:
    return target.name == artifact_id or target.name.startswith(f".{artifact_id}.")


def _forbidden_key_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = (*path, str(key))
            if str(key).casefold() in FORBIDDEN_PUBLIC_INPUT_KEYS:
                found.append(".".join(child_path))
            found.extend(_forbidden_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_key_paths(child, (*path, str(index))))
    return found


def _artifact_locator(
    *, target: Path, artifact_kind: str, artifact_id: str, payload_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": "stage1-locator-ref/v1",
        "artifact_kind": artifact_kind,
        "artifact_id": artifact_id,
        "target_path": str(target.resolve()),
        "payload_manifest_sha256": payload_sha256,
    }


def _profile_id(profile: Mapping[str, Any]) -> str:
    value = profile.get("profile_version")
    if value != PROFILE_ID:
        raise G3DevelopmentRunError("G3 profile is not the frozen full-v2 profile")
    return str(value)


def _load_profile(
    profile_path: str | Path, *, workspace_root: str | Path
) -> dict[str, Any]:
    profile = _load_object(profile_path, "G3 profile")
    try:
        return validate_g3_profile(profile, workspace_root=workspace_root)
    except CandidateGeneratorError as exc:
        raise G3DevelopmentRunError(str(exc)) from exc


def _validate_public_frame_payload(
    frame_dir: str | Path, *, expected_case_count: int
) -> dict[str, Any]:
    """Validate only the immutable public projection of the S2.1 frame.

    In particular, this function does not call ``validate_development_frame``:
    that older validator deliberately opens a sibling private source map and
    resolves formal fit artifacts.  Neither is an input to G3.
    """

    target = Path(frame_dir).resolve()
    try:
        payload_sha256 = validate_payload_manifest(target)
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
        raise G3DevelopmentRunError(str(exc)) from exc
    manifest = _load_object(target / "manifest.json", "public frame manifest")
    cases_raw = _load_array(target / "cases.json", "public frame cases")
    forbidden = _forbidden_key_paths(
        {"manifest": manifest, "cases": cases_raw}
    )
    if forbidden:
        raise G3DevelopmentRunError(
            "public frame contains forbidden task/data keys: "
            + ", ".join(forbidden)
        )
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
    frame_id = "wp3devframe-" + canonical_sha256(identity)
    if (
        manifest.get("schema_version") != PUBLIC_FRAME_SCHEMA_VERSION
        or manifest.get("artifact_kind") != PUBLIC_FRAME_ARTIFACT_KIND
        or manifest.get("frame_id") != frame_id
        or not _target_matches(target, frame_id)
        or manifest.get("case_count") != expected_case_count
        or len(cases_raw) != expected_case_count
        or manifest.get("cases_sha256") != canonical_sha256(cases_raw)
        or manifest.get("source_policy")
        != "fit-content-only-no-task-fields/v1"
        or manifest.get("task_label_contribution_count") != 0
        or manifest.get("calibration_dev_test_intersection_count") != 0
        or manifest.get("scientific_eligible") is not False
        or manifest.get("sealed") is not False
        or manifest.get("human_visible_phase_a_fields")
        != ["case_id", "blind_alias", "content"]
    ):
        raise G3DevelopmentRunError("public development-frame bindings differ")
    cases: list[dict[str, str]] = []
    seen_case_ids: set[str] = set()
    seen_aliases: set[str] = set()
    for raw in cases_raw:
        case = _object(raw, "public frame case")
        if set(case) != {"case_id", "blind_alias", "content"}:
            raise G3DevelopmentRunError("public frame exposes non-canonical fields")
        case_id = _trimmed_text(case["case_id"], "case_id", maximum=40)
        alias = _trimmed_text(case["blind_alias"], "blind_alias", maximum=80)
        content = _trimmed_text(case["content"], "content")
        if case_id in seen_case_ids or alias in seen_aliases:
            raise G3DevelopmentRunError("public frame case IDs or aliases are duplicated")
        seen_case_ids.add(case_id)
        seen_aliases.add(alias)
        cases.append({"case_id": case_id, "blind_alias": alias, "content": content})
    return {
        "frame_id": frame_id,
        "target": str(target),
        "payload_manifest_sha256": payload_sha256,
        "manifest": manifest,
        "cases": cases,
    }


def validate_public_development_frame(frame_dir: str | Path) -> dict[str, Any]:
    """Validate the production 424-case public frame without private inputs."""

    return _validate_public_frame_payload(
        frame_dir, expected_case_count=EXPECTED_CASE_COUNT
    )


def _reference_dependency_target(
    *, reference_target: Path, dependency: Mapping[str, Any], directory: str
) -> Path:
    row = _object(dependency, f"reference {directory} dependency")
    if set(row) != {"artifact_kind", "artifact_id", "payload_manifest_sha256"}:
        raise G3DevelopmentRunError("reference dependency fields are not canonical")
    expected_kind = (
        SOURCE_BUNDLE_ARTIFACT_KIND
        if directory == "public_source_bundles"
        else FORM_FRAME_ARTIFACT_KIND
    )
    if row.get("artifact_kind") != expected_kind:
        raise G3DevelopmentRunError("reference dependency kind differs")
    artifact_id = _trimmed_text(row.get("artifact_id"), "dependency artifact ID")
    target = reference_target.parent.parent / directory / artifact_id
    if not target.is_dir() or target.is_symlink() or target.name != artifact_id:
        raise G3DevelopmentRunError("reference lifecycle dependency is missing")
    try:
        payload = validate_payload_manifest(target)
    except TrainingArtifactError as exc:
        raise G3DevelopmentRunError(str(exc)) from exc
    if payload != row.get("payload_manifest_sha256"):
        raise G3DevelopmentRunError("reference lifecycle dependency hash differs")
    return target


def validate_finalized_form_reference(
    reference_dir: str | Path,
    *,
    workspace_root: str | Path,
    profile_path: str | Path,
    profile: Mapping[str, Any] | None = None,
    romanizer: Callable[[str], list[str]] | None = None,
    require_current_implementation: bool = True,
) -> dict[str, Any]:
    """Independently replay a finalized, human-declared form reference."""

    target = Path(reference_dir).resolve()
    manifest = _load_object(target / "manifest.json", "form-reference manifest")
    source_target = _reference_dependency_target(
        reference_target=target,
        dependency=manifest.get("source_bundle_dependency"),
        directory="public_source_bundles",
    )
    frame_target = _reference_dependency_target(
        reference_target=target,
        dependency=manifest.get("frame_dependency"),
        directory="review_frames",
    )
    normalized_profile = (
        dict(profile)
        if profile is not None
        else _load_profile(profile_path, workspace_root=workspace_root)
    )
    runtime_romanizer = romanizer
    if runtime_romanizer is None:
        try:
            runtime_romanizer = build_pypinyin_romanizer(
                normalized_profile, workspace_root=workspace_root
            )
        except CandidateGeneratorError as exc:
            raise G3DevelopmentRunError(str(exc)) from exc
    try:
        result = validate_form_reference(
            target,
            frame_dir=frame_target,
            source_bundle_dir=source_target,
            workspace_root=workspace_root,
            romanizer_profile_path=profile_path,
            reference_schema_path=Path(workspace_root) / REFERENCE_SCHEMA_PATH,
            require_current_implementation=require_current_implementation,
            romanizer=runtime_romanizer,
            initials_builder=pinyin_initials,
        )
    except G3FormReferenceError as exc:
        raise G3DevelopmentRunError(str(exc)) from exc
    if result["manifest"].get("profile_sha256") != canonical_sha256(
        normalized_profile
    ):
        raise G3DevelopmentRunError("form reference is bound to a different G3 profile")
    if result["reference"].get("reference_role") != REFERENCE_ROLE:
        raise G3DevelopmentRunError("form reference role differs")
    return {**result, "profile": normalized_profile, "romanizer": runtime_romanizer}


def _expected_rule_source(
    *, reference: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "kind": "rule",
        "rule_version": profile["rule_version"],
        "reference_sha256": canonical_sha256(reference["rows"]),
        "profile_version": profile["profile_version"],
        "profile_sha256": canonical_sha256(profile),
        "romanizer_backend": profile["romanizer"]["backend_id"],
        "romanizer_resource_sha256": profile["romanizer"][
            "resource_manifest_sha256"
        ],
    }
    return {**payload, "source_id": "rulesrc-" + canonical_sha256(payload)[:32]}


def _run_cases(
    *,
    cases: Sequence[Mapping[str, str]],
    reference: Mapping[str, Any],
    profile: Mapping[str, Any],
    romanizer: Callable[[str], Sequence[str]],
    generator: Callable[..., list[dict[str, Any]]],
    workspace_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_source = _expected_rule_source(reference=reference, profile=profile)
    observation_schema = Path(workspace_root) / OBSERVATION_SCHEMA_PATH
    for case in cases:
        case_id = str(case["case_id"])
        content = str(case["content"])
        try:
            rows = generator(
                record_id=case_id,
                content=content,
                reference_document=reference,
                romanizer=romanizer,
                profile=profile,
                max_surface_chars=int(profile["limits"]["max_surface_chars"]),
                mixed_script_max_chars=int(
                    profile["limits"]["mixed_script_max_chars"]
                ),
                phonetic_max_distance=int(profile["phonetic_max_distance"]),
            )
            if not isinstance(rows, list):
                raise CandidateGeneratorError("G3 generator returned a non-array")
            record_rows: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for raw in rows:
                row = validate_observation(raw, content=content)
                validate_json_schema(row, observation_schema)
                family = row.get("generator_variant")
                expected_mechanism = (
                    "abbreviation" if family == "pinyin_initials" else family
                )
                if (
                    row.get("record_id") != case_id
                    or row.get("generator") != "g3_form_rule"
                    or family not in FAMILY_ORDER
                    or row.get("mechanism") != expected_mechanism
                    or row.get("source") != expected_source
                    or row["observation_id"] in seen_ids
                ):
                    raise CandidateGeneratorError(
                        "G3 observation provenance or family differs"
                    )
                seen_ids.add(str(row["observation_id"]))
                record_rows.append(row)
            observations.extend(record_rows)
        except CandidateGeneratorError:
            failures.append(
                {
                    "case_id": case_id,
                    "error_code": "generator-contract-error",
                    "failed_families": list(FAMILY_ORDER),
                }
            )
        except Exception:
            # Exception text is intentionally excluded: the public artifact is
            # a stable completion credential, not a runtime-debugging vault.
            failures.append(
                {
                    "case_id": case_id,
                    "error_code": "unexpected-runtime-error",
                    "failed_families": list(FAMILY_ORDER),
                }
            )
    observations.sort(key=lambda row: str(row["observation_id"]))
    failures.sort(key=lambda row: str(row["case_id"]))
    if len({row["observation_id"] for row in observations}) != len(observations):
        raise G3DevelopmentRunError("G3 observation IDs are duplicated across cases")
    return observations, failures


def _derive_completion_receipt(
    *,
    cases: Sequence[Mapping[str, str]],
    observations: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    frame_dependency: Mapping[str, Any],
    reference_dependency: Mapping[str, Any],
    reference_id: str,
    reference_sha256: str,
    profile: Mapping[str, Any],
    expected_case_count: int,
) -> dict[str, Any]:
    case_by_id = {str(case["case_id"]): case for case in cases}
    failure_ids: set[str] = set()
    for raw in failures:
        failure = _object(raw, "G3 record failure")
        if set(failure) != {"case_id", "error_code", "failed_families"}:
            raise G3DevelopmentRunError("G3 record failure fields differ")
        case_id = str(failure.get("case_id"))
        if (
            case_id not in case_by_id
            or case_id in failure_ids
            or failure.get("error_code")
            not in {"generator-contract-error", "unexpected-runtime-error"}
            or failure.get("failed_families") != list(FAMILY_ORDER)
        ):
            raise G3DevelopmentRunError("G3 record failure is invalid")
        failure_ids.add(case_id)
    replayed = 0
    family_observations: Counter[str] = Counter()
    for raw in observations:
        row = _object(raw, "G3 observation")
        case_id = str(row.get("record_id"))
        case = case_by_id.get(case_id)
        if case is None or case_id in failure_ids:
            raise G3DevelopmentRunError("G3 observation is outside completed records")
        content = str(case["content"])
        try:
            validated = validate_observation(row, content=content)
            start, end = resolve_exact_span(
                content,
                str(validated["surface"]),
                int(validated["occurrence_ordinal"]),
            )
        except CandidateGeneratorError as exc:
            raise G3DevelopmentRunError(f"G3 offset replay failed: {exc}") from exc
        if (start, end) != (validated["start"], validated["end"]):
            raise G3DevelopmentRunError("G3 occurrence offsets do not replay")
        family = str(validated["generator_variant"])
        if family not in FAMILY_ORDER:
            raise G3DevelopmentRunError("G3 observation family is not enabled")
        family_observations[family] += 1
        replayed += 1
    successful_records = len(cases) - len(failure_ids)
    enabled = [family for family in FAMILY_ORDER if profile["families"].get(family)]
    families = {
        family: {
            "enabled": profile["families"].get(family) is True,
            "record_completion_count": successful_records,
            "record_error_count": len(failure_ids),
            "observation_count": family_observations[family],
        }
        for family in FAMILY_ORDER
    }
    replay_rate = 1 if replayed == len(observations) else 0
    complete = (
        len(cases) == expected_case_count
        and successful_records == expected_case_count
        and not failure_ids
        and enabled == list(FAMILY_ORDER)
        and all(
            row["record_completion_count"] == expected_case_count
            and row["record_error_count"] == 0
            for row in families.values()
        )
        and replay_rate == 1
    )
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "artifact_kind": RUN_ARTIFACT_KIND,
        "scope": SCOPE,
        "scientific_eligible": False,
        "sealed": False,
        "frame_dependency": copy.deepcopy(dict(frame_dependency)),
        "reference_dependency": copy.deepcopy(dict(reference_dependency)),
        "reference_id": reference_id,
        "reference_sha256": reference_sha256,
        "profile_id": _profile_id(profile),
        "profile_sha256": canonical_sha256(profile),
        "romanizer_backend_id": profile["romanizer"]["backend_id"],
        "romanizer_resource_sha256": profile["romanizer"][
            "resource_manifest_sha256"
        ],
        "case_count": len(cases),
        "successful_record_count": successful_records,
        "terminal_failure_count": len(failure_ids),
        "enabled_families": enabled,
        "families": families,
        "observation_count": len(observations),
        "offset_occurrence_count": len(observations),
        "offset_replay_count": replayed,
        "offset_replay_rate": replay_rate,
        "complete": complete,
    }


def _run_manifest(
    *,
    frame_dependency: Mapping[str, Any],
    reference_dependency: Mapping[str, Any],
    reference_id: str,
    reference_sha256: str,
    profile: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": RUN_SCHEMA_VERSION,
        "artifact_kind": RUN_ARTIFACT_KIND,
        "scope": SCOPE,
        "scientific_eligible": False,
        "sealed": False,
        "frame_dependency": copy.deepcopy(dict(frame_dependency)),
        "reference_dependency": copy.deepcopy(dict(reference_dependency)),
        "profile_id": _profile_id(profile),
        "profile_sha256": canonical_sha256(profile),
        "reference_id": reference_id,
        "reference_sha256": reference_sha256,
        "romanizer_backend_id": profile["romanizer"]["backend_id"],
        "romanizer_resource_sha256": profile["romanizer"][
            "resource_manifest_sha256"
        ],
        "enabled_families": list(receipt["enabled_families"]),
        "case_count": receipt["case_count"],
        "terminal_failure_count": receipt["terminal_failure_count"],
        "observation_count": len(observations),
        "observations_sha256": canonical_sha256(list(observations)),
        "record_failures_sha256": canonical_sha256(list(failures)),
        "completion_receipt_sha256": canonical_sha256(dict(receipt)),
        "offset_occurrence_count": receipt["offset_occurrence_count"],
        "offset_replay_count": receipt["offset_replay_count"],
        "offset_replay_rate": receipt["offset_replay_rate"],
        "complete": receipt["complete"],
        "generator_implementation_sha256": sha256_file(
            Path(__file__).with_name("terminology_candidate_generators_v2.py")
        ),
        "lifecycle_implementation_sha256": sha256_file(Path(__file__)),
        "observation_schema_sha256": sha256_file(
            Path(__file__).resolve().parents[2] / OBSERVATION_SCHEMA_PATH
        ),
    }
    return {**identity, "artifact_id": RUN_ID_PREFIX + canonical_sha256(identity)}


def _validate_run_payload(
    target: Path,
    *,
    cases: Sequence[Mapping[str, str]],
    reference: Mapping[str, Any],
    profile: Mapping[str, Any],
    romanizer: Callable[[str], Sequence[str]],
    frame_dependency: Mapping[str, Any],
    reference_dependency: Mapping[str, Any],
    reference_id: str,
    reference_sha256: str,
    expected_case_count: int,
    generator: Callable[..., list[dict[str, Any]]],
    require_current_implementation: bool,
) -> dict[str, Any]:
    try:
        payload_sha256 = validate_payload_manifest(target)
        ensure_exact_file_set(
            target,
            {
                "manifest.json",
                "profile.json",
                "observations.jsonl",
                "record_failures.json",
                "completion_receipt.json",
                "payload_manifest.json",
            },
        )
        observations = load_jsonl(target / "observations.jsonl")
    except TrainingArtifactError as exc:
        raise G3DevelopmentRunError(str(exc)) from exc
    manifest = _load_object(target / "manifest.json", "G3 run manifest")
    stored_profile = _load_object(target / "profile.json", "frozen G3 profile")
    failures = _load_array(target / "record_failures.json", "G3 record failures")
    receipt = _load_object(target / "completion_receipt.json", "G3 completion receipt")
    if stored_profile != dict(profile):
        raise G3DevelopmentRunError("embedded G3 profile differs")
    expected_observations, expected_failures = _run_cases(
        cases=cases,
        reference=reference,
        profile=profile,
        romanizer=romanizer,
        generator=generator,
        workspace_root=Path(__file__).resolve().parents[2],
    )
    if observations != expected_observations or failures != expected_failures:
        raise G3DevelopmentRunError("G3 payload does not replay from public inputs")
    expected_receipt = _derive_completion_receipt(
        cases=cases,
        observations=expected_observations,
        failures=expected_failures,
        frame_dependency=frame_dependency,
        reference_dependency=reference_dependency,
        reference_id=reference_id,
        reference_sha256=reference_sha256,
        profile=profile,
        expected_case_count=expected_case_count,
    )
    if receipt != expected_receipt:
        raise G3DevelopmentRunError("G3 completion receipt is not validator-derived")
    expected_manifest = _run_manifest(
        frame_dependency=frame_dependency,
        reference_dependency=reference_dependency,
        reference_id=reference_id,
        reference_sha256=reference_sha256,
        profile=profile,
        observations=expected_observations,
        failures=expected_failures,
        receipt=expected_receipt,
    )
    if manifest != expected_manifest:
        raise G3DevelopmentRunError("G3 run manifest bindings differ")
    artifact_id = str(expected_manifest["artifact_id"])
    if not _target_matches(target, artifact_id):
        raise G3DevelopmentRunError("G3 run content-addressed target differs")
    if require_current_implementation and (
        manifest.get("lifecycle_implementation_sha256") != sha256_file(Path(__file__))
        or manifest.get("generator_implementation_sha256")
        != sha256_file(Path(__file__).with_name("terminology_candidate_generators_v2.py"))
    ):
        raise G3DevelopmentRunError("G3 implementation hash drifted")
    return {
        "artifact_id": artifact_id,
        "target": str(target),
        "payload_manifest_sha256": payload_sha256,
        "manifest": manifest,
        "completion_receipt": receipt,
        "complete": receipt["complete"],
    }


def _build_from_validated_inputs(
    *,
    workspace_root: str | Path,
    cases: Sequence[Mapping[str, str]],
    frame_dependency: Mapping[str, Any],
    reference: Mapping[str, Any],
    reference_dependency: Mapping[str, Any],
    reference_id: str,
    reference_sha256: str,
    profile: Mapping[str, Any],
    romanizer: Callable[[str], Sequence[str]],
    output_root: str | Path,
    write_ref: str | Path | None,
    expected_case_count: int,
    generator: Callable[..., list[dict[str, Any]]] = generate_g3_observations,
) -> dict[str, Any]:
    """Build from already validated inputs; ``expected_case_count`` is test-only."""

    if len(cases) != expected_case_count:
        raise G3DevelopmentRunError("public frame case count differs")
    if set(G3_FAMILIES) != set(FAMILY_ORDER):
        raise G3DevelopmentRunError("generator and lifecycle family sets differ")
    try:
        normalized_profile = validate_g3_profile(
            profile, workspace_root=workspace_root
        )
        normalized_reference = validate_form_reference_document(reference)
        validate_dependency_ref(frame_dependency, expected_kind=PUBLIC_FRAME_ARTIFACT_KIND)
        validate_dependency_ref(reference_dependency, expected_kind=REFERENCE_ARTIFACT_KIND)
    except (CandidateGeneratorError, TrainingArtifactError) as exc:
        raise G3DevelopmentRunError(str(exc)) from exc
    if set(normalized_profile["families"]) != set(FAMILY_ORDER) or any(
        normalized_profile["families"][family] is not True for family in FAMILY_ORDER
    ):
        raise G3DevelopmentRunError("all eight G3 families must be enabled")
    if (
        reference_dependency.get("artifact_id") != reference_id
        or not reference_id.startswith("wp3g3formref-")
        or reference_sha256 != canonical_sha256(normalized_reference)
    ):
        raise G3DevelopmentRunError("form-reference identity or document hash differs")
    observations, failures = _run_cases(
        cases=cases,
        reference=normalized_reference,
        profile=normalized_profile,
        romanizer=romanizer,
        generator=generator,
        workspace_root=workspace_root,
    )
    receipt = _derive_completion_receipt(
        cases=cases,
        observations=observations,
        failures=failures,
        frame_dependency=frame_dependency,
        reference_dependency=reference_dependency,
        reference_id=reference_id,
        reference_sha256=reference_sha256,
        profile=normalized_profile,
        expected_case_count=expected_case_count,
    )
    manifest = _run_manifest(
        frame_dependency=frame_dependency,
        reference_dependency=reference_dependency,
        reference_id=reference_id,
        reference_sha256=reference_sha256,
        profile=normalized_profile,
        observations=observations,
        failures=failures,
        receipt=receipt,
    )
    artifact_id = str(manifest["artifact_id"])
    parent = Path(output_root).resolve()
    target = parent / artifact_id
    if not target.exists():
        staging = new_staging_directory(parent, artifact_id)
        try:
            write_canonical_json(staging / "manifest.json", manifest)
            write_canonical_json(staging / "profile.json", normalized_profile)
            write_canonical_jsonl(
                staging / "observations.jsonl",
                observations,
                key="observation_id",
            )
            write_canonical_json(staging / "record_failures.json", failures)
            write_canonical_json(staging / "completion_receipt.json", receipt)
            payload_sha256 = finalize_target_atomic(
                staging,
                target,
                validate_staging=lambda candidate: _validate_run_payload(
                    candidate,
                    cases=cases,
                    reference=normalized_reference,
                    profile=normalized_profile,
                    romanizer=romanizer,
                    frame_dependency=frame_dependency,
                    reference_dependency=reference_dependency,
                    reference_id=reference_id,
                    reference_sha256=reference_sha256,
                    expected_case_count=expected_case_count,
                    generator=generator,
                    require_current_implementation=True,
                ),
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    result = _validate_run_payload(
        target,
        cases=cases,
        reference=normalized_reference,
        profile=normalized_profile,
        romanizer=romanizer,
        frame_dependency=frame_dependency,
        reference_dependency=reference_dependency,
        reference_id=reference_id,
        reference_sha256=reference_sha256,
        expected_case_count=expected_case_count,
        generator=generator,
        require_current_implementation=True,
    )
    if write_ref is not None:
        try:
            write_locator_ref(
                write_ref,
                artifact_kind=RUN_ARTIFACT_KIND,
                artifact_id=artifact_id,
                target=target,
                payload_manifest_sha256=result["payload_manifest_sha256"],
            )
        except TrainingArtifactError as exc:
            raise G3DevelopmentRunError(str(exc)) from exc
    return result


def build_g3_development_run(
    *,
    workspace_root: str | Path,
    reference_dir: str | Path,
    frame_dir: str | Path,
    profile_path: str | Path,
    output_root: str | Path,
    write_ref: str | Path | None = None,
) -> dict[str, Any]:
    """Run full G3 over exactly 424 public cases and publish its credential."""

    root = Path(workspace_root).resolve()
    profile = _load_profile(profile_path, workspace_root=root)
    try:
        romanizer = build_pypinyin_romanizer(profile, workspace_root=root)
    except CandidateGeneratorError as exc:
        raise G3DevelopmentRunError(str(exc)) from exc
    reference = validate_finalized_form_reference(
        reference_dir,
        workspace_root=root,
        profile_path=profile_path,
        profile=profile,
        romanizer=romanizer,
    )
    frame = validate_public_development_frame(frame_dir)
    reference_target = Path(reference["target"])
    frame_target = Path(frame["target"])
    try:
        reference_dependency = portable_dependency(
            _artifact_locator(
                target=reference_target,
                artifact_kind=REFERENCE_ARTIFACT_KIND,
                artifact_id=reference["reference_id"],
                payload_sha256=reference["payload_manifest_sha256"],
            ),
            reference_target,
            root,
        )
        frame_dependency = portable_dependency(
            _artifact_locator(
                target=frame_target,
                artifact_kind=PUBLIC_FRAME_ARTIFACT_KIND,
                artifact_id=frame["frame_id"],
                payload_sha256=frame["payload_manifest_sha256"],
            ),
            frame_target,
            root,
        )
    except TrainingArtifactError as exc:
        raise G3DevelopmentRunError(str(exc)) from exc
    return _build_from_validated_inputs(
        workspace_root=root,
        cases=frame["cases"],
        frame_dependency=frame_dependency,
        reference=reference["reference"],
        reference_dependency=reference_dependency,
        reference_id=reference["reference_id"],
        reference_sha256=reference["manifest"]["reference_sha256"],
        profile=profile,
        romanizer=romanizer,
        output_root=output_root,
        write_ref=write_ref,
        expected_case_count=EXPECTED_CASE_COUNT,
    )


def validate_g3_development_run(
    run_dir: str | Path,
    *,
    workspace_root: str | Path,
    reference_dir: str | Path,
    frame_dir: str | Path,
    profile_path: str | Path,
) -> dict[str, Any]:
    """Independently rerun G3 and derive completion from public inputs."""

    root = Path(workspace_root).resolve()
    profile = _load_profile(profile_path, workspace_root=root)
    try:
        romanizer = build_pypinyin_romanizer(profile, workspace_root=root)
    except CandidateGeneratorError as exc:
        raise G3DevelopmentRunError(str(exc)) from exc
    reference = validate_finalized_form_reference(
        reference_dir,
        workspace_root=root,
        profile_path=profile_path,
        profile=profile,
        romanizer=romanizer,
    )
    frame = validate_public_development_frame(frame_dir)
    reference_target = Path(reference["target"])
    frame_target = Path(frame["target"])
    try:
        reference_dependency = portable_dependency(
            _artifact_locator(
                target=reference_target,
                artifact_kind=REFERENCE_ARTIFACT_KIND,
                artifact_id=reference["reference_id"],
                payload_sha256=reference["payload_manifest_sha256"],
            ),
            reference_target,
            root,
        )
        frame_dependency = portable_dependency(
            _artifact_locator(
                target=frame_target,
                artifact_kind=PUBLIC_FRAME_ARTIFACT_KIND,
                artifact_id=frame["frame_id"],
                payload_sha256=frame["payload_manifest_sha256"],
            ),
            frame_target,
            root,
        )
    except TrainingArtifactError as exc:
        raise G3DevelopmentRunError(str(exc)) from exc
    return _validate_run_payload(
        Path(run_dir).resolve(),
        cases=frame["cases"],
        reference=reference["reference"],
        profile=profile,
        romanizer=romanizer,
        frame_dependency=frame_dependency,
        reference_dependency=reference_dependency,
        reference_id=reference["reference_id"],
        reference_sha256=reference["manifest"]["reference_sha256"],
        expected_case_count=EXPECTED_CASE_COUNT,
        generator=generate_g3_observations,
        require_current_implementation=True,
    )


def validate_g3_run_ref_for_planning(
    g3_run_ref: str | Path,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Return the runner's exact projection after a full independent replay.

    Unlike a manifest-only projection, this entry point resolves the embedded
    portable public-frame and form-reference dependencies, validates the
    finalized human-reference lifecycle, reruns all 424 records, and derives
    ``complete`` from the resulting receipt.
    """

    root = Path(workspace_root).resolve()
    try:
        locator, target = resolve_locator_ref(g3_run_ref, RUN_ARTIFACT_KIND)
        manifest = _object(load_json(target / "manifest.json"), "G3 run manifest")
        frame_target = resolve_dependency_target(manifest["frame_dependency"], root)
        reference_target = resolve_dependency_target(
            manifest["reference_dependency"], root
        )
    except (KeyError, TrainingArtifactError) as exc:
        raise G3DevelopmentRunError(str(exc)) from exc
    validated = validate_g3_development_run(
        target,
        workspace_root=root,
        reference_dir=reference_target,
        frame_dir=frame_target,
        profile_path=target / "profile.json",
    )
    if (
        locator.get("artifact_id") != validated["artifact_id"]
        or locator.get("payload_manifest_sha256")
        != validated["payload_manifest_sha256"]
    ):
        raise G3DevelopmentRunError("G3 locator differs from validated artifact")
    frozen = validated["manifest"]
    projection = {
        "artifact_kind": RUN_ARTIFACT_KIND,
        "artifact_id": validated["artifact_id"],
        "payload_manifest_sha256": validated["payload_manifest_sha256"],
        "profile_id": frozen["profile_id"],
        "profile_sha256": frozen["profile_sha256"],
        "reference_id": frozen["reference_id"],
        "reference_sha256": frozen["reference_sha256"],
        "romanizer_backend_id": frozen["romanizer_backend_id"],
        "case_count": frozen["case_count"],
        "enabled_families": list(frozen["enabled_families"]),
        "terminal_failure_count": frozen["terminal_failure_count"],
        "offset_replay_count": frozen["offset_replay_count"],
        "offset_occurrence_count": frozen["offset_occurrence_count"],
        "offset_replay_rate": frozen["offset_replay_rate"],
        "complete": validated["completion_receipt"]["complete"],
    }
    if (
        projection["case_count"] != EXPECTED_CASE_COUNT
        or projection["enabled_families"] != list(FAMILY_ORDER)
        or projection["terminal_failure_count"] != 0
        or projection["offset_replay_count"]
        != projection["offset_occurrence_count"]
        or projection["offset_replay_rate"] != 1
        or projection["complete"] is not True
    ):
        raise G3DevelopmentRunError("G3 run is not a full completion credential")
    return projection


__all__ = [
    "EXPECTED_CASE_COUNT",
    "FAMILY_ORDER",
    "G3DevelopmentRunError",
    "RUN_ARTIFACT_KIND",
    "build_g3_development_run",
    "validate_finalized_form_reference",
    "validate_g3_development_run",
    "validate_g3_run_ref_for_planning",
    "validate_public_development_frame",
]
