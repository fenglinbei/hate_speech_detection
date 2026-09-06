"""Offline WP3 G3 public-source catalog and bundle lifecycle, version 2.

Version 1 remains implemented in :mod:`terminology_g3_form_reference` and is
not interpreted by this module.  The v2 builder consumes two already frozen
dependencies: the 42-target zero-proxy capture and a user-supplied manual
archive intake.  It performs no network access.  Failed capture responses are
retained only in the acquisition receipt; only validated, normalized local
components can be copied into the relation-review source bundle.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

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


CATALOG_SCHEMA_VERSION = "wp3-g3-public-source-catalog/v2"
ACQUISITION_RECEIPT_SCHEMA_VERSION = (
    "wp3-g3-public-source-acquisition-receipt/v2"
)
COMPONENTS_SCHEMA_VERSION = "wp3-g3-public-source-components/v2"
SOURCE_BUNDLE_SCHEMA_VERSION = "wp3-g3-public-source-bundle/v2"

CATALOG_ID = "wp3-g3-fixed-public-priors/v2"
ACQUISITION_RECEIPT_ARTIFACT_KIND = "wp3-g3-public-source-acquisition-receipt"
SOURCE_BUNDLE_ARTIFACT_KIND = "wp3-g3-public-source-bundle"
CAPTURE_ARTIFACT_KIND = "wp3-external-page-capture"
MANUAL_INTAKE_ARTIFACT_KIND = "wp3-g3-manual-source-intake"
ACQUISITION_RECEIPT_ID_PREFIX = "wp3g3acq-"
SOURCE_BUNDLE_ID_PREFIX = "wp3g3sources-"

CATALOG_SCHEMA_PATH = "schemas/wp3_g3_public_source_catalog_v2.schema.json"
ACQUISITION_RECEIPT_SCHEMA_PATH = (
    "schemas/wp3_g3_public_source_acquisition_receipt_v2.schema.json"
)
COMPONENTS_SCHEMA_PATH = "schemas/wp3_g3_public_source_components_v2.schema.json"
SOURCE_BUNDLE_SCHEMA_PATH = "schemas/wp3_g3_public_source_bundle_v2.schema.json"

FROZEN_TARGET_COUNT = 42
FROZEN_TARGET_URLS_SHA256 = (
    "2c95ea9709e1b4c916c38bb1748c4d0eb9c1097341d35f91df880a1a9249f844"
)
SOURCE_POLICY_ID = "closed-42-target-offline-two-dependency/v2"
REFERENCE_ROLE = "form-only-label-free-non-lexicon"
SCOPE = "development-only"

ACQUISITION_MODES = (
    "direct_fetch",
    "publisher_pdf",
    "user_supplied_archive",
)
SOURCE_ROLES = (
    "candidate_pool",
    "direct_evidence",
    "method_only",
    "prevalence_or_taxonomy",
)
DISPOSITIONS = ("accepted", "companion", "duplicate", "rejected")
EVIDENCE_ROLES = frozenset({"candidate_pool", "direct_evidence"})
CAPTURE_COMPLETE_STATUS = "downloaded_usable"

CHIME_COMMIT = "865ef186a0e797ec5ac242524a3c45b30a429542"
CHIME_LICENSE_SHA256 = (
    "b1205c11b8450decda38571230806045aa8c3951b84919b2da99fe3efd286e6d"
)
WIKIMEDIA_OLDIDS = {
    "wikipedia-mainland-internet-language-page": "93974959",
    "wikipedia-internet-language-page": "93933509",
}
MANUAL_COMPONENT_IDS = frozenset(
    {
        "chime-repository-archive",
        "chime-data-json",
        "chime-license",
        "wikipedia-mainland-internet-language-page",
        "wikipedia-internet-language-page",
        "fx361-letter-abbreviations-page",
        "lingoace-internet-buzzwords-page",
        "ctgoodjobs-mainland-internet-language-page",
        "people-weibo-neologisms-page-1",
        "people-weibo-neologisms-page-2",
        "cppcc-network-language-page",
        "hanspub-homophonic-words-pdf",
    }
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,119}$")
MAX_NORMALIZED_COMPONENT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_NORMALIZED_BYTES = 256 * 1024 * 1024


class G3SourceCatalogV2Error(RuntimeError):
    """Raised when a v2 catalog, dependency, receipt, or bundle fails closed."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise G3SourceCatalogV2Error(f"{label} must be an object")
    return dict(value)


def _objects(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise G3SourceCatalogV2Error(f"{label} must be an array of objects")
    return [dict(row) for row in value]


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        return _object(load_json(path), label)
    except TrainingArtifactError as exc:
        raise G3SourceCatalogV2Error(str(exc)) from exc


def _validate_schema(value: Mapping[str, Any], schema_path: Path) -> None:
    try:
        validate_json_schema(value, schema_path)
    except TrainingArtifactError as exc:
        if "jsonschema is required" in str(exc):
            schema = _load_object(schema_path, "JSON schema")
            if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
                raise G3SourceCatalogV2Error("JSON schema draft binding differs") from exc
            return
        raise G3SourceCatalogV2Error(str(exc)) from exc


def _target_name_matches(path: Path, artifact_id: str) -> bool:
    return path.name == artifact_id or path.name.startswith(f".{artifact_id}.")


def _safe_file(root: Path, logical_path: Any, *, label: str) -> Path:
    if not isinstance(logical_path, str) or not logical_path:
        raise G3SourceCatalogV2Error(f"{label} path is missing")
    relative = Path(logical_path)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != logical_path:
        raise G3SourceCatalogV2Error(f"{label} path is unsafe")
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise G3SourceCatalogV2Error(f"{label} escapes its artifact") from exc
    if not target.is_file() or target.is_symlink():
        raise G3SourceCatalogV2Error(f"{label} is unavailable or unsafe")
    return target


def _safe_workspace_file(root: Path, logical_path: Any, *, label: str) -> Path:
    if not isinstance(logical_path, str) or not logical_path:
        raise G3SourceCatalogV2Error(f"{label} path is missing")
    relative = Path(logical_path)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != logical_path:
        raise G3SourceCatalogV2Error(f"{label} path is not portable")
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise G3SourceCatalogV2Error(f"{label} escapes the workspace") from exc
    if not target.is_file() or target.is_symlink():
        raise G3SourceCatalogV2Error(f"{label} is unavailable or unsafe")
    return target


def _url(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise G3SourceCatalogV2Error(f"{label} is missing")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise G3SourceCatalogV2Error(f"{label} is malformed") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise G3SourceCatalogV2Error(
            f"{label} must be HTTP(S) without userinfo or fragment"
        )
    return value


def _schema_hashes(root: Path) -> dict[str, str]:
    return {
        "catalog_schema_sha256": sha256_file(root / CATALOG_SCHEMA_PATH),
        "acquisition_receipt_schema_sha256": sha256_file(
            root / ACQUISITION_RECEIPT_SCHEMA_PATH
        ),
        "components_schema_sha256": sha256_file(root / COMPONENTS_SCHEMA_PATH),
        "source_bundle_schema_sha256": sha256_file(root / SOURCE_BUNDLE_SCHEMA_PATH),
    }


def _validate_catalog_policy(catalog: Mapping[str, Any]) -> None:
    policy = _object(catalog.get("policy"), "catalog policy")
    expected = {
        "allowlist_closed": True,
        "follow_links": False,
        "bundle_builder_network_access": False,
        "same_host_protocol_downgrade": "allowed-only-when-recorded",
        "relation_review_required": True,
        "normalized_text_required_for_evidence": True,
        "pricing_sources_excluded": True,
        "acquisition_modes": list(ACQUISITION_MODES),
        "source_roles": list(SOURCE_ROLES),
        "dispositions": list(DISPOSITIONS),
        "extraction_scope": [
            "attested_written_variant",
            "homophone_or_near_homophone",
            "orthographic_substitution",
            "pinyin_initials",
        ],
        "excluded_scope": [
            "abc_or_r_tiers",
            "definitions",
            "fit_or_gold_records",
            "offensiveness_or_profanity",
            "pricing_material",
            "semantic_slang_without_form_relation",
            "stance",
            "task_labels",
        ],
    }
    if policy != expected:
        raise G3SourceCatalogV2Error("v2 catalog policy differs from the frozen policy")


def load_source_catalog_v2(
    catalog_path: str | Path,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Load and fully validate the exact closed 42-target v2 catalog."""

    root = Path(workspace_root).resolve()
    catalog = _load_object(catalog_path, "G3 source catalog v2")
    _validate_schema(catalog, root / CATALOG_SCHEMA_PATH)
    if (
        catalog.get("schema_version") != CATALOG_SCHEMA_VERSION
        or catalog.get("catalog_id") != CATALOG_ID
    ):
        raise G3SourceCatalogV2Error("v2 catalog identity differs")
    _validate_catalog_policy(catalog)
    handbook = _object(catalog.get("handbook"), "catalog handbook")
    expected_handbook_fields = {"source_id", "role", "version", "path", "sha256"}
    if (
        set(handbook) != expected_handbook_fields
        or handbook.get("source_id") != "wp3-handbook-v1"
        or handbook.get("role") != "rubric-only-not-form-evidence"
        or handbook.get("version") != "wp3-terminology-evidence-handbook/v1.0"
    ):
        raise G3SourceCatalogV2Error("v2 catalog handbook contract differs")
    handbook_path = _safe_workspace_file(root, handbook.get("path"), label="handbook")
    if sha256_file(handbook_path) != handbook.get("sha256"):
        raise G3SourceCatalogV2Error("catalog handbook hash drifted")

    targets = _objects(catalog.get("targets"), "catalog targets")
    if len(targets) != FROZEN_TARGET_COUNT:
        raise G3SourceCatalogV2Error("v2 catalog must contain exactly 42 targets")
    expected_target_fields = {
        "target_id",
        "publisher",
        "requested_url",
        "acquisition_mode",
        "source_role",
        "disposition",
        "evidence_allowed",
        "disposition_reason",
        "components",
    }
    target_ids: set[str] = set()
    target_urls: set[str] = set()
    component_ids: set[str] = set()
    manual_component_ids: set[str] = set()
    for target in targets:
        if set(target) != expected_target_fields:
            raise G3SourceCatalogV2Error("catalog target fields differ")
        target_id = target.get("target_id")
        if not isinstance(target_id, str) or not ID_RE.fullmatch(target_id):
            raise G3SourceCatalogV2Error("catalog target ID is invalid")
        requested_url = _url(target.get("requested_url"), "catalog requested URL")
        if target_id in target_ids or requested_url in target_urls:
            raise G3SourceCatalogV2Error("catalog target IDs or URLs are duplicated")
        target_ids.add(target_id)
        target_urls.add(requested_url)
        if "bigmodel.cn/pricing" in requested_url.casefold():
            raise G3SourceCatalogV2Error("pricing references cannot enter the G3 catalog")
        mode = target.get("acquisition_mode")
        role = target.get("source_role")
        disposition = target.get("disposition")
        evidence_allowed = target.get("evidence_allowed")
        if mode not in ACQUISITION_MODES or role not in SOURCE_ROLES:
            raise G3SourceCatalogV2Error("catalog acquisition mode or source role is invalid")
        if disposition not in DISPOSITIONS or not isinstance(evidence_allowed, bool):
            raise G3SourceCatalogV2Error("catalog disposition or evidence gate is invalid")
        if evidence_allowed and (
            role not in EVIDENCE_ROLES or disposition != "accepted"
        ):
            raise G3SourceCatalogV2Error(
                "only accepted direct/candidate sources may allow relation evidence"
            )
        if role in {"method_only", "prevalence_or_taxonomy"} and evidence_allowed:
            raise G3SourceCatalogV2Error("non-evidence source role cannot produce relations")
        reason = target.get("disposition_reason")
        if not isinstance(reason, str) or not ID_RE.fullmatch(reason):
            raise G3SourceCatalogV2Error("catalog disposition reason is invalid")
        components = _objects(target.get("components"), "target components")
        if disposition == "rejected":
            if components or evidence_allowed:
                raise G3SourceCatalogV2Error("rejected catalog target must have no components")
        elif not components:
            raise G3SourceCatalogV2Error("non-rejected catalog target lacks components")
        ordinals: list[int] = []
        for component in components:
            expected_component_fields = {
                "component_id",
                "component_kind",
                "component_url",
                "page_ordinal",
                "wikimedia_oldid",
                "required_commit",
                "required_license_sha256",
            }
            if set(component) != expected_component_fields:
                raise G3SourceCatalogV2Error("catalog component fields differ")
            component_id = component.get("component_id")
            if not isinstance(component_id, str) or not ID_RE.fullmatch(component_id):
                raise G3SourceCatalogV2Error("catalog component ID is invalid")
            if component_id in component_ids:
                raise G3SourceCatalogV2Error("catalog component IDs are duplicated")
            component_ids.add(component_id)
            _url(component.get("component_url"), "catalog component URL")
            ordinal = component.get("page_ordinal")
            if ordinal is not None:
                if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
                    raise G3SourceCatalogV2Error("component page ordinal is invalid")
                ordinals.append(ordinal)
            oldid = component.get("wikimedia_oldid")
            expected_oldid = WIKIMEDIA_OLDIDS.get(component_id)
            if oldid != expected_oldid:
                raise G3SourceCatalogV2Error("Wikimedia oldid binding differs")
            if oldid is not None:
                query = parse_qs(urlsplit(str(component["component_url"])).query)
                if query.get("oldid") != [oldid]:
                    raise G3SourceCatalogV2Error("Wikimedia component URL is not oldid-pinned")
            commit = component.get("required_commit")
            license_sha = component.get("required_license_sha256")
            if commit is not None and commit != CHIME_COMMIT:
                raise G3SourceCatalogV2Error("CHIME commit binding differs")
            if license_sha is not None and license_sha != CHIME_LICENSE_SHA256:
                raise G3SourceCatalogV2Error("CHIME license binding differs")
            if component_id in MANUAL_COMPONENT_IDS:
                manual_component_ids.add(component_id)
        if ordinals and sorted(ordinals) != list(range(1, len(ordinals) + 1)):
            raise G3SourceCatalogV2Error("catalog pagination is not contiguous")
    if canonical_sha256(sorted(target_urls)) != FROZEN_TARGET_URLS_SHA256:
        raise G3SourceCatalogV2Error("catalog URL allowlist differs from the frozen 42 targets")
    if manual_component_ids != MANUAL_COMPONENT_IDS:
        raise G3SourceCatalogV2Error("catalog manual-component coverage differs")
    if sum(target["disposition"] == "rejected" for target in targets) != 1:
        raise G3SourceCatalogV2Error("catalog must contain exactly one explicit rejection")
    rejected = next(target for target in targets if target["disposition"] == "rejected")
    if rejected["target_id"] != "bupt-internet-slang-project":
        raise G3SourceCatalogV2Error("the frozen rejected target differs")
    return catalog


def _dependency(
    *, artifact_kind: str, artifact_id: Any, payload_manifest_sha256: Any
) -> dict[str, str]:
    if not isinstance(artifact_id, str) or not artifact_id:
        raise G3SourceCatalogV2Error("dependency artifact ID is invalid")
    if not isinstance(payload_manifest_sha256, str) or not SHA256_RE.fullmatch(
        payload_manifest_sha256
    ):
        raise G3SourceCatalogV2Error("dependency payload hash is invalid")
    return {
        "artifact_kind": artifact_kind,
        "artifact_id": artifact_id,
        "payload_manifest_sha256": payload_manifest_sha256,
    }


def _load_capture_dependency(
    capture_directory: str | Path,
) -> tuple[dict[str, str], Path, dict[str, Any]]:
    try:
        from build_lex.terminology_external_page_capture import (
            validate_external_page_capture,
        )

        report = validate_external_page_capture(capture_directory)
    except Exception as exc:
        raise G3SourceCatalogV2Error(
            "zero-proxy capture dependency failed independent validation"
        ) from exc
    root = Path(report["target"]).resolve()
    manifest = _load_object(root / "manifest.json", "zero-proxy capture manifest")
    pages = _objects(manifest.get("pages"), "capture pages")
    urls = [row.get("requested_url") for row in pages]
    if (
        len(urls) != FROZEN_TARGET_COUNT
        or any(not isinstance(url, str) for url in urls)
        or len(set(urls)) != FROZEN_TARGET_COUNT
        or canonical_sha256(sorted(urls)) != FROZEN_TARGET_URLS_SHA256
    ):
        raise G3SourceCatalogV2Error(
            "capture dependency does not exactly cover the frozen 42 targets"
        )
    dependency = _dependency(
        artifact_kind=CAPTURE_ARTIFACT_KIND,
        artifact_id=report.get("capture_id"),
        payload_manifest_sha256=report.get("payload_manifest_sha256"),
    )
    return dependency, root, manifest


def _load_manual_dependency(
    intake_directory: str | Path,
) -> tuple[dict[str, str], Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from build_lex.terminology_g3_manual_source_intake import (
            validate_manual_source_intake,
        )

        report = validate_manual_source_intake(intake_directory)
    except Exception as exc:
        raise G3SourceCatalogV2Error(
            "manual-source intake dependency failed independent validation"
        ) from exc
    root = Path(report["target"]).resolve()
    manifest = _object(report.get("manifest"), "manual intake manifest")
    components = _objects(report.get("components"), "manual intake components")
    aliases = _objects(report.get("aliases"), "manual intake aliases")
    dependency = _dependency(
        artifact_kind=MANUAL_INTAKE_ARTIFACT_KIND,
        artifact_id=report.get("intake_id"),
        payload_manifest_sha256=report.get("payload_manifest_sha256"),
    )
    component_ids = [row.get("component_id") for row in components]
    if (
        len(component_ids) != len(set(component_ids))
        or set(component_ids) != MANUAL_COMPONENT_IDS
    ):
        raise G3SourceCatalogV2Error(
            "manual intake component coverage differs from the frozen catalog"
        )
    return dependency, root, manifest, components, aliases


def _capture_audit_row(page: Mapping[str, Any]) -> dict[str, Any]:
    requested_url = _url(page.get("requested_url"), "capture requested URL")
    final_url = page.get("final_url")
    if final_url is not None:
        final_url = _url(final_url, "capture final URL")
    response_sha = page.get("response_sha256")
    projection_sha = page.get("text_projection_sha256")
    for label, value in (
        ("capture response SHA", response_sha),
        ("capture projection SHA", projection_sha),
    ):
        if value is not None and (
            not isinstance(value, str) or not SHA256_RE.fullmatch(value)
        ):
            raise G3SourceCatalogV2Error(f"{label} is invalid")
    analysis = page.get("content_analysis")
    content_validation = None
    media_kind = None
    if isinstance(analysis, Mapping):
        content_validation = analysis.get("content_validation")
        media_kind = analysis.get("media_kind")
    return {
        "requested_url": requested_url,
        "fetch_status": page.get("fetch_status"),
        "final_url": final_url,
        "status_code": page.get("status_code"),
        "response_saved": page.get("response_file") is not None,
        "response_size_bytes": page.get("response_size_bytes"),
        "response_sha256": response_sha,
        "normalized_projection_available": page.get("text_projection_file")
        is not None,
        "text_projection_sha256": projection_sha,
        "content_validation": content_validation,
        "media_kind": media_kind,
        "wikimedia_revision_id": page.get("wikimedia_revision_id"),
        "transport_downgrade": page.get("transport_downgrade"),
        "cross_host_redirect": page.get("cross_host_redirect"),
        "redirect_chain_sha256": canonical_sha256(page.get("redirect_chain", [])),
    }


def _manual_alias_projection(
    components: Sequence[Mapping[str, Any]],
    aliases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_component = {str(row["component_id"]): row for row in components}
    if len(by_component) != len(components):
        raise G3SourceCatalogV2Error("manual intake component IDs are duplicated")
    projected: list[dict[str, Any]] = []
    alias_ids: set[str] = set()
    expected_alias_ids: set[str] = set()
    for component in components:
        raw_aliases = component.get("aliases")
        if not isinstance(raw_aliases, list) or any(
            not isinstance(alias_id, str) or not alias_id for alias_id in raw_aliases
        ):
            raise G3SourceCatalogV2Error("manual component aliases are invalid")
        expected_alias_ids.update(raw_aliases)
    for alias in aliases:
        expected_fields = {
            "alias_id",
            "source_id",
            "component_id",
            "attachment_locator",
            "expected_size_bytes",
            "expected_sha256",
        }
        if set(alias) != expected_fields:
            raise G3SourceCatalogV2Error("manual alias fields differ")
        alias_id = alias.get("alias_id")
        component_id = alias.get("component_id")
        if (
            not isinstance(alias_id, str)
            or not alias_id
            or alias_id in alias_ids
            or component_id not in by_component
        ):
            raise G3SourceCatalogV2Error("manual alias identity or component differs")
        alias_ids.add(alias_id)
        component = by_component[str(component_id)]
        if (
            alias.get("source_id") != component.get("source_id")
            or alias.get("expected_sha256") != component.get("raw_sha256")
            or alias.get("expected_size_bytes") != component.get("raw_size_bytes")
        ):
            raise G3SourceCatalogV2Error("manual alias does not replay its component")
        projected.append(
            {
                "alias_id": alias_id,
                "source_id": alias["source_id"],
                "component_id": component_id,
                "requested_url": component["requested_url"],
                "snapshot_url": component["snapshot_url"],
                "attachment_locator_sha256": hashlib.sha256(
                    str(alias["attachment_locator"]).encode("utf-8")
                ).hexdigest(),
                "size_bytes": alias["expected_size_bytes"],
                "raw_sha256": alias["expected_sha256"],
            }
        )
    if alias_ids != expected_alias_ids:
        raise G3SourceCatalogV2Error("manual alias coverage differs from components")
    projected.sort(key=lambda row: row["alias_id"])
    return projected


def _catalog_indexes(
    catalog: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    targets = _objects(catalog.get("targets"), "catalog targets")
    by_url = {str(row["requested_url"]): row for row in targets}
    component_to_target: dict[str, dict[str, Any]] = {}
    component_to_url: dict[str, str] = {}
    for target in targets:
        for component in target["components"]:
            component_id = str(component["component_id"])
            component_to_target[component_id] = target
            component_to_url[component_id] = str(component["component_url"])
    return by_url, component_to_target, component_to_url


def _recorded_transport_downgrade(requested_url: str, final_url: str) -> bool:
    requested = urlsplit(requested_url)
    final = urlsplit(final_url)
    downgraded = requested.scheme == "https" and final.scheme == "http"
    if downgraded and (
        requested.hostname != final.hostname or requested.port != final.port
    ):
        raise G3SourceCatalogV2Error(
            "protocol downgrade is allowed only on the exact requested host/port"
        )
    return downgraded


def _normalized_component(
    *,
    component_id: str,
    source_id: str,
    target: Mapping[str, Any],
    catalog_component: Mapping[str, Any],
    dependency_origin: str,
    requested_url: str,
    final_url: str,
    snapshot_url: str,
    raw_sha256: str,
    normalized_file: str | None,
    normalized_sha256: str | None,
    normalized_size_bytes: int | None,
    normalization_metadata: Mapping[str, Any],
    candidate_projection: Mapping[str, Any] | None,
    source_acquisition_mode: str,
    intake_disposition: str | None,
) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(raw_sha256):
        raise G3SourceCatalogV2Error("selected component raw SHA is invalid")
    if normalized_file is None:
        if normalized_sha256 is not None or normalized_size_bytes is not None:
            raise G3SourceCatalogV2Error("normalized component metadata is incomplete")
    else:
        if (
            not isinstance(normalized_sha256, str)
            or not SHA256_RE.fullmatch(normalized_sha256)
            or isinstance(normalized_size_bytes, bool)
            or not isinstance(normalized_size_bytes, int)
            or normalized_size_bytes <= 0
        ):
            raise G3SourceCatalogV2Error("normalized component hash/size is invalid")
    evidence_eligible = bool(
        normalized_file is not None
        and target["evidence_allowed"]
        and target["source_role"] in EVIDENCE_ROLES
        and target["disposition"] == "accepted"
    )
    return {
        "component_id": component_id,
        "source_id": source_id,
        "target_id": target["target_id"],
        "publisher": target["publisher"],
        "requested_url": requested_url,
        "final_url": final_url,
        "snapshot_url": snapshot_url,
        "acquisition_mode": target["acquisition_mode"],
        "source_acquisition_mode": source_acquisition_mode,
        "source_role": target["source_role"],
        "disposition": target["disposition"],
        "intake_disposition": intake_disposition,
        "component_kind": catalog_component["component_kind"],
        "page_ordinal": catalog_component["page_ordinal"],
        "wikimedia_oldid": catalog_component["wikimedia_oldid"],
        "transport_downgrade": _recorded_transport_downgrade(
            requested_url, final_url
        ),
        "raw_sha256": raw_sha256,
        "normalized_file": normalized_file,
        "normalized_sha256": normalized_sha256,
        "normalized_size_bytes": normalized_size_bytes,
        "normalization_metadata": copy.deepcopy(dict(normalization_metadata)),
        "candidate_projection": (
            None
            if candidate_projection is None
            else copy.deepcopy(dict(candidate_projection))
        ),
        "evidence_eligible": evidence_eligible,
        "human_review_required": evidence_eligible,
        "dependency_origin": dependency_origin,
    }


def _validate_chime_projection(path: Path, metadata: Mapping[str, Any]) -> None:
    value = load_json(path)
    projection = _object(value, "CHIME candidate projection")
    expected_fields = {
        "schema_version",
        "source_component_id",
        "raw_sha256",
        "source_record_count",
        "allowed_types",
        "candidate_count",
        "rows",
    }
    if (
        set(projection) != expected_fields
        or projection.get("schema_version")
        != "wp3-g3-chime-form-candidate-projection/v1"
        or projection.get("source_component_id") != "chime-data-json"
        or projection.get("source_record_count") != 1458
        or projection.get("allowed_types") != ["abbreviation", "homophonic pun"]
        or projection.get("candidate_count") != 185
        or metadata.get("count") != 185
        or metadata.get("allowed_types") != ["abbreviation", "homophonic pun"]
    ):
        raise G3SourceCatalogV2Error("CHIME candidate projection contract differs")
    rows = _objects(projection.get("rows"), "CHIME candidate rows")
    counts: Counter[str] = Counter()
    for row in rows:
        if set(row) != {
            "source_row_ordinal",
            "meme",
            "meaning",
            "origin",
            "type_cn",
            "type_en",
        }:
            raise G3SourceCatalogV2Error("CHIME candidate row fields differ")
        if row["type_en"] not in {"abbreviation", "homophonic pun"}:
            raise G3SourceCatalogV2Error("CHIME candidate type is outside form scope")
        counts[row["type_en"]] += 1
    if counts != Counter({"homophonic pun": 133, "abbreviation": 52}):
        raise G3SourceCatalogV2Error("CHIME candidate type counts differ")


def _validate_chime_provenance(manual_manifest: Mapping[str, Any]) -> None:
    contracts = _objects(
        manual_manifest.get("companion_contracts"), "manual companion contracts"
    )
    matching = [
        row
        for row in contracts
        if row.get("archive_component_id") == "chime-repository-archive"
        and row.get("data_component_id") == "chime-data-json"
        and row.get("license_component_id") == "chime-license"
    ]
    if len(matching) != 1 or matching[0].get("repository_commit") != CHIME_COMMIT:
        raise G3SourceCatalogV2Error("CHIME companion contract or commit differs")


def _assemble_selected_components(
    *,
    catalog: Mapping[str, Any],
    capture_root: Path,
    capture_manifest: Mapping[str, Any],
    manual_root: Path,
    manual_manifest: Mapping[str, Any],
    manual_components: Sequence[Mapping[str, Any]],
    manual_aliases: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Return selected components, coverage, and projected manual aliases.

    Private ``_normalized_*`` and ``_candidate_source`` keys in the returned
    selected rows are copy instructions.  They are removed before publication.
    """

    by_url, component_to_target, _component_to_url = _catalog_indexes(catalog)
    capture_pages = {
        str(row["requested_url"]): row
        for row in _objects(capture_manifest.get("pages"), "capture pages")
    }
    if set(capture_pages) != set(by_url):
        raise G3SourceCatalogV2Error("capture/catalog target coverage differs")
    manual_by_id = {str(row["component_id"]): row for row in manual_components}
    if set(manual_by_id) != MANUAL_COMPONENT_IDS:
        raise G3SourceCatalogV2Error("manual component coverage differs")
    _validate_chime_provenance(manual_manifest)
    if manual_by_id["chime-license"].get("raw_sha256") != CHIME_LICENSE_SHA256:
        raise G3SourceCatalogV2Error("CHIME LICENSE payload differs")
    for component_id, oldid in WIKIMEDIA_OLDIDS.items():
        component = manual_by_id[component_id]
        if component.get("wikimedia_revision_id") != oldid:
            raise G3SourceCatalogV2Error("manual Wikimedia revision differs")
        snapshot_url = _url(
            component.get("snapshot_url"), "manual Wikimedia snapshot URL"
        )
        if parse_qs(urlsplit(snapshot_url).query).get("oldid") != [oldid]:
            raise G3SourceCatalogV2Error("manual Wikimedia snapshot is not oldid-pinned")

    alias_projection = _manual_alias_projection(manual_components, manual_aliases)
    manual_aliases_by_component: dict[str, list[str]] = {}
    for alias in alias_projection:
        component_id = str(alias["component_id"])
        if component_id not in component_to_target:
            raise G3SourceCatalogV2Error("manual alias component is outside the catalog")
        manual_aliases_by_component.setdefault(component_id, []).append(
            str(alias["alias_id"])
        )

    selected: list[dict[str, Any]] = []
    selected_by_target: dict[str, list[str]] = {}
    for target in catalog["targets"]:
        if target["disposition"] != "accepted":
            continue
        capture_page = capture_pages[target["requested_url"]]
        for catalog_component in target["components"]:
            component_id = str(catalog_component["component_id"])
            if component_id in manual_by_id:
                source = manual_by_id[component_id]
                requested_url = _url(
                    source.get("requested_url"), "manual component requested URL"
                )
                final_url = _url(source.get("final_url"), "manual component final URL")
                snapshot_url = _url(
                    source.get("snapshot_url"), "manual component snapshot URL"
                )
                if catalog_component["component_kind"] in {"page", "paper", "dataset"}:
                    expected_component_url = str(catalog_component["component_url"])
                    if (
                        catalog_component["wikimedia_oldid"] is None
                        and snapshot_url != expected_component_url
                        and requested_url != expected_component_url
                    ):
                        raise G3SourceCatalogV2Error(
                            "manual component URL differs from the catalog component"
                        )
                normalized_logical = source.get("normalized_text_file")
                if normalized_logical is None:
                    raise G3SourceCatalogV2Error(
                        f"accepted manual component {component_id} lacks normalized text"
                    )
                normalized_source = _safe_file(
                    manual_root,
                    normalized_logical,
                    label="manual normalized component",
                )
                normalized_size = source.get("normalized_text_size_bytes")
                normalized_sha = source.get("normalized_text_sha256")
                if (
                    normalized_source.stat().st_size != normalized_size
                    or sha256_file(normalized_source) != normalized_sha
                ):
                    raise G3SourceCatalogV2Error(
                        "manual normalized component does not replay"
                    )
                candidate = source.get("candidate_projection")
                candidate_metadata: dict[str, Any] | None = None
                candidate_source: Path | None = None
                if candidate is not None:
                    source_candidate = _object(candidate, "manual candidate projection")
                    candidate_source = _safe_file(
                        manual_root,
                        source_candidate.get("file"),
                        label="manual candidate projection",
                    )
                    if (
                        candidate_source.stat().st_size != source_candidate.get("size_bytes")
                        or sha256_file(candidate_source) != source_candidate.get("sha256")
                    ):
                        raise G3SourceCatalogV2Error(
                            "manual candidate projection does not replay"
                        )
                    candidate_metadata = {
                        "schema_version": source_candidate["schema_version"],
                        "file": f"candidate_projections/{component_id}.json",
                        "size_bytes": source_candidate["size_bytes"],
                        "sha256": source_candidate["sha256"],
                        "count": source_candidate["count"],
                        "allowed_types": source_candidate["allowed_types"],
                    }
                    _validate_chime_projection(candidate_source, candidate_metadata)
                row = _normalized_component(
                    component_id=component_id,
                    source_id=str(source["source_id"]),
                    target=target,
                    catalog_component=catalog_component,
                    dependency_origin="manual_intake",
                    requested_url=requested_url,
                    final_url=final_url,
                    snapshot_url=snapshot_url,
                    raw_sha256=str(source["raw_sha256"]),
                    normalized_file=f"normalized/{component_id}.txt",
                    normalized_sha256=str(normalized_sha),
                    normalized_size_bytes=int(normalized_size),
                    normalization_metadata=_object(
                        source.get("format_metadata"), "manual normalization metadata"
                    ),
                    candidate_projection=candidate_metadata,
                    source_acquisition_mode=str(source["acquisition_mode"]),
                    intake_disposition=str(source["disposition"]),
                )
                row["_normalized_source"] = str(normalized_source)
                if candidate_source is not None:
                    row["_candidate_source"] = str(candidate_source)
                selected.append(row)
                selected_by_target.setdefault(str(target["target_id"]), []).append(
                    component_id
                )
                continue

            if capture_page.get("fetch_status") != CAPTURE_COMPLETE_STATUS:
                raise G3SourceCatalogV2Error(
                    f"accepted direct target {target['target_id']} lacks a usable capture"
                )
            response_file = capture_page.get("response_file")
            if response_file is None:
                raise G3SourceCatalogV2Error("usable capture lacks a response payload")
            response_path = _safe_file(
                capture_root, response_file, label="capture response"
            )
            response_sha = capture_page.get("response_sha256")
            if sha256_file(response_path) != response_sha:
                raise G3SourceCatalogV2Error("capture response hash differs")
            final_url = _url(capture_page.get("final_url"), "capture final URL")
            projection_file = capture_page.get("text_projection_file")
            normalization_metadata: dict[str, Any]
            normalized_source: Path | None = None
            normalized_bytes: bytes | None = None
            if projection_file is not None:
                normalized_source = _safe_file(
                    capture_root, projection_file, label="capture text projection"
                )
                normalized_sha = sha256_file(normalized_source)
                if normalized_sha != capture_page.get("text_projection_sha256"):
                    raise G3SourceCatalogV2Error("capture text projection hash differs")
                normalized_size = normalized_source.stat().st_size
                normalization_metadata = {
                    "backend": "zero-proxy-capture-visible-text/v1",
                    "capture_text_projection_sha256": normalized_sha,
                }
            else:
                analysis = _object(
                    capture_page.get("content_analysis"), "capture content analysis"
                )
                if analysis.get("media_kind") != "pdf":
                    raise G3SourceCatalogV2Error(
                        "accepted capture lacks normalized text and is not a PDF"
                    )
                try:
                    from build_lex.terminology_g3_manual_source_intake import (
                        normalize_publisher_pdf_bytes,
                    )

                    normalized_bytes, normalization_metadata = (
                        normalize_publisher_pdf_bytes(
                            response_path.read_bytes(),
                            component_id=component_id,
                            requested_url=str(target["requested_url"]),
                            final_url=final_url,
                        )
                    )
                except Exception as exc:
                    raise G3SourceCatalogV2Error(
                        "direct-capture publisher PDF normalization failed"
                    ) from exc
                normalized_sha = hashlib.sha256(normalized_bytes).hexdigest()
                normalized_size = len(normalized_bytes)
            row = _normalized_component(
                component_id=component_id,
                source_id=str(target["target_id"]),
                target=target,
                catalog_component=catalog_component,
                dependency_origin="zero_proxy_capture",
                requested_url=str(target["requested_url"]),
                final_url=final_url,
                snapshot_url=final_url,
                raw_sha256=str(response_sha),
                normalized_file=f"normalized/{component_id}.txt",
                normalized_sha256=str(normalized_sha),
                normalized_size_bytes=int(normalized_size),
                normalization_metadata=normalization_metadata,
                candidate_projection=None,
                source_acquisition_mode="direct_fetch",
                intake_disposition=None,
            )
            if normalized_source is not None:
                row["_normalized_source"] = str(normalized_source)
            else:
                row["_normalized_bytes"] = normalized_bytes
            selected.append(row)
            selected_by_target.setdefault(str(target["target_id"]), []).append(
                component_id
            )

    selected_ids = [str(row["component_id"]) for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise G3SourceCatalogV2Error("selected component IDs are duplicated")
    coverage: list[dict[str, Any]] = []
    aliases_by_target: dict[str, list[str]] = {}
    for component_id, alias_ids in manual_aliases_by_component.items():
        target_id = str(component_to_target[component_id]["target_id"])
        aliases_by_target.setdefault(target_id, []).extend(alias_ids)
    for target in sorted(catalog["targets"], key=lambda row: row["target_id"]):
        target_id = str(target["target_id"])
        chosen = sorted(selected_by_target.get(target_id, []))
        if target["disposition"] == "accepted" and not chosen:
            raise G3SourceCatalogV2Error(
                f"accepted target {target_id} has no selected normalized component"
            )
        if target["disposition"] != "accepted" and chosen:
            raise G3SourceCatalogV2Error(
                "non-accepted target unexpectedly selected a component"
            )
        coverage.append(
            {
                "target_id": target_id,
                "requested_url": target["requested_url"],
                "acquisition_mode": target["acquisition_mode"],
                "source_role": target["source_role"],
                "disposition": target["disposition"],
                "disposition_reason": target["disposition_reason"],
                "evidence_allowed": target["evidence_allowed"],
                "catalog_component_ids": sorted(
                    component["component_id"] for component in target["components"]
                ),
                "selected_component_ids": chosen,
                "manual_alias_ids": sorted(aliases_by_target.get(target_id, [])),
                "capture_audit": _capture_audit_row(
                    capture_pages[str(target["requested_url"])]
                ),
            }
        )
    selected.sort(key=lambda row: row["component_id"])
    return selected, coverage, alias_projection


def _public_component(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if not str(key).startswith("_")
    }


def _build_acquisition_receipt(
    *,
    catalog: Mapping[str, Any],
    capture_dependency: Mapping[str, Any],
    manual_dependency: Mapping[str, Any],
    coverage: Sequence[Mapping[str, Any]],
    manual_aliases: Sequence[Mapping[str, Any]],
    selected_components: Sequence[Mapping[str, Any]],
    workspace_root: Path,
) -> dict[str, Any]:
    ordered_coverage = [copy.deepcopy(dict(row)) for row in coverage]
    ordered_aliases = [copy.deepcopy(dict(row)) for row in manual_aliases]
    target_ids = [str(row["target_id"]) for row in ordered_coverage]
    if (
        len(ordered_coverage) != FROZEN_TARGET_COUNT
        or len(set(target_ids)) != FROZEN_TARGET_COUNT
    ):
        raise G3SourceCatalogV2Error("acquisition receipt target coverage differs")
    schema_hashes = _schema_hashes(workspace_root)
    identity = {
        "schema_version": ACQUISITION_RECEIPT_SCHEMA_VERSION,
        "artifact_kind": ACQUISITION_RECEIPT_ARTIFACT_KIND,
        "catalog_id": CATALOG_ID,
        "catalog_sha256": canonical_sha256(catalog),
        "catalog_schema_sha256": schema_hashes["catalog_schema_sha256"],
        "capture_dependency": copy.deepcopy(dict(capture_dependency)),
        "manual_intake_dependency": copy.deepcopy(dict(manual_dependency)),
        "bundle_builder_network_access_performed": False,
        "complete": True,
        "target_count": len(ordered_coverage),
        "target_ids": target_ids,
        "target_urls_sha256": FROZEN_TARGET_URLS_SHA256,
        "manual_alias_count": len(ordered_aliases),
        "manual_aliases": ordered_aliases,
        "manual_aliases_sha256": canonical_sha256(ordered_aliases),
        "selected_component_count": len(selected_components),
        "target_coverage": ordered_coverage,
        "target_coverage_sha256": canonical_sha256(ordered_coverage),
    }
    receipt = {
        **identity,
        "acquisition_receipt_id": ACQUISITION_RECEIPT_ID_PREFIX
        + canonical_sha256(identity),
    }
    _validate_schema(receipt, workspace_root / ACQUISITION_RECEIPT_SCHEMA_PATH)
    return receipt


def _validate_components_document(
    document: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    bundle_root: Path,
) -> list[dict[str, Any]]:
    if set(document) != {"schema_version", "components"} or document.get(
        "schema_version"
    ) != COMPONENTS_SCHEMA_VERSION:
        raise G3SourceCatalogV2Error("selected-components document identity differs")
    components = _objects(document.get("components"), "selected components")
    if not components:
        raise G3SourceCatalogV2Error("source bundle has no selected components")
    by_url, component_to_target, _component_to_url = _catalog_indexes(catalog)
    del by_url
    expected_fields = {
        "component_id",
        "source_id",
        "target_id",
        "publisher",
        "requested_url",
        "final_url",
        "snapshot_url",
        "acquisition_mode",
        "source_acquisition_mode",
        "source_role",
        "disposition",
        "intake_disposition",
        "component_kind",
        "page_ordinal",
        "wikimedia_oldid",
        "transport_downgrade",
        "raw_sha256",
        "normalized_file",
        "normalized_sha256",
        "normalized_size_bytes",
        "normalization_metadata",
        "candidate_projection",
        "evidence_eligible",
        "human_review_required",
        "dependency_origin",
    }
    component_ids: set[str] = set()
    total_size = 0
    selected_by_target: dict[str, list[str]] = {}
    for component in components:
        if set(component) != expected_fields:
            raise G3SourceCatalogV2Error("selected component fields differ")
        component_id = component.get("component_id")
        if (
            not isinstance(component_id, str)
            or component_id in component_ids
            or component_id not in component_to_target
        ):
            raise G3SourceCatalogV2Error("selected component identity differs")
        component_ids.add(component_id)
        target = component_to_target[component_id]
        catalog_component = next(
            row for row in target["components"] if row["component_id"] == component_id
        )
        if (
            target["disposition"] != "accepted"
            or component.get("target_id") != target["target_id"]
            or component.get("publisher") != target["publisher"]
            or component.get("acquisition_mode") != target["acquisition_mode"]
            or component.get("source_role") != target["source_role"]
            or component.get("disposition") != target["disposition"]
            or component.get("component_kind") != catalog_component["component_kind"]
            or component.get("page_ordinal") != catalog_component["page_ordinal"]
            or component.get("wikimedia_oldid")
            != catalog_component["wikimedia_oldid"]
        ):
            raise G3SourceCatalogV2Error(
                "selected component differs from its catalog target"
            )
        for label in ("requested_url", "final_url", "snapshot_url"):
            _url(component.get(label), f"component {label}")
        expected_downgrade = _recorded_transport_downgrade(
            str(component["requested_url"]), str(component["final_url"])
        )
        if component.get("transport_downgrade") is not expected_downgrade:
            raise G3SourceCatalogV2Error("component protocol downgrade metadata differs")
        raw_sha = component.get("raw_sha256")
        if not isinstance(raw_sha, str) or not SHA256_RE.fullmatch(raw_sha):
            raise G3SourceCatalogV2Error("component raw SHA is invalid")
        normalized_file = component.get("normalized_file")
        if normalized_file != f"normalized/{component_id}.txt":
            raise G3SourceCatalogV2Error("component normalized path differs")
        normalized_path = _safe_file(
            bundle_root, normalized_file, label="bundle normalized component"
        )
        size = normalized_path.stat().st_size
        total_size += size
        if (
            size <= 0
            or size > MAX_NORMALIZED_COMPONENT_BYTES
            or total_size > MAX_TOTAL_NORMALIZED_BYTES
            or component.get("normalized_size_bytes") != size
            or component.get("normalized_sha256") != sha256_file(normalized_path)
        ):
            raise G3SourceCatalogV2Error("normalized component payload does not replay")
        try:
            text = normalized_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise G3SourceCatalogV2Error(
                "normalized component is not UTF-8 text"
            ) from exc
        if not text.strip():
            raise G3SourceCatalogV2Error("normalized component text is empty")
        if not isinstance(component.get("normalization_metadata"), Mapping):
            raise G3SourceCatalogV2Error("normalization metadata is malformed")
        evidence_expected = bool(
            target["evidence_allowed"] and target["source_role"] in EVIDENCE_ROLES
        )
        if (
            component.get("evidence_eligible") is not evidence_expected
            or component.get("human_review_required") is not evidence_expected
        ):
            raise G3SourceCatalogV2Error("component relation-review gate differs")
        dependency_origin = component.get("dependency_origin")
        if component_id in MANUAL_COMPONENT_IDS:
            if dependency_origin != "manual_intake":
                raise G3SourceCatalogV2Error("manual component origin differs")
        elif dependency_origin != "zero_proxy_capture":
            raise G3SourceCatalogV2Error("capture component origin differs")
        candidate = component.get("candidate_projection")
        if candidate is not None:
            candidate_row = _object(candidate, "bundle candidate projection")
            if set(candidate_row) != {
                "schema_version",
                "file",
                "size_bytes",
                "sha256",
                "count",
                "allowed_types",
            }:
                raise G3SourceCatalogV2Error("candidate projection fields differ")
            if component_id != "chime-data-json" or component["source_role"] != "candidate_pool":
                raise G3SourceCatalogV2Error(
                    "only the frozen CHIME candidate component may carry a projection"
                )
            expected_candidate_file = f"candidate_projections/{component_id}.json"
            if candidate_row.get("file") != expected_candidate_file:
                raise G3SourceCatalogV2Error("candidate projection path differs")
            candidate_path = _safe_file(
                bundle_root, expected_candidate_file, label="candidate projection"
            )
            if (
                candidate_path.stat().st_size != candidate_row.get("size_bytes")
                or sha256_file(candidate_path) != candidate_row.get("sha256")
            ):
                raise G3SourceCatalogV2Error("candidate projection payload differs")
            _validate_chime_projection(candidate_path, candidate_row)
        elif component_id == "chime-data-json":
            raise G3SourceCatalogV2Error("CHIME component lacks its candidate projection")
        selected_by_target.setdefault(str(target["target_id"]), []).append(component_id)
    if components != sorted(components, key=lambda row: row["component_id"]):
        raise G3SourceCatalogV2Error("selected components are not canonically ordered")
    for target in catalog["targets"]:
        selected = selected_by_target.get(str(target["target_id"]), [])
        if target["disposition"] == "accepted" and not selected:
            raise G3SourceCatalogV2Error("accepted target lacks a bundle component")
        if target["disposition"] != "accepted" and selected:
            raise G3SourceCatalogV2Error("non-accepted target has a bundle component")
    return components


def _validate_acquisition_receipt(
    receipt: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    components: Sequence[Mapping[str, Any]],
    workspace_root: Path,
    capture_dependency: Mapping[str, Any] | None = None,
    capture_manifest: Mapping[str, Any] | None = None,
    manual_dependency: Mapping[str, Any] | None = None,
    manual_components: Sequence[Mapping[str, Any]] | None = None,
    manual_aliases: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = _object(receipt, "v2 acquisition receipt")
    _validate_schema(normalized, workspace_root / ACQUISITION_RECEIPT_SCHEMA_PATH)
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "catalog_id",
        "catalog_sha256",
        "catalog_schema_sha256",
        "capture_dependency",
        "manual_intake_dependency",
        "bundle_builder_network_access_performed",
        "complete",
        "target_count",
        "target_ids",
        "target_urls_sha256",
        "manual_alias_count",
        "manual_aliases",
        "manual_aliases_sha256",
        "selected_component_count",
        "target_coverage",
        "target_coverage_sha256",
        "acquisition_receipt_id",
    }
    if set(normalized) != expected_fields:
        raise G3SourceCatalogV2Error("acquisition receipt fields differ")
    schema_hashes = _schema_hashes(workspace_root)
    aliases = _objects(normalized.get("manual_aliases"), "receipt manual aliases")
    coverage = _objects(normalized.get("target_coverage"), "receipt target coverage")
    identity = {
        key: copy.deepcopy(value)
        for key, value in normalized.items()
        if key != "acquisition_receipt_id"
    }
    expected_id = ACQUISITION_RECEIPT_ID_PREFIX + canonical_sha256(identity)
    if (
        normalized.get("schema_version") != ACQUISITION_RECEIPT_SCHEMA_VERSION
        or normalized.get("artifact_kind") != ACQUISITION_RECEIPT_ARTIFACT_KIND
        or normalized.get("catalog_id") != CATALOG_ID
        or normalized.get("catalog_sha256") != canonical_sha256(catalog)
        or normalized.get("catalog_schema_sha256")
        != schema_hashes["catalog_schema_sha256"]
        or normalized.get("bundle_builder_network_access_performed") is not False
        or normalized.get("complete") is not True
        or normalized.get("target_count") != FROZEN_TARGET_COUNT
        or normalized.get("target_urls_sha256") != FROZEN_TARGET_URLS_SHA256
        or normalized.get("manual_alias_count") != len(aliases)
        or normalized.get("manual_aliases_sha256") != canonical_sha256(aliases)
        or normalized.get("selected_component_count") != len(components)
        or normalized.get("target_coverage_sha256") != canonical_sha256(coverage)
        or normalized.get("acquisition_receipt_id") != expected_id
    ):
        raise G3SourceCatalogV2Error("acquisition receipt bindings differ")
    for key, kind in (
        ("capture_dependency", CAPTURE_ARTIFACT_KIND),
        ("manual_intake_dependency", MANUAL_INTAKE_ARTIFACT_KIND),
    ):
        dependency = _object(normalized.get(key), key)
        if (
            set(dependency)
            != {"artifact_kind", "artifact_id", "payload_manifest_sha256"}
            or dependency.get("artifact_kind") != kind
            or not isinstance(dependency.get("artifact_id"), str)
            or not isinstance(dependency.get("payload_manifest_sha256"), str)
            or not SHA256_RE.fullmatch(dependency["payload_manifest_sha256"])
        ):
            raise G3SourceCatalogV2Error("acquisition dependency projection differs")
    catalog_by_id = {str(row["target_id"]): row for row in catalog["targets"]}
    components_by_target: dict[str, list[str]] = {}
    for component in components:
        components_by_target.setdefault(str(component["target_id"]), []).append(
            str(component["component_id"])
        )
    if len(coverage) != FROZEN_TARGET_COUNT:
        raise G3SourceCatalogV2Error("acquisition target coverage count differs")
    coverage_ids: list[str] = []
    alias_ids = {str(row.get("alias_id")) for row in aliases}
    if len(alias_ids) != len(aliases) or None in alias_ids:
        raise G3SourceCatalogV2Error("receipt manual aliases are duplicated")
    for row in coverage:
        expected_coverage_fields = {
            "target_id",
            "requested_url",
            "acquisition_mode",
            "source_role",
            "disposition",
            "disposition_reason",
            "evidence_allowed",
            "catalog_component_ids",
            "selected_component_ids",
            "manual_alias_ids",
            "capture_audit",
        }
        if set(row) != expected_coverage_fields:
            raise G3SourceCatalogV2Error("acquisition target-coverage fields differ")
        target_id = str(row.get("target_id"))
        target = catalog_by_id.get(target_id)
        if target is None:
            raise G3SourceCatalogV2Error("receipt target is outside the catalog")
        coverage_ids.append(target_id)
        expected_selected = sorted(components_by_target.get(target_id, []))
        if (
            any(
                row.get(field) != target[field]
                for field in (
                    "requested_url",
                    "acquisition_mode",
                    "source_role",
                    "disposition",
                    "disposition_reason",
                    "evidence_allowed",
                )
            )
            or row.get("catalog_component_ids")
            != sorted(component["component_id"] for component in target["components"])
            or row.get("selected_component_ids") != expected_selected
        ):
            raise G3SourceCatalogV2Error("receipt target coverage differs from bundle/catalog")
        row_alias_ids = row.get("manual_alias_ids")
        if (
            not isinstance(row_alias_ids, list)
            or row_alias_ids != sorted(row_alias_ids)
            or any(alias_id not in alias_ids for alias_id in row_alias_ids)
        ):
            raise G3SourceCatalogV2Error("receipt target manual-alias coverage differs")
        audit = _object(row.get("capture_audit"), "receipt capture audit")
        if audit.get("requested_url") != target["requested_url"]:
            raise G3SourceCatalogV2Error("receipt capture audit target differs")
        if audit.get("fetch_status") != CAPTURE_COMPLETE_STATUS and expected_selected:
            if not any(
                component["dependency_origin"] == "manual_intake"
                for component in components
                if component["target_id"] == target_id
            ):
                raise G3SourceCatalogV2Error(
                    "failed direct capture was incorrectly used as evidence"
                )
    if (
        coverage_ids != sorted(catalog_by_id)
        or normalized.get("target_ids") != coverage_ids
    ):
        raise G3SourceCatalogV2Error("receipt target ordering or identity differs")
    used_alias_ids = {
        alias_id for row in coverage for alias_id in row["manual_alias_ids"]
    }
    if used_alias_ids != alias_ids:
        raise G3SourceCatalogV2Error("receipt does not cover every manual alias")

    if capture_dependency is not None:
        if normalized["capture_dependency"] != dict(capture_dependency):
            raise G3SourceCatalogV2Error("capture dependency differs from receipt")
        if capture_manifest is None:
            raise G3SourceCatalogV2Error("capture manifest is required for replay")
        capture_pages = {
            str(row["requested_url"]): row
            for row in _objects(capture_manifest.get("pages"), "capture pages")
        }
        for row in coverage:
            if row["capture_audit"] != _capture_audit_row(
                capture_pages[row["requested_url"]]
            ):
                raise G3SourceCatalogV2Error("capture audit does not replay upstream")
    if manual_dependency is not None:
        if normalized["manual_intake_dependency"] != dict(manual_dependency):
            raise G3SourceCatalogV2Error("manual dependency differs from receipt")
        if manual_components is None or manual_aliases is None:
            raise G3SourceCatalogV2Error("manual inputs are required for replay")
        expected_aliases = _manual_alias_projection(manual_components, manual_aliases)
        if aliases != expected_aliases:
            raise G3SourceCatalogV2Error("manual aliases do not replay upstream")
    return normalized


def build_public_source_bundle_v2(
    *,
    workspace_root: str | Path,
    catalog_path: str | Path,
    capture_directory: str | Path,
    manual_intake_directory: str | Path,
    output_root: str | Path,
    write_ref: str | Path | None = None,
) -> dict[str, Any]:
    """Build the v2 bundle offline from two independently validated artifacts."""

    root = Path(workspace_root).resolve()
    catalog = load_source_catalog_v2(catalog_path, workspace_root=root)
    capture_dependency, capture_root, capture_manifest = _load_capture_dependency(
        capture_directory
    )
    (
        manual_dependency,
        manual_root,
        manual_manifest,
        manual_components,
        manual_aliases,
    ) = _load_manual_dependency(manual_intake_directory)
    selected_private, coverage, projected_aliases = _assemble_selected_components(
        catalog=catalog,
        capture_root=capture_root,
        capture_manifest=capture_manifest,
        manual_root=manual_root,
        manual_manifest=manual_manifest,
        manual_components=manual_components,
        manual_aliases=manual_aliases,
    )
    components = [_public_component(row) for row in selected_private]
    components_document = {
        "schema_version": COMPONENTS_SCHEMA_VERSION,
        "components": components,
    }
    _validate_schema(components_document, root / COMPONENTS_SCHEMA_PATH)
    receipt = _build_acquisition_receipt(
        catalog=catalog,
        capture_dependency=capture_dependency,
        manual_dependency=manual_dependency,
        coverage=coverage,
        manual_aliases=projected_aliases,
        selected_components=components,
        workspace_root=root,
    )
    _validate_acquisition_receipt(
        receipt,
        catalog=catalog,
        components=components,
        workspace_root=root,
        capture_dependency=capture_dependency,
        capture_manifest=capture_manifest,
        manual_dependency=manual_dependency,
        manual_components=manual_components,
        manual_aliases=manual_aliases,
    )
    schema_hashes = _schema_hashes(root)
    identity = {
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "artifact_kind": SOURCE_BUNDLE_ARTIFACT_KIND,
        "catalog_id": CATALOG_ID,
        "catalog_sha256": canonical_sha256(catalog),
        "catalog_schema_sha256": schema_hashes["catalog_schema_sha256"],
        "acquisition_receipt_id": receipt["acquisition_receipt_id"],
        "acquisition_receipt_sha256": canonical_sha256(receipt),
        "acquisition_receipt_schema_sha256": schema_hashes[
            "acquisition_receipt_schema_sha256"
        ],
        "components_sha256": canonical_sha256(components_document),
        "components_schema_sha256": schema_hashes["components_schema_sha256"],
        "source_count": len({row["source_id"] for row in components}),
        "component_count": len(components),
        "normalized_component_count": len(components),
        "candidate_projection_count": sum(
            row["candidate_projection"] is not None for row in components
        ),
        "source_policy": SOURCE_POLICY_ID,
        "reference_role": REFERENCE_ROLE,
        "scope": SCOPE,
        "scientific_eligible": False,
        "sealed": False,
        "builder_implementation_sha256": sha256_file(Path(__file__)),
    }
    source_bundle_id = SOURCE_BUNDLE_ID_PREFIX + canonical_sha256(identity)
    manifest = {**identity, "source_bundle_id": source_bundle_id}
    _validate_schema(manifest, root / SOURCE_BUNDLE_SCHEMA_PATH)
    output_parent = Path(output_root).resolve()
    target = output_parent / source_bundle_id
    if not target.exists():
        staging = new_staging_directory(output_parent, source_bundle_id)
        try:
            write_canonical_json(staging / "manifest.json", manifest)
            write_canonical_json(staging / "catalog.json", catalog)
            write_canonical_json(staging / "acquisition_receipt.json", receipt)
            write_canonical_json(staging / "components.json", components_document)
            normalized_root = staging / "normalized"
            normalized_root.mkdir(mode=0o700)
            candidate_root = staging / "candidate_projections"
            if any(row.get("_candidate_source") for row in selected_private):
                candidate_root.mkdir(mode=0o700)
            for row in selected_private:
                destination = staging / str(row["normalized_file"])
                if "_normalized_source" in row:
                    shutil.copyfile(str(row["_normalized_source"]), destination)
                else:
                    payload = row.get("_normalized_bytes")
                    if not isinstance(payload, bytes):
                        raise G3SourceCatalogV2Error(
                            "selected component lacks a normalized copy source"
                        )
                    destination.write_bytes(payload)
                os.chmod(destination, 0o600)
                candidate_source = row.get("_candidate_source")
                if candidate_source is not None:
                    candidate_destination = staging / str(
                        row["candidate_projection"]["file"]
                    )
                    shutil.copyfile(str(candidate_source), candidate_destination)
                    os.chmod(candidate_destination, 0o600)
            rubric = staging / "rubric"
            rubric.mkdir(mode=0o700)
            handbook = _safe_workspace_file(
                root, catalog["handbook"]["path"], label="handbook"
            )
            shutil.copyfile(handbook, rubric / "handbook.md")
            os.chmod(rubric / "handbook.md", 0o600)
            for path in (
                staging / "manifest.json",
                staging / "catalog.json",
                staging / "acquisition_receipt.json",
                staging / "components.json",
            ):
                os.chmod(path, 0o600)
            payload_hash = finalize_target_atomic(
                staging,
                target,
                validate_staging=lambda directory: validate_public_source_bundle_v2(
                    directory,
                    workspace_root=root,
                    require_current_implementation=True,
                ),
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    else:
        existing = validate_public_source_bundle_v2(
            target,
            workspace_root=root,
            require_current_implementation=True,
            capture_directory=capture_directory,
            manual_intake_directory=manual_intake_directory,
        )
        payload_hash = existing["payload_manifest_sha256"]
    if write_ref is not None:
        write_locator_ref(
            write_ref,
            artifact_kind=SOURCE_BUNDLE_ARTIFACT_KIND,
            artifact_id=source_bundle_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    return {
        "source_bundle_id": source_bundle_id,
        "target": str(target),
        "payload_manifest_sha256": payload_hash,
        "manifest": manifest,
        "catalog": catalog,
        "acquisition_receipt": receipt,
        "components": components,
    }


def validate_public_source_bundle_v2(
    source_bundle_dir: str | Path,
    *,
    workspace_root: str | Path,
    require_current_implementation: bool = True,
    capture_directory: str | Path | None = None,
    manual_intake_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Independently replay one v2 bundle, optionally through both upstreams."""

    root = Path(workspace_root).resolve()
    target = Path(source_bundle_dir).resolve()
    try:
        payload_hash = validate_payload_manifest(target)
    except TrainingArtifactError as exc:
        raise G3SourceCatalogV2Error(str(exc)) from exc
    manifest = _load_object(target / "manifest.json", "v2 source-bundle manifest")
    catalog = load_source_catalog_v2(target / "catalog.json", workspace_root=root)
    receipt = _load_object(
        target / "acquisition_receipt.json", "v2 acquisition receipt"
    )
    components_document = _load_object(
        target / "components.json", "v2 selected components"
    )
    _validate_schema(components_document, root / COMPONENTS_SCHEMA_PATH)
    components = _validate_components_document(
        components_document, catalog=catalog, bundle_root=target
    )
    expected_files = {
        "manifest.json",
        "catalog.json",
        "acquisition_receipt.json",
        "components.json",
        "rubric/handbook.md",
        "payload_manifest.json",
    }
    expected_files.update(str(row["normalized_file"]) for row in components)
    expected_files.update(
        str(row["candidate_projection"]["file"])
        for row in components
        if row["candidate_projection"] is not None
    )
    try:
        ensure_exact_file_set(target, expected_files)
    except TrainingArtifactError as exc:
        raise G3SourceCatalogV2Error(str(exc)) from exc
    if sha256_file(target / "rubric/handbook.md") != catalog["handbook"]["sha256"]:
        raise G3SourceCatalogV2Error("frozen rubric hash differs")

    capture_dependency = None
    capture_manifest = None
    manual_dependency = None
    manual_components = None
    manual_aliases = None
    if (capture_directory is None) != (manual_intake_directory is None):
        raise G3SourceCatalogV2Error(
            "upstream replay requires both capture and manual intake dependencies"
        )
    if capture_directory is not None:
        capture_dependency, _capture_root, capture_manifest = _load_capture_dependency(
            capture_directory
        )
        (
            manual_dependency,
            _manual_root,
            _manual_manifest,
            manual_components,
            manual_aliases,
        ) = _load_manual_dependency(manual_intake_directory)
    receipt = _validate_acquisition_receipt(
        receipt,
        catalog=catalog,
        components=components,
        workspace_root=root,
        capture_dependency=capture_dependency,
        capture_manifest=capture_manifest,
        manual_dependency=manual_dependency,
        manual_components=manual_components,
        manual_aliases=manual_aliases,
    )
    schema_hashes = _schema_hashes(root)
    identity = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "source_bundle_id"
    }
    source_bundle_id = SOURCE_BUNDLE_ID_PREFIX + canonical_sha256(identity)
    _validate_schema(manifest, root / SOURCE_BUNDLE_SCHEMA_PATH)
    expected_manifest_fields = {
        "schema_version",
        "artifact_kind",
        "catalog_id",
        "catalog_sha256",
        "catalog_schema_sha256",
        "acquisition_receipt_id",
        "acquisition_receipt_sha256",
        "acquisition_receipt_schema_sha256",
        "components_sha256",
        "components_schema_sha256",
        "source_count",
        "component_count",
        "normalized_component_count",
        "candidate_projection_count",
        "source_policy",
        "reference_role",
        "scope",
        "scientific_eligible",
        "sealed",
        "builder_implementation_sha256",
        "source_bundle_id",
    }
    if set(manifest) != expected_manifest_fields:
        raise G3SourceCatalogV2Error("v2 source-bundle manifest fields differ")
    if (
        manifest.get("schema_version") != SOURCE_BUNDLE_SCHEMA_VERSION
        or manifest.get("artifact_kind") != SOURCE_BUNDLE_ARTIFACT_KIND
        or manifest.get("catalog_id") != CATALOG_ID
        or manifest.get("catalog_sha256") != canonical_sha256(catalog)
        or manifest.get("catalog_schema_sha256")
        != schema_hashes["catalog_schema_sha256"]
        or manifest.get("acquisition_receipt_id")
        != receipt["acquisition_receipt_id"]
        or manifest.get("acquisition_receipt_sha256") != canonical_sha256(receipt)
        or manifest.get("acquisition_receipt_schema_sha256")
        != schema_hashes["acquisition_receipt_schema_sha256"]
        or manifest.get("components_sha256")
        != canonical_sha256(components_document)
        or manifest.get("components_schema_sha256")
        != schema_hashes["components_schema_sha256"]
        or manifest.get("source_count")
        != len({row["source_id"] for row in components})
        or manifest.get("component_count") != len(components)
        or manifest.get("normalized_component_count") != len(components)
        or manifest.get("candidate_projection_count")
        != sum(row["candidate_projection"] is not None for row in components)
        or manifest.get("source_policy") != SOURCE_POLICY_ID
        or manifest.get("reference_role") != REFERENCE_ROLE
        or manifest.get("scope") != SCOPE
        or manifest.get("scientific_eligible") is not False
        or manifest.get("sealed") is not False
        or manifest.get("source_bundle_id") != source_bundle_id
        or not _target_name_matches(target, source_bundle_id)
    ):
        raise G3SourceCatalogV2Error("v2 source-bundle manifest bindings differ")
    if require_current_implementation and manifest.get(
        "builder_implementation_sha256"
    ) != sha256_file(Path(__file__)):
        raise G3SourceCatalogV2Error("v2 source-bundle implementation drifted")
    return {
        "source_bundle_id": source_bundle_id,
        "target": str(target),
        "payload_manifest_sha256": payload_hash,
        "manifest": manifest,
        "catalog": catalog,
        "acquisition_receipt": receipt,
        "components": components,
    }


__all__ = [
    "ACQUISITION_RECEIPT_ARTIFACT_KIND",
    "ACQUISITION_RECEIPT_SCHEMA_VERSION",
    "CATALOG_ID",
    "CATALOG_SCHEMA_VERSION",
    "COMPONENTS_SCHEMA_VERSION",
    "G3SourceCatalogV2Error",
    "SOURCE_BUNDLE_ARTIFACT_KIND",
    "SOURCE_BUNDLE_SCHEMA_VERSION",
    "build_public_source_bundle_v2",
    "load_source_catalog_v2",
    "validate_public_source_bundle_v2",
]
