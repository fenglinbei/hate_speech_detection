"""Closed-source-catalog lifecycle for the WP3 G3 form reference.

Artifact building is intentionally offline and accepts only snapshots carrying
a content-bound receipt from the explicit network synchronizer, plus a
controlled extraction document.  That synchronizer can fetch only the seven
catalog URLs without unsafe redirects or link traversal; artifact validation
never accesses the network.  The
immutable source bundle and review frame contain no fit records, task labels,
gold annotations, legacy lexicon, or candidate proposals.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import os
import re
import shutil
import stat
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlsplit

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


CATALOG_SCHEMA_VERSION = "wp3-g3-public-source-catalog/v1"
SNAPSHOT_INDEX_SCHEMA_VERSION = "wp3-g3-frozen-snapshot-index/v1"
SYNC_RECEIPT_SCHEMA_VERSION = "wp3-g3-public-source-sync-receipt/v1"
EXTRACTION_SCHEMA_VERSION = "wp3-g3-form-extraction/v1"
SOURCE_BUNDLE_SCHEMA_VERSION = "wp3-g3-public-source-bundle/v1"
FRAME_SCHEMA_VERSION = "wp3-g3-form-review-frame/v1"
SESSION_SCHEMA_VERSION = "wp3-g3-form-review-session/v1"
REFERENCE_SCHEMA_VERSION = "wp3-g3-form-reference/v1"
DECLARATION_SCHEMA_VERSION = "wp3-g3-form-reviewer-declaration/v1"

SOURCE_BUNDLE_ARTIFACT_KIND = "wp3-g3-public-source-bundle"
SYNC_RECEIPT_ARTIFACT_KIND = "wp3-g3-public-source-sync-receipt"
FRAME_ARTIFACT_KIND = "wp3-g3-form-review-frame"
REFERENCE_ARTIFACT_KIND = "wp3-g3-form-reference"

SOURCE_BUNDLE_ID_PREFIX = "wp3g3sources-"
SYNC_RECEIPT_ID_PREFIX = "wp3g3sync-"
FRAME_ID_PREFIX = "wp3g3formframe-"
REFERENCE_ID_PREFIX = "wp3g3formref-"

REFERENCE_ROLE = "form-only-label-free-non-lexicon"
SCOPE = "development-only"
FAMILIES = frozenset(
    {"known_variant", "phonetic_variant", "orthographic_variant"}
)
ACTIONS = frozenset({"accept", "reject", "edit", "defer"})
MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
MAX_TOTAL_SNAPSHOT_BYTES = 40 * 1024 * 1024
FETCH_CONNECT_TIMEOUT_SECONDS = 10
FETCH_READ_TIMEOUT_SECONDS = 60
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CATALOG_SCHEMA_PATH = "schemas/wp3_g3_public_source_catalog_v1.schema.json"
SNAPSHOT_SCHEMA_PATH = "schemas/wp3_g3_frozen_source_input_v1.schema.json"
SYNC_RECEIPT_SCHEMA_PATH = "schemas/wp3_g3_public_source_sync_receipt_v1.schema.json"
EXTRACTION_SCHEMA_PATH = "schemas/wp3_g3_form_extraction_v1.schema.json"
REFERENCE_SCHEMA_PATH = "schemas/wp3_g3_form_reference_v1.schema.json"

FROZEN_SOURCE_URLS = {
    "moe-network-language-experts": "https://www.moe.gov.cn/jyb_xwfb/xw_ft/moe_46/moe_1055/tnull_11014.html",
    "moe-network-language-impact": "https://www.moe.gov.cn/jyb_xwfb/xw_ft/moe_46/moe_1055/tnull_11604.html",
    "wikipedia-mainland-internet-language": "https://zh.wikipedia.org/wiki/中国大陆网络用语列表",
    "wikipedia-internet-language": "https://zh.wikipedia.org/wiki/互联网用语",
    "china-daily-eight-internet-expressions": "https://language.chinadaily.com.cn/a/202408/08/WS66b491a7a3104e74fddb91d7.html",
    "china-daily-yunvwuagua-awsl": "https://language.chinadaily.com.cn/a/201908/23/WS5d5f57a8a310cf3e3556787c.html",
    "china-daily-yyds-nbcs-hhh": "https://language.chinadaily.com.cn/a/202108/06/WS610cd28da310efa1bd667316.html",
}
WIKIMEDIA_SOURCE_IDS = frozenset(
    {
        "wikipedia-mainland-internet-language",
        "wikipedia-internet-language",
    }
)
SYNC_NETWORK_POLICY = "closed-seven-source-https-no-link-follow/v1"
SOURCE_BUNDLE_POLICY = "sync-receipt-bound-public-snapshots-no-link-follow/v1"
FORBIDDEN_REFERENCE_KEYS = frozenset(
    {
        "definition",
        "definitions",
        "label",
        "labels",
        "abc",
        "a_candidate",
        "b_candidate",
        "c_candidate",
        "r",
        "verdict",
        "record_id",
        "case_id",
        "task",
        "task_label",
        "targeted_group",
        "hateful",
        "gold",
        "fit",
        "legacy_lexicon",
    }
)


class G3FormReferenceError(RuntimeError):
    """Raised when a source, review, or reference contract fails closed."""


class G3FormReviewConflict(G3FormReferenceError):
    """Raised when a mutable session revision changed concurrently."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise G3FormReferenceError(f"{label} must be an object")
    return dict(value)


def _array_of_objects(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise G3FormReferenceError(f"{label} must be an array of objects")
    return [dict(row) for row in value]


def _trimmed_text(value: Any, label: str, *, maximum: int = 1000) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise G3FormReferenceError(f"{label} must be non-empty trimmed text")
    return value


def _optional_text(value: Any, label: str, *, maximum: int = 4000) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise G3FormReferenceError(f"{label} must be text no longer than {maximum}")
    return value


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        return _object(load_json(path), label)
    except TrainingArtifactError as exc:
        raise G3FormReferenceError(str(exc)) from exc


def _forbidden_key_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = (*path, str(key))
            if str(key).casefold() in FORBIDDEN_REFERENCE_KEYS:
                found.append(".".join(child_path))
            found.extend(_forbidden_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_key_paths(child, (*path, str(index))))
    return found


def _target_name_matches(target: Path, artifact_id: str) -> bool:
    return target.name == artifact_id or target.name.startswith(f".{artifact_id}.")


def _safe_workspace_file(root: Path, logical_path: str) -> Path:
    relative = Path(logical_path)
    if relative.is_absolute() or ".." in relative.parts or logical_path != relative.as_posix():
        raise G3FormReferenceError("catalog handbook path is not portable")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise G3FormReferenceError("catalog handbook escapes workspace") from exc
    if not target.is_file() or target.is_symlink():
        raise G3FormReferenceError("catalog handbook is not a regular file")
    return target


def _safe_snapshot_file(root: Path, logical_path: str, source_id: str) -> Path:
    relative = Path(logical_path)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or ".." in relative.parts
        or relative.stem != source_id
        or logical_path != relative.as_posix()
    ):
        raise G3FormReferenceError("snapshot_file must be a source-ID basename")
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise G3FormReferenceError("snapshot_file escapes snapshot root") from exc
    if not target.is_file() or target.is_symlink():
        raise G3FormReferenceError("snapshot_file is not a regular file")
    return target


def _validate_schema(value: Mapping[str, Any], schema: Path) -> None:
    try:
        validate_json_schema(value, schema)
    except TrainingArtifactError as exc:
        # This repository's minimal execution environment does not always ship
        # jsonschema.  Every lifecycle document is also validated field by
        # field below; in that environment we still bind and parse the exact
        # schema file rather than silently skipping the contract altogether.
        if "jsonschema is required" in str(exc):
            schema_value = _load_object(schema, "JSON schema")
            if schema_value.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
                raise G3FormReferenceError("JSON schema draft binding differs") from exc
            return
        raise G3FormReferenceError(str(exc)) from exc


def load_source_catalog(
    catalog_path: str | Path,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Load the exact closed catalog and verify its local rubric binding."""

    root = Path(workspace_root).resolve()
    catalog_file = Path(catalog_path).resolve()
    catalog = _load_object(catalog_file, "G3 source catalog")
    _validate_schema(catalog, root / CATALOG_SCHEMA_PATH)
    sources = _array_of_objects(catalog.get("external_sources"), "external_sources")
    source_urls = {str(row.get("source_id")): row.get("requested_url") for row in sources}
    if source_urls != FROZEN_SOURCE_URLS or len(source_urls) != len(sources):
        raise G3FormReferenceError("public source allowlist differs from the frozen catalog")
    for row in sources:
        expected_revision = str(row["source_id"]) in WIKIMEDIA_SOURCE_IDS
        if row.get("wikimedia_revision_required") is not expected_revision:
            raise G3FormReferenceError("Wikimedia revision policy differs")
    policy = _object(catalog.get("policy"), "catalog policy")
    if (
        policy.get("allowlist_closed") is not True
        or policy.get("follow_links") is not False
        or policy.get("network_fetch_implemented") is not True
    ):
        raise G3FormReferenceError("catalog is not closed and offline")
    handbook = _object(catalog.get("handbook"), "catalog handbook")
    if handbook.get("role") != "rubric-only-not-form-evidence":
        raise G3FormReferenceError("handbook must remain rubric-only")
    handbook_file = _safe_workspace_file(root, str(handbook.get("path", "")))
    if sha256_file(handbook_file) != handbook.get("sha256"):
        raise G3FormReferenceError("handbook hash drifted")
    return catalog


def _url_identity(url: str) -> tuple[str, str, int | None, str, str, str]:
    parsed = urlsplit(url)
    return (
        parsed.scheme.casefold(),
        (parsed.hostname or "").casefold(),
        parsed.port,
        unquote(parsed.path),
        parsed.query,
        parsed.fragment,
    )


def _bounded_http_get(session: Any, url: str) -> tuple[bytes, str, str]:
    """Fetch one URL with at most three explicit same-origin HTTPS redirects."""

    original = urlsplit(url)
    current = url
    response = None
    for _redirect_count in range(4):
        try:
            response = session.get(
            current,
            headers={
                "Accept": "text/html,text/plain;q=0.9",
                "User-Agent": "WP3-G3-public-source-freezer/1.0",
            },
            timeout=(FETCH_CONNECT_TIMEOUT_SECONDS, FETCH_READ_TIMEOUT_SECONDS),
            allow_redirects=False,
            stream=True,
            )
        except Exception as exc:
            raise G3FormReferenceError(f"public source fetch failed for {current}") from exc
        if str(response.url) and _url_identity(str(response.url)) != _url_identity(current):
            response.close()
            raise G3FormReferenceError("HTTP client returned a different unrequested URL")
        if response.status_code not in {301, 302, 303, 307, 308}:
            break
        location = response.headers.get("Location")
        response.close()
        if not isinstance(location, str) or not location:
            raise G3FormReferenceError("public source redirect lacks Location")
        redirected = urljoin(current, location)
        parsed = urlsplit(redirected)
        if (
            parsed.scheme != "https"
            or parsed.hostname != original.hostname
            or parsed.port != original.port
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise G3FormReferenceError(
                f"public source redirect from {current} to {redirected} escaped "
                f"the HTTPS origin of {url}"
            )
        current = redirected
    else:
        raise G3FormReferenceError("public source exceeded three redirects")
    if response is None:
        raise G3FormReferenceError("public source produced no response")
    try:
        if response.status_code != 200:
            raise G3FormReferenceError(
                f"public source response must end in 200 for {current}"
            )
        content_type = str(response.headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
        if content_type not in {"text/html", "text/plain"}:
            raise G3FormReferenceError("public source content type is not text/html or text/plain")
        length_header = response.headers.get("Content-Length")
        if length_header is not None:
            try:
                declared_length = int(length_header)
            except (TypeError, ValueError) as exc:
                raise G3FormReferenceError("public source Content-Length is invalid") from exc
            if declared_length <= 0 or declared_length > MAX_SNAPSHOT_BYTES:
                raise G3FormReferenceError("public source declared body size is invalid")
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not isinstance(chunk, bytes):
                raise G3FormReferenceError("public source body chunk is not bytes")
            total += len(chunk)
            if total > MAX_SNAPSHOT_BYTES:
                raise G3FormReferenceError("public source body exceeds the byte limit")
            chunks.append(chunk)
        body = b"".join(chunks)
        if not body:
            raise G3FormReferenceError("public source body is empty")
        try:
            body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise G3FormReferenceError("public source body must be UTF-8") from exc
        return body, content_type, current
    finally:
        response.close()


def _wikimedia_revision_id(body: bytes) -> str:
    text = body.decode("utf-8")
    patterns = (
        r'"wgRevisionId"\s*:\s*([1-9][0-9]*)',
        r'"revisionId"\s*:\s*([1-9][0-9]*)',
        r'"revision"\s*:\s*"?([1-9][0-9]*)"?',
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    raise G3FormReferenceError("Wikimedia page does not expose a current revision ID")


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() in {"script", "style", "noscript", "template"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "noscript", "template"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data)


def _visible_text_snapshot(body: bytes, media_type: str) -> bytes:
    text = body.decode("utf-8")
    if media_type == "text/html":
        parser = _VisibleTextParser()
        try:
            parser.feed(text)
            parser.close()
        except Exception as exc:
            raise G3FormReferenceError("HTML visible-text projection failed") from exc
        text = " ".join(parser.parts)
    normalized = " ".join(text.split())
    if not normalized:
        raise G3FormReferenceError("visible public-source text is empty")
    return (normalized + "\n").encode("utf-8")


def fetch_public_source_snapshots(
    *,
    workspace_root: str | Path,
    catalog_path: str | Path,
    output_directory: str | Path,
    session_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Synchronize the exact allowlist into a new local snapshot directory.

    This is the only network-capable operation in the lifecycle.  It never
    follows HTML links or implicit/unsafe redirects.  Wikimedia pages are
    fetched once to discover ``wgRevisionId`` and then fetched again at the
    derived ``oldid`` URL; only the revision-locked body is retained.
    """

    root = Path(workspace_root).resolve()
    catalog = load_source_catalog(catalog_path, workspace_root=root)
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise G3FormReferenceError(
            "source synchronization refuses to overwrite an existing path"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        os.path.realpath(
            os.path.join(
                destination.parent,
                f".{destination.name}.{os.getpid()}.{hashlib.sha256(os.urandom(16)).hexdigest()[:12]}",
            )
        )
    )
    temporary.mkdir(mode=0o700)
    if session_factory is None:
        try:
            import requests
        except ImportError as exc:
            raise G3FormReferenceError("requests is required for explicit source sync") from exc
        session_factory = requests.Session
    session = session_factory()
    if hasattr(session, "trust_env"):
        session.trust_env = False
    snapshots: list[dict[str, Any]] = []
    try:
        for catalog_row in catalog["external_sources"]:
            source_id = str(catalog_row["source_id"])
            requested_url = str(catalog_row["requested_url"])
            try:
                body, media_type, current_url = _bounded_http_get(session, requested_url)
            except G3FormReferenceError as exc:
                raise G3FormReferenceError(
                    f"source {source_id} ({requested_url}) failed: {exc}"
                ) from exc
            revision: str | None = None
            final_url = current_url
            if source_id in WIKIMEDIA_SOURCE_IDS:
                revision = _wikimedia_revision_id(body)
                separator = "&" if urlsplit(current_url).query else "?"
                locked_url = f"{current_url}{separator}oldid={revision}"
                try:
                    body, media_type, final_url = _bounded_http_get(session, locked_url)
                except G3FormReferenceError as exc:
                    raise G3FormReferenceError(
                        f"source {source_id} revision {revision} ({locked_url}) failed: {exc}"
                    ) from exc
                locked_revision = _wikimedia_revision_id(body)
                if locked_revision != revision:
                    raise G3FormReferenceError(
                        "Wikimedia oldid response revision differs from requested revision"
                    )
            body = _visible_text_snapshot(body, media_type)
            media_type = "text/plain"
            filename = source_id + ".txt"
            snapshot_path = temporary / filename
            snapshot_path.write_bytes(body)
            os.chmod(snapshot_path, 0o600)
            snapshots.append(
                {
                    "source_id": source_id,
                    "requested_url": requested_url,
                    "final_url": final_url,
                    "fetched_at": _utc_now(),
                    "media_type": media_type,
                    "snapshot_file": filename,
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                    "wikimedia_revision_id": revision,
                }
            )
        index = {
            "schema_version": SNAPSHOT_INDEX_SCHEMA_VERSION,
            "snapshots": sorted(snapshots, key=lambda row: row["source_id"]),
        }
        _validate_schema(index, root / SNAPSHOT_SCHEMA_PATH)
        receipt = _build_source_sync_receipt(
            catalog=catalog,
            snapshot_index=index,
            workspace_root=root,
        )
        _validate_schema(receipt, root / SYNC_RECEIPT_SCHEMA_PATH)
        write_canonical_json(temporary / "snapshot_index.json", index)
        write_canonical_json(temporary / "sync_receipt.json", receipt)
        os.chmod(temporary / "snapshot_index.json", 0o600)
        os.chmod(temporary / "sync_receipt.json", 0o600)
        os.replace(temporary, destination)
    finally:
        try:
            session.close()
        except Exception:
            pass
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "snapshot_root": str(destination),
        "snapshot_index": str(destination / "snapshot_index.json"),
        "sync_receipt": str(destination / "sync_receipt.json"),
        "sync_receipt_id": receipt["sync_receipt_id"],
        "sync_receipt_sha256": canonical_sha256(receipt),
        "source_count": len(snapshots),
        "catalog_sha256": canonical_sha256(catalog),
    }


def _validate_final_url(snapshot: Mapping[str, Any], catalog_row: Mapping[str, Any]) -> None:
    requested = str(snapshot.get("requested_url", ""))
    final = str(snapshot.get("final_url", ""))
    if requested != catalog_row.get("requested_url"):
        raise G3FormReferenceError("snapshot requested URL differs from catalog")
    source_id = str(snapshot.get("source_id", ""))
    revision = snapshot.get("wikimedia_revision_id")
    requested_parts = urlsplit(requested)
    final_parts = urlsplit(final)
    if (
        final_parts.scheme != "https"
        or final_parts.hostname != requested_parts.hostname
        or final_parts.port != requested_parts.port
        or final_parts.username is not None
        or final_parts.password is not None
        or final_parts.fragment
    ):
        raise G3FormReferenceError("snapshot final URL escaped the catalog HTTPS origin")
    if source_id not in WIKIMEDIA_SOURCE_IDS:
        if revision is not None or final_parts.query:
            raise G3FormReferenceError("non-Wikimedia final URL or revision differs")
        return
    if not isinstance(revision, str) or not revision.isdigit() or revision.startswith("0"):
        raise G3FormReferenceError("Wikimedia snapshot lacks a concrete revision ID")
    parsed_final = final_parts
    if (
        parsed_final.scheme != "https"
        or parse_qs(parsed_final.query, keep_blank_values=True) != {"oldid": [revision]}
        or parsed_final.fragment
    ):
        raise G3FormReferenceError("Wikimedia final URL is not the requested page plus oldid")


def _source_schema_hashes(workspace_root: Path) -> dict[str, str]:
    return {
        "catalog_schema_sha256": sha256_file(workspace_root / CATALOG_SCHEMA_PATH),
        "snapshot_index_schema_sha256": sha256_file(
            workspace_root / SNAPSHOT_SCHEMA_PATH
        ),
        "sync_receipt_schema_sha256": sha256_file(
            workspace_root / SYNC_RECEIPT_SCHEMA_PATH
        ),
        "extraction_schema_sha256": sha256_file(
            workspace_root / EXTRACTION_SCHEMA_PATH
        ),
    }


def _sync_source_coverage(
    *,
    catalog: Mapping[str, Any],
    snapshot_index: Mapping[str, Any],
    workspace_root: Path,
) -> list[dict[str, Any]]:
    """Return the exact seven-source projection bound by a sync receipt."""

    _validate_schema(snapshot_index, workspace_root / SNAPSHOT_SCHEMA_PATH)
    snapshots = _array_of_objects(snapshot_index.get("snapshots"), "snapshots")
    snapshot_by_id = {str(row.get("source_id")): row for row in snapshots}
    catalog_rows = {
        str(row["source_id"]): row
        for row in _array_of_objects(catalog.get("external_sources"), "external_sources")
    }
    if (
        len(snapshot_by_id) != len(snapshots)
        or set(snapshot_by_id) != set(FROZEN_SOURCE_URLS)
        or set(catalog_rows) != set(FROZEN_SOURCE_URLS)
    ):
        raise G3FormReferenceError(
            "sync receipt coverage must exactly equal the closed seven-source allowlist"
        )
    expected_snapshot_fields = {
        "source_id",
        "requested_url",
        "final_url",
        "fetched_at",
        "media_type",
        "snapshot_file",
        "body_sha256",
        "wikimedia_revision_id",
    }
    coverage: list[dict[str, Any]] = []
    for source_id in sorted(FROZEN_SOURCE_URLS):
        snapshot = snapshot_by_id[source_id]
        if set(snapshot) != expected_snapshot_fields:
            raise G3FormReferenceError("sync snapshot fields are not canonical")
        _validate_final_url(snapshot, catalog_rows[source_id])
        body_sha256 = str(snapshot.get("body_sha256", ""))
        if not SHA256_RE.fullmatch(body_sha256):
            raise G3FormReferenceError("sync snapshot body hash is invalid")
        coverage.append(
            {
                "source_id": source_id,
                "requested_url": snapshot["requested_url"],
                "final_url": snapshot["final_url"],
                "snapshot_file": snapshot["snapshot_file"],
                "body_sha256": body_sha256,
                "wikimedia_revision_id": snapshot["wikimedia_revision_id"],
            }
        )
    return coverage


def _build_source_sync_receipt(
    *,
    catalog: Mapping[str, Any],
    snapshot_index: Mapping[str, Any],
    workspace_root: Path,
    fetch_implementation_sha256: str | None = None,
) -> dict[str, Any]:
    implementation_sha256 = (
        sha256_file(Path(__file__))
        if fetch_implementation_sha256 is None
        else fetch_implementation_sha256
    )
    if not SHA256_RE.fullmatch(str(implementation_sha256)):
        raise G3FormReferenceError("sync receipt fetch implementation hash is invalid")
    coverage = _sync_source_coverage(
        catalog=catalog,
        snapshot_index=snapshot_index,
        workspace_root=workspace_root,
    )
    schema_hashes = _source_schema_hashes(workspace_root)
    identity = {
        "schema_version": SYNC_RECEIPT_SCHEMA_VERSION,
        "artifact_kind": SYNC_RECEIPT_ARTIFACT_KIND,
        "catalog_id": catalog["catalog_id"],
        "catalog_sha256": canonical_sha256(catalog),
        "catalog_schema_sha256": schema_hashes["catalog_schema_sha256"],
        "snapshot_index_sha256": canonical_sha256(snapshot_index),
        "snapshot_index_schema_sha256": schema_hashes[
            "snapshot_index_schema_sha256"
        ],
        "sync_receipt_schema_sha256": schema_hashes[
            "sync_receipt_schema_sha256"
        ],
        "fetch_implementation_sha256": str(implementation_sha256),
        "network_policy": SYNC_NETWORK_POLICY,
        "network_access_performed": True,
        "complete": True,
        "source_count": len(coverage),
        "source_ids": [row["source_id"] for row in coverage],
        "source_coverage_sha256": canonical_sha256(coverage),
    }
    return {
        **identity,
        "sync_receipt_id": SYNC_RECEIPT_ID_PREFIX + canonical_sha256(identity),
    }


def _validate_source_sync_receipt(
    receipt: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    snapshot_index: Mapping[str, Any],
    workspace_root: Path,
    require_current_implementation: bool,
) -> dict[str, Any]:
    normalized = _object(receipt, "public-source sync receipt")
    _validate_schema(normalized, workspace_root / SYNC_RECEIPT_SCHEMA_PATH)
    declared_implementation = normalized.get("fetch_implementation_sha256")
    expected = _build_source_sync_receipt(
        catalog=catalog,
        snapshot_index=snapshot_index,
        workspace_root=workspace_root,
        fetch_implementation_sha256=(
            None if require_current_implementation else str(declared_implementation)
        ),
    )
    if normalized != expected:
        raise G3FormReferenceError(
            "public-source sync receipt does not replay from the catalog and snapshots"
        )
    return normalized


def _load_source_sync_receipt(
    *, snapshot_root: Path, sync_receipt_path: str | Path
) -> dict[str, Any]:
    expected = (snapshot_root.resolve() / "sync_receipt.json").resolve()
    supplied = Path(sync_receipt_path)
    if supplied.resolve() != expected:
        raise G3FormReferenceError(
            "production freeze requires sync_receipt.json from the snapshot directory"
        )
    if not supplied.is_file() or supplied.is_symlink():
        raise G3FormReferenceError("public-source sync receipt is missing or not regular")
    return _load_object(supplied, "public-source sync receipt")


def _occurrence_span(content: str, surface: str, ordinal: int) -> tuple[int, int]:
    cursor = 0
    found = 0
    while cursor <= len(content) - len(surface):
        start = content.find(surface, cursor)
        if start < 0:
            break
        found += 1
        if found == ordinal:
            return start, start + len(surface)
        cursor = start + 1
    raise G3FormReferenceError("evidence quote occurrence does not replay")


def _validate_source_inputs(
    *,
    catalog: Mapping[str, Any],
    snapshot_index: Mapping[str, Any],
    extraction: Mapping[str, Any],
    snapshot_root: Path,
    workspace_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    _validate_schema(snapshot_index, workspace_root / SNAPSHOT_SCHEMA_PATH)
    _validate_schema(extraction, workspace_root / EXTRACTION_SCHEMA_PATH)
    forbidden = _forbidden_key_paths(extraction)
    if forbidden:
        raise G3FormReferenceError(
            "form extraction contains forbidden task/data keys: " + ", ".join(forbidden)
        )
    snapshots = _array_of_objects(snapshot_index.get("snapshots"), "snapshots")
    catalog_rows = {
        str(row["source_id"]): row
        for row in _array_of_objects(catalog.get("external_sources"), "external_sources")
    }
    snapshot_by_id = {str(row.get("source_id")): row for row in snapshots}
    if len(snapshot_by_id) != len(snapshots) or set(snapshot_by_id) != set(catalog_rows):
        raise G3FormReferenceError("snapshot coverage must exactly equal the closed allowlist")
    snapshot_text: dict[str, str] = {}
    source_files: dict[str, str] = {}
    total_size = 0
    for source_id, snapshot in snapshot_by_id.items():
        _validate_final_url(snapshot, catalog_rows[source_id])
        source_file = _safe_snapshot_file(
            snapshot_root, str(snapshot.get("snapshot_file", "")), source_id
        )
        size = source_file.stat().st_size
        total_size += size
        if size <= 0 or size > MAX_SNAPSHOT_BYTES or total_size > MAX_TOTAL_SNAPSHOT_BYTES:
            raise G3FormReferenceError("snapshot byte budget exceeded")
        if sha256_file(source_file) != snapshot.get("body_sha256"):
            raise G3FormReferenceError("snapshot body hash differs")
        try:
            snapshot_text[source_id] = source_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise G3FormReferenceError("snapshot must be valid UTF-8 text") from exc
        source_files[source_id] = str(source_file)

    evidence = _array_of_objects(extraction.get("evidence"), "extraction evidence")
    evidence_by_id = {str(row.get("evidence_id")): row for row in evidence}
    if len(evidence_by_id) != len(evidence):
        raise G3FormReferenceError("evidence IDs are duplicated")
    for row in evidence:
        source_id = str(row.get("source_id", ""))
        if source_id not in snapshot_by_id:
            raise G3FormReferenceError("evidence source is outside the closed allowlist")
        quote = _trimmed_text(row.get("quote"), "evidence quote", maximum=4000)
        ordinal = row.get("occurrence_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
            raise G3FormReferenceError("evidence occurrence ordinal is invalid")
        _occurrence_span(snapshot_text[source_id], quote, ordinal)

    items = _array_of_objects(extraction.get("items"), "extraction items")
    item_by_id = {str(row.get("item_id")): row for row in items}
    if not items:
        raise G3FormReferenceError("form extraction must produce a non-empty review queue")
    if len(item_by_id) != len(items):
        raise G3FormReferenceError("form item IDs are duplicated")
    for row in items:
        surface = _trimmed_text(row.get("surface"), "form surface", maximum=160)
        canonical = _trimmed_text(row.get("canonical"), "form canonical", maximum=160)
        if surface == canonical:
            raise G3FormReferenceError("form relation surface must differ from canonical")
        if row.get("proposed_family") not in FAMILIES:
            raise G3FormReferenceError("form relation family is invalid")
        evidence_ids = row.get("evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or len(evidence_ids) != len(set(evidence_ids))
            or any(evidence_id not in evidence_by_id for evidence_id in evidence_ids)
        ):
            raise G3FormReferenceError("form relation evidence IDs are invalid")
        selected_evidence = [evidence_by_id[evidence_id] for evidence_id in evidence_ids]
        if len({row["source_id"] for row in selected_evidence}) != 1:
            raise G3FormReferenceError("one form relation must use evidence from one source")
        quotes = [str(row["quote"]) for row in selected_evidence]
        if not any(surface in quote for quote in quotes) or not any(
            canonical in quote for quote in quotes
        ):
            raise G3FormReferenceError(
                "surface and canonical must both replay in the selected source evidence"
            )
    evidence.sort(key=lambda row: str(row["evidence_id"]))
    items.sort(key=lambda row: str(row["item_id"]))
    snapshots.sort(key=lambda row: str(row["source_id"]))
    return snapshots, evidence, items, source_files


def sync_public_sources(
    *,
    workspace_root: str | Path,
    catalog_path: str | Path,
    snapshot_root: str | Path,
    snapshot_index_path: str | Path,
    sync_receipt_path: str | Path,
    extraction_path: str | Path,
    output_root: str | Path,
    write_ref: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze only snapshots carrying the synchronizer's replayable receipt."""

    root = Path(workspace_root).resolve()
    catalog_file = Path(catalog_path).resolve()
    catalog = load_source_catalog(catalog_file, workspace_root=root)
    snapshot_directory = Path(snapshot_root).resolve()
    snapshot_index = _load_object(snapshot_index_path, "snapshot index")
    sync_receipt = _load_source_sync_receipt(
        snapshot_root=snapshot_directory,
        sync_receipt_path=sync_receipt_path,
    )
    extraction = _load_object(extraction_path, "form extraction")
    snapshots, evidence, items, source_files = _validate_source_inputs(
        catalog=catalog,
        snapshot_index=snapshot_index,
        extraction=extraction,
        snapshot_root=snapshot_directory,
        workspace_root=root,
    )
    normalized_extraction = {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "extractor": extraction["extractor"],
        "evidence": evidence,
        "items": items,
    }
    normalized_index = {
        "schema_version": SNAPSHOT_INDEX_SCHEMA_VERSION,
        "snapshots": snapshots,
    }
    sync_receipt = _validate_source_sync_receipt(
        sync_receipt,
        catalog=catalog,
        snapshot_index=normalized_index,
        workspace_root=root,
        require_current_implementation=True,
    )
    schema_hashes = _source_schema_hashes(root)
    identity = {
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "artifact_kind": SOURCE_BUNDLE_ARTIFACT_KIND,
        "catalog_id": catalog["catalog_id"],
        "catalog_sha256": canonical_sha256(catalog),
        "catalog_schema_sha256": schema_hashes["catalog_schema_sha256"],
        "snapshot_index_sha256": canonical_sha256(normalized_index),
        "snapshot_index_schema_sha256": schema_hashes[
            "snapshot_index_schema_sha256"
        ],
        "sync_receipt_id": sync_receipt["sync_receipt_id"],
        "sync_receipt_sha256": canonical_sha256(sync_receipt),
        "sync_receipt_schema_sha256": schema_hashes[
            "sync_receipt_schema_sha256"
        ],
        "extraction_sha256": canonical_sha256(normalized_extraction),
        "extraction_schema_sha256": schema_hashes["extraction_schema_sha256"],
        "handbook_sha256": catalog["handbook"]["sha256"],
        "source_count": len(snapshots),
        "evidence_count": len(evidence),
        "item_count": len(items),
        "source_policy": SOURCE_BUNDLE_POLICY,
        "reference_role": REFERENCE_ROLE,
        "scope": SCOPE,
        "scientific_eligible": False,
        "sealed": False,
        "builder_implementation_sha256": sha256_file(Path(__file__)),
    }
    artifact_id = SOURCE_BUNDLE_ID_PREFIX + canonical_sha256(identity)
    manifest = {**identity, "source_bundle_id": artifact_id}
    output_parent = Path(output_root).resolve()
    target = output_parent / artifact_id
    if not target.exists():
        staging = new_staging_directory(output_parent, artifact_id)
        try:
            write_canonical_json(staging / "manifest.json", manifest)
            write_canonical_json(staging / "catalog.json", catalog)
            write_canonical_json(staging / "snapshot_index.json", normalized_index)
            write_canonical_json(staging / "sync_receipt.json", sync_receipt)
            write_canonical_json(staging / "extraction.json", normalized_extraction)
            snapshot_dir = staging / "snapshots"
            snapshot_dir.mkdir()
            for row in snapshots:
                source_id = str(row["source_id"])
                shutil.copyfile(source_files[source_id], snapshot_dir / row["snapshot_file"])
            rubric_dir = staging / "rubric"
            rubric_dir.mkdir()
            handbook_file = _safe_workspace_file(root, str(catalog["handbook"]["path"]))
            shutil.copyfile(handbook_file, rubric_dir / "handbook.md")
            payload_hash = finalize_target_atomic(
                staging,
                target,
                validate_staging=lambda staged: validate_public_source_bundle(
                    staged,
                    workspace_root=root,
                    require_current_implementation=True,
                ),
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    else:
        validated = validate_public_source_bundle(
            target,
            workspace_root=root,
            require_current_implementation=True,
        )
        payload_hash = validated["payload_manifest_sha256"]
    if write_ref is not None:
        write_locator_ref(
            write_ref,
            artifact_kind=SOURCE_BUNDLE_ARTIFACT_KIND,
            artifact_id=artifact_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    return {
        "source_bundle_id": artifact_id,
        "target": str(target),
        "payload_manifest_sha256": payload_hash,
        "manifest": manifest,
        "sync_receipt": sync_receipt,
    }


def validate_public_source_bundle(
    source_bundle_dir: str | Path,
    *,
    workspace_root: str | Path,
    require_current_implementation: bool = True,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    target = Path(source_bundle_dir).resolve()
    try:
        payload_hash = validate_payload_manifest(target)
    except TrainingArtifactError as exc:
        raise G3FormReferenceError(str(exc)) from exc
    manifest = _load_object(target / "manifest.json", "source-bundle manifest")
    catalog = _load_object(target / "catalog.json", "frozen catalog")
    snapshot_index = _load_object(target / "snapshot_index.json", "frozen snapshot index")
    sync_receipt = _load_object(
        target / "sync_receipt.json", "frozen public-source sync receipt"
    )
    extraction = _load_object(target / "extraction.json", "frozen extraction")
    expected_files = {
        "manifest.json",
        "catalog.json",
        "snapshot_index.json",
        "sync_receipt.json",
        "extraction.json",
        "rubric/handbook.md",
        "payload_manifest.json",
    } | {
        f"snapshots/{row['snapshot_file']}"
        for row in _array_of_objects(snapshot_index.get("snapshots"), "snapshots")
    }
    try:
        ensure_exact_file_set(target, expected_files)
    except TrainingArtifactError as exc:
        raise G3FormReferenceError(str(exc)) from exc
    _validate_schema(catalog, root / CATALOG_SCHEMA_PATH)
    if {
        str(row.get("source_id")): row.get("requested_url")
        for row in _array_of_objects(catalog.get("external_sources"), "external_sources")
    } != FROZEN_SOURCE_URLS:
        raise G3FormReferenceError("frozen source catalog differs")
    if sha256_file(target / "rubric/handbook.md") != catalog["handbook"]["sha256"]:
        raise G3FormReferenceError("frozen rubric hash differs")
    snapshots, evidence, items, _ = _validate_source_inputs(
        catalog=catalog,
        snapshot_index=snapshot_index,
        extraction=extraction,
        snapshot_root=target / "snapshots",
        workspace_root=root,
    )
    normalized_extraction = {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "extractor": extraction["extractor"],
        "evidence": evidence,
        "items": items,
    }
    normalized_index = {
        "schema_version": SNAPSHOT_INDEX_SCHEMA_VERSION,
        "snapshots": snapshots,
    }
    sync_receipt = _validate_source_sync_receipt(
        sync_receipt,
        catalog=catalog,
        snapshot_index=normalized_index,
        workspace_root=root,
        require_current_implementation=require_current_implementation,
    )
    schema_hashes = _source_schema_hashes(root)
    identity = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "source_bundle_id"}
    expected_id = SOURCE_BUNDLE_ID_PREFIX + canonical_sha256(identity)
    if manifest.get("source_bundle_id") != expected_id or not _target_name_matches(target, expected_id):
        raise G3FormReferenceError("source-bundle content-addressed identity differs")
    if (
        manifest.get("schema_version") != SOURCE_BUNDLE_SCHEMA_VERSION
        or manifest.get("artifact_kind") != SOURCE_BUNDLE_ARTIFACT_KIND
        or manifest.get("catalog_sha256") != canonical_sha256(catalog)
        or manifest.get("catalog_schema_sha256")
        != schema_hashes["catalog_schema_sha256"]
        or manifest.get("snapshot_index_sha256") != canonical_sha256(normalized_index)
        or manifest.get("snapshot_index_schema_sha256")
        != schema_hashes["snapshot_index_schema_sha256"]
        or manifest.get("sync_receipt_id") != sync_receipt["sync_receipt_id"]
        or manifest.get("sync_receipt_sha256") != canonical_sha256(sync_receipt)
        or manifest.get("sync_receipt_schema_sha256")
        != schema_hashes["sync_receipt_schema_sha256"]
        or manifest.get("extraction_sha256") != canonical_sha256(normalized_extraction)
        or manifest.get("extraction_schema_sha256")
        != schema_hashes["extraction_schema_sha256"]
        or manifest.get("source_count") != len(snapshots)
        or manifest.get("evidence_count") != len(evidence)
        or manifest.get("item_count") != len(items)
        or manifest.get("source_policy") != SOURCE_BUNDLE_POLICY
        or manifest.get("reference_role") != REFERENCE_ROLE
        or manifest.get("scope") != SCOPE
        or manifest.get("scientific_eligible") is not False
        or manifest.get("sealed") is not False
    ):
        raise G3FormReferenceError("source-bundle manifest bindings differ")
    if require_current_implementation and manifest.get("builder_implementation_sha256") != sha256_file(Path(__file__)):
        raise G3FormReferenceError("source-bundle implementation drifted")
    return {
        "source_bundle_id": expected_id,
        "target": str(target),
        "payload_manifest_sha256": payload_hash,
        "manifest": manifest,
        "catalog": catalog,
        "snapshot_index": normalized_index,
        "sync_receipt": sync_receipt,
        "extraction": normalized_extraction,
    }


def _dependency(result: Mapping[str, Any], kind: str, identifier_key: str) -> dict[str, Any]:
    return {
        "artifact_kind": kind,
        "artifact_id": result[identifier_key],
        "payload_manifest_sha256": result["payload_manifest_sha256"],
    }


def build_form_review_frame(
    *,
    source_bundle_dir: str | Path,
    workspace_root: str | Path,
    output_root: str | Path,
    write_ref: str | Path | None = None,
) -> dict[str, Any]:
    source = validate_public_source_bundle(
        source_bundle_dir,
        workspace_root=workspace_root,
        require_current_implementation=True,
    )
    catalog_by_id = {
        row["source_id"]: row for row in source["catalog"]["external_sources"]
    }
    snapshot_by_id = {
        row["source_id"]: row for row in source["snapshot_index"]["snapshots"]
    }
    evidence = []
    for row in source["extraction"]["evidence"]:
        source_id = row["source_id"]
        snapshot_row = snapshot_by_id[source_id]
        snapshot_text = (
            Path(source["target"])
            / "snapshots"
            / str(snapshot_row["snapshot_file"])
        ).read_text(encoding="utf-8")
        start, end = _occurrence_span(
            snapshot_text, str(row["quote"]), int(row["occurrence_ordinal"])
        )
        evidence.append(
            {
                **copy.deepcopy(row),
                "snapshot_start": start,
                "snapshot_end": end,
                "evidence_text_sha256": hashlib.sha256(
                    str(row["quote"]).encode("utf-8")
                ).hexdigest(),
                "publisher": catalog_by_id[source_id]["publisher"],
                "requested_url": catalog_by_id[source_id]["requested_url"],
                "final_url": snapshot_by_id[source_id]["final_url"],
                "wikimedia_revision_id": snapshot_by_id[source_id]["wikimedia_revision_id"],
            }
        )
    items = copy.deepcopy(source["extraction"]["items"])
    evidence.sort(key=lambda row: row["evidence_id"])
    items.sort(key=lambda row: row["item_id"])
    identity = {
        "schema_version": FRAME_SCHEMA_VERSION,
        "artifact_kind": FRAME_ARTIFACT_KIND,
        "source_bundle_dependency": _dependency(
            source, SOURCE_BUNDLE_ARTIFACT_KIND, "source_bundle_id"
        ),
        "items_sha256": canonical_sha256(items),
        "evidence_sha256": canonical_sha256(evidence),
        "item_count": len(items),
        "evidence_count": len(evidence),
        "decision_actions": sorted(ACTIONS),
        "variant_families": sorted(FAMILIES),
        "reference_role": REFERENCE_ROLE,
        "scope": SCOPE,
        "scientific_eligible": False,
        "sealed": False,
        "review_policy": "explicit-human-confirmation-no-auto-accept/v1",
        "builder_implementation_sha256": sha256_file(Path(__file__)),
    }
    frame_id = FRAME_ID_PREFIX + canonical_sha256(identity)
    manifest = {**identity, "frame_id": frame_id}
    output_parent = Path(output_root).resolve()
    target = output_parent / frame_id
    if not target.exists():
        staging = new_staging_directory(output_parent, frame_id)
        try:
            write_canonical_json(staging / "manifest.json", manifest)
            write_canonical_json(staging / "items.json", items)
            write_canonical_json(staging / "evidence.json", evidence)
            payload_hash = finalize_target_atomic(
                staging,
                target,
                validate_staging=lambda staged: validate_form_review_frame(
                    staged,
                    source_bundle_dir=source_bundle_dir,
                    workspace_root=workspace_root,
                    require_current_implementation=True,
                ),
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    else:
        validated = validate_form_review_frame(
            target,
            source_bundle_dir=source_bundle_dir,
            workspace_root=workspace_root,
            require_current_implementation=True,
        )
        payload_hash = validated["payload_manifest_sha256"]
    if write_ref is not None:
        write_locator_ref(
            write_ref,
            artifact_kind=FRAME_ARTIFACT_KIND,
            artifact_id=frame_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    return {
        "frame_id": frame_id,
        "target": str(target),
        "payload_manifest_sha256": payload_hash,
        "manifest": manifest,
        "items": items,
        "evidence": evidence,
    }


def validate_form_review_frame(
    frame_dir: str | Path,
    *,
    source_bundle_dir: str | Path,
    workspace_root: str | Path,
    require_current_implementation: bool = True,
) -> dict[str, Any]:
    target = Path(frame_dir).resolve()
    # The v2 source lifecycle is intentionally implemented in a separate
    # module so the frozen seven-source v1 contract remains byte-for-byte
    # interpretable.  Dispatch only on the exact immutable frame schema; all
    # validation still happens fail-closed in the v2 validator.
    try:
        schema_version = _load_object(
            target / "manifest.json", "form-review frame manifest"
        ).get("schema_version")
    except (G3FormReferenceError, OSError):
        schema_version = None
    if schema_version == "wp3-g3-form-review-frame/v2":
        from build_lex.terminology_g3_relation_review import (
            validate_form_relation_review_frame_v2,
        )

        return validate_form_relation_review_frame_v2(
            target,
            source_bundle_dir=source_bundle_dir,
            workspace_root=workspace_root,
            require_current_implementation=require_current_implementation,
        )
    try:
        payload_hash = validate_payload_manifest(target)
        ensure_exact_file_set(
            target,
            {"manifest.json", "items.json", "evidence.json", "payload_manifest.json"},
        )
    except TrainingArtifactError as exc:
        raise G3FormReferenceError(str(exc)) from exc
    source = validate_public_source_bundle(
        source_bundle_dir,
        workspace_root=workspace_root,
        require_current_implementation=require_current_implementation,
    )
    manifest = _load_object(target / "manifest.json", "form-review frame manifest")
    items = _array_of_objects(load_json(target / "items.json"), "form-review items")
    evidence = _array_of_objects(load_json(target / "evidence.json"), "form-review evidence")
    identity = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "frame_id"}
    frame_id = FRAME_ID_PREFIX + canonical_sha256(identity)
    if manifest.get("frame_id") != frame_id or not _target_name_matches(target, frame_id):
        raise G3FormReferenceError("form-review frame content-addressed identity differs")
    expected_dependency = _dependency(
        source, SOURCE_BUNDLE_ARTIFACT_KIND, "source_bundle_id"
    )
    if (
        manifest.get("schema_version") != FRAME_SCHEMA_VERSION
        or manifest.get("artifact_kind") != FRAME_ARTIFACT_KIND
        or manifest.get("source_bundle_dependency") != expected_dependency
        or manifest.get("items_sha256") != canonical_sha256(items)
        or manifest.get("evidence_sha256") != canonical_sha256(evidence)
        or manifest.get("item_count") != len(items)
        or manifest.get("evidence_count") != len(evidence)
        or manifest.get("decision_actions") != sorted(ACTIONS)
        or manifest.get("variant_families") != sorted(FAMILIES)
        or manifest.get("review_policy") != "explicit-human-confirmation-no-auto-accept/v1"
        or manifest.get("reference_role") != REFERENCE_ROLE
        or manifest.get("scope") != SCOPE
        or manifest.get("scientific_eligible") is not False
        or manifest.get("sealed") is not False
    ):
        raise G3FormReferenceError("form-review frame bindings differ")
    if require_current_implementation and manifest.get("builder_implementation_sha256") != sha256_file(Path(__file__)):
        raise G3FormReferenceError("form-review frame implementation drifted")
    if items != source["extraction"]["items"]:
        raise G3FormReferenceError("form-review items differ from frozen extraction")
    evidence_projection = [
        {
            key: row[key]
            for key in (
                "evidence_id",
                "source_id",
                "quote",
                "occurrence_ordinal",
                "relation_note",
            )
        }
        for row in evidence
    ]
    if evidence_projection != source["extraction"]["evidence"]:
        raise G3FormReferenceError("form-review evidence differs from frozen extraction")
    evidence_ids = {row["evidence_id"] for row in evidence}
    if any(
        not isinstance(row.get("evidence_ids"), list)
        or any(evidence_id not in evidence_ids for evidence_id in row["evidence_ids"])
        for row in items
    ):
        raise G3FormReferenceError("form-review evidence coverage differs")
    return {
        "frame_id": frame_id,
        "target": str(target),
        "payload_manifest_sha256": payload_hash,
        "manifest": manifest,
        "items": items,
        "evidence": evidence,
        "source": source,
    }


def _session_payload(session: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in session.items()
        if key != "revision"
    }


def _with_session_revision(session: Mapping[str, Any]) -> dict[str, Any]:
    result = _session_payload(session)
    result["revision"] = canonical_sha256(result)
    return result


@contextmanager
def _form_review_session_lock(
    path: str | Path, *, exclusive: bool
) -> Iterator[None]:
    """Hold a stable sibling lock across one session read or mutation.

    The session itself is atomically replaced, so locking that inode would let
    another process open the replacement and bypass the lock.  The sibling
    lock file is persistent and must never be removed during the lifecycle.
    """

    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise G3FormReferenceError(
            "form-review session parent must be a regular directory"
        )
    lock_path = destination.with_name(destination.name + ".lock")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise G3FormReferenceError(
            "form-review session lock cannot be safely opened"
        ) from exc
    locked = False
    try:
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise G3FormReferenceError(
                    "form-review session lock must be a regular file"
                )
            os.fchmod(descriptor, 0o600)
            fcntl.flock(
                descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            )
            locked = True
            current = lock_path.lstat()
        except OSError as exc:
            raise G3FormReferenceError(
                "form-review session lock cannot be acquired"
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != metadata.st_dev
            or current.st_ino != metadata.st_ino
            or current.st_mode & 0o077
        ):
            raise G3FormReferenceError(
                "form-review session lock identity or mode changed"
            )
        yield
    finally:
        if locked:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)


def _read_form_review_session_unlocked(path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    try:
        metadata = destination.lstat()
    except OSError as exc:
        raise G3FormReferenceError("form-review session is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or destination.is_symlink()
        or metadata.st_mode & 0o077
    ):
        raise G3FormReferenceError(
            "form-review session must be a regular owner-only file"
        )
    session = _load_object(path, "form-review session")
    if session.get("revision") != canonical_sha256(_session_payload(session)):
        raise G3FormReferenceError("form-review session revision is invalid")
    return session


def read_form_review_session(path: str | Path) -> dict[str, Any]:
    with _form_review_session_lock(path, exclusive=False):
        return _read_form_review_session_unlocked(path)


def _write_session_cas_unlocked(
    path: str | Path,
    session: Mapping[str, Any],
    *,
    expected_revision: str | None,
) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        current = _read_form_review_session_unlocked(destination)
        if expected_revision is not None and current["revision"] != expected_revision:
            raise G3FormReviewConflict("form-review session changed concurrently")
    elif expected_revision is not None:
        raise G3FormReviewConflict("form-review session does not exist")
    result = _with_session_revision(session)
    write_canonical_json(destination, result)
    os.chmod(destination, 0o600)
    return result


def _write_session_cas(
    path: str | Path,
    session: Mapping[str, Any],
    *,
    expected_revision: str | None,
) -> dict[str, Any]:
    with _form_review_session_lock(path, exclusive=True):
        return _write_session_cas_unlocked(
            path, session, expected_revision=expected_revision
        )


def create_form_review_session(
    *,
    frame_dir: str | Path,
    source_bundle_dir: str | Path,
    workspace_root: str | Path,
    session_path: str | Path,
    reviewer_id: str,
) -> dict[str, Any]:
    frame = validate_form_review_frame(
        frame_dir,
        source_bundle_dir=source_bundle_dir,
        workspace_root=workspace_root,
        require_current_implementation=True,
    )
    reviewer = _trimmed_text(reviewer_id, "reviewer_id", maximum=100)
    destination = Path(session_path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination.parent, 0o700)
    with _form_review_session_lock(destination, exclusive=True):
        if destination.exists():
            existing = _read_form_review_session_unlocked(destination)
            if (
                existing.get("frame_id") != frame["frame_id"]
                or existing.get("frame_payload_manifest_sha256")
                != frame["payload_manifest_sha256"]
                or existing.get("reviewer_id") != reviewer
            ):
                raise G3FormReferenceError(
                    "existing form-review session belongs to another frame/reviewer"
                )
            return existing
        decisions = {
            row["item_id"]: {
                "status": "draft",
                "action": "defer",
                "surface": row["surface"],
                "canonical": row["canonical"],
                "family": row["proposed_family"],
                "phonetic_scan_enabled": row["phonetic_scan_enabled"],
                "evidence_ids": list(row["evidence_ids"]),
                "notes": "",
            }
            for row in frame["items"]
        }
        session = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "frame_id": frame["frame_id"],
            "frame_payload_manifest_sha256": frame["payload_manifest_sha256"],
            "reviewer_id": reviewer,
            "decisions": decisions,
            "amendments": [],
            "finalized_reference_id": None,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        return _write_session_cas_unlocked(
            destination, session, expected_revision=None
        )


def _decision_replays(
    decision: Mapping[str, Any],
    *,
    item: Mapping[str, Any],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_fields = {
        "action",
        "surface",
        "canonical",
        "family",
        "phonetic_scan_enabled",
        "evidence_ids",
        "notes",
    }
    if set(decision) != expected_fields:
        raise G3FormReferenceError("form decision fields are not canonical")
    action = decision.get("action")
    if action not in ACTIONS:
        raise G3FormReferenceError("form decision action is invalid")
    surface = _trimmed_text(decision.get("surface"), "decision surface", maximum=160)
    canonical = _trimmed_text(
        decision.get("canonical"), "decision canonical", maximum=160
    )
    if surface == canonical:
        raise G3FormReferenceError("decision surface must differ from canonical")
    family = decision.get("family")
    if family not in FAMILIES:
        raise G3FormReferenceError("decision family is invalid")
    scan = decision.get("phonetic_scan_enabled")
    if not isinstance(scan, bool):
        raise G3FormReferenceError("phonetic_scan_enabled must be boolean")
    evidence_ids = decision.get("evidence_ids")
    allowed_evidence = set(item["evidence_ids"])
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or len(evidence_ids) != len(set(evidence_ids))
        or any(
            not isinstance(evidence_id, str)
            or evidence_id not in allowed_evidence
            or evidence_id not in evidence_by_id
            for evidence_id in evidence_ids
        )
    ):
        raise G3FormReferenceError("decision evidence IDs are invalid")
    notes = _optional_text(decision.get("notes"), "decision notes", maximum=2000)
    proposed = {
        "surface": item["surface"],
        "canonical": item["canonical"],
        "family": item["proposed_family"],
        "phonetic_scan_enabled": item["phonetic_scan_enabled"],
        "evidence_ids": item["evidence_ids"],
    }
    selected = {
        "surface": surface,
        "canonical": canonical,
        "family": family,
        "phonetic_scan_enabled": scan,
        "evidence_ids": evidence_ids,
    }
    if action in {"accept", "reject", "defer"} and selected != proposed:
        raise G3FormReferenceError(f"{action} must preserve the extracted proposal")
    if action == "edit" and selected == proposed:
        raise G3FormReferenceError("edit must change at least one proposal field")
    if action in {"accept", "edit"}:
        selected_evidence = [evidence_by_id[evidence_id] for evidence_id in evidence_ids]
        if len({row["source_id"] for row in selected_evidence}) != 1:
            raise G3FormReferenceError("one form decision must use one public source")
        component_ids = {
            row.get("component_id")
            for row in selected_evidence
            if row.get("relation_contract")
            == "single-quote-surface-and-canonical/v2"
        }
        if component_ids and (
            len(component_ids) != 1 or None in component_ids
        ):
            raise G3FormReferenceError(
                "one v2 form decision must use one frozen source component"
            )
        quotes = [str(row["quote"]) for row in selected_evidence]
        strict_single_quote = any(
            row.get("relation_contract")
            == "single-quote-surface-and-canonical/v2"
            for row in selected_evidence
        )
        replays = (
            any(surface in quote and canonical in quote for quote in quotes)
            if strict_single_quote
            else (
                any(surface in quote for quote in quotes)
                and any(canonical in quote for quote in quotes)
            )
        )
        if not replays:
            raise G3FormReferenceError(
                "edited surface and canonical must replay in selected evidence"
            )
    return {
        "action": action,
        "surface": surface,
        "canonical": canonical,
        "family": family,
        "phonetic_scan_enabled": scan,
        "evidence_ids": list(evidence_ids),
        "notes": notes,
    }


def save_form_decision(
    *,
    frame_dir: str | Path,
    source_bundle_dir: str | Path,
    workspace_root: str | Path,
    session_path: str | Path,
    item_id: str,
    decision: Mapping[str, Any],
    confirm: bool,
    expected_revision: str,
) -> dict[str, Any]:
    frame = validate_form_review_frame(
        frame_dir,
        source_bundle_dir=source_bundle_dir,
        workspace_root=workspace_root,
        require_current_implementation=True,
    )
    return save_form_decision_from_validated_frame(
        frame=frame,
        session_path=session_path,
        item_id=item_id,
        decision=decision,
        confirm=confirm,
        expected_revision=expected_revision,
    )


def save_form_decision_from_validated_frame(
    *,
    frame: Mapping[str, Any],
    session_path: str | Path,
    item_id: str,
    decision: Mapping[str, Any],
    confirm: bool,
    expected_revision: str,
) -> dict[str, Any]:
    """Save one decision against an already fail-closed validated frame.

    Long-running review services validate their immutable, content-addressed
    frame once during startup.  This entry point retains all decision replay,
    session integrity, locking, and CAS checks without re-reading and replaying
    the complete frozen source bundle for every autosave.
    """

    item_by_id = {row["item_id"]: row for row in frame["items"]}
    evidence_by_id = {row["evidence_id"]: row for row in frame["evidence"]}
    if item_id not in item_by_id:
        raise G3FormReferenceError("unknown form-review item")
    normalized = _decision_replays(
        decision,
        item=item_by_id[item_id],
        evidence_by_id=evidence_by_id,
    )
    if not isinstance(confirm, bool):
        raise G3FormReferenceError("confirm must be boolean")
    with _form_review_session_lock(session_path, exclusive=True):
        session = _read_form_review_session_unlocked(session_path)
        if session.get("finalized_reference_id") is not None:
            raise G3FormReferenceError("finalized form-review session is immutable")
        current = session.get("decisions", {}).get(item_id)
        if not isinstance(current, Mapping):
            raise G3FormReferenceError("form-review item state is missing")
        if current.get("status") == "confirmed":
            raise G3FormReferenceError(
                "confirmed form decision must be explicitly reopened"
            )
        updated = copy.deepcopy(session)
        updated["decisions"][item_id] = {
            "status": "confirmed" if confirm else "draft",
            **normalized,
        }
        updated["updated_at"] = _utc_now()
        return _write_session_cas_unlocked(
            session_path, updated, expected_revision=expected_revision
        )


def reopen_form_decision(
    *,
    session_path: str | Path,
    item_id: str,
    reason: str,
    expected_revision: str,
) -> dict[str, Any]:
    amendment_reason = _trimmed_text(reason, "amendment reason", maximum=1000)
    with _form_review_session_lock(session_path, exclusive=True):
        session = _read_form_review_session_unlocked(session_path)
        if session.get("finalized_reference_id") is not None:
            raise G3FormReferenceError("finalized form-review session is immutable")
        decision = session.get("decisions", {}).get(item_id)
        if not isinstance(decision, Mapping):
            raise G3FormReferenceError("unknown form-review item")
        if decision.get("status") != "confirmed":
            raise G3FormReferenceError(
                "only a confirmed form decision can be reopened"
            )
        updated = copy.deepcopy(session)
        updated["decisions"][item_id]["status"] = "draft"
        updated["amendments"].append(
            {
                "item_id": item_id,
                "reason": amendment_reason,
                "prior_revision": session["revision"],
                "reopened_at": _utc_now(),
            }
        )
        updated["updated_at"] = _utc_now()
        return _write_session_cas_unlocked(
            session_path, updated, expected_revision=expected_revision
        )


def validate_form_review_session(
    *,
    frame: Mapping[str, Any],
    session: Mapping[str, Any],
    require_complete: bool,
) -> dict[str, Any]:
    if (
        session.get("schema_version") != SESSION_SCHEMA_VERSION
        or session.get("frame_id") != frame["frame_id"]
        or session.get("frame_payload_manifest_sha256")
        != frame["payload_manifest_sha256"]
    ):
        raise G3FormReferenceError("form-review session/frame binding differs")
    _trimmed_text(session.get("reviewer_id"), "reviewer_id", maximum=100)
    if session.get("finalized_reference_id") is not None and not isinstance(
        session.get("finalized_reference_id"), str
    ):
        raise G3FormReferenceError("finalized reference identity is invalid")
    item_by_id = {row["item_id"]: row for row in frame["items"]}
    evidence_by_id = {row["evidence_id"]: row for row in frame["evidence"]}
    decisions = session.get("decisions")
    if not isinstance(decisions, Mapping) or set(decisions) != set(item_by_id):
        raise G3FormReferenceError("form-review decision coverage differs")
    confirmed = 0
    deferred = 0
    action_counts: Counter[str] = Counter()
    for item_id, stored in decisions.items():
        if not isinstance(stored, Mapping) or set(stored) != {
            "status",
            "action",
            "surface",
            "canonical",
            "family",
            "phonetic_scan_enabled",
            "evidence_ids",
            "notes",
        }:
            raise G3FormReferenceError("stored form decision fields are not canonical")
        status = stored.get("status")
        if status not in {"draft", "confirmed"}:
            raise G3FormReferenceError("stored form decision status is invalid")
        _decision_replays(
            {key: copy.deepcopy(value) for key, value in stored.items() if key != "status"},
            item=item_by_id[item_id],
            evidence_by_id=evidence_by_id,
        )
        confirmed += int(status == "confirmed")
        deferred += int(status == "confirmed" and stored.get("action") == "defer")
        if status == "confirmed":
            action_counts[str(stored["action"])] += 1
    amendments = session.get("amendments")
    if not isinstance(amendments, list):
        raise G3FormReferenceError("form-review amendments must be an array")
    for row in amendments:
        if not isinstance(row, Mapping) or set(row) != {
            "item_id",
            "reason",
            "prior_revision",
            "reopened_at",
        }:
            raise G3FormReferenceError("form-review amendment fields are not canonical")
        if row.get("item_id") not in item_by_id or not SHA256_RE.fullmatch(
            str(row.get("prior_revision", ""))
        ):
            raise G3FormReferenceError("form-review amendment identity is invalid")
        _trimmed_text(row.get("reason"), "amendment reason", maximum=1000)
        _trimmed_text(row.get("reopened_at"), "amendment timestamp", maximum=100)
    if require_complete:
        if confirmed != len(item_by_id):
            raise G3FormReferenceError(
                f"form review is incomplete: {len(item_by_id) - confirmed} item(s) remain"
            )
        if deferred:
            raise G3FormReferenceError("deferred form decisions must be resolved before finalize")
    return {
        "item_count": len(item_by_id),
        "confirmed_count": confirmed,
        "deferred_count": deferred,
        "amendment_count": len(amendments),
        "action_counts": dict(sorted(action_counts.items())),
    }


def form_review_status(
    *,
    frame_dir: str | Path,
    source_bundle_dir: str | Path,
    workspace_root: str | Path,
    session_path: str | Path,
) -> dict[str, Any]:
    frame = validate_form_review_frame(
        frame_dir,
        source_bundle_dir=source_bundle_dir,
        workspace_root=workspace_root,
        require_current_implementation=True,
    )
    session = read_form_review_session(session_path)
    summary = validate_form_review_session(
        frame=frame, session=session, require_complete=False
    )
    return {
        "frame_id": frame["frame_id"],
        "reviewer_id": session["reviewer_id"],
        "revision": session["revision"],
        "finalized_reference_id": session["finalized_reference_id"],
        **summary,
    }


def _load_runtime_profile(
    profile_path: str | Path,
    *,
    romanizer: Callable[[str], list[str]] | None,
    initials_builder: Callable[[Sequence[str]], str] | None,
) -> tuple[dict[str, Any], Callable[[str], list[str]], Callable[[Sequence[str]], str]]:
    profile = _load_object(profile_path, "G3 runtime profile")
    if romanizer is not None and initials_builder is not None:
        return profile, romanizer, initials_builder
    try:
        from build_lex.terminology_candidate_generators_v2 import (
            build_pypinyin_romanizer,
            pinyin_initials,
            validate_g3_profile,
        )

        validated = validate_g3_profile(profile)
        if isinstance(validated, Mapping):
            profile = dict(validated)
        runtime_romanizer = build_pypinyin_romanizer(profile)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise G3FormReferenceError(
            "the frozen G3 romanizer/profile implementation is unavailable"
        ) from exc
    return profile, runtime_romanizer, pinyin_initials


def _build_reference_rows(
    *,
    frame: Mapping[str, Any],
    session: Mapping[str, Any],
    romanizer: Callable[[str], list[str]],
    initials_builder: Callable[[Sequence[str]], str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scans_by_canonical: dict[str, set[bool]] = defaultdict(set)
    evidence_by_id = {row["evidence_id"]: row for row in frame["evidence"]}
    for item_id in sorted(session["decisions"]):
        decision = session["decisions"][item_id]
        if decision["status"] != "confirmed":
            raise G3FormReferenceError("reference cannot use a draft decision")
        if decision["action"] not in {"accept", "edit"}:
            continue
        selected_quotes = [
            str(evidence_by_id[evidence_id]["quote"])
            for evidence_id in decision["evidence_ids"]
        ]
        if not any(decision["surface"] in quote for quote in selected_quotes) or not any(
            decision["canonical"] in quote for quote in selected_quotes
        ):
            raise G3FormReferenceError("accepted decision no longer replays from evidence")
        grouped[str(decision["canonical"])].append(
            {
                "surface": decision["surface"],
                "family": decision["family"],
                "evidence_ids": sorted(decision["evidence_ids"]),
            }
        )
        scans_by_canonical[str(decision["canonical"])].add(
            bool(decision["phonetic_scan_enabled"])
        )
    rows: list[dict[str, Any]] = []
    for canonical in sorted(grouped):
        if len(scans_by_canonical[canonical]) != 1:
            raise G3FormReferenceError(
                "accepted decisions disagree on canonical phonetic_scan_enabled"
            )
        try:
            pinyin = romanizer(canonical)
        except Exception as exc:  # backend exceptions are normalized at this boundary
            raise G3FormReferenceError(
                f"romanizer failed for accepted canonical {canonical!r}"
            ) from exc
        if (
            not isinstance(pinyin, list)
            or not pinyin
            or any(
                not isinstance(syllable, str)
                or not re.fullmatch(r"[a-z]+", syllable)
                for syllable in pinyin
            )
        ):
            raise G3FormReferenceError("romanizer returned a non-canonical syllable list")
        computed_initials = initials_builder(pinyin)
        if not isinstance(computed_initials, str) or not re.fullmatch(
            r"[a-z]+", computed_initials
        ):
            raise G3FormReferenceError("pinyin initials are invalid")
        merged: dict[tuple[str, str], set[str]] = defaultdict(set)
        family_by_surface: dict[str, set[str]] = defaultdict(set)
        for variant in grouped[canonical]:
            key = (str(variant["surface"]), str(variant["family"]))
            merged[key].update(str(value) for value in variant["evidence_ids"])
            family_by_surface[key[0]].add(key[1])
        conflicts = sorted(
            surface for surface, families in family_by_surface.items() if len(families) > 1
        )
        if conflicts:
            raise G3FormReferenceError(
                "one canonical/surface pair has conflicting families: "
                + ", ".join(conflicts)
            )
        variants = [
            {
                "surface": surface,
                "family": family,
                "evidence_ids": sorted(evidence_ids),
            }
            for (surface, family), evidence_ids in sorted(merged.items())
        ]
        initials = (
            computed_initials
            if any(
                variant["family"] == "phonetic_variant"
                and str(variant["surface"]).casefold() == computed_initials
                for variant in variants
            )
            else None
        )
        rows.append(
            {
                "canonical": canonical,
                "pinyin": list(pinyin),
                "initials": initials,
                "phonetic_scan_enabled": next(iter(scans_by_canonical[canonical])),
                "variants": variants,
            }
        )
    if not rows:
        raise G3FormReferenceError("final reference would be empty")
    return rows


def _reference_document(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "reference_role": REFERENCE_ROLE,
        "scope": SCOPE,
        "scientific_eligible": False,
        "sealed": False,
        "rows": [copy.deepcopy(dict(row)) for row in rows],
    }


def _profile_id(profile: Mapping[str, Any]) -> str:
    for key in ("profile_id", "profile_version", "schema_version"):
        value = profile.get(key)
        if isinstance(value, str) and value:
            return value
    raise G3FormReferenceError("G3 runtime profile has no stable identity")


def finalize_form_reference(
    *,
    frame_dir: str | Path,
    source_bundle_dir: str | Path,
    workspace_root: str | Path,
    session_path: str | Path,
    reviewer_id: str,
    romanizer_profile_path: str | Path,
    output_root: str | Path,
    reference_schema_path: str | Path,
    write_ref: str | Path | None = None,
    romanizer: Callable[[str], list[str]] | None = None,
    initials_builder: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, Any]:
    """Finalize one immutable snapshot while excluding all session writers."""

    with _form_review_session_lock(session_path, exclusive=True):
        return _finalize_form_reference_locked(
            frame_dir=frame_dir,
            source_bundle_dir=source_bundle_dir,
            workspace_root=workspace_root,
            session_path=session_path,
            reviewer_id=reviewer_id,
            romanizer_profile_path=romanizer_profile_path,
            output_root=output_root,
            reference_schema_path=reference_schema_path,
            write_ref=write_ref,
            romanizer=romanizer,
            initials_builder=initials_builder,
        )


def _finalize_form_reference_locked(
    *,
    frame_dir: str | Path,
    source_bundle_dir: str | Path,
    workspace_root: str | Path,
    session_path: str | Path,
    reviewer_id: str,
    romanizer_profile_path: str | Path,
    output_root: str | Path,
    reference_schema_path: str | Path,
    write_ref: str | Path | None = None,
    romanizer: Callable[[str], list[str]] | None = None,
    initials_builder: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, Any]:
    """Finalize only a fully confirmed review; caller holds the session lock."""

    frame = validate_form_review_frame(
        frame_dir,
        source_bundle_dir=source_bundle_dir,
        workspace_root=workspace_root,
        require_current_implementation=True,
    )
    session_file = Path(session_path)
    session_before = session_file.read_bytes()
    session = _read_form_review_session_unlocked(session_file)
    reviewer = _trimmed_text(reviewer_id, "reviewer_id", maximum=100)
    if session.get("reviewer_id") != reviewer:
        raise G3FormReferenceError("finalize reviewer does not own the session")
    if session.get("finalized_reference_id") is not None:
        existing = Path(output_root).resolve() / str(session["finalized_reference_id"])
        if not existing.is_dir():
            raise G3FormReferenceError("session names a missing finalized reference")
        return validate_form_reference(
            existing,
            frame_dir=frame_dir,
            source_bundle_dir=source_bundle_dir,
            workspace_root=workspace_root,
            romanizer_profile_path=romanizer_profile_path,
            reference_schema_path=reference_schema_path,
            romanizer=romanizer,
            initials_builder=initials_builder,
        )
    summary = validate_form_review_session(
        frame=frame, session=session, require_complete=True
    )
    profile, runtime_romanizer, runtime_initials = _load_runtime_profile(
        romanizer_profile_path,
        romanizer=romanizer,
        initials_builder=initials_builder,
    )
    rows = _build_reference_rows(
        frame=frame,
        session=session,
        romanizer=runtime_romanizer,
        initials_builder=runtime_initials,
    )
    reference = _reference_document(rows)
    schema = _load_object(reference_schema_path, "form-reference schema")
    _validate_schema(reference, Path(reference_schema_path).resolve())
    forbidden = _forbidden_key_paths(reference)
    if forbidden:
        raise G3FormReferenceError(
            "final reference contains forbidden keys: " + ", ".join(forbidden)
        )
    decisions = {
        item_id: copy.deepcopy(session["decisions"][item_id])
        for item_id in sorted(session["decisions"])
    }
    declaration = {
        "schema_version": DECLARATION_SCHEMA_VERSION,
        "reviewer_id": reviewer,
        "frame_id": frame["frame_id"],
        "source_session_revision": session["revision"],
        "confirmed_item_count": summary["confirmed_count"],
        "deferred_item_count": summary["deferred_count"],
        "amendment_count": summary["amendment_count"],
        "attestations": {
            "every_item_was_human_confirmed": True,
            "deferred_items_were_resolved": True,
            "accepted_forms_replay_from_frozen_public_evidence": True,
            "handbook_was_rubric_only": True,
            "task_labels_fit_gold_and_legacy_lexicon_were_not_used": True,
            "reference_is_form_only_label_free_non_lexicon": True,
            "development_only_nonsealed_nonscientific": True,
        },
        "declared_at": _utc_now(),
    }
    identity = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "artifact_kind": REFERENCE_ARTIFACT_KIND,
        "frame_dependency": _dependency(frame, FRAME_ARTIFACT_KIND, "frame_id"),
        "source_bundle_dependency": _dependency(
            frame["source"], SOURCE_BUNDLE_ARTIFACT_KIND, "source_bundle_id"
        ),
        "source_session_revision": session["revision"],
        "profile_id": _profile_id(profile),
        "profile_sha256": canonical_sha256(profile),
        "reference_schema_sha256": canonical_sha256(schema),
        "reference_sha256": canonical_sha256(reference),
        "decisions_sha256": canonical_sha256(decisions),
        "declaration_sha256": canonical_sha256(declaration),
        "canonical_count": len(rows),
        "variant_count": sum(len(row["variants"]) for row in rows),
        "action_counts": summary["action_counts"],
        "deferred_count": 0,
        "reference_role": REFERENCE_ROLE,
        "scope": SCOPE,
        "scientific_eligible": False,
        "sealed": False,
        "validator_implementation_sha256": sha256_file(Path(__file__)),
    }
    reference_id = REFERENCE_ID_PREFIX + canonical_sha256(identity)
    manifest = {**identity, "reference_id": reference_id}
    output_parent = Path(output_root).resolve()
    target = output_parent / reference_id
    if not target.exists():
        staging = new_staging_directory(output_parent, reference_id)
        try:
            write_canonical_json(staging / "manifest.json", manifest)
            write_canonical_json(staging / "reference.json", reference)
            write_canonical_json(staging / "decisions.json", decisions)
            write_canonical_json(staging / "reviewer_declaration.json", declaration)
            write_canonical_json(staging / "profile.json", profile)
            write_canonical_json(staging / "schema.json", schema)
            payload_hash = finalize_target_atomic(
                staging,
                target,
                validate_staging=lambda staged: _validate_form_reference_payload(
                    staged,
                    frame=frame,
                    expected_profile=profile,
                    schema_path=Path(reference_schema_path).resolve(),
                    romanizer=runtime_romanizer,
                    initials_builder=runtime_initials,
                    require_current_implementation=True,
                ),
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    else:
        payload_hash = validate_payload_manifest(target)
        _validate_form_reference_payload(
            target,
            frame=frame,
            expected_profile=profile,
            schema_path=Path(reference_schema_path).resolve(),
            romanizer=runtime_romanizer,
            initials_builder=runtime_initials,
            require_current_implementation=True,
        )
    if session_file.read_bytes() != session_before:
        raise G3FormReviewConflict("form-review session changed during finalize")
    updated = copy.deepcopy(session)
    updated["finalized_reference_id"] = reference_id
    updated["updated_at"] = _utc_now()
    _write_session_cas_unlocked(
        session_file, updated, expected_revision=session["revision"]
    )
    if write_ref is not None:
        write_locator_ref(
            write_ref,
            artifact_kind=REFERENCE_ARTIFACT_KIND,
            artifact_id=reference_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    return {
        "reference_id": reference_id,
        "target": str(target),
        "payload_manifest_sha256": payload_hash,
        "manifest": manifest,
        "reference": reference,
        "declaration": declaration,
    }


def _validate_form_reference_payload(
    target: Path,
    *,
    frame: Mapping[str, Any],
    expected_profile: Mapping[str, Any],
    schema_path: Path,
    romanizer: Callable[[str], list[str]],
    initials_builder: Callable[[Sequence[str]], str],
    require_current_implementation: bool,
) -> dict[str, Any]:
    try:
        payload_hash = validate_payload_manifest(target)
        ensure_exact_file_set(
            target,
            {
                "manifest.json",
                "reference.json",
                "decisions.json",
                "reviewer_declaration.json",
                "profile.json",
                "schema.json",
                "payload_manifest.json",
            },
        )
    except TrainingArtifactError as exc:
        raise G3FormReferenceError(str(exc)) from exc
    manifest = _load_object(target / "manifest.json", "form-reference manifest")
    reference = _load_object(target / "reference.json", "form reference")
    decisions = _load_object(target / "decisions.json", "frozen decisions")
    declaration = _load_object(
        target / "reviewer_declaration.json", "reviewer declaration"
    )
    profile = _load_object(target / "profile.json", "frozen G3 profile")
    schema = _load_object(target / "schema.json", "frozen form-reference schema")
    if profile != expected_profile or schema != _load_object(schema_path, "current form-reference schema"):
        raise G3FormReferenceError("form-reference profile or schema drifted")
    _validate_schema(reference, schema_path)
    forbidden = _forbidden_key_paths(reference)
    if forbidden:
        raise G3FormReferenceError("form reference contains forbidden task/data keys")
    reconstructed_session = {
        "decisions": decisions,
    }
    expected_rows = _build_reference_rows(
        frame=frame,
        session=reconstructed_session,
        romanizer=romanizer,
        initials_builder=initials_builder,
    )
    if reference != _reference_document(expected_rows):
        raise G3FormReferenceError("form reference does not replay from frozen decisions")
    expected_attestations = {
        "every_item_was_human_confirmed",
        "deferred_items_were_resolved",
        "accepted_forms_replay_from_frozen_public_evidence",
        "handbook_was_rubric_only",
        "task_labels_fit_gold_and_legacy_lexicon_were_not_used",
        "reference_is_form_only_label_free_non_lexicon",
        "development_only_nonsealed_nonscientific",
    }
    attestations = declaration.get("attestations")
    if (
        declaration.get("schema_version") != DECLARATION_SCHEMA_VERSION
        or not isinstance(attestations, Mapping)
        or set(attestations) != expected_attestations
        or any(attestations.get(key) is not True for key in expected_attestations)
        or declaration.get("deferred_item_count") != 0
    ):
        raise G3FormReferenceError("form-reference reviewer declaration is incomplete")
    identity = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "reference_id"}
    reference_id = REFERENCE_ID_PREFIX + canonical_sha256(identity)
    if manifest.get("reference_id") != reference_id or not _target_name_matches(target, reference_id):
        raise G3FormReferenceError("form-reference content-addressed identity differs")
    if (
        manifest.get("schema_version") != REFERENCE_SCHEMA_VERSION
        or manifest.get("artifact_kind") != REFERENCE_ARTIFACT_KIND
        or manifest.get("frame_dependency")
        != _dependency(frame, FRAME_ARTIFACT_KIND, "frame_id")
        or manifest.get("source_bundle_dependency")
        != _dependency(frame["source"], SOURCE_BUNDLE_ARTIFACT_KIND, "source_bundle_id")
        or manifest.get("profile_id") != _profile_id(profile)
        or manifest.get("profile_sha256") != canonical_sha256(profile)
        or manifest.get("reference_schema_sha256") != canonical_sha256(schema)
        or manifest.get("reference_sha256") != canonical_sha256(reference)
        or manifest.get("decisions_sha256") != canonical_sha256(decisions)
        or manifest.get("declaration_sha256") != canonical_sha256(declaration)
        or manifest.get("canonical_count") != len(reference["rows"])
        or manifest.get("variant_count")
        != sum(len(row["variants"]) for row in reference["rows"])
        or manifest.get("deferred_count") != 0
        or manifest.get("reference_role") != REFERENCE_ROLE
        or manifest.get("scope") != SCOPE
        or manifest.get("scientific_eligible") is not False
        or manifest.get("sealed") is not False
    ):
        raise G3FormReferenceError("form-reference manifest bindings differ")
    if require_current_implementation and manifest.get("validator_implementation_sha256") != sha256_file(Path(__file__)):
        raise G3FormReferenceError("form-reference validator implementation drifted")
    return {
        "reference_id": reference_id,
        "target": str(target),
        "payload_manifest_sha256": payload_hash,
        "manifest": manifest,
        "reference": reference,
        "declaration": declaration,
    }


def validate_form_reference(
    reference_dir: str | Path,
    *,
    frame_dir: str | Path,
    source_bundle_dir: str | Path,
    workspace_root: str | Path,
    romanizer_profile_path: str | Path,
    reference_schema_path: str | Path,
    require_current_implementation: bool = True,
    romanizer: Callable[[str], list[str]] | None = None,
    initials_builder: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, Any]:
    frame = validate_form_review_frame(
        frame_dir,
        source_bundle_dir=source_bundle_dir,
        workspace_root=workspace_root,
        require_current_implementation=require_current_implementation,
    )
    profile, runtime_romanizer, runtime_initials = _load_runtime_profile(
        romanizer_profile_path,
        romanizer=romanizer,
        initials_builder=initials_builder,
    )
    return _validate_form_reference_payload(
        Path(reference_dir).resolve(),
        frame=frame,
        expected_profile=profile,
        schema_path=Path(reference_schema_path).resolve(),
        romanizer=runtime_romanizer,
        initials_builder=runtime_initials,
        require_current_implementation=require_current_implementation,
    )


__all__ = [
    "ACTIONS",
    "FAMILIES",
    "FRAME_ARTIFACT_KIND",
    "G3FormReferenceError",
    "G3FormReviewConflict",
    "REFERENCE_ARTIFACT_KIND",
    "SOURCE_BUNDLE_ARTIFACT_KIND",
    "build_form_review_frame",
    "create_form_review_session",
    "fetch_public_source_snapshots",
    "finalize_form_reference",
    "form_review_status",
    "load_source_catalog",
    "read_form_review_session",
    "reopen_form_decision",
    "save_form_decision",
    "sync_public_sources",
    "validate_form_reference",
    "validate_form_review_frame",
    "validate_form_review_session",
    "validate_public_source_bundle",
]
