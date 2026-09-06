"""Offline, auditable intake for user-supplied WP3 G3 source archives.

The intake lifecycle is intentionally separate from the G3 source catalog and
form-reference lifecycle.  It copies explicitly mapped attachment payloads
into a content-addressed artifact, validates the archive containers, and emits
deterministic text/candidate projections.  It never performs network access
and it never promotes a source into the executable G3 reference.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlsplit
from zipfile import BadZipFile, ZipFile

from data.training_artifacts import (
    TrainingArtifactError,
    canonical_json_bytes,
    canonical_sha256,
    finalize_target_atomic,
    new_staging_directory,
    sha256_file,
    validate_payload_manifest,
    write_bytes_atomic,
    write_canonical_json,
)


SOURCE_MAP_SCHEMA_VERSION = "wp3-g3-manual-source-map/v1"
INTAKE_SCHEMA_VERSION = "wp3-g3-manual-source-intake/v1"
INTAKE_ARTIFACT_KIND = "wp3-g3-manual-source-intake"
INTAKE_ID_PREFIX = "wp3manual-"
INTAKE_POLICY_ID = "explicit-relative-attachment-offline-replay/v1"
CHIME_PROJECTION_SCHEMA_VERSION = "wp3-g3-chime-form-candidate-projection/v1"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PDF_HEADER_RE = re.compile(br"^%PDF-([12]\.[0-9])(?:\r?\n|\r)")
META_CHARSET_RE = re.compile(
    br"<meta\s+[^>]*(?:charset\s*=\s*['\"]?([^\s'\"/>;]+)|"
    br"content\s*=\s*['\"][^'\"]*charset=([^\s'\"/>;]+))",
    re.IGNORECASE,
)

MAX_ATTACHMENT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 256 * 1024 * 1024
MAX_ZIP_ENTRIES = 10_000
MAX_ZIP_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 200
MAX_PDF_PAGES = 1_000
MAX_PDF_TEXT_CHARACTERS = 20_000_000

PYPDF_VERSION = "6.0.0"
PYPDF_WHEEL_SHA256 = (
    "56ea60100ce9f11fc3eec4f359da15e9aec3821b036c1f06d2b660d35683abb8"
)
PDF_BACKEND_ID = "pypdf-6.0.0/extract-text-per-page-layout-false/v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEPENDENCY_LOCK_PATH = (
    REPOSITORY_ROOT / "config/stage1/wp3_g3_source_intake_dependency_lock_v1.json"
)
REQUIREMENTS_LOCK_PATH = REPOSITORY_ROOT / "environment/wp3-g3-source-intake-requirements.lock"
SOURCE_MAP_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/wp3_g3_manual_source_map_v1.schema.json"
INTAKE_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/wp3_g3_manual_source_intake_v1.schema.json"

FORM_CANDIDATE_TYPES = ("abbreviation", "homophonic pun")
CHIME_TYPE_PAIRS = {
    "abbreviation": "缩写",
    "experience": "现象",
    "homophonic pun": "谐音",
    "quotation": "引用",
    "slang": "俗语",
    "stylistic device": "修辞",
}
CHIME_ROW_FIELDS = {
    "meme",
    "meaning",
    "origin",
    "examples",
    "profanity",
    "offense",
    "type_cn",
    "type_en",
}

FORMATS = frozenset({"mhtml", "pdf", "json", "zip", "text"})
ACQUISITION_MODES = frozenset(
    {
        "user_supplied_archive",
        "publisher_pdf",
        "repository_archive",
        "repository_data",
        "license_file",
    }
)
SOURCE_ROLES = frozenset(
    {
        "direct_evidence",
        "candidate_pool",
        "prevalence_or_taxonomy",
        "license_or_provenance",
    }
)
DISPOSITIONS = frozenset(
    {"evidence_eligible", "candidate_only", "acquisition_only"}
)

MAP_TOP_FIELDS = {
    "schema_version",
    "map_id",
    "scope",
    "network_access_allowed",
    "attachments",
    "companion_contracts",
}
ATTACHMENT_FIELDS = {
    "alias_id",
    "source_id",
    "component_id",
    "attachment_locator",
    "requested_url",
    "final_url",
    "snapshot_url",
    "acquired_at",
    "title",
    "acquisition_mode",
    "source_role",
    "format",
    "disposition",
    "expected_size_bytes",
    "expected_sha256",
    "pagination",
    "wikimedia_revision_id",
}
CONTRACT_FIELDS = {
    "contract_id",
    "repository_commit",
    "archive_component_id",
    "data_component_id",
    "license_component_id",
    "archive_comment",
    "member_bindings",
    "expected_record_count",
    "expected_candidate_count",
    "expected_candidate_type_counts",
}


class ManualSourceIntakeError(RuntimeError):
    """Raised when a manual archive cannot be safely imported or replayed."""


class _VisibleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ignored_depth = 0
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.title_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript", "template", "svg"}:
            self.ignored_depth += 1
        if normalized == "title":
            self.title_depth += 1

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript", "template", "svg"}:
            self.ignored_depth = max(0, self.ignored_depth - 1)
        if normalized == "title":
            self.title_depth = max(0, self.title_depth - 1)

    def handle_data(self, data: str) -> None:
        if self.title_depth and data.strip():
            self.title_parts.append(data)
        if not self.ignored_depth and data.strip():
            self.parts.append(data)


def _reject_constant(value: str) -> Any:
    raise ManualSourceIntakeError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManualSourceIntakeError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ManualSourceIntakeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManualSourceIntakeError(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def _strict_json_file(path: Path, *, label: str) -> Any:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ManualSourceIntakeError(f"cannot read {label}: {path}") from exc
    return _strict_json_bytes(payload, label=label)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManualSourceIntakeError(f"{label} must be an object")
    return dict(value)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise ManualSourceIntakeError(f"{label} must be a lowercase identifier")
    return value


def _http_url(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ManualSourceIntakeError(f"{label} must be an HTTP(S) URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ManualSourceIntakeError(f"{label} is malformed") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ManualSourceIntakeError(f"{label} must be a safe absolute HTTP(S) URL")
    return value


def _utc_or_offset_datetime(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:Z|[+-][0-9]{2}:[0-9]{2})",
        value,
    ):
        raise ManualSourceIntakeError(f"{label} must be a second-precision datetime")
    return value


def _relative_locator(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ManualSourceIntakeError("attachment_locator is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ManualSourceIntakeError("attachment_locator must be a normalized relative path")
    if path.as_posix() != value:
        raise ManualSourceIntakeError("attachment_locator is not canonical")
    return value


def _nullable_pagination(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    row = _object(value, "pagination")
    if set(row) != {"page_index", "page_count"}:
        raise ManualSourceIntakeError("pagination fields differ")
    page_index = row.get("page_index")
    page_count = row.get("page_count")
    if (
        isinstance(page_index, bool)
        or not isinstance(page_index, int)
        or isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count < 2
        or not 1 <= page_index <= page_count
    ):
        raise ManualSourceIntakeError("pagination values are invalid")
    return {"page_index": page_index, "page_count": page_count}


def _validate_attachment_row(raw: Any) -> dict[str, Any]:
    row = _object(raw, "attachment mapping")
    if set(row) != ATTACHMENT_FIELDS:
        raise ManualSourceIntakeError("attachment mapping fields differ")
    for field in ("alias_id", "source_id", "component_id"):
        row[field] = _identifier(row.get(field), field)
    row["attachment_locator"] = _relative_locator(row.get("attachment_locator"))
    for field in ("requested_url", "final_url", "snapshot_url"):
        row[field] = _http_url(row.get(field), field)
    row["acquired_at"] = _utc_or_offset_datetime(row.get("acquired_at"), "acquired_at")
    if row.get("title") is not None and (
        not isinstance(row["title"], str) or not row["title"].strip()
    ):
        raise ManualSourceIntakeError("title must be null or a non-empty string")
    if row.get("acquisition_mode") not in ACQUISITION_MODES:
        raise ManualSourceIntakeError("acquisition_mode is unsupported")
    if row.get("source_role") not in SOURCE_ROLES:
        raise ManualSourceIntakeError("source_role is unsupported")
    if row.get("format") not in FORMATS:
        raise ManualSourceIntakeError("format is unsupported")
    if row.get("disposition") not in DISPOSITIONS:
        raise ManualSourceIntakeError("disposition is unsupported")
    size = row.get("expected_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_ATTACHMENT_BYTES:
        raise ManualSourceIntakeError("expected_size_bytes is invalid")
    if not isinstance(row.get("expected_sha256"), str) or not SHA256_RE.fullmatch(
        row["expected_sha256"]
    ):
        raise ManualSourceIntakeError("expected_sha256 is invalid")
    row["pagination"] = _nullable_pagination(row.get("pagination"))
    revision = row.get("wikimedia_revision_id")
    if revision is not None and (
        not isinstance(revision, str) or not re.fullmatch(r"[1-9][0-9]*", revision)
    ):
        raise ManualSourceIntakeError("wikimedia_revision_id is invalid")
    return row


def _validate_companion_contract(raw: Any) -> dict[str, Any]:
    row = _object(raw, "companion contract")
    if set(row) != CONTRACT_FIELDS:
        raise ManualSourceIntakeError("companion contract fields differ")
    row["contract_id"] = _identifier(row.get("contract_id"), "contract_id")
    commit = row.get("repository_commit")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise ManualSourceIntakeError("repository_commit is invalid")
    for field in ("archive_component_id", "data_component_id", "license_component_id"):
        row[field] = _identifier(row.get(field), field)
    if row.get("archive_comment") != commit:
        raise ManualSourceIntakeError("archive_comment must exactly pin repository_commit")
    bindings = row.get("member_bindings")
    if not isinstance(bindings, list) or len(bindings) < 2:
        raise ManualSourceIntakeError("member_bindings must contain data and license")
    seen_members: set[str] = set()
    seen_components: set[str] = set()
    normalized_bindings: list[dict[str, str]] = []
    for raw_binding in bindings:
        binding = _object(raw_binding, "member binding")
        if set(binding) != {"member_path", "component_id"}:
            raise ManualSourceIntakeError("member binding fields differ")
        member_path = _relative_locator(binding.get("member_path"))
        component_id = _identifier(binding.get("component_id"), "member component_id")
        if member_path in seen_members or component_id in seen_components:
            raise ManualSourceIntakeError("member bindings are duplicated")
        seen_members.add(member_path)
        seen_components.add(component_id)
        normalized_bindings.append(
            {"member_path": member_path, "component_id": component_id}
        )
    if {
        row["data_component_id"],
        row["license_component_id"],
    } != seen_components:
        raise ManualSourceIntakeError("member bindings do not bind data and license companions")
    row["member_bindings"] = sorted(
        normalized_bindings, key=lambda value: value["member_path"]
    )
    for field in ("expected_record_count", "expected_candidate_count"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ManualSourceIntakeError(f"{field} is invalid")
    counts = _object(
        row.get("expected_candidate_type_counts"), "expected_candidate_type_counts"
    )
    if set(counts) != set(FORM_CANDIDATE_TYPES) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts.values()
    ):
        raise ManualSourceIntakeError("expected candidate type counts are invalid")
    if sum(counts.values()) != row["expected_candidate_count"]:
        raise ManualSourceIntakeError("candidate type counts do not sum to expected count")
    row["expected_candidate_type_counts"] = {
        key: counts[key] for key in FORM_CANDIDATE_TYPES
    }
    return row


def validate_manual_source_map_value(value: Any) -> dict[str, Any]:
    """Validate and canonicalize a manual-source mapping value."""

    mapping = _object(value, "manual source map")
    if set(mapping) != MAP_TOP_FIELDS:
        raise ManualSourceIntakeError("manual source map fields differ")
    if mapping.get("schema_version") != SOURCE_MAP_SCHEMA_VERSION:
        raise ManualSourceIntakeError("manual source map schema differs")
    mapping["map_id"] = _identifier(mapping.get("map_id"), "map_id")
    if mapping.get("scope") != "development-only-form-only-label-free-non-lexicon":
        raise ManualSourceIntakeError("manual source map scope differs")
    if mapping.get("network_access_allowed") is not False:
        raise ManualSourceIntakeError("manual source map must forbid network access")
    raw_attachments = mapping.get("attachments")
    if not isinstance(raw_attachments, list) or not raw_attachments:
        raise ManualSourceIntakeError("manual source map has no attachments")
    attachments = [_validate_attachment_row(row) for row in raw_attachments]
    aliases = [row["alias_id"] for row in attachments]
    locators = [row["attachment_locator"] for row in attachments]
    if len(aliases) != len(set(aliases)) or len(locators) != len(set(locators)):
        raise ManualSourceIntakeError("attachment aliases and locators must be unique")
    by_component: dict[str, list[dict[str, Any]]] = {}
    for row in attachments:
        by_component.setdefault(row["component_id"], []).append(row)
    invariant_fields = ATTACHMENT_FIELDS - {
        "alias_id",
        "attachment_locator",
    }
    for component_id, rows in by_component.items():
        first = rows[0]
        for row in rows[1:]:
            if any(row[field] != first[field] for field in invariant_fields):
                raise ManualSourceIntakeError(
                    f"aliases for component {component_id} disagree"
                )
    raw_contracts = mapping.get("companion_contracts")
    if not isinstance(raw_contracts, list):
        raise ManualSourceIntakeError("companion_contracts must be an array")
    contracts = [_validate_companion_contract(row) for row in raw_contracts]
    contract_ids = [row["contract_id"] for row in contracts]
    if len(contract_ids) != len(set(contract_ids)):
        raise ManualSourceIntakeError("companion contract IDs are duplicated")
    component_ids = set(by_component)
    claimed_components: set[str] = set()
    for contract in contracts:
        required = {
            contract["archive_component_id"],
            contract["data_component_id"],
            contract["license_component_id"],
        }
        if not required.issubset(component_ids):
            raise ManualSourceIntakeError("companion contract references unknown component")
        if claimed_components.intersection(required):
            raise ManualSourceIntakeError("a component occurs in multiple companion contracts")
        claimed_components.update(required)
        formats = {
            row["component_id"]: row["format"]
            for row in attachments
            if row["component_id"] in required
        }
        if (
            formats[contract["archive_component_id"]] != "zip"
            or formats[contract["data_component_id"]] != "json"
            or formats[contract["license_component_id"]] != "text"
        ):
            raise ManualSourceIntakeError("companion component formats differ")
    mapping["attachments"] = sorted(attachments, key=lambda row: row["alias_id"])
    mapping["companion_contracts"] = sorted(
        contracts, key=lambda row: row["contract_id"]
    )
    return mapping


def load_manual_source_map(path: str | Path) -> dict[str, Any]:
    """Load a strict, duplicate-key-free manual attachment mapping."""

    return validate_manual_source_map_value(
        _strict_json_file(Path(path), label="manual source map")
    )


def _safe_attachment(root: Path, locator: str) -> Path:
    if not root.is_dir() or root.is_symlink():
        raise ManualSourceIntakeError("attachment root must be a non-symlink directory")
    current = root
    parts = PurePosixPath(_relative_locator(locator)).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            status = os.lstat(current)
        except OSError as exc:
            raise ManualSourceIntakeError(f"attachment is unavailable: {locator}") from exc
        if stat.S_ISLNK(status.st_mode):
            raise ManualSourceIntakeError(f"attachment path contains a symlink: {locator}")
        if index < len(parts) - 1 and not stat.S_ISDIR(status.st_mode):
            raise ManualSourceIntakeError(f"attachment parent is not a directory: {locator}")
        if index == len(parts) - 1 and not stat.S_ISREG(status.st_mode):
            raise ManualSourceIntakeError(f"attachment is not a regular file: {locator}")
    return current


def _read_attachment(root: Path, row: Mapping[str, Any]) -> bytes:
    path = _safe_attachment(root, str(row["attachment_locator"]))
    before = path.stat()
    if before.st_size != row["expected_size_bytes"] or before.st_size > MAX_ATTACHMENT_BYTES:
        raise ManualSourceIntakeError(
            f"attachment size differs: {row['attachment_locator']}"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ManualSourceIntakeError(
            f"cannot read attachment: {row['attachment_locator']}"
        ) from exc
    after = path.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ManualSourceIntakeError("attachment changed while it was read")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != row["expected_sha256"]:
        raise ManualSourceIntakeError(
            f"attachment SHA256 differs: {row['attachment_locator']}"
        )
    return payload


def _decode_html(payload: bytes, content_type_charset: str | None) -> tuple[str, str]:
    encodings: list[str] = []
    if payload.startswith(b"\xef\xbb\xbf"):
        encodings.append("utf-8-sig")
    if content_type_charset:
        encodings.append(content_type_charset)
    match = META_CHARSET_RE.search(payload[:65536])
    if match:
        charset = next((part for part in match.groups() if part), None)
        if charset:
            try:
                encodings.append(charset.decode("ascii", errors="strict"))
            except UnicodeDecodeError:
                pass
    encodings.extend(["utf-8", "gb18030", "big5"])
    tried: set[str] = set()
    for encoding in encodings:
        normalized = encoding.casefold()
        if normalized in tried:
            continue
        tried.add(normalized)
        try:
            return payload.decode(encoding, errors="strict"), normalized
        except (LookupError, UnicodeDecodeError):
            continue
    raise ManualSourceIntakeError("MHTML root HTML has no supported strict encoding")


def _visible_text(html: str) -> tuple[bytes, str | None, int]:
    if "\x00" in html:
        raise ManualSourceIntakeError("MHTML root HTML contains NUL")
    parser = _VisibleHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:
        raise ManualSourceIntakeError("MHTML root HTML cannot be parsed") from exc
    visible = " ".join(" ".join(parser.parts).split())
    title = " ".join(" ".join(parser.title_parts).split()) or None
    if len(visible) < 80:
        raise ManualSourceIntakeError("MHTML visible-text projection is too short")
    return (visible + "\n").encode("utf-8"), title, len(visible)


def _analyze_mhtml(
    payload: bytes, component: Mapping[str, Any]
) -> tuple[bytes, dict[str, Any], bytes | None]:
    try:
        message = BytesParser(policy=policy.default).parsebytes(payload)
    except Exception as exc:
        raise ManualSourceIntakeError("MHTML MIME parsing failed") from exc
    if (
        message.defects
        or not message.is_multipart()
        or message.get_content_type() != "multipart/related"
    ):
        raise ManualSourceIntakeError("MHTML must be a defect-free multipart/related archive")
    snapshot_location = message.get("Snapshot-Content-Location")
    date_header = message.get("Date")
    if snapshot_location != component["snapshot_url"]:
        raise ManualSourceIntakeError("MHTML Snapshot-Content-Location differs")
    if not isinstance(date_header, str):
        raise ManualSourceIntakeError("MHTML Date header is missing")
    try:
        parsed_date = parsedate_to_datetime(date_header)
    except (TypeError, ValueError) as exc:
        raise ManualSourceIntakeError("MHTML Date header is invalid") from exc
    if parsed_date.tzinfo is None:
        raise ManualSourceIntakeError("MHTML Date header lacks an offset")
    dated = parsed_date.isoformat(timespec="seconds")
    if dated != component["acquired_at"]:
        raise ManualSourceIntakeError("MHTML Date does not match mapped acquired_at")
    body_parts = message.get_payload()
    if not isinstance(body_parts, list) or not body_parts:
        raise ManualSourceIntakeError("MHTML multipart/related body is empty")
    related_start = message.get_param("start", header="Content-Type")
    if related_start is None:
        root = body_parts[0]
        root_selection = "first-body-part"
    else:
        start_matches = [
            part for part in message.walk() if part.get("Content-ID") == related_start
        ]
        if len(start_matches) != 1:
            raise ManualSourceIntakeError("MHTML related start does not resolve uniquely")
        root = start_matches[0]
        root_selection = "content-type-start"
    if (
        root.get_content_type() not in {"text/html", "application/xhtml+xml"}
        or root.get("Content-Location") != snapshot_location
    ):
        raise ManualSourceIntakeError("MHTML related root is not the snapshot HTML")
    if root.defects:
        raise ManualSourceIntakeError("MHTML root HTML MIME part is defective")
    root_payload = root.get_payload(decode=True)
    if not isinstance(root_payload, bytes) or not root_payload:
        raise ManualSourceIntakeError("MHTML root HTML MIME payload is missing")
    html, encoding = _decode_html(root_payload, root.get_content_charset())
    projection, html_title, visible_characters = _visible_text(html)
    declared_title = component.get("title")
    subject = str(message.get("Subject") or "").strip() or None
    if declared_title is not None and declared_title not in {subject, html_title}:
        raise ManualSourceIntakeError("MHTML declared title does not match Subject or HTML title")
    revision = component.get("wikimedia_revision_id")
    if revision is not None:
        snapshot_query = parse_qs(urlsplit(str(snapshot_location)).query)
        if snapshot_query.get("oldid") != [revision]:
            raise ManualSourceIntakeError("MHTML snapshot URL does not bind Wikimedia oldid")
    duplicate_location_parts: list[dict[str, Any]] = []
    for part in message.walk():
        if part is root or part.get("Content-Location") != snapshot_location:
            continue
        duplicate_payload = part.get_payload(decode=True)
        if not isinstance(duplicate_payload, bytes):
            raise ManualSourceIntakeError("MHTML duplicate-location MIME part is invalid")
        duplicate_location_parts.append(
            {
                "content_type": part.get_content_type(),
                "content_id": part.get("Content-ID"),
                "transfer_encoding": part.get("Content-Transfer-Encoding"),
                "size_bytes": len(duplicate_payload),
                "sha256": hashlib.sha256(duplicate_payload).hexdigest(),
            }
        )
    metadata = {
        "container": "multipart/related",
        "related_start": related_start,
        "root_selection": root_selection,
        "snapshot_content_location": snapshot_location,
        "date_header": date_header,
        "date_iso8601": dated,
        "subject": subject,
        "root_content_type": root.get_content_type(),
        "root_content_location": root.get("Content-Location"),
        "root_content_id": root.get("Content-ID"),
        "root_transfer_encoding": root.get("Content-Transfer-Encoding"),
        "root_size_bytes": len(root_payload),
        "root_sha256": hashlib.sha256(root_payload).hexdigest(),
        "root_text_encoding": encoding,
        "html_title": html_title,
        "visible_text_characters": visible_characters,
        "mime_part_count": sum(1 for _ in message.walk()),
        "duplicate_location_count": len(duplicate_location_parts),
        "duplicate_location_parts": duplicate_location_parts,
    }
    return projection, metadata, None


def _normalize_pdf_page(text: str) -> str:
    value = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\v\f]+", " ", line).strip() for line in value.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    compact: list[str] = []
    for line in lines:
        if not line and compact and not compact[-1]:
            continue
        compact.append(line)
    return "\n".join(compact)


def _scan_pdf_dangerous_objects(value: Any, *, seen: set[int], depth: int = 0) -> None:
    if depth > 128:
        raise ManualSourceIntakeError("PDF object graph is too deep")
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    try:
        from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject
    except ImportError as exc:  # pragma: no cover - guarded by _analyze_pdf
        raise ManualSourceIntakeError("pypdf backend is unavailable") from exc
    if isinstance(value, IndirectObject):
        try:
            resolved = value.get_object()
        except Exception as exc:
            raise ManualSourceIntakeError("PDF indirect object cannot be resolved") from exc
        _scan_pdf_dangerous_objects(resolved, seen=seen, depth=depth + 1)
        return
    if isinstance(value, DictionaryObject):
        forbidden_keys = {"/OpenAction", "/JavaScript", "/JS", "/Launch", "/EmbeddedFiles"}
        if forbidden_keys.intersection(str(key) for key in value.keys()):
            raise ManualSourceIntakeError("PDF contains an active or embedded-content key")
        action = value.get("/S")
        if str(action) in {"/JavaScript", "/Launch"}:
            raise ManualSourceIntakeError("PDF contains a forbidden action")
        for child in value.values():
            _scan_pdf_dangerous_objects(child, seen=seen, depth=depth + 1)
    elif isinstance(value, ArrayObject):
        for child in value:
            _scan_pdf_dangerous_objects(child, seen=seen, depth=depth + 1)


def _analyze_pdf(
    payload: bytes, component: Mapping[str, Any]
) -> tuple[bytes, dict[str, Any], bytes | None]:
    del component
    if not PDF_HEADER_RE.search(payload) or b"%%EOF" not in payload[-8192:]:
        raise ManualSourceIntakeError("PDF basic container markers are invalid")
    startxref = re.search(br"startxref\s+([0-9]+)\s+%%EOF", payload[-16384:])
    if startxref is None or int(startxref.group(1)) >= len(payload):
        raise ManualSourceIntakeError("PDF startxref is missing or out of bounds")
    try:
        import pypdf
        from pypdf import PdfReader
    except ImportError as exc:
        raise ManualSourceIntakeError(
            f"PDF intake requires fixed pypdf=={PYPDF_VERSION}"
        ) from exc
    if getattr(pypdf, "__version__", None) != PYPDF_VERSION:
        raise ManualSourceIntakeError(
            f"PDF intake requires fixed pypdf=={PYPDF_VERSION}"
        )
    try:
        reader = PdfReader(io.BytesIO(payload), strict=True)
    except Exception as exc:
        raise ManualSourceIntakeError("pypdf rejected the PDF container") from exc
    if reader.is_encrypted:
        raise ManualSourceIntakeError("encrypted PDF is forbidden")
    if not 0 < len(reader.pages) <= MAX_PDF_PAGES:
        raise ManualSourceIntakeError("PDF page count is invalid")
    try:
        root = reader.trailer["/Root"]
    except Exception as exc:
        raise ManualSourceIntakeError("PDF catalog is unavailable") from exc
    _scan_pdf_dangerous_objects(root, seen=set())
    page_texts: list[str] = []
    total_characters = 0
    for page in reader.pages:
        try:
            extracted = page.extract_text()
        except Exception as exc:
            raise ManualSourceIntakeError("pypdf page text extraction failed") from exc
        if not isinstance(extracted, str):
            extracted = ""
        normalized = _normalize_pdf_page(extracted)
        total_characters += len(normalized)
        if total_characters > MAX_PDF_TEXT_CHARACTERS:
            raise ManualSourceIntakeError("PDF text projection is too large")
        page_texts.append(normalized)
    if not any(page_texts):
        raise ManualSourceIntakeError("PDF has no extractable text")
    joined = ""
    pages: list[dict[str, Any]] = []
    for index, page_text in enumerate(page_texts, start=1):
        if index > 1:
            joined += "\n\n"
        start = len(joined)
        joined += page_text
        pages.append(
            {
                "page_number": index,
                "start_offset": start,
                "end_offset": len(joined),
                "text_characters": len(page_text),
                "text_sha256": hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
            }
        )
    projection = (joined + "\n").encode("utf-8")
    metadata = {
        "container_validation": "valid-pdf-pypdf-strict",
        "pdf_header_version": PDF_HEADER_RE.search(payload).group(1).decode("ascii"),
        "backend_id": PDF_BACKEND_ID,
        "pypdf_version": PYPDF_VERSION,
        "pypdf_wheel_sha256": PYPDF_WHEEL_SHA256,
        "encrypted": False,
        "active_content_scan": "passed",
        "page_count": len(pages),
        "text_characters": len(joined),
        "pages": pages,
    }
    return projection, metadata, None


def normalize_publisher_pdf_bytes(
    payload: bytes,
    *,
    component_id: str,
    requested_url: str,
    final_url: str,
) -> tuple[bytes, dict[str, Any]]:
    """Normalize one publisher PDF through the frozen fail-closed backend.

    This helper performs no I/O beyond verifying the repository-owned parser
    dependency locks.  It is shared by catalog-v2 direct-capture migration so
    manually supplied and directly captured PDFs use identical extraction and
    active-content gates.
    """

    _identifier(component_id, "component_id")
    _http_url(requested_url, "requested_url")
    _http_url(final_url, "final_url")
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_ATTACHMENT_BYTES:
        raise ManualSourceIntakeError("publisher PDF payload size is invalid")
    _dependency_binding()
    projection, metadata, candidate = _analyze_pdf(payload, {})
    if candidate is not None:  # pragma: no cover - internal invariant
        raise ManualSourceIntakeError("PDF unexpectedly produced a candidate projection")
    return projection, metadata


def _validate_chime_rows(
    parsed: Any,
    *,
    component_id: str,
    raw_sha256: str,
    contract: Mapping[str, Any],
) -> tuple[bytes, bytes, dict[str, Any]]:
    if not isinstance(parsed, list) or len(parsed) != contract["expected_record_count"]:
        raise ManualSourceIntakeError("CHIME record count differs")
    candidate_rows: list[dict[str, Any]] = []
    all_type_counts: Counter[str] = Counter()
    for ordinal, raw_row in enumerate(parsed, start=1):
        row = _object(raw_row, f"CHIME row {ordinal}")
        if set(row) != CHIME_ROW_FIELDS:
            raise ManualSourceIntakeError(f"CHIME row {ordinal} fields differ")
        if any(
            not isinstance(row[field], str) or not row[field].strip()
            for field in ("meme", "meaning", "type_cn", "type_en")
        ):
            raise ManualSourceIntakeError(f"CHIME row {ordinal} string fields are invalid")
        origin = row.get("origin")
        if origin is not None and (not isinstance(origin, str) or not origin.strip()):
            raise ManualSourceIntakeError(f"CHIME row {ordinal} origin is invalid")
        examples = row.get("examples")
        if not isinstance(examples, list) or not 1 <= len(examples) <= 100 or any(
            not isinstance(example, str) or not example.strip() for example in examples
        ):
            raise ManualSourceIntakeError(f"CHIME row {ordinal} examples are invalid")
        if not isinstance(row.get("profanity"), bool) or not isinstance(row.get("offense"), bool):
            raise ManualSourceIntakeError(f"CHIME row {ordinal} label fields are invalid")
        type_en = row["type_en"]
        if type_en not in CHIME_TYPE_PAIRS or row["type_cn"] != CHIME_TYPE_PAIRS[type_en]:
            raise ManualSourceIntakeError(f"CHIME row {ordinal} type pair is invalid")
        all_type_counts[type_en] += 1
        if type_en in FORM_CANDIDATE_TYPES:
            candidate_rows.append(
                {
                    "source_row_ordinal": ordinal,
                    "meme": row["meme"],
                    "meaning": row["meaning"],
                    "origin": origin,
                    "type_cn": row["type_cn"],
                    "type_en": type_en,
                }
            )
    candidate_counts = Counter(row["type_en"] for row in candidate_rows)
    if len(candidate_rows) != contract["expected_candidate_count"] or {
        key: candidate_counts[key] for key in FORM_CANDIDATE_TYPES
    } != contract["expected_candidate_type_counts"]:
        raise ManualSourceIntakeError("CHIME candidate counts differ")
    projection = {
        "schema_version": CHIME_PROJECTION_SCHEMA_VERSION,
        "source_component_id": component_id,
        "raw_sha256": raw_sha256,
        "source_record_count": len(parsed),
        "allowed_types": list(FORM_CANDIDATE_TYPES),
        "candidate_count": len(candidate_rows),
        "rows": candidate_rows,
    }
    text_parts: list[str] = []
    for row in candidate_rows:
        text_parts.extend(
            [
                f"[source_row_ordinal={row['source_row_ordinal']}]",
                f"meme: {row['meme']}",
                f"meaning: {row['meaning']}",
                f"origin: {row['origin'] if row['origin'] is not None else 'null'}",
                f"type_en: {row['type_en']}",
                "",
            ]
        )
    text_bytes = ("\n".join(text_parts).rstrip() + "\n").encode("utf-8")
    metadata = {
        "json_validation": "strict-utf8-no-duplicate-keys-no-nonfinite",
        "dataset_contract": "chime-full/v1",
        "record_count": len(parsed),
        "type_counts": dict(sorted(all_type_counts.items())),
        "candidate_count": len(candidate_rows),
        "candidate_type_counts": {
            key: candidate_counts[key] for key in FORM_CANDIDATE_TYPES
        },
        "excluded_fields_from_candidate_projection": [
            "examples",
            "offense",
            "profanity",
        ],
    }
    return text_bytes, canonical_json_bytes(projection) + b"\n", metadata


def _analyze_json(
    payload: bytes,
    component: Mapping[str, Any],
    *,
    raw_sha256: str,
    companion_contract: Mapping[str, Any] | None,
) -> tuple[bytes, dict[str, Any], bytes | None]:
    parsed = _strict_json_bytes(payload, label=f"JSON component {component['component_id']}")
    if (
        companion_contract is not None
        and component["component_id"] == companion_contract["data_component_id"]
    ):
        projection, candidates, metadata = _validate_chime_rows(
            parsed,
            component_id=str(component["component_id"]),
            raw_sha256=raw_sha256,
            contract=companion_contract,
        )
        return projection, metadata, candidates
    normalized = canonical_json_bytes(parsed) + b"\n"
    return (
        normalized,
        {
            "json_validation": "strict-utf8-no-duplicate-keys-no-nonfinite",
            "root_type": type(parsed).__name__,
            "item_count": len(parsed) if isinstance(parsed, (list, dict)) else None,
        },
        None,
    )


def _analyze_text(
    payload: bytes, component: Mapping[str, Any]
) -> tuple[bytes | None, dict[str, Any], bytes | None]:
    del component
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ManualSourceIntakeError("text attachment is not strict UTF-8") from exc
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ManualSourceIntakeError("text attachment is empty")
    normalized = (text.rstrip() + "\n").encode("utf-8")
    return normalized, {"text_encoding": "utf-8", "text_characters": len(text.rstrip())}, None


def _zip_member_path(value: str) -> str:
    if "\\" in value or "\x00" in value:
        raise ManualSourceIntakeError("ZIP member path is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ManualSourceIntakeError("ZIP member path is unsafe")
    return path.as_posix()


def _analyze_zip(
    payload: bytes,
    component: Mapping[str, Any],
    *,
    companion_contract: Mapping[str, Any] | None,
    component_payloads: Mapping[str, bytes],
) -> tuple[None, dict[str, Any], None]:
    if (
        companion_contract is None
        or component["component_id"]
        != companion_contract["archive_component_id"]
    ):
        raise ManualSourceIntakeError("ZIP component lacks a companion contract")
    try:
        archive = ZipFile(io.BytesIO(payload), mode="r")
    except BadZipFile as exc:
        raise ManualSourceIntakeError("ZIP container is invalid") from exc
    with archive:
        infos = archive.infolist()
        if not 0 < len(infos) <= MAX_ZIP_ENTRIES:
            raise ManualSourceIntakeError("ZIP entry count is invalid")
        names: set[str] = set()
        total_uncompressed = 0
        for info in infos:
            name = _zip_member_path(info.filename.rstrip("/")) if info.filename.rstrip("/") else ""
            if not name or name in names:
                if name in names:
                    raise ManualSourceIntakeError("ZIP member names are duplicated")
                if not info.is_dir():
                    raise ManualSourceIntakeError("ZIP contains an empty member name")
            if name:
                names.add(name)
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise ManualSourceIntakeError("ZIP symlink members are forbidden")
            if info.flag_bits & 0x1:
                raise ManualSourceIntakeError("encrypted ZIP members are forbidden")
            if info.file_size > MAX_ZIP_MEMBER_BYTES:
                raise ManualSourceIntakeError("ZIP member is too large")
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise ManualSourceIntakeError("ZIP uncompressed budget is exceeded")
            if (
                (info.file_size > 0 and info.compress_size == 0)
                or (
                    info.compress_size > 0
                    and info.file_size / info.compress_size
                    > MAX_ZIP_COMPRESSION_RATIO
                )
            ):
                raise ManualSourceIntakeError("ZIP compression ratio is unsafe")
        try:
            corrupt = archive.testzip()
        except (BadZipFile, RuntimeError, OSError) as exc:
            raise ManualSourceIntakeError("ZIP CRC replay failed") from exc
        if corrupt is not None:
            raise ManualSourceIntakeError(f"ZIP CRC replay failed for {corrupt}")
        try:
            comment = archive.comment.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise ManualSourceIntakeError("ZIP archive comment is not ASCII") from exc
        if comment != companion_contract["archive_comment"]:
            raise ManualSourceIntakeError("ZIP archive comment does not pin the commit")
        bound: list[dict[str, Any]] = []
        for binding in companion_contract["member_bindings"]:
            member = binding["member_path"]
            component_id = binding["component_id"]
            if member not in names or component_id not in component_payloads:
                raise ManualSourceIntakeError("ZIP companion member is unavailable")
            try:
                member_payload = archive.read(member)
            except (BadZipFile, RuntimeError, OSError, KeyError) as exc:
                raise ManualSourceIntakeError("ZIP companion member cannot be read") from exc
            companion_payload = component_payloads[component_id]
            if member_payload != companion_payload:
                raise ManualSourceIntakeError("ZIP companion member differs byte-for-byte")
            bound.append(
                {
                    "member_path": member,
                    "component_id": component_id,
                    "size_bytes": len(member_payload),
                    "sha256": hashlib.sha256(member_payload).hexdigest(),
                }
            )
    metadata = {
        "container_validation": "zip-crc-bounded-no-traversal-no-symlinks",
        "archive_comment": comment,
        "repository_commit": companion_contract["repository_commit"],
        "entry_count": len(infos),
        "total_uncompressed_bytes": total_uncompressed,
        "member_bindings": sorted(bound, key=lambda row: row["member_path"]),
    }
    return None, metadata, None


def _component_contract(
    component_id: str, contracts: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    for contract in contracts:
        if component_id in {
            contract["archive_component_id"],
            contract["data_component_id"],
            contract["license_component_id"],
        }:
            return dict(contract)
    return None


def _analyze_component(
    component: Mapping[str, Any],
    payload: bytes,
    *,
    component_payloads: Mapping[str, bytes],
    contracts: Sequence[Mapping[str, Any]],
) -> tuple[bytes | None, dict[str, Any], bytes | None]:
    component_id = str(component["component_id"])
    contract = _component_contract(component_id, contracts)
    raw_sha256 = hashlib.sha256(payload).hexdigest()
    format_name = component["format"]
    if format_name == "mhtml":
        return _analyze_mhtml(payload, component)
    if format_name == "pdf":
        return _analyze_pdf(payload, component)
    if format_name == "json":
        return _analyze_json(
            payload,
            component,
            raw_sha256=raw_sha256,
            companion_contract=contract,
        )
    if format_name == "zip":
        return _analyze_zip(
            payload,
            component,
            companion_contract=contract,
            component_payloads=component_payloads,
        )
    if format_name == "text":
        return _analyze_text(payload, component)
    raise ManualSourceIntakeError("component format is unsupported")


def _component_specs(mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_component: dict[str, list[dict[str, Any]]] = {}
    for row in mapping["attachments"]:
        by_component.setdefault(row["component_id"], []).append(dict(row))
    specs: list[dict[str, Any]] = []
    for component_id in sorted(by_component):
        rows = sorted(by_component[component_id], key=lambda row: row["alias_id"])
        first = rows[0]
        specs.append(
            {
                key: first[key]
                for key in (
                    "source_id",
                    "component_id",
                    "requested_url",
                    "final_url",
                    "snapshot_url",
                    "acquired_at",
                    "title",
                    "acquisition_mode",
                    "source_role",
                    "format",
                    "disposition",
                    "pagination",
                    "wikimedia_revision_id",
                )
            }
            | {"aliases": [row["alias_id"] for row in rows]}
        )
    return specs


def _raw_extension(format_name: str) -> str:
    return {"mhtml": ".mhtml", "pdf": ".pdf", "json": ".json", "zip": ".zip", "text": ".txt"}[
        format_name
    ]


def _safe_artifact_file(directory: Path, logical_path: Any) -> Path:
    if not isinstance(logical_path, str) or not logical_path:
        raise ManualSourceIntakeError("artifact file path is missing")
    path = PurePosixPath(logical_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != logical_path
    ):
        raise ManualSourceIntakeError("artifact file path is unsafe")
    target = directory.joinpath(*path.parts)
    if not target.is_file() or target.is_symlink():
        raise ManualSourceIntakeError("artifact file is unavailable or unsafe")
    return target


def _sha256sums(directory: Path, relative_paths: Sequence[str]) -> bytes:
    return "".join(
        f"{sha256_file(directory / relative)}  {relative}\n"
        for relative in sorted(relative_paths)
    ).encode("utf-8")


def _summary(
    components: Sequence[Mapping[str, Any]],
    aliases: Sequence[Mapping[str, Any]],
    assets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "alias_count": len(aliases),
        "component_count": len(components),
        "unique_raw_payload_count": len(assets),
        "deduplicated_alias_count": len(aliases) - len(components),
        "source_count": len({row["source_id"] for row in components}),
        "format_counts": dict(sorted(Counter(row["format"] for row in components).items())),
        "normalized_component_count": sum(
            row.get("normalized_text_file") is not None for row in components
        ),
        "candidate_projection_count": sum(
            row.get("candidate_projection") is not None for row in components
        ),
    }


def _dependency_binding() -> dict[str, Any]:
    lock = _object(
        _strict_json_file(DEPENDENCY_LOCK_PATH, label="source-intake dependency lock"),
        "source-intake dependency lock",
    )
    expected_lock_fields = {
        "schema_version",
        "scope",
        "runtime_python_abi",
        "requirements_lock_path",
        "requirements_lock_sha256",
        "pdf_backend",
    }
    if set(lock) != expected_lock_fields:
        raise ManualSourceIntakeError("source-intake dependency lock fields differ")
    if (
        lock.get("schema_version") != "wp3-g3-source-intake-dependency-lock/v1"
        or lock.get("scope") != "wp3-g3-public-source-intake-only"
        or lock.get("requirements_lock_path")
        != "environment/wp3-g3-source-intake-requirements.lock"
    ):
        raise ManualSourceIntakeError("source-intake dependency lock identity differs")
    backend = _object(lock.get("pdf_backend"), "PDF backend lock")
    if (
        backend.get("backend_id") != PDF_BACKEND_ID
        or backend.get("package_name") != "pypdf"
        or backend.get("package_version") != PYPDF_VERSION
        or backend.get("wheel_filename") != "pypdf-6.0.0-py3-none-any.whl"
        or backend.get("wheel_sha256") != PYPDF_WHEEL_SHA256
        or backend.get("encrypted_pdf_allowed") is not False
        or backend.get("active_content_allowed") is not False
        or backend.get("network_fallback_allowed") is not False
        or backend.get("ocr_fallback_allowed") is not False
    ):
        raise ManualSourceIntakeError("fixed PDF backend lock differs")
    requirements_sha256 = sha256_file(REQUIREMENTS_LOCK_PATH)
    if requirements_sha256 != lock.get("requirements_lock_sha256"):
        raise ManualSourceIntakeError("source-intake requirements lock SHA differs")
    for schema_path, expected_id in (
        (
            SOURCE_MAP_SCHEMA_PATH,
            "https://local.invalid/schemas/wp3_g3_manual_source_map_v1.schema.json",
        ),
        (
            INTAKE_SCHEMA_PATH,
            "https://local.invalid/schemas/wp3_g3_manual_source_intake_v1.schema.json",
        ),
    ):
        schema = _object(_strict_json_file(schema_path, label="intake schema"), "intake schema")
        if (
            schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
            or schema.get("$id") != expected_id
        ):
            raise ManualSourceIntakeError("manual-source schema identity differs")
    implementation_sha256 = sha256_file(Path(__file__))
    return {
        "dependency_lock_path": "config/stage1/wp3_g3_source_intake_dependency_lock_v1.json",
        "dependency_lock_sha256": sha256_file(DEPENDENCY_LOCK_PATH),
        "requirements_lock_path": "environment/wp3-g3-source-intake-requirements.lock",
        "requirements_lock_sha256": requirements_sha256,
        "source_map_schema_path": "schemas/wp3_g3_manual_source_map_v1.schema.json",
        "source_map_schema_sha256": sha256_file(SOURCE_MAP_SCHEMA_PATH),
        "intake_schema_path": "schemas/wp3_g3_manual_source_intake_v1.schema.json",
        "intake_schema_sha256": sha256_file(INTAKE_SCHEMA_PATH),
        "builder_implementation_path": "src/build_lex/terminology_g3_manual_source_intake.py",
        "builder_implementation_sha256": implementation_sha256,
        "parser_implementation_path": "src/build_lex/terminology_g3_manual_source_intake.py",
        "parser_implementation_sha256": implementation_sha256,
        "pdf_backend_id": PDF_BACKEND_ID,
        "pypdf_wheel_sha256": PYPDF_WHEEL_SHA256,
    }


def build_manual_source_intake(
    *,
    attachment_root: str | Path,
    source_map_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build and atomically publish one immutable offline intake artifact."""

    root = Path(attachment_root)
    if not root.is_absolute():
        raise ManualSourceIntakeError("attachment_root must be an explicit absolute path")
    mapping = load_manual_source_map(source_map_path)
    destination_root = Path(output_root).resolve()
    staging = new_staging_directory(destination_root, "wp3manual-pending")
    os.chmod(staging, 0o700)
    try:
        aliases: list[dict[str, Any]] = []
        component_payloads: dict[str, bytes] = {}
        total_bytes = 0
        for row in mapping["attachments"]:
            payload = _read_attachment(root, row)
            total_bytes += len(payload)
            if total_bytes > MAX_TOTAL_ATTACHMENT_BYTES:
                raise ManualSourceIntakeError("total attachment budget is exceeded")
            existing = component_payloads.setdefault(row["component_id"], payload)
            if existing != payload:
                raise ManualSourceIntakeError("aliases of one component differ byte-for-byte")
            aliases.append(
                {
                    "alias_id": row["alias_id"],
                    "source_id": row["source_id"],
                    "component_id": row["component_id"],
                    "attachment_locator": row["attachment_locator"],
                    "expected_size_bytes": row["expected_size_bytes"],
                    "expected_sha256": row["expected_sha256"],
                }
            )
        components: list[dict[str, Any]] = []
        raw_by_sha: dict[str, dict[str, Any]] = {}
        for spec in _component_specs(mapping):
            payload = component_payloads[spec["component_id"]]
            raw_sha256 = hashlib.sha256(payload).hexdigest()
            asset = raw_by_sha.get(raw_sha256)
            if asset is None:
                raw_file = f"raw/sha256/{raw_sha256}{_raw_extension(spec['format'])}"
                write_bytes_atomic(staging / raw_file, payload)
                os.chmod(staging / raw_file, 0o600)
                asset = {
                    "raw_sha256": raw_sha256,
                    "size_bytes": len(payload),
                    "file": raw_file,
                    "formats": [spec["format"]],
                    "component_ids": [],
                    "alias_ids": [],
                }
                raw_by_sha[raw_sha256] = asset
            elif len(payload) != asset["size_bytes"]:
                raise ManualSourceIntakeError("raw SHA collision has a different size")
            if spec["format"] not in asset["formats"]:
                asset["formats"].append(spec["format"])
            asset["component_ids"].append(spec["component_id"])
            asset["alias_ids"].extend(spec["aliases"])
            normalized, format_metadata, candidate_projection = _analyze_component(
                spec,
                payload,
                component_payloads=component_payloads,
                contracts=mapping["companion_contracts"],
            )
            component = dict(spec)
            component.update(
                {
                    "raw_file": asset["file"],
                    "raw_size_bytes": len(payload),
                    "raw_sha256": raw_sha256,
                    "normalized_text_file": None,
                    "normalized_text_size_bytes": None,
                    "normalized_text_sha256": None,
                    "candidate_projection": None,
                    "format_metadata": format_metadata,
                }
            )
            if normalized is not None:
                normalized_file = f"normalized/{spec['component_id']}.txt"
                write_bytes_atomic(staging / normalized_file, normalized)
                os.chmod(staging / normalized_file, 0o600)
                component.update(
                    {
                        "normalized_text_file": normalized_file,
                        "normalized_text_size_bytes": len(normalized),
                        "normalized_text_sha256": hashlib.sha256(normalized).hexdigest(),
                    }
                )
            if candidate_projection is not None:
                projection_file = f"derived/{spec['component_id']}-form-candidates.json"
                write_bytes_atomic(staging / projection_file, candidate_projection)
                os.chmod(staging / projection_file, 0o600)
                projection_value = _strict_json_bytes(
                    candidate_projection, label="candidate projection"
                )
                component["candidate_projection"] = {
                    "schema_version": CHIME_PROJECTION_SCHEMA_VERSION,
                    "file": projection_file,
                    "size_bytes": len(candidate_projection),
                    "sha256": hashlib.sha256(candidate_projection).hexdigest(),
                    "count": projection_value["candidate_count"],
                    "allowed_types": list(FORM_CANDIDATE_TYPES),
                }
            components.append(component)
        assets = []
        for raw_sha256 in sorted(raw_by_sha):
            asset = raw_by_sha[raw_sha256]
            asset["formats"] = sorted(asset["formats"])
            asset["component_ids"] = sorted(asset["component_ids"])
            asset["alias_ids"] = sorted(asset["alias_ids"])
            assets.append(asset)
        aliases.sort(key=lambda row: row["alias_id"])
        components.sort(key=lambda row: row["component_id"])
        source_map_file = "inputs/source_map.json"
        write_canonical_json(staging / source_map_file, mapping)
        os.chmod(staging / source_map_file, 0o600)
        source_map_sha256 = sha256_file(staging / source_map_file)
        identity = {
            "schema_version": INTAKE_SCHEMA_VERSION,
            "artifact_kind": INTAKE_ARTIFACT_KIND,
            "scope": "development-only-form-only-label-free-non-lexicon",
            "network_access_performed": False,
            "policy_id": INTAKE_POLICY_ID,
            "dependency_binding": _dependency_binding(),
            "map_id": mapping["map_id"],
            "source_map_sha256": source_map_sha256,
            "source_map_canonical_sha256": canonical_sha256(mapping),
            "assets": assets,
            "components": components,
            "aliases": aliases,
            "companion_contracts": mapping["companion_contracts"],
        }
        intake_id = INTAKE_ID_PREFIX + canonical_sha256(identity)
        manifest = dict(identity)
        manifest["intake_id"] = intake_id
        manifest["source_map_file"] = source_map_file
        manifest["summary"] = _summary(components, aliases, assets)
        manifest["promotion"] = {
            "g3_catalog_applied": False,
            "g3_source_bundle_created": False,
            "g3_form_reference_created": False,
            "human_relation_review_required": True,
        }
        write_canonical_json(staging / "manifest.json", manifest)
        os.chmod(staging / "manifest.json", 0o600)
        referenced = [
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file() and path.name not in {"SHA256SUMS", "payload_manifest.json"}
        ]
        write_bytes_atomic(staging / "SHA256SUMS", _sha256sums(staging, referenced))
        os.chmod(staging / "SHA256SUMS", 0o600)
        target = destination_root / intake_id
        payload_manifest_sha256 = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda directory: validate_manual_source_intake(
                directory, require_directory_id=False
            ),
        )
        report = validate_manual_source_intake(target)
        if report["payload_manifest_sha256"] != payload_manifest_sha256:
            raise ManualSourceIntakeError("published payload manifest hash differs")
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def validate_manual_source_intake(
    intake_directory: str | Path,
    *,
    require_directory_id: bool = True,
) -> dict[str, Any]:
    """Independently replay a published manual-source intake artifact."""

    directory = Path(intake_directory).resolve()
    try:
        payload_manifest_sha256 = validate_payload_manifest(directory)
    except TrainingArtifactError as exc:
        raise ManualSourceIntakeError(str(exc)) from exc
    manifest = _object(
        _strict_json_file(directory / "manifest.json", label="intake manifest"),
        "intake manifest",
    )
    expected_top = {
        "schema_version",
        "artifact_kind",
        "scope",
        "network_access_performed",
        "policy_id",
        "dependency_binding",
        "map_id",
        "source_map_sha256",
        "source_map_canonical_sha256",
        "assets",
        "components",
        "aliases",
        "companion_contracts",
        "intake_id",
        "source_map_file",
        "summary",
        "promotion",
    }
    if set(manifest) != expected_top:
        raise ManualSourceIntakeError("intake manifest fields differ")
    if (
        manifest.get("schema_version") != INTAKE_SCHEMA_VERSION
        or manifest.get("artifact_kind") != INTAKE_ARTIFACT_KIND
        or manifest.get("scope") != "development-only-form-only-label-free-non-lexicon"
        or manifest.get("network_access_performed") is not False
        or manifest.get("policy_id") != INTAKE_POLICY_ID
    ):
        raise ManualSourceIntakeError("intake manifest identity differs")
    identity_fields = (
        "schema_version",
        "artifact_kind",
        "scope",
        "network_access_performed",
        "policy_id",
        "dependency_binding",
        "map_id",
        "source_map_sha256",
        "source_map_canonical_sha256",
        "assets",
        "components",
        "aliases",
        "companion_contracts",
    )
    identity = {field: manifest[field] for field in identity_fields}
    intake_id = manifest.get("intake_id")
    if intake_id != INTAKE_ID_PREFIX + canonical_sha256(identity):
        raise ManualSourceIntakeError("intake ID does not bind its payload identity")
    if require_directory_id and directory.name != intake_id:
        raise ManualSourceIntakeError("intake directory name differs from intake ID")
    if manifest.get("dependency_binding") != _dependency_binding():
        raise ManualSourceIntakeError("source-intake dependency binding drifted")
    source_map_file = _safe_artifact_file(directory, manifest.get("source_map_file"))
    if sha256_file(source_map_file) != manifest.get("source_map_sha256"):
        raise ManualSourceIntakeError("source map snapshot SHA differs")
    mapping = validate_manual_source_map_value(
        _strict_json_file(source_map_file, label="source map snapshot")
    )
    if (
        canonical_sha256(mapping) != manifest.get("source_map_canonical_sha256")
        or mapping["map_id"] != manifest.get("map_id")
        or mapping["companion_contracts"] != manifest.get("companion_contracts")
    ):
        raise ManualSourceIntakeError("source map snapshot binding differs")
    raw_assets = manifest.get("assets")
    components = manifest.get("components")
    aliases = manifest.get("aliases")
    if (
        not isinstance(raw_assets, list)
        or not isinstance(components, list)
        or not isinstance(aliases, list)
    ):
        raise ManualSourceIntakeError("intake arrays are malformed")
    expected_specs = {row["component_id"]: row for row in _component_specs(mapping)}
    component_rows: dict[str, dict[str, Any]] = {}
    component_payloads: dict[str, bytes] = {}
    referenced = {
        "manifest.json",
        "SHA256SUMS",
        "payload_manifest.json",
        str(manifest["source_map_file"]),
    }
    expected_component_fields = set(next(iter(expected_specs.values()))) | {
        "raw_file",
        "raw_size_bytes",
        "raw_sha256",
        "normalized_text_file",
        "normalized_text_size_bytes",
        "normalized_text_sha256",
        "candidate_projection",
        "format_metadata",
    }
    for raw_component in components:
        component = _object(raw_component, "intake component")
        if set(component) != expected_component_fields:
            raise ManualSourceIntakeError("intake component fields differ")
        component_id = component.get("component_id")
        if component_id in component_rows or component_id not in expected_specs:
            raise ManualSourceIntakeError("intake component ID is invalid or duplicated")
        if any(
            component.get(field) != value
            for field, value in expected_specs[component_id].items()
        ):
            raise ManualSourceIntakeError("intake component differs from source map")
        raw_file = _safe_artifact_file(directory, component.get("raw_file"))
        referenced.add(str(component["raw_file"]))
        payload = raw_file.read_bytes()
        if (
            len(payload) != component.get("raw_size_bytes")
            or hashlib.sha256(payload).hexdigest() != component.get("raw_sha256")
        ):
            raise ManualSourceIntakeError("raw component replay failed")
        component_rows[component_id] = component
        component_payloads[component_id] = payload
    if set(component_rows) != set(expected_specs):
        raise ManualSourceIntakeError("intake component coverage differs")
    expected_aliases = [
        {
            "alias_id": row["alias_id"],
            "source_id": row["source_id"],
            "component_id": row["component_id"],
            "attachment_locator": row["attachment_locator"],
            "expected_size_bytes": row["expected_size_bytes"],
            "expected_sha256": row["expected_sha256"],
        }
        for row in mapping["attachments"]
    ]
    expected_aliases.sort(key=lambda row: row["alias_id"])
    if aliases != expected_aliases:
        raise ManualSourceIntakeError("intake aliases differ from source map")
    expected_assets: dict[str, dict[str, Any]] = {}
    for component_id, component in component_rows.items():
        sha = component["raw_sha256"]
        asset = expected_assets.setdefault(
            sha,
            {
                "raw_sha256": sha,
                "size_bytes": component["raw_size_bytes"],
                "file": component["raw_file"],
                "formats": [],
                "component_ids": [],
                "alias_ids": [],
            },
        )
        if (
            asset["file"] != component["raw_file"]
            or asset["size_bytes"] != component["raw_size_bytes"]
        ):
            raise ManualSourceIntakeError("raw deduplication differs")
        asset["formats"].append(component["format"])
        asset["component_ids"].append(component_id)
        asset["alias_ids"].extend(component["aliases"])
    expected_asset_rows = []
    for sha in sorted(expected_assets):
        asset = expected_assets[sha]
        asset["formats"] = sorted(set(asset["formats"]))
        asset["component_ids"] = sorted(asset["component_ids"])
        asset["alias_ids"] = sorted(asset["alias_ids"])
        expected_asset_rows.append(asset)
    if raw_assets != expected_asset_rows:
        raise ManualSourceIntakeError("raw asset index does not replay")
    for component_id in sorted(component_rows):
        component = component_rows[component_id]
        normalized, metadata, candidate_projection = _analyze_component(
            expected_specs[component_id],
            component_payloads[component_id],
            component_payloads=component_payloads,
            contracts=mapping["companion_contracts"],
        )
        if metadata != component.get("format_metadata"):
            raise ManualSourceIntakeError("component format metadata does not replay")
        normalized_file = component.get("normalized_text_file")
        if normalized is None:
            if any(
                component.get(field) is not None
                for field in (
                    "normalized_text_file",
                    "normalized_text_size_bytes",
                    "normalized_text_sha256",
                )
            ):
                raise ManualSourceIntakeError("component has an unexpected normalized projection")
        else:
            normalized_path = _safe_artifact_file(directory, normalized_file)
            referenced.add(str(normalized_file))
            if (
                normalized_path.read_bytes() != normalized
                or len(normalized) != component.get("normalized_text_size_bytes")
                or hashlib.sha256(normalized).hexdigest() != component.get("normalized_text_sha256")
            ):
                raise ManualSourceIntakeError("normalized text projection does not replay")
        candidate = component.get("candidate_projection")
        if candidate_projection is None:
            if candidate is not None:
                raise ManualSourceIntakeError("component has an unexpected candidate projection")
        else:
            candidate_row = _object(candidate, "candidate projection reference")
            expected_candidate_fields = {
                "schema_version",
                "file",
                "size_bytes",
                "sha256",
                "count",
                "allowed_types",
            }
            if set(candidate_row) != expected_candidate_fields:
                raise ManualSourceIntakeError("candidate projection reference fields differ")
            candidate_path = _safe_artifact_file(directory, candidate_row.get("file"))
            referenced.add(str(candidate_row["file"]))
            value = _strict_json_bytes(candidate_projection, label="candidate projection replay")
            if (
                candidate_path.read_bytes() != candidate_projection
                or candidate_row.get("schema_version") != CHIME_PROJECTION_SCHEMA_VERSION
                or candidate_row.get("size_bytes") != len(candidate_projection)
                or candidate_row.get("sha256") != hashlib.sha256(candidate_projection).hexdigest()
                or candidate_row.get("count") != value["candidate_count"]
                or candidate_row.get("allowed_types") != list(FORM_CANDIDATE_TYPES)
            ):
                raise ManualSourceIntakeError("candidate projection does not replay")
    if manifest.get("summary") != _summary(components, aliases, raw_assets):
        raise ManualSourceIntakeError("intake summary does not replay")
    promotion = _object(manifest.get("promotion"), "promotion")
    if promotion != {
        "g3_catalog_applied": False,
        "g3_source_bundle_created": False,
        "g3_form_reference_created": False,
        "human_relation_review_required": True,
    }:
        raise ManualSourceIntakeError("intake was incorrectly promoted")
    payload_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if payload_files != referenced:
        raise ManualSourceIntakeError("intake contains unreferenced or missing payload files")
    sums_lines = (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    sum_paths = sorted(referenced - {"SHA256SUMS", "payload_manifest.json"})
    expected_sums = [
        f"{sha256_file(directory / relative)}  {relative}" for relative in sum_paths
    ]
    if sums_lines != expected_sums:
        raise ManualSourceIntakeError("SHA256SUMS does not replay")
    ordered_components = [component_rows[key] for key in sorted(component_rows)]
    return {
        "intake_id": intake_id,
        "target": str(directory),
        "payload_manifest_sha256": payload_manifest_sha256,
        "manifest": manifest,
        "components": ordered_components,
        "aliases": aliases,
    }


def load_normalized_components(intake_directory: str | Path) -> list[dict[str, Any]]:
    """Validate an intake and return its stable component contract."""

    return validate_manual_source_intake(intake_directory)["components"]


__all__ = [
    "CHIME_PROJECTION_SCHEMA_VERSION",
    "INTAKE_ARTIFACT_KIND",
    "INTAKE_SCHEMA_VERSION",
    "ManualSourceIntakeError",
    "SOURCE_MAP_SCHEMA_VERSION",
    "build_manual_source_intake",
    "load_manual_source_map",
    "load_normalized_components",
    "normalize_publisher_pdf_bytes",
    "validate_manual_source_intake",
    "validate_manual_source_map_value",
]
