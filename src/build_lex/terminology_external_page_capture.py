"""Auditable direct-network capture for proposed WP3 public pages.

This module is deliberately separate from the frozen G3 source lifecycle.  It
captures every URL explicitly registered in a Markdown source register and
preserves successful and failed acquisition evidence.  Library callers can
add an explicitly authorized URL, but the WP3 CLI defaults to the source
register alone.  A capture does not promote a page into the G3 catalog, create
a source sync receipt, or make pricing executable.

The network client ignores environment proxy configuration, performs no
automatic retry, follows only bounded public HTTP(S) redirects, and stores the
response entity bytes together with a replayable manifest.
"""

from __future__ import annotations

import codecs
import hashlib
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

from data.training_artifacts import (
    TrainingArtifactError,
    canonical_sha256,
    finalize_target_atomic,
    load_json,
    new_staging_directory,
    sha256_file,
    validate_payload_manifest,
    write_bytes_atomic,
    write_canonical_json,
)


CAPTURE_SCHEMA_VERSION = "wp3-external-page-capture/v1"
CAPTURE_ARTIFACT_KIND = "wp3-external-page-capture"
CAPTURE_ID_PREFIX = "wp3capture-"
CAPTURE_POLICY_ID = "explicit-register-zero-proxy-single-attempt/v1"
BROWSER_RENDER_SCHEMA_VERSION = "wp3-external-page-browser-render/v1"
BROWSER_RENDER_ARTIFACT_KIND = "wp3-external-page-browser-render"
BROWSER_RENDER_ID_PREFIX = "wp3render-"
BROWSER_RENDER_POLICY_ID = "chromium-zero-proxy-dump-dom-single-navigation/v1"
REGISTER_URL_RE = re.compile(r"<(https?://[^>\s]+)>")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([^;\s\"']+)", re.IGNORECASE)
WIKIMEDIA_REVISION_PATTERNS = (
    re.compile(r'"wgRevisionId"\s*:\s*([1-9][0-9]*)'),
    re.compile(r'"revisionId"\s*:\s*([1-9][0-9]*)'),
)

PROXY_ENVIRONMENT_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECTS = 10
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_CAPTURE_BYTES = 256 * 1024 * 1024
CONNECT_TIMEOUT_SECONDS = 15
READ_TIMEOUT_SECONDS = 90
DEFAULT_WORKERS = 4
BROWSER_TIMEOUT_SECONDS = 60
BROWSER_VIRTUAL_TIME_BUDGET_MS = 20000
MAX_RENDERED_DOM_BYTES = 32 * 1024 * 1024

DOWNLOAD_COMPLETE_STATUSES = frozenset({"downloaded_usable"})
FETCH_STATUSES = frozenset(
    {
        "downloaded_usable",
        "downloaded_needs_review",
        "downloaded_needs_browser_archive",
        "downloaded_error_or_challenge",
        "http_error",
        "network_error",
        "unsafe_target",
        "redirect_error",
        "body_too_large",
        "invalid_body",
    }
)

CHALLENGE_MARKERS = (
    "access denied",
    "request blocked",
    "forbidden",
    "verify you are human",
    "captcha",
    "just a moment",
    "cloudflare ray id",
    "proxy error",
    "bad gateway",
    "robot check",
    "访问被拒绝",
    "访问异常",
    "安全验证",
    "请输入验证码",
    "请求被拒绝",
    "页面不存在",
)
JAVASCRIPT_SHELL_MARKERS = (
    "enable javascript",
    "please enable javascript",
    "javascript is required",
    "请启用 javascript",
    "请开启 javascript",
)


class ExternalPageCaptureError(RuntimeError):
    """Raised when a proposed-source capture cannot be built or replayed."""


class _VisibleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._title_depth = 0
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.script_count = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized == "script":
            self.script_count += 1
        if normalized in {"script", "style", "noscript", "template"}:
            self._ignored_depth += 1
        if normalized == "title":
            self._title_depth += 1

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        if normalized == "title":
            self._title_depth = max(0, self._title_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._title_depth and data.strip():
            self.title_parts.append(data)
        if not self._ignored_depth and data.strip():
            self.parts.append(data)


class _CaptureBudget:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.used = 0
        self._lock = threading.Lock()

    def consume(self, amount: int) -> None:
        with self._lock:
            if amount < 0 or self.used + amount > self.maximum:
                raise ExternalPageCaptureError("capture body budget would be exceeded")
            self.used += amount


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalPageCaptureError(f"{label} must be an object")
    return dict(value)


def _relative_workspace_path(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError as exc:
        raise ExternalPageCaptureError("capture input must remain inside the workspace") from exc


def _heading_path(headings: dict[int, str]) -> list[str]:
    return [headings[level] for level in sorted(headings)]


def extract_registered_urls(register_path: str | Path) -> list[dict[str, Any]]:
    """Extract and de-duplicate only angle-bracketed HTTP(S) URLs.

    Heading paths and line numbers are retained so a URL repeated under both a
    source section and a pending-download priority remains one network target
    with multiple register occurrences.
    """

    path = Path(register_path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ExternalPageCaptureError(f"cannot read source register: {path}") from exc
    headings: dict[int, str] = {}
    by_url: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            level = len(heading.group(1))
            headings = {
                current_level: value
                for current_level, value in headings.items()
                if current_level < level
            }
            headings[level] = heading.group(2)
        for match in REGISTER_URL_RE.finditer(line):
            url = match.group(1)
            row = by_url.setdefault(
                url,
                {
                    "requested_url": url,
                    "roles": ["g3_source_candidate"],
                    "register_occurrences": [],
                },
            )
            occurrence = {
                "line": line_number,
                "heading_path": _heading_path(headings),
            }
            if occurrence not in row["register_occurrences"]:
                row["register_occurrences"].append(occurrence)
    if not by_url:
        raise ExternalPageCaptureError("source register contains no angle-bracketed URLs")
    return [by_url[url] for url in sorted(by_url)]


def _validate_url_shape(url: str) -> tuple[str, str, int]:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ExternalPageCaptureError("URL is malformed") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ExternalPageCaptureError("URL must be public HTTP(S) without userinfo or fragment")
    return parsed.scheme, parsed.hostname, port or (443 if parsed.scheme == "https" else 80)


def _resolve_public_addresses(url: str) -> list[str]:
    _scheme, hostname, port = _validate_url_shape(url)
    try:
        answers = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ExternalPageCaptureError("public hostname resolution failed") from exc
    addresses = sorted({str(answer[4][0]) for answer in answers})
    if not addresses:
        raise ExternalPageCaptureError("public hostname produced no addresses")
    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ExternalPageCaptureError("resolver returned a malformed address") from exc
        if not parsed.is_global:
            raise ExternalPageCaptureError(
                "public page resolved to a non-global address and was not requested"
            )
    return addresses


def _safe_header(response: Any, name: str) -> str | None:
    value = response.headers.get(name)
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered[:2000] if rendered else None


def _response_headers(response: Any) -> dict[str, str]:
    fields = (
        "Content-Type",
        "Content-Length",
        "Content-Encoding",
        "ETag",
        "Last-Modified",
        "Cache-Control",
    )
    return {
        name.casefold().replace("-", "_"): value
        for name in fields
        if (value := _safe_header(response, name)) is not None
    }


def _read_bounded_body(response: Any) -> bytes:
    declared = response.headers.get("Content-Length")
    if declared is not None:
        try:
            declared_size = int(declared)
        except (TypeError, ValueError) as exc:
            raise ExternalPageCaptureError("response Content-Length is invalid") from exc
        if declared_size < 0 or declared_size > MAX_RESPONSE_BYTES:
            raise ExternalPageCaptureError("response body exceeds the per-page limit")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not isinstance(chunk, bytes):
            raise ExternalPageCaptureError("response body chunk is not bytes")
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise ExternalPageCaptureError("response body exceeds the per-page limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_text(body: bytes, content_type: str) -> tuple[str | None, str | None]:
    candidates: list[str] = []
    match = CHARSET_RE.search(content_type)
    if match:
        candidates.append(match.group(1))
    if body.startswith(codecs.BOM_UTF8):
        candidates.append("utf-8-sig")
    candidates.extend(("utf-8", "gb18030"))
    tried: set[str] = set()
    for candidate in candidates:
        normalized = candidate.casefold()
        if normalized in tried:
            continue
        tried.add(normalized)
        try:
            codecs.lookup(candidate)
            return body.decode(candidate, errors="strict"), candidate
        except (LookupError, UnicodeDecodeError):
            continue
    return None, None


def _content_type(response: Any) -> str:
    return str(response.headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()


def _media_kind(body: bytes, declared_type: str, final_url: str) -> str:
    leading = body[:1024].lstrip()
    lowered = leading.lower()
    path = unquote(urlsplit(final_url).path).casefold()
    if leading.startswith(b"%PDF-"):
        return "pdf"
    if declared_type in {"application/json", "application/ld+json"} or path.endswith(".json"):
        return "json"
    if path.endswith(".pdf") or "pdf" in declared_type:
        return "pdf_claimed"
    if (
        declared_type in {"text/html", "application/xhtml+xml"}
        or lowered.startswith(b"<!doctype html")
        or lowered.startswith(b"<html")
    ):
        return "html"
    if declared_type.startswith("text/"):
        return "text"
    return "binary"


def _normalize_visible_text(parts: Sequence[str]) -> str:
    return " ".join(" ".join(parts).split())


def _classify_body(
    body: bytes,
    *,
    declared_type: str,
    full_content_type: str,
    final_url: str,
) -> tuple[dict[str, Any], bytes | None]:
    media_kind = _media_kind(body, declared_type, final_url)
    result: dict[str, Any] = {
        "media_kind": media_kind,
        "declared_media_type": declared_type or None,
        "content_validation": "unvalidated_binary",
        "warnings": [],
    }
    projection: bytes | None = None
    if media_kind == "pdf":
        eof_present = b"%%EOF" in body[-8192:]
        result["content_validation"] = "valid_pdf_container" if eof_present else "truncated_pdf"
        if not eof_present:
            result["warnings"].append("pdf_eof_marker_missing")
        if declared_type not in {"application/pdf", "application/octet-stream", ""}:
            result["warnings"].append("media_type_mismatch")
        return result, None
    if media_kind == "pdf_claimed":
        result["content_validation"] = "invalid_pdf_container"
        result["warnings"].append("pdf_magic_missing")
        return result, None
    if media_kind == "json":
        text, encoding = _decode_text(body, full_content_type)
        result["text_encoding"] = encoding
        if text is None:
            result["content_validation"] = "invalid_json_encoding"
            return result, None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            result["content_validation"] = "invalid_json"
            return result, None
        result["content_validation"] = "valid_json"
        if isinstance(parsed, (list, dict)):
            result["json_item_count"] = len(parsed)
        return result, None
    if media_kind in {"html", "text"}:
        text, encoding = _decode_text(body, full_content_type)
        result["text_encoding"] = encoding
        if text is None or "\x00" in text:
            result["content_validation"] = "undecodable_text"
            return result, None
        if media_kind == "html":
            parser = _VisibleHTMLParser()
            try:
                parser.feed(text)
                parser.close()
            except Exception:
                result["content_validation"] = "invalid_html"
                return result, None
            visible = _normalize_visible_text(parser.parts)
            title = _normalize_visible_text(parser.title_parts)
            result["title"] = title[:1000] or None
            result["script_count"] = parser.script_count
        else:
            visible = _normalize_visible_text([text])
            title = ""
            result["title"] = None
            result["script_count"] = 0
        projection = (visible + "\n").encode("utf-8") if visible else None
        result["visible_text_characters"] = len(visible)
        lowered = f"{title}\n{visible[:10000]}".casefold()
        raw_lowered = text[:20000].casefold()
        if any(marker in lowered for marker in CHALLENGE_MARKERS) or any(
            marker in raw_lowered
            for marker in ("acw_sc__v2", "document.cookie", "waf challenge")
        ):
            result["content_validation"] = "error_or_challenge"
        elif any(marker in lowered for marker in JAVASCRIPT_SHELL_MARKERS) or (
            media_kind == "html" and len(visible) < 200 and parser.script_count >= 2
        ):
            result["content_validation"] = "likely_javascript_shell"
        elif len(visible) < 80:
            result["content_validation"] = "very_short_text"
            result["warnings"].append("visible_text_too_short")
        else:
            result["content_validation"] = "usable_static_text"
        return result, projection
    return result, None


def _file_basename(url: str, media_kind: str) -> str:
    parsed = urlsplit(url)
    candidate = Path(unquote(parsed.path)).name or "index"
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-.") or "page"
    stem = Path(candidate).stem[:48] or "page"
    extension = {
        "pdf": ".pdf",
        "pdf_claimed": ".bin",
        "json": ".json",
        "html": ".html",
        "text": ".txt",
        "binary": ".bin",
    }[media_kind]
    host = re.sub(r"[^A-Za-z0-9.-]+", "-", parsed.hostname or "unknown")[:64]
    identity = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{host}--{stem}--{identity}{extension}"


def _status_for_response(status_code: int, analysis: Mapping[str, Any], body: bytes) -> str:
    if status_code != 200:
        return "http_error"
    if not body:
        return "invalid_body"
    validation = analysis.get("content_validation")
    if validation in {"valid_pdf_container", "valid_json", "usable_static_text"}:
        return "downloaded_usable"
    if validation == "likely_javascript_shell":
        return "downloaded_needs_browser_archive"
    if validation == "error_or_challenge":
        return "downloaded_error_or_challenge"
    if validation in {"very_short_text", "unvalidated_binary"}:
        return "downloaded_needs_review"
    return "invalid_body"


def _wikimedia_revision(body: bytes, encoding: str | None) -> str | None:
    if encoding is None:
        return None
    try:
        text = body.decode(encoding, errors="strict")
    except (LookupError, UnicodeDecodeError):
        return None
    for pattern in WIKIMEDIA_REVISION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def _fetch_one(
    row: Mapping[str, Any],
    *,
    staging: Path,
    budget: _CaptureBudget,
) -> dict[str, Any]:
    url = str(row["requested_url"])
    result = {
        "requested_url": url,
        "roles": list(row["roles"]),
        "register_occurrences": list(row["register_occurrences"]),
        "attempt_count": 1,
        "captured_at": _utc_now(),
        "redirect_chain": [],
        "final_url": None,
        "transport_downgrade": False,
        "cross_host_redirect": False,
        "status_code": None,
        "response_headers": {},
        "response_file": None,
        "response_size_bytes": 0,
        "response_sha256": None,
        "text_projection_file": None,
        "text_projection_sha256": None,
        "wikimedia_revision_id": None,
        "fetch_status": "network_error",
        "error": None,
    }
    original = urlsplit(url)
    current = url
    visited: set[str] = set()
    response = None
    session = None
    try:
        import requests

        session = requests.Session()
        session.trust_env = False
        for redirect_index in range(MAX_REDIRECTS + 1):
            if current in visited:
                result["fetch_status"] = "redirect_error"
                result["error"] = "redirect loop detected"
                return result
            visited.add(current)
            try:
                resolved = _resolve_public_addresses(current)
            except ExternalPageCaptureError as exc:
                result["fetch_status"] = "unsafe_target"
                result["error"] = str(exc)
                return result
            response = session.get(
                current,
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/pdf,application/json,text/plain;q=0.9,*/*;q=0.5",
                    "Accept-Encoding": "identity",
                    "User-Agent": "WP3-public-evidence-capture/1.0",
                },
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                allow_redirects=False,
                stream=True,
            )
            location = _safe_header(response, "Location")
            result["redirect_chain"].append(
                {
                    "request_url": current,
                    "response_url": str(response.url),
                    "status_code": int(response.status_code),
                    "location": location,
                    "resolved_addresses": resolved,
                }
            )
            if response.status_code not in REDIRECT_CODES:
                break
            response.close()
            response = None
            if redirect_index == MAX_REDIRECTS or not location:
                result["fetch_status"] = "redirect_error"
                result["error"] = "redirect limit exceeded or Location missing"
                return result
            redirected = urljoin(current, location)
            try:
                redirected_scheme, redirected_host, _port = _validate_url_shape(redirected)
            except ExternalPageCaptureError as exc:
                result["fetch_status"] = "redirect_error"
                result["error"] = str(exc)
                return result
            if original.scheme == "https" and redirected_scheme == "http":
                result["transport_downgrade"] = True
            if redirected_host.casefold() != (original.hostname or "").casefold():
                result["cross_host_redirect"] = True
            current = redirected
        if response is None:
            result["fetch_status"] = "network_error"
            result["error"] = "request produced no terminal response"
            return result
        result["final_url"] = str(response.url)
        result["status_code"] = int(response.status_code)
        result["response_headers"] = _response_headers(response)
        try:
            body = _read_bounded_body(response)
        except ExternalPageCaptureError as exc:
            result["fetch_status"] = "body_too_large"
            result["error"] = str(exc)
            return result
        budget.consume(len(body))
        full_content_type = str(response.headers.get("Content-Type", ""))
        analysis, projection = _classify_body(
            body,
            declared_type=_content_type(response),
            full_content_type=full_content_type,
            final_url=str(response.url),
        )
        result["content_analysis"] = analysis
        result["fetch_status"] = _status_for_response(
            int(response.status_code), analysis, body
        )
        basename = _file_basename(url, str(analysis["media_kind"]))
        response_path = staging / "responses" / basename
        write_bytes_atomic(response_path, body)
        os.chmod(response_path, 0o600)
        result["response_file"] = response_path.relative_to(staging).as_posix()
        result["response_size_bytes"] = len(body)
        result["response_sha256"] = hashlib.sha256(body).hexdigest()
        if projection is not None:
            projection_path = staging / "text" / f"{Path(basename).stem}.txt"
            write_bytes_atomic(projection_path, projection)
            os.chmod(projection_path, 0o600)
            result["text_projection_file"] = projection_path.relative_to(staging).as_posix()
            result["text_projection_sha256"] = hashlib.sha256(projection).hexdigest()
        if (urlsplit(url).hostname or "").casefold().endswith("wikipedia.org"):
            result["wikimedia_revision_id"] = _wikimedia_revision(
                body, analysis.get("text_encoding")
            )
        return result
    except ExternalPageCaptureError as exc:
        if "budget" in str(exc):
            result["fetch_status"] = "body_too_large"
        else:
            result["fetch_status"] = "network_error"
        result["error"] = str(exc)[:2000]
        return result
    except Exception as exc:
        result["fetch_status"] = "network_error"
        result["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        return result
    finally:
        if response is not None:
            response.close()
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def _sha256sums(directory: Path, relative_paths: Sequence[str]) -> bytes:
    lines: list[str] = []
    for relative in sorted(relative_paths):
        target = directory / relative
        lines.append(f"{sha256_file(target)}  {relative}\n")
    return "".join(lines).encode("utf-8")


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row["fetch_status"]) for row in rows)
    return {
        "target_count": len(rows),
        "complete_download_count": sum(
            count for status, count in statuses.items() if status in DOWNLOAD_COMPLETE_STATUSES
        ),
        "response_saved_count": sum(1 for row in rows if row.get("response_file")),
        "failed_or_incomplete_count": sum(
            count for status, count in statuses.items() if status not in DOWNLOAD_COMPLETE_STATUSES
        ),
        "status_counts": dict(sorted(statuses.items())),
        "transport_downgrade_count": sum(
            1 for row in rows if row.get("transport_downgrade") is True
        ),
        "cross_host_redirect_count": sum(
            1 for row in rows if row.get("cross_host_redirect") is True
        ),
    }


def _write_capture_payload(
    *,
    staging: Path,
    register_path: Path,
    workspace_root: Path,
    rows: Sequence[Mapping[str, Any]],
    started_at: str,
    finished_at: str,
) -> tuple[str, dict[str, Any]]:
    inputs = staging / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    register_copy = inputs / register_path.name
    shutil.copyfile(register_path, register_copy)
    os.chmod(register_copy, 0o600)
    acquisition = {
        "policy_id": CAPTURE_POLICY_ID,
        "proxy_environment_cleared": list(PROXY_ENVIRONMENT_VARIABLES),
        "requests_trust_env": False,
        "automatic_retries": 0,
        "redirect_limit": MAX_REDIRECTS,
        "accepted_schemes": ["http", "https"],
        "public_address_only": True,
        "tls_verification": True,
        "connect_timeout_seconds": CONNECT_TIMEOUT_SECONDS,
        "read_timeout_seconds": READ_TIMEOUT_SECONDS,
        "per_response_limit_bytes": MAX_RESPONSE_BYTES,
        "capture_limit_bytes": MAX_CAPTURE_BYTES,
    }
    identity = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "artifact_kind": CAPTURE_ARTIFACT_KIND,
        "scope": "development-only-candidate-acquisition",
        "source_register_sha256": sha256_file(register_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "acquisition": acquisition,
        "pages": list(rows),
    }
    capture_id = CAPTURE_ID_PREFIX + canonical_sha256(identity)
    manifest = dict(identity)
    manifest["capture_id"] = capture_id
    manifest["source_register"] = {
        "workspace_path": _relative_workspace_path(register_path, workspace_root),
        "snapshot_file": register_copy.relative_to(staging).as_posix(),
        "sha256": sha256_file(register_copy),
    }
    manifest["summary"] = _summary(rows)
    manifest["promotion"] = {
        "g3_catalog_applied": False,
        "g3_source_sync_receipt_created": False,
        "pricing_executable": False,
        "human_evidence_review_required": True,
    }
    write_canonical_json(staging / "manifest.json", manifest)
    os.chmod(staging / "manifest.json", 0o600)
    sum_paths = [
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "payload_manifest.json"}
    ]
    write_bytes_atomic(staging / "SHA256SUMS", _sha256sums(staging, sum_paths))
    os.chmod(staging / "SHA256SUMS", 0o600)
    return capture_id, manifest


def _safe_capture_file(directory: Path, logical_path: Any) -> Path:
    if not isinstance(logical_path, str) or not logical_path:
        raise ExternalPageCaptureError("capture file path is missing")
    relative = Path(logical_path)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != logical_path:
        raise ExternalPageCaptureError("capture file path is unsafe")
    target = directory / relative
    if not target.is_file() or target.is_symlink():
        raise ExternalPageCaptureError("capture file is unavailable or unsafe")
    return target


def validate_external_page_capture(
    capture_directory: str | Path,
    *,
    require_directory_id: bool = True,
) -> dict[str, Any]:
    directory = Path(capture_directory).resolve()
    try:
        payload_manifest_sha256 = validate_payload_manifest(directory)
        manifest = _object(load_json(directory / "manifest.json"), "capture manifest")
    except TrainingArtifactError as exc:
        raise ExternalPageCaptureError(str(exc)) from exc
    expected_top = {
        "schema_version",
        "artifact_kind",
        "scope",
        "source_register_sha256",
        "started_at",
        "finished_at",
        "acquisition",
        "pages",
        "capture_id",
        "source_register",
        "summary",
        "promotion",
    }
    if set(manifest) != expected_top:
        raise ExternalPageCaptureError("capture manifest fields differ")
    if (
        manifest.get("schema_version") != CAPTURE_SCHEMA_VERSION
        or manifest.get("artifact_kind") != CAPTURE_ARTIFACT_KIND
        or manifest.get("scope") != "development-only-candidate-acquisition"
    ):
        raise ExternalPageCaptureError("capture identity differs")
    capture_id = str(manifest.get("capture_id", ""))
    identity = {
        key: manifest[key]
        for key in (
            "schema_version",
            "artifact_kind",
            "scope",
            "source_register_sha256",
            "started_at",
            "finished_at",
            "acquisition",
            "pages",
        )
    }
    if capture_id != CAPTURE_ID_PREFIX + canonical_sha256(identity):
        raise ExternalPageCaptureError("capture ID does not bind its acquisition results")
    if require_directory_id and directory.name != capture_id:
        raise ExternalPageCaptureError("capture directory name differs from capture ID")
    acquisition = _object(manifest.get("acquisition"), "acquisition")
    if (
        acquisition.get("policy_id") != CAPTURE_POLICY_ID
        or acquisition.get("proxy_environment_cleared")
        != list(PROXY_ENVIRONMENT_VARIABLES)
        or acquisition.get("requests_trust_env") is not False
        or acquisition.get("automatic_retries") != 0
    ):
        raise ExternalPageCaptureError("zero-proxy single-attempt evidence differs")
    source_register = _object(manifest.get("source_register"), "source_register")
    register_copy = _safe_capture_file(directory, source_register.get("snapshot_file"))
    if (
        sha256_file(register_copy) != source_register.get("sha256")
        or source_register.get("sha256") != manifest.get("source_register_sha256")
    ):
        raise ExternalPageCaptureError("source register snapshot hash differs")
    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ExternalPageCaptureError("capture pages must be a non-empty array")
    urls: set[str] = set()
    referenced_files = {
        "manifest.json",
        "SHA256SUMS",
        "payload_manifest.json",
        str(source_register["snapshot_file"]),
    }
    for raw_row in pages:
        row = _object(raw_row, "capture page")
        url = row.get("requested_url")
        if not isinstance(url, str) or url in urls:
            raise ExternalPageCaptureError("capture requested URLs are invalid or duplicated")
        _validate_url_shape(url)
        urls.add(url)
        if row.get("attempt_count") != 1 or row.get("fetch_status") not in FETCH_STATUSES:
            raise ExternalPageCaptureError("capture attempt or status differs")
        response_file = row.get("response_file")
        if response_file is None:
            if row.get("response_sha256") is not None or row.get("response_size_bytes") != 0:
                raise ExternalPageCaptureError("missing response has payload metadata")
        else:
            response = _safe_capture_file(directory, response_file)
            referenced_files.add(str(response_file))
            if (
                response.stat().st_size != row.get("response_size_bytes")
                or sha256_file(response) != row.get("response_sha256")
            ):
                raise ExternalPageCaptureError("response payload replay failed")
        projection_file = row.get("text_projection_file")
        if projection_file is not None:
            projection = _safe_capture_file(directory, projection_file)
            referenced_files.add(str(projection_file))
            if sha256_file(projection) != row.get("text_projection_sha256"):
                raise ExternalPageCaptureError("text projection replay failed")
    if manifest.get("summary") != _summary(pages):
        raise ExternalPageCaptureError("capture summary does not replay")
    promotion = _object(manifest.get("promotion"), "promotion")
    if any(
        promotion.get(field) is not expected
        for field, expected in {
            "g3_catalog_applied": False,
            "g3_source_sync_receipt_created": False,
            "pricing_executable": False,
            "human_evidence_review_required": True,
        }.items()
    ):
        raise ExternalPageCaptureError("capture was incorrectly promoted")
    sums_file = directory / "SHA256SUMS"
    try:
        sums_lines = sums_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ExternalPageCaptureError("cannot read SHA256SUMS") from exc
    expected_sum_paths = sorted(referenced_files - {"SHA256SUMS", "payload_manifest.json"})
    expected_sums = [
        f"{sha256_file(directory / relative)}  {relative}" for relative in expected_sum_paths
    ]
    if sums_lines != expected_sums:
        raise ExternalPageCaptureError("SHA256SUMS does not replay")
    return {
        "capture_id": capture_id,
        "target": str(directory),
        "payload_manifest_sha256": payload_manifest_sha256,
        "summary": manifest["summary"],
    }


def capture_registered_pages(
    *,
    workspace_root: str | Path,
    register_path: str | Path,
    output_root: str | Path,
    extra_urls: Sequence[Mapping[str, Any]] = (),
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Capture all unique register URLs and explicit extra pages once each."""

    root = Path(workspace_root).resolve()
    register = Path(register_path).resolve()
    _relative_workspace_path(register, root)
    destination_root = Path(output_root).resolve()
    try:
        destination_root.relative_to(root)
    except ValueError as exc:
        raise ExternalPageCaptureError("capture output root must remain in workspace") from exc
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ExternalPageCaptureError("workers must be between 1 and 8")
    rows = extract_registered_urls(register)
    by_url = {str(row["requested_url"]): row for row in rows}
    for raw_extra in extra_urls:
        extra = _object(raw_extra, "extra URL")
        url = extra.get("requested_url")
        role = extra.get("role")
        if not isinstance(url, str) or not isinstance(role, str) or not role.strip():
            raise ExternalPageCaptureError("extra URL requires requested_url and role")
        _validate_url_shape(url)
        if url in by_url:
            if role not in by_url[url]["roles"]:
                by_url[url]["roles"].append(role)
        else:
            by_url[url] = {
                "requested_url": url,
                "roles": [role],
                "register_occurrences": [],
            }
    ordered = [by_url[url] for url in sorted(by_url)]
    for name in PROXY_ENVIRONMENT_VARIABLES:
        os.environ.pop(name, None)
    started_at = _utc_now()
    staging = new_staging_directory(destination_root, "wp3capture-pending")
    os.chmod(staging, 0o700)
    (staging / "responses").mkdir(mode=0o700)
    (staging / "text").mkdir(mode=0o700)
    budget = _CaptureBudget(MAX_CAPTURE_BYTES)
    results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_fetch_one, row, staging=staging, budget=budget): row
                for row in ordered
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda row: str(row["requested_url"]))
        finished_at = _utc_now()
        capture_id, _manifest = _write_capture_payload(
            staging=staging,
            register_path=register,
            workspace_root=root,
            rows=results,
            started_at=started_at,
            finished_at=finished_at,
        )
        target = destination_root / capture_id
        payload_manifest_sha256 = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda directory: validate_external_page_capture(
                directory, require_directory_id=False
            ),
        )
        report = validate_external_page_capture(target)
        if report["payload_manifest_sha256"] != payload_manifest_sha256:
            raise ExternalPageCaptureError("published payload manifest hash differs")
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def derive_capture_excluding_urls(
    *,
    parent_capture_directory: str | Path,
    output_root: str | Path,
    excluded_urls: Sequence[str],
) -> dict[str, Any]:
    """Publish a replay-equivalent capture with explicitly unwanted URLs removed.

    No network operation occurs.  The retained response bytes and per-URL
    acquisition receipts are copied from a fully validated immutable parent.
    """

    parent_report = validate_external_page_capture(parent_capture_directory)
    parent = Path(parent_report["target"])
    parent_manifest = _object(load_json(parent / "manifest.json"), "parent capture")
    excluded = set(excluded_urls)
    if not excluded or any(not isinstance(url, str) or not url for url in excluded):
        raise ExternalPageCaptureError("excluded URLs must be non-empty exact strings")
    parent_urls = {str(page["requested_url"]) for page in parent_manifest["pages"]}
    if not excluded.issubset(parent_urls):
        raise ExternalPageCaptureError("an excluded URL is absent from the parent capture")
    pages = [
        page
        for page in parent_manifest["pages"]
        if str(page["requested_url"]) not in excluded
    ]
    if not pages:
        raise ExternalPageCaptureError("derived capture cannot remove every page")
    output = Path(output_root).resolve()
    staging = new_staging_directory(output, "wp3capture-pending")
    os.chmod(staging, 0o700)
    try:
        for page in pages:
            for field in ("response_file", "text_projection_file"):
                logical = page.get(field)
                if logical is None:
                    continue
                source = _safe_capture_file(parent, logical)
                destination = staging / str(logical)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copyfile(source, destination)
                os.chmod(destination, 0o600)
        source_register = _object(
            parent_manifest["source_register"], "parent source register"
        )
        source_register_file = _safe_capture_file(
            parent, source_register["snapshot_file"]
        )
        register_destination = staging / str(source_register["snapshot_file"])
        register_destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source_register_file, register_destination)
        os.chmod(register_destination, 0o600)
        identity = {
            key: parent_manifest[key]
            for key in (
                "schema_version",
                "artifact_kind",
                "scope",
                "source_register_sha256",
                "started_at",
                "finished_at",
                "acquisition",
            )
        }
        identity["pages"] = pages
        capture_id = CAPTURE_ID_PREFIX + canonical_sha256(identity)
        manifest = dict(identity)
        manifest["capture_id"] = capture_id
        manifest["source_register"] = source_register
        manifest["summary"] = _summary(pages)
        manifest["promotion"] = dict(parent_manifest["promotion"])
        write_canonical_json(staging / "manifest.json", manifest)
        os.chmod(staging / "manifest.json", 0o600)
        sum_paths = [
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file() and path.name not in {"SHA256SUMS", "payload_manifest.json"}
        ]
        write_bytes_atomic(staging / "SHA256SUMS", _sha256sums(staging, sum_paths))
        os.chmod(staging / "SHA256SUMS", 0o600)
        target = output / capture_id
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda directory: validate_external_page_capture(
                directory, require_directory_id=False
            ),
        )
        report = validate_external_page_capture(target)
        if report["payload_manifest_sha256"] != payload_hash:
            raise ExternalPageCaptureError("derived capture payload hash changed")
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _browser_flags(user_data_directory: Path) -> list[str]:
    return [
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--no-proxy-server",
        "--proxy-server=direct://",
        "--proxy-bypass-list=*",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-extensions",
        "--disable-sync",
        "--metrics-recording-only",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-features=OptimizationGuideModelDownloading",
        f"--user-data-dir={user_data_directory}",
        f"--virtual-time-budget={BROWSER_VIRTUAL_TIME_BUDGET_MS}",
        "--dump-dom",
    ]


def _browser_version(chrome_binary: Path, environment: Mapping[str, str]) -> str:
    try:
        completed = subprocess.run(
            [str(chrome_binary), "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            env=dict(environment),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExternalPageCaptureError("cannot identify the browser binary") from exc
    try:
        version = completed.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ExternalPageCaptureError("browser version is not UTF-8") from exc
    if not version or len(version) > 500:
        raise ExternalPageCaptureError("browser version is invalid")
    return version


def _render_one(
    page: Mapping[str, Any],
    *,
    staging: Path,
    chrome_binary: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    url = str(page["requested_url"])
    result: dict[str, Any] = {
        "requested_url": url,
        "roles": list(page["roles"]),
        "parent_fetch_status": str(page["fetch_status"]),
        "navigation_count": 1,
        "started_at": _utc_now(),
        "finished_at": None,
        "exit_code": None,
        "stderr_size_bytes": 0,
        "stderr_sha256": None,
        "render_status": "browser_error",
        "rendered_dom_file": None,
        "rendered_dom_size_bytes": 0,
        "rendered_dom_sha256": None,
        "text_projection_file": None,
        "text_projection_sha256": None,
        "error": None,
    }
    try:
        profile = Path(tempfile.mkdtemp(prefix="wp3-browser-profile-"))
        try:
            command = [
                str(chrome_binary),
                *_browser_flags(profile),
                url,
            ]
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=BROWSER_TIMEOUT_SECONDS,
                env=dict(environment),
            )
        finally:
            shutil.rmtree(profile, ignore_errors=True)
        result["exit_code"] = completed.returncode
        result["stderr_size_bytes"] = len(completed.stderr)
        result["stderr_sha256"] = hashlib.sha256(completed.stderr).hexdigest()
        body = completed.stdout
        if completed.returncode != 0:
            result["error"] = "browser process returned a non-zero exit code"
            return result
        if not body:
            result["render_status"] = "browser_empty"
            result["error"] = "browser produced no rendered DOM"
            return result
        if len(body) > MAX_RENDERED_DOM_BYTES:
            result["render_status"] = "browser_output_too_large"
            result["error"] = "rendered DOM exceeds the byte limit"
            return result
        analysis, projection = _classify_body(
            body,
            declared_type="text/html",
            full_content_type="text/html; charset=utf-8",
            final_url=url,
        )
        result["content_analysis"] = analysis
        basename = _file_basename(url, "html").removesuffix(".html")
        dom_path = staging / "rendered_dom" / f"{basename}.rendered.html"
        write_bytes_atomic(dom_path, body)
        os.chmod(dom_path, 0o600)
        result["rendered_dom_file"] = dom_path.relative_to(staging).as_posix()
        result["rendered_dom_size_bytes"] = len(body)
        result["rendered_dom_sha256"] = hashlib.sha256(body).hexdigest()
        if projection is not None:
            text_path = staging / "text" / f"{basename}.rendered.txt"
            write_bytes_atomic(text_path, projection)
            os.chmod(text_path, 0o600)
            result["text_projection_file"] = text_path.relative_to(staging).as_posix()
            result["text_projection_sha256"] = hashlib.sha256(projection).hexdigest()
        validation = analysis.get("content_validation")
        if validation == "usable_static_text":
            if "provider_pricing_glm" in result["roles"]:
                rendered = (projection or b"").decode("utf-8", errors="ignore").casefold()
                model_present = "glm-5.3" in rendered
                tariff_context_present = (
                    "token" in rendered
                    and any(marker in rendered for marker in ("输入", "input"))
                    and any(marker in rendered for marker in ("输出", "output"))
                )
                result["pricing_candidate_checks"] = {
                    "requested_model_text_present": model_present,
                    "input_output_token_context_present": tariff_context_present,
                }
                result["render_status"] = (
                    "rendered_usable"
                    if model_present and tariff_context_present
                    else "rendered_needs_pricing_review"
                )
            else:
                result["render_status"] = "rendered_usable"
        elif validation == "error_or_challenge":
            result["render_status"] = "rendered_error_or_challenge"
        else:
            result["render_status"] = "rendered_incomplete"
        return result
    except subprocess.TimeoutExpired:
        result["render_status"] = "browser_timeout"
        result["error"] = "browser render exceeded the timeout"
        return result
    except (OSError, ExternalPageCaptureError) as exc:
        result["error"] = str(exc)[:2000]
        return result
    finally:
        result["finished_at"] = _utc_now()


def _render_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row["render_status"]) for row in rows)
    return {
        "target_count": len(rows),
        "rendered_usable_count": statuses.get("rendered_usable", 0),
        "rendered_dom_saved_count": sum(
            1 for row in rows if row.get("rendered_dom_file") is not None
        ),
        "status_counts": dict(sorted(statuses.items())),
    }


def _validate_browser_render(
    directory: Path,
    *,
    require_directory_id: bool,
) -> dict[str, Any]:
    try:
        payload_hash = validate_payload_manifest(directory)
        manifest = _object(load_json(directory / "manifest.json"), "browser render manifest")
    except TrainingArtifactError as exc:
        raise ExternalPageCaptureError(str(exc)) from exc
    required = {
        "schema_version",
        "artifact_kind",
        "scope",
        "render_id",
        "parent_capture",
        "browser",
        "acquisition",
        "pages",
        "summary",
        "promotion",
    }
    if set(manifest) != required:
        raise ExternalPageCaptureError("browser render manifest fields differ")
    if (
        manifest.get("schema_version") != BROWSER_RENDER_SCHEMA_VERSION
        or manifest.get("artifact_kind") != BROWSER_RENDER_ARTIFACT_KIND
        or manifest.get("scope") != "development-only-candidate-acquisition"
    ):
        raise ExternalPageCaptureError("browser render identity differs")
    identity = {
        key: manifest[key]
        for key in (
            "schema_version",
            "artifact_kind",
            "scope",
            "parent_capture",
            "browser",
            "acquisition",
            "pages",
        )
    }
    render_id = BROWSER_RENDER_ID_PREFIX + canonical_sha256(identity)
    if manifest.get("render_id") != render_id:
        raise ExternalPageCaptureError("browser render ID does not replay")
    if require_directory_id and directory.name != render_id:
        raise ExternalPageCaptureError("browser render directory name differs")
    acquisition = _object(manifest.get("acquisition"), "browser acquisition")
    if (
        acquisition.get("policy_id") != BROWSER_RENDER_POLICY_ID
        or acquisition.get("proxy_environment_cleared")
        != list(PROXY_ENVIRONMENT_VARIABLES)
        or acquisition.get("navigation_runs_per_url") != 1
    ):
        raise ExternalPageCaptureError("browser zero-proxy evidence differs")
    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ExternalPageCaptureError("browser render pages are invalid")
    urls: set[str] = set()
    referenced = {"manifest.json", "SHA256SUMS", "payload_manifest.json"}
    for raw_row in pages:
        row = _object(raw_row, "browser render page")
        url = row.get("requested_url")
        if not isinstance(url, str) or url in urls or row.get("navigation_count") != 1:
            raise ExternalPageCaptureError("browser render URL or count differs")
        urls.add(url)
        for path_key, hash_key, size_key in (
            ("rendered_dom_file", "rendered_dom_sha256", "rendered_dom_size_bytes"),
            ("text_projection_file", "text_projection_sha256", None),
        ):
            logical = row.get(path_key)
            if logical is None:
                continue
            target = _safe_capture_file(directory, logical)
            referenced.add(str(logical))
            if sha256_file(target) != row.get(hash_key):
                raise ExternalPageCaptureError("browser render payload hash differs")
            if size_key is not None and target.stat().st_size != row.get(size_key):
                raise ExternalPageCaptureError("browser render payload size differs")
    if manifest.get("summary") != _render_summary(pages):
        raise ExternalPageCaptureError("browser render summary does not replay")
    if manifest.get("promotion") != {
        "g3_catalog_applied": False,
        "pricing_executable": False,
        "human_review_required": True,
    }:
        raise ExternalPageCaptureError("browser render was incorrectly promoted")
    expected_sum_paths = sorted(referenced - {"SHA256SUMS", "payload_manifest.json"})
    expected_sums = [
        f"{sha256_file(directory / relative)}  {relative}" for relative in expected_sum_paths
    ]
    try:
        stored_sums = (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ExternalPageCaptureError("cannot read browser SHA256SUMS") from exc
    if stored_sums != expected_sums:
        raise ExternalPageCaptureError("browser SHA256SUMS does not replay")
    return {
        "render_id": render_id,
        "target": str(directory),
        "payload_manifest_sha256": payload_hash,
        "summary": manifest["summary"],
    }


def validate_browser_render_capture(
    render_directory: str | Path,
) -> dict[str, Any]:
    return _validate_browser_render(
        Path(render_directory).resolve(), require_directory_id=True
    )


def capture_incomplete_pages_with_browser(
    *,
    parent_capture_directory: str | Path,
    output_root: str | Path,
    chrome_binary: str | Path,
) -> dict[str, Any]:
    """Render only response-bearing incomplete pages from an immutable capture."""

    parent_report = validate_external_page_capture(parent_capture_directory)
    parent = Path(parent_report["target"])
    manifest = _object(load_json(parent / "manifest.json"), "parent capture manifest")
    selected_statuses = {
        "downloaded_needs_browser_archive",
        "downloaded_needs_review",
        "downloaded_error_or_challenge",
        "http_error",
    }
    selected = [
        page
        for page in manifest["pages"]
        if page.get("fetch_status") in selected_statuses and page.get("response_file")
    ]
    if not selected:
        raise ExternalPageCaptureError("parent capture has no browser-render targets")
    chrome = Path(chrome_binary).resolve()
    if not chrome.is_file() or chrome.is_symlink() or not os.access(chrome, os.X_OK):
        raise ExternalPageCaptureError("browser binary is unavailable or unsafe")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in PROXY_ENVIRONMENT_VARIABLES
    }
    browser = {
        "version": _browser_version(chrome, environment),
        "binary_sha256": sha256_file(chrome),
        "headless": True,
        "tls_verification_disabled": False,
        "proxy_flags": [
            "--no-proxy-server",
            "--proxy-server=direct://",
            "--proxy-bypass-list=*",
        ],
        "virtual_time_budget_ms": BROWSER_VIRTUAL_TIME_BUDGET_MS,
        "timeout_seconds": BROWSER_TIMEOUT_SECONDS,
    }
    output = Path(output_root).resolve()
    staging = new_staging_directory(output, "wp3render-pending")
    os.chmod(staging, 0o700)
    (staging / "rendered_dom").mkdir(mode=0o700)
    (staging / "text").mkdir(mode=0o700)
    try:
        rows = [
            _render_one(
                page,
                staging=staging,
                chrome_binary=chrome,
                environment=environment,
            )
            for page in selected
        ]
        rows.sort(key=lambda row: str(row["requested_url"]))
        acquisition = {
            "policy_id": BROWSER_RENDER_POLICY_ID,
            "proxy_environment_cleared": list(PROXY_ENVIRONMENT_VARIABLES),
            "navigation_runs_per_url": 1,
            "selected_parent_statuses": sorted(selected_statuses),
        }
        identity = {
            "schema_version": BROWSER_RENDER_SCHEMA_VERSION,
            "artifact_kind": BROWSER_RENDER_ARTIFACT_KIND,
            "scope": "development-only-candidate-acquisition",
            "parent_capture": {
                "capture_id": parent_report["capture_id"],
                "payload_manifest_sha256": parent_report["payload_manifest_sha256"],
            },
            "browser": browser,
            "acquisition": acquisition,
            "pages": rows,
        }
        render_id = BROWSER_RENDER_ID_PREFIX + canonical_sha256(identity)
        render_manifest = dict(identity)
        render_manifest["render_id"] = render_id
        render_manifest["summary"] = _render_summary(rows)
        render_manifest["promotion"] = {
            "g3_catalog_applied": False,
            "pricing_executable": False,
            "human_review_required": True,
        }
        write_canonical_json(staging / "manifest.json", render_manifest)
        os.chmod(staging / "manifest.json", 0o600)
        sum_paths = [
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file() and path.name not in {"SHA256SUMS", "payload_manifest.json"}
        ]
        write_bytes_atomic(staging / "SHA256SUMS", _sha256sums(staging, sum_paths))
        os.chmod(staging / "SHA256SUMS", 0o600)
        target = output / render_id
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda directory: _validate_browser_render(
                directory, require_directory_id=False
            ),
        )
        report = validate_browser_render_capture(target)
        if report["payload_manifest_sha256"] != payload_hash:
            raise ExternalPageCaptureError("browser render payload hash changed")
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "CAPTURE_ARTIFACT_KIND",
    "CAPTURE_SCHEMA_VERSION",
    "BROWSER_RENDER_ARTIFACT_KIND",
    "BROWSER_RENDER_SCHEMA_VERSION",
    "ExternalPageCaptureError",
    "capture_incomplete_pages_with_browser",
    "capture_registered_pages",
    "derive_capture_excluding_urls",
    "extract_registered_urls",
    "validate_browser_render_capture",
    "validate_external_page_capture",
]
