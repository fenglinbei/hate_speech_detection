#!/usr/bin/env python3
"""Serve the local-only WP3 G3 form-reference reviewer."""

from __future__ import annotations

import ipaddress
import json
import re
import secrets
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from build_lex.terminology_g3_form_reference import (  # noqa: E402
    G3FormReferenceError,
    G3FormReviewConflict,
    create_form_review_session,
    read_form_review_session,
    reopen_form_decision,
    save_form_decision_from_validated_frame,
    validate_form_review_session,
    validate_form_review_frame,
)
from data.training_artifacts import canonical_json_bytes  # noqa: E402


BOOTSTRAP_SCHEMA_VERSION = "wp3-g3-form-review-bootstrap/v1"
ITEM_SCHEMA_VERSION = "wp3-g3-form-review-item/v1"
MUTATION_SCHEMA_VERSION = "wp3-g3-form-review-mutation/v1"
MAX_REQUEST_BYTES = 64 * 1024
REQUEST_ID_HEADER = "X-Request-ID"
CLIENT_INSTANCE_HEADER = "X-Review-Client-Instance"
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9_-]{16,128}\Z")
_CLIENT_INSTANCE_RE = re.compile(r"[A-Za-z0-9_-]{16,64}\Z")
ASSET_NAMES = frozenset(
    {
        "index.html",
        "app.js",
        "core.js",
        "styles.css",
        "review-base.css",
        "review-core.js",
    }
)
SHARED_REVIEW_ASSETS = {
    "review-base.css": REPOSITORY_ROOT
    / "tools"
    / "wp3_candidate_review_ui"
    / "styles.css",
    "review-core.js": REPOSITORY_ROOT
    / "tools"
    / "wp3_candidate_review_ui"
    / "core.js",
}


class WebFormReviewError(RuntimeError):
    """Safe request-edge error."""


def _loopback_authority(value: str) -> bool:
    if not value or len(value) > 255:
        return False
    try:
        parsed = urlsplit("//" + value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return False
    if (
        hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _loopback_origin(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
        and _loopback_authority(parsed.netloc)
    )


def _loopback_host(value: str) -> str:
    if value == "localhost":
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise WebFormReviewError("host must be localhost or a loopback IP") from exc
    if not address.is_loopback:
        raise WebFormReviewError("form-review service may bind only to loopback")
    return value


class FormReviewService:
    def __init__(
        self,
        *,
        workspace_root: Path,
        source_bundle_dir: Path,
        frame_dir: Path,
        session_path: Path,
        reviewer_id: str,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.source_bundle_dir = source_bundle_dir.resolve()
        self.frame_dir = frame_dir.resolve()
        self.session_path = session_path.resolve()
        self.asset_root = Path(__file__).resolve().parent
        self.frame = validate_form_review_frame(
            self.frame_dir,
            source_bundle_dir=self.source_bundle_dir,
            workspace_root=self.workspace_root,
            require_current_implementation=True,
        )
        self.item_by_id = {row["item_id"]: row for row in self.frame["items"]}
        self.evidence_by_id = {
            row["evidence_id"]: row for row in self.frame["evidence"]
        }
        self.session_token = secrets.token_urlsafe(32)
        self.allowed_hosts: set[str] = set()
        self.allowed_origins: set[str] = set()
        self._lock = threading.RLock()
        create_form_review_session(
            frame_dir=self.frame_dir,
            source_bundle_dir=self.source_bundle_dir,
            workspace_root=self.workspace_root,
            session_path=self.session_path,
            reviewer_id=reviewer_id,
        )

    def configure_network(
        self, port: int, public_origin: str | None = None
    ) -> None:
        self.allowed_hosts = {
            f"127.0.0.1:{port}",
            f"localhost:{port}",
            f"[::1]:{port}",
        }
        self.allowed_origins = {
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
            f"http://[::1]:{port}",
        }
        if public_origin is not None:
            try:
                parsed = urlsplit(public_origin)
                _ = parsed.port
            except ValueError as exc:
                raise WebFormReviewError("public origin is invalid") from exc
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise WebFormReviewError(
                    "public origin must be an https origin without a path"
                )
            self.allowed_hosts.add(parsed.netloc)
            self.allowed_origins.add(public_origin)

    def _status(self, session: Mapping[str, Any]) -> dict[str, Any]:
        summary = validate_form_review_session(
            frame=self.frame,
            session=session,
            require_complete=False,
        )
        return {
            "frame_id": self.frame["frame_id"],
            "reviewer_id": session["reviewer_id"],
            "revision": session["revision"],
            "finalized_reference_id": session["finalized_reference_id"],
            **summary,
        }

    def _summary(self, session: Mapping[str, Any], item_id: str) -> dict[str, Any]:
        decision = session["decisions"][item_id]
        item = self.item_by_id[item_id]
        evidence = [
            self.evidence_by_id[evidence_id] for evidence_id in item["evidence_ids"]
        ]
        return {
            "item_id": item_id,
            "surface": item["surface"],
            "canonical": item["canonical"],
            "proposed_family": item["proposed_family"],
            "publishers": sorted({str(row["publisher"]) for row in evidence}),
            "source_roles": sorted(
                {
                    str(row["source_role"])
                    for row in evidence
                    if row.get("source_role")
                }
            ),
            "evidence_count": len(evidence),
            "status": decision["status"],
            "action": decision["action"],
        }

    def bootstrap(self) -> dict[str, Any]:
        with self._lock:
            session = read_form_review_session(self.session_path)
            return {
                "schema_version": BOOTSTRAP_SCHEMA_VERSION,
                "session_token": self.session_token,
                "frame_id": self.frame["frame_id"],
                "reviewer_id": session["reviewer_id"],
                "revision": session["revision"],
                "status": self._status(session),
                "actions": ["accept", "reject", "edit", "defer"],
                "families": [
                    "known_variant",
                    "phonetic_variant",
                    "orthographic_variant",
                ],
                "items": [
                    self._summary(session, item["item_id"])
                    for item in self.frame["items"]
                ],
                "warnings": [
                    "DEVELOPMENT ONLY / NON-SEALED / NON-SCIENTIFIC",
                    "The handbook is a rubric and cannot prove a form relation.",
                    "No item is accepted automatically.",
                ],
            }

    def item_state(self, item_id: str) -> dict[str, Any]:
        with self._lock:
            if item_id not in self.item_by_id:
                raise WebFormReviewError("unknown form-review item")
            session = read_form_review_session(self.session_path)
            item = self.item_by_id[item_id]
            return {
                "schema_version": ITEM_SCHEMA_VERSION,
                "revision": session["revision"],
                "item": item,
                "evidence": [
                    self.evidence_by_id[evidence_id]
                    for evidence_id in item["evidence_ids"]
                ],
                "decision": session["decisions"][item_id],
                "item_summary": self._summary(session, item_id),
            }

    def save(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "session_token",
            "expected_revision",
            "item_id",
            "decision",
            "confirm",
        }
        if set(payload) != required or not isinstance(payload.get("confirm"), bool):
            raise WebFormReviewError("form save fields are invalid")
        item_id = str(payload["item_id"])
        with self._lock:
            session = save_form_decision_from_validated_frame(
                frame=self.frame,
                session_path=self.session_path,
                item_id=item_id,
                decision=payload["decision"],
                confirm=payload["confirm"],
                expected_revision=str(payload["expected_revision"]),
            )
            return {
                "schema_version": MUTATION_SCHEMA_VERSION,
                "revision": session["revision"],
                "status": self._status(session),
                "decision": session["decisions"][item_id],
                "item_summary": self._summary(session, item_id),
            }

    def reopen(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "session_token",
            "expected_revision",
            "item_id",
            "reason",
        }
        if set(payload) != required:
            raise WebFormReviewError("form reopen fields are invalid")
        item_id = str(payload["item_id"])
        with self._lock:
            session = reopen_form_decision(
                session_path=self.session_path,
                item_id=item_id,
                reason=str(payload["reason"]),
                expected_revision=str(payload["expected_revision"]),
            )
            return {
                "schema_version": MUTATION_SCHEMA_VERSION,
                "revision": session["revision"],
                "status": self._status(session),
                "decision": session["decisions"][item_id],
                "item_summary": self._summary(session, item_id),
            }


class FormReviewRequestHandler(BaseHTTPRequestHandler):
    service: FormReviewService

    def _start_observation(self) -> None:
        self._request_started_at = time.monotonic()
        forwarded_request_id = self.headers.get(REQUEST_ID_HEADER, "")
        self._request_id = (
            forwarded_request_id
            if _REQUEST_ID_RE.fullmatch(forwarded_request_id)
            else secrets.token_hex(16)
        )
        client_instance = self.headers.get(CLIENT_INSTANCE_HEADER, "")
        self._client_instance_id = (
            client_instance
            if _CLIENT_INSTANCE_RE.fullmatch(client_instance)
            else "-"
        )

    def log_message(self, format_string: str, *args: Any) -> None:
        elapsed_ms = max(
            0.0,
            (time.monotonic() - getattr(self, "_request_started_at", time.monotonic()))
            * 1000,
        )
        request_id = getattr(self, "_request_id", "-")
        client_instance = getattr(self, "_client_instance_id", "-")
        sys.stderr.write(
            "[wp3-g3-form-review] "
            f"request_id={request_id} client_instance={client_instance} "
            f"duration_ms={elapsed_ms:.3f} "
            + (format_string % args)
            + "\n"
        )

    def _host_allowed(self) -> bool:
        return self.headers.get("Host", "") in self.service.allowed_hosts

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in self.service.allowed_origins

    def _headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(REQUEST_ID_HEADER, getattr(self, "_request_id", ""))
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
        )

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status.value)
        self._headers(content_type, len(body))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
        self._send(status, canonical_json_bytes(value), "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"status": status.value, "error": message})

    def _asset(self, name: str) -> None:
        if name not in ASSET_NAMES:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        path = SHARED_REVIEW_ASSETS.get(name, self.service.asset_root / name)
        if not path.is_file() or path.is_symlink():
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "UI asset missing")
            return
        content_type = {
            "index.html": "text/html; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
            "core.js": "text/javascript; charset=utf-8",
            "review-core.js": "text/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
            "review-base.css": "text/css; charset=utf-8",
        }[name]
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def _read_payload(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError as exc:
            raise WebFormReviewError("invalid Content-Length") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise WebFormReviewError("request body size is invalid")
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            raise WebFormReviewError("request must use application/json")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebFormReviewError("request JSON is invalid") from exc
        if not isinstance(value, dict):
            raise WebFormReviewError("request JSON must be an object")
        return value

    def _authorized(self, payload: Mapping[str, Any]) -> bool:
        token = payload.get("session_token")
        return isinstance(token, str) and secrets.compare_digest(
            token, self.service.session_token
        )

    def do_GET(self) -> None:  # noqa: N802
        self._start_observation()
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Host not allowed")
            return
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self._asset("index.html")
        elif path in {
            "/app.js",
            "/core.js",
            "/review-core.js",
            "/styles.css",
            "/review-base.css",
        }:
            self._asset(path[1:])
        elif path == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
        elif path == "/api/bootstrap":
            try:
                self._json(HTTPStatus.OK, self.service.bootstrap())
            except (G3FormReferenceError, OSError):
                self._error(HTTPStatus.CONFLICT, "frozen form-review state is invalid")
        elif path.startswith("/api/items/"):
            item_id = path.removeprefix("/api/items/")
            if not item_id or "/" in item_id:
                self._error(HTTPStatus.NOT_FOUND, "item not found")
                return
            try:
                self._json(HTTPStatus.OK, self.service.item_state(item_id))
            except WebFormReviewError:
                self._error(HTTPStatus.NOT_FOUND, "item not found")
            except (G3FormReferenceError, OSError):
                self._error(HTTPStatus.CONFLICT, "frozen form-review state is invalid")
        elif path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
        else:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        self._start_observation()
        if not self._host_allowed() or not self._origin_allowed():
            self.close_connection = True
            self._error(HTTPStatus.FORBIDDEN, "request origin not allowed")
            return
        try:
            payload = self._read_payload()
            if not self._authorized(payload):
                self.close_connection = True
                self._error(HTTPStatus.FORBIDDEN, "invalid session token")
                return
            path = urlsplit(self.path).path
            if path == "/api/save":
                self._json(HTTPStatus.OK, self.service.save(payload))
            elif path == "/api/reopen":
                self._json(HTTPStatus.OK, self.service.reopen(payload))
            else:
                self._error(HTTPStatus.NOT_FOUND, "API route not found")
        except G3FormReviewConflict as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except (G3FormReferenceError, WebFormReviewError, TypeError, ValueError) as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except OSError:
            self._error(HTTPStatus.CONFLICT, "form-review state could not be written")
        except Exception as exc:  # pragma: no cover
            print(
                f"[wp3-g3-form-review] unexpected failure: {type(exc).__name__}",
                file=sys.stderr,
            )
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "unexpected local error")


def run_server(
    *,
    workspace_root: Path,
    source_bundle_dir: Path,
    frame_dir: Path,
    session_path: Path,
    reviewer_id: str,
    host: str,
    port: int,
    check: bool = False,
    public_origin: str | None = None,
) -> int:
    host = _loopback_host(host)
    if not 0 <= port <= 65535:
        raise WebFormReviewError("port must be in 0..65535")
    service = FormReviewService(
        workspace_root=workspace_root,
        source_bundle_dir=source_bundle_dir,
        frame_dir=frame_dir,
        session_path=session_path,
        reviewer_id=reviewer_id,
    )
    if check:
        # ``--check`` must exercise the same origin validation as the live
        # server even though it deliberately avoids opening a socket.
        service.configure_network(port, public_origin=public_origin)
        state = service.bootstrap()
        print(
            json.dumps(
                {
                    "frame_id": state["frame_id"],
                    "confirmed": state["status"]["confirmed_count"],
                    "total": state["status"]["item_count"],
                    "development_only": True,
                    "sealed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    handler = type(
        "BoundFormReviewRequestHandler",
        (FormReviewRequestHandler,),
        {"service": service},
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    actual_port = int(server.server_address[1])
    try:
        service.configure_network(actual_port, public_origin=public_origin)
    except Exception:
        server.server_close()
        raise
    print(
        json.dumps(
            {
                "url": f"http://127.0.0.1:{actual_port}/",
                "public_origin": public_origin,
                "frame_id": service.frame["frame_id"],
                "session_file": str(service.session_path),
                "development_only": True,
                "sealed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = [
    "FormReviewRequestHandler",
    "FormReviewService",
    "MAX_REQUEST_BYTES",
    "WebFormReviewError",
    "_loopback_authority",
    "_loopback_origin",
    "run_server",
]
