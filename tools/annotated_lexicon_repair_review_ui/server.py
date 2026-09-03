#!/usr/bin/env python3
"""Serve the adapted annotated-lexicon repair review workbench."""

from __future__ import annotations

import ipaddress
import json
import re
import secrets
import sys
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

from build_lex.annotated_lexicon_repair import (  # noqa: E402
    LexiconRepairConflict,
    LexiconRepairError,
    SpanGoldReviewStore,
    canonical_bytes,
)


MAX_REQUEST_BYTES = 64 * 1024
REQUEST_ID_HEADER = "X-Request-ID"
CLIENT_INSTANCE_HEADER = "X-Review-Client-Instance"
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9_-]{16,128}\Z")
_CLIENT_INSTANCE_RE = re.compile(r"[A-Za-z0-9_-]{16,64}\Z")
ASSET_NAMES = frozenset(
    {"index.html", "app.js", "core.js", "styles.css", "review-base.css", "review-core.js"}
)
SHARED_ASSETS = {
    "review-base.css": REPOSITORY_ROOT / "tools/wp3_candidate_review_ui/styles.css",
    "review-core.js": REPOSITORY_ROOT / "tools/wp3_candidate_review_ui/core.js",
}


class ReviewWebError(RuntimeError):
    """Safe HTTP-edge error."""


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


def _loopback_host(value: str) -> str:
    if value == "localhost":
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ReviewWebError("host must be localhost or a loopback IP") from exc
    if not address.is_loopback:
        raise ReviewWebError("review service may bind only to loopback")
    return value


class SpanGoldWebService:
    def __init__(self, *, frame_path: Path, session_path: Path, reviewer_id: str) -> None:
        self.store = SpanGoldReviewStore(
            frame_path=frame_path,
            session_path=session_path,
            reviewer_id=reviewer_id,
        )
        self.asset_root = Path(__file__).resolve().parent
        self.session_token = secrets.token_urlsafe(32)
        self.allowed_hosts: set[str] = set()
        self.allowed_origins: set[str] = set()

    def configure_network(self, port: int, public_origin: str | None = None) -> None:
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
                raise ReviewWebError("public origin is invalid") from exc
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ReviewWebError(
                    "public origin must be an HTTPS origin without a path"
                )
            self.allowed_hosts.add(parsed.netloc)
            self.allowed_origins.add(public_origin)

    def authorized(self, payload: Mapping[str, Any]) -> bool:
        token = payload.get("session_token")
        return isinstance(token, str) and secrets.compare_digest(token, self.session_token)

    def bootstrap(self) -> dict[str, Any]:
        return {**self.store.bootstrap(), "session_token": self.session_token}

    def save(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {"session_token", "expected_revision", "item_id", "decision", "confirm"}
        if set(payload) != required or not isinstance(payload.get("confirm"), bool):
            raise ValueError("save request fields differ")
        return self.store.save(
            expected_revision=str(payload["expected_revision"]),
            item_id=str(payload["item_id"]),
            decision=payload["decision"],
            confirm=bool(payload["confirm"]),
        )

    def reopen(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {"session_token", "expected_revision", "item_id", "reason"}
        if set(payload) != required:
            raise ValueError("reopen request fields differ")
        return self.store.reopen(
            expected_revision=str(payload["expected_revision"]),
            item_id=str(payload["item_id"]),
            reason=str(payload["reason"]),
        )

    def export(self, payload: Mapping[str, Any]) -> bytes:
        required = {"session_token", "expected_revision"}
        if set(payload) != required:
            raise ValueError("export request fields differ")
        snapshot = self.store.snapshot()
        if snapshot["session"]["revision"] != str(payload["expected_revision"]):
            raise LexiconRepairConflict("span-gold session changed concurrently")
        return canonical_bytes(snapshot)


class SpanGoldRequestHandler(BaseHTTPRequestHandler):
    service: SpanGoldWebService

    def _start_observation(self) -> None:
        self._request_started_at = time.monotonic()
        forwarded = self.headers.get(REQUEST_ID_HEADER, "")
        self._request_id = forwarded if _REQUEST_ID_RE.fullmatch(forwarded) else secrets.token_hex(16)
        client = self.headers.get(CLIENT_INSTANCE_HEADER, "")
        self._client_instance_id = client if _CLIENT_INSTANCE_RE.fullmatch(client) else "-"

    def log_message(self, format_string: str, *args: Any) -> None:
        elapsed_ms = max(
            0.0,
            (time.monotonic() - getattr(self, "_request_started_at", time.monotonic())) * 1000,
        )
        sys.stderr.write(
            "[annotated-lexicon-span-gold] "
            f"request_id={getattr(self, '_request_id', '-')} "
            f"client_instance={getattr(self, '_client_instance_id', '-')} "
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

    def _send(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        disposition: str | None = None,
    ) -> None:
        self.send_response(status.value)
        self._headers(content_type, len(body))
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
        self._send(status, canonical_bytes(value), "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"status": status.value, "error": message})

    def _asset(self, name: str) -> None:
        if name not in ASSET_NAMES:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        path = SHARED_ASSETS.get(name, self.service.asset_root / name)
        if not path.is_file() or path.is_symlink():
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "review UI asset missing")
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
            raise ReviewWebError("invalid Content-Length") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ReviewWebError("request body size is invalid")
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            raise ReviewWebError("request must use application/json")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewWebError("request JSON is invalid") from exc
        if not isinstance(value, dict):
            raise ReviewWebError("request JSON must be an object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        self._start_observation()
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Host not allowed")
            return
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self._asset("index.html")
        elif path in {"/app.js", "/core.js", "/review-core.js", "/styles.css", "/review-base.css"}:
            self._asset(path[1:])
        elif path == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ok", "stage": "span-gold"})
        elif path == "/api/bootstrap":
            self._json(HTTPStatus.OK, self.service.bootstrap())
        elif path.startswith("/api/items/"):
            item_id = path.removeprefix("/api/items/")
            if not item_id or "/" in item_id:
                self._error(HTTPStatus.NOT_FOUND, "item not found")
                return
            try:
                self._json(HTTPStatus.OK, self.service.store.item_state(item_id))
            except ValueError:
                self._error(HTTPStatus.NOT_FOUND, "item not found")
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
            if not self.service.authorized(payload):
                self.close_connection = True
                self._error(HTTPStatus.FORBIDDEN, "invalid session token")
                return
            path = urlsplit(self.path).path
            if path == "/api/save":
                self._json(HTTPStatus.OK, self.service.save(payload))
            elif path == "/api/reopen":
                self._json(HTTPStatus.OK, self.service.reopen(payload))
            elif path == "/api/export":
                body = self.service.export(payload)
                self._send(
                    HTTPStatus.OK,
                    body,
                    "application/json; charset=utf-8",
                    disposition='attachment; filename="span-gold-review-snapshot.json"',
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, "API route not found")
        except LexiconRepairConflict as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except (LexiconRepairError, ReviewWebError, TypeError, ValueError) as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except OSError:
            self._error(HTTPStatus.CONFLICT, "review state could not be written")
        except Exception as exc:  # pragma: no cover
            print(
                f"[annotated-lexicon-span-gold] unexpected failure: {type(exc).__name__}",
                file=sys.stderr,
            )
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "unexpected local error")


def run_server(
    *,
    frame_path: Path,
    session_path: Path,
    reviewer_id: str,
    host: str,
    port: int,
    check: bool = False,
    public_origin: str | None = None,
) -> int:
    host = _loopback_host(host)
    if not 0 <= port <= 65535:
        raise ReviewWebError("port must be in 0..65535")
    service = SpanGoldWebService(
        frame_path=frame_path,
        session_path=session_path,
        reviewer_id=reviewer_id,
    )
    service.configure_network(port, public_origin=public_origin)
    if check:
        state = service.bootstrap()
        print(
            json.dumps(
                {
                    "frame_id": state["frame_id"],
                    "confirmed": state["status"]["confirmed_count"],
                    "total": state["status"]["item_count"],
                    "development_only": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    handler = type("BoundSpanGoldRequestHandler", (SpanGoldRequestHandler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    actual_port = int(server.server_address[1])
    if actual_port != port:
        service.configure_network(actual_port, public_origin=public_origin)
    print(
        json.dumps(
            {
                "url": f"http://127.0.0.1:{actual_port}/",
                "public_origin": public_origin,
                "frame_id": service.store.frame["manifest"]["frame_id"],
                "session_file": str(service.store.session_path),
                "development_only": True,
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
    "MAX_REQUEST_BYTES",
    "ReviewWebError",
    "SpanGoldRequestHandler",
    "SpanGoldWebService",
    "_loopback_authority",
    "run_server",
]
