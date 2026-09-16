"""Serve the paired-case workbench using the existing review HTTP conventions."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from build_lex.annotated_lexicon_repair import canonical_bytes  # noqa: E402
from tools.annotated_lexicon_operation_review_ui.server import (  # noqa: E402
    OperationRequestHandler,
    OperationWebService,
    ReviewWebError,
    _loopback_host,
)
from tools.general_model_paired_review_ui.store import (  # noqa: E402
    PairedReviewStore,
    ReviewConflict,
    ReviewError,
)
from tools.general_model_paired_review_ui.evidence_store import EvidenceReviewStore  # noqa: E402


EXPERIMENT = REPOSITORY_ROOT / "exps/causal_context/general_model_ld_nolabel_paired_cases_v1"
DEFAULT_DATA = EXPERIMENT / "results/paired-cases-02"
DEFAULT_SESSION = EXPERIMENT / "reviews/paired-cases-02/session.json"


class PairedWebService(OperationWebService):
    """Reuse origin/token checks and common assets, with a separate review store."""

    def __init__(self, *, data_dir: Path, session_path: Path, reviewer_id: str,
                 evidence_bundle: Path | None = None, evidence_session: Path | None = None,
                 evidence_policy: Path | None = None) -> None:
        self.store = PairedReviewStore(
            data_dir=data_dir, session_path=session_path, reviewer_id=reviewer_id,
        )
        self.asset_root = Path(__file__).resolve().parent
        self.session_token = secrets.token_urlsafe(32)
        self.allowed_hosts: set[str] = set()
        self.allowed_origins: set[str] = set()
        if bool(evidence_bundle) != bool(evidence_session):
            raise ReviewError("证据审核包和独立会话路径须同时提供。")
        if evidence_policy is not None and evidence_bundle is None:
            raise ReviewError("独立规则修订需要已配置证据审核包及会话。")
        self.evidence_store = EvidenceReviewStore(
            bundle_path=evidence_bundle, session_path=evidence_session, reviewer_id=reviewer_id,
            policy_path=evidence_policy,
        ) if evidence_bundle else None

    def evidence_bootstrap(self) -> dict:
        if self.evidence_store is None:
            raise ReviewError("证据审核模式尚未配置。")
        return {**self.evidence_store.bootstrap(), "session_token": self.session_token}


class PairedRequestHandler(OperationRequestHandler):
    service: PairedWebService

    def _asset(self, name: str) -> None:
        evidence_assets = {"evidence.html": "text/html", "evidence.js": "text/javascript",
                           "evidence_core.js": "text/javascript", "evidence.css": "text/css"}
        if name not in evidence_assets:
            return super()._asset(name)
        path = self.service.asset_root / name
        if not path.is_file() or path.is_symlink():
            self._error(HTTPStatus.NOT_FOUND, "review UI asset missing")
            return
        self._send(HTTPStatus.OK, path.read_bytes(), evidence_assets[name] + "; charset=utf-8")

    def log_message(self, format_string: str, *args: object) -> None:
        sys.stderr.write("[paired-human-review] " + (format_string % args) + "\n")

    def _read_payload(self) -> dict:
        # The paired form has more text fields than the lexicon operation form.
        if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
            raise ReviewWebError("exactly one Content-Length is required")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError as exc:
            raise ReviewWebError("invalid Content-Length") from exc
        if not 0 < length <= 128 * 1024:
            raise ReviewWebError("request body size is invalid")
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            raise ReviewWebError("request must use application/json")
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewWebError("invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ReviewWebError("request JSON must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        self._start_observation()
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Host not allowed")
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            if path in {"/", "/index.html"}:
                self._asset("index.html")
            elif path in {"/evidence", "/evidence/"} and self.service.evidence_store:
                self._asset("evidence.html")
            elif path in {"/evidence/evidence.js", "/evidence/evidence.css", "/evidence/evidence_core.js"} and self.service.evidence_store:
                self._asset(path.rsplit("/", 1)[1])
            elif path in {"/app.js", "/core.js", "/styles.css", "/review-base.css", "/review-core.js"}:
                self._asset(path[1:])
            elif path == "/api/health":
                self._json(HTTPStatus.OK, {"status": "ok", "stage": "paired-human-review"})
            elif path == "/api/bootstrap":
                self._json(HTTPStatus.OK, self.service.bootstrap())
            elif path == "/api/evidence/bootstrap":
                self._json(HTTPStatus.OK, self.service.evidence_bootstrap())
            elif path.startswith("/api/evidence/items/") and self.service.evidence_store:
                self._json(HTTPStatus.OK, self.service.evidence_store.item_state(path.removeprefix("/api/evidence/items/")))
            elif path.startswith("/api/items/"):
                self._json(HTTPStatus.OK, self.service.store.item_state(path.removeprefix("/api/items/")))
            elif path.startswith("/api/prompt/"):
                query = parse_qs(parsed.query)
                self._json(HTTPStatus.OK, self.service.store.prompt(
                    path.removeprefix("/api/prompt/"),
                    query.get("condition", [""])[0], query.get("task", [""])[0],
                ))
            elif path == "/favicon.ico":
                self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            else:
                self._error(HTTPStatus.NOT_FOUND, "页面或案例不存在。")
        except (ReviewError, KeyError, ValueError) as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except OSError:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "无法读取审核记录。")

    def do_POST(self) -> None:  # noqa: N802
        self._start_observation()
        if not self._host_allowed() or not self._origin_allowed():
            self.close_connection = True
            self._error(HTTPStatus.FORBIDDEN, "request origin not allowed")
            return
        try:
            payload = self._read_payload()
            if not self.service.authorized(payload):
                self._error(HTTPStatus.FORBIDDEN, "页面会话已失效，请刷新后重试。")
                return
            path = urlsplit(self.path).path
            common = {"session_token", "expected_revision"}
            if not isinstance(payload.get("expected_revision"), str):
                raise ReviewError("请求缺少记录版本。")
            if path.startswith("/api/evidence/"):
                self._evidence_post(path, payload, common)
                return
            if path == "/api/export":
                if set(payload) != common | {"format"} or payload["format"] not in {"json", "csv"}:
                    raise ReviewError("导出格式不正确。")
                snapshot = self.service.store.snapshot(payload["expected_revision"])
                csv_format = payload["format"] == "csv"
                data = self.service.store.export_csv(snapshot) if csv_format else canonical_bytes(snapshot)
                self._send(
                    HTTPStatus.OK, data,
                    "text/csv; charset=utf-8" if csv_format else "application/json; charset=utf-8",
                    disposition='attachment; filename="paired-human-review.' + payload["format"] + '"',
                )
                return
            actions = {
                "/api/save": "save", "/api/reveal": "reveal",
                "/api/reveal-ai": "reveal_ai", "/api/confirm": "confirm",
                "/api/reopen": "reopen",
            }
            if path not in actions:
                self._error(HTTPStatus.NOT_FOUND, "API route not found")
                return
            action = actions[path]
            field = "reason" if action == "reopen" else "notes"
            if set(payload) != common | {"item_id", field} or not isinstance(payload.get("item_id"), str):
                raise ReviewError("请求字段不正确。")
            result = self.service.store.mutate(
                expected_revision=payload["expected_revision"], item_id=payload["item_id"],
                action=action, **{field: payload[field]},
            )
            result["bootstrap"]["session_token"] = self.service.session_token
            self._json(HTTPStatus.OK, result)
        except ReviewConflict as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except (ReviewError, ReviewWebError, TypeError, ValueError) as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except OSError:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "保存失败，当前页面中的草稿仍保留。")

    def _evidence_post(self, path: str, payload: dict, common: set[str]) -> None:
        store = self.service.evidence_store
        if store is None:
            raise ReviewError("证据审核模式尚未配置。")
        action = path.removeprefix("/api/evidence/").replace("-", "_")
        if action == "export":
            if set(payload) != common | {"format"} or payload["format"] not in {"json", "csv"}:
                raise ReviewError("导出格式不正确。")
            snapshot = store.snapshot(payload["expected_revision"])
            csv_format = payload["format"] == "csv"
            self._send(HTTPStatus.OK, store.export_csv(snapshot) if csv_format else canonical_bytes(snapshot),
                       "text/csv; charset=utf-8" if csv_format else "application/json; charset=utf-8",
                       disposition='attachment; filename="evidence-review.' + payload["format"] + '"')
            return
        fields = {
            "save_object": {"object_id", "object_version", "values"},
            "confirm_object": {"object_id", "object_version", "values"},
            "reopen_object": {"object_id", "object_version", "reason"},
            "confirm_batch": {"objects"}, "reveal": set(),
            "save_assessment": {"assessment"}, "confirm": {"assessment"}, "reopen": {"reason"},
            "save_exposure": {"prior_exposure"},
        }
        optional = {"prior_exposure"} if "prior_exposure" in payload else set()
        if action not in fields or set(payload) != common | {"item_id"} | fields[action] | optional or not isinstance(payload.get("item_id"), str):
            raise ReviewError("证据审核请求字段不正确。")
        result = store.mutate(expected_revision=payload["expected_revision"], item_id=payload["item_id"],
                              action=action, **{k: payload[k] for k in fields[action] | optional})
        result["bootstrap"]["session_token"] = self.service.session_token
        self._json(HTTPStatus.OK, result)


def create_server(
    *, data_dir: Path, session_path: Path, reviewer_id: str,
    host: str = "127.0.0.1", port: int = 8772, public_origin: str | None = None,
    evidence_bundle: Path | None = None, evidence_session: Path | None = None,
    evidence_policy: Path | None = None,
) -> ThreadingHTTPServer:
    _loopback_host(host)
    if not 0 <= port <= 65535:
        raise ReviewWebError("port must be in 0..65535")
    service = PairedWebService(data_dir=data_dir, session_path=session_path, reviewer_id=reviewer_id,
                               evidence_bundle=evidence_bundle, evidence_session=evidence_session, evidence_policy=evidence_policy)
    handler = type("BoundPairedReviewHandler", (PairedRequestHandler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    service.configure_network(int(server.server_address[1]), public_origin=public_origin)
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description="NoCat 配对案例人工复核工作台")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--session-file", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--reviewer-id", default="liaozijie")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--public-origin", default=None)
    parser.add_argument("--evidence-bundle", type=Path)
    parser.add_argument("--evidence-session", type=Path)
    parser.add_argument("--evidence-policy", type=Path)
    args = parser.parse_args()
    authority_marker = args.session_file.with_name(args.session_file.name + ".remote-authority.json")
    if authority_marker.exists() or authority_marker.is_symlink():
        parser.exit(
            2,
            "审核记录已迁移至 digitalocean-sgp，本地旧会话已停止接收写入。\n"
            "请访问 https://hsd.fenglin.pro/，或运行 "
            "bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh start "
            "后打开 http://127.0.0.1:8772/。\n"
            f"迁移标记：{authority_marker}\n",
        )
    server = create_server(
        data_dir=args.data_dir, session_path=args.session_file, reviewer_id=args.reviewer_id,
        host=args.host, port=args.port, public_origin=args.public_origin,
        evidence_bundle=args.evidence_bundle, evidence_session=args.evidence_session,
        evidence_policy=args.evidence_policy,
    )
    print(json.dumps({
        "url": "http://127.0.0.1:" + str(server.server_address[1]) + "/",
        "session_file": str(args.session_file.resolve()), "reviewer_id": args.reviewer_id,
    }, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
