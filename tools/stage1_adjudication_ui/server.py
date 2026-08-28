#!/usr/bin/env python3
"""Serve the local Stage 1 P0 human-adjudication interface.

The browser receives only the same allowlisted, model-output-free projection as
the frozen terminal reviewer.  Confirmed decisions are checked by the existing
authoritative validator and atomically written to the two human-review JSONL
files.  This tool never merges the sealed automatic rows and never signs a
reviewer declaration.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import ipaddress
import json
import secrets
import sys
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.training_artifacts import canonical_json_bytes, sha256_file  # noqa: E402
from review.human_adjudication import (  # noqa: E402
    FROZEN_REVIEWER_ID,
    HumanAdjudicationError,
    HumanAdjudicationIncomplete,
    HumanReviewWorkspace,
    _blind_alias,
    _blind_order_key,
    _terminal_safe,
    _validate_one_completed,
    _write_rows_atomic,
    load_human_review_workspace,
    validate_human_review_workspace,
    workspace_status,
)


STATE_SCHEMA_VERSION = "stage1-adjudication-web-state/v1"
EXPORT_SCHEMA_VERSION = "stage1-adjudication-web-export/v1"
MAX_REQUEST_BYTES = 64 * 1024
ASSET_NAMES = frozenset({"index.html", "app.js", "styles.css"})
WEB_CONTEXT_FIELDS = {
    "group-hate": ("content", "tuple_before"),
    "field-type": (
        "content",
        "tuple_before",
        "field",
        "observed_type",
        "allowed_correction_types",
    ),
}


class WebReviewError(RuntimeError):
    """Safe, user-facing review error."""


class WebReviewConflict(WebReviewError):
    """Raised when another tab or process changed an adjudication file."""


class WebReviewValidationError(WebReviewError):
    """Raised when a proposed decision violates the web input contract."""


@dataclass(frozen=True)
class WorkspaceConfig:
    scope: str
    title: str
    packet_file: Path
    adjudication_file: Path
    review_ref: Path | None


def _safe_projection(value: Any) -> Any:
    """Escape controls/bidi markers while preserving the JSON value shape."""

    if isinstance(value, str):
        return _terminal_safe(value)
    if isinstance(value, list):
        return [_safe_projection(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_projection(item) for key, item in value.items()}
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _loopback_authority(value: str) -> bool:
    """Accept only localhost or a literal loopback IP, with any valid port."""

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
    if hostname.lower() == "localhost":
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
    if parsed.scheme != "http" or parsed.path or parsed.query or parsed.fragment:
        return False
    return _loopback_authority(parsed.netloc)


class ReviewService:
    """Own the two validated human-review workspaces for one local session."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        group_adjudication_file: Path | None = None,
        field_adjudication_file: Path | None = None,
        audit_ref: Path | None = None,
        group_review_ref: Path | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve(strict=True)
        self.asset_root = Path(__file__).resolve().parent
        review_root = self.workspace_root / "exps/causal_context/stage1_p0/review_inputs"
        refs_root = self.workspace_root / "exps/causal_context/stage1_p0/refs"
        self.audit_ref = (
            Path(audit_ref) if audit_ref is not None else refs_root / "data_audit_ref.json"
        )
        self.configs: dict[str, WorkspaceConfig] = {
            "group-hate": WorkspaceConfig(
                scope="group-hate",
                title="Group–hate 二轮盲审",
                packet_file=review_root / "group_hate_human_packets.jsonl",
                adjudication_file=(
                    Path(group_adjudication_file)
                    if group_adjudication_file is not None
                    else review_root / "group_hate_human_adjudication.jsonl"
                ),
                review_ref=(
                    Path(group_review_ref)
                    if group_review_ref is not None
                    else refs_root / "data_blind_review_ref.json"
                ),
            ),
            "field-type": WorkspaceConfig(
                scope="field-type",
                title="字段类型复核",
                packet_file=review_root / "field_type_human_packets.jsonl",
                adjudication_file=(
                    Path(field_adjudication_file)
                    if field_adjudication_file is not None
                    else review_root / "field_type_human_adjudication.jsonl"
                ),
                review_ref=None,
            ),
        }
        self._lock = threading.RLock()
        self.session_token = secrets.token_urlsafe(32)
        self.allowed_hosts: set[str] = set()
        self.allowed_origins: set[str] = set()
        self.workspaces = {
            scope: self._load_workspace(config)
            for scope, config in self.configs.items()
        }
        self.workspace_fingerprint = self._workspace_fingerprint()
        self._assert_frame_contract()

    def _load_workspace(self, config: WorkspaceConfig) -> HumanReviewWorkspace:
        arguments: dict[str, Any] = {
            "audit_ref": self.audit_ref,
            "packet_file": config.packet_file,
            "adjudication_file": config.adjudication_file,
            "workspace_root": self.workspace_root,
            "reviewer_id": FROZEN_REVIEWER_ID,
        }
        if config.review_ref is not None:
            arguments["review_ref"] = config.review_ref
        return load_human_review_workspace(**arguments)

    def _workspace_fingerprint(self) -> str:
        identity = {
            "schema_version": STATE_SCHEMA_VERSION,
            "reviewer_id": FROZEN_REVIEWER_ID,
            "audit_ref_sha256": sha256_file(self.audit_ref),
            "scopes": {
                scope: {
                    "packet_sha256": sha256_file(config.packet_file),
                    "review_ref_sha256": (
                        sha256_file(config.review_ref)
                        if config.review_ref is not None
                        else None
                    ),
                }
                for scope, config in sorted(self.configs.items())
            },
        }
        return hashlib.sha256(_canonical_json(identity)).hexdigest()

    def _assert_frame_contract(
        self, workspaces: Mapping[str, HumanReviewWorkspace] | None = None
    ) -> None:
        selected = self.workspaces if workspaces is None else workspaces
        expected = {"group-hate": 20, "field-type": 4}
        actual = {scope: len(workspace.rows) for scope, workspace in selected.items()}
        if actual != expected:
            raise WebReviewError(
                "裁决 frame 与冻结 P0 交接不一致；应为 group–hate 20 条、field-type 4 条。"
            )
        for scope, workspace in selected.items():
            aliases = [
                _blind_alias(str(workspace.locator["artifact_id"]), str(row["issue_id"]))
                for row in workspace.rows
            ]
            if len(aliases) != len(set(aliases)):
                raise WebReviewError(f"{scope} 的盲化别名发生碰撞。")
            for issue in workspace.issues_by_id.values():
                locations = issue.get("locations")
                if not isinstance(locations, list) or len(locations) != 1:
                    raise WebReviewError("当前网页只支持每条异常对应一个冻结 source location。")

    def configure_network(self, *, port: int) -> None:
        self.allowed_hosts = {
            f"127.0.0.1:{port}",
            f"localhost:{port}",
            f"[::1]:{port}",
            "127.0.0.1",
            "localhost",
            "[::1]",
        }
        self.allowed_origins = {
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
            f"http://[::1]:{port}",
        }

    def _reload_all(self) -> None:
        """Revalidate every frozen dependency and both live adjudication files."""

        refreshed = {
            scope: self._load_workspace(config)
            for scope, config in self.configs.items()
        }
        self._assert_frame_contract(refreshed)
        if self._workspace_fingerprint() != self.workspace_fingerprint:
            raise WebReviewConflict(
                "冻结 audit、packet 或 review ref 已变化；请停止当前会话并重新核对。"
            )
        self.workspaces = refreshed

    @staticmethod
    def _alias_maps(workspace: HumanReviewWorkspace) -> tuple[dict[str, str], dict[str, str]]:
        audit_id = str(workspace.locator["artifact_id"])
        issue_to_alias = {
            str(row["issue_id"]): _blind_alias(audit_id, str(row["issue_id"]))
            for row in workspace.rows
        }
        alias_to_issue = {alias: issue_id for issue_id, alias in issue_to_alias.items()}
        return issue_to_alias, alias_to_issue

    @staticmethod
    def _safe_context(issue: Mapping[str, Any]) -> dict[str, Any]:
        issue_kind = str(issue.get("issue_kind"))
        allowed = WEB_CONTEXT_FIELDS.get(issue_kind)
        context = issue.get("review_context")
        if allowed is None or not isinstance(context, dict):
            raise WebReviewError("该异常没有安全的浏览器展示投影。")
        return {
            field: _safe_projection(context[field])
            for field in allowed
            if field in context
        }

    def _scope_state(self, scope: str) -> dict[str, Any]:
        workspace = self.workspaces[scope]
        config = self.configs[scope]
        status = workspace_status(workspace)
        issue_to_alias, _ = self._alias_maps(workspace)
        audit_id = str(workspace.locator["artifact_id"])
        row_by_id = {str(row["issue_id"]): row for row in workspace.rows}
        ordered_issue_ids = sorted(
            row_by_id,
            key=lambda issue_id: _blind_order_key(audit_id, issue_id),
        )
        issues: list[dict[str, Any]] = []
        for ordinal, issue_id in enumerate(ordered_issue_ids, 1):
            issue = workspace.issues_by_id[issue_id]
            row = row_by_id[issue_id]
            reason_codes = (
                workspace.rubric_meta.get("reason_codes", {}).get(
                    str(issue.get("issue_code")), {}
                )
            )
            current_edits = [
                {
                    "json_pointer": str(edit["json_pointer"]),
                    "value": _safe_projection(edit.get("value")),
                }
                for edit in row.get("edits", [])
                if isinstance(edit, dict)
            ]
            issues.append(
                {
                    "alias": issue_to_alias[issue_id],
                    "ordinal": ordinal,
                    "issue_kind": str(issue["issue_kind"]),
                    "issue_code": str(issue["issue_code"]),
                    "accept_allowed": issue.get("accept_allowed") is True,
                    "allowed_edit_paths": [
                        str(path) for path in issue.get("allowed_edit_paths", [])
                    ],
                    "context": self._safe_context(issue),
                    "reason_codes": {
                        "accepted": list(reason_codes.get("accepted", [])),
                        "corrected": list(reason_codes.get("corrected", [])),
                    },
                    "current": {
                        "decision": str(row.get("decision", "")),
                        "edits": current_edits,
                        "reason_code": str(row.get("reason_code", "")),
                        "reason": _safe_projection(str(row.get("reason", ""))),
                        "reviewed_at": str(row.get("reviewed_at", "")),
                    },
                }
            )
        return {
            "scope": scope,
            "title": config.title,
            "revision": workspace.adjudication_sha256,
            "row_count": int(status["row_count"]),
            "complete_count": int(status["complete_count"]),
            "issues": issues,
        }

    def _state_from_loaded(self) -> dict[str, Any]:
        scopes = [self._scope_state(scope) for scope in ("group-hate", "field-type")]
        total = sum(int(scope["row_count"]) for scope in scopes)
        complete = sum(int(scope["complete_count"]) for scope in scopes)
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "session_token": self.session_token,
            "workspace_fingerprint": self.workspace_fingerprint,
            "reviewer_id": FROZEN_REVIEWER_ID,
            "all_complete": complete == total,
            "total_count": total,
            "complete_count": complete,
            "scopes": scopes,
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            self._reload_all()
            return self._state_from_loaded()

    @staticmethod
    def _validate_reason(reason: Any) -> str:
        if not isinstance(reason, str):
            raise WebReviewValidationError("裁决理由必须是文本。")
        stripped = reason.strip()
        if not stripped:
            raise WebReviewValidationError("裁决理由不能为空。")
        if len(stripped) > 2000:
            raise WebReviewValidationError("裁决理由不能超过 2000 个字符。")
        if _terminal_safe(stripped) != stripped:
            raise WebReviewValidationError("裁决理由不能包含控制字符或双向覆盖字符。")
        return stripped

    def commit(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        expected_keys = {
            "session_token",
            "scope",
            "alias",
            "revision",
            "decision",
            "edits",
            "reason_code",
            "reason",
        }
        if set(payload) != expected_keys:
            raise WebReviewValidationError("提交字段集合不合法。")
        token = payload.get("session_token")
        if not isinstance(token, str) or not secrets.compare_digest(
            token, self.session_token
        ):
            raise WebReviewValidationError("会话令牌无效。")
        scope = payload.get("scope")
        if scope not in self.configs:
            raise WebReviewValidationError("未知裁决范围。")
        with self._lock:
            self._reload_all()
            workspace = self.workspaces[str(scope)]
            if payload.get("revision") != workspace.adjudication_sha256:
                raise WebReviewConflict("裁决文件已在另一标签页或进程中变化，请重新加载。")
            _, alias_to_issue = self._alias_maps(workspace)
            alias = payload.get("alias")
            if not isinstance(alias, str) or alias not in alias_to_issue:
                raise WebReviewValidationError("未知盲化编号。")
            issue_id = alias_to_issue[alias]
            issue = workspace.issues_by_id[issue_id]
            decision = payload.get("decision")
            if decision not in {"accepted", "corrected"}:
                raise WebReviewValidationError("必须选择接受或修正。")
            if decision == "accepted" and issue.get("accept_allowed") is not True:
                raise WebReviewValidationError("该异常不能接受原标注。")
            reason_code = payload.get("reason_code")
            allowed_codes = (
                workspace.rubric_meta.get("reason_codes", {})
                .get(str(issue.get("issue_code")), {})
                .get(str(decision), [])
            )
            if not isinstance(reason_code, str) or reason_code not in allowed_codes:
                raise WebReviewValidationError("reason code 与异常类型或裁决不匹配。")
            reason = self._validate_reason(payload.get("reason"))
            raw_edits = payload.get("edits")
            if not isinstance(raw_edits, list):
                raise WebReviewValidationError("edits 必须是列表。")
            if decision == "accepted" and raw_edits:
                raise WebReviewValidationError("接受原标注时不能包含 edits。")
            if decision == "corrected" and not raw_edits:
                raise WebReviewValidationError("修正至少需要一个 edit。")
            allowed_paths = set(str(path) for path in issue.get("allowed_edit_paths", []))
            edits: list[dict[str, Any]] = []
            seen_paths: set[str] = set()
            for raw_edit in raw_edits:
                if not isinstance(raw_edit, dict) or set(raw_edit) != {
                    "json_pointer",
                    "value",
                }:
                    raise WebReviewValidationError("edit 字段集合不合法。")
                pointer = raw_edit.get("json_pointer")
                if not isinstance(pointer, str) or pointer not in allowed_paths:
                    raise WebReviewValidationError("edit 路径不在冻结 allowlist 中。")
                if pointer in seen_paths:
                    raise WebReviewValidationError("同一路径不能重复编辑。")
                seen_paths.add(pointer)
                edits.append(
                    {
                        "location_index": 0,
                        "op": "set",
                        "json_pointer": pointer,
                        "value": raw_edit.get("value"),
                    }
                )
            edits.sort(key=lambda edit: (edit["location_index"], edit["json_pointer"]))
            row_index = next(
                index
                for index, row in enumerate(workspace.rows)
                if str(row["issue_id"]) == issue_id
            )
            if workspace.rows[row_index].get("decision"):
                raise WebReviewConflict(
                    "该条裁决已经确认；已确认结果不可在网页中覆盖。"
                )
            candidate = dict(workspace.rows[row_index])
            candidate.update(
                {
                    "decision": decision,
                    "edits": edits,
                    "reason_code": reason_code,
                    "reason": reason,
                    "reviewed_at": _utc_now(),
                }
            )
            try:
                completed = _validate_one_completed(
                    candidate,
                    issue,
                    workspace.rubric_meta,
                    reviewer_id=workspace.reviewer_id,
                )
            except HumanAdjudicationError as exc:
                raise WebReviewValidationError(
                    "未通过冻结裁决规则；请检查修正值、前后字段和 reason code。"
                ) from exc
            previous = workspace.rows[row_index]
            workspace.rows[row_index] = completed
            try:
                workspace_status(workspace)
                _write_rows_atomic(workspace)
            except Exception as exc:
                workspace.rows[row_index] = previous
                try:
                    self.workspaces[str(scope)] = self._load_workspace(
                        self.configs[str(scope)]
                    )
                except (HumanAdjudicationError, OSError):
                    pass
                if isinstance(exc, HumanAdjudicationError):
                    if "concurrent" in str(exc).lower() or "changed" in str(exc).lower():
                        raise WebReviewConflict(
                            "裁决依赖或文件已变化，请重新加载后再提交。"
                        ) from exc
                    raise WebReviewValidationError(
                        "提交未通过冻结 workspace 校验。"
                    ) from exc
                raise WebReviewConflict(
                    "写入结果状态无法确认；请重新加载核对，切勿直接重复提交。"
                ) from exc
            return self._state_from_loaded()

    def export_zip(
        self,
        *,
        expected_workspace_fingerprint: str,
        expected_revisions: Mapping[str, Any],
    ) -> bytes:
        with self._lock:
            self._reload_all()
            if expected_workspace_fingerprint != self.workspace_fingerprint:
                raise WebReviewConflict("工作区身份已变化，请重新加载后再导出。")
            if set(expected_revisions) != set(self.configs) or any(
                not isinstance(value, str) for value in expected_revisions.values()
            ):
                raise WebReviewValidationError("导出 revision 集合不合法。")
            current_revisions = {
                scope: workspace.adjudication_sha256
                for scope, workspace in self.workspaces.items()
            }
            if dict(expected_revisions) != current_revisions:
                raise WebReviewConflict(
                    "裁决文件已在另一标签页或进程中变化，请重新加载后再导出。"
                )
            reports: dict[str, dict[str, Any]] = {}
            for scope, workspace in self.workspaces.items():
                try:
                    reports[scope] = validate_human_review_workspace(workspace)
                except HumanAdjudicationIncomplete as exc:
                    raise WebReviewConflict("24 条人工裁决全部完成后才能导出结果包。") from exc
            files = {
                "group_hate_human_adjudication.jsonl": b"".join(
                    _canonical_json(row) + b"\n"
                    for row in self.workspaces["group-hate"].rows
                ),
                "field_type_human_adjudication.jsonl": b"".join(
                    _canonical_json(row) + b"\n"
                    for row in self.workspaces["field-type"].rows
                ),
            }
            manifest = {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "workspace_fingerprint": self.workspace_fingerprint,
                "reviewer_id": FROZEN_REVIEWER_ID,
                "exported_at": _utc_now(),
                "files": {
                    name: {
                        "row_count": 20 if name.startswith("group_") else 4,
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                    for name, content in files.items()
                },
                "status": {
                    scope: {
                        "row_count": int(report["row_count"]),
                        "complete_count": int(report["complete_count"]),
                    }
                    for scope, report in sorted(reports.items())
                },
            }
            next_steps = (
                "Stage 1 P0 人工裁决结果包\n\n"
                "本包只包含 20 条 group–hate 与 4 条 field-type 人工结果。\n"
                "请将两个 JSONL 放回 review_inputs 对应路径，并按\n"
                "docs/research/experiment-plans/stage1-p0-human-review-handoff.md\n"
                "依次执行 validate、merge-human、merge-data-adjudication、\n"
                "prepare-data-declaration。核对声明 hash 后再由人类确认签署。\n"
            ).encode("utf-8")
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, content in files.items():
                    archive.writestr(name, content)
                archive.writestr("manifest.json", _canonical_json(manifest) + b"\n")
                archive.writestr("NEXT_STEPS.txt", next_steps)
            return buffer.getvalue()


class ReviewRequestHandler(BaseHTTPRequestHandler):
    """Fixed-route HTTP handler; no arbitrary filesystem paths are accepted."""

    protocol_version = "HTTP/1.1"
    service: ReviewService

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stderr.write("[stage1-review-ui] " + (format_string % args) + "\n")

    def _host_allowed(self) -> bool:
        host = self.headers.get("Host", "")
        allowed = host in self.service.allowed_hosts or _loopback_authority(host)
        if not allowed:
            safe_host = _terminal_safe(host[:256])
            print(
                f"[stage1-review-ui] rejected Host header: {safe_host!r}",
                file=sys.stderr,
            )
        return allowed

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return (
            origin is None
            or origin in self.service.allowed_origins
            or _loopback_origin(origin)
        )

    def _base_headers(self, *, content_type: str, content_length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
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

    def _send_bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        *,
        content_type: str,
        extra_headers: Mapping[str, str] | None = None,
        head_only: bool = False,
    ) -> None:
        self.send_response(status.value)
        self._base_headers(content_type=content_type, content_length=len(payload))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(payload)

    def _send_json(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
        self._send_bytes(
            status,
            _canonical_json(value),
            content_type="application/json; charset=utf-8",
        )

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message, "status": status.value})

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise WebReviewValidationError("请求缺少 Content-Length。")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise WebReviewValidationError("Content-Length 不合法。") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise WebReviewValidationError("请求体大小不合法。")
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            raise WebReviewValidationError("请求必须使用 application/json。")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebReviewValidationError("请求 JSON 不合法。") from exc
        if not isinstance(value, dict):
            raise WebReviewValidationError("请求 JSON 顶层必须是对象。")
        return value

    def _authorized(self, payload: Mapping[str, Any]) -> bool:
        token = payload.get("session_token")
        return isinstance(token, str) and secrets.compare_digest(
            token, self.service.session_token
        )

    def _serve_asset(self, name: str, *, head_only: bool = False) -> None:
        if name not in ASSET_NAMES:
            self._send_error_json(HTTPStatus.NOT_FOUND, "页面不存在。")
            return
        path = self.service.asset_root / name
        if path.is_symlink() or not path.is_file():
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "界面资源缺失。")
            return
        content_types = {
            "index.html": "text/html; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
        }
        self._send_bytes(
            HTTPStatus.OK,
            path.read_bytes(),
            content_type=content_types[name],
            head_only=head_only,
        )

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._send_error_json(HTTPStatus.FORBIDDEN, "Host 不允许。")
            return
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self._serve_asset("index.html", head_only=True)
        elif path == "/app.js":
            self._serve_asset("app.js", head_only=True)
        elif path == "/styles.css":
            self._serve_asset("styles.css", head_only=True)
        else:
            self._send_error_json(HTTPStatus.NOT_FOUND, "页面不存在。")

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._send_error_json(HTTPStatus.FORBIDDEN, "Host 不允许。")
            return
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self._serve_asset("index.html")
        elif path == "/app.js":
            self._serve_asset("app.js")
        elif path == "/styles.css":
            self._serve_asset("styles.css")
        elif path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
        elif path == "/api/state":
            try:
                self._send_json(HTTPStatus.OK, self.service.state())
            except (HumanAdjudicationError, OSError, WebReviewError):
                self._send_error_json(
                    HTTPStatus.CONFLICT,
                    "裁决 workspace 已变化或无效，请停止并在终端核对冻结依赖。",
                )
        elif path == "/favicon.ico":
            self._send_bytes(HTTPStatus.NO_CONTENT, b"", content_type="image/x-icon")
        else:
            self._send_error_json(HTTPStatus.NOT_FOUND, "页面不存在。")

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_allowed() or not self._origin_allowed():
            self._send_error_json(HTTPStatus.FORBIDDEN, "请求来源不允许。")
            return
        try:
            payload = self._read_json_body()
            if not self._authorized(payload):
                self._send_error_json(HTTPStatus.FORBIDDEN, "会话令牌无效。")
                return
            path = urlsplit(self.path).path
            if path == "/api/decision":
                state = self.service.commit(payload)
                self._send_json(HTTPStatus.OK, state)
            elif path == "/api/export":
                if set(payload) != {
                    "session_token",
                    "workspace_fingerprint",
                    "revisions",
                }:
                    raise WebReviewValidationError("导出请求字段集合不合法。")
                fingerprint = payload.get("workspace_fingerprint")
                revisions = payload.get("revisions")
                if not isinstance(fingerprint, str) or not isinstance(revisions, dict):
                    raise WebReviewValidationError("导出工作区身份或 revisions 不合法。")
                archive = self.service.export_zip(
                    expected_workspace_fingerprint=fingerprint,
                    expected_revisions=revisions,
                )
                self._send_bytes(
                    HTTPStatus.OK,
                    archive,
                    content_type="application/zip",
                    extra_headers={
                        "Content-Disposition": (
                            'attachment; filename="stage1-p0-human-adjudication-results.zip"'
                        )
                    },
                )
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "接口不存在。")
        except WebReviewConflict as exc:
            self._send_error_json(HTTPStatus.CONFLICT, str(exc))
        except WebReviewValidationError as exc:
            self._send_error_json(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except (HumanAdjudicationError, OSError):
            self._send_error_json(
                HTTPStatus.CONFLICT,
                "冻结 workspace 校验失败；本次请求没有写入，请在终端核对状态。",
            )
        except Exception as exc:  # pragma: no cover - fail closed at the HTTP edge.
            print(f"[stage1-review-ui] unexpected request failure: {type(exc).__name__}", file=sys.stderr)
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "本地服务发生未预期错误；本次请求没有写入。",
            )


def _loopback_host(value: str) -> str:
    if value == "localhost":
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("host 必须是 loopback IP 或 localhost") from exc
    if not address.is_loopback:
        raise argparse.ArgumentTypeError("为保护盲审数据，服务只能绑定 loopback")
    return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    root.add_argument("--host", type=_loopback_host, default="127.0.0.1")
    root.add_argument("--port", type=int, default=8765)
    root.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs and print a safe summary without starting the server.",
    )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        print("error: port 必须在 0..65535", file=sys.stderr)
        return 2
    try:
        service = ReviewService(args.workspace_root)
        state = service.state()
    except (HumanAdjudicationError, OSError, WebReviewError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.check:
        print(
            json.dumps(
                {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "total_count": state["total_count"],
                    "complete_count": state["complete_count"],
                    "all_complete": state["all_complete"],
                    "scopes": [
                        {
                            "scope": scope["scope"],
                            "row_count": scope["row_count"],
                            "complete_count": scope["complete_count"],
                        }
                        for scope in state["scopes"]
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    handler = type("BoundReviewRequestHandler", (ReviewRequestHandler,), {"service": service})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    port = int(server.server_address[1])
    service.configure_network(port=port)
    url = f"http://127.0.0.1:{port}/"
    print(
        json.dumps(
            {
                "url": url,
                "total_count": state["total_count"],
                "complete_count": state["complete_count"],
                "binding": "loopback-only",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
