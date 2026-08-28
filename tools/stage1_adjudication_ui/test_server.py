from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TOOL_ROOT.parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

import server as review_server


class ReviewServiceTests(unittest.TestCase):
    @staticmethod
    def _write_pristine_copy(source: Path, destination: Path) -> None:
        rows = [json.loads(line) for line in source.read_text().splitlines() if line]
        for row in rows:
            row.update(
                {
                    "decision": "",
                    "edits": [],
                    "reason_code": "",
                    "reason": "",
                    "reviewed_at": "",
                }
            )
        destination.write_bytes(
            b"".join(review_server._canonical_json(row) + b"\n" for row in rows)
        )

    @staticmethod
    def _write_locator_ref(
        destination: Path, *, artifact_kind: str, target: Path
    ) -> None:
        payload_manifest = target / "payload_manifest.json"
        destination.write_bytes(
            review_server._canonical_json(
                {
                    "schema_version": "stage1-locator-ref/v1",
                    "artifact_kind": artifact_kind,
                    "artifact_id": target.name,
                    "target_path": str(target.resolve()),
                    "payload_manifest_sha256": hashlib.sha256(
                        payload_manifest.read_bytes()
                    ).hexdigest(),
                }
            )
            + b"\n"
        )

    @classmethod
    def _build_service(
        cls, temporary_root: Path, *, group_file: Path, field_file: Path
    ) -> review_server.ReviewService:
        stage1_root = REPOSITORY_ROOT / "exps/causal_context/stage1_p0"
        first_group_row = json.loads(
            next(
                line
                for line in group_file.read_text().splitlines()
                if line.strip()
            )
        )
        audit_target = stage1_root / "data_audits" / first_group_row["data_audit_id"]
        review_targets = sorted(
            path
            for path in (stage1_root / "data_blind_reviews").iterdir()
            if path.is_dir()
        )
        if len(review_targets) != 1:
            raise AssertionError("expected exactly one committed data blind-review target")
        audit_ref = temporary_root / "data_audit_ref.json"
        review_ref = temporary_root / "data_blind_review_ref.json"
        cls._write_locator_ref(audit_ref, artifact_kind="data-audit", target=audit_target)
        cls._write_locator_ref(
            review_ref,
            artifact_kind="data-blind-review",
            target=review_targets[0],
        )
        return review_server.ReviewService(
            REPOSITORY_ROOT,
            group_adjudication_file=group_file,
            field_adjudication_file=field_file,
            audit_ref=audit_ref,
            group_review_ref=review_ref,
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary.name)
        source_root = (
            REPOSITORY_ROOT / "exps/causal_context/stage1_p0/review_inputs"
        )
        self.group_file = temporary_root / "group_hate_human_adjudication.jsonl"
        self.field_file = temporary_root / "field_type_human_adjudication.jsonl"
        self._write_pristine_copy(
            source_root / "group_hate_human_adjudication.jsonl", self.group_file
        )
        self._write_pristine_copy(
            source_root / "field_type_human_adjudication.jsonl", self.field_file
        )
        self.service = self._build_service(
            temporary_root,
            group_file=self.group_file,
            field_file=self.field_file,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _scope(state: dict, name: str) -> dict:
        return next(scope for scope in state["scopes"] if scope["scope"] == name)

    def _payload(
        self,
        *,
        state: dict,
        scope_name: str,
        issue: dict,
        decision: str,
        edits: list[dict],
        reason_code: str,
    ) -> dict:
        scope = self._scope(state, scope_name)
        return {
            "session_token": state["session_token"],
            "scope": scope_name,
            "alias": issue["alias"],
            "revision": scope["revision"],
            "decision": decision,
            "edits": edits,
            "reason_code": reason_code,
            "reason": "依据冻结规则核对原文与当前四元组后完成裁决。",
        }

    @staticmethod
    def _export_arguments(state: dict) -> dict:
        return {
            "expected_workspace_fingerprint": state["workspace_fingerprint"],
            "expected_revisions": {
                scope["scope"]: scope["revision"] for scope in state["scopes"]
            },
        }

    def test_browser_state_is_exactly_24_rows_and_contains_no_hidden_fields(self) -> None:
        state = self.service.state()
        self.assertEqual(
            set(state),
            {
                "schema_version",
                "session_token",
                "workspace_fingerprint",
                "reviewer_id",
                "all_complete",
                "total_count",
                "complete_count",
                "scopes",
            },
        )
        self.assertEqual(state["total_count"], 24)
        self.assertEqual(state["complete_count"], 0)
        self.assertEqual(self._scope(state, "group-hate")["row_count"], 20)
        self.assertEqual(self._scope(state, "field-type")["row_count"], 4)
        rendered = json.dumps(state, ensure_ascii=False, sort_keys=True)
        real_issue_ids = {
            str(row["issue_id"])
            for workspace in self.service.workspaces.values()
            for row in workspace.rows
        }
        for issue_id in real_issue_ids:
            self.assertNotIn(issue_id, rendered)
        for scope in state["scopes"]:
            self.assertEqual(
                set(scope),
                {
                    "scope",
                    "title",
                    "revision",
                    "row_count",
                    "complete_count",
                    "issues",
                },
            )
            for issue in scope["issues"]:
                self.assertEqual(
                    set(issue),
                    {
                        "alias",
                        "ordinal",
                        "issue_kind",
                        "issue_code",
                        "accept_allowed",
                        "allowed_edit_paths",
                        "context",
                        "reason_codes",
                        "current",
                    },
                )
                self.assertRegex(issue["alias"], r"^H-[0-9a-f]{12}$")
                self.assertNotIn("issue_id", issue)
                expected_context = (
                    {"content", "tuple_before"}
                    if issue["issue_kind"] == "group-hate"
                    else {
                        "content",
                        "tuple_before",
                        "field",
                        "observed_type",
                        "allowed_correction_types",
                    }
                )
                self.assertEqual(set(issue["context"]), expected_context)
                self.assertEqual(
                    set(issue["current"]),
                    {"decision", "edits", "reason_code", "reason", "reviewed_at"},
                )

    def test_accept_commit_is_atomic_and_stale_revision_is_rejected(self) -> None:
        state = self.service.state()
        group = self._scope(state, "group-hate")
        issue = next(item for item in group["issues"] if item["accept_allowed"])
        payload = self._payload(
            state=state,
            scope_name="group-hate",
            issue=issue,
            decision="accepted",
            edits=[],
            reason_code="valid-independent-label-combination",
        )
        updated = self.service.commit(payload)
        self.assertEqual(self._scope(updated, "group-hate")["complete_count"], 1)
        with self.assertRaises(review_server.WebReviewConflict):
            self.service.commit(payload)
        refreshed_group = self._scope(updated, "group-hate")
        refreshed_issue = next(
            item for item in refreshed_group["issues"] if item["alias"] == issue["alias"]
        )
        rewrite = self._payload(
            state=updated,
            scope_name="group-hate",
            issue=refreshed_issue,
            decision="accepted",
            edits=[],
            reason_code="valid-independent-label-combination",
        )
        with self.assertRaises(review_server.WebReviewConflict):
            self.service.commit(rewrite)
        rows = [json.loads(line) for line in self.group_file.read_text().splitlines()]
        completed = [row for row in rows if row["decision"]]
        self.assertEqual(len(completed), 1)
        self.assertRegex(
            completed[0]["reviewed_at"],
            r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+00:00$",
        )

    def test_direct_service_commit_rejects_wrong_session_token(self) -> None:
        state = self.service.state()
        group = self._scope(state, "group-hate")
        issue = next(item for item in group["issues"] if item["accept_allowed"])
        payload = self._payload(
            state=state,
            scope_name="group-hate",
            issue=issue,
            decision="accepted",
            edits=[],
            reason_code="valid-independent-label-combination",
        )
        payload["session_token"] = "wrong"
        with self.assertRaises(review_server.WebReviewValidationError):
            self.service.commit(payload)
        self.assertEqual(self._scope(self.service.state(), "group-hate")["complete_count"], 0)

    def test_state_rejects_changed_immutable_packet_identity(self) -> None:
        config = self.service.configs["group-hate"]
        packet_copy = Path(self.temporary.name) / "group-packet-copy.jsonl"
        packet_bytes = config.packet_file.read_bytes()
        packet_copy.write_bytes(packet_bytes.replace(b"\n", b" \n", 1))
        self.service.configs["group-hate"] = replace(
            config, packet_file=packet_copy
        )
        with self.assertRaises(review_server.WebReviewConflict):
            self.service.state()

    def test_authoritative_validator_rejects_semantically_invalid_group_edit(self) -> None:
        state = self.service.state()
        group = self._scope(state, "group-hate")
        issue = next(
            item for item in group["issues"] if item["issue_code"] == "hateful-null-sentinel"
        )
        payload = self._payload(
            state=state,
            scope_name="group-hate",
            issue=issue,
            decision="corrected",
            edits=[
                {
                    "json_pointer": issue["allowed_edit_paths"][0],
                    "value": "non-hate",
                }
            ],
            reason_code="replace-legacy-null",
        )
        with self.assertRaises(review_server.WebReviewValidationError):
            self.service.commit(payload)
        self.assertEqual(self._scope(self.service.state(), "group-hate")["complete_count"], 0)

    def test_field_edit_requires_value_and_reason_code_consistency(self) -> None:
        state = self.service.state()
        field_scope = self._scope(state, "field-type")
        issue = field_scope["issues"][0]
        pointer = issue["allowed_edit_paths"][0]
        before_value = issue["context"]["tuple_before"][pointer.rsplit("/", 1)[-1]]
        invalid = self._payload(
            state=state,
            scope_name="field-type",
            issue=issue,
            decision="corrected",
            edits=[{"json_pointer": pointer, "value": str(before_value)}],
            reason_code="set-explicit-null",
        )
        with self.assertRaises(review_server.WebReviewValidationError):
            self.service.commit(invalid)
        valid = dict(invalid)
        valid["reason_code"] = "coerce-numeric-annotation"
        updated = self.service.commit(valid)
        self.assertEqual(self._scope(updated, "field-type")["complete_count"], 1)

    def _complete_every_row(self) -> dict:
        state = self.service.state()
        while not state["all_complete"]:
            pending: tuple[str, dict] | None = None
            for scope in state["scopes"]:
                for issue in scope["issues"]:
                    if not issue["current"]["decision"]:
                        pending = (scope["scope"], issue)
                        break
                if pending is not None:
                    break
            assert pending is not None
            scope_name, issue = pending
            if issue["accept_allowed"]:
                decision = "accepted"
                edits: list[dict] = []
                reason_code = "valid-independent-label-combination"
            elif issue["issue_code"] == "hateful-null-sentinel":
                decision = "corrected"
                edits = [
                    {
                        "json_pointer": issue["allowed_edit_paths"][0],
                        "value": "hate",
                    }
                ]
                reason_code = "replace-legacy-null"
            elif issue["issue_code"] == "group-hate-conflict":
                decision = "corrected"
                group_path = next(
                    path for path in issue["allowed_edit_paths"] if path.endswith("targeted_group")
                )
                edits = [{"json_pointer": group_path, "value": "others"}]
                reason_code = "resolve-group-hate-conflict"
            else:
                decision = "corrected"
                pointer = issue["allowed_edit_paths"][0]
                field = pointer.rsplit("/", 1)[-1]
                before = issue["context"]["tuple_before"][field]
                edits = [{"json_pointer": pointer, "value": str(before)}]
                reason_code = "coerce-numeric-annotation"
            state = self.service.commit(
                self._payload(
                    state=state,
                    scope_name=scope_name,
                    issue=issue,
                    decision=decision,
                    edits=edits,
                    reason_code=reason_code,
                )
            )
        return state

    def test_export_is_blocked_until_complete_then_contains_valid_20_and_4_rows(self) -> None:
        initial = self.service.state()
        with self.assertRaises(review_server.WebReviewConflict):
            self.service.export_zip(**self._export_arguments(initial))
        state = self._complete_every_row()
        self.assertTrue(state["all_complete"])
        archive_bytes = self.service.export_zip(**self._export_arguments(state))
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {
                    "group_hate_human_adjudication.jsonl",
                    "field_type_human_adjudication.jsonl",
                    "manifest.json",
                    "NEXT_STEPS.txt",
                },
            )
            group_rows = archive.read("group_hate_human_adjudication.jsonl").splitlines()
            field_rows = archive.read("field_type_human_adjudication.jsonl").splitlines()
            self.assertEqual(len(group_rows), 20)
            self.assertEqual(len(field_rows), 4)
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(
                manifest["schema_version"], review_server.EXPORT_SCHEMA_VERSION
            )
            for name in (
                "group_hate_human_adjudication.jsonl",
                "field_type_human_adjudication.jsonl",
            ):
                self.assertEqual(
                    manifest["files"][name]["sha256"],
                    hashlib.sha256(archive.read(name)).hexdigest(),
                )

        rows = [json.loads(line) for line in self.group_file.read_text().splitlines()]
        rows[0]["reason"] += "（并发合法修订）"
        self.group_file.write_bytes(
            b"".join(review_server._canonical_json(row) + b"\n" for row in rows)
        )
        with self.assertRaises(review_server.WebReviewConflict):
            self.service.export_zip(**self._export_arguments(state))


class HttpBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary.name)
        source_root = (
            REPOSITORY_ROOT / "exps/causal_context/stage1_p0/review_inputs"
        )
        group_file = temporary_root / "group.jsonl"
        field_file = temporary_root / "field.jsonl"
        ReviewServiceTests._write_pristine_copy(
            source_root / "group_hate_human_adjudication.jsonl", group_file
        )
        ReviewServiceTests._write_pristine_copy(
            source_root / "field_type_human_adjudication.jsonl", field_file
        )
        service = ReviewServiceTests._build_service(
            temporary_root,
            group_file=group_file,
            field_file=field_file,
        )
        handler = type(
            "TestReviewRequestHandler",
            (review_server.ReviewRequestHandler,),
            {"service": service},
        )
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.port = int(self.httpd.server_address[1])
        service.configure_network(port=self.port)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        self.temporary.cleanup()

    def test_state_and_static_assets_are_local_csp_protected(self) -> None:
        base = f"http://127.0.0.1:{self.port}"
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
            self.assertIn("text/html", response.headers["Content-Type"])
            self.assertIn("camera=()", response.headers["Permissions-Policy"])
        with urllib.request.urlopen(base + "/app.js", timeout=5) as response:
            script = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertNotIn("innerHTML", script)
            self.assertNotIn("http://", script)
            self.assertNotIn("https://", script)
        with urllib.request.urlopen(base + "/api/state", timeout=5) as response:
            state_payload = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertNotIn("dissue:", state_payload)
            self.assertNotIn("queue_reasons", state_payload)
            state = json.loads(state_payload)
            self.assertEqual(state["total_count"], 24)
        proxied_state = urllib.request.Request(
            base + "/api/state", headers={"Host": "localhost:8210"}
        )
        with urllib.request.urlopen(proxied_state, timeout=5) as response:
            self.assertEqual(json.loads(response.read())["total_count"], 24)
        export_payload = {
            "session_token": state["session_token"],
            "workspace_fingerprint": state["workspace_fingerprint"],
            "revisions": {
                scope["scope"]: scope["revision"] for scope in state["scopes"]
            },
        }
        incomplete_export = urllib.request.Request(
            base + "/api/export",
            data=json.dumps(export_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Host": "localhost:8210",
                "Origin": "http://localhost:8210",
            },
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(incomplete_export, timeout=5)
        self.assertEqual(caught.exception.code, 409)
        request = urllib.request.Request(
            base + "/api/export",
            data=b'{"session_token":"wrong"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 403)

        hostile_origin = urllib.request.Request(
            base + "/api/export",
            data=json.dumps(export_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": "https://attacker.invalid",
            },
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(hostile_origin, timeout=5)
        self.assertEqual(caught.exception.code, 403)

        hostile_host = urllib.request.Request(
            base + "/api/state", headers={"Host": "attacker.invalid"}
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(hostile_host, timeout=5)
        self.assertEqual(caught.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
