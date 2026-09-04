from __future__ import annotations

import copy
import http.client
import importlib.util
import json
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parent / "server.py"
SPEC = importlib.util.spec_from_file_location("annotated_lexicon_operation_review_server_test", MODULE_PATH)
assert SPEC and SPEC.loader
SERVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVER
# Boundary tests can also run while the independently implemented store is not
# installed yet. Every service below explicitly receives a controlled fake.
STORE_MODULE_NAME = "build_lex.annotated_lexicon_operation_review"
STORE_MODULE_PATH = MODULE_PATH.parents[2] / "src/build_lex/annotated_lexicon_operation_review.py"
if STORE_MODULE_PATH.is_file():
    SPEC.loader.exec_module(SERVER)
else:
    stub = types.ModuleType(STORE_MODULE_NAME)
    stub.OperationReviewStore = object
    with patch.dict(sys.modules, {STORE_MODULE_NAME: stub}):
        SPEC.loader.exec_module(SERVER)


class FakeStore:
    def __init__(self) -> None:
        self.revision = "revision-1"
        self.session_path = Path("operation-session.json")
        self.decisions: list[dict] = []

    def bootstrap(self) -> dict:
        return {
            "schema_version": "annotated-lexicon-operation-bootstrap/v1",
            "frame_id": "operation-frame-1",
            "revision": self.revision,
            "status": {"item_count": 1, "confirmed_count": 0, "open_count": 1},
            "items": [{"item_id": "operation-1"}],
            "warnings": [],
        }

    def item_state(self, item_id: str) -> dict:
        if item_id != "operation-1":
            raise ValueError("unknown item")
        return {"item_id": item_id, "revision": self.revision}

    def save(self, **kwargs) -> dict:
        if kwargs["expected_revision"] != self.revision:
            raise SERVER.LexiconRepairConflict("repair-operation session changed concurrently")
        self.decisions.append(copy.deepcopy(kwargs))
        self.revision = "revision-2"
        return {"revision": self.revision, "item": {"item_id": kwargs["item_id"]}}

    def reopen(self, **kwargs) -> dict:
        return self.save(**kwargs)

    def snapshot(self) -> dict:
        return {"session": {"revision": self.revision}, "checksums": {}, "frame": {}}


class OperationServerBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeStore()
        with patch.object(SERVER, "OperationReviewStore", return_value=self.store):
            self.service = SERVER.OperationWebService(
                frame_path=Path("frame.json"), session_path=Path("session.json"), reviewer_id="test"
            )
        self.handler = type(
            "QuietOperationHandler", (SERVER.OperationRequestHandler,),
            {"service": self.service, "log_message": lambda self, *args: None},
        )
        self.server = SERVER.ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        self.server.daemon_threads = True
        self.port = int(self.server.server_address[1])
        self.service.configure_network(self.port, "https://hsd.fenglin.pro")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method: str, path: str, payload=None, **headers) -> tuple[int, dict, bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        defaults = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers={**defaults, **headers})
        response = connection.getresponse()
        value = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return value

    def save_payload(self) -> dict:
        return {
            "session_token": self.service.session_token,
            "expected_revision": self.store.revision,
            "item_id": "operation-1",
            "decision": {"resolution": "approve", "entry": None, "notes": "checked"},
            "confirm": True,
        }

    def test_loopback_and_public_origin_boundaries(self) -> None:
        self.assertTrue(SERVER._loopback_authority("127.0.0.1:8769"))
        self.assertFalse(SERVER._loopback_authority("user@127.0.0.1:8769"))
        with self.assertRaises(SERVER.ReviewWebError):
            SERVER._loopback_host("0.0.0.0")
        for origin in ("http://hsd.fenglin.pro", "https://hsd.fenglin.pro/path", "https://user@hsd.fenglin.pro"):
            with self.subTest(origin=origin), self.assertRaises(SERVER.ReviewWebError):
                self.service.configure_network(self.port, origin)

    def test_health_bootstrap_headers_and_stage_identity(self) -> None:
        status, headers, body = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["stage"], "repair-operation")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        status, _, body = self.request("GET", "/api/bootstrap", Host="hsd.fenglin.pro")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["session_token"], self.service.session_token)
        self.assertEqual(json.loads(body)["frame_id"], "operation-frame-1")

    def test_untrusted_host_origin_and_token_cannot_write(self) -> None:
        self.assertEqual(self.request("GET", "/api/bootstrap", Host="attacker.example")[0], 403)
        self.assertEqual(self.request("POST", "/api/save", self.save_payload(), Origin="https://attacker.example")[0], 403)
        self.assertEqual(self.request("POST", "/api/save", {**self.save_payload(), "session_token": "wrong"})[0], 403)
        self.assertEqual(self.store.decisions, [])

    def test_save_cas_and_export_snapshot(self) -> None:
        payload = self.save_payload()
        self.assertEqual(self.request("POST", "/api/save", payload, Origin="https://hsd.fenglin.pro")[0], 200)
        self.assertEqual(len(self.store.decisions), 1)
        self.assertEqual(self.request("POST", "/api/save", payload)[0], 409)
        export = {"session_token": self.service.session_token, "expected_revision": "revision-1"}
        self.assertEqual(self.request("POST", "/api/export", export)[0], 409)
        status, headers, body = self.request("POST", "/api/export", {**export, "expected_revision": self.store.revision})
        self.assertEqual(status, 200)
        self.assertIn("repair-operation-review-snapshot.json", headers["Content-Disposition"])
        self.assertEqual(json.loads(body)["session"]["revision"], self.store.revision)

    def test_bad_request_and_oversize_body_are_rejected(self) -> None:
        payload = self.save_payload()
        bad_payloads = [{**payload, "confirm": "true"}, {**payload, "decision": []}, {**payload, "item_id": None}, {**payload, "extra": 1}]
        for bad in bad_payloads:
            with self.subTest(payload=bad):
                self.assertEqual(self.request("POST", "/api/save", bad)[0], 422)
        self.assertEqual(self.request("POST", "/api/save", payload, **{"Content-Type": "text/plain"})[0], 422)
        self.assertEqual(self.request("POST", "/api/save", {**payload, "padding": "x" * SERVER.MAX_REQUEST_BYTES})[0], 422)
        self.assertEqual(self.store.decisions, [])

    def test_item_lookup_and_path_traversal_are_closed(self) -> None:
        self.assertEqual(self.request("GET", "/api/items/operation-1")[0], 200)
        self.assertEqual(self.request("GET", "/api/items/missing")[0], 404)
        self.assertEqual(self.request("GET", "/api/items/../bootstrap")[0], 404)
        self.assertEqual(self.request("GET", "/../../config/secrets")[0], 404)


if __name__ == "__main__":
    unittest.main()
