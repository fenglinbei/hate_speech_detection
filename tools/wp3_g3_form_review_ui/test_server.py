from __future__ import annotations

import http.client
import importlib.util
import json
import sys
import threading
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src", Path(__file__).resolve().parent):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

_SERVER_SPEC = importlib.util.spec_from_file_location(
    "wp3_g3_form_review_server", Path(__file__).resolve().parent / "server.py"
)
if _SERVER_SPEC is None or _SERVER_SPEC.loader is None:
    raise RuntimeError("cannot load G3 form-review server")
form_server = importlib.util.module_from_spec(_SERVER_SPEC)
sys.modules[_SERVER_SPEC.name] = form_server
_SERVER_SPEC.loader.exec_module(form_server)


class _UiContractParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.asset_urls: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])
        for name in ("href", "src"):
            if values.get(name):
                self.asset_urls.append(values[name])


class _FakeService:
    def __init__(self) -> None:
        self.asset_root = Path(__file__).resolve().parent
        self.session_token = "test-token"
        self.allowed_hosts = {"127.0.0.1"}
        self.allowed_origins = {"http://127.0.0.1"}

    def bootstrap(self):
        return {"schema_version": "test", "session_token": self.session_token}

    def item_state(self, item_id):
        if item_id != "g3form-one":
            raise form_server.WebFormReviewError("unknown")
        return {"schema_version": "test-item", "item": {"item_id": item_id}}

    def save(self, payload):
        return {"schema_version": "saved", "item_id": payload["item_id"]}

    def reopen(self, payload):
        return {"schema_version": "reopened", "item_id": payload["item_id"]}


class FormReviewPublicOriginTests(unittest.TestCase):
    def test_public_origin_is_exact_https_origin_only_without_binding(self):
        service = form_server.FormReviewService.__new__(
            form_server.FormReviewService
        )
        service.allowed_hosts = set()
        service.allowed_origins = set()
        service.configure_network(8767, public_origin="https://hsd.fenglin.pro")
        self.assertIn("hsd.fenglin.pro", service.allowed_hosts)
        self.assertIn("https://hsd.fenglin.pro", service.allowed_origins)
        self.assertNotIn("http://hsd.fenglin.pro", service.allowed_origins)

        invalid = (
            "http://hsd.fenglin.pro",
            "https://hsd.fenglin.pro/",
            "https://hsd.fenglin.pro/review",
            "https://hsd.fenglin.pro?x=1",
            "https://user@hsd.fenglin.pro",
            "https://",
        )
        for origin in invalid:
            with self.subTest(origin=origin):
                with self.assertRaises(form_server.WebFormReviewError):
                    service.configure_network(8767, public_origin=origin)

    def test_check_mode_validates_public_origin_before_returning(self):
        service = form_server.FormReviewService.__new__(
            form_server.FormReviewService
        )
        service.allowed_hosts = set()
        service.allowed_origins = set()
        service.bootstrap = lambda: {  # type: ignore[method-assign]
            "frame_id": "frame",
            "status": {"confirmed_count": 0, "item_count": 1},
        }
        with patch.object(form_server, "FormReviewService", return_value=service):
            with self.assertRaises(form_server.WebFormReviewError):
                form_server.run_server(
                    workspace_root=Path("."),
                    source_bundle_dir=Path("."),
                    frame_dir=Path("."),
                    session_path=Path("session.json"),
                    reviewer_id="reviewer",
                    host="127.0.0.1",
                    port=8767,
                    public_origin="https://hsd.fenglin.pro/review",
                    check=True,
                )


class FormReviewUiAssetTests(unittest.TestCase):
    def test_shared_assets_are_fixed_to_the_existing_s21_review_project(self):
        self.assertEqual(
            set(form_server.SHARED_REVIEW_ASSETS),
            {"review-base.css", "review-core.js"},
        )
        expected_root = REPOSITORY_ROOT / "tools" / "wp3_candidate_review_ui"
        for path in form_server.SHARED_REVIEW_ASSETS.values():
            self.assertEqual(path.parent, expected_root)
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
        self.assertTrue(set(form_server.SHARED_REVIEW_ASSETS) <= form_server.ASSET_NAMES)

    def test_ui_has_unique_ids_and_same_origin_assets_only(self):
        parser = _UiContractParser()
        html = (Path(__file__).resolve().parent / "index.html").read_text(
            encoding="utf-8"
        )
        parser.feed(html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertEqual(
            parser.asset_urls,
            [
                "/review-base.css",
                "/styles.css",
                "/review-core.js",
                "/core.js",
                "/app.js",
            ],
        )
        self.assertNotIn("A_candidate", html)
        self.assertNotIn("ABC", html)
        action_keys = (
            ("accept", "1"),
            ("edit", "2"),
            ("reject", "3"),
            ("defer", "4"),
        )
        for action, key in action_keys:
            self.assertIn(f'data-action="{action}"', html)
            self.assertIn(f"<kbd>{key}</kbd>", html)

        app = (Path(__file__).resolve().parent / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("X-Review-Client-Instance", app)

    def test_nginx_template_logs_latency_request_and_client_instance_ids(self):
        nginx = (REPOSITORY_ROOT / "deploy/wp3_g3/hsd.fenglin.pro.nginx").read_text(
            encoding="utf-8"
        )
        for field in (
            "$request_time",
            "$upstream_response_time",
            "$request_id",
            "$http_x_review_client_instance",
        ):
            self.assertIn(field, nginx)
        self.assertIn("proxy_set_header X-Request-ID $request_id", nginx)
        self.assertIn(
            "proxy_set_header X-Review-Client-Instance "
            "$http_x_review_client_instance",
            nginx,
        )

    def test_item_summary_adds_only_frozen_evidence_provenance(self):
        service = form_server.FormReviewService.__new__(
            form_server.FormReviewService
        )
        service.item_by_id = {
            "g3form-one": {
                "item_id": "g3form-one",
                "surface": "永远滴神",
                "canonical": "永远的神",
                "proposed_family": "phonetic_variant",
                "evidence_ids": ["ev-one", "ev-two"],
            }
        }
        service.evidence_by_id = {
            "ev-one": {"publisher": "来源乙", "source_role": "primary"},
            "ev-two": {"publisher": "来源甲", "source_role": "supporting"},
        }
        session = {
            "decisions": {
                "g3form-one": {"status": "draft", "action": "defer"}
            }
        }
        self.assertEqual(
            service._summary(session, "g3form-one"),
            {
                "item_id": "g3form-one",
                "surface": "永远滴神",
                "canonical": "永远的神",
                "proposed_family": "phonetic_variant",
                "publishers": ["来源乙", "来源甲"],
                "source_roles": ["primary", "supporting"],
                "evidence_count": 2,
                "status": "draft",
                "action": "defer",
            },
        )


class FormReviewHttpSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type(
            "TestFormHandler",
            (form_server.FormReviewRequestHandler,),
            {"service": _FakeService()},
        )
        try:
            self.server = form_server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        except PermissionError:
            self.skipTest("sandbox does not permit loopback sockets")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_host_origin_token_csp_and_fixed_routes(self):
        status, headers, _ = self._request("GET", "/", headers={"Host": "127.0.0.1"})
        self.assertEqual(status, 200)
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        status, _, _ = self._request(
            "GET", "/api/bootstrap", headers={"Host": "attacker.invalid"}
        )
        self.assertEqual(status, 403)
        payload = json.dumps(
            {
                "session_token": "test-token",
                "expected_revision": "rev",
                "item_id": "g3form-one",
                "decision": {},
                "confirm": False,
            }
        )
        status, _, _ = self._request(
            "POST",
            "/api/save",
            body=payload,
            headers={
                "Host": "127.0.0.1",
                "Origin": "https://attacker.invalid",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 403)
        bad = json.loads(payload)
        bad["session_token"] = "wrong"
        status, _, _ = self._request(
            "POST",
            "/api/save",
            body=json.dumps(bad),
            headers={"Host": "127.0.0.1", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 403)
        status, _, _ = self._request(
            "GET", "/etc/passwd", headers={"Host": "127.0.0.1"}
        )
        self.assertEqual(status, 404)

        for asset, content_type in (
            ("/review-base.css", "text/css; charset=utf-8"),
            ("/review-core.js", "text/javascript; charset=utf-8"),
            ("/core.js", "text/javascript; charset=utf-8"),
        ):
            with self.subTest(asset=asset):
                status, asset_headers, body = self._request(
                    "GET", asset, headers={"Host": "127.0.0.1"}
                )
                self.assertEqual(status, 200)
                self.assertEqual(asset_headers["Content-Type"], content_type)
                self.assertTrue(body)

    def test_request_size_limit(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.putrequest("POST", "/api/save", skip_host=True)
        connection.putheader("Host", "127.0.0.1")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(form_server.MAX_REQUEST_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 422)
        response.read()
        connection.close()

    def test_request_and_anonymous_client_ids_are_safe_and_echoed(self):
        request_id = "0123456789abcdef0123456789abcdef"
        client_id = "11111111-2222-4333-8444-555555555555"
        status, headers, _ = self._request(
            "GET",
            "/api/bootstrap",
            headers={
                "Host": "127.0.0.1",
                "X-Request-ID": request_id,
                "X-Review-Client-Instance": client_id,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Request-ID"], request_id)

        status, headers, _ = self._request(
            "GET",
            "/api/bootstrap",
            headers={
                "Host": "127.0.0.1",
                "X-Request-ID": "bad id with spaces",
                "X-Review-Client-Instance": "bad/client",
            },
        )
        self.assertEqual(status, 200)
        self.assertRegex(headers["X-Request-ID"], r"^[0-9a-f]{32}$")
        self.assertNotEqual(headers["X-Request-ID"], "bad id with spaces")

    def test_non_loopback_bind_is_rejected(self):
        with self.assertRaises(form_server.WebFormReviewError):
            form_server._loopback_host("0.0.0.0")

    def test_public_origin_is_exact_https_origin_only(self):
        service = form_server.FormReviewService.__new__(
            form_server.FormReviewService
        )
        service.allowed_hosts = set()
        service.allowed_origins = set()
        service.configure_network(8767, public_origin="https://hsd.fenglin.pro")
        self.assertIn("hsd.fenglin.pro", service.allowed_hosts)
        self.assertIn("https://hsd.fenglin.pro", service.allowed_origins)
        self.assertNotIn("http://hsd.fenglin.pro", service.allowed_origins)

        invalid = (
            "http://hsd.fenglin.pro",
            "https://hsd.fenglin.pro/",
            "https://hsd.fenglin.pro/review",
            "https://hsd.fenglin.pro?x=1",
            "https://user@hsd.fenglin.pro",
            "https://",
        )
        for origin in invalid:
            with self.subTest(origin=origin):
                with self.assertRaises(form_server.WebFormReviewError):
                    service.configure_network(8767, public_origin=origin)

        bound = self.server.RequestHandlerClass.service
        bound.allowed_hosts.add("hsd.fenglin.pro")
        bound.allowed_origins.add("https://hsd.fenglin.pro")
        status, _, _ = self._request(
            "GET", "/api/bootstrap", headers={"Host": "hsd.fenglin.pro"}
        )
        self.assertEqual(status, 200)
        payload = json.dumps(
            {
                "session_token": "test-token",
                "expected_revision": "rev",
                "item_id": "g3form-one",
                "decision": {},
                "confirm": False,
            }
        )
        status, _, _ = self._request(
            "POST",
            "/api/save",
            body=payload,
            headers={
                "Host": "hsd.fenglin.pro",
                "Origin": "https://hsd.fenglin.pro",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
