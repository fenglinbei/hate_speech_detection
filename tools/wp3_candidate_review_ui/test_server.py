from __future__ import annotations

import http.client
import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from urllib.parse import quote


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for root in (REPOSITORY_ROOT, SRC_ROOT, Path(__file__).resolve().parent):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

_SERVER_SPEC = importlib.util.spec_from_file_location(
    "wp3_candidate_review_server", Path(__file__).resolve().parent / "server.py"
)
if _SERVER_SPEC is None or _SERVER_SPEC.loader is None:
    raise RuntimeError("cannot load WP3 candidate-review server")
review_server = importlib.util.module_from_spec(_SERVER_SPEC)
sys.modules[_SERVER_SPEC.name] = review_server
_SERVER_SPEC.loader.exec_module(review_server)
from build_lex.terminology_candidate_review import (  # noqa: E402
    _read_session,
    _with_session_revision,
    lock_raw_phase,
)
from data.training_artifacts import (  # noqa: E402
    resolve_locator_ref,
    write_canonical_json,
)


FRAME_REF = (
    REPOSITORY_ROOT
    / "exps/causal_context/stage1_p0/wp3_candidate_generators_v2/refs/development_frame_ref.json"
)
CONFIG_PATH = REPOSITORY_ROOT / "config/stage1/wp3_candidate_generators_v1.json"


class _FakeService:
    def __init__(self) -> None:
        self.asset_root = Path(__file__).resolve().parent
        self.session_token = "test-token"
        self.allowed_hosts = {"127.0.0.1"}
        self.allowed_origins = {"http://127.0.0.1"}

    def state(self):
        return {"schema_version": "test", "session_token": self.session_token}

    def bootstrap(self):
        return {
            "schema_version": "test-bootstrap",
            "session_token": self.session_token,
            "case_summaries": [],
        }

    def case_state(self, case_id):
        if case_id != "S21-001":
            raise review_server.WebReviewError("unknown development case")
        return {"schema_version": "test-case", "case": {"case_id": case_id}}

    def search_cases(self, query, mode):
        if mode not in review_server.SEARCH_MODES:
            raise review_server.WebReviewError("search mode is invalid")
        return {
            "schema_version": review_server.SEARCH_SCHEMA_VERSION,
            "frame_id": "test-frame",
            "query": query,
            "mode": mode,
            "matches": [],
        }

    def save_raw(self, payload):
        return {"saved": "raw"}

    def lock_raw(self, payload):
        return {"locked": True}

    def save_diagnostic(self, payload):
        return {"saved": "diagnostic"}

    def reopen(self, payload):
        return {"reopened": True}

    def export_zip(self, payload):
        return b"PK-test"


class HttpSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        service = _FakeService()
        handler = type(
            "TestHandler",
            (review_server.ReviewRequestHandler,),
            {"service": service},
        )
        try:
            self.server = review_server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        except PermissionError:
            self.skipTest("sandbox does not permit loopback sockets")
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method, path, *, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        wire = response.read()
        result = response.status, dict(response.getheaders()), wire
        connection.close()
        return result

    def test_csp_host_origin_token_and_fixed_routes(self) -> None:
        status, headers, _ = self._request(
            "GET", "/", headers={"Host": "127.0.0.1"}
        )
        self.assertEqual(status, 200)
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(headers["Cache-Control"], "no-store")

        status, _, wire = self._request(
            "GET", "/api/bootstrap", headers={"Host": "127.0.0.1"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(wire)["schema_version"], "test-bootstrap")

        status, _, wire = self._request(
            "GET", "/api/cases/S21-001", headers={"Host": "127.0.0.1"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(wire)["case"]["case_id"], "S21-001")

        status, _, wire = self._request(
            "GET",
            f"/api/cases/search?q={quote('冻结原文')}&mode=literal",
            headers={"Host": "127.0.0.1"},
        )
        self.assertEqual(status, 200)
        search = json.loads(wire)
        self.assertEqual(search["schema_version"], review_server.SEARCH_SCHEMA_VERSION)
        self.assertEqual(search["query"], "冻结原文")

        status, _, _ = self._request(
            "GET",
            "/api/cases/search?q=test&mode=unknown",
            headers={"Host": "127.0.0.1"},
        )
        self.assertEqual(status, 422)

        status, _, _ = self._request(
            "GET", "/api/cases/S21-999", headers={"Host": "127.0.0.1"}
        )
        self.assertEqual(status, 404)

        status, _, _ = self._request(
            "GET", "/api/state", headers={"Host": "attacker.invalid"}
        )
        self.assertEqual(status, 403)

        body = json.dumps(
            {
                "session_token": "test-token",
                "expected_revision": "rev",
            }
        )
        status, _, _ = self._request(
            "POST",
            "/api/lock-raw",
            body=body,
            headers={
                "Host": "127.0.0.1",
                "Origin": "https://attacker.invalid",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 403)

        bad_token = json.dumps(
            {"session_token": "wrong", "expected_revision": "rev"}
        )
        status, _, _ = self._request(
            "POST",
            "/api/lock-raw",
            body=bad_token,
            headers={"Host": "127.0.0.1", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 403)

        status, _, _ = self._request(
            "GET", "/not-a-file", headers={"Host": "127.0.0.1"}
        )
        self.assertEqual(status, 404)

    def test_request_size_limit_fails_closed(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.putrequest("POST", "/api/lock-raw", skip_host=True)
        connection.putheader("Host", "127.0.0.1")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(review_server.MAX_REQUEST_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 422)
        response.read()
        connection.close()

    def test_public_origin_must_be_exact_https_origin(self) -> None:
        service = _FakeService()
        # Exercise the production validator without constructing the full frame.
        review_server.ReviewService.configure_network(
            service, 8766, public_origin="https://hsd.fenglin.pro"
        )
        self.assertIn("hsd.fenglin.pro", service.allowed_hosts)
        self.assertIn("https://hsd.fenglin.pro", service.allowed_origins)
        for invalid in (
            "http://hsd.fenglin.pro",
            "https://hsd.fenglin.pro/path",
            "https://user@hsd.fenglin.pro",
        ):
            with self.assertRaises(review_server.WebReviewError):
                review_server.ReviewService.configure_network(
                    service, 8766, public_origin=invalid
                )


class SearchProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = object.__new__(review_server.ReviewService)
        self.service.frame = {"frame_id": "search-frame"}
        self.service.cases = [
            {
                "case_id": "S21-001",
                "blind_alias": "苹果主题",
                "content": (
                    "前文冻结原文包含快捷标注表达，后文仅用于摘要测试。"
                ),
                "proposals": [{"secret": "must-not-leak"}],
            },
            {
                "case_id": "S21-002",
                "blind_alias": "香蕉主题",
                "content": "第二条冻结原文包含快捷标住表达。",
            },
            {
                "case_id": "S21-003",
                "blind_alias": "快捷标住表达",
                "content": "标识符精确命中应优先于正文模糊命中。",
            },
            {
                "case_id": "S21-004",
                "blind_alias": "其他主题",
                "content": "相差过多的快捷标签说法不应进入结果。",
            },
        ]

    def test_literal_matches_identifiers_and_content_in_frame_order(self) -> None:
        identifier = self.service.search_cases("s21-002", "literal")
        self.assertEqual(identifier["frame_id"], "search-frame")
        self.assertEqual(identifier["matches"][0]["matched_field"], "case_id")
        self.assertEqual(identifier["matches"][0]["case_id"], "S21-002")

        result = self.service.search_cases("快捷标注表达", "literal")
        self.assertEqual(result["schema_version"], review_server.SEARCH_SCHEMA_VERSION)
        self.assertEqual(result["mode"], "literal")
        self.assertEqual([row["case_id"] for row in result["matches"]], ["S21-001"])
        match = result["matches"][0]
        self.assertEqual(
            match["snippet"][match["match_start"] : match["match_end"]],
            "快捷标注表达",
        )
        self.assertEqual(
            set(match),
            {
                "case_id",
                "matched_field",
                "snippet",
                "match_start",
                "match_end",
                "distance",
            },
        )
        self.assertNotIn("proposals", json.dumps(result, ensure_ascii=False))
        self.assertNotIn("must-not-leak", json.dumps(result, ensure_ascii=False))
        self.assertNotIn(self.service.cases[0]["content"], match["snippet"])

    def test_all_terms_requires_every_deduplicated_term_across_fields(self) -> None:
        result = self.service.search_cases("苹果 快捷 苹果", "all_terms")
        self.assertEqual([row["case_id"] for row in result["matches"]], ["S21-001"])
        match = result["matches"][0]
        self.assertEqual(match["matched_field"], "content")
        self.assertEqual(
            match["snippet"][match["match_start"] : match["match_end"]],
            "快捷",
        )
        self.assertEqual(
            self.service.search_cases("苹果 不存在", "all_terms")["matches"],
            [],
        )

    def test_fuzzy_search_ranks_and_honors_threshold(self) -> None:
        result = self.service.search_cases("快捷标住表达", "fuzzy")
        self.assertEqual(
            [row["case_id"] for row in result["matches"]],
            ["S21-003", "S21-002", "S21-001"],
        )
        self.assertEqual(
            [row["distance"] for row in result["matches"]], [0, 0, 1]
        )
        self.assertNotIn("S21-004", [row["case_id"] for row in result["matches"]])

        # Fuzzy queries shorter than three characters deliberately fall back
        # to literal matching, while retaining the requested response mode.
        short = self.service.search_cases("苹杲", "fuzzy")
        self.assertEqual(short["mode"], "fuzzy")
        self.assertEqual(short["matches"], [])

    def test_search_validation_and_snippet_bound(self) -> None:
        for query, mode in (("", "literal"), ("   ", "all_terms"), ("x", "bad")):
            with self.subTest(query=query, mode=mode):
                with self.assertRaises(review_server.WebReviewError):
                    self.service.search_cases(query, mode)
        with self.assertRaises(review_server.WebReviewError):
            self.service.search_cases(
                "x" * (review_server.MAX_SEARCH_QUERY_CHARS + 1), "literal"
            )

        self.service.cases = [
            {
                "case_id": "S21-999",
                "blind_alias": "长文本",
                "content": "前" * 100 + "命中词" + "后" * 100,
            }
        ]
        match = self.service.search_cases("命中词", "literal")["matches"][0]
        self.assertLessEqual(
            len(match["snippet"]), review_server.SEARCH_SNIPPET_CHARS + 2
        )
        self.assertTrue(match["snippet"].startswith("…"))
        self.assertTrue(match["snippet"].endswith("…"))

        self.service.cases = [
            {
                "case_id": "S21-998",
                "blind_alias": "短文本",
                "content": "完整命中",
            }
        ]
        short_match = self.service.search_cases("完整命中", "literal")[
            "matches"
        ][0]
        self.assertNotEqual(short_match["snippet"], "完整命中")
        self.assertNotIn("完整命中", short_match["snippet"])

        self.service.cases = [
            {
                "case_id": f"S21-{index:03d}",
                "blind_alias": f"别名 {index}",
                "content": "共同命中词",
            }
            for index in range(review_server.MAX_SEARCH_RESULTS + 7)
        ]
        self.assertEqual(
            len(self.service.search_cases("共同", "literal")["matches"]),
            review_server.MAX_SEARCH_RESULTS,
        )


@unittest.skipUnless(FRAME_REF.is_file(), "materialized S2.1 frame is unavailable")
class RealServiceProjectionTests(unittest.TestCase):
    def test_phase_a_state_hides_every_proposal_and_export_hashes_verify(self) -> None:
        _, frame_dir = resolve_locator_ref(
            FRAME_REF, "wp3-s21-development-frame"
        )
        with tempfile.TemporaryDirectory() as directory:
            service = review_server.ReviewService(
                workspace_root=REPOSITORY_ROOT,
                frame_dir=frame_dir,
                generator_config_path=CONFIG_PATH,
                session_path=Path(directory) / "session.json",
                reviewer_id="ui-reviewer-test",
            )
            state = service.state()
            self.assertEqual(state["phase"], "raw")
            self.assertEqual(len(state["cases"]), 424)
            self.assertTrue(all("proposals" not in row for row in state["cases"]))
            bootstrap = service.bootstrap()
            self.assertEqual(len(bootstrap["case_summaries"]), 424)
            self.assertEqual(bootstrap["status"]["diagnostic"]["total"], 0)
            self.assertGreater(state["status"]["diagnostic"]["total"], 0)
            bootstrap_wire = json.dumps(bootstrap, ensure_ascii=False)
            self.assertNotIn('"content"', bootstrap_wire)
            self.assertNotIn('"proposals"', bootstrap_wire)
            self.assertLess(len(bootstrap_wire.encode("utf-8")), 150_000)
            projected = service.case_state("S21-001")
            self.assertNotIn("proposals", projected["case"])
            first_case = service.cases[0]
            surface = next(
                character
                for character in str(first_case["content"])
                if not character.isspace()
            )
            mutation = service.save_raw(
                {
                    "session_token": service.session_token,
                    "expected_revision": state["revision"],
                    "case_id": first_case["case_id"],
                    "annotation": {
                        "needs_explanation": True,
                        "mentions": [
                            {
                                "surface": surface,
                                "occurrence_ordinal": 1,
                                "provisional_route": "A_candidate",
                                "reason_codes": ["stable_core_candidate"],
                                "notes": "",
                            }
                        ],
                        "notes": "",
                    },
                    "confirm": False,
                }
            )
            self.assertIn("raw_annotation", mutation)
            self.assertNotIn("cases", mutation)
            self.assertNotIn("case_summaries", mutation)
            state = service.state()
            wire = service.export_zip(
                {
                    "session_token": service.session_token,
                    "expected_revision": state["revision"],
                }
            )
            with zipfile.ZipFile(io.BytesIO(wire)) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {
                        "annotations.json",
                        "reviewer_declaration.json",
                        "manifest.json",
                        "SHA256SUMS",
                    },
                )
                sums = archive.read("SHA256SUMS").decode("utf-8").splitlines()
                for line in sums:
                    digest, name = line.split("  ", 1)
                    import hashlib

                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)

            session = _read_session(service.session_path)
            for annotation in session["raw_annotations"].values():
                annotation["status"] = "confirmed"
            session = _with_session_revision(session)
            write_canonical_json(service.session_path, session)
            lock_raw_phase(
                session_path=service.session_path,
                expected_revision=session["revision"],
            )
            revealed = service.state()
            self.assertEqual(revealed["phase"], "diagnostic")
            self.assertTrue(any(row["proposals"] for row in revealed["cases"]))
            revealed_case = service.case_state("S21-001")
            self.assertIn("proposals", revealed_case["case"])

    def test_static_assets_contain_no_external_requests(self) -> None:
        asset_root = Path(__file__).resolve().parent
        wire = "\n".join(
            (asset_root / name).read_text(encoding="utf-8")
            for name in ("index.html", "core.js", "app.js", "styles.css")
        )
        self.assertNotIn("https://", wire)
        self.assertNotIn("http://", wire)
        self.assertNotIn("telemetry", wire.casefold())


if __name__ == "__main__":
    unittest.main()
