from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_lex.terminology_external_page_capture import (  # noqa: E402
    _classify_body,
    _status_for_response,
    extract_registered_urls,
)


class ExternalPageCaptureUnitTests(unittest.TestCase):
    def test_project_register_has_42_g3_urls_and_no_pricing_page(self) -> None:
        rows = extract_registered_urls(
            ROOT
            / "docs/research/experiment-plans/"
            "wp3-g3-source-policy-and-candidate-register.md"
        )
        urls = {row["requested_url"] for row in rows}
        self.assertEqual(len(urls), 42)
        self.assertNotIn("https://bigmodel.cn/pricing", urls)

    def test_register_urls_are_deduplicated_with_all_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            register = Path(temporary) / "register.md"
            register.write_text(
                "# Sources\n## Fixed\n<xhttps://invalid>\n"
                "<https://example.com/a>\n## Pending\n"
                "<https://example.com/a>\n<http://example.org/b>\n",
                encoding="utf-8",
            )
            rows = extract_registered_urls(register)
        self.assertEqual(
            [row["requested_url"] for row in rows],
            ["http://example.org/b", "https://example.com/a"],
        )
        repeated = rows[1]
        self.assertEqual(len(repeated["register_occurrences"]), 2)
        self.assertEqual(
            repeated["register_occurrences"][0]["heading_path"],
            ["Sources", "Fixed"],
        )
        self.assertEqual(
            repeated["register_occurrences"][1]["heading_path"],
            ["Sources", "Pending"],
        )

    def test_pdf_requires_container_magic_and_eof(self) -> None:
        analysis, projection = _classify_body(
            b"%PDF-1.7\nbody\n%%EOF\n",
            declared_type="application/pdf",
            full_content_type="application/pdf",
            final_url="https://example.com/a.pdf",
        )
        self.assertIsNone(projection)
        self.assertEqual(analysis["content_validation"], "valid_pdf_container")
        self.assertEqual(_status_for_response(200, analysis, b"x"), "downloaded_usable")

        invalid, _ = _classify_body(
            b"<html><body>gateway error</body></html>",
            declared_type="application/pdf",
            full_content_type="application/pdf",
            final_url="https://example.com/a.pdf",
        )
        self.assertEqual(invalid["content_validation"], "invalid_pdf_container")
        self.assertEqual(_status_for_response(200, invalid, b"x"), "invalid_body")

    def test_json_and_javascript_shell_are_distinguished(self) -> None:
        body = json.dumps({"items": [1, 2]}, ensure_ascii=False).encode("utf-8")
        parsed, _ = _classify_body(
            body,
            declared_type="application/json",
            full_content_type="application/json; charset=utf-8",
            final_url="https://example.com/data.json",
        )
        self.assertEqual(parsed["content_validation"], "valid_json")
        shell, projection = _classify_body(
            b"<html><head><title>App</title><script></script><script></script>"
            b"</head><body>Please enable JavaScript</body></html>",
            declared_type="text/html",
            full_content_type="text/html; charset=utf-8",
            final_url="https://example.com/app",
        )
        self.assertIsNotNone(projection)
        self.assertEqual(shell["content_validation"], "likely_javascript_shell")
        self.assertEqual(
            _status_for_response(200, shell, b"x"),
            "downloaded_needs_browser_archive",
        )

    def test_cookie_challenge_is_not_treated_as_short_page(self) -> None:
        challenged, projection = _classify_body(
            b"<html><script>document.cookie='acw_sc__v2=value';location.reload()"
            b"</script></html>",
            declared_type="text/html",
            full_content_type="text/html; charset=utf-8",
            final_url="https://example.com/download",
        )
        self.assertIsNone(projection)
        self.assertEqual(challenged["content_validation"], "error_or_challenge")


if __name__ == "__main__":
    unittest.main()
