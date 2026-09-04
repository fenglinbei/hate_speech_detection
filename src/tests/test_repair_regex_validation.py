from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from build_lex.annotated_lexicon_repair import LexiconRepairError
from build_lex import repair_regex_validation as validation


def policy(pattern: str = "^基本$", *, rule_id: str = "exclude-basic") -> dict:
    return {"exclude_any": [{"rule_id": rule_id, "target": "context", "pattern": pattern}]}


class RepairRegexValidationTests(unittest.TestCase):
    def test_empty_policy_does_not_start_worker(self) -> None:
        with patch.object(validation, "_run_worker") as worker:
            self.assertEqual(validation.validate_match_policy({}), {"require_any": [], "exclude_any": []})
            worker.assert_not_called()

    def test_pinned_runtime_check_and_isolated_compilation(self) -> None:
        runtime = validation.check_regex_runtime()
        self.assertEqual(runtime["regex_version"], "2026.4.4")
        self.assertEqual(runtime["validation_isolation"], "resource-limited-subprocess/v1")
        self.assertEqual(validation.validate_match_policy(policy())["exclude_any"][0]["pattern"], "^基本$")

    def test_invalid_expression_fails_closed(self) -> None:
        with self.assertRaisesRegex(LexiconRepairError, "compilation failed"):
            validation.validate_match_policy(policy("(?P<"))

    def test_invalid_structure_never_reaches_worker(self) -> None:
        invalid = [
            [],
            {"other": []},
            {"require_any": "^foo$"},
            {"require_any": [{}]},
            {"require_any": [{"rule_id": 1, "target": "left", "pattern": "a"}]},
            {"require_any": [{"rule_id": "a", "target": "arbitrary", "pattern": "a"}]},
            policy(""),
            policy("a" * 257),
            policy("\ud800"),
            policy(rule_id=" " * 2),
            policy(rule_id="a" * 129),
        ]
        with patch.object(validation, "_run_worker") as worker:
            for value in invalid:
                with self.subTest(value=value), self.assertRaises(LexiconRepairError):
                    validation.validate_match_policy(value)
            worker.assert_not_called()

    def test_rule_count_and_cross_group_ids_are_bounded(self) -> None:
        row = {"rule_id": "a", "target": "left", "pattern": "a"}
        with self.assertRaisesRegex(LexiconRepairError, "duplicated"):
            validation.validate_match_policy({"require_any": [row], "exclude_any": [row]})
        with self.assertRaisesRegex(LexiconRepairError, "exceeds 8"):
            validation.validate_match_policy({"require_any": [{**row, "rule_id": str(i)} for i in range(9)]})

    def test_worker_uses_current_isolated_interpreter_and_stdin(self) -> None:
        result = subprocess.CompletedProcess([], 0, b'{"ok":true,"regex_version":"2026.4.4","compiled_count":1}', b"")
        with patch.object(validation.subprocess, "run", return_value=result) as run:
            validation.validate_match_policy(policy())
        arguments, keywords = run.call_args
        self.assertEqual(arguments[0][:3], [validation.sys.executable, "-I", "-c"])
        self.assertNotIn("^基本$", arguments[0][-1])
        self.assertIn("^基本$".encode("utf-8"), keywords["input"])
        self.assertEqual(keywords["timeout"], 3.0)
        self.assertTrue(keywords["close_fds"])

    def test_worker_timeout_crash_and_malformed_output_fail_closed(self) -> None:
        cases = [
            subprocess.TimeoutExpired("regex-worker", 3.0),
            subprocess.CompletedProcess([], -9, b"", b""),
            subprocess.CompletedProcess([], 0, b"{}", b""),
            subprocess.CompletedProcess([], 0, b'{"ok":true,"regex_version":"old","compiled_count":0}', b""),
            subprocess.CompletedProcess([], 0, b"[]", b""),
            subprocess.CompletedProcess([], 0, b"x" * 4097, b""),
        ]
        for case in cases:
            configuration = {"side_effect": case} if isinstance(case, Exception) else {"return_value": case}
            with self.subTest(case=case), patch.object(validation.subprocess, "run", **configuration):
                with self.assertRaises(LexiconRepairError):
                    validation.check_regex_runtime()


if __name__ == "__main__":
    unittest.main()
