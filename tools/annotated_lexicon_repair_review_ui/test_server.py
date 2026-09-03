from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "server.py"
SPEC = importlib.util.spec_from_file_location("annotated_lexicon_repair_review_server", MODULE_PATH)
assert SPEC and SPEC.loader
SERVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


class SpanGoldServerBoundaryTests(unittest.TestCase):
    def test_loopback_authority(self) -> None:
        self.assertTrue(SERVER._loopback_authority("127.0.0.1:8769"))
        self.assertTrue(SERVER._loopback_authority("localhost:8769"))
        self.assertFalse(SERVER._loopback_authority("hsd.fenglin.pro"))
        self.assertFalse(SERVER._loopback_authority("user@127.0.0.1:8769"))

    def test_non_loopback_bind_is_rejected(self) -> None:
        with self.assertRaises(SERVER.ReviewWebError):
            SERVER._loopback_host("0.0.0.0")


if __name__ == "__main__":
    unittest.main()
