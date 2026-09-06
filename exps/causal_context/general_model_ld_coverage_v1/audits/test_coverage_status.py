"""Read-only status helper tests using synthetic receipts and opaque row values."""

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SPEC = importlib.util.spec_from_file_location("coverage_status", Path(__file__).with_name("coverage_status.py"))
status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(status)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class StatusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name) / "synthetic-run"
        self.plan_id = "gmlcoverage-synthetic"

    def terminal(self, state="running", published=False):
        write_json(self.run / "run_manifest.json", {"schema_version": "general-model-coverage-run/v1",
                   "plan_id": self.plan_id, "status": state, "analysis_published": published})

    def checkpoint(self, directory, device, count):
        path = directory / "shards" / str(device) / directory.name / "checkpoint.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE blocks (key TEXT PRIMARY KEY, payload TEXT)")
            connection.executemany("INSERT INTO blocks VALUES (?, ?)", [(str(i), "unparseable opaque score") for i in range(count)])
        return path

    def seal_pass(self, directory, count):
        write_json(directory / "manifest.json", {"status": "complete", "blocks": count,
                   "identity": {"plan_id": self.plan_id, "pass_name": directory.name}})

    def test_missing_files_are_unknown_and_do_not_create_directories_or_databases(self):
        observed = status.snapshot(self.run)
        self.assertEqual(observed["run_state"], "unknown")
        self.assertIsNone(observed["full_dev"]["committed_blocks"])
        self.assertIsNone(observed["full_dev"]["known_committed_subtotal"])
        self.assertEqual(observed["unknown_preflight_pass_manifests"], 18)
        self.assertIn("unknown/10288", status.render_text(observed))
        self.assertFalse(self.run.exists())

    def test_running_counts_are_read_only_and_missing_shard_does_not_become_zero(self):
        self.terminal()
        directory = self.run / "preflight/regression-b1-r0"
        paths = [self.checkpoint(directory, device, 3) for device in range(3)]
        before = {path: path.read_bytes() for path in paths}
        calls, statements = [], []
        real_connect = sqlite3.connect

        def readonly_connect(database, **kwargs):
            calls.append((database, kwargs))
            connection = real_connect(database, **kwargs)
            connection.set_trace_callback(statements.append)
            return connection

        with mock.patch.object(status.sqlite3, "connect", side_effect=readonly_connect):
            observed = status.snapshot(self.run)
        row = observed["preflight_passes"][0]
        self.assertEqual(observed["run_state"], "running")
        self.assertIsNone(row["committed_blocks"])
        self.assertEqual(row["known_committed_subtotal"], 9)
        self.assertEqual(row["readable_shards"], 3)
        self.assertTrue(all("mode=ro" in uri and opts["uri"] for uri, opts in calls))
        self.assertTrue(any("busy_timeout" in statement for statement in statements))
        self.assertTrue(all(statement.startswith("PRAGMA ") or statement == "SELECT COUNT(*) FROM blocks" for statement in statements))
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        self.checkpoint(directory, 3, 0)
        self.assertEqual(status.snapshot(self.run)["preflight_passes"][0]["committed_blocks"], 9)
        self.terminal("interrupted")
        self.assertEqual(status.snapshot(self.run)["run_state"], "interrupted")

    def test_complete_receipts_show_eighteen_seals_and_published_analysis_without_opening_scores(self):
        self.terminal("complete", True)
        for cohort, count in status.COHORT_BLOCKS.items():
            for suffix in status.PASS_SUFFIXES:
                self.seal_pass(self.run / "preflight" / f"{cohort}-b1-{suffix}", count)
        self.seal_pass(self.run / "dev-b1", status.DEV_BLOCKS)
        write_json(self.run / "preflight/preflight_report.json", {"schema_version": "general-model-coverage-preflight/v1",
                   "plan_id": self.plan_id, "passed": True, "complete": True})
        write_json(self.run / "analysis/manifest.json", {"schema_version": "general-model-coverage-analysis/v1",
                   "plan_id": self.plan_id, "gold_join_after_raw_sealed": True, "test_content_read": False})
        (self.run / "analysis/analysis.json").write_text("must not be parsed")
        real_read = status.read_receipt
        with mock.patch.object(status, "read_receipt", wraps=real_read) as reader:
            observed = status.snapshot(self.run)
        self.assertEqual(observed["sealed_preflight_passes"], 18)
        self.assertEqual(observed["preflight_state"], "passed")
        self.assertEqual(observed["analysis_state"], "published (receipt)")
        self.assertTrue(observed["full_dev"]["sealed_manifest"])
        self.assertIsNone(observed["full_dev"]["committed_blocks"])
        self.assertTrue(all(Path(call.args[0]).name in {"run_manifest.json", "manifest.json", "preflight_report.json"}
                            for call in reader.call_args_list))
        self.assertFalse(observed["score_payload_read"])
        self.assertFalse(observed["query_gold_read"])

    def test_malformed_or_locked_database_and_bad_receipts_are_unknown(self):
        self.run.mkdir(parents=True)
        (self.run / "run_manifest.json").write_text("broken json")
        observed = status.snapshot(self.run)
        self.assertEqual(observed["run_state"], "unknown")
        self.assertIn("JSONDecodeError", observed["run_receipt_error"])
        bad = self.run / "bad.sqlite3"
        bad.write_text("not a database")
        self.assertIsNone(status.checkpoint_count(bad)["count"])
        self.terminal()
        directory = self.run / "preflight/regression-b1-r0"
        path = self.checkpoint(directory, 0, 3)
        connection = sqlite3.connect(path)
        try:
            connection.execute("BEGIN EXCLUSIVE")
            observed = status.checkpoint_count(path, busy_timeout_ms=1)
            self.assertIsNone(observed["count"])
            self.assertIn("locked", observed["error"])
        finally:
            connection.rollback()
            connection.close()
        write_json(directory / "manifest.json", {"status": "complete", "blocks": 999,
                   "identity": {"plan_id": self.plan_id, "pass_name": directory.name}})
        self.assertIsNone(status.snapshot(self.run)["preflight_passes"][0]["sealed_manifest"])


if __name__ == "__main__":
    unittest.main()
