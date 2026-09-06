import datetime as dt
import fcntl
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from diagnostics import general_model_gpu_queue as queue


def iso_after(seconds):
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)).isoformat()


def config_fixture(root, jobs=None, *, stop=2.5, hard=5):
    return {"schema_version": queue.SCHEMA, "graceful_stop_at": iso_after(stop),
            "hard_deadline": iso_after(hard), "poll_seconds": .05,
            "interrupt_grace_seconds": .5, "terminate_grace_seconds": .3,
            "hard_release_margin_seconds": .3,
            "phases": [{"name": "preflight", "jobs": jobs or []}]}


def job(root, name, code, **extra):
    return {"id": name, "cwd": str(root), "command": [sys.executable, "-c", code], **extra}


def wait_until(predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.05)
    raise AssertionError("timed out waiting for detached queue")


def stop_test_child(child):
    if child.poll() is None:
        child.kill()
    child.wait(timeout=5)


class QueueConfigTests(unittest.TestCase):
    def test_proc_scan_race_rechecks_identity_without_stale_scan_lookup(self):
        identity = {"pid": 777, "ppid": 1, "pgid": 777, "sid": 777, "start_ticks": 123, "state": "R"}
        ownership = {"jobs": [{"id": "job", "token": "token", "process": identity}]}
        with patch.object(queue, "read", return_value=ownership), \
                patch.object(queue.Path, "iterdir", return_value=iter([])), \
                patch.object(queue, "process_identity", return_value=identity):
            self.assertEqual(queue.owned_processes(Path("/unread")), [identity])
        with patch.object(queue, "read", return_value=ownership), \
                patch.object(queue.Path, "iterdir", return_value=iter([])), \
                patch.object(queue, "process_identity", return_value={**identity, "start_ticks": 124}):
            self.assertEqual(queue.owned_processes(Path("/unread")), [])

    def test_changed_source_snapshot_is_rejected(self):
        source = Path(queue.__file__).resolve()
        receipt = {"source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        queue.verify_source_identity(receipt)
        receipt["source_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "source snapshot"):
            queue.verify_source_identity(receipt)

    def test_pid_namespace_mismatch_is_rejected_before_launch_or_signals(self):
        with patch.object(queue.os, "readlink", return_value=str(os.getpid() + 1)):
            with self.assertRaisesRegex(RuntimeError, "PID namespace"):
                queue.require_consistent_pid_namespace()
            with patch.object(queue.os, "kill") as kill, self.assertRaises(RuntimeError):
                queue.send_owned(Path("/unread-ownership"), signal.SIGKILL)
            kill.assert_not_called()

    def test_dependency_must_precede_job_and_times_require_timezone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = config_fixture(root, [job(root, "p14", "pass"), job(root, "p27", "pass"),
                job(root, "d14", "pass", requires_success=["p14", "p27"])])
            queue.validate_config(config)
            config["phases"][0]["jobs"][0]["requires_success"] = ["p27"]
            with self.assertRaises(ValueError):
                queue.validate_config(config)
            with self.assertRaises(ValueError):
                queue.timestamp("2026-09-06T10:00:00")

    def test_complete_manifest_alone_does_not_bypass_held_writer_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, lock = root / "run.json", root / ".writer.lock"
            queue.atomic(manifest, {"status": "complete", "plan_id": "p"})
            config = config_fixture(root)
            config["wait_for"] = {"manifest_path": str(manifest), "required_status": "complete",
                                  "writer_lock": str(lock), "plan_id": "p"}
            with lock.open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(queue.wait_ready(config), (False, "source-writer-lock-not-released"))
            self.assertEqual(queue.wait_ready(config), (True, None))
            config["wait_for"]["process"] = queue.process_identity(os.getpid())
            self.assertEqual(queue.wait_ready(config), (False, "source-process-still-alive"))


class DetachedDeadlineTests(unittest.TestCase):
    def launch(self, root, config):
        path = root / "config.json"
        queue.atomic(path, config)
        output = root / "detached"
        receipt = queue.launch(path, output)
        self.addCleanup(self.cleanup, output, receipt)
        return output, receipt

    def cleanup(self, output, receipt):
        if (output / "ownership.json").exists():
            queue.send_owned(output, signal.SIGKILL)
        for name in ("supervisor", "watchdog"):
            identity = receipt[name]
            if queue.alive(identity):
                os.kill(identity["pid"], signal.SIGKILL)
            try:
                os.waitpid(identity["pid"], 0)
            except ChildProcessError:
                pass

    def released(self, output):
        path = output / "watchdog.json"
        return path.exists() and queue.read(path).get("status") == "released"

    def test_phases_are_serial_and_failed_preflight_does_not_skip_other_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preflight = [job(root, "p14", "raise SystemExit(2)"),
                         job(root, "p27", "from pathlib import Path; Path('p27-done').touch()")]
            config = config_fixture(root, preflight, stop=10, hard=12)
            config["phases"].append({"name": "dev", "jobs": [
                job(root, "d14", "from pathlib import Path; Path('d14-ran').touch()", requires_success=["p14", "p27"]),
                job(root, "d27", "from pathlib import Path; Path('d27-ran').touch()", requires_success=["p14", "p27"])]})
            output, _ = self.launch(root, config)
            wait_until(lambda: self.released(output))
            state = queue.read(output / "state.json")
            self.assertEqual(state["status"], "completed_with_failures")
            self.assertEqual(state["jobs"]["p14"]["status"], "failed")
            self.assertEqual(state["jobs"]["p27"]["status"], "succeeded")
            self.assertEqual(state["jobs"]["d14"]["status"], "skipped")
            self.assertEqual(state["jobs"]["d27"]["status"], "skipped")
            self.assertTrue((root / "p27-done").exists())
            self.assertFalse((root / "d14-ran").exists())
            self.assertFalse((root / "d27-ran").exists())
            self.assertEqual(len(queue.read(output / "ownership.json")["jobs"]), 2)

    def test_detached_queue_outlives_its_launcher_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = config_fixture(root, [job(root, "later", "import time; from pathlib import Path; time.sleep(.4); Path('done-after-launcher-exit').touch()")], stop=10, hard=12)
            queue.atomic(root / "config.json", config)
            output = root / "detached"
            result = subprocess.run([sys.executable, str(Path(queue.__file__).resolve()), "launch",
                "--config", str(root / "config.json"), "--output", str(output)], capture_output=True, text=True, check=True)
            receipt = queue.read(output / "launch.json")
            self.addCleanup(self.cleanup, output, receipt)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(receipt["source_sha256"], hashlib.sha256((output / "queue_source.py").read_bytes()).hexdigest())
            self.assertEqual(queue.read(output / "state.json")["source_sha256"], receipt["source_sha256"])
            wait_until(lambda: self.released(output))
            self.assertTrue((root / "done-after-launcher-exit").exists())
            self.assertEqual(queue.read(output / "state.json")["status"], "complete")
            self.assertTrue((output / "logs/later.log").is_file())

    def test_zero_exit_with_failed_manifest_cannot_unlock_dev(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preflight = job(root, "p14", "from pathlib import Path; Path('result.json').write_text('{\"status\": \"failed\"}')",
                            result_manifest=str(root / "result.json"), success_statuses=["preflight_passed"])
            config = config_fixture(root, [preflight], stop=10, hard=12)
            config["phases"].append({"name": "dev", "jobs": [job(root, "d14", "raise RuntimeError('must not run')", requires_success=["p14"])]})
            output, _ = self.launch(root, config)
            wait_until(lambda: self.released(output))
            state = queue.read(output / "state.json")
            self.assertEqual(state["jobs"]["p14"]["status"], "failed")
            self.assertEqual(state["jobs"]["d14"]["status"], "skipped")

    def test_deadline_sends_sigint_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = "import signal,time; from pathlib import Path\ndef stop(sig, frame):\n Path('checkpoint').write_text('preserved'); raise SystemExit(130)\nsignal.signal(signal.SIGINT,stop)\nPath('started').touch()\ntime.sleep(60)"
            output, _ = self.launch(root, config_fixture(root, [job(root, "long", code)]))
            wait_until(lambda: (root / "started").exists())
            wait_until(lambda: self.released(output))
            self.assertEqual((root / "checkpoint").read_text(), "preserved")
            self.assertFalse(queue.owned_processes(output))
            actions = queue.read(output / "watchdog-actions.json")
            self.assertIn(signal.SIGINT, [row["signal"] for row in actions["signals"]])

    def test_hard_cleanup_reaps_owned_descendant_but_not_unrelated_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            self.addCleanup(stop_test_child, unrelated)
            grandchild = "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
            code = f"import signal,time,subprocess,sys; from pathlib import Path; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); p=subprocess.Popen([sys.executable,'-c',{grandchild!r}],start_new_session=True); Path('grandchild').write_text(str(p.pid)); time.sleep(60)"
            output, _ = self.launch(root, config_fixture(root, [job(root, "stubborn", code)]))
            wait_until(lambda: (root / "grandchild").exists())
            descendant = queue.process_identity(int((root / "grandchild").read_text()))
            wait_until(lambda: self.released(output))
            self.assertFalse(queue.alive(descendant))
            self.assertFalse(queue.owned_processes(output))
            self.assertIsNone(unrelated.poll())
            self.assertLess(time.time(), queue.timestamp(queue.read(output / "config.json")["hard_deadline"]))
            signals = [row["signal"] for row in queue.read(output / "watchdog-actions.json")["signals"]]
            self.assertIn(signal.SIGTERM, signals)
            self.assertIn(signal.SIGKILL, signals)

    def test_independent_watchdog_survives_supervisor_sigkill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = "import signal,time; from pathlib import Path; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); Path('started').touch(); time.sleep(60)"
            output, receipt = self.launch(root, config_fixture(root, [job(root, "long", code)], stop=15, hard=20))
            wait_until(lambda: (root / "started").exists())
            os.kill(receipt["supervisor"]["pid"], signal.SIGKILL)
            wait_until(lambda: self.released(output))
            self.assertEqual(queue.read(output / "watchdog.json")["reason"], "supervisor-exited")
            self.assertFalse(queue.owned_processes(output))

    def test_guard_does_not_exec_before_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "barrier"
            output.mkdir()
            config = config_fixture(root, [job(root, "guarded", "from pathlib import Path; Path('unsafe-exec').touch()")])
            queue.atomic(output / "config.json", config)
            queue.atomic(output / "ownership.json", {"jobs": []})
            me = queue.process_identity(os.getpid())
            queue.atomic(output / "launch.json", {"supervisor": me, "watchdog": me})
            queue.atomic(output / "watchdog-ready.json", {"process": me})
            token = uuid.uuid4().hex
            child = subprocess.Popen([sys.executable, str(Path(queue.__file__).resolve()), "_guard", "--output", str(output),
                                      "--job", "guarded"], env={**os.environ, queue.TOKEN_ENV: token})
            self.addCleanup(stop_test_child, child)
            time.sleep(.3)
            self.assertIsNone(child.poll())
            self.assertFalse((root / "unsafe-exec").exists())
            queue.request_stop(output, "test-stop")
            self.assertEqual(child.wait(timeout=5), 130)
            self.assertFalse((root / "unsafe-exec").exists())


if __name__ == "__main__":
    unittest.main()
