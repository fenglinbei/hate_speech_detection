import datetime as dt
import errno
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from diagnostics import general_model_existing_run_watchdog as guard


def wait_until(predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError("timeout waiting for existing-run watchdog")


def iso_after(seconds):
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)).isoformat()


def stop_child(child):
    if child.poll() is None:
        child.kill()
    child.wait(timeout=5)


class ExistingRunWatchdogTests(unittest.TestCase):
    def setUp(self):
        try:
            guard.require_process_safety()
        except RuntimeError as error:
            self.skipTest(str(error))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def target(self, body):
        root = self.root
        plan_dir, run = root / "plan", root / "run"
        plan_dir.mkdir()
        run.mkdir()
        guard.atomic(plan_dir / "plan.json", {"plan_id": "test-frozen-plan"})
        ref = root / "plan_ref.json"
        guard.atomic(ref, {"plan_id": "test-frozen-plan", "target_path": str(plan_dir),
                          "plan_sha256": guard.digest(plan_dir / "plan.json")})
        guard.atomic(run / "run_manifest.json", {"plan_id": "test-frozen-plan", "status": "running"})
        guard.atomic(run / "binding.json", {"plan_id": "test-frozen-plan"})
        script = root / "model.py"
        script.write_text("import os,sys,signal,time,subprocess,fcntl\nfrom pathlib import Path\n"
            "lock=Path('run/.writer.lock').open('a+')\nfcntl.flock(lock,fcntl.LOCK_EX)\n" + body)
        child = subprocess.Popen([sys.executable, str(script), "run", "--plan", str(ref), "--output", str(run)],
                                 cwd=root, start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(stop_child, child)
        wait_until(lambda: (root / "started").exists())
        if (root / "grandchild").exists():
            descendant = guard.identity(int((root / "grandchild").read_text()))
            self.addCleanup(guard.signal_process, descendant, signal.SIGKILL)
        identity = guard.identity(child.pid)
        target = {"pid": child.pid, "start_ticks": identity["start_ticks"], "python_path": str(Path(sys.executable).resolve()),
                  "entrypoint": str(script), "cwd": str(root), "plan_ref": str(ref),
                  "plan_id": "test-frozen-plan", "run_output": str(run)}
        return child, target

    def config(self, target, *, stop=2.2, hard=4):
        return {"schema_version": guard.SCHEMA, "target": target, "graceful_stop_at": iso_after(stop),
                "hard_deadline": iso_after(hard), "poll_seconds": .05, "interrupt_grace_seconds": .4,
                "terminate_grace_seconds": .3, "hard_release_margin_seconds": .2}

    def launch(self, config, *, separate_launcher=False):
        config_path = self.root / "config.json"
        guard.atomic(config_path, config)
        output = self.root / "watchdog"
        if separate_launcher:
            result = subprocess.run([sys.executable, str(Path(guard.__file__).resolve()), "launch", "--config", str(config_path),
                                     "--output", str(output)], capture_output=True, text=True, check=True)
            self.assertEqual(result.returncode, 0)
        else:
            guard.launch(config_path, output)
        if (output / "launch.json").exists():
            self.addCleanup(self.cleanup_guard, output)
        return output

    def cleanup_guard(self, output):
        if (output / "ownership.json").exists():
            for record in guard.read(output / "ownership.json")["processes"]:
                guard.signal_process(record, signal.SIGKILL)
        process = guard.read(output / "launch.json")["watchdog"]
        guard.signal_process(process, signal.SIGKILL)
        try:
            os.waitpid(process["pid"], 0)
        except ChildProcessError:
            pass

    def released(self, output):
        return guard.read(output / "state.json")["status"] == "released"

    def test_graceful_cutoff_is_detached_and_preserves_checkpoint(self):
        _, target = self.target("def stop(sig,frame):\n Path('checkpoint').write_text('saved'); raise SystemExit(130)\n"
                                "signal.signal(signal.SIGINT,stop)\nPath('started').touch()\ntime.sleep(60)\n")
        output = self.launch(self.config(target), separate_launcher=True)
        wait_until(lambda: self.released(output))
        state = guard.read(output / "state.json")
        self.assertEqual((self.root / "checkpoint").read_text(), "saved")
        self.assertEqual(state["remaining_processes"], [])
        self.assertEqual([row["signal"] for row in state["signals"]], [signal.SIGINT])
        self.assertEqual(guard.read(output / "binding.json")["source_sha256"], guard.digest(output / "watchdog_source.py"))

    def test_wrong_start_time_or_command_is_not_adopted(self):
        _, target = self.target("Path('started').touch()\ntime.sleep(60)\n")
        for altered in ({**target, "start_ticks": target["start_ticks"] + 1},
                        {**target, "entrypoint": str(self.root / "other.py")}):
            guard.atomic(self.root / "config.json", self.config(altered))
            with self.assertRaises(ValueError):
                guard.launch(self.root / "config.json", self.root / "must-not-start")
        self.assertFalse((self.root / "must-not-start").exists())
        self.assertIsNotNone(guard.current(target))

    def test_hard_cutoff_reclaims_private_descendants_not_other_jobs(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        self.addCleanup(stop_child, unrelated)
        grandchild = "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
        _, target = self.target("signal.signal(signal.SIGINT,signal.SIG_IGN)\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}],start_new_session=True)\n"
            "Path('grandchild').write_text(str(p.pid))\nPath('started').touch()\ntime.sleep(60)\n")
        descendant = guard.identity(int((self.root / "grandchild").read_text()))
        config = self.config(target)
        output = self.launch(config)
        wait_until(lambda: self.released(output))
        self.assertIsNone(guard.current(descendant))
        self.assertIsNone(unrelated.poll())
        state = guard.read(output / "state.json")
        self.assertIn(signal.SIGTERM, [row["signal"] for row in state["signals"]])
        self.assertIn(signal.SIGKILL, [row["signal"] for row in state["signals"]])
        self.assertLess(time.time(), guard.epoch(config["hard_deadline"]))

    def test_main_sigkill_leaves_detached_watchdog_to_reclaim_recorded_orphan(self):
        grandchild = "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
        child, target = self.target(f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}],start_new_session=True)\n"
            "Path('grandchild').write_text(str(p.pid))\nPath('started').touch()\ntime.sleep(60)\n")
        output = self.launch(self.config(target, stop=15, hard=20))
        child.kill()
        child.wait(timeout=5)
        wait_until(lambda: self.released(output))
        self.assertEqual(guard.read(output / "state.json")["remaining_processes"], [])

    def test_pidfd_does_not_signal_a_reused_pid(self):
        original = {"pid": 777, "start_ticks": 123}
        reused = {"pid": 777, "start_ticks": 124, "state": "R"}
        with patch.object(guard, "require_process_safety"), patch.object(guard.os, "pidfd_open", return_value=999), \
                patch.object(guard.os, "close"), patch.object(guard, "identity", return_value=reused), \
                patch.object(guard.signal, "pidfd_send_signal") as send:
            self.assertFalse(guard.signal_process(original, signal.SIGKILL, "pidfd"))
        send.assert_not_called()

    def test_backend_only_falls_back_for_kernel_enosys(self):
        with patch.object(guard.os, "pidfd_open", side_effect=OSError(errno.ENOSYS, "not implemented")):
            self.assertEqual(guard.signal_backend(), "pid-start-ticks-recheck-kill-enosys")
        with patch.object(guard.os, "pidfd_open", side_effect=PermissionError(errno.EPERM, "not permitted")):
            with self.assertRaises(PermissionError):
                guard.signal_backend()

    def test_fallback_rejects_reused_pid_or_changed_session(self):
        original = {"pid": 777, "start_ticks": 123, "sid": 777, "pgid": 777}
        for actual in ({**original, "start_ticks": 124, "state": "R"},
                       {**original, "sid": 888, "state": "R"}):
            with patch.object(guard, "identity", return_value=actual), patch.object(guard.os, "kill") as send:
                self.assertFalse(guard.signal_process(original, signal.SIGKILL, "pid-start-ticks-recheck-kill-enosys"))
            send.assert_not_called()

    def test_live_complete_unlocked_root_retains_deadline_protection(self):
        _, target = self.target("def stop(sig,frame):\n Path('checkpoint').write_text('saved'); raise SystemExit(130)\n"
            "signal.signal(signal.SIGINT,stop)\nfcntl.flock(lock,fcntl.LOCK_UN)\nPath('started').touch()\ntime.sleep(60)\n")
        guard.atomic(self.root / "run/run_manifest.json", {"plan_id": "test-frozen-plan", "status": "complete"})
        self.assertTrue(guard.completed_and_unlocked(target))
        output = self.launch(self.config(target))
        time.sleep(.15)
        self.assertEqual(guard.read(output / "state.json")["status"], "watching")
        self.assertIsNotNone(guard.current(target))
        wait_until(lambda: self.released(output))
        self.assertEqual((self.root / "checkpoint").read_text(), "saved")

    def test_completed_and_unlocked_run_needs_no_adoption(self):
        child, target = self.target("Path('started').touch()\ntime.sleep(60)\n")
        child.kill()
        child.wait(timeout=5)
        guard.atomic(self.root / "run/run_manifest.json", {"plan_id": "test-frozen-plan", "status": "complete"})
        output = self.launch(self.config(target))
        self.assertTrue(self.released(output))
        self.assertFalse((output / "launch.json").exists())


if __name__ == "__main__":
    unittest.main()
