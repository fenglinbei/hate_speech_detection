"""Identity-scoped lifecycle guard for the approved until-complete no-category run."""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path


SCHEMA = "general-model-nolabel-watchdog/v1"


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def epoch(value):
    if value is None:
        return float("inf")
    value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("deadline must contain an explicit timezone")
    return value.timestamp()


def require_process_safety():
    if int(os.readlink("/proc/self")) != os.getpid():
        raise RuntimeError("existing-run watchdog requires the host PID namespace matching /proc")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("PID-reuse-safe pidfd signaling is required")


def signal_backend():
    require_process_safety()
    try:
        descriptor = os.pidfd_open(os.getpid())
    except OSError as error:
        if error.errno != errno.ENOSYS:
            raise
        return "pid-start-ticks-recheck-kill-enosys"
    os.close(descriptor)
    return "pidfd"


def identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rindex(")") + 2:].split()
        return {"pid": int(pid), "ppid": int(fields[1]), "pgid": int(fields[2]),
                "sid": int(fields[3]), "start_ticks": int(fields[19]), "state": fields[0]}
    except (OSError, ValueError, IndexError):
        return None


def current(record):
    actual = identity(record["pid"])
    return actual if actual and actual["state"] != "Z" and actual["start_ticks"] == record["start_ticks"] else None


def target_command(target):
    actual = current(target)
    if actual is None:
        return None
    try:
        cwd = Path(os.readlink(f"/proc/{target['pid']}/cwd")).resolve()
        executable = Path(os.readlink(f"/proc/{target['pid']}/exe")).resolve()
        raw = Path(f"/proc/{target['pid']}/cmdline").read_bytes()
    except OSError:
        if current(target) is None:
            return None
        raise
    argv = [value.decode() for value in raw.split(b"\0") if value]
    if (cwd != Path(target["cwd"]).resolve() or executable != Path(target["python_path"]).resolve()
            or len(argv) != 7 or argv[2] != "run" or argv[3] != "--plan" or argv[5] != "--output"
            or (cwd / argv[0]).resolve() != executable
            or (cwd / argv[1]).resolve() != Path(target["entrypoint"]).resolve()
            or (cwd / argv[4]).resolve() != Path(target["plan_ref"]).resolve()
            or (cwd / argv[6]).resolve() != Path(target["run_output"]).resolve()):
        raise ValueError("target PID command, interpreter, cwd, plan, or output differs")
    return {"process": actual, "argv": argv, "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
            "cwd": str(cwd), "executable": str(executable)}


def verify_plan(target):
    reference = Path(target["plan_ref"])
    ref = read(reference)
    plan_path = Path(ref["target_path"]) / "plan.json"
    run = Path(target["run_output"])
    if (ref["plan_id"] != target["plan_id"] or read(plan_path)["plan_id"] != target["plan_id"]
            or digest(plan_path) != ref["plan_sha256"]
            or read(run / "run_manifest.json")["plan_id"] != target["plan_id"]
            or read(run / "binding.json")["plan_id"] != target["plan_id"]):
        raise ValueError("target is not bound to the explicitly registered experiment plan")
    return {"plan_ref_sha256": digest(reference), "plan_sha256": ref["plan_sha256"],
            "run_binding_sha256": digest(run / "binding.json")}


def completed_and_unlocked(target):
    run = Path(target["run_output"])
    if read(run / "run_manifest.json").get("status") != "complete":
        return False
    try:
        with (run / ".writer.lock").open("r") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
        return True
    except (BlockingIOError, FileNotFoundError):
        return False


def validate_config(config):
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unknown existing-run watchdog configuration")
    target = config["target"]
    if type(target["pid"]) is not int or target["pid"] <= 1 or type(target["start_ticks"]) is not int:
        raise ValueError("explicit PID and start_ticks are required")
    for key in ("cwd", "python_path", "entrypoint", "plan_ref", "run_output"):
        if not Path(target[key]).is_absolute():
            raise ValueError("target paths must be absolute")
    graceful, hard = epoch(config["graceful_stop_at"]), epoch(config["hard_deadline"])
    until_complete = (config.get("run_until_complete") is True
                      and config["graceful_stop_at"] is None and config["hard_deadline"] is None)
    if not until_complete and graceful >= hard:
        raise ValueError("graceful stop must precede hard release deadline")
    for key, default in (("poll_seconds", 1), ("interrupt_grace_seconds", 120),
                         ("terminate_grace_seconds", 30), ("hard_release_margin_seconds", 5)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or value <= 0:
            raise ValueError("invalid watchdog timer")
    if config.get("poll_seconds", 1) > 5 or config.get("hard_release_margin_seconds", 5) >= hard - graceful:
        raise ValueError("watchdog polling or deadline margin is unsafe")


def refresh_owned(target, proof, previous):
    scanned = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            row = identity(int(path.name))
            if row and row["state"] != "Z":
                scanned[row["pid"]] = row
    owned = {}
    for record in previous:
        row = current(record)
        if row:
            owned[row["pid"]] = row
    root = target_command(target)
    if root:
        if root["cmdline_sha256"] != proof["cmdline_sha256"]:
            raise ValueError("the registered root changed its command after adoption")
        owned[root["process"]["pid"]] = root["process"]
    while True:
        additions = {pid: row for pid, row in scanned.items() if pid not in owned and row["ppid"] in owned}
        if not additions:
            break
        owned.update(additions)
    # A still-live member prevents reuse of this initially private session ID.
    source_sid = proof["process"]["sid"]
    if any(row["sid"] == source_sid for row in owned.values()):
        for pid, row in scanned.items():
            if row["sid"] == source_sid and row["start_ticks"] >= target["start_ticks"]:
                owned[pid] = row
    return sorted(owned.values(), key=lambda row: row["pid"])


def signal_process(record, sig, backend=None):
    require_process_safety()
    backend = backend or signal_backend()
    if backend == "pid-start-ticks-recheck-kill-enosys":
        actual = current(record)
        if actual is None or any(actual[key] != record[key] for key in ("sid", "pgid")):
            return False
        # This kernel lacks pidfd. There remains a narrow check-to-kill race.
        try:
            os.kill(record["pid"], sig)
            return True
        except ProcessLookupError:
            return False
    if backend != "pidfd":
        raise ValueError("unknown signal backend")
    try:
        descriptor = os.pidfd_open(record["pid"])
    except ProcessLookupError:
        return False
    try:
        if current(record) is None:
            return False
        signal.pidfd_send_signal(descriptor, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(descriptor)


def watch(output):
    require_process_safety()
    config, binding = read(output / "config.json"), read(output / "binding.json")
    if (digest(output / "config.json") != binding["config_sha256"]
            or digest(Path(__file__)) != binding["source_sha256"]):
        raise ValueError("existing-run watchdog source/config snapshot changed")
    backend = signal_backend()
    if backend != binding["signal_backend"]:
        raise ValueError("signal backend changed after watchdog launch")
    target, proof = config["target"], binding["target_proof"]
    lock = (output / ".watchdog.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owned = read(output / "ownership.json")["processes"]
    state = {"schema_version": SCHEMA, "status": "watching", "watchdog": identity(os.getpid()),
             "target": target, "source_sha256": binding["source_sha256"], "signal_backend": backend,
             "signal_backend_limitation": "check-to-kill PID reuse race" if backend != "pidfd" else None,
             "ready_at": now(), "signals": []}
    atomic(output / "state.json", state)
    stop_started = None
    sent_int, sent_term = set(), set()
    try:
        while True:
            owned = refresh_owned(target, proof, owned)
            atomic(output / "ownership.json", {"processes": owned, "updated_at": now()})
            root = current(target)
            if not owned:
                state.update(status="released", reason="cutoff-cleanup-complete" if stop_started else "registered-run-finished",
                             released_at=now(), remaining_processes=[])
                atomic(output / "state.json", state)
                return 0
            if stop_started is None and (time.time() >= epoch(config["graceful_stop_at"]) or root is None or (output / "stop.json").exists()):
                stop_started = time.time()
                state.update(status="stopping", stop_started_at=now(),
                             reason=("user-stop-request" if (output / "stop.json").exists() else
                                     "reservation-graceful-stop") if root else "registered-main-exited-with-descendants")
            if stop_started is not None:
                kill_at = min(stop_started + config.get("interrupt_grace_seconds", 120) + config.get("terminate_grace_seconds", 30),
                              epoch(config["hard_deadline"]) - config.get("hard_release_margin_seconds", 5))
                term_at = min(stop_started + config.get("interrupt_grace_seconds", 120),
                              kill_at - config.get("terminate_grace_seconds", 30))
                for row in owned:
                    key = (row["pid"], row["start_ticks"])
                    if row["pid"] == target["pid"] and target_command(target) is None:
                        continue
                    sig = None
                    if time.time() >= kill_at:
                        sig = signal.SIGKILL
                    elif time.time() >= term_at and key not in sent_term:
                        sig = signal.SIGTERM
                        sent_term.add(key)
                    elif key not in sent_int and (root is None or row["pid"] == target["pid"]):
                        sig = signal.SIGINT
                        sent_int.add(key)
                    if sig is not None and row["pid"] == target["pid"] and verify_plan(target) != binding["plan_proof"]:
                        raise ValueError("registered plan binding changed before signaling its root")
                    if sig is not None and signal_process(row, sig, backend):
                        state["signals"].append({"pid": row["pid"], "start_ticks": row["start_ticks"],
                                                 "signal": sig, "sent_at": now()})
                state.update(remaining_processes=owned, hard_deadline_breached=time.time() >= epoch(config["hard_deadline"]))
                atomic(output / "state.json", state)
            time.sleep(min(config.get("poll_seconds", 1), .2) if stop_started else config.get("poll_seconds", 1))
    except BaseException as error:
        state.update(status="needs_attention", error_type=type(error).__name__, error=str(error), remaining_processes=owned)
        atomic(output / "state.json", state)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def launch(config_path, output):
    require_process_safety()
    backend = signal_backend()
    config = read(config_path)
    validate_config(config)
    target = config["target"]
    plan_proof = verify_plan(target)
    proof = target_command(target)
    output = output.resolve()
    if output.exists():
        raise ValueError("existing-run watchdog output must be a new directory")
    if proof is None:
        if not completed_and_unlocked(target):
            raise ValueError("target PID identity is unavailable and completion/cleanup is not proven")
        output.mkdir(parents=True)
        state = {"schema_version": SCHEMA, "status": "released", "reason": "already-complete-and-unlocked",
                 "target": target, "plan_proof": plan_proof, "released_at": now(), "remaining_processes": []}
        atomic(output / "state.json", state)
        return state
    if proof["process"]["sid"] != target["pid"] or proof["process"]["pgid"] != target["pid"]:
        raise ValueError("adoption requires the target to lead its own private session and process group")
    owned = refresh_owned(target, proof, [proof["process"]])
    output.mkdir(parents=True)
    atomic(output / "config.json", config)
    source = output / "watchdog_source.py"
    source.write_bytes(Path(__file__).read_bytes())
    binding = {"schema_version": SCHEMA, "target_proof": proof, "plan_proof": plan_proof,
               "config_sha256": digest(output / "config.json"), "source_sha256": digest(source),
               "signal_backend": backend, "created_at": now()}
    atomic(output / "binding.json", binding)
    atomic(output / "ownership.json", {"processes": owned, "updated_at": now()})
    with (output / "watchdog.log").open("ab", buffering=0) as handle:
        child = subprocess.Popen([sys.executable, str(source), "_watch", "--output", str(output)],
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"output": str(output), "watchdog": identity(child.pid), "source_sha256": binding["source_sha256"],
               "signal_backend": backend}
    atomic(output / "launch.json", receipt)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if (output / "state.json").exists():
            state = read(output / "state.json")
            if state["status"] in {"watching", "stopping", "released"}:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    del child
                return {**receipt, "status": state["status"]}
        if child.poll() is not None:
            raise RuntimeError("existing-run watchdog exited before it acknowledged readiness")
        time.sleep(.05)
    raise RuntimeError("existing-run watchdog did not acknowledge readiness")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("launch")
    start.add_argument("--config", type=Path, required=True)
    start.add_argument("--output", type=Path, required=True)
    for name in ("status", "_watch"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "_watch":
        return watch(args.output)
    result = launch(args.config, args.output) if args.command == "launch" else read(args.output / "state.json")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
