"""Detached local GPU queue with an independently surviving deadline watchdog."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from pathlib import Path


TOKEN_ENV = "GENERAL_MODEL_GPU_QUEUE_TOKEN"
SCHEMA = "general-model-gpu-queue/v1"


def require_consistent_pid_namespace():
    if int(os.readlink("/proc/self")) != os.getpid():
        raise RuntimeError("GPU watchdog requires a /proc view matching the process PID namespace; launch outside the exec sandbox")


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("queue times must include an explicit timezone")
    return parsed.timestamp()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def process_identity(pid):
    try:
        value = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = value[value.rindex(")") + 2:].split()
        return {"pid": int(pid), "ppid": int(fields[1]), "pgid": int(fields[2]),
                "sid": int(fields[3]), "start_ticks": int(fields[19]), "state": fields[0]}
    except (OSError, ValueError, IndexError):
        return None


def alive(identity):
    actual = process_identity(identity["pid"]) if identity else None
    return bool(actual and actual["state"] != "Z" and actual["start_ticks"] == identity["start_ticks"])


def owned_processes(output):
    ownership = read(output / "ownership.json")
    tokens = {row["token"] for row in ownership["jobs"]}
    identities, found = {}, {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        identity = process_identity(int(path.name))
        if identity is None or identity["state"] == "Z":
            continue
        identities[identity["pid"]] = identity
        try:
            environment = (path / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(f"{TOKEN_ENV}={token}".encode() in environment for token in tokens):
            found[identity["pid"]] = identity
    for row in ownership["jobs"]:
        identity = row["process"]
        actual = process_identity(identity["pid"])
        if actual and actual["state"] != "Z" and actual["start_ticks"] == identity["start_ticks"]:
            found[actual["pid"]] = actual
    # Also recognize descendants that intentionally replace their environment.
    while True:
        new = {pid: identity for pid, identity in identities.items()
               if pid not in found and identity["ppid"] in found}
        if not new:
            break
        found.update(new)
    return list(found.values())


def send_owned(output, sig, *, main_only=False):
    require_consistent_pid_namespace()
    ownership = read(output / "ownership.json")
    processes = owned_processes(output)
    if main_only:
        roots = {row["process"]["pid"] for row in ownership["jobs"] if alive(row["process"])}
        selected = [identity for identity in processes if identity["pid"] in roots]
        if not selected:
            selected = processes
    else:
        selected = sorted(processes, key=lambda row: row["pid"], reverse=True)
    sent = []
    for identity in selected:
        if not alive(identity):
            continue
        try:
            os.kill(identity["pid"], sig)
            sent.append({"pid": identity["pid"], "start_ticks": identity["start_ticks"], "signal": sig})
        except ProcessLookupError:
            pass
    return sent


def validate_config(config):
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unknown GPU queue configuration")
    graceful, hard = timestamp(config["graceful_stop_at"]), timestamp(config["hard_deadline"])
    if graceful >= hard:
        raise ValueError("graceful stop must precede hard deadline")
    for key, default in (("poll_seconds", 1), ("interrupt_grace_seconds", 120),
                         ("terminate_grace_seconds", 30), ("hard_release_margin_seconds", 5)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"invalid queue timer: {key}")
    if config.get("poll_seconds", 1) > 5:
        raise ValueError("deadline polling may not exceed five seconds")
    if config.get("hard_release_margin_seconds", 5) >= hard - graceful:
        raise ValueError("hard release margin must fit in the stop window")
    if not isinstance(config.get("phases"), list) or not config["phases"]:
        raise ValueError("queue requires sequential phases")
    ids = set()
    for phase in config["phases"]:
        if not phase.get("name") or not isinstance(phase.get("jobs"), list):
            raise ValueError("invalid queue phase")
        for job in phase["jobs"]:
            if not job.get("id") or job["id"] in ids or "/" in job["id"]:
                raise ValueError("job IDs must be unique filenames")
            if not set(job.get("requires_success", [])) <= ids:
                raise ValueError("job dependencies must precede the job")
            ids.add(job["id"])
            if (not isinstance(job.get("command"), list) or not job["command"]
                    or not all(isinstance(arg, str) and arg for arg in job["command"])):
                raise ValueError("job command must be a nonempty argv, not a shell string")
            if not Path(job["cwd"]).is_absolute():
                raise ValueError("job cwd must be absolute")
            if "result_manifest" in job:
                if not Path(job["result_manifest"]).is_absolute() or not job.get("success_statuses"):
                    raise ValueError("result manifest requires an absolute path and success statuses")
            if TOKEN_ENV in job.get("env", {}):
                raise ValueError("jobs cannot override the ownership token")
    wait = config.get("wait_for")
    if wait:
        if not Path(wait["manifest_path"]).is_absolute() or not wait.get("required_status"):
            raise ValueError("waiting requires an absolute manifest and terminal status")
        if not wait.get("writer_lock") and not wait.get("process"):
            raise ValueError("waiting requires a writer lock or PID identity to prove GPU cleanup")
        if wait.get("writer_lock") and not Path(wait["writer_lock"]).is_absolute():
            raise ValueError("writer lock path must be absolute")
        if wait.get("process") and (type(wait["process"].get("pid")) is not int
                                    or type(wait["process"].get("start_ticks")) is not int):
            raise ValueError("wait process requires PID and start_ticks identity")


def wait_ready(config):
    wait = config.get("wait_for")
    if not wait:
        return True, None
    path = Path(wait["manifest_path"])
    if not path.exists():
        return False, "source-manifest-not-present"
    manifest = read(path)
    if manifest.get("status") in {"failed", "preflight_failed", "interrupted"}:
        return False, "source-needs-user-attention"
    if manifest.get("status") != wait["required_status"]:
        return False, "source-not-complete"
    if wait.get("plan_id") and manifest.get("plan_id") != wait["plan_id"]:
        return False, "source-plan-identity-differs"
    if wait.get("process") and alive(wait["process"]):
        return False, "source-process-still-alive"
    if wait.get("writer_lock"):
        try:
            with Path(wait["writer_lock"]).open("r") as handle:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
        except (BlockingIOError, FileNotFoundError):
            return False, "source-writer-lock-not-released"
    return True, None


def stop_requested(output, config):
    return (output / "stop.json").exists() or time.time() >= timestamp(config["graceful_stop_at"])


def request_stop(output, reason):
    path = output / "stop.json"
    if not path.exists():
        atomic(path, {"reason": reason, "requested_at": utc_now()})


def finish_stopped(output, state):
    for row in state["jobs"].values():
        if row["status"] == "queued":
            row.update(status="stopped", reason="not-started-before-reservation-stop")
    state.update(status="stopped", finished_at=utc_now())
    atomic(output / "state.json", state)


def guard(output, job_id):
    require_consistent_pid_namespace()
    config = read(output / "config.json")
    me = process_identity(os.getpid())
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if stop_requested(output, config):
            return 130
        ownership = read(output / "ownership.json")
        rows = [row for row in ownership["jobs"] if row["id"] == job_id
                and row["process"]["pid"] == me["pid"] and row["process"]["start_ticks"] == me["start_ticks"]]
        launch = read(output / "launch.json") if (output / "launch.json").exists() else {}
        if not alive(launch.get("supervisor")) or not alive(launch.get("watchdog")):
            return 130
        if rows and (output / "watchdog-ready.json").exists():
            verify_source_identity(launch)
            if hashlib.sha256((output / "config.json").read_bytes()).hexdigest() != launch["config_sha256"]:
                raise ValueError("registered guard configuration changed")
            row = rows[0]
            if os.environ.get(TOKEN_ENV) != row["token"]:
                raise ValueError("guard ownership token differs")
            job = next(job for phase in config["phases"] for job in phase["jobs"] if job["id"] == job_id)
            os.chdir(job["cwd"])
            env = {**os.environ, **job.get("env", {})}
            os.execvpe(job["command"][0], job["command"], env)
        time.sleep(.05)
    return 130


def verify_source_identity(launch):
    source = Path(__file__).resolve()
    if (Path(launch["source_path"]).resolve() != source
            or hashlib.sha256(source.read_bytes()).hexdigest() != launch["source_sha256"]):
        raise ValueError("registered queue source snapshot changed")


def startup(output, role):
    require_consistent_pid_namespace()
    deadline = time.monotonic() + 30
    me = process_identity(os.getpid())
    while time.monotonic() < deadline:
        path = output / "launch.json"
        if path.exists():
            launch = read(path)
            if launch.get(role, {}).get("pid") != me["pid"] or launch[role]["start_ticks"] != me["start_ticks"]:
                raise ValueError("detached launch process identity differs")
            verify_source_identity(launch)
            return launch
        time.sleep(.05)
    raise RuntimeError("detached startup was not registered")


def supervisor(output):
    launch = startup(output, "supervisor")
    config = read(output / "config.json")
    state = read(output / "state.json")
    config_hash = hashlib.sha256((output / "config.json").read_bytes()).hexdigest()
    if config_hash != launch["config_sha256"]:
        raise ValueError("registered queue configuration changed")
    poll = config.get("poll_seconds", 1)
    lock = (output / ".supervisor.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        while not (output / "watchdog-ready.json").exists():
            if not alive(launch["watchdog"]):
                raise RuntimeError("deadline watchdog failed before queue startup")
            time.sleep(.05)
        state.update(status="waiting", supervisor=launch["supervisor"], watchdog=launch["watchdog"])
        atomic(output / "state.json", state)
        while True:
            if stop_requested(output, config):
                request_stop(output, "reservation-graceful-stop")
                finish_stopped(output, state)
                return 0
            if not alive(launch["watchdog"]):
                raise RuntimeError("deadline watchdog unexpectedly exited")
            ready, reason = wait_ready(config)
            if ready:
                break
            if reason in {"source-needs-user-attention", "source-plan-identity-differs"}:
                state.update(status="blocked", wait_reason=reason, finished_at=utc_now())
                atomic(output / "state.json", state)
                return 2
            if state.get("wait_reason") != reason:
                state["wait_reason"] = reason
                atomic(output / "state.json", state)
            time.sleep(poll)
        for phase in config["phases"]:
            state.update(status="running", phase=phase["name"], wait_reason=None)
            atomic(output / "state.json", state)
            for job in phase["jobs"]:
                if stop_requested(output, config):
                    request_stop(output, "reservation-graceful-stop")
                    finish_stopped(output, state)
                    return 0
                if not alive(launch["watchdog"]):
                    raise RuntimeError("deadline watchdog unexpectedly exited")
                if any(state["jobs"].get(key, {}).get("status") != "succeeded" for key in job.get("requires_success", [])):
                    state["jobs"][job["id"]] = {"status": "skipped", "reason": "dependency-did-not-pass"}
                    atomic(output / "state.json", state)
                    continue
                token = uuid.uuid4().hex
                log_path = output / "logs" / f"{job['id']}.log"
                with log_path.open("ab", buffering=0) as handle:
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_guard",
                        "--output", str(output), "--job", job["id"]], stdin=subprocess.DEVNULL,
                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                        env={**os.environ, TOKEN_ENV: token})
                identity = process_identity(child.pid)
                if not identity:
                    raise RuntimeError("guard exited before its ownership could be registered")
                ownership = read(output / "ownership.json")
                ownership["jobs"].append({"id": job["id"], "token": token, "process": identity,
                                          "registered_at": utc_now()})
                atomic(output / "ownership.json", ownership)
                state["jobs"][job["id"]] = {"status": "running", "process": identity,
                    "started_at": utc_now(), "log_path": str(log_path), "command": job["command"]}
                atomic(output / "state.json", state)
                while child.poll() is None:
                    if not alive(launch["watchdog"]):
                        raise RuntimeError("deadline watchdog unexpectedly exited during a job")
                    time.sleep(poll)
                job_state = state["jobs"][job["id"]]
                result = read(Path(job["result_manifest"])) if job.get("result_manifest") and Path(job["result_manifest"]).exists() else None
                success = child.returncode == 0 and (not job.get("result_manifest") or (
                    result is not None and result.get("status") in job["success_statuses"]
                    and (not job.get("plan_id") or result.get("plan_id") == job["plan_id"])))
                job_state.update(status="succeeded" if success else "stopped" if stop_requested(output, config) else "failed",
                                 exit_code=child.returncode, finished_at=utc_now(),
                                 result_status=result.get("status") if result else None)
                atomic(output / "state.json", state)
                if stop_requested(output, config):
                    finish_stopped(output, state)
                    return 0
                cleanup_deadline = time.monotonic() + config.get("interrupt_grace_seconds", 120)
                while owned_processes(output):
                    if stop_requested(output, config) or time.monotonic() >= cleanup_deadline:
                        raise RuntimeError("job exited with remaining owned descendants")
                    time.sleep(poll)
        state.update(status="complete" if all(row["status"] == "succeeded" for row in state["jobs"].values())
                     else "completed_with_failures", finished_at=utc_now())
        atomic(output / "state.json", state)
        return 0 if state["status"] == "complete" else 2
    except BaseException as error:
        request_stop(output, "supervisor-failed")
        state.update(status="failed", error_type=type(error).__name__, error=str(error), finished_at=utc_now())
        atomic(output / "state.json", state)
        if not alive(launch["watchdog"]):
            # Reciprocal protection if the independent watchdog itself was lost.
            stop_owned(output, config, reason="watchdog-lost")
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def stop_owned(output, config, *, reason):
    request_stop(output, reason)
    start = time.time()
    hard = timestamp(config["hard_deadline"])
    kill_at = min(start + config.get("interrupt_grace_seconds", 120) + config.get("terminate_grace_seconds", 30),
                  hard - config.get("hard_release_margin_seconds", 5))
    term_at = min(start + config.get("interrupt_grace_seconds", 120),
                  kill_at - config.get("terminate_grace_seconds", 30))
    sent = send_owned(output, signal.SIGINT, main_only=True)
    atomic(output / "watchdog-actions.json", {"reason": reason, "started_at": utc_now(), "signals": sent})
    term_sent = False
    while True:
        owned = owned_processes(output)
        if not owned:
            return sent
        now = time.time()
        if now >= kill_at:
            sent.extend(send_owned(output, signal.SIGKILL))
        elif now >= term_at and not term_sent:
            sent.extend(send_owned(output, signal.SIGTERM))
            term_sent = True
        atomic(output / "watchdog-actions.json", {"reason": reason, "started_epoch": start,
               "signals": sent, "remaining": owned, "hard_deadline_epoch": hard})
        time.sleep(min(config.get("poll_seconds", 1), .2))


def watchdog(output):
    launch = startup(output, "watchdog")
    config = read(output / "config.json")
    if hashlib.sha256((output / "config.json").read_bytes()).hexdigest() != launch["config_sha256"]:
        raise ValueError("registered watchdog configuration changed")
    atomic(output / "watchdog-ready.json", {"process": launch["watchdog"], "ready_at": utc_now()})
    atomic(output / "watchdog.json", {"status": "watching", "process": launch["watchdog"]})
    while True:
        state = read(output / "state.json")
        if state["status"] in {"complete", "completed_with_failures", "blocked", "stopped"} and not owned_processes(output):
            atomic(output / "watchdog.json", {"status": "released", "reason": "queue-terminal-no-owned-processes",
                   "released_at": utc_now(), "remaining_owned_processes": []})
            return 0
        reason = ("supervisor-exited" if not alive(launch["supervisor"]) else
                  "stop-requested" if (output / "stop.json").exists() else
                  "reservation-graceful-stop" if time.time() >= timestamp(config["graceful_stop_at"]) else None)
        if reason:
            signals = stop_owned(output, config, reason=reason)
            atomic(output / "watchdog.json", {"status": "released", "reason": reason,
                   "released_at": utc_now(), "remaining_owned_processes": [], "signals": signals})
            return 0
        time.sleep(config.get("poll_seconds", 1))


def launch(config_path, output):
    require_consistent_pid_namespace()
    config = read(config_path)
    validate_config(config)
    if time.time() >= timestamp(config["graceful_stop_at"]):
        raise ValueError("reservation graceful-stop time has already arrived")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    source = output / "queue_source.py"
    source.write_bytes(Path(__file__).resolve().read_bytes())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    atomic(output / "config.json", config)
    config_hash = hashlib.sha256((output / "config.json").read_bytes()).hexdigest()
    atomic(output / "state.json", {"schema_version": SCHEMA, "status": "starting",
           "source_path": str(source), "source_sha256": source_hash,
           "jobs": {job["id"]: {"status": "queued"} for phase in config["phases"] for job in phase["jobs"]},
           "created_at": utc_now()})
    atomic(output / "ownership.json", {"jobs": []})
    processes = {}
    for role in ("watchdog", "supervisor"):
        with (output / "logs" / f"{role}.log").open("ab", buffering=0) as handle:
            child = subprocess.Popen([sys.executable, str(source), "_" + role,
                "--output", str(output)], stdin=subprocess.DEVNULL, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True)
        identity = process_identity(child.pid)
        if identity is None:
            raise RuntimeError(f"detached {role} did not start")
        processes[role] = identity
        # These children deliberately outlive the launcher; the watchdog owns cleanup.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            del child
    receipt = {"schema_version": SCHEMA, "output": str(output), "config_sha256": config_hash,
               "source_path": str(source), "source_sha256": source_hash, "created_at": utc_now(), **processes}
    atomic(output / "launch.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("launch")
    start.add_argument("--config", type=Path, required=True)
    start.add_argument("--output", type=Path, required=True)
    status = sub.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    for name in ("_supervisor", "_watchdog", "_guard"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "_guard":
            command.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    if args.command == "launch":
        result = launch(args.config, args.output)
    elif args.command == "status":
        result = {"queue": read(args.output / "state.json"),
                  "watchdog": read(args.output / "watchdog.json") if (args.output / "watchdog.json").exists() else None,
                  "owned_processes": owned_processes(args.output)}
    elif args.command == "_supervisor":
        return supervisor(args.output)
    elif args.command == "_watchdog":
        return watchdog(args.output)
    else:
        return guard(args.output, args.job)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
