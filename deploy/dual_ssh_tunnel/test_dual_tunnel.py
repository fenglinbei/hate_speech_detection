"""Isolated tests. Integration uses temporary local sshd, never the real relays."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import dual_tunnel as d


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def banner_server(banner, delay=0):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(2)
    port = listener.getsockname()[1]

    def serve():
        try:
            conn, _ = listener.accept()
            with conn:
                time.sleep(delay)
                conn.sendall(banner)
        except OSError:
            pass

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        yield port
    finally:
        listener.close()
        thread.join(timeout=3)


class DetectionTests(unittest.TestCase):
    def test_custom_ssh_port_with_prebanner(self):
        with banner_server(b"Welcome\r\nSSH-2.0-test\r\n") as port:
            d.probe_ssh(port)

    def test_refused(self):
        with self.assertRaisesRegex(d.Failure, "连接被拒绝"):
            d.probe_ssh(free_port())

    def test_non_ssh(self):
        with banner_server(b"HTTP/1.0 200 OK\r\n") as port:
            with self.assertRaisesRegex(d.Failure, "未返回有效 SSH"):
                d.probe_ssh(port)

    def test_timeout(self):
        with banner_server(b"", delay=0.3) as port:
            with self.assertRaisesRegex(d.Failure, "超时"):
                d.probe_ssh(port, timeout=0.05)

    def test_failed_probe_does_not_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = d.Manager(Path(tmp) / "state")
            with patch.object(d, "dependencies"), patch.object(d, "probe_ssh", side_effect=d.Failure("bad SSH")):
                with self.assertRaises(d.Failure):
                    manager.install(argparse.Namespace(local_port=22))
            self.assertFalse((manager.root / "config.json").exists())

    def test_failed_probe_does_not_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = d.Manager(Path(tmp) / "state")
            with patch.object(d, "dependencies"), patch.object(manager, "config", return_value={"local_port": 22}), \
                    patch.object(d, "probe_ssh", side_effect=d.Failure("bad SSH")), patch.object(manager, "tmux") as tmux:
                with self.assertRaises(d.Failure):
                    manager.start(list(d.RELAYS))
                tmux.assert_not_called()
            shutil.rmtree(manager.runtime, ignore_errors=True)

    def test_invalid_port(self):
        for value in ("0", "65536", "abc"):
            with self.assertRaises(argparse.ArgumentTypeError):
                d.port_number(value)


@unittest.skipUnless(os.environ.get("DUAL_TUNNEL_INTEGRATION") == "1",
                     "Set DUAL_TUNNEL_INTEGRATION=1 for isolated local sshd test (root required)")
class IntegrationTests(unittest.TestCase):
    def test_lifecycle_and_reconnect(self):
        self.assertEqual(os.getuid(), 0, "Temporary sshd fixture requires root")
        with tempfile.TemporaryDirectory(prefix="dual-tunnel-test-") as tmp:
            root = Path(tmp)
            key = root / "key"
            ssh = shutil.which("ssh")
            for name in ("key", "host"):
                subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(root / name)], check=True)
            port, local_port, aliyun_port, sgp_port = [free_port() for _ in range(4)]
            self.assertEqual(len({port, local_port, aliyun_port, sgp_port}), 4)
            # One sshd also serves as the target; two remote forwards use distinct fixture ports.
            config = root / "sshd_config"
            config.write_text(f"""ListenAddress 127.0.0.1
Port {port}
Port {local_port}
HostKey {root / 'host'}
PidFile {root / 'sshd.pid'}
AuthorizedKeysFile {key}.pub
StrictModes no
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin yes
UsePAM no
AllowTcpForwarding yes
GatewayPorts no
LogLevel ERROR
""")
            known = root / "known_hosts"
            known.write_text(f"[127.0.0.1]:{port} " + (root / "host.pub").read_text())
            bindir = root / "bin"
            bindir.mkdir()
            # Only map endpoints to the temporary loopback fixture. All SSH is real OpenSSH.
            shim = bindir / "ssh"
            shim.write_text(f"""#!/usr/bin/env python3
import os, sys
a = sys.argv[1:]
relay = 'aliyun' if any('39.106.154.233' in x for x in a) else 'sgp'
forward_port = {{'aliyun': {aliyun_port}, 'sgp': {sgp_port}}}[relay]
for i, value in enumerate(a):
    if value.startswith('UserKnownHostsFile='):
        a[i] = 'UserKnownHostsFile=' + {str(known)!r}
    elif value in ('root@39.106.154.233', 'root@165.22.48.237'):
        a[i] = 'root@127.0.0.1'
    elif value == '22' and i and a[i-1] == '-p':
        a[i] = {str(port)!r}
    elif value.startswith('127.0.0.1:6001:'):
        a[i] = value.replace(':6001:', ':' + str(forward_port) + ':', 1)
    elif 'sport = :6001' in value:
        a[i] = value.replace('sport = :6001', 'sport = :' + str(forward_port))
os.execv({ssh!r}, [{ssh!r}] + a)
""")
            shim.chmod(0o700)
            manager = d.Manager(root / "state")
            daemon_log = (root / "sshd.log").open("w")
            daemon = None

            def launch():
                return subprocess.Popen(["/usr/sbin/sshd", "-D", "-e", "-f", str(config)],
                                        stdout=daemon_log, stderr=daemon_log)

            def wait_for(predicate, description, timeout=18):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if predicate():
                        return
                    time.sleep(0.2)
                self.fail(description + "\n" + "\n".join(
                    p.read_text() for p in manager.root.glob("*.log")) +
                    "\nsshd: " + (root / "sshd.log").read_text())

            def reachable(p):
                try:
                    d.probe_ssh(p, timeout=0.3)
                    return True
                except d.Failure:
                    return False

            with patch.dict(os.environ, {"PATH": str(bindir) + os.pathsep + os.environ["PATH"]}):
                try:
                    daemon = launch()
                    wait_for(lambda: reachable(local_port), "sshd did not start")
                    args = argparse.Namespace(identity=str(key), target_user="root", local_port=local_port,
                                              remote_port=6001, client_identity="~/.ssh/id_ed25519")
                    manager.install(args)
                    original = (manager.root / "config.json").read_bytes()
                    manager.install(args)
                    self.assertEqual(original, (manager.root / "config.json").read_bytes())
                    manager.start(list(d.RELAYS))
                    wait_for(lambda: reachable(aliyun_port) and reachable(sgp_port), "both tunnels not ready")
                    wait_for(lambda: all((manager.root / (r + ".ready")).exists() for r in d.RELAYS),
                             "forward readiness not recorded")
                    status = io.StringIO()
                    with contextlib.redirect_stdout(status):
                        manager.status(list(d.RELAYS))
                    self.assertEqual(status.getvalue().count("SSH 转发已建立"), 2)
                    manager.start(list(d.RELAYS))
                    manager.install(args)  # Same config while running is idempotent.
                    sessions = manager.tmux("list-sessions", "-F", "#{session_name}").stdout.splitlines()
                    self.assertEqual(sorted(sessions), ["aliyun", "sgp"])
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        manager.print_config()
                    client_config = root / "client-config"
                    client_config.write_text(output.getvalue())
                    for relay in d.RELAYS:
                        rendered = subprocess.run([ssh, "-G", "-F", str(client_config), "new-server-via-" + relay],
                                                  text=True, capture_output=True, check=True).stdout
                        self.assertIn("hostkeyalias new-server-dual-tunnel", rendered)
                        self.assertIn("proxyjump new-server-jump-" + relay, rendered)
                    # Verify real authentication and command execution through each forward.
                    for forward in (aliyun_port, sgp_port):
                        result = subprocess.run([ssh, "-F", "/dev/null", "-i", str(key), "-p", str(forward),
                                                 "-o", "StrictHostKeyChecking=accept-new",
                                                 "-o", "UserKnownHostsFile=" + str(root / "target_hosts"),
                                                 "-o", "BatchMode=yes", "root@127.0.0.1", "printf tunnel-ok"],
                                                capture_output=True, text=True, timeout=10)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stdout, "tunnel-ok")
                    # Terminate only aliyun's authenticated master; supervisor must reconnect.
                    subprocess.run([ssh, "-F", "/dev/null", "-S", manager.control("aliyun"),
                                    "-O", "exit", "root@127.0.0.1"], check=True, capture_output=True)
                    wait_for(lambda: not reachable(aliyun_port), "aliyun did not disconnect")
                    self.assertTrue(reachable(sgp_port))
                    wait_for(lambda: reachable(aliyun_port), "aliyun did not reconnect")
                    manager.stop(["aliyun"])
                    self.assertFalse(reachable(aliyun_port))
                    self.assertTrue(reachable(sgp_port))
                    with socket.socket() as occupied:
                        occupied.bind(("127.0.0.1", aliyun_port))
                        occupied.listen()
                        with self.assertRaisesRegex(d.Failure, "已占用"):
                            manager.start(["aliyun"])
                    manager.start(["aliyun"])
                    wait_for(lambda: reachable(aliyun_port), "restart after conflict failed")
                    manager.stop(list(d.RELAYS))
                    manager.stop(list(d.RELAYS))
                    self.assertFalse(reachable(aliyun_port))
                    self.assertFalse(reachable(sgp_port))
                finally:
                    if manager.runtime.exists():
                        with contextlib.suppress(Exception):
                            manager.stop(list(d.RELAYS))
                        manager.tmux("kill-server")
                    if daemon:
                        daemon.terminate()
                        daemon.wait(timeout=5)
                    daemon_log.close()
                    shutil.rmtree(manager.runtime, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
