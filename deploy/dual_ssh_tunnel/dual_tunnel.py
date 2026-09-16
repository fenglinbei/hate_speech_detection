#!/usr/bin/env python3
"""User-owned, dual reverse SSH tunnels. Standard library only; Linux containers."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time


RELAYS = {
    "aliyun": ("39.106.154.233", "AAAAC3NzaC1lZDI1NTE5AAAAIM/oz0kPp8gFaB/dhCdgYb8ZkGK5pnLleqDG7k9t2kyx"),
    "sgp": ("165.22.48.237", "AAAAC3NzaC1lZDI1NTE5AAAAIH7hYJN5v7NBTB7uVhPmRbqyNrhE3Riem+QwVF3OGO9t"),
}
# Host keys read through existing, strictly verified SSH connections on 2026-09-10.
USAGE_NOTE = "aliyun：国内、有流量费用，适合命令/VS Code；sgp：国外、无流量限制，适合大文件。"
SSH_FIX = "请自行检查/启动容器 SSH 服务或指定正确端口，再重新运行。参见 README.md 的 SSH 配置步骤。"


class Failure(Exception):
    pass


def run(argv, timeout=20):
    try:
        return subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise Failure("操作超时：请检查网络或服务状态。") from exc


def probe_ssh(port, timeout=4):
    """Validate an SSH banner, including legal pre-banner lines, within one deadline."""
    deadline = time.monotonic() + timeout
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
            data = b""
            while len(data) < 8192:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                conn.settimeout(remaining)
                chunk = conn.recv(512)
                if not chunk:
                    break
                data += chunk
                for line in data.split(b"\n")[:-1]:
                    if line.startswith((b"SSH-2.0-", b"SSH-1.99-")):
                        return
                if data.startswith(b"HTTP/"):
                    break
    except ConnectionRefusedError as exc:
        raise Failure(f"127.0.0.1:{port} 连接被拒绝，可能没有 SSH 服务监听。{SSH_FIX}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise Failure(f"127.0.0.1:{port} 连接或等待 SSH 标识超时。{SSH_FIX}") from exc
    except OSError as exc:
        raise Failure(f"127.0.0.1:{port} 不可达：{exc}。{SSH_FIX}") from exc
    raise Failure(f"127.0.0.1:{port} 未返回有效 SSH 标识，可能是其他服务。{SSH_FIX}")


def dependencies():
    missing = [name for name in ("ssh", "tmux") if not shutil.which(name)]
    if missing:
        raise Failure("缺少依赖：" + ", ".join(missing) +
                      "。Debian/Ubuntu：sudo apt-get update && sudo apt-get install -y openssh-client tmux python3；"
                      "RHEL/Fedora：sudo dnf install openssh-clients tmux python3；"
                      "Alpine：sudo apk add bash openssh-client tmux python3。无 sudo 时请联系管理员。")


def port_number(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("端口必须是整数") from exc
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("端口必须在 1–65535 之间")
    return number


def atomic_write(path, text):
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Manager:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        digest = hashlib.sha256(str(self.root).encode()).hexdigest()[:12]
        # Unix sockets have short path limits. Dedicated UID-owned directory in /tmp.
        self.runtime = Path(tempfile.gettempdir()) / f"dual-ssh-{os.getuid()}-{digest}"

    def prepare(self):
        for path in (self.root, self.runtime):
            if path.is_symlink():
                raise Failure(f"拒绝使用符号链接运行目录：{path}")
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            if path.stat().st_uid != os.getuid():
                raise Failure(f"目录不属于当前用户：{path}")
            path.chmod(0o700)

    @contextlib.contextmanager
    def locked(self):
        self.prepare()
        with (self.root / "operation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def config(self):
        path = self.root / "config.json"
        if not path.is_file():
            raise Failure("尚未安装，请先运行 setup-dual-tunnel.sh install，或无参数启动向导。")
        return json.loads(path.read_text())

    def tmux(self, *args):
        return run(["tmux", "-S", str(self.runtime / "tmux.sock"), "-f", "/dev/null", *args])

    def alive(self, relay):
        return self.tmux("has-session", "-t", "=" + relay).returncode == 0

    def control(self, relay):
        return str(self.runtime / (relay + ".sock"))

    def ssh(self, config, relay, known_hosts=None):
        host = RELAYS[relay][0]
        return ["ssh", "-F", "/dev/null", "-i", config["identity"], "-p", "22",
                "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", "HostKeyAlgorithms=ssh-ed25519",
                "-o", "UserKnownHostsFile=" + str(known_hosts or self.root / "known_hosts"),
                "-o", "GlobalKnownHostsFile=/dev/null", "-o", "UpdateHostKeys=no",
                "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=3", "-o", "ControlMaster=no",
                "-o", "ControlPath=none", "-o", "ExitOnForwardFailure=yes",
                "-o", "ForwardAgent=no", "-o", "PermitLocalCommand=no",
                "-o", "RequestTTY=no", "root@" + host]

    def preflight(self, config, relay, known_hosts=None):
        # Port is parsed as an integer; command contains no user-controlled shell text.
        command = "command -v ss >/dev/null || exit 71; ss -H -lnt 'sport = :%d'" % config["remote_port"]
        result = run(self.ssh(config, relay, known_hosts) + [command])
        if result.returncode:
            raise Failure(f"{relay} 检查失败：{result.stderr.strip() or '需要跳板安装 ss（iproute2）'}。"
                          "检查私钥授权/权限、网络、主机密钥；加密私钥需已解锁的 SSH agent。")
        if result.stdout.strip():
            raise Failure(f"{relay} 的 {config['remote_port']} 端口已占用；请选择其他端口，不会停止占用者。")

    def install(self, args):
        dependencies()
        probe_ssh(args.local_port)
        print(f"[通过] 本机 {args.local_port} 提供 SSH 服务；尚未验证目标账户登录。", flush=True)
        identity = Path(args.identity).expanduser().resolve()
        if not identity.is_file() or identity.name.endswith(".pub"):
            raise Failure("请提供存在的私钥文件路径，不能填写 .pub 公钥。")
        if identity.stat().st_mode & 0o077:
            raise Failure(f"私钥权限过宽，请执行 chmod 600 {shlex.quote(str(identity))} 后重试。")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*\$?", args.target_user):
            raise Failure("登录用户名格式无效。请填写新容器中实际存在的 Linux 用户名。")
        if any(c in args.client_identity for c in '\r\n\x00"'):
            raise Failure("电脑私钥路径不能包含换行或双引号。")
        if args.remote_port == 6000:
            raise Failure("6000 已用于原服务器，请使用 6001 或其他空闲端口。")
        config = dict(identity=str(identity), local_port=args.local_port,
                      remote_port=args.remote_port, target_user=args.target_user,
                      client_identity=args.client_identity)
        with self.locked():
            if any(self.alive(r) for r in RELAYS):
                if self.config() == config:
                    print("相同配置已经运行，无需重复安装。")
                    return
                raise Failure("隧道正在运行；如需修改配置，请先 stop，再 install。")
            hosts = "".join(f"{host} ssh-ed25519 {key}\n" for host, key in RELAYS.values())
            with tempfile.NamedTemporaryFile(mode="w", dir=self.root) as known:
                known.write(hosts)
                known.flush()
                for relay in RELAYS:
                    self.preflight(config, relay, known.name)
                    print(f"[通过] {relay} 密钥登录正常，转发端口空闲。", flush=True)
            # Keep an installed copy so moving the downloaded files does not break restarts.
            atomic_write(self.root / "manager.py", Path(__file__).read_text())
            atomic_write(self.root / "known_hosts", hosts)
            atomic_write(self.root / "config.json", json.dumps(config, indent=2) + "\n")
        print("配置已保存：" + str(self.root))
        print(USAGE_NOTE)
        print("下一步：运行 start，再将 print-config 输出追加到电脑 SSH config，验证两个入口。")

    def start(self, relays):
        dependencies()
        with self.locked():
            config = self.config()
            probe_ssh(config["local_port"])
            failures = []
            for relay in relays:
                if self.alive(relay):
                    print(f"{relay}：守护进程已运行，不重复启动。")
                    continue
                try:
                    self.preflight(config, relay)
                    command = shlex.join([sys.executable, str(self.root / "manager.py"),
                                          "--state-dir", str(self.root), "_run", relay])
                    # Update the agent environment even if the other session kept this server alive.
                    env = ["-e", "SSH_AUTH_SOCK=" + os.environ.get("SSH_AUTH_SOCK", "")]
                    result = self.tmux("new-session", "-d", *env, "-s", relay, command)
                    if result.returncode:
                        raise Failure(result.stderr.strip())
                    print(f"{relay}：守护进程已启动。", flush=True)
                except Failure as exc:
                    failures.append(f"{relay}：{exc}")
            if failures:
                raise Failure("\n".join(failures) + "\n其他成功启动的入口保持运行；修复后重新 start。")
        print("运行 status 查看连接状态，并在电脑端验证登录。关闭终端后继续运行；容器重启后需再次 start。")

    def stop(self, relays):
        with self.locked():
            for relay in relays:
                if self.alive(relay):
                    result = self.tmux("kill-session", "-t", "=" + relay)
                    if result.returncode:
                        raise Failure(result.stderr.strip())
                # Wait for the runner's signal handler to reap its own SSH child.
                with (self.root / (relay + ".lock")).open("a") as lock:
                    deadline = time.monotonic() + 8
                    while True:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() > deadline:
                                raise Failure(f"{relay} 停止尚未完成，请查看日志后重试。")
                            time.sleep(0.1)
                print(f"{relay}：已停止。")

    def status(self, relays):
        config = self.config()
        try:
            probe_ssh(config["local_port"])
            print("本机 SSH：可达（未验证账户认证）。")
        except Failure as exc:
            print("本机 SSH：" + str(exc))
        for relay in relays:
            if not self.alive(relay):
                state = "守护进程未运行"
            else:
                result = run(["ssh", "-F", "/dev/null", "-S", self.control(relay),
                              "-O", "check", "root@" + RELAYS[relay][0]], timeout=5)
                ready = self.root / (relay + ".ready")
                state = ("SSH 转发已建立；电脑端登录仍需验证" if result.returncode == 0 and ready.exists()
                         else "守护进程运行，正在连接或等待重试；请查看日志")
            print(relay + "：" + state)
        print(USAGE_NOTE)

    def runner(self, relay):
        self.prepare()
        config = self.config()
        with (self.root / (relay + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            stopping = False
            child = None

            def stop_signal(signum, frame):
                nonlocal stopping
                stopping = True
                if child is not None and child.poll() is None:
                    child.terminate()

            for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                signal.signal(sig, stop_signal)
            ready = self.root / (relay + ".ready")
            try:
                while not stopping:
                    ready.unlink(missing_ok=True)
                    # Previous runner is excluded by flock; only remove our own stale socket.
                    Path(self.control(relay)).unlink(missing_ok=True)
                    log = self.root / (relay + ".log")
                    if log.exists() and log.stat().st_size > 2 * 1024 * 1024:
                        log.replace(self.root / (relay + ".log.1"))
                    with log.open("a") as output:
                        output.write(time.strftime("%Y-%m-%d %H:%M:%S") + " connecting\n")
                        output.flush()
                        # Start an authenticated master, then explicitly request the forward.
                        # The control request succeeds only after the server accepts the listener.
                        base = self.ssh(config, relay)
                        destination = base.pop()
                        base[base.index("ControlMaster=no")] = "ControlMaster=yes"
                        base[base.index("ControlPath=none")] = "ControlPath=" + self.control(relay)
                        command = base + ["-nNT", destination]
                        child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                                 stdout=output, stderr=output)
                        if stopping:
                            child.terminate()
                        deadline = time.monotonic() + 15
                        while not stopping and child.poll() is None and time.monotonic() < deadline:
                            if Path(self.control(relay)).exists():
                                break
                            time.sleep(0.1)
                        if not stopping and child.poll() is None:
                            try:
                                result = run(["ssh", "-F", "/dev/null", "-S", self.control(relay),
                                              "-O", "forward", "-R",
                                              f"127.0.0.1:{config['remote_port']}:127.0.0.1:{config['local_port']}",
                                              destination], timeout=5)
                                if result.returncode:
                                    raise Failure(result.stderr.strip())
                                ready.touch(mode=0o600)
                                output.write("Remote forward accepted\n")
                                output.flush()
                            except Failure as exc:
                                output.write(str(exc) + "\n")
                                child.terminate()
                        while child.poll() is None:
                            if stopping:
                                try:
                                    child.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    child.kill()
                            else:
                                time.sleep(0.2)
                        child.wait()
                        ready.unlink(missing_ok=True)
                        output.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                                     f" SSH exited {child.returncode}; retry in 3s\n")
                    for _ in range(30):
                        if stopping:
                            break
                        time.sleep(0.1)
            finally:
                if child is not None and child.poll() is None:
                    child.kill()
                    child.wait()
                ready.unlink(missing_ok=True)

    def print_config(self):
        config = self.config()
        print("# 将以下片段追加到你电脑的 ~/.ssh/config；路径指电脑上的私钥。")
        print("# " + USAGE_NOTE)
        for relay, (host, _) in RELAYS.items():
            print(f'''\nHost new-server-jump-{relay}
    HostName {host}
    User root
    Port 22
    IdentityFile "{config['client_identity']}"
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3

Host new-server-via-{relay}
    HostName 127.0.0.1
    Port {config['remote_port']}
    User {config['target_user']}
    ProxyJump new-server-jump-{relay}
    HostKeyAlias new-server-dual-tunnel
    IdentityFile "{config['client_identity']}"
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3''')


def parser():
    result = argparse.ArgumentParser(description="双跳板 SSH 隧道工具（在新 Linux 容器运行）。" + USAGE_NOTE)
    result.add_argument("--state-dir", default="~/.local/share/dual-ssh-tunnel", help="独立配置目录；放在子命令之前")
    commands = result.add_subparsers(dest="command", metavar="COMMAND")
    install = commands.add_parser("install", help="检查并安装配置，不启动隧道")
    install.add_argument("--identity", required=True, help="新容器上的已有私钥路径，不是 .pub")
    install.add_argument("--target-user", required=True, help="电脑登录新容器时使用的用户名")
    install.add_argument("--local-port", default=22, type=port_number)
    install.add_argument("--remote-port", default=6001, type=port_number)
    install.add_argument("--client-identity", default="~/.ssh/id_ed25519", help="电脑上的私钥路径")
    descriptions = {"start": "检查并启动隧道", "stop": "停止本工具的隧道", "restart": "重新启动",
                    "status": "检查本机 SSH 和隧道状态", "logs": "显示最近日志"}
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description)
        command.add_argument("relay", nargs="?", default="all", choices=["all", *RELAYS])
    commands.add_parser("print-config", help="输出电脑 SSH config 片段")
    commands.add_parser("help", help="显示命令帮助")
    internal = commands.add_parser("_run")
    internal.add_argument("relay", choices=list(RELAYS))
    return result


def wizard(manager):
    if not sys.stdin.isatty():
        raise Failure("非交互环境请使用 install --identity 私钥路径 --target-user 用户名；使用 help 查看命令。")
    print("在新容器上配置：新容器主动连接两个公网跳板，电脑再经跳板登录新容器。")
    print(USAGE_NOTE)
    print("先检查本机 SSH；失败时需要你自行配置，工具不会修改 SSH 服务。")
    local = port_number(input("本机 SSH 端口 [22]：").strip() or "22")
    probe_ssh(local)
    identity = input("新容器上已授权给两台跳板的私钥路径（不是 .pub）：").strip()
    target = input("电脑登录新容器的用户名（在容器执行 whoami 可查看当前用户）：").strip()
    client = input("电脑上的登录私钥路径 [~/.ssh/id_ed25519]：").strip() or "~/.ssh/id_ed25519"
    remote = port_number(input("两台跳板上的空闲转发端口 [6001]：").strip() or "6001")
    manager.install(argparse.Namespace(identity=identity, target_user=target, client_identity=client,
                                       local_port=local, remote_port=remote))
    manager.start(list(RELAYS))
    manager.print_config()
    print("请在电脑上分别测试 ssh new-server-via-aliyun 和 ssh new-server-via-sgp。")


def main(argv=None):
    os.umask(0o077)
    cli = parser()
    args = cli.parse_args(argv)
    manager = Manager(args.state_dir)
    try:
        if args.command == "help":
            cli.print_help()
        elif args.command is None:
            wizard(manager)
        elif args.command == "install":
            manager.install(args)
        elif args.command == "print-config":
            manager.print_config()
        elif args.command == "_run":
            manager.runner(args.relay)
        else:
            relays = list(RELAYS) if args.relay == "all" else [args.relay]
            if args.command == "logs":
                for relay in relays:
                    path = manager.root / (relay + ".log")
                    print(f"--- {relay}: {path}（最近 60 行）---")
                    print("\n".join(path.read_text(errors="replace").splitlines()[-60:]) if path.exists() else "暂无日志")
            elif args.command == "restart":
                manager.stop(relays)
                manager.start(relays)
            else:
                getattr(manager, args.command)(relays)
    except (Failure, OSError, ValueError, argparse.ArgumentTypeError) as exc:
        print("错误：" + str(exc), file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("已取消。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
