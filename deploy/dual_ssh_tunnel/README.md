# 新容器双跳板 SSH 配置指南

这套工具在**新 Linux 容器**中运行，不要求新容器有公网 IP。容器需要能够访问两台跳板的 TCP 22 端口。

```text
你的电脑 ── aliyun（国内）── SSH 隧道 ── 新容器的 SSH 服务
        └─ sgp（国外）──── SSH 隧道 ── 同一个新容器

隧道由新容器主动连接跳板建立，并持续保持。
```

| 入口 | 位置及流量情况（由服务器所有者提供） | 建议用途 |
| --- | --- | --- |
| aliyun：39.106.154.233 | 国内，有流量费用 | 普通命令执行、VS Code 远程连接，不适合大文件传输 |
| sgp：165.22.48.237 | 国外，无流量限制 | 大文件上传下载，也支持命令执行和 VS Code |

所有经过隧道的数据都会经过对应跳板。VS Code 上传下载、文件同步也产生流量，大文件请选择 **sgp**。两个入口手动选择，不自动切换。

## 1. 准备工具（在新容器操作）

将本目录的 `setup-dual-tunnel.sh` 和 `dual_tunnel.py` **放在同一个目录**，例如 `~/dual-ssh-tunnel/`。保留本指南方便排查。

需要 Bash、Python 3.8+、OpenSSH 客户端、tmux。Python 只使用标准库，无须 pip。检查命令：

```bash
command -v bash python3 ssh tmux
```

预期每个程序显示一个路径。缺少时，根据容器系统选择一条安装命令，有 root 权限可以省略 `sudo`：

```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y bash python3 openssh-client tmux
# RHEL / Fedora
sudo dnf install bash python3 openssh-clients tmux
# Alpine
sudo apk add bash python3 openssh-client tmux
```

无安装权限请联系容器管理员。工具不会自动安装软件或改动 SSH 服务。

## 2. 检查容器的 SSH 服务（在新容器操作）

22 只是默认 SSH 端口，**不能假定它已经可用**。向导会尝试连接 `127.0.0.1:22` 并检查 SSH 协议标识。失败时停止安装，需自行配置后重新运行；若服务使用其他端口，在向导中填写该端口。

可先检查监听端口和当前用户名：

```bash
whoami
ss -lnt
```

如果容器没有 SSH 服务，下面是 **Debian/Ubuntu 容器的手动配置示例**，需要管理员权限，其他系统请使用相应软件包和服务方式：

```bash
sudo apt-get update && sudo apt-get install -y openssh-server
sudo ssh-keygen -A
sudo mkdir -p /run/sshd
sudo /usr/sbin/sshd -t
```

`sshd -t` 没有输出且成功退出表示配置语法正确。根据需要由管理员调整 `/etc/ssh/sshd_config`：端口必须与向导一致，监听地址需包含 `127.0.0.1`，目标账户需允许公钥登录。只通过隧道使用时可监听 `127.0.0.1`；若还有其他访问需求，应保留相应地址。不要覆盖已有配置。

确认没有现有 sshd 监听冲突后，在无 systemd 的容器手动启动：

```bash
sudo /usr/sbin/sshd
ss -lnt
```

启动报错时先修复报错，不要重复启动多个实例。无管理员权限时，请让管理员配置 SSH 服务。容器里的 `systemctl` 可能不可用。

目标登录账户还必须授权**你电脑的公钥**。在电脑查看 `~/.ssh/id_ed25519.pub`，把完整单行公钥追加到新容器**目标用户**的 `~/.ssh/authorized_keys` 中。以目标用户操作，保留已有条目：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
# 使用编辑器向 ~/.ssh/authorized_keys 追加你的电脑公钥，不要粘贴私钥。
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

若使用 root 登录，还需管理员确认 sshd 允许 root 公钥认证；不要为此开放 root 密码登录。工具的端口检测不会验证账号授权，最终需在电脑实际登录验证。

## 3. 准备密钥（分清电脑与新容器）

- **公钥**通常以 `.pub` 结尾，可以交给服务器管理员授权。
- **私钥**通常是对应的不带 `.pub` 的文件，不要发送给别人，也不要粘贴进配置说明。
- 新容器 → 两台跳板：工具需要新容器上一份**已经被两台跳板 root 账户授权**的私钥。
- 电脑 → 跳板、电脑 → 新容器：电脑需要相应私钥，它的公钥须分别被跳板和新容器目标账户授权。前一项能连接，不代表后一项已授权。

前面已添加到两台跳板的公钥，只有持有其**对应私钥**才能使用。把你打算使用的已有私钥通过可信方式放到新容器，记录路径，例如 `~/.ssh/id_ed25519`，设置权限：

```bash
chmod 600 ~/.ssh/id_ed25519
```

如果私钥带口令，可以在新容器先启动并解锁 SSH agent：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

agent 必须在隧道运行期间持续可用。容器重启后需重新解锁；工具不保存私钥口令。更换 agent 后运行 `restart` 使新连接使用新的 agent。无交互的容器启动需要预先解决密钥解锁，不能等待口令输入。

## 4. 运行中文向导（在新容器操作）

```bash
cd ~/dual-ssh-tunnel
bash setup-dual-tunnel.sh
```

依次填写本机 SSH 端口、容器私钥路径、新容器登录用户名、电脑私钥路径和跳板转发端口。默认转发端口为 `6001`；现有服务器正在使用 `6000`，不要填写它。

预期显示本机 SSH、两个跳板密钥登录及端口检查通过，随后保存配置、启动守护进程并输出电脑配置。如果一个入口启动失败，另一个成功入口保持运行；修复后重新 `start`。

两台跳板主机公钥已通过原有可信连接核对并固定在工具内（2026-09-10）。主机密钥不匹配时联系管理员核验，不要关闭主机检查。配置检查会登录跳板执行只读的监听端口检查，不更改远端 SSH 配置、防火墙或授权。

非交互安装示例（将 `YOUR_USER` 换成新容器用户名）：

```bash
bash setup-dual-tunnel.sh install \
  --identity ~/.ssh/id_ed25519 \
  --target-user YOUR_USER \
  --local-port 22 \
  --remote-port 6001 \
  --client-identity '~/.ssh/id_ed25519'
bash setup-dual-tunnel.sh start
bash setup-dual-tunnel.sh status
bash setup-dual-tunnel.sh print-config
```

`--client-identity` 是**电脑路径**，这里的引号用于防止容器提前展开 `~`。实际配置默认保存在 `~/.local/share/dual-ssh-tunnel/`，包含独立管理程序副本、配置、固定主机公钥和日志，不包含复制的私钥。

## 5. 配置并登录（在你的电脑操作）

将 `print-config` 输出追加到电脑 SSH config 中，保留原有内容：

- macOS / Linux：`~/.ssh/config`，可执行 `mkdir -p ~/.ssh` 后用编辑器打开。
- Windows OpenSSH：`C:\Users\你的用户名\.ssh\config`，文件名是 `config`，不是 `config.txt`。PowerShell 可运行 `notepad "$env:USERPROFILE\.ssh\config"`；目录不存在时先创建。

生成配置中的 `IdentityFile` 必须指向**电脑上的私钥**。默认电脑登录跳板和新容器使用同一私钥；若不同，分别修改 `new-server-jump-*` 和 `new-server-via-*` 块中的 `IdentityFile`。Windows 路径可写为 `C:/Users/用户名/.ssh/id_ed25519`。

先在新容器查看主机指纹（可能需要管理员代查）：

```bash
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

电脑首次连接会询问主机指纹，应核对一致再接受。跳板的预期 Ed25519 指纹可在新容器运行以下命令查看：

```bash
ssh-keygen -lf ~/.local/share/dual-ssh-tunnel/known_hosts
```

随后在电脑分别执行：

```bash
ssh new-server-via-aliyun
ssh new-server-via-sgp
```

预期两者都进入**同一个新容器**，可以用 `hostname` 和 `whoami` 核对。`HostKeyAlias` 让两个入口共用该容器的主机身份。若以后配置另一台容器，应同时更换别名及 `HostKeyAlias`，不要让不同容器共用身份。

VS Code：安装 Microsoft 的 Remote - SSH 扩展，打开命令面板，选择 `Remote-SSH: Connect to Host...`，再选择上述任一别名。大文件任务请选择 sgp。

## 6. 大文件通过 sgp 传输（在电脑操作）

先用小文件验证路径和权限，确认后再传大文件。把示例路径换成真实路径：

```bash
# 上传到新容器当前用户的主目录
scp ./example.zip new-server-via-sgp:~/
# 从新容器下载到电脑当前目录
scp new-server-via-sgp:~/example.zip ./
# 交互式文件传输，可用 put 上传、get 下载、bye 退出
sftp new-server-via-sgp
```

## 7. 持久运行和容器重启（在新容器操作）

- 关闭终端或退出登录：tmux 中的隧道继续运行。
- 网络断开：每 30 秒检测连接，连续三次失败断开，每三秒尝试重连。两条隧道独立运行。
- **容器重启**：tmux 和 SSH 服务都会结束，需要通过容器启动入口先恢复 sshd，再运行本工具 `start`。仅安装工具不会自动修改容器启动命令。

日常命令（可选 `aliyun` 或 `sgp` 只管理一个入口，默认两个）：

```bash
bash setup-dual-tunnel.sh status
bash setup-dual-tunnel.sh logs sgp
bash setup-dual-tunnel.sh restart sgp
bash setup-dual-tunnel.sh stop aliyun
bash setup-dual-tunnel.sh start aliyun
```

状态中的“守护进程运行，正在连接”不代表转发成功。“SSH 转发已建立”代表跳板接受了转发请求，电脑最终登录仍取决于目标服务和账号认证。

启动入口集成示例：请容器维护者把以下逻辑接入**现有启动脚本**，保留原来的主进程命令。示例中的 `APP_USER`、`/home/APP_USER` 和 `原有主进程及参数` 必须替换，不要原样执行。无 root 权限时由平台启动 sshd，跳过第一步。

```bash
# 示例：容器以 root 初始化，尚未启动 sshd
mkdir -p /run/sshd
/usr/sbin/sshd -t && /usr/sbin/sshd
# 用安装工具的同一用户启动；私钥与配置必须已挂载且可读。
# 容器刚重启时网络可能尚未就绪，在后台重试，不阻塞应用启动。
su - APP_USER -c 'while ! python3 /home/APP_USER/.local/share/dual-ssh-tunnel/manager.py start; do sleep 3; done' &
exec 原有主进程及参数
```

如果安装用户就是容器当前用户，可直接使用 `python3 ~/.local/share/dual-ssh-tunnel/manager.py start`。容器主进程退出时容器通常会结束，tmux 不能让已停止的容器继续运行。

配置目录、原私钥文件和新容器 `/etc/ssh/ssh_host_*` 主机密钥需要持久保存；不要持久化 `/tmp` 中的运行套接字。否则容器重建后配置丢失或主机指纹变化。自定义状态目录时，所有命令均在子命令前加 `--state-dir /持久路径`。

## 常见问题

| 现象 | 下一步 |
| --- | --- |
| 本机连接被拒绝 | 手动启动 sshd，核对监听地址/端口，再运行向导 |
| 本机端口返回非 SSH 标识 | 该端口可能是 Web 等服务，填写真正的 SSH 端口 |
| 连接/SSH 标识超时 | 检查服务日志、监听地址及网络策略，不能只看端口有监听 |
| `Permission denied (publickey)` | 检查发生在跳板还是目标容器，核对对应私钥、公钥授权、用户名和文件权限 |
| 跳板连接超时 | 检查容器到该跳板公网 TCP 22 的出站连通性 |
| `6001` 已占用 | 选择其他空闲端口；工具不终止占用者，修改后重新生成电脑配置 |
| 运行中修改配置被拒绝 | 先 `stop`，再 `install` 和 `start` |
| 主机指纹变化 | 联系管理员核验是否重装或更换密钥，不要直接关闭校验 |
| 守护进程运行但一直未连接 | 查看对应入口 `logs`，检查网络、认证、端口冲突和 agent |
| 容器重启后失联 | 恢复 sshd、持久化配置/私钥以及必要的 agent，再运行 `start` |
| ssh 成功但 SFTP 失败 | 由管理员检查新容器 sshd 的 SFTP 子系统配置 |

停止工具只终止本工具的独立 tmux 会话及其 SSH 子进程，不操作其他 SSH 隧道，也不改动两台跳板上的业务服务。

## 维护者测试

在仓库根目录运行下列命令。集成测试需要 root、`/usr/sbin/sshd`、`ssh-keygen`、`ss` 和 tmux，以及允许回环网络；所有主机、密钥、转发端口和 tmux 会话均为临时隔离资源，不连接正式跳板。

```bash
bash -n deploy/dual_ssh_tunnel/setup-dual-tunnel.sh
DUAL_TUNNEL_INTEGRATION=1 python3 -m unittest discover -s deploy/dual_ssh_tunnel -p 'test_*.py' -v
```

测试覆盖拒绝连接、非 SSH 服务、超时、自定义端口、安装/启动门禁，以及真实 OpenSSH 的双转发登录、就绪状态、重复安装/启动、单入口重连、端口冲突和停止清理。未设置环境变量时跳过真实 sshd 集成测试，其余检测测试仍需回环网络。
