# 配对案例人审页面：SSH 私有转发

通过既有 `aliyun` SSH 身份访问开发机上的三栏人审工作台。

```text
浏览器电脑 127.0.0.1:8772
  -> SSH 本地转发
  -> aliyun 127.0.0.1:18772
  -> 开发机建立的 SSH 反向隧道
  -> 开发机 127.0.0.1:8772
  -> 配对案例人审工作台
```

## 开发机：启动、检查与停止

在项目根目录运行：

```bash
bash deploy/general_model_paired_review/review-forward.sh start
bash deploy/general_model_paired_review/review-forward.sh status
bash deploy/general_model_paired_review/review-forward.sh stop
```

依赖 Bash、tmux、curl、OpenSSH，以及可运行项目的 `/usr/bin/python`（Python ≥3.10）。
复用 `ssh aliyun` 配置和现有密钥，转发只监听回环地址。

本次使用独立 tmux socket `hsd-general-model-paired-review`、session `paired-review`，
`web` 和 `tunnel` 分别管理页面与 SSH 子进程。进程退出后等待 3 秒重启；
SSH 使用已知主机校验、批处理认证、转发失败即退出、30 秒保活和 3 次失败上限。

`start` 可重复运行；已有托管会话时只显示状态，其他进程占用 8772 时拒绝启动。
`stop` 仅结束本次专属会话，保留人工记录。
关闭终端不会停止服务；开发容器重启后需重新运行 `start`。

## 浏览器电脑：连接并打开页面

在**使用浏览器的个人电脑**上运行：

```bash
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8772:127.0.0.1:18772 aliyun
```

随后打开 `http://127.0.0.1:8772/`。保留该 SSH 连接，结束访问时按 `Ctrl+C`。
这条链路使用 SSH 身份验证，供私人访问使用。

浏览器侧使用 8772，以匹配应用允许的 Host/Origin。开发机已经运行 8772 服务，
不要在开发机重复执行这条客户端命令；本地端口被占用时先核对其归属。
Codex 中已经打开的本地预览也可以继续使用。

## 记录与续审

正式记录保存在：

`exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/session.json`

复核人为 `liaozijie`，重启后读取同一份会话。记录和日志均在被 Git 忽略的 `reviews/` 内；
同级 `runtime/` 包含 `web.log`、`tunnel.log`、最近的健康检查与迁移前会话备份。
本次进程恢复验证未写入人工决定，正式记录保持原样。

## 验证与排查

在开发机或已完成本地转发的浏览器电脑上检查：

```bash
curl --noproxy 127.0.0.1 --fail http://127.0.0.1:8772/api/health
```

单独检查 aliyun 上的反向隧道时，使用应用允许的 Host：

```bash
ssh aliyun 'curl --noproxy 127.0.0.1 --fail -H "Host: 127.0.0.1:8772" http://127.0.0.1:18772/api/health'
```

预期返回 `{"stage":"paired-human-review","status":"ok"}`。
开发机健康而远端失败时检查 `runtime/tunnel.log`；两端正常但浏览器无法访问时，
检查个人电脑的 SSH 连接、本地监听端口，以及 Host/Origin 是否匹配。

已验证页面和隧道子进程退出后自动重启、重复启动保护、正式会话保持原样，以及远端静态资源与本地版本一致。
保存、确认和导出等交互测试使用独立测试会话；正式人审记录仅由实际复核操作产生。
