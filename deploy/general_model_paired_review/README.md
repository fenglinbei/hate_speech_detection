# 配对案例人审服务：访问与迁移

**当前正式记录仍在开发机，原服务和 aliyun 反向隧道可继续使用。** DigitalOcean 部署已准备，
真实会话尚待明确授权迁移；`https://hsd.fenglin.pro/` 目前返回维护响应 503。
远端正式 unit 已安装但未启用、未运行，也没有正式会话。详见 [DigitalOcean 部署说明](digitalocean-sgp/README.md)。

## 当前使用：aliyun 私有转发

```text
浏览器电脑 127.0.0.1:8772
  -> SSH 本地转发 -> aliyun 127.0.0.1:18772
  -> 开发机反向隧道 -> 开发机 127.0.0.1:8772 -> 当前正式会话
```

在项目根目录管理开发机服务和反向隧道：

```bash
bash deploy/general_model_paired_review/review-forward.sh start
bash deploy/general_model_paired_review/review-forward.sh status
bash deploy/general_model_paired_review/review-forward.sh stop
```

依赖 Bash、tmux、curl、OpenSSH，以及 `/usr/bin/python`（Python ≥3.10）。
独立 tmux socket 为 `hsd-general-model-paired-review`，session 为 `paired-review`；
`web` 和 `tunnel` 分别管理页面与隧道。关闭终端不停止服务，容器重启后重新运行 `start`。
已有托管会话不重复启动；端口被其他进程占用时拒绝启动，`stop` 只结束本任务会话。

在**使用浏览器的电脑**上运行：

```bash
ssh -NT -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8772:127.0.0.1:18772 aliyun
```

打开 `http://127.0.0.1:8772/`，保留 SSH 连接，结束访问时按 `Ctrl+C`。
开发机本身已占用 8772，不要在开发机重复执行浏览器电脑的转发命令。

## 记录与迁移保护

当前正式记录：

```text
exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/session.json
```

复核人为 `liaozijie`，重启读取同一会话。2026-09-08 检查时已有 3 条确认，实际进度以页面为准。
记录、备份与 `runtime/` 日志均在被 Git 忽略的 `reviews/` 内；自动测试不得填写正式会话。

得到真实会话迁移授权后，先停旧写入，备份、迁移并校验最新会话，验证远端重启续读，
再将远端设为唯一正式服务。本地副本保留为备份，并在旁边建立 `session.json.remote-authority.json`。
存在标记时，CLI 和旧脚本的 `start`、`run-web` 会拒绝旧写入并提示新入口。
当前尚未建立此标记；它不随正式会话复制到远端，也不应通过删除标记来恢复陈旧进度。

## 切换完成后的访问

公开入口为 `https://hsd.fenglin.pro/`，由 Nginx 提供 TLS 与登录验证。
私人登录文件仅保存在被忽略的
`exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/runtime/digitalocean-login.json`；
不要将内容写入文档、命令参数、日志或 Git。

切换后如需本开发机的 `http://127.0.0.1:8772/` 入口：

```bash
bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh start
bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh status
bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh stop
```

该脚本只转发本机 `127.0.0.1:8772` 至 `digitalocean-sgp 127.0.0.1:8772`，不运行本地 writer。
需要 Bash、tmux、curl、OpenSSH、Python 3 和 `flock`。独立 socket 为
`hsd-general-model-paired-review-sgp-private`，session 为 `paired-review-private-access`；
日志位于 `runtime/digitalocean-sgp-private-access/`。`run` 为前台模式，以 `Ctrl+C` 结束。

两种托管转发都使用已知主机校验、批处理认证、30 秒保活、3 次失败上限及 3 秒重连。
本地端口须保持 8772，以匹配应用允许的 Host/Origin；容器重启后重新运行相应 `start`。

## 健康检查与回退

在开发机或已完成转发的浏览器电脑上检查：

```bash
curl --noproxy 127.0.0.1 --fail http://127.0.0.1:8772/api/health
```

切换前单独检查 aliyun 反向监听：

```bash
ssh aliyun 'curl --noproxy 127.0.0.1 --fail -H "Host: 127.0.0.1:8772" http://127.0.0.1:18772/api/health'
```

健康响应为 `{"stage":"paired-human-review","status":"ok"}`。
旧隧道异常看 `runtime/tunnel.log`；切换后的直接转发看
`runtime/digitalocean-sgp-private-access/tunnel.log`。

代码回退必须保留最新的唯一正式会话。远端成为正式服务后，只回退代码，不能重新启动本机旧备份。
如需迁回本机，应停远端写入、迁回其最新记录并验证完整状态。保存、确认和导出测试均用隔离会话。
