# 配对案例人审服务：访问与续审

**正式服务已迁至 `digitalocean-sgp`（165.22.48.237），打开 https://hsd.fenglin.pro/ 即可登录复核。**
2026-09-08 切换保留首批 **3/12** 已确认记录；后续进度以页面为准。
正式 systemd 服务已启用并运行，开发机的私有转发也已连接，可继续使用 `http://127.0.0.1:8772/`。
部署与发布校验见 [DigitalOcean 部署说明](digitalocean-sgp/README.md)。

## 当前访问

HTTPS 入口由 Nginx 提供 TLS 和登录验证。当前私人登录文件仅保存在被忽略的
`exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/runtime/digitalocean-login.json`；
不将凭据写入文档、命令参数、日志、代码包或 Git。

本开发机已运行以下私有访问链路：

```text
浏览器 127.0.0.1:8772
  -> SSH 本地转发 -> digitalocean-sgp 127.0.0.1:8772 -> 远端正式会话
```

在项目根目录管理转发：

```bash
bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh start
bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh status
bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh stop
```

脚本只转发，不启动本地审核服务。需要 Bash、tmux、curl、OpenSSH、Python 3 和 `flock`。
独立 socket 为 `hsd-general-model-paired-review-sgp-private`，session 为 `paired-review-private-access`；
日志位于 `reviews/paired-cases-02/runtime/digitalocean-sgp-private-access/`。
`run` 为前台模式，以 `Ctrl+C` 结束；`stop` 只结束本任务的 tmux，会保留远端服务和记录。

脚本使用已知主机校验、批处理认证、转发失败即退出、30 秒保活、3 次失败上限及 3 秒重连。
已有会话不重复启动，其他进程占用 8772 时拒绝启动。关闭终端不会中断转发；容器重启后重新运行 `start`。

其他已配置该 SSH alias 的浏览器电脑可以运行：

```bash
ssh -NT -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8772:127.0.0.1:8772 digitalocean-sgp
```

再打开 `http://127.0.0.1:8772/`。不要在已经占用 8772 的开发机重复执行此命令；
浏览器侧端口必须与应用允许的 Host/Origin 保持一致。

## 正式记录与旧入口保护

唯一正式记录位于远端：

```text
/var/lib/hsd-general-model-paired-review/session.json
```

复核人为 `liaozijie`。会话迁移前后字节哈希一致，远端重启续读也已验证。
本机 `exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/session.json`
仅保留为旧备份，旁边已有 `session.json.remote-authority.json`。

旧本地 writer 和 aliyun 反向隧道已停止，旧 `review-forward.sh` 仅供归档参考。
CLI 及旧脚本的 `start`、`run-web` 会因迁移标记拒绝旧写入；不要删除标记来启用陈旧记录，
也不要把此标记复制到远端正式会话旁。自动化保存、确认和导出测试必须使用隔离会话。

## 检查、回退与同机服务

在开发机或已完成私有转发的浏览器电脑检查：

```bash
curl --noproxy 127.0.0.1 --fail http://127.0.0.1:8772/api/health
```

健康响应为 `{"stage":"paired-human-review","status":"ok"}`。
转发异常查看 `runtime/digitalocean-sgp-private-access/tunnel.log`；远端服务状态及日志见部署说明。

代码回退只切换兼容 release，保留最新远端正式记录。若需迁回本机，应先停远端写入、
迁回其最新会话并核对完整状态，不能启动本机旧备份。

用户要求本任务只修改 hsd 站点和服务，保持 `pdf.fenglin.pro` 的配置、8787 上游和服务不变。
此次切换仅经 `nginx -t` 后平滑重载；PDF 的配置、unit、242 个静态文件、后端 PID/启动时间均未变，
重启次数仍为 0，HTTPS 仍返回 200、证书验证通过且页面内容哈希一致。
