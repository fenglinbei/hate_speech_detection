# 配对案例人审页面：aliyun 转发

配置日期：2026-09-07。页面代码版本：`304afafa13190d9e93f1d3d5eaffcd75667461c1`。

**状态：转发已配置，隧道与服务器本机 HTTPS 检查通过；公网访问仍被阿里云备案检查拦截。**
公网 HTTP 返回 `403`、`Server: Beaver` 和 `Non-compliance ICP Filing` 页面；公网 HTTPS 连接被重置。
不能将这个状态描述为已完成公网登录访问验证。

用户已确认该域名未备案，另一个已备案域名的业务范围不包含此审核。登录凭据作用于 Nginx，
不能解除请求到达 Nginx 前发生的云平台拦截。本次未替换为另一个已备案域名。

## 路径与进程

```text
https://hsd.fenglin.pro
  -> aliyun / Nginx（已有证书与 Basic Auth）
  -> aliyun 127.0.0.1:18772
  -> SSH 反向隧道
  -> 开发机 127.0.0.1:8772
  -> 配对案例三栏人审工作台
```

复用既有 `ssh aliyun` 配置。SSH 使用已知主机校验、批处理认证、转发失败即退出、30 秒保活和 3 次失败上限；远端只监听回环地址。
页面启动时指定 `--public-origin https://hsd.fenglin.pro`，允许正确的 Host/Origin。

开发容器没有可用的 systemd 总线。本次使用独立 tmux socket
`hsd-general-model-paired-review`、session `paired-review`，`web` 和 `tunnel` 各占一个窗口。
每个窗口只管理自己的子进程，退出后等待 3 秒重启；关闭终端不会结束服务。
**这不是开机自启动：开发容器重启后，需要重新运行 `start`。**

在项目根目录运行：

```bash
bash deploy/general_model_paired_review/review-forward.sh start
bash deploy/general_model_paired_review/review-forward.sh status
bash deploy/general_model_paired_review/review-forward.sh stop
```

依赖：Bash、tmux、curl、OpenSSH、项目可运行的 `/usr/bin/python`（Python ≥3.10）。
`start` 可重复运行；已有托管会话时只显示状态。发现 8772 被其他进程占用时拒绝启动，须先核对进程归属。
`stop` 仅结束本次专属 tmux 会话，保留人工记录。

## 当前使用：SSH 私有访问

在**使用浏览器的个人电脑**上，通过已有的 `aliyun` SSH 身份连接远端回环隧道：

```bash
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8772:127.0.0.1:18772 aliyun
```

随后在该电脑打开 `http://127.0.0.1:8772/`。这条链路依靠 SSH 身份验证，供私人访问使用，
不经过未备案域名的公网 HTTP/HTTPS 入口。保留本地 8772 端口，以匹配应用允许的 Host/Origin。
开发机本身已有 8772 服务，不要在开发机重复执行此命令；本地端口被占用时先核对其归属。
Codex 中已经打开的本地预览也可以继续使用。

## 数据与续审

正式记录继续保存在：

`exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/session.json`

复核人仍为 `liaozijie`。本次转发和进程恢复测试后为首批 **0/12**、全部 **0/32**，未写入任何人工决定。
记录和日志都在被 Git 忽略的 `reviews/` 内。`runtime/` 包含 `web.log`、`tunnel.log`、最近一次本地健康检查以及切换前备份
`session.pre-public-forward-20260907.json`。

切换前后正式会话 SHA-256 相同：
`31790fe50dcec495bd70dc456593816d9f5000e321edc3b37af51c5d70e2bac8`。

## Nginx 切换与回退

已将 aliyun 的 `/etc/nginx/sites-available/hsd.fenglin.pro` 中唯一的上游
从 `127.0.0.1:8769` 改为 `127.0.0.1:18772`。站点的其他设置保持原样，原账号密码继续有效。
本目录的 `hsd.fenglin.pro.nginx` 是本次启用配置的副本。

- 切换前配置 SHA-256：`f229557832b65fa18926891c036b7caab974d94a167b909755e50481c28f9e06`。
- 切换后配置 SHA-256：`67f3bb00c005e4cdb23913583fc94c575985fc7a8375e956be0d16f560e44407`。
- 远端备份：`/etc/nginx/sites-available/hsd.fenglin.pro.pre-paired-review-20260907-f2295578`。
- 修改前核对旧校验值，原子替换后执行 `nginx -t` 与 `systemctl reload nginx`；失败时恢复备份。

回退前先核对现有配置是否仍属于本次切换，避免覆盖后续部署。恢复上述备份，执行 `nginx -t`，
通过后重新加载 Nginx，再停止本次转发。旧 8769 服务原本已停，恢复旧配置本身不会恢复旧工作台；
不要因此自动重开已经暂停的词典审核。已有 WP3 服务、词典审核会话及认证文件的校验值均未改变。

## 验证与排查

本地健康检查：

```bash
curl --noproxy 127.0.0.1 --fail http://127.0.0.1:8772/api/health
```

远端隧道检查必须发送允许的 Host：

```bash
ssh aliyun 'curl --noproxy 127.0.0.1 --fail -H "Host: hsd.fenglin.pro" http://127.0.0.1:18772/api/health'
```

服务器本机 TLS/SNI 检查（无凭据应为 401）：

```bash
ssh aliyun 'curl --noproxy hsd.fenglin.pro --resolve hsd.fenglin.pro:443:127.0.0.1 --head https://hsd.fenglin.pro/'
```

外网检查：

```bash
curl --noproxy hsd.fenglin.pro --max-time 15 --head https://hsd.fenglin.pro/
curl --noproxy hsd.fenglin.pro --max-time 15 --include http://hsd.fenglin.pro/
```

已验证页面与隧道子进程各退出一次后自动重启、重复启动不生成第二套进程、正式会话保持原样。
隧道上的健康接口和六个页面/静态资源返回 200；错误来源及无令牌写入返回 403；
Nginx 配置检查通过，服务器本机 TLS/SNI 请求返回原有 Basic Auth 的 401。
这些检查不导出完整审核记录，也不替代公网登录及保存验证。

目前公网阻塞发生在云平台访问层，且用户已确认本域名未备案。继续使用这个公网入口前，需要处理
网站备案及阿里云接入问题；已有其他服务商备案时也需检查接入信息，参见
[阿里云官方备案拦截排查说明](https://help.aliyun.com/en/icp-filing/basic-icp-service/support/web-site-suddenly-appeared-for-the-record-to-block-or-hang-and-so-on-and-so-forth)。
备案或接入状态恢复后，重新检查公网 401、使用原有账号登录，并在独立测试会话验证保存。
