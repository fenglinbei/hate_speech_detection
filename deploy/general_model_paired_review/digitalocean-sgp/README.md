# DigitalOcean Singapore 部署说明

## group 范围 v2 的代码与规则迁移

用户于 2026-09-09 接受 group 范围修订。生效文档与完整规则、父版哈希、用户确认、
discovery 影响 ID 清单保存在独立 policy JSON 中。v1 AI bundle 和原 manifest 保留原字节；
新规则不意味着旧的人审决定已按 v2 复核。2026-09-09 01:15:38（Asia/Shanghai）已上线，
 release 为 `af7a61fe2b2324f6d2954ae07c04f7b2736a34b3eb2791e3049e0a6f6dffa5a3`；
[执行与验收记录](../../../exps/causal_context/general_model_evidence_applicability_v1/execution-status.md)保留具体回执。

2026-09-09 02:14:23（Asia/Shanghai）材料导航静态更新已上线，当前 release 为
`211f4b4f2da3a2cfd0725e5cfba4c75f57c41db5426bc4e95190a4de4e668037`。
本次沿用同一 v2 unit、policy 与最新会话，仅替换三份证据 UI 文件；
使用下文 `update_static_release.py`，没有重跑一次性规则迁移。

```bash
python3 -B -S deploy/general_model_paired_review/build_release.py \
  --output-dir /tmp/hsd-evidence-policy-release-NEW \
  --evidence-bundle exps/causal_context/general_model_evidence_applicability_v1/bundle/evidence_bundle.json \
  --evidence-policy exps/causal_context/general_model_evidence_applicability_v1/policies/group-scope-v2/policy_amendment.json
```

构建仍只包含运行闭包和 discovery 输入；独立 policy 文件随 manifest 逐文件校验，
使用 `python -I -B -S` 在隔离会话检验新规则启动、零自动确认和续读。
正式 unit 仅在 `ExecStart` 末尾追加固定的 `--evidence-policy` 路径，其他设置不变。

使用 [update_evidence_policy_release.py](update_evidence_policy_release.py)，不要重跑首次激活脚本。
将它与 [update_static_release.py](update_static_release.py) 上传至同一私有操作目录，
另传入发布压缩包及本仓库的候选 unit。先执行以下只读预检；`ARCHIVE_SHA256` 和
`UNIT_SHA256` 均须使用本机计算、独立核对的完整哈希：

```bash
python3 -B -S update_evidence_policy_release.py \
  --archive /tmp/hsd-evidence-policy-release.tar.gz --sha256 ARCHIVE_SHA256 \
  --unit /tmp/hsd-evidence-policy.service --unit-sha256 UNIT_SHA256 --check-only
```

预检在 0700 临时目录复制当前记录，执行真实迁移和续读，再删除临时副本；
不写正式会话、unit、release 链接，也不控制任何服务。去掉 `--check-only` 执行上线：

1. 取得与静态更新共用的部署锁，验证新代码/规则和 frozen 输入字节。
2. 仅停止 HSD，并核对无运行中的主进程；以 0600 备份两层最新会话与原 unit。
3. 对停写后的最新快照再次隔离验证，仅显式迁移 evidence 层；保留全部旧对象、裁决、
   材料快照、历史事件与暴露记录。受影响决定通过规则版本等待重核，不自动确认。
4. 原子切换代码和 unit，启动 HSD，只读 health/bootstrap 核对进度与续读，
   允许启动后用户正常保存并保留这些最新写入。部署回执不包含人审正文或凭据。

迁移尚未改变正式 evidence 文件时，失败只回退代码与 unit，沿用最新记录。
**一旦 evidence 字节改变，失败会保持新代码/规则与最新会话，并停止 HSD 等待向前修复。**
即使迁移子进程在原子写入后异常退出，也按磁盘实际字节判定；不得启动不兼容的 v1 writer，
更不能用部署前备份覆盖后续人审决定。此流程不修改或 reload Nginx，不控制 PDF 服务；
执行前后运行只读 audit 证明 PDF 的配置、静态资源、PID/启动时间/重启数和 HTTPS 不变。

该迁移是已经完成的一次性 v1 → v2 转换，不能对当前会话再次执行。
后续代码更新（包括纯前端修复）必须继续携带相同的 `--evidence-policy` 与原 AI bundle，
保留已迁移的最新会话；不能用下文历史首次上线命令构建缺少 active policy 的生产版本。
`smoke_evidence_readonly.cjs` 可设置 `HSD_EXPECT_POLICY_VERSION=evidence-applicability-annotation-policy/v2`
验证生效规则、旧 AI 版本、无自动确认、完整规则的手机阅读及零 POST。
另设 `HSD_CHECK_MATERIAL_NAVIGATION=1` 验证上一／下一材料按钮、`Alt + ↑/↓`、
筛选边界与三种视口的按钮可达性；脚本拦截所有 API 写请求。

## 证据适用性附属模式（2026-09-08）

新模式使用同一服务的 `/evidence/` 与 `/api/evidence/`，新会话独立保存在
`/var/lib/hsd-general-model-paired-review/evidence-applicability-v1/session.json`。
原 `/` 模式继续使用原会话；两者均有单独的非空启动检查。仅替换 hsd 代码和 unit，
不修改或 reload Nginx，不操作 PDF 服务。

带初稿包的发布命令如下；输出目录须新建且为空：

```bash
python3 -B -S deploy/general_model_paired_review/build_release.py \
  --output-dir /tmp/hsd-evidence-release-NEW \
  --evidence-bundle exps/causal_context/general_model_evidence_applicability_v1/bundle/evidence_bundle.json
```

包中新增证据模式的两份 Python 源码、三份静态文件及一份只读 AI bundle。
原始 manifest 字节仍保持不变；不包含真实人工会话、凭据、测试会话或 reserve。
构建脚本对两模式分别创建临时隔离会话，验证隐藏、零人工确认和重启续读。

首次上线使用 [activate_evidence_release.py](activate_evidence_release.py)：先验证上传压缩包、
逐文件清单与 unit 哈希，检查旧会话仍与只读父记录引用一致；停止 hsd 后备份最新原记录，
以 `hsd-review` 初始化全新的独立会话，再原子切换代码和启动服务。
该脚本拒绝覆盖已有的新模式会话；后续更新应复用其最新记录，不能再次初始化。
失败只回退代码和 unit，保留所有最新会话。只读检查使用
[audit_readonly.py](audit_readonly.py)，比较 PDF 配置、242 个静态文件、进程和 HTTPS。

已有两层会话时，证据模式的纯前端修复可使用
[update_static_release.py](update_static_release.py)。先传入 `--archive`、`--sha256` 和
`--check-only` 做只读预检，再以同一参数去掉 `--check-only` 执行更新。
该脚本仅允许 `evidence.html`、`evidence.css`、`evidence.js` 改变，验证所有其他包内文件字节一致；
停写后以 0600 备份两层最新会话，只切换代码并启动 hsd，失败只回退代码。
不初始化或恢复会话、不改 unit 或 Nginx；启动后的人工作业也保留。

线上进度及最新 release 身份见
[证据审核执行记录](../../../exps/causal_context/general_model_evidence_applicability_v1/execution-status.md)。
下文为原 paired 模式的迁移背景与通用操作，旧 release 哈希仅表示首次切换时状态。

**2026-09-08 已完成正式切换：`https://hsd.fenglin.pro/` 已上线，首批 3/12 确认进度完整保留。**
正式会话位于本服务器，unit 已 `enabled/active`，来源与会话哈希、重启续读均已核验。
本机旧 writer 和 aliyun 反向隧道已停止，旧会话保留为带迁移标记的备份。
[直接私有访问](../README.md) 已运行，本机 `http://127.0.0.1:8772/` 继续访问远端正式服务。

## 路径与配置

| 项目 | 配置 |
| --- | --- |
| SSH 主机 | `digitalocean-sgp`，`165.22.48.237` |
| 正式入口 | `https://hsd.fenglin.pro/` |
| 后端 | `127.0.0.1:8772` |
| systemd | `hsd-general-model-paired-review.service`，用户/组 `hsd-review` |
| 发布目录 | `/opt/hsd-general-model-paired-review/releases/<完整压缩包 SHA-256>/` |
| 当前代码链接 | `/opt/hsd-general-model-paired-review/current` |
| 唯一正式会话 | `/var/lib/hsd-general-model-paired-review/session.json` |

[服务配置](hsd-general-model-paired-review.service) 用 `/usr/bin/python3 -B -S` 启动，
`ExecStartPre` 要求正式会话文件已存在且非空，避免误启动生成空会话。失败后 3 秒重试，
60 秒内最多启动 5 次；持久目录 0700、会话 0600，代码只读，正式记录写入独立状态目录。
会话文件本身不能是符号链接。

当前使用 [正式 Nginx 配置](hsd.fenglin.pro.nginx)，对所有 HTTPS 应用路径执行 Basic 登录验证，
保留 Host，并代理到 loopback 8772。应用 `--public-origin` 必须为 `https://hsd.fenglin.pro`，
不带尾斜杠；页面使用 `/` 开始的资源路径，需部署在域名根目录。
证书位于 `/etc/letsencrypt/live/hsd.fenglin.pro/`，登录哈希文件为 `/etc/nginx/.htpasswd-hsd-review`，
ACME 根目录为 `/var/www/hsd-acme`。无证书时的 [初始化配置](hsd.fenglin.pro.bootstrap.nginx) 仅开放 ACME。

私人登录文件仅保存在本机被 Git 忽略的
`exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/runtime/digitalocean-login.json`。
保持 0600，不打包，不把密码放进 URL、命令参数、操作输出、日志或提交；用户明确索取时可直接提供给用户。

## 构建和校验发布包

在仓库根目录执行，输出目录必须在 `/tmp` 下且为空或不存在：

```bash
HSD_BUILD_DIR=$(mktemp -d /tmp/hsd-review-release.XXXXXX)
python3 -B -S deploy/general_model_paired_review/build_release.py --output-dir "$HSD_BUILD_DIR"
sha256sum "$HSD_BUILD_DIR/release.tar.gz"
```

构建与生产运行仅需 Linux、Python ≥3.10 标准库，无第三方运行依赖。
输出 `release/` 和 `release.tar.gz`；固定归属、权限、顺序和时间戳，使同一输入快照可复现。
包内是 8 个 Python 文件、6 个静态文件、36 个数据文件，以及 `release_manifest.json`，共 51 个文件。
数据仅含原始 manifest、案例索引、首批名单、AI 初读 CSV 和 32 张 discovery JSON 卡片（首批 12 条）。
不带人审会话、凭据、日志、模型、训练数据、reserve 或未用 `inputs/`。

原始 `data/manifest.json` 逐字节保留，不能删去其中未打包的实验产物条目，避免破坏会话来源绑定。
发布清单另行列出实际 50 个应用/数据文件的大小及 SHA-256；清单自身和压缩包的哈希由构建脚本输出。
当前 `current` 指向正在运行的 release，其完整压缩包 SHA-256 为：

```text
704ffe68544f5ba576c64de04934204dd3064a4bfdb6abd13322c05ecc3074ce
```

只上传压缩包，真实会话不随代码发布自动迁移：

```bash
scp "$HSD_BUILD_DIR/release.tar.gz" digitalocean-sgp:/tmp/hsd-review-release.tar.gz
```

服务器先比对压缩包 SHA-256 与本机构建输出，再将包内 `release/` 去掉一层目录后解压到新的
`releases/<完整 SHA-256>/`，不覆盖旧 release。设 `HSD_RELEASE_DIR` 为新目录并校验：

```bash
python3 -B -S - "$HSD_RELEASE_DIR" <<'PYVERIFY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "release_manifest.json").read_bytes())
actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
assert actual == set(manifest["files"]) | {"release_manifest.json"}
for name, expected in manifest["files"].items():
    path = root / name
    assert not path.is_symlink() and path.resolve().is_relative_to(root)
    content = path.read_bytes()
    assert len(content) == expected["bytes"]
    assert hashlib.sha256(content).hexdigest() == expected["sha256"], name
assert hashlib.sha256((root / "data/manifest.json").read_bytes()).hexdigest() == manifest["source_manifest_sha256"]
print("PASS: verified", len(manifest["files"]), "payload files")
PYVERIFY
```

校验后以同目录临时链接和原子重命名更新 `current`。运行用户只读代码；会话在独立持久目录，
不会随 release 更换。后续构建须使用新输出的 SHA-256，继续读取现有远端正式会话。

## 隔离测试

构建脚本默认用 `python -I -B -S` 从 release 导入后端，在 `/tmp` 临时测试会话检查全部材料、
阶段隐藏和重启续读，完成后清理测试会话。

[smoke_https.cjs](smoke_https.cjs) 需要本机 Node ≥20、Playwright、`@playwright/test` 和 Chromium，
验证 HTTPS 登录、保存/刷新、阶段解锁、确认/下一条、手机标注说明。测试工具不属于生产包。

**该脚本会写测试记录。** 运行前须在独立路径创建全新测试会话，复核人精确设为
`automated-deployment-test`，并在维护窗口将域名临时代理到测试服务。脚本不会创建服务器或切换代理，
会在修改前检查复核人、32 条案例和 0 条已确认，拒绝正式会话。DNS 在浏览器内固定到该 DO IP，
仍验证域名证书。

```bash
HSD_REVIEW_CREDENTIALS_FILE="$PWD/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/runtime/digitalocean-login.json" \
HSD_REVIEW_SCREENSHOTS=/tmp/hsd-review-https-smoke \
node deploy/general_model_paired_review/digitalocean-sgp/smoke_https.cjs
```

测试后停止测试服务并恢复正式代理，保持正式会话与测试会话隔离。测试会话不能改名为正式会话，
测试确认数不能计入人审。正式进度仅以只读 health/bootstrap 检查。

## 已完成的切换与同机服务边界

正式会话经授权迁入独立持久目录，迁移前后字节哈希一致；复核人仍为 `liaozijie`，3/12 确认记录保留。
远端正式 unit 已启用，重启续读通过；本机旧 writer/aliyun 隧道已停止，旧会话旁已有
`session.json.remote-authority.json`，CLI 和旧脚本 `start`、`run-web` 均拒绝旧写入。
远端不携带此标记，本机只运行到远端的 SSH 转发。后续代码更新不迁移或覆盖正式会话。

**用户约束：本任务只修改 hsd 站点/服务，保持 `pdf.fenglin.pro` 不受影响。**
PDF 的 `/etc/nginx/conf.d/pdf-translate-reader.conf`、`127.0.0.1:8787` 上游与
`pdf-translate-reader.service` 必须保持不变；先记录基线，再 `nginx -t`，仅平滑重载 Nginx，
不重启 PDF 服务，之后复查配置、静态资源、后端进程和 HTTPS。
此次前后检查确认配置/unit/242 个静态文件哈希不变，PDF 仍为 PID `1270223`、
启动时间 `2026-08-20 17:38:12 UTC`、重启次数 0；HTTPS 返回 200，证书验证成功，首页哈希一致。

## 当前私有访问和回退

```bash
bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh start
bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh status
bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh stop
```

脚本只维护本机 `127.0.0.1:8772` 到 `digitalocean-sgp 127.0.0.1:8772` 的 SSH 转发，
使用独立 tmux、任务锁、严格已知主机校验、批处理认证、`ExitOnForwardFailure`、30 秒保活、
3 次失败上限及 3 秒重连。端口被占用时拒绝启动；`stop` 只停止该 tmux 会话，保留远端服务。
`run` 为前台模式，以 `Ctrl+C` 结束；容器重启后重新执行 `start`。

其他已配置 SSH alias 的浏览器电脑可以保持以下连接，再打开 `http://127.0.0.1:8772/`：

```bash
ssh -NT -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8772:127.0.0.1:8772 digitalocean-sgp
```

服务器只读检查使用 `systemctl status hsd-general-model-paired-review.service`、
`journalctl -u hsd-general-model-paired-review.service` 和
`curl --noproxy 127.0.0.1 --fail http://127.0.0.1:8772/api/health`。
HTTPS 问题看 `/var/log/nginx/hsd.fenglin.pro.access.log`、`hsd.fenglin.pro.error.log`；
本机私有转发看 `reviews/paired-cases-02/runtime/digitalocean-sgp-private-access/tunnel.log`。

代码回退只切换兼容的 release，继续使用当时最新的唯一正式会话；操作前停写、备份并核对 schema/来源。
不得恢复本机旧备份或仅删除 marker 来重启旧 writer。
如需迁回本机，应停远端写入、迁回远端最新记录并验证完整状态，作为另一次明确的状态迁移处理。
# Final annotation writeback, 2026-09-12

The user explicitly accepted the completed 954-object local result. The evidence
session now contains 1,072 confirmed materials, with its 118 earlier confirmations
and all 32 separate case records preserved. The case-level stage remains 3/32.

The compatible release is
`25d100d40d3107cc8d016ed6a5c11b445ceb250ed56da95a76fab3b87a385785`.
The final artifact SHA-256 is
`737ccd7c81ade5a7c9617be0e6afb68f4e966028a20c8237b906e038a84f5177`.
The artifact and writeback receipt are under the evidence state's
`final-results/<artifact-sha256>/`; exact pre-writeback copies of both sessions
are in `/var/lib/hsd-general-model-paired-review/final-annotation-backup-di3mqzyp`.

`finalize_annotation_release.py` performed the explicit, version-checked operation
with a stopped writer. Do not rerun its original import against later decisions.
Code rollback must retain a finalization-compatible reader and the latest session;
never restore a session backup to make old code start. Nginx, the service unit and
the PDF service were not changed by this operation.

The UI and JSON/CSV exports retain `final_annotation` receipts, including severity
and adopted definition wording, while the original frozen input definitions stay
intact. These receipts record bulk acceptance and preserve pre-acceptance field
provenance. Blank demo original-label verdicts were outside this sentence-label
review; they must not be converted into invented source-error confirmations.
