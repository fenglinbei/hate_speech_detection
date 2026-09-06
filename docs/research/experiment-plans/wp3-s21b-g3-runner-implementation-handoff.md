# WP3 S2.1b G3 与双模型 runner 实施交接

> 状态：`OFFLINE IMPLEMENTED / BLOCKED AT SOURCE AND HUMAN GATES / NO MODEL CALLS`
>
> 日期：2026-08-30（Asia/Shanghai）
>
> 范围：`development-only / non-sealed / non-scientific`

> **2026-08-30 来源获取与价格边界修订：** 已完成候选登记 42 个唯一 URL 的一次零代理抓取，32 个通过
> 静态正文/PDF 完整性检查；该产物是 candidate acquisition，不是正式 source bundle，详见
> [`wp3-g3-zero-proxy-capture-report-20260830.md`](wp3-g3-zero-proxy-capture-report-20260830.md)。GLM
> 价格页按用户决定不做本地下载，只作人工费用估算参考；因此没有生成 pricing snapshot 或 executable
> pricing bundle。当前 runner 的价格 evidence gate 尚未随之放宽，需在预检前另行修订契约。

## 已落地

- 固定七个公开 URL 的 source catalog、受限同步器、Wikimedia oldid 锁定、正文快照与同步凭据；
- 37 项 form relation 人工审核队列，支持 `accept/reject/edit/defer`、逐项 CAS、跨进程文件锁、reopen amendment 与 defer=0 finalize gate；
- `pypinyin==0.55.0` 的独立 CPython 3.12 环境锁、wheel SHA、85-member 安装分发清单、两份资源清单和六组固定向量；
- `wp3-g3-profile/full-v2` 八 family、distance=0、typed reference、多 canonical hypotheses 与 424-case offset replay 完成凭据；
- 三个 prompt v2、严格 JSON/schema normalizer、GLM/DeepSeek 同请求体的六调用合成预检；
- 单写者 checkpoint、reservation-before-send、崩溃 ambiguous、可恢复 slot、私有 vault、公开 immutable run、独立授权 receipt 与自动 evaluator；
- 官方价格页面本地快照、精确模型/费率证据重放及 plan 绑定机制（已实现但当前不为 GLM 生成快照）。

实现不会生成 `lexicon_ref.json`，不会物化 S2.2 或 Phase-B-v2，也不会在 run/evaluation 前读取 locked raw gold。

## 当前硬阻断

1. 候选抓取已证明两个固定教育部页及其余五个教育部页可经 HTTPS→同站 HTTP 取得完整正文；当前 v1
   同步器仍按协议拒绝降级，因此这些 bytes 尚不是正式 source bundle。必须先提升 catalog/policy 版本并
   记录 downgrade，不能用候选 capture 或 `freeze-g3-public-sources` 绕过。
2. 尚未完成人工 form review；因此不存在 finalized reference，也不存在合法的 424-case G3 completion receipt。
3. GLM 价格页被确定为 reference-only，不保存本地快照。当前 successor plan 仍要求可重放价格 evidence，
   因而 live preflight 保持阻断；若改成估价参考而非 executable evidence，必须另立预算契约修订。

截至本交接，只有独立 candidate-acquisition capture；无正式 G3 source/reference/run artifact、
live-preflight checkpoint、provider vault、正式 2,544-slot run 或 evaluation 被创建，也没有模型请求被发送。

## 可复现执行顺序

先构建并自检隔离环境：

```bash
scripts/stage1/bootstrap_wp3_g3_env.sh
PYTHONPATH=src .venv/wp3-g3/bin/python \
  scripts/stage1/wp3_g3_environment.py check
```

只有在正式实现 catalog/policy v2（含 HTTP/downgrade 与 acquisition-mode 记录）后，才使用对应的 v2
同步命令。下面仍是 v1 CLI 示例，它不能接收本次候选 capture，也不能用于绕过版本提升：

```bash
PYTHONPATH=src .venv/wp3-g3/bin/python \
  scripts/stage1/wp3_g3_form_reference.py sync-g3-public-sources \
  --output-directory /ABSOLUTE/NEW/SNAPSHOT/DIRECTORY
```

同步成功会同时生成 `snapshot_index.json` 和 `sync_receipt.json`。随后依次运行
`freeze-g3-public-sources`、`build-g3-reference`、`serve-g3-reference`；人工 100% confirmed 且 defer=0 后运行
`finalize-g3-reference`。再执行：

```bash
PYTHONPATH=src .venv/wp3-g3/bin/python \
  scripts/stage1/wp3_g3_dev_run.py run-g3-dev
PYTHONPATH=src .venv/wp3-g3/bin/python \
  scripts/stage1/wp3_g3_dev_run.py validate-g3-dev
```

当前不要为 GLM 运行 `wp3_provider_pricing.py verify-pricing` 或生成本地页面快照。应先把“价格页只作估价
参考”的新决定落实为独立、可测试的预算契约修订；在此之前，现有 `build-run-plan` 仍会因缺少
`--pricing-evidence` 而 fail-closed。预算契约修订通过后，才继续 `preflight-offline`，最后才允许带显式
`--confirm-six-paid-calls` 的 `run-live-preflight`。

六调用预检只发送合成注入文本。任一 alias、共同参数、usage、finish reason、严格 JSON、normalizer 或预算门失败，都以非零状态停止；不得放宽 provider 参数。2,544-slot 正式批次仍需另行生成 full-run authorization receipt。
