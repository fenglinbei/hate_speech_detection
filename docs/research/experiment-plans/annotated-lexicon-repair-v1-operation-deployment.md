# 词典修复 v1：第二阶段词条操作审核

日期：2026-09-04（Asia/Shanghai）。状态：第一阶段已冻结，第二阶段已在既有公网入口上线，等待人工单次审核。未生成最终运行词典，未启动 GPU 任务。

## 审核入口与操作

入口：<https://hsd.fenglin.pro>，继续使用原有 Basic Auth。第二阶段是独立的 `repair-operation` 会话，上线检查为 **0/56**，并未复制第一阶段的确认状态。

页面沿用第一阶段的三栏操作模式：左侧搜索/过滤队列，中间编辑，右侧来源与句子证据。每卡对应一个受控词条操作，不再重新标注 span，不显示模型输出或任务标签。

1. 核对原词条、保留 span、删除反例以及待判断问题。
2. 审核各义项的定义、对象、用途、必要语境和多类别；义项 ID 保持稳定。
3. 只登记已核实的显式变体，再检查上下文资格规则。
4. 选择批准、修改后采用、暂缓或不采用；必要时填写理由，最后确认。

| 快捷键 | 动作 |
| --- | --- |
| `1` / `2` | 批准提案 / 修改后采用，只选择、不确认 |
| `3` / `4` | 暂缓 / 不采用提案，只选择、不确认 |
| `[` / `]` | 上一条 / 下一条；有改动先保存，失败则停留 |
| `Ctrl/⌘ S` | 保存草稿 |
| `Ctrl/⌘ Enter` | 校验、确认并继续 |
| `?` / `Esc` | 打开审核说明 / 关闭弹窗 |

停笔 5 秒自动保存草稿，但不代表批准。单键在输入框内不生效；输入法组词、弹窗、请求中暂停审核快捷键。已确认/暂缓项锁定，单条重开须填写理由。CAS 冲突保留本地草稿，不自动覆盖服务器版本。支持快照导出、桌面和手机布局，没有批量确认。

## 边界与待核实内容

- 56 卡由 41 个既有词条组（含全部 11 个重复组）和 15 个新增词面候选组成；新增卡不等于自动增加运行词条，可在闭合词面范围内判断是否应作为显式变体。
- 8 卡仍含空释义：`YP` 的第二义项、呆比、宁、掉的一批、盖盖、蝲蝲蛄、褐兰州、黑人喃。它们阻止直接采用，须补足依据/释义或暂缓；不能把猜测当成定论。
- “它们”带 `policy_unresolved`，不允许以当前未解决的规则直接采用。“它”保留用户对“乱吠”协同贬损的判断，同时提示该远处线索超出左右各 16 字的规则窗口；局部规则草案不是语义证明。
- “屌”补充当前骂人用途；“老黑”“yp”取消绝对化中性/单义判断；“丰县”带入既有三项研究来源，2022 年事件仅为待核实关联，不等于每次出现地名都指向该事件。
- 本阶段只做结构与隔离子进程内的正则编译检查，不提供运行匹配预览。编译通过不代表正确覆盖。运行时每规则 5ms、每查询 100ms 的预算、整体词面冲突、封闭样本覆盖和独立盲审仍是后续门槛。
- 多义结构确定性渲染不等于逐处义项消歧。不采用提案也不等于删除原词条或解除冻结 span 的覆盖要求；暂缓不计入冻结完成。
- 审核通过之前，不发布物化词典、不把草案 dry-run 当作正式实验结果，也不开展第二遍全量人工审查。

## 冻结与可追溯性

唯一原词典 `data/lexicon/annotated_lexicon.json` 保持不变，SHA-256：
`a4a2d1e7826419a365962ded70806a610345d21a998454adeda8b1f999738565`。

第一阶段经核验的 39/39 会话在停止写入、保留完整快照和远端备份后冻结：

- 冻结前审核 revision：`45345a8cb4e48e1de1f287da0658b3c64c4bb8259bfe2bb80effcfa437009778`。
- reference ID：`span-gold-ref-d9aab23d4de89bf0864294c7c71acb8bc584b5c065c404d0321c41603240e81e`。
- reference 文件：`exps/causal_context/annotated_lexicon_repair_v1/span_gold/reference.json`，文件 SHA-256 `4be8a43659048360222b77f9d9f3d77f3fc189590deb5ab51754e8e1dca60b16`。
- 冻结后会话 revision：`b21d8a4256ca3b244b74bb23113c40a6860adbcd4ccefd9589d3b2c25080561f`，文件 SHA-256 `3d81ad20d2cdd20809518303f15763b4c0aacf1709310bac99042b970d9172e1`。
- 冻结前远端备份：`/var/lib/hsd-annotated-lexicon-repair-review/backups/span-gold-pre-stage2-45345a8cb4e4.json`。
- 冻结后本地检查点：`exps/causal_context/annotated_lexicon_repair_v1/span_gold/checkpoints/frozen-d9aab23d4de8/session.json`（按既有规则忽略）。

第二阶段提案：`config/stage1/annotated_lexicon_repair_operation_proposals_v1.json`。

- frame 文件：`exps/causal_context/annotated_lexicon_repair_v1/repair_operation/frame.json`。
- frame ID：`operation-frame-060160e33c7dc0bd518932b10eeb1aaf5bd7ef215ca677943320399175e76045`。
- frame 文件 SHA-256：`3bb50f77b7d9f7d194a1491cffc518a61df1743a94afa467eb561a6697cdb075`。
- 独立远端 session：`/var/lib/hsd-annotated-lexicon-repair-review/repair-operation-session.json`，属主 `hsd-review`、权限 `0600`。
- 上线时 revision：`3c85ecf95cea14781fddb288483cb5569699d4b2746282aebbfd3526561c5752`。
- 上线时文件 SHA-256：`fc7af9ce694f5cab7722ce518e11391cdebe3bb3d7f078956c8ed8c1c6c25b41`，0 确认、0 暂缓、0 修订记录、56 草稿。

## 发布与验证

- 最小发布包 ID / SHA-256：`ee997c8f8114a5cc38b04ac7e8e5725df90baf9287e40c7acd9baf96c484035b`。
- `current` 指向 `/opt/hsd-annotated-lexicon-repair-review/releases/` 下同名目录；包内仅相关服务代码、共享静态资源和冻结 frame，不含完整训练数据、词典全库、示例库、标签、模型权重或会话文件。
- 沿用 `hsd-annotated-lexicon-repair-review.service` 与 `127.0.0.1:8769`；新模板为 `deploy/annotated_lexicon_repair_v1/hsd-annotated-lexicon-operation-review.service`。
- 独立 Python 3.10 runtime：`/opt/hsd-annotated-lexicon-repair-review/runtimes/regex-2026.4.4-0540e5b7`。未修改系统 Python 包。
- `regex==2026.4.4` cp310 manylinux wheel SHA-256 `0540e5b733618a2f84e9cb3e812c8afa82e151ca8e19cf6c4e95c5a65198236f`，与 PyPI 官方版本元数据一致。规则编译子进程限制 3 秒墙钟、256MiB 地址空间、CPU/文件/句柄，失败关闭。
- 本地隔离发布包和远端 `--check` 均通过，准确返回 0/56 及固定正则运行时。
- 回归：43 项 Python/HTTP 测试及 40 个 subtests 通过；Node 核心测试、JS 语法检查、8 组模拟浏览器测试通过。
- 临时独立 session 的真实 56 卡接口烟测：保存→暂缓→重新打开→恢复草稿→导出；正式 session 未提交烟测决定。
- 桌面 1440/1024px 与手机 390px 布局、帮助弹窗、中文输入法防误触、自动保存、CAS 冲突均覆盖。
- 上线后服务 active/running，0 次重启，仅监听回环；health 为 `repair-operation`。只读 bootstrap、关键卡及全部静态资源校验通过。
- 原 Nginx 与认证未改，配置 SHA-256 仍为 `f229557832b65fa18926891c036b7caab974d94a167b909755e50481c28f9e06`，`nginx -t` 通过。服务器本机 TLS/SNI 经 Nginx 无凭据访问返回 401；未声称已从外部客户端完成登录态端到端验证。
- 第一阶段冻结会话 SHA-256 未变；正式 G3 服务继续运行，session SHA-256 仍为 `2065a3685454d764ab6be6e2a33489906967c35e062f6ac80f5198ab08c69207`。

## 回滚

第一阶段旧 release：`1c2687f552da04de7aded1b34f79956fb30a79ca6e7af5ccc60029087a9e9fb7`。
旧 unit 备份：`/etc/systemd/system/hsd-annotated-lexicon-repair-review.service.pre-stage2-20260904`。

若只回滚页面，停止修复服务、将 `current` 原子切回旧 release、恢复旧 unit、daemon-reload 后启动即可；Nginx 不变。第一阶段维持已冻结状态，第二阶段 session 必须保留，不能拷贝成第一阶段会话。若要恢复正式 G3 公网入口，按第一阶段部署记录恢复其 Nginx 备份并停止修复服务。所有版本、会话和备份均保留，不做删除。
