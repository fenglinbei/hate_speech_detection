# UI 易用性第二轮实施记录

对应规范：[README.md](README.md)  
实施日期：2026-08-29 至 2026-08-30  
记录状态：实现、验收与生产上线完成

## 实施元数据

| 字段 | 结果 |
| --- | --- |
| 实施状态 | `DEPLOYED / VERIFIED` |
| 实施提交/工作树 | 生产不可变 release 内容 ID `eb4f9215074d1776f534c1f389101c6946cdfd21f4e430031eb22a391aa3bd9c` |
| 上一生产 release | `7c8513a0402aef9e4c6d0dae57925afa5b050be52f1a28ec4a98ad976fa7e0fd` |
| 新生产 release | `eb4f9215074d1776f534c1f389101c6946cdfd21f4e430031eb22a391aa3bd9c` |
| 部署时间（Asia/Shanghai） | `2026-08-30T00:35:27+08:00` |
| 回滚锚点 | 上一生产 release `7c8513a0402aef9e4c6d0dae57925afa5b050be52f1a28ec4a98ad976fa7e0fd` |
| 实施负责人/任务 | WP3 S2.1 标注项目 UI、检索与队列导航升级任务 |

## 实际改动摘要

- UI 布局与滚动层级：右栏使用 `head/actions/error/editor` 命名区域；操作按钮采用固定正常高度并移至详情上方。桌面/平板将页面和决策外壳设为非滚动，只让中央内容与右侧编辑器滚动；移动端改为决策区内 sticky 操作栏。桌面原因列表取消内部滚动，Mention 删除操作进入独立危险区。
- A/B/C 快捷标注与快捷键：选区栏新增 A/B/C 三入口，1/2/3 映射到对应 route/default reason；创建后打开 mention 详情但不确认。阶段 B 移除 1–6 键盘动作及按钮数字前缀，保留 J/K、`[`/`]` 和组合键。
- 冻结原文搜索接口与三模式：新增连续、多词、模糊搜索及单按钮循环；客户端实现 300ms 防抖、AbortController、请求序号、安全 `<mark>` 摘要和错误/空结果状态。
- 统一可见队列与左栏定位：搜索结果与状态筛选合成为唯一队列，Case 导航、快捷键、确认后继续和预取共享该顺序；普通重绘保留用户滚动，条件更新、导航或保存成功后可重新显示当前项。
- 兼容性或偏离计划之处：持久化英文代码、offset 解析、CAS revision、session schema v1、导出合同及既有写接口均未改变。过滤队列确认完成和 J/K 已补充端到端断言；手动滚动保留及软键盘遮挡仍没有独立专项断言。

## 发布 blocker 关闭记录

| Blocker | 风险 | 修复 | 复核结论 |
| --- | --- | --- | --- |
| 搜索竞态 | 旧查询触发的延迟 Case 加载可在新查询后覆盖活动页面 | 将 search generation 传入结果应用与 Case 加载，并结合 AbortController 丢弃所有过期提交 | 新增延迟 Case 响应后立即改查无结果的 Playwright 场景；旧响应未覆盖新状态，五视口通过 |
| 确认期编辑未锁 | confirm 在途期间继续编辑可能追加 post-confirm draft，破坏逐项审计顺序 | `confirmBusy` 使编辑器、原文、选区、提案条、搜索、筛选和导航不可交互，同时取消待执行搜索 | 新增慢 confirm 场景；只观察到一次 confirm 写入，确认后无额外草稿，五视口通过 |
| 短原文摘要泄露 | 短于摘要上限的冻结正文可能完整出现在搜索响应中 | 正文搜索摘要强制省略；即使整段命中也裁去至少一个 code point，极短值不返回全文 | 短正文整段命中断言随最终 Python 合并回归通过 |

## 接口记录

### `GET /api/cases/search`

| 项目 | 实际结果 |
| --- | --- |
| Schema | `wp3-s21-review-search/v1` |
| 支持模式 | `literal`、`all_terms`、`fuzzy` |
| 查询限制 | 非空普通文本，最多 80 个字符；不接受未知模式或正则语义 |
| 最大结果数 | 424 |
| 响应字段 | 顶层 `schema_version/frame_id/query/mode/matches`；匹配项 `case_id/matched_field/snippet/match_start/match_end/distance` |
| 隐私/盲化检查 | 独立服务测试确认结果不含完整原文、提案内容、来源或模型身份 |

其他 API、持久化 payload、offset 解析、CAS、session schema 或导出合同变化：无。

性能记录：本地使用 424 个 Case 和 80 字模糊查询实测约 243ms；这是单次开发环境观测值，不定义生产 SLA。

## 自动测试结果

所有命令、通过数、跳过数和失败数必须填写实际值；不要只写“通过”。

| 测试层 | 命令 | 结果 |
| --- | --- | --- |
| JavaScript controller/unit | `node tools/wp3_candidate_review_ui/test_core.cjs` | 11/11 passed |
| 搜索服务专项 | 搜索 projection 与 HTTP 路由专项，纳入 Python 合并回归 | 9/9 targeted checks passed；424 Case、80 字模糊查询约 243ms |
| Python 合并回归 | WP3 reviewer 服务、搜索、HTTP 安全及相关生命周期选择集 | 27 tests OK, 3 skipped；跳过项均因本地缺少 materialized frame |
| Playwright 五视口 | `npm run test:ui-e2e` | 104 passed, 11 expected skipped, 115 total；1440×900、1280×720、1024×768、412×915、375×667 |
| Frame/session 投影 | 本地真实 frame 用例及生产只读投影核对 | 本地 3 项因缺少 materialized frame 跳过；生产 bootstrap、会话元数据与三模式搜索只读验证通过 |

覆盖确认：

- [x] 点击和键盘创建 A/B/C。
- [x] 搜索模式循环与连续、多词、模糊结果。
- [x] 搜索响应及其后续延迟 Case 加载的乱序抑制。
- [x] 可见队列的页面按钮与 `[`/`]` 导航。
- [x] 可见队列中的确认后继续及结果全部完成链路。
- [x] 确认期间编辑锁定且不会产生 post-confirm draft。
- [x] 短冻结原文不会作为完整搜索摘要返回。
- [x] 普通左栏重绘保留滚动位置，业务事件后重新定位当前 Case。
- [x] 桌面单滚动容器、按钮高度、移动端粘顶和五视口无横向溢出。

## 发布前会话快照

| 字段 | 值 |
| --- | --- |
| Phase | `raw`（Phase A） |
| Revision | `77f244e47dd4542d058bf695966aee268fb7ca0df329c93ac7369574eefc52a5` |
| 已确认/总数 | `274 / 424` |
| `session.json` mtime | `2026-08-29 23:26:02.828576863 +0800` |
| `session.json` size | `114140` bytes |
| `session.json` SHA-256 | `6655f2b8f06743d6f3e1bffddeaf67ba4a8fd847139ec66839e65e5103930fa9` |
| 只读备份路径 | `/var/lib/hsd-wp3-review/session.backup-20260830T003527+0800.json`（`root:root`、`0400`） |
| 备份 SHA-256 | `6655f2b8f06743d6f3e1bffddeaf67ba4a8fd847139ec66839e65e5103930fa9` |

## 发布与现网验证

| 检查 | 结果 |
| --- | --- |
| 不可变 release 创建及文件哈希 | `eb4f9215074d1776f534c1f389101c6946cdfd21f4e430031eb22a391aa3bd9c` 已创建并作为内容 ID 验证 |
| `current` 原子切换 | `2026-08-30T00:35:27+08:00` 从 previous release 切换到 new release |
| 服务 active / restart count | `hsd-wp3-review` active，`NRestarts=0` |
| Nginx / Basic Auth / HSTS | Nginx active；未认证请求返回 401 challenge；HSTS 验证通过 |
| bootstrap 和搜索接口只读冒烟 | bootstrap 状态正确；连续、多词、模糊三模式 GET 搜索均通过 |
| 桌面与手机只读浏览器冒烟 | 真实 Chromium 1440×900 与 375×667 完成 GET、搜索、对话框和滚动检查，无横向溢出 |
| 生产环境未执行保存或确认 | 冒烟期间无写请求；会话 revision、mtime、size 与 SHA-256 前后完全一致 |

发布覆盖的五个应用文件与生产 staging 校验值：

| 文件 | SHA-256 |
| --- | --- |
| `index.html` | `80109623dede6272e75438b3a7e8bb5171ee5c3d30cfe6dee008eae826fb49ca` |
| `core.js` | `f993e2b5e9437a12aa30f7ff5018c87651f07925aa3b67dc7d07008c4b57d5dd` |
| `app.js` | `0afa5da3965cde6e161f103593bf3cc4e370ae13ee63bc5e4d6f2140e3fe891a` |
| `styles.css` | `7478e55f1c7a7a74e53b081d444ccaac25d8ac73db96dc9e8dac1d003e54f284` |
| `server.py` | `fc5e98b04e1afa44bc2e1cadac3ba2eb97a06957a45de1600fca334f1d103712` |

## 发布后会话核对

| 字段 | 值 |
| --- | --- |
| Phase | `raw`（Phase A） |
| Revision | `77f244e47dd4542d058bf695966aee268fb7ca0df329c93ac7369574eefc52a5` |
| 已确认/总数 | `274 / 424` |
| `session.json` mtime | `2026-08-29 23:26:02.828576863 +0800` |
| `session.json` size | `114140` bytes |
| `session.json` SHA-256 | `6655f2b8f06743d6f3e1bffddeaf67ba4a8fd847139ec66839e65e5103930fa9` |
| 与发布前一致性结论 | Phase、进度、revision、mtime、size、SHA-256 全部一致；部署和只读冒烟未修改会话 |

## 回滚说明

- 回滚操作：将 `current` 原子指回 release `7c8513a0402aef9e4c6d0dae57925afa5b050be52f1a28ec4a98ad976fa7e0fd`，随后重启 `hsd-wp3-review`。
- 回滚只切换 release，不恢复、替换或覆盖最新 `session.json`。
- 实际是否执行回滚：未执行；新 release 验证通过并保持 active。
- 未解决问题和后续工作：无已知发布 blocker；真实移动端软键盘弹出没有独立自动化场景，现有 375×667 真实浏览器滚动冒烟和移动端 Playwright 矩阵均通过。
