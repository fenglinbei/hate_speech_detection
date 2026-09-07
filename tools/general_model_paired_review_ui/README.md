# NoCat 配对案例人工复核工作台

复用现有 WP3 人审工作台的三栏布局与共享组件：左侧固定案例队列，中间查询/词典/示例/预测材料，右侧人工记录。默认按冻结顺序复核首批12条，可切换全部32条discovery；16条reserve不进入工作台。

## 启动

在项目根目录运行：

```bash
python scripts/stage1/general_model_paired_review.py --reviewer-id liaozijie --port 8772
```

浏览器打开 `http://127.0.0.1:8772/`。从另一台机器访问时，将服务所在机器的8772端口转发到本机后打开。

使用 `aliyun` 和 `hsd.fenglin.pro` 的转发入口、进程管理、回退与公网访问状态见
[部署说明](../../deploy/general_model_paired_review/README.md)。该方式会管理本地8772服务，使用前不要重复启动上面的独立进程。

默认读取 `exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/`，人工记录独立保存到同实验的 `reviews/paired-cases-02/session.json`。该目录默认忽略入库。重启同一命令自动续审，不修改封存结果中的人工复核状态。

不同复核人使用不同 `--session-file`；现有会话绑定来源manifest与复核人，不能换人覆盖。可用 `--data-dir` 指定另一份结构相同的完整配对案例产物。

## 操作

1. 先读查询、义项与固定示例，填写语义/立场、定义适配、类别关系、示例对应四项简短观察；允许明确写“不确定”“无相关词条”。
2. 点击“保存初读 · 查看轨迹”。最初观察保持原样，随后开放Gold、六条件预测、配对分数及完整提示。
3. 填写Gold判断、候选解释、替代解释，并选择“进入输入对照 / 先核验再使用 / 暂缓使用”。进入输入对照需写明可检验的下一步；先核验/暂缓需说明原因。
4. 可以在完成人工初判后点击“保存初判并查看AI”。AI初读单列，打开前的人工初判另存快照，之后可补充不同意见。
5. 点击“确认并继续”。按当前队列跳转下一条未完成案例；已确认记录可填写原因后重新打开，原确认记录保留。

本轮记录为非盲人工复核。人工用途判断只影响下一阶段；Gold争议单列，不改原标签、资源、模型结果或冻结名单。

草稿在停止输入5秒后自动保存。保存进行期间继续输入的内容会在下一次保存中保留；切换案例前先保存。版本冲突时保留本地草稿，点击恢复并核对后再手动保存。手机页面提供“查看材料 / 填写复核”切换入口。

- `[` / `]`：上一条 / 下一条。
- `Ctrl/⌘ + S`：保存草稿。
- `Ctrl/⌘ + Enter`：保存初读，或确认并继续。
- `1 / 2 / 3`：选择下一阶段用途。输入框、输入法组合期间及弹窗内不会误触用途快捷键。
- 顶栏“导出”：CSV工作表或包含快照/更正记录的JSON。JSON保留原始文本；CSV对可能被表格软件执行为公式的文本加前导单引号。

## 复用与验证

共享CSS、搜索模式、队列导航和文本输入判别来自 `tools/wp3_candidate_review_ui/`。原子JSON写入、文件锁与版本计算来自 `src/build_lex/annotated_lexicon_repair.py`；HTTP资源、请求来源和会话令牌处理复用现有词典操作人审服务。

```bash
python -m unittest tools.general_model_paired_review_ui.test_store -v
node tools/general_model_paired_review_ui/test_core.cjs
node tools/general_model_paired_review_ui/test_browser.cjs
```

浏览器测试需要Node ≥20及已安装的Playwright Chromium。测试脚本启动临时端口、使用 `/tmp` 下独立会话，复核人为 `automated-browser-test`。它验证实际HTTP服务与页面交互，不写入用户真实人工记录。可设置 `PAIRED_REVIEW_SCREENSHOTS` 指定截图目录。

接口只提供静态页面资源及受限的discovery材料。资源初读、AI展开和确认的顺序也在服务器检查；完整提示在保存初读后按案例、任务和条件读取。服务默认仅绑定回环地址，现有HTTPS反代可通过 `--public-origin` 配置对应来源。
