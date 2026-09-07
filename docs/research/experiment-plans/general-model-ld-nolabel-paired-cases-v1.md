# Qwen3-8B NoCat 逐查询配对分析与候选集：精简执行版 v1

用户于 2026-09-07 确认按审阅后的顺序执行。本文件记录实际执行规则；原附件是提案材料，其旧的“尚未实现”状态不代表当前代码状态。结果以新运行目录的 manifest 和 REPORT 为准。

## 范围与来源

- 来源提交：`f80fe355263c267606c9601ca5d12aabee26da10`；来源运行：`nolabel-01`。
- 全部 643 条 dev、六条件、hate/group 两任务；7,716 blocks、131,172 candidates。
- 仅 CPU 事后分析。没有新 forward、模型/分词器加载、test 数据读取或 Gold 修改。
- 原 plan、raw、analysis、词典、示例、报告均只读。显式绝对路径及内容 SHA256 固定在配置中。
- 实际代码版本采用 Git HEAD 加模块/入口/纯数值依赖/测试的 SHA256 与源码快照，不为运行而强制创建新提交。
- 本轮是已经查看总体结果及只读候选数量后的探索性扩展，不追溯称为原预注册结果。

## 三个自动检查

1. 来源：complete 状态、输入哈希及 manifest 绑定；运行前后检查所有显式绑定输入文件的哈希。已有 GPU 与提示构建审计不重做。
2. 配对：唯一字符串 ID、plan.frame 顺序、完整条件任务矩阵；raw 候选身份/顺序/有限性及分数算术，预测、Gold 和实际使用的 readout 对齐；原分类点值与计数闭合。
3. 选样：类型规则、对应任务/条件的数值门槛、固定 hash 轮询、query/family 去重、无跨侧泄漏；同配置重放选样完全一致。

原四个区间在新增 bootstrap 的同一次计算中回归核对，不再另启一套独立复算。新增逻辑保留聚焦的 CPU 测试。

输入缺失或身份、配对不一致时停止相应分析链。只有上下文元数据缺失时可交付行为统计，候选标记 `context_incomplete`，不得作为完整干预材料验收。

## 预测和分析

六条件顺序固定为 `C0, CLnew, CD, CLDnew, CLnewNoCat, CLDnewNoCat`。
解释记号分别为 0、SG、D、SGD、S、SD。四位轨迹固定 `(C0, CLnewNoCat, CD, CLDnewNoCat)`。

主预测取不含 EOS 的 `answer_sum` argmax；精确并列取最小 canonical ordinal。
Hate 为两个固定类；group 为五类的 32 个候选集合整体 argmax，不逐标签阈值化，不按 hate 清空。
正确性直接比较预测；group 忽略集合顺序。不得用 gold_margin > 0 替代预测正确性。

全部 643 条均进入总体统计，包括精确/近并列。epsilon 为 `0.0013427734375`。
主要机制候选只检查其定义涉及的任务及条件：`tied_top_count == 1` 且 `gap > 2*epsilon`。
四种既存 score mode 重取 argmax，仅标记口径敏感，不据此更换预测或剔除总体样本。

六个双向对比：SGD→SD、SG→S、D→SD、0→S、0→D、S→SD。
两个任务分别保留 00/01/10/11；hate 记录 FN/TP、FP/TN 的双向变化；group 再按五标签记录转换、预测基数、对称差错误数及“集合仍错但误差减少”。
四条件的全部 16 位型保留零计数，不强合并成四类。

固定分层：all、Lq_hit/no_hit、Gold hate/non-hate、Gold group 大小 0/1/≥2，以及 Lq_hit/no_hit × Gold hate/non-hate。每层给 n、构成和点值，空层为 null + reason。

连续 gold_margin 读数：

```text
E_remove_with_D = margin_SD - margin_SGD
E_remove_without_D = margin_S - margin_SG
E_S_given_D = margin_SD - margin_D
I_S_D = margin_SD - margin_S - margin_D + margin_0
```

另报 hate_logodds 的同样差值；给均值、中位数、四分位数、正负零比例，不增加连续读数区间。
逐任务/条件长度来自冻结上下文，Lq/Ld-only 来源来自封存的父 coverage inventory；示例答案构成来自固定 fit catalog 与原 demo 顺序。按转换给长度和词条数的描述性表。

## 四项新增区间

两个任务各计算：

```text
E_S_given_D_F1(t) = F_t(SD) - F_t(D)
J_remove_by_D_F1(t) = [F_t(SD) - F_t(SGD)] - [F_t(S) - F_t(SG)]
```

Hate 使用两类 Macro-F1，group 使用五类 Micro-F1。完整查询配对 bootstrap，10,000 次、NumPy PCG64、seed 42。所有条件、任务和端点共用查询索引；每次汇总 TP/FP/FN 后重算 F1。缺类仍保留固定类别，零分母为 0。95% percentile、`method="linear"`。原四端点另存核验表。

新增区间为逐点、描述性、未校正多重比较，不输出确认性 p 值。分层、逐标签、连续读数只给点值。

## 候选与选样

| 标签 | 任务及正确性条件 | 数值门槛涉及条件 |
|---|---|---|
| H_rescue | hate: D 对、SGD 错、SD 对 | CD、CLDnew、CLDnewNoCat |
| H_residual | hate: D 对、SGD 错、SD 错 | CD、CLDnew、CLDnewNoCat |
| H_removal_harm | hate: SGD 对、SD 错 | CLDnew、CLDnewNoCat |
| G_category_support | group: SG 对、S 错 | CLnew、CLnewNoCat |
| G_category_harm | group: SG 错、S 对 | CLnew、CLnewNoCat |
| H/G_joint_only | 对应任务 core_mask=0001 | 四个 core 条件 |
| Stable_correct / Stable_wrong | 对应任务六条件全对/全错 | 六条件 |

完整成员资格允许重叠。主配额桶顺序为 H_rescue、H_residual、H_removal_harm、G_category_support、H_joint_only、G_joint_only、Stable_correct、Stable_wrong。每桶 discovery 上限 4、reserve 上限 2，总目标 32+16。G_category_harm 保留完整清单，不另设名额。

查询文本仅作 NFC 与 CRLF/CR→LF 规范化，SHA256 得到 family_key，不 strip、casefold 或修改原 prompt。

```text
h = SHA256("paired-cases-v1|20260907|split|" + family_key)
reserve iff int(h[:16], 16) % 3 == 0; otherwise discovery
```

同 family 全部查询/任务/条件放同侧。桶内按 hate 的 Gold label × Lq_hit，或 group 的 Gold size × Lq_hit 轮询。
subcell 使用 `{"task":...,"gold_layer":...,"lq_hit":...}` 的 UTF-8、sort_keys、紧凑 JSON。group gold_layer 编码为 `"0"/"1"/"2"`（2 表示 ≥2）。
子层按 SHA256(seed|bucket|subcell_key) 排序；层内查询按 SHA256(seed|bucket|query_id)、字符串 ID 排序。
稳定桶每侧从 hate 开始轮换 hate/group，在各任务内轮询子层；一侧耗尽后继续另一任务。
更早桶选中的 query/family 后续跳过。无跨桶、跨侧补位，不放宽门槛，逐次记录原因。

## 案例审阅

导出全部 discovery 的两页 Markdown 卡与包含全部冻结上下文的 JSON。第一页只展示查询和资源，第二页才展示 Gold、所有条件预测/分数及候选标签。reserve 不生成或展开案例卡。

首批 12 条按主桶轮询、桶内已冻结选中顺序取样，不改变正式名单。先记录资源页观察，再查看轨迹页。AI 可作辅助初读，但必须单列为 AI-assisted，不冒充人工、独立审阅或未暴露预测的真正盲审。
Gold 争议单列，不改标签；未知干预只写可证伪假设，不能填入未运行的结果。

## 执行命令

```bash
PAIRED_REPO=/data/liaozijie/hate_speech_detection
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PAIRED_REPO/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PAIRED_REPO/.conda/stage1-p0/bin/python" \
  "$PAIRED_REPO/scripts/stage1/general_model_nolabel_paired_cases.py" \
  --config "$PAIRED_REPO/config/stage1/general_model_ld_nolabel_paired_cases_v1.json" \
  --phase all
```

阶段按 `validate → analyze → select → review-export` 执行。`report` 可在单列审阅记录完成后更新汇总。
同身份已完成阶段跳过；已有目录身份不同则拒绝写入。改变配置、筛选或源码时使用新版本/新运行目录，保留旧产物。

## 解释边界

类别删除也改变长度和位置；S 本身可含类别语义，而且词条来自查询与固定示例并集。行为模式及分数交互不能证明内部机制，Gold mass 不是校准的现实正确概率。reserve 来自已暴露 dev，仅用于后续未参与路径定位的机制评估，不是 test。完整案例与提示默认保留本地。下一阶段输入对照与激活干预另立协议。
