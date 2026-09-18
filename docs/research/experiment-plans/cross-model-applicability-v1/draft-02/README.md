# 四词项 CPU 扩充包

入口是[本轮审核页](REVIEW.md)。这版只将首批已交付内容记为整批采纳；新增材料和S7–S8没有人工决定。原始draft-01、四个旧脚本、历史实验选择器及所有旧评分保持不变。

| 项目 | 首批已采纳 | 新增待审 | 合计 |
|---|---:|---:|---:|
| 词项家族 | 2 | 2 | 4 |
| 查询 | 8 | 8 | 16 |
| 原材料审核行 | 24 | 24 | 48 |
| 关系 | 120 | 120 | 240 |
| 科学条件 | 192 | 192 | 384 |
| 线性比较式 | 576 | 576 | 1152 |

查询参考、严重度与材料来源保存在materials.json；关系的当前状态在relations.json；relations-ai.json保留原AI建议。旧关系采纳后ID增加-adopt01并通过supersedes追溯；关系中的资料质量说明保留采纳前原话，当前人审状态看provenance。单独的[首批采纳目录](../adopted-01/README.md)固定24项原文、120条关系、S1–S6和用户原话。

[材料](MATERIALS.md)、[矩阵](condition-matrix.tsv)、[模型可见消息](PROMPTS.md)、[关系](RELATIONS.md)、[分析边界](ANALYSIS-AND-EXPOSURE.md)可对应查看。三模型沿用[首批模型身份说明](../draft-01/MODELS.md)及精确本地快照；GLM本地实现与已核查上游的差别不被抹掉。

本包含384新输入及156旧输入，每个模型各渲染540份，共1620份CPU prompt和3240个候选续接边界。这里包括只需复用历史结果的旧8B输入，不表示全部都要重新进行科学评分。未来新增科学预算仍是1152＋312＝1464次prompt-only前向，资格、格式诊断和必要桥接另计。

本目录封存后不可原地修改。复核命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .conda/stage1-p0/bin/python scripts/review/check_cross_model_applicability_expansion_v1.py
```

CPU检查可以确认字节、结构、符号与状态，没有替代新增材料人审或GPU数值资格。运行代码和设备安排尚未冻结，没有run入口、任务队列或模型预测。
