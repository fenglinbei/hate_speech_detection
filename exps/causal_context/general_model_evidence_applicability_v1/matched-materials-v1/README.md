# 第一批位置、长度匹配的语义对照材料

**两批均已获整体无异议反馈，并完成正式输入与评分方案冻结。** [正式冻结与执行入口](../matched-input-freeze-v1/README.md)；[两次反馈记录](../matched-materials-review-v1/README.md)；[第二批位置与长短材料](../position-length-materials-v1/README.md)。以下准备文件保留发布时的草稿身份和字段来源，反馈独立保存。

[打开逐项材料](../reviews/analysis-freeze-20260912/matched-materials-v1/prepared-01/MATERIALS.md)；[CSV 审核表](../reviews/analysis-freeze-20260912/matched-materials-v1/prepared-01/materials.csv)；[准备指针](current.json)；[核验回执](validation.json)。

用户同意“好的，那现在可以先完成材料”。本批落实此前建议的 541/hate 和 3169/hate：每例两段内容替换 C1/C2、两段语义保持改写 D1/D2。每个新文本都有原文、查询、拟保留或去掉的语义、AI 标签与依据、解释限制和空白人工字段；其人工确认状态均为 pending。

| 案例 | 操作对象 | 原始 A | 历史删除 B | C1/C2/D1/D2 完整提示 |
| --- | --- | ---: | ---: | ---: |
| 541/hate | 第 5 个示例 826，non-hate | 918 tokens | 855 tokens | 各 918 tokens |
| 3169/hate | 第 3 个示例 3660，hate | 742 tokens | 712 tokens | 各 742 tokens |

全部 C/D 保持原词典、10 个示例及其答案分布，所有示例边界、示例答案、查询及生成起点的 token 位置一致；目标正文前后 token ID 完全相同。3169 的两版 D 还保留“嘿嘿”的原 token 位置。正文末尾与换行可能合并分词，因此所有核验均在完整提示中执行。

541 的 C 版改为普通生活观察，D 版保留反对由个人行为泛化到整个性取向群体的含义。3169 的 C 版为可单独成立的个人辱骂，D 版保留原示例“被嘿嘿玩过的”及其贬损暗示。3169 的 C 同时改变攻击对象与命题，结果只能先检验整个语义组合，不能直接解释为单词或种族线索的独立效应。新标签与严重度是 AI 提案，不继承原材料的人工确认。

6 项边界测试通过；4 个 A/B 历史文本与 token 精确重放，8 个 C/D 的位置检查和 24 个候选答案边界通过；材料可逐字节重建。另用独立解析核对全部示例正文、答案、标点／表情和 42 个来源文件的哈希。22 条分词检查过的草稿都保存在 [authoring_history.jsonl](../reviews/analysis-freeze-20260912/matched-materials-v1/prepared-01/authoring_history.jsonl)，未使用新模型评分挑选措辞。

审核重点是文字自然度、新文本标签及严重度、C 的语义移除或 D 的语义保持是否成立，以及有无未记录的变化。记录应绑定材料 ID 和正文哈希；修改需另出版本并重新核验。审核后再冻结正式评分协议。此次未运行 GPU，位置交换、短／长版本和激活实验属于后续独立批次。

[8 项待审队列](../reviews/analysis-freeze-20260912/matched-materials-v1/prepared-01/review_queue.json)已备好，尚未接入线上标注页面。接入时沿用现有三栏工作台并使用独立的新材料记录；不能覆盖原 1,072 项材料或将本批 AI 语义意见记成人工确认。

重建检查：

```bash
.conda/stage1-p0/bin/python scripts/review/prepare_evidence_matched_materials.py --output exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260912/matched-materials-v1/prepared-01 --check
```

输出目录拒绝覆盖。`current.json` 仅选择本批材料草稿，不改变原分析参考、案例用途或已完成输入干预的指针。
