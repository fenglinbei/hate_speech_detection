# 3169 词条依赖与普通笑声适用边界

**已完成 GPU 运行与独立核算：** [结果解读](../lexicon-scope-results-v1/INTERPRETATION.md)、[完成结果选择器](../lexicon-scope-results-v1/current.json)。以下保留准备与执行说明。

响应本轮用户请求建立的新实验，沿用两套“嘿嘿地笑／哈哈地笑”示例。删除完整 lex-0419，配套原位等长替换、同一替换移至词典末尾，以及原定义追加重述／普通笑声排除说明的匹配比较。

[设计与解释边界](PLAN.md) · [新增词典材料](materials-source-01.json) · [冻结选择器](current.json)

`current.json` 记录第一版准备封存时的状态。[执行版本](execution-model-path-02.json) 选择 `frozen-02`／`run-02`；完成后另建 `../lexicon-scope-results-v1/current.json`，不改写准备历史。

第一次启动 `run-01` 在模型加载前终止：共享的五个权重分片已成为软链接，原加载器拒绝其文件类型。本实验建立 16,397,431,454 字节的独立普通文件副本，11 个文件的哈希全部匹配原登记；新执行版本只修改加载路径及其来源记录。全部科学输入逐字保留，沿用原模型加载前后全量哈希和文件锁检查。原软链接、共享模型和失败记录均保留。路径适配另有 3 项通过的 CPU 测试。

- 40 个真实条件，320 提示、640 候选，32 个历史重放提示。
- 20 个行内词形配对；每套示例有 13 个词典条件转换，每项分别计算两种示例的位移、差中差及共同位移，共 124 项比较。
- 8 轮共 3968 次候选评估；保留原总分／均分、NCC、A/B 正反映射、含 EOS 辅助模式以及全部单探针和留一诊断。
- 7 项 CPU 测试通过，覆盖完整合成运行、封存后禁止重复 forward、提前读取参考的阻断及独立 Decimal 分析核验。
- 输入重建核验 205 个来源、320 个提示、640 个候选边界、352 项几何检查及 32 个历史候选载荷。

运行使用物理 GPU 1、2；每次模型加载前核验冻结 UUID、空闲显存与利用率。所有词典新文本是助手编写的实验变体，未改写原人审定义或人工机制判断。保留全部旧材料、输入、结果、参考和用途指针。

```bash
lexicon_work=exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/lexicon-scope-v1
.conda/stage1-p0/bin/python scripts/review/freeze_evidence_lexicon_execution_v2.py --check
.conda/stage1-p0/bin/python scripts/review/run_evidence_lexicon_scope.py validate --plan "$lexicon_work/frozen-02"
.conda/stage1-p0/bin/python scripts/review/run_evidence_lexicon_scope.py run --plan "$lexicon_work/frozen-02" --run "$lexicon_work/run-02"
.conda/stage1-p0/bin/python scripts/review/run_evidence_lexicon_scope.py check --plan "$lexicon_work/frozen-02" --run "$lexicon_work/run-02"
.conda/stage1-p0/bin/python scripts/review/run_evidence_lexicon_scope.py analyze --plan "$lexicon_work/frozen-02" --run "$lexicon_work/run-02" --output "$lexicon_work/results-02"
.conda/stage1-p0/bin/python scripts/review/analyze_evidence_lexicon_scope.py --check --plan "$lexicon_work/frozen-02" --run "$lexicon_work/run-02" --output "$lexicon_work/results-02"
.conda/stage1-p0/bin/python scripts/review/audit_evidence_lexicon_scope.py inputs --plan "$lexicon_work/frozen-02"
.conda/stage1-p0/bin/python scripts/review/audit_evidence_lexicon_scope.py results --plan "$lexicon_work/frozen-02" --run "$lexicon_work/run-02" --results "$lexicon_work/results-02"
```

完整运行达到终态后，`run` 拒绝重新 forward；`check`、`analyze` 和独立审计不加载模型。新增材料或实现修改须新建版本。
