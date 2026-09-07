# NoCat 逐查询配对分析与候选集

独立 CPU 事后分析入口，复用 `general_model_ld_nolabel_v1/runs/nolabel-01` 的封存分数和逐查询结果。

- [执行协议](../../../docs/research/experiment-plans/general-model-ld-nolabel-paired-cases-v1.md)
- [配置](../../../config/stage1/general_model_ld_nolabel_paired_cases_v1.json)
- [运行报告与汇总表](results/paired-cases-02/REPORT.md)
- [结果解释与下一阶段优先事项](RESULTS.md)
- [运行身份与状态](results/paired-cases-02/manifest.json)
- [入库产物清单](results/paired-cases-02/export_manifest.json)
- [32条discovery案例索引](results/paired-cases-02/cases/cards_index.json)
- [首批12条AI辅助初读](results/paired-cases-02/cases/AI_REVIEW.md)
- [完整dev查询](results/paired-cases-02/inputs/queries.dev.jsonl)、[完整冻结上下文与提示（gzip）](results/paired-cases-02/inputs/contexts.dev.jsonl.gz)
- [人工复核页面与运行说明](../../../tools/general_model_paired_review_ui/README.md)：三栏工作台，支持首批12条/全部32条、自动保存、阶段展开与CSV/JSON导出；人工记录单独保存于`reviews/`。

`paired-cases-02` 已完整完成：643条dev的全量配对分析、四项新增探索性区间、32条discovery和16条reserve选样，以及全部discovery案例导出。16项CPU测试通过。首批12条已作AI辅助初读；人工复核尚未完成，reserve正文未在本任务展开。

`paired-cases-01` 的来源核验、统计与选样已完成，但案例导出因示例答案的原生字段格式适配失败。其产物和失败记录保留。修复仅影响案例导出，不修改统计、划分或选样规则；正式完整交付使用 `paired-cases-02`。

完整命令见执行协议。入口支持 `validate / analyze / select / review-export / all`；单列案例审阅完成后可用 `report` 刷新汇总。
系统默认 Python 可能缺少 NumPy，请使用协议中的项目 Python 路径。只导入 NumPy 与纯 CPU 分数工具。

只有来源、配对、选样三个自动检查。原分类点值和四项区间在同一分析链核对；不重做 GPU 预检或提示构建审计。

`results/paired-cases-02/` 收录完整运行产物的逐字节副本：统计、逐查询数据、32条定位案例全文与完整提示、AI审阅记录及源码快照。另收录原始643条dev查询和无损压缩的完整冻结上下文。`export_manifest.json` 记录所有文件哈希及解压后上下文哈希。

用户在执行结束后授权体积适中的案例全文、查询数据和完整提示随提交发布。因此封存报告、执行协议中“保留本地”的文字描述执行时状态；当前发布范围以本README及导出清单为准。上传完整输入不表示已审阅reserve正文；本任务仍只详读12条discovery。

`runs/` 保持默认忽略，原位执行产物和失败尝试留在本地。上游259 MB原始评分与105 MB完整分析文件不随本次提交；本次发布的全部结果和压缩输入合计约37.4 MB，单个文件最大约9.7 MB。
