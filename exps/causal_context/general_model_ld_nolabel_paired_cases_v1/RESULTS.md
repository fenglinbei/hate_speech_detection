# NoCat 配对案例阶段结果

`paired-cases-02` 已完成全部643条dev的六条件、两任务CPU配对分析，包含7,716条条件任务记录。32条discovery与16条reserve按固定规则选满，32条discovery案例卡已导出，首批12条完成AI辅助初读。人工复核尚未完成；reserve正文未在本任务展开。没有新增模型forward。

## 主要结果

删除显式类别字段只部分缓解hate的组合损失。有示例时，删除字段纠正57条hate查询，也使27条退化；去类别组合SD相对仅示例D仍有60条纠正、103条退化，Hate Macro-F1低4.48个百分点。

group呈不同方向：无示例删除字段时，集合预测有37条纠正、60条退化；有示例时为38条纠正、18条退化。SD相对D有52条纠正、26条退化，Group Micro-F1高2.64个百分点。集合转换计数与Micro-F1的贡献不同，逐标签转换另见[汇总表](results/paired-cases-02/tables/per_label_transitions.csv)。

以下为新增的95%配对bootstrap区间，单位为百分点。全部是已暴露dev上的探索性、逐点、描述性区间，未作多重比较校正。

| 对比 | 指标 | 差值 | 95%区间 |
|---|---|---:|---:|
| SD − D | Hate Macro-F1 | −4.48 | [−8.26, −0.63] |
| SD − D | Group Micro-F1 | +2.64 | [+0.12, +5.19] |
| (SD − SGD) − (S − SG) | Hate Macro-F1 | +3.41 | [−0.13, +7.04] |
| (SD − SGD) − (S − SG) | Group Micro-F1 | +8.67 | [+4.67, +12.82] |

S指删除类别字段后的词典，SG指原词典，D指固定示例。后两行在F1尺度直接比较有/无D时的删除效应。hate的差中差区间跨零；group呈正向差中差。删除同时改变长度和位置，S本身仍可带类别语义，这些结果不能证明内部机制。

## 候选与下一阶段

全量H_rescue候选50条，均通过对应gap门槛；H_residual为89条、筛后88条；H/G_joint_only分别9/17条。八个主桶各选discovery 4条、reserve 2条，没有放宽门槛或跨桶补位。候选标签可以重叠；它们是行为分型，不是机制标签。

AI初读提示后续需要分别验证两种恢复方向（FN→TP、FP→TN）、残余组合干扰、删除有害反例以及稳定对照。“只有组合正确”也可能与负向gold_margin交互同时出现。部分候选对计分口径敏感，应先核验重复性，再做受控输入对照。

下一阶段先比较类别语义与同位置/长度对照、查询词条与示例贡献词条、定义适配与示例对应，再制定激活采集和干预协议。当前仅记录可证伪假设，未执行这些干预。reserve来自已暴露dev，用于后续未参与路径定位的机制评估，不是新的test。

## 核验与产物

- 三项自动检查通过：来源一致、全量配对闭合、固定选样可重放。原分类点值及四项区间复现，绑定输入哈希在运行前后一致。
- 16项CPU测试通过，见[测试记录](results/paired-cases-02/audit/cpu-tests.txt)。
- 首次`paired-cases-01`因示例答案原生字段格式导致案例导出失败，保留失败记录；修复后的`paired-cases-02`成功。两次统计及选样产物逐文件一致，见[回执](results/paired-cases-02/audit/retry_consistency.json)。
- [完整统计报告](results/paired-cases-02/REPORT.md)、[新增区间](results/paired-cases-02/tables/exploratory_ci.csv)、[选样回执](results/paired-cases-02/cases/selection_manifest.json)、[入库清单与哈希](results/paired-cases-02/export_manifest.json)。

按用户后续授权，逐查询表、完整提示、案例卡和[AI详读](results/paired-cases-02/cases/AI_REVIEW.md)随完整运行副本收录于`results/paired-cases-02/`。另附643条dev查询和完整冻结上下文的gzip副本，解压后哈希与冻结输入一致。原位运行目录保留本地；封存文档中的原发布范围不影响本次明确授权。

正式运行身份绑定执行时源码SHA256和快照；结果生成时的Git提交仍为来源提交，之后的发布提交不改变运行记录。
