# 嘿嘿释义与语境：运行完成

本轮9个输入已完成，GPU已释放。最新结果由 [results-current.json](/data/liaozijie/hate_speech_detection/docs/research/experiment-plans/hehe-sense-context-v1/results-current.json) 指向，运行处于终态，不得重启。

| 查询 | 参考 | 原释义 D01 | 普通义 D02 | 双义并列 D03 |
| --- | --- | --- | --- | --- |
| Q01 #3169 普通聊天 | 无 | 有 | 无 | 有 |
| Q02 #3660 直接贬损 | 有 | 有 | 无 | 有 |
| Q03 引用并反对 | 无 | 无 | 无 | 无 |

普通义版本修复Q01，却损害Q02；本次双义排法没有修复Q01。Q03在三种条件下均输出“无”。全部分数、9个配对差值、释义分段注意力及解释范围见[完整结果报告](/data/liaozijie/hate_speech_detection/reviews/hehe-sense-context-v1/report-01/REPORT.md)。这些是依赖的探索性材料，不能当作总体准确率或独立确认结论。

- [注意力交互查看器](/data/liaozijie/hate_speech_detection/reviews/hehe-sense-context-v1/results-01/index.html)：保留36层、32头及全部读取位置，支持两段释义分别查看、条件与读取位置差分、SVG导出。
- [全部9份完整材料](/data/liaozijie/hate_speech_detection/reviews/hehe-sense-context-v1/prepared-01/ALL-PROMPTS.md)：与已审核稿逐字节一致。
- [采用记录](/data/liaozijie/hate_speech_detection/reviews/hehe-sense-context-v1/adopted-01/ADOPTION.md)：Q02、Q03、D03已明确通过；Q03为已采用的AI改写，参考无／0级。
- [完成记录与验证入口](/data/liaozijie/hate_speech_detection/reviews/hehe-sense-context-v1/closeout-01/README.md)：包含数值、浏览器及资源释放证据。

用户授权“现在可以直接开始，直至运行结束”。仅使用GPU0，共81次前向；两阶段含装载和释放共114.16秒，从工程启动至最终工作进程释放共223.97秒，最终释放于2026-09-19 13:49:04（北京时间）。9项输出格式和所有原定数值门槛均通过，独立复核及本地浏览器检查通过。没有重试或强制停止。

直接打开HTML时可选择同目录对应的 `.view.json`；通过本地HTTP打开结果目录时可按条件自动加载。本轮线上网站未修改，报告另附PNG、PDF、SVG图。

本轮封存记录SHA256：`0bc1bec325b92063148fc0a8c28e3c4eb3cc72ce8dacc7a085cdd0baf580738e`。原 [current.json](/data/liaozijie/hate_speech_detection/docs/research/experiment-plans/hehe-sense-context-v1/current.json) 保留为准备材料选择器；完成状态请读取 `results-current.json`。所有已封存来源、脚本和历史实验保持原样。
