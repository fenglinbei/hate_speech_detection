# 第17层替换后的单分支恢复：已完成

本轮于2026年9月20日10:35启动，10:38:34（北京时间）正常释放GPU，完成152次前向。12个原生自身控制、8个条件自身控制、16个单标签后EOS端点均通过。独立审核核对152个完整向量、116份轨迹和22,032个汇总数值。四条既有输入，所有层号0起。

[完整报告及解读](../../../../reviews/hehe-branch-restore-v1/report-02/REPORT.md) · [全部结果JSON](../../../../reviews/hehe-branch-restore-v1/results-01/results.json) · [恢复比较TSV](../../../../reviews/hehe-branch-restore-v1/results-01/restoration-contrasts.tsv) · [独立审核](../../../../reviews/hehe-branch-restore-v1/result-audit-01.json)

单独恢复答案前26层注意力或28层MLP，在全部四个方向都削弱第17层嘿嘿替换的最终效应。26注意力削弱约12.2%–46.6%，28MLP约17.9%–23.9%。Q01普通义→原义的修复被任一恢复撤销；Q02的修复虽减弱但保留。比例不可相加为中介份额，未确立唯一自然词义路径。

GPU运行与独立审核自动完成。后续报告封存脚本误将新解读正文纳入复制等值检查，自动流程安全停止；现已在独立v2脚本和report-02修复完成。全部19个真正复制的数据/图均保持逐字节一致，旧日志和半成品report-01保留；未重跑GPU或放宽数值审核。

权威索引为results-current.json；schedule-current.json已记录CPU修复后的完成状态。所有本任务定时、控制器、工作进程和报告子进程均已退出；不要重启任何已完成实验。本轮结果尚未增量部署网站。

[已冻结协议](../../../../reviews/hehe-branch-restore-v1/prepared-01/PROTOCOL.md) · [四份完整输入](../../../../reviews/hehe-branch-restore-v1/prepared-01/ALL-PROMPTS.md) · [修复记录](../../../../reviews/hehe-branch-restore-v1/recovery-01/diagnosis.json) · [封存记录](../../../../reviews/hehe-branch-restore-v1/closeout-01/closeout.json)
