# 第17层替换后的单分支恢复：定时待运行

已完成CPU准备并封存。沿用Q01/Q02×D01/D02四份完整输入，只增加分别恢复答案前第26层注意力输出、第28层MLP输出的配置。所有层号0起；两处不会合并恢复。没有新的科学GPU前向。

一次性启动时间：**2026年9月20日10:35，北京时间（Asia/Shanghai）**。届时重新确认空闲GPU；仍忙则等待，不停止其他进程。取一张空闲L20完成工程复测、正式运行、正常释放、独立数值审核、报告和封存。正常预算152次前向，GPU预计约3–6分钟，含CPU审核和报告预留约10–15分钟（不含等待GPU）。

[已冻结协议](../../../../reviews/hehe-branch-restore-v1/prepared-01/PROTOCOL.md) · [四份完整输入](../../../../reviews/hehe-branch-restore-v1/prepared-01/ALL-PROMPTS.md)

运行状态在reviews/hehe-branch-restore-v1/schedule-01/state.json；完整报告完成后写入reviews/hehe-branch-restore-v1/report-01/REPORT.md。当前应用内定时接口不可调用，使用服务器后台一次性定时进程，因此不会出现在应用的计划任务列表；服务器需持续运行。

取消排程：创建reviews/hehe-branch-restore-v1/schedule-01/CANCEL。若GPU阶段已开始，会写入本轮run-01/STOP并等待安全释放；不会终止其他人的进程。任一验证失败即停止并保留记录，不自动重试，不重启任何已完成实验。

launch-01是因新时间安排而在绑定前正常结束的等待尝试；后续仅使用launch-02。旧科学结果和网站保持不变。
