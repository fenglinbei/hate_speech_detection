# 本轮执行与停止

本轮命名空间为reviews/jingba-attn-restore-v1。先读launch-01/state.json与run-01/state.json（若存在）；已有运行不可重复启动。旧实验均保持终态。

CPU检查通过并封存后，执行决定execution-decision-01.json须明确本轮授权且绑定prepared-01/manifest.json。启动器在授权核验前不查询GPU；核验后只选新近检查为空闲、至少44000MiB的一卡。流程为engineering/check/full/check，失败与完成均终态，无自动重试。

GPU启动入口：scripts/review/launch_jingba_attn_restore_v1.py --decision reviews/jingba-attn-restore-v1/execution-decision-01.json。

取消尚未绑定或阶段间的启动：创建launch-01/CANCEL。运行中安全停止：创建run-01/STOP，工作进程在提交边界正常退出。只管理本轮归属进程，不对其他使用者进程发信号。无固定截止，正常执行到完成。

GPU工作进程正常退出后：CPU analyze至全新results-01；运行独立audit_jingba_attn_restore_results_v1.py；核验所有归属PID退出；人工查看图表；运行closeout_jingba_attn_restore_v1.py建立新的科学结果选择器。来源、输入、运行、结果均不可覆盖；问题保留原状态，另立恢复版本，不放宽数值门槛或重启GPU。

网站与所有旧科学结果选择器均不改动。本轮不含头扫描或任何自动后续实验。
