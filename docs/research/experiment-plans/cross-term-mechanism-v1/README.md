本轮已完成。京巴普通犬种句复现了“贬损义导致误判、固定目标词状态替换修复”的现象；垃圾和公交车没有出现相同的标签修复，后续两处分支的作用也存在反例。

- [先读：结果解读](../../../../reviews/cross-term-mechanism-v1/interpretation-01/REPORT.md)
- [完整原文、数表与全部图](../../../../reviews/cross-term-mechanism-v1/results-01/REPORT.md)
- [科学收尾记录](../../../../reviews/cross-term-mechanism-v1/closeout-03/closeout.json)
- [独立复核及CPU恢复说明](../../../../reviews/cross-term-mechanism-v1/recovery-01/README.md)
- [完整36份prompt](../../../../reviews/cross-term-mechanism-v1/prepared-01/ALL-PROMPTS.md)
- [冻结协议与准备记录](../../../../reviews/cross-term-mechanism-v1/prepared-01/PREPARATION.md)

京巴、垃圾、公交车各4条查询，共12条。无词典D00、贬损义D01、普通义D02均为8/12正确，但京巴J01（普通犬种）与J03（反对辱称）的正确与错误恰好交换。将D02条件下第17层目标词向量放入D01，京巴4/4正确，全部12条为9/12；新增修复只有J01，J03仍正确但接近分界。12条开发材料、其中5条此前审核的AI构造，不能当作通用修复验证。

24个方向的目标词替换效应绝对值均大于前置对照；19个朝供体分数移动，5个反向。72个单独或联合分支恢复对比中，56个削弱效应、12个增强、4个在工程界内未分清。保留所有36层、反向与未修复案例，没有重新找层或头。注意力分支向量与注意力权重是不同读数。

2026-09-20用户新授权后，GPU0完成1476次前向，23:19:56正常释放；首次阶段开始到最后释放约22.5分钟。192项自身控制、156个单token有／无后EOS端点及独立数值复核通过，所有本轮GPU进程均已退出。结束符续算的审计比较对象经过CPU修正，8处使用同一因果前缀下既有右填充对照精确重建；原失败和7项回归检查保留。后续一处归档自引用问题也已[单独记录并修复](../../../../reviews/cross-term-mechanism-v1/recovery-01/closeout-note-01.md)。原始数据、科学代码和数值门槛未变，没有GPU重跑。

results-current.json选择已完成的科学结果；current.json继续选择不可变的CPU准备包。GPU运行处于终态，不可重启。本轮没有发布网站。
