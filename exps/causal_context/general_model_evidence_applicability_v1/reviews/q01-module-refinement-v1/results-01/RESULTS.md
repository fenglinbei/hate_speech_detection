# Q01 六模块细化结果

全部 12 遍封存和数值验收已通过；这里仍只涉及一个已暴露查询。

| 目标 | 提名单元 | 四组双向最小 closeness gain |
|---|---|---:|
| C | attention 35 / pre_answer | 0.040501 |
| I | mlp 34 / pre_answer | 0.586958 |

A/B、NCC、探针和控制结果均保留在 comparison-summary.csv 与 comparisons.jsonl；辅助口径异常见 nomination.json。
提名依据原总分的数值可分辨双向恢复；没有按分类翻转选择，也没有自动替换赢家或启动查询转移实验。
C 与 I 分别解释。机制 readiness 仍为 false，不能从单例局部干预推出独立词形原因或完整证据使用机制。
