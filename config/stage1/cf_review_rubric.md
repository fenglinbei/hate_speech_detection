# Stage 1 Counterfactual Blind-Review Rubric v1

本轮只判断一个候选值能否作为同一 query、同一 gold tuple、同一字段的局部反事实。审查员不得查看实验 condition、模型输出、log-probability、margin 或其他审查员意见。

## 通用通过条件

候选必须同时满足：

1. canonical 后与 gold value 不同；
2. 代回原四元组时只改变指定字段，其他 tuple 和字段逐字保持；
3. 候选可由原 query 自身支持，不依赖另一 query 的背景；
4. 候选不是乱码、截断到失去语义的残片，也不引入原句不存在的新事实；
5. 候选与 gold 构成有意义、可解释的局部偏好比较。

## target / argument

- `boundary-edit`：只有删除边界冗余成分后仍是原句中的自然、指代清楚的 span 才可通过；改变核心指称或命题应拒绝。
- `multi-tuple-binding-swap`：交换后必须仍是原 query 中真实存在的 span，并确实测试 tuple 绑定；若两个值语义等价或交换没有造成局部对照，应拒绝。
- `query-local-distractor-span`：必须来自同一 query，语法上能占据该字段位置，但不应与 gold 等价；明显无关符号、停用成分或跨边界拼接应拒绝。

## targeted_group / hateful

这两个字段使用确定性 canonical 候选，标记为 `not_required`，不由模型主观审查。`targeted_group` 与 `hateful` 必须独立替换，禁止由其中一个自动重写另一个。

## 决策与 reason code

- `pass / valid-local-foil`：满足全部质量门；
- `reject / equivalent-to-gold`：语义上未形成有效变化；
- `reject / invalid-boundary-fragment`：边界编辑产生不自然或缺义残片；
- `reject / unsupported-by-query`：候选不受同一 query 支持；
- `reject / changes-more-than-one-field`：实际改变了多个字段或 tuple；
- `reject / malformed-candidate`：类型或 canonical schema 非法；
- `not_required / deterministic-label-foil`：仅用于已由程序验证的 group/hate 候选。

不确定时必须拒绝并在 note 中简述原因；不得为了提高 coverage 放宽标准。
