# 49-case 正确 span 人工修订包

本包完全离线运行。它只包含上一轮 80-case 中被标为碎片或句子的 49 条记录。

## 使用

1. 解压 ZIP，保持文件相对位置不变。
2. 直接打开 `index.html`。
3. 每条选择“无有效词条”，或选择上下文并填写一个或多个正确 span。
4. 正确 span 必须逐字、连续出现在所选上下文中；页面会在保存时校验。
5. 完成 49 条后导出 `lexicon_span_revision_annotations.json` 并交回。

正确 span 只收录 `独立群体词` 或 `可生产词干`。如果句中只有普通词、通用辱骂、
行为描述，或现有上下文不足以确定稳定词条，请选择“无有效词条”。

## 冻结信息

- package ID: `lcsr-5ab78e12d5dd89403af86cb1f9dca60b2de179063e978da0e024ad2f26aadbc3`
- parent package ID: `lcgate-2bb8d8fa91a79500a5e2cce5cf5e968b47436842dc24eafb5717f858011142d6`
- source annotation SHA-256: `4074484c928188c2e29b0bb9d53d0688c4512dce4e248d92bc0d0cac234dda41`
- case count: `49`
- source surface forms: fragment 28 / sentence 21

页面不展示候选分数、数据标签、旧模型判断或 Web 结果，不发起网络请求。
`manifest.json`、`cases.json`、`audit/source_annotations.json` 和 `SHA256SUMS`
用于复核来源与包完整性。
