# 双模型 span 人工检查包

- package: `dmspan-review-30583732ec1adb2f`
- frame: `spanframe-66a04b4dda19d4988e6f401e1d5d1938717b07f34ac0cfb053159c6e4a9ade26`
- 页面 case：288 条（实际 240 条唯一原始记录）
- 模型任务：每模型 288 条，其中 48 条为隐藏复测

## 使用

直接双击 `index.html`。页面完全离线，不发送网络请求；标注草稿保存在浏览器 localStorage。完成或中途备份时点击“导出 JSON”。

页面只呈现盲化原文及模型 A/B 的本轮输出，不呈现数据集类别、target、argument、旧候选、原始记录 ID，也不标记哪 48 条是重复记录。模型 A/B 的真实映射位于 `audit/model_mapping.json`，重复配对位于 `audit/source_map.json`；建议完成独立人工判断后再打开。

人工判断不是新的 gold set；它用于抽查双模型策略是否值得采用或需要继续优化。`corrected spans` 每行一个，必须是原文逐字子串。
