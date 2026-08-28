# 词典候选 80-case 离线人工门

本包完全离线运行，不需要安装依赖，也不会发起网络请求。

## 使用

1. 解压 ZIP，并保持包内文件相对位置不变。
2. 用 Chrome、Edge 或 Firefox 直接打开 `index.html`。
3. 标注会自动保存在当前浏览器；建议中途使用“导出备份”。
4. 完成 80 条后点击“导出完整结果”，得到 `lexicon_candidate_annotations.json`。
5. 如需换电脑或浏览器，可用页面的“导入结果/备份”恢复。

只判断 exact candidate，不要把整句标签、相邻辱骂词或主题类别投射到候选本身。
页面故意不显示候选排序、分数、语料标签、模型票据和 Web 结果。

## 判定口径

- `完整词/短语`：形式完整，可独立分析。
- `碎片或句子`：截断 ngram、语法残片或整句。
- 只有 `独立群体词` 和 `可生产词干` 会自动导出为 `provider_eligible=yes`。
- `仅完整短语成立`、`通用辱骂`、`行为/现象`、`非仇恨/其他`、`上下文碎片`均为 `no`。
- 无法可靠判断时选择“不确定”，不要勉强二选一。

## 冻结信息

- package ID: `lcgate-2bb8d8fa91a79500a5e2cce5cf5e968b47436842dc24eafb5717f858011142d6`
- fit records: `5165`
- provider-bound candidate frame: `1000`
- sample size: `80`
- sampling policy: `track-quantile-hash-sample/v1`

`audit/candidate_sample.jsonl` 和 `manifest.json` 用于复核抽样与来源，不是页面标注输入。
`SHA256SUMS` 可用于检查包内文件是否被改动。
