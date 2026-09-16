# 第六批：逐句核对与用户裁定

本批为安全输入队列的第 151–180 条，共 30 条，均已复读全文并参考前五批的明确裁决。6 条疑点已由用户逐条回答，现有 6 条用户严重度数值、24 条 AI 严重度数值，没有本批待定项。本轮收尾于第六批，第七批尚未开始。

[合并讨论表](../../batches/batch-06/review.md)、[JSON](../../batches/batch-06/review.json)的 `records[].values` 与 [CSV](../../batches/batch-06/review.csv)展示当前值。0→non-hate，1–4→hate，null→null。原文、候选分与讨论前倾向已保存在[裁定前快照](before-user-decisions-01/manifest.json)，当前记录不再保留已解决的候选状态。

## 六条明确回复

| 对象 | 严重度 | 映射 hate | 当前 group 及来源 | 用户原回复 |
| --- | --- | --- | --- | --- |
| #5865 | 0 | non-hate | []（用户） | 0 / []：整句仍只谈抽象政治、资本或制度 |
| #5900 | 0 | non-hate | Racism、Region（AI） | 0 级：当前文字没有明确贬损 |
| #5919 | 1 | hate | LGBTQ、others（AI） | 1 级：轻度人格贬评和强硬争辩 |
| #5998 | 1 | hate | others（用户） | 1 / others：只确认对具体人的能力讥讽 |
| #6074 | 0 | non-hate | Sexism（AI） | 0 级：当前没有具体攻击命题 |
| #6263 | 0 | non-hate | Region（AI） | 0 级：只是描述或区分概念，没有认可攻击 |

本轮明确裁定的是 6 个分数和 2 个 group，见[完整回复及类别裁决](user-decisions-01.json)、[分数裁决](../../severity/batch-06/discussions/user-decisions-01.json)。6 个 hate 均由用户分数按已接受的规则派生，没有新增直接 hate 裁决；其余 4 个 group 仍是 AI 草稿。攻击形式、证据高亮和辅助解释继续保留 AI 来源。没有把疑点处理完成记作 30 条完整人工审核。

## 已直接校准的边界

- #5819 明确邀请组队猎杀实际身份群体，拟 4／Racism、Sexism；#6413 的“鲨光男人”则是被批评的引文，按全文另有的性别、个人及角色污名取 2／Sexism、others。
- #5855 反讽的实际含义是河南、河北也有同性恋，未产生新的攻击命题，拟 0／Region、LGBTQ。#5924 的反复疾病、混乱概括拟 2／LGBTQ、others。两者均采用 #5416 的“检查实际含义”方法，未把所有反讽统一判成某一类。
- #5780、#6035、#6442 采用可见文字标准，不由缺失事件、“你这样”或语气符号补出具体攻击；分别拟 0／[]、0／others、0／Sexism。
- #6114 用猴与香蕉把越南伴侣动物化，拟 3；参考用户 #528 的国民动物辱称分类，取 Region、Sexism、others，没有仅凭越南国籍与动物词增加 Racism。
- #5690、#5708 保留实际个人辱骂对应的 others。用户 #5615 的仅 Racism 已作为个案明确保存；未从这一个未附理由的类别修正推断全局规则被撤销。
- #6006、#6291、#6307 反对相应的刻板印象或双重标准，未另行发起攻击，取 0。#6224 反驳阶层、职业、学历及年龄偏见，拟 0／others；没有具体地域对象，不因“地域偏见”字面加 Region。
- #6183 只叙述某 ID 支持黑人；未据“爱狗人士”的 ID 和黑人相邻出现而补出动物类比，拟 0／Racism。
- #6458 同时展示河南居民与欺骗的实际负面属性，并借个人经历为整体污名背书，拟 2／Region，与 #4744 没有具体对象和内容的情况分别处理。

## 有限词义查证

[上海大学档案馆《百年上大画传》](https://dangan.shu.edu.cn/info/2541/20896.htm)直接使用“上大人”称学校共同体成员，为 #6035 的学校身份读法提供线索。这不确定当前句子的褒贬，也不将学校所在地自动视为被评价的地域。

关于“默”，[公开使用者讨论](https://www.reddit.com/r/chonglangTV/comments/qiwoqw)明确提到“默（黑犬）”的写法，支持它可以作为种族化贬称。这里仅引用词语用例，不采用讨论中的其他观点，也不据单字推断所有场景的所指。

“阿娜”的女性贬称解释来自安全词典输入中的候选义项，未作为本句已确认事实。检索得到 [Blizzard 的 Ana-Mouz 角色介绍](https://worldofwarcraft.blizzard.com/pt-br/news/20307794)和[使用中文角色名的游戏说明](https://ol.3dmgame.com/gl/190713.html)，可以支持角色存在，但不能证明当前作者用它指女性；本轮没有检得能够确认这一语境适配的可靠用例。用户随后明确选择 1／others，只确认对具体人的能力讥讽；这项个案裁决不用于确认或否定其他语境的词义。

所有来源、查询和使用限制记录于 [calibration.json](calibration.json)。没有为了历史或游戏背景补造原帖上下文，没有用本批原答案或预测决定标签；原答案对照仅在草稿落盘后由程序生成，仍是独立建议层。

## 来源与状态

第六批此前没有暂存标签，本次直接按当前规则生成第一稿；不伪造“修订前”的第六批记录。[初始标签参照](initial-label-reference/labels-06.json)单独冻结，严重度选择及输入绑定该版本；[policy-application.json](policy-application.json)记录全批检查。

第五批最新的 [7 条标签裁决](../batch-05/user-decisions-01.json)、[7 条分数裁决](../../severity/batch-05/discussions/user-decisions-01.json)和[校准回执](../../severity/batch-05/calibration-summary.json)已作为参考。第六批的[校准回执](../../severity/batch-06/calibration-summary.json)绑定本批最终导出和裁决来源；[校验记录](validation.json)核对当前映射、逐字段来源与历史快照。未被用户明确裁定的字段继续保留 AI 来源，没有新增线上材料确认。
