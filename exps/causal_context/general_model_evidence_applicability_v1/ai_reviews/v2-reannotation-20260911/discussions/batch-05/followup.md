# 第五批：默认 hate 映射后的校准

30 条授权文本均已逐条复读。原有 7 个疑点均已由用户逐条裁定 hate、严重度和 group。本批现在有 7 个用户严重度数值、23 个 AI 数值，全部可映射；0 个待讨论对象。其他 23 条仍是 AI 草稿，随后进入第六批。

[合并讨论表](../../batches/batch-05/review.md)提供全部疑点原文与候选；[JSON](../../batches/batch-05/review.json)的 `records[].values` 和 [CSV](../../batches/batch-05/review.csv)是当前结果。hate 按活动严重度默认生成，0→non-hate、1–4→hate，待定分数暂不生成结果。候选分和倾向分仍是讨论信息。

## 已直接采用的校准

- #478 沿用 #3194、#4744 的可见文本标准，当前取 0；不推断缺失图片中的攻击内容。女性明确作为被解释对象，group 仍取 Sexism。
- #5425 沿用用户 #1729 的 lkd 释义，取 2／Sexism、others；#5532 的具体受话者辱骂取 2／others。已明确的个人辱骂范围不再另设 hate 问题。
- #5466 针对具体举报者使用女性辱称，加入 others；#5408、#5507 是整体群体表述，没有仅凭“群体由人组成”加 others。
- #5211 取 0／Racism、Region、Sexism，区分批评殴打、预测暴力后果与实际赞许暴力。#5230 取 0／Region、Sexism、others，按明确国别、性别与工人角色处理，未凭国别词补种族。
- #476 的种族化婚恋污名取 2；允许接纳性侵受害者，不等于赞成实施性侵。#51 的女性本质化与工具化、#5125 的婚育及后代身份排斥取 3；不因出现性内容自动升 4。
- #4839 对态度群体及亲邻的连带歧视取 3／others；没有明确女性信息，不自行补 Sexism。“火烧”在此作为殃及自身的比喻，没有转成纵火主张。
- #5282 当前出现针对上海居民的直接污名化表述，取 2／Region；与 #4744 没有展示具体攻击对象和内容的情形分开。#5473 的个人审美讨论保留 0／Racism。
- #5666 在删帖叙述中实际使用含轻蔑意的称谓指人，参照 #4392 的使用与释义区分，拟 1／LGBTQ、others。此判断为 AI 草稿，不是把 #4392 的人工裁决转移给本例。

## 本批用户裁决

| 对象 | hate | 严重度 | group | 用户理由 |
| --- | --- | --- | --- | --- |
| #4810 | hate | 2 | Racism、Sexism、others | 美国佬为政治代理指控不参与讨论 |
| #5113 | non-hate | 0 | LGBTQ、Sexism、others | 仅为客观描述事件，主体是那个”男的“，并未直接攻击同性恋群体 |
| #528 | hate | 3 | Region、Sexism | 使用明确辱称，且没证据表明是无害讨论 |
| #5365 | non-hate | 0 | Sexism、others | 只在谈网络梗和生活选择 |
| #5394 | hate | 3 | LGBTQ、others | 只是表达漠视、未认可伤害，没到达预定程度 |
| #5416 | hate | 2 | LGBTQ、others | 归谬得到”男同最会传播艾滋病“带有污名化意，且整体带有阴阳怪气 |
| #5615 | hate | 2 | Racism | 未提供理由 |

[标签裁决](user-decisions-01.json)和[分数裁决](../../severity/batch-05/discussions/user-decisions-01.json)分别保存原文、正文哈希、对象版本与政策。本轮用户明确给出全部三个字段，区别于仅由分数自动推导的 hate。辅助形式、证据标记和整理文字仍为 AI 来源。

#5416 按归谬的实际污名化含义取 2，不把反讽默认作无害表达。#5394 删除暴力认可候选；#5113 删除作者认可个体辱骂的假设；#5365 保留讨论的 Sexism、others，但没有攻击。#5615 服从用户仅 Racism 的个案裁决；用户未给出一般规则变更，不自行推广为“有专门类别就不加 others”。

原候选、分数与说明已存于 [before-user-decisions-01](before-user-decisions-01/manifest.json)。本轮三个字段均与暂行映射一致，不再保留旧疑点为活动待办。

## 词义查证与暴露

检索“租鸡青一”发现[2019 年的使用者实例](https://m.hupu.com/bbs/31351347.html)，将其展开为青年租房、购买性服务的生活选择说法。这使 #5365 的早期“无具体身份对象”判断需要重看。该资料只解释词语，不提供本句缺失上下文；用户最终判定本例仅讨论网络梗和生活选择，取 0／Sexism、others。

[《汉语群体称谓语冒犯性分析及数据集构建》](https://www.colips.org/conferences/clsw2023/proceedings/papers/107.pdf)的可见检索摘要将“基佬”列入含冒犯义的称谓；PDF 直接打开失败，没有据此声称读完全文。#5666 的具体 1 级判断来自本句用法和既有校准，并非论文直接给出的等级。

另读过安全定义输入中“幕刃”“基”“基佬”“棒畜”的候选义项。它们作为词义线索另行登记，未当作已确认的人类裁决，也未直接采用词典类别完成 group。没有读取本批原答案或模型预测来标注；原答案对照由导出程序在标签落盘后另行生成。

## 记录身份

[calibration.json](calibration.json)记录全部 30 条修改、人工参照及查证来源；[policy-application.json](policy-application.json)记录个人辱骂计入 others 的规则应用。早期暂存稿保存在 [before-calibration](before-calibration/labels-05.json)，初始标签参照另行冻结于 [initial-label-reference](initial-label-reference/labels-05.json)。严重度输入绑定该冻结版本。

导出校验发现 #5532 的旧辅助选项“无群体评价目标”与增补后的 others 不兼容，已改成“其他原因”并保留个人辱骂说明；hate、group 和分数没有因此改变，见 [validation-fixes.json](validation-fixes.json)。没有为通过校验修改线上规则或真实人审记录。

[第五批校准回执](../../severity/batch-05/calibration-summary.json)固定本阶段裁决与导出身份，供第六批引用；没有写入线上权威会话。
