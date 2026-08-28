# 词典注入帮助划定混淆跨度边界的定性案例

说明：以下案例均来自完整方法 `output/runner/method_comparison/ours_prompt_al1280.json`，并与全局检索基线 `exps/emonstration_selection/global/exp_744c07ab8e/runner_output/exp_744c07ab8e_s42.json`、单样本 SRAG `output/runner/no-rag-1-llamafactory-2ep.json`、零样本 `output/runner/method_comparison/general_prompt.json` 对比。所有案例均满足：完整方法 `hard_tp>0`，三个比较设置均 `hard_tp=0`。下文同时报告 hard/soft 两种复核结果；若基线 `hard_tp=0, soft_tp=1`，记为 soft-only 命中，即语义近似但边界不完全一致，也可按软命中使用。每个案例都列出未 hard 命中基线的实际输出和边界偏差。

## Case 1: id=382

原句：现实里女拳没网上猖狂可以了

Exact 匹配结果与词条解释：

- 女拳 (Sexism)：`女拳` 是 `女权` 的中文谐音，常用于对中国女权主义者的贬义表达。

Semantic 匹配结果与词条解释：

- 田园女拳 (Sexism)：`田园女拳` 为 `田园女权` 的中文谐音，指以女权为借口追求女性收益最大化的群体。
- 女拳王 (Sexism)：常用于对中国女权主义者的贬义表达。

补全作用证据：

- 女拳 (Sexism) 直接命中句中目标词 `女拳`。
- 田园女拳 (Sexism) 和 女拳王 (Sexism) 补充了同类贬义女权词条，帮助模型保留完整评价范围 `现实里女拳没网上猖狂可以了`，而不是只抽后半句。

匹配复核：

- 完整方法：hard_tp=1, soft_tp=1, soft_only_tp=0
- 全局检索基线：hard_tp=0, soft_tp=1, soft_only_tp=1
- 单样本 SRAG：hard_tp=0, soft_tp=1, soft_only_tp=1
- 零样本：hard_tp=0, soft_tp=1, soft_only_tp=1

GT：

```text
女拳 | 现实里女拳没网上猖狂可以了 | Sexism | hate
```

完整方法输出：

```text
女拳 | 现实里女拳没网上猖狂可以了 | Sexism | hate
```

未 hard 命中基线的实际输出与失效说明：

```text
全局检索基线（soft-only）：女拳 | 没网上猖狂可以了 | Sexism | hate
  偏差：argument 左边界漏掉“现实里女拳”。

单样本 SRAG（soft-only）：女拳 | 没网上猖狂可以了 | Sexism | hate
  偏差：argument 左边界漏掉“现实里女拳”。

零样本（soft-only）：女拳 | 没网上猖狂可以了 | Sexism | hate
  偏差：argument 左边界漏掉“现实里女拳”。
```

解释：三个基线都漏掉了 argument 左边界 `现实里女拳`。词典注入把 `女拳` 作为 Sexism 贬义词固定下来，并通过相邻女权贬义词条强化该表达的语义背景，使完整方法抽取到与 GT 完全一致的跨度。

## Case 2: id=5025

原句：当光棍足以成为一支军队的时候女性就再没有恋爱自由的权利了

Exact 匹配结果与词条解释：无。

Semantic 匹配结果与词条解释：

- 妈宝女 (Sexism)：多用于形容成年后仍对母亲言听计从、缺乏独立生活能力和自主思想的女性。
- 厌男 (Sexism)：指女性对于普通男性的厌恶，一说，是对男性的歧视。
- 田园女拳 (Sexism)：指要求男女平等却要男性承担主要责任，以女权为借口追求女性收益最大化的群体。
- 索玛女王 (Sexism)：指常发表极端言论、煽动性别对抗的女权博主。
- 慰安妇 (Sexism)：常用于对女性的侮辱性称呼。

补全作用证据：

- 妈宝女、厌男、田园女拳、索玛女王、慰安妇 的定义/类别均连接 `女性` 与 Sexism。
- 在没有 exact 词条时，semantic 词条为模型提供了“该句应围绕女性受限权利判断 Sexism”的背景。

匹配复核：

- 完整方法：hard_tp=1, soft_tp=1, soft_only_tp=0
- 全局检索基线：hard_tp=0, soft_tp=0, soft_only_tp=0
- 单样本 SRAG：hard_tp=0, soft_tp=1, soft_only_tp=1
- 零样本：hard_tp=0, soft_tp=0, soft_only_tp=0

GT：

```text
女性 | 没有恋爱自由的权利 | Sexism | hate
```

完整方法输出：

```text
女性 | 没有恋爱自由的权利 | Sexism | hate
```

未 hard 命中基线的实际输出与失效说明：

```text
全局检索基线（soft 未命中）：女性 | 当光棍足以成为一支军队的时候女性就再没有恋爱自由的权利了 | Sexism | hate
  偏差：argument 过长，把条件背景“当光棍足以成为一支军队的时候”纳入了混淆跨度。

单样本 SRAG（soft-only）：女性 | 再没有恋爱自由的权利了 | Sexism | hate
  偏差：argument 左边界漏掉核心否定成分“没有”。

零样本（soft 未命中）：女性 | 当光棍足以成为一支军队的时候女性就再没有恋爱自由的权利了 | Sexism | hate
  偏差：argument 过长，把条件背景一并纳入了混淆跨度。
```

解释：全局检索和零样本把原因状语 `当光棍足以成为一支军队的时候` 一并纳入 argument，单样本 SRAG 又漏掉了核心否定边界 `没有`。semantic 词典补全帮助完整方法把混淆跨度收缩到真正的歧视性论断 `没有恋爱自由的权利`。

## Case 3: id=1200

原句：默就是默，非我中华民族必为异类，畏威不畏德。

Exact 匹配结果与词条解释：

- 默 (Racism)：`默` 是 `黑犬` 的拼字，指黑色的狗，对黑人群体的侮辱性称呼。

Semantic 匹配结果与词条解释：

- 反默 (Racism)：`默` 是 `黑犬` 的拼字，指对黑人群体的迫害行为。
- 猎默 (Racism)：`默` 是 `黑犬` 的拼字，指对黑人群体的迫害行为。
- 默妖 (Racism)：`默` 是 `黑犬` 的拼字，对黑人群体的侮辱性称呼。
- 灭默 (Racism)：`默` 是 `黑犬` 的拼字，指对黑人群体的迫害行为。

补全作用证据：

- 默 (Racism) 直接命中句中目标。
- 反默、猎默、默妖、灭默 的定义/词面均连接了 `默`，强化其 Racism 目标群体属性。

匹配复核：

- 完整方法：hard_tp=1, soft_tp=1, soft_only_tp=0
- 全局检索基线：hard_tp=0, soft_tp=1, soft_only_tp=1
- 单样本 SRAG：hard_tp=0, soft_tp=1, soft_only_tp=1
- 零样本：hard_tp=0, soft_tp=1, soft_only_tp=1

GT：

```text
默 | 非我中华民族必为异类 | Racism | hate
```

完整方法输出：

```text
默 | 非我中华民族必为异类 | Racism | hate
```

未 hard 命中基线的实际输出与失效说明：

```text
全局检索基线（soft-only）：默 | 非我中华民族必为异类，畏威不畏德 | Racism | hate
  偏差：argument 右边界过长，额外合并了“畏威不畏德”。

单样本 SRAG（soft-only）：默 | 非我中华民族必为异类，畏威不畏德。 | Racism | hate
  偏差：argument 右边界过长，额外合并了“畏威不畏德”。

零样本（soft-only）：默 | 非我中华民族必为异类，畏威不畏德 | Racism | hate
  偏差：argument 右边界过长，额外合并了“畏威不畏德”。
```

解释：三个基线都把后续评价 `畏威不畏德` 合并进 argument。词典注入明确了 `默` 是隐晦种族侮辱词，使模型聚焦于与该目标直接绑定的核心排斥断言 `非我中华民族必为异类`，从而正确切分右边界。

## Case 4: id=5728

原句：天生的没办法，有些gay就是恶心，骚扰人，这纯纯没素质行为。但有些就干干净净，有干净的圈子和让人舒适的人品，让许多人愿意与他交朋友。再遇到这种骚扰人的你完全可以削他🤔，咱有理

Exact 匹配结果与词条解释：无。

Semantic 匹配结果与词条解释：

- 搅屎棍 (LGBTQ)：网络中被用于对男同性恋群体的侮辱性称呼。
- 腐癌 (Sexism, LGBTQ)：与 BL/性取向相关的极端腐文化现象。
- 色狼 (others)：指一些对他人作出性骚扰的人。
- 基 (LGBTQ)：`基` 是 `gay` 的粤语谐音，意为男同性恋者。

补全作用证据：

- 基 (LGBTQ) 的定义连接了 `gay`。
- 色狼 (others) 的定义连接了 `骚扰`，帮助确定 `恶心，骚扰人` 是评价片段。

匹配复核：

- 完整方法：hard_tp=1, soft_tp=1, soft_only_tp=0
- 全局检索基线：hard_tp=0, soft_tp=1, soft_only_tp=1
- 单样本 SRAG：hard_tp=0, soft_tp=1, soft_only_tp=1
- 零样本：hard_tp=0, soft_tp=0, soft_only_tp=0

GT：

```text
gay | 恶心，骚扰人 | LGBTQ | hate
```

完整方法输出：

```text
gay | 恶心，骚扰人 | LGBTQ | hate
```

未 hard 命中基线的实际输出与失效说明：

```text
全局检索基线（soft-only）：gay | 就是恶心，骚扰人 | LGBTQ | hate
  偏差：argument 左边界多抽了语气/判断词“就是”。

单样本 SRAG（soft-only）：有些gay | 恶心，骚扰人 | LGBTQ | hate
  偏差：target 左边界多抽了量化修饰“有些”。

零样本（soft 未命中）：gay | 恶心 | LGBTQ | hate
  偏差：argument 右边界漏掉“骚扰人”，没有覆盖完整负面行为。
```

解释：全局检索多抽了语气词 `就是`，单样本 SRAG 多抽了量化修饰 `有些`，零样本漏掉 `骚扰人`。semantic 词条同时连接 `gay` 与 LGBTQ、`骚扰` 与负面行为，使完整方法精确保留目标和评价片段边界。

## Case 5: id=6341

原句：但是有一说一，我从事旅游行业，在青岛，我们这对东北人的看法非常统一大家都挺反感他们的，发生矛盾也几乎都是跟东北人有些人是非常不讲理还感觉自己很有理，这不是少数现象，10次吵架8次东北的。所以我打心底里反感东北人，但是也有好的10个里面可能有两个

Exact 匹配结果与词条解释：无。

Semantic 匹配结果与词条解释：

- 低温人 (Region)：指东北人，因东北地区天气寒冷得来，常含贬义。
- 东百 (Region)：`东百` 是 `东北` 的中文谐音，常用于指代东北地区，含贬义。
- 九头鸟 (Region)：来源于地域刻板印象谚语，影射湖北人好战、不团结、精明等。
- 江浙人种 (Region)：讽刺部分江浙人对经济欠发达地区存在地域歧视。

补全作用证据：

- 低温人 (Region) 的定义连接了 `东北人`。
- 东百 (Region) 的定义连接了 `东北`。

匹配复核：

- 完整方法：hard_tp=1, soft_tp=1, soft_only_tp=0
- 全局检索基线：hard_tp=0, soft_tp=0, soft_only_tp=0
- 单样本 SRAG：hard_tp=0, soft_tp=0, soft_only_tp=0
- 零样本：hard_tp=0, soft_tp=0, soft_only_tp=0

GT：

```text
东北人 | 打心底里反感 | Region | hate
```

完整方法输出：

```text
东北人 | 大家都挺反感他们的 | Region | hate
东北人 | 非常不讲理还感觉自己很有理 | Region | hate
东北人 | 打心底里反感 | Region | hate
```

未 hard 命中基线的实际输出与失效说明：

```text
全局检索基线（soft 未命中）：
东北人 | 大家都挺反感他们 | Region | hate
东北人 | 不是少数现象，10次吵架8次东北的 | Region | hate
  偏差：抽取了前文泛化评价和统计式表达，漏掉 GT 指定的“打心底里反感”。

单样本 SRAG（soft 未命中）：
东北人 | 挺反感 | Region | hate
有些人 | 非常不讲理还感觉自己很有理 | Region | hate
东北的 | 10次吵架8次 | Region | hate
  偏差：第一条过度截短，后两条目标或 argument 与 GT 不一致，漏掉“东北人 | 打心底里反感”。

零样本（soft 未命中）：
东北人 | 我们这对东北人的看法非常统一大家都挺反感他们的 | Region | hate
有些人 | 非常不讲理还感觉自己很有理 | Region | hate
  偏差：抽取了更早的群体评价句，漏掉句末核心主观评价“打心底里反感”。
```

解释：该句包含多个相近地域评价片段，基线模型容易被前文 `大家都挺反感`、`10次吵架8次东北的` 等局部线索吸引，漏掉 GT 指定的核心主观评价 `打心底里反感`。semantic 词条把 `东北人/东北` 映射到 Region 贬义背景，完整方法虽然额外抽取了其他相关片段，但包含了与 GT 完全一致的关键跨度。

## 补充 Case：完整方法 hard 命中，三个基线均 hard 未命中

以下补充案例均满足：完整方法 `hard_tp>0`；全局检索基线、单样本 SRAG、零样本三者均 `hard_tp=0`。

## Case 6: id=5373

原句：这些黑人崇拜者为黑人做了大量正面宣传。应该占据了国内所有渠道对黑人的正面宣传的99%以上吧。而且是完全自愿，义务，积极，和免费的。宣传多了，那么自然就会有越来越多的女孩倒贴黑人。

Exact 匹配结果与词条解释：无。

Semantic 匹配结果与词条解释：

- 舔黑 (Racism)：描述一种极端崇拜和追求与黑人相关事物的社会现象，常含贬义。
- 黑吹 (Racism)：指一种极端崇拜和追求与黑人相关事物的社会群体。
- 黑桃女 (Racism, Sexism)：指偏好黑人男性的女性，是对崇拜黑人男性的女性的标签化称呼，常含贬义。
- 黑爹 (Racism)：用于描述对某种过度崇拜黑人群体的现象，常含贬义。

补全作用证据：

- 舔黑、黑吹、黑桃女、黑爹 的定义/词面连接了 `崇拜`、`黑人`。
- 这些词条把 `黑人崇拜者` 与 Racism 背景对齐，并帮助模型保留 `为黑人做了大量正面宣传` 的完整 predicate 边界。

匹配复核：

- 完整方法：hard_tp=1, soft_tp=1, soft_only_tp=0
- 全局检索基线：hard_tp=0, soft_tp=0, soft_only_tp=0
- 单样本 SRAG：hard_tp=0, soft_tp=1, soft_only_tp=1
- 零样本：hard_tp=0, soft_tp=0, soft_only_tp=0

GT：

```text
黑人崇拜者 | 为黑人做了大量正面宣传 | Racism | hate
女孩 | 倒贴黑人 | Sexism, Racism | hate
```

完整方法输出：

```text
黑人崇拜者 | 为黑人做了大量正面宣传 | Racism | hate
这些黑人崇拜者 | 宣传多了，那么自然就会有越来越多的女孩倒贴黑人 | Sexism, Racism | hate
```

未 hard 命中基线的实际输出与失效说明：

```text
全局检索基线（soft 未命中）：
这些黑人崇拜者 | 为黑人做了大量正面宣传 | Sexism, Racism | hate
  偏差：target 多抽了“这些”，且类别多抽 Sexism，未 hard 命中 GT 的第一条。

单样本 SRAG（soft-only）：
黑人崇拜者 | 做了大量正面宣传 | Racism | hate
倒贴黑人 | 越来越多了 | Sexism, Racism | hate
  偏差：第一条 argument 左边界漏掉“为黑人”；第二条 target/argument 切分倒置，未 hard 命中。

零样本（soft 未命中）：
黑人崇拜者 | 倒贴黑人 | Racism | hate
  偏差：把第一条 target 与第二条动作片段错误拼接，混淆了两个事件。
```

解释：semantic 词条集中补充了 `黑人崇拜` 相关概念，使完整方法能把 `黑人崇拜者` 与 `为黑人做了大量正面宣传` 作为一条完整关系抽出；基线要么多抽类别，要么漏掉 predicate 的关键左边界。

## Case 7: id=5252

原句：本安徽人来说一句。浙江人是不屑于地域黑的，他们都钻到钱眼里去了。没时间没精力地域黑。他们的好邻居江苏那可真是地域黑十斗，江苏人独占八斗。比完经济比教育，比完教育比两院，比完两院比状元。身在江苏，没几个屌钱。动辄我苏天下无敌，宇宙苏。身处陋室，捡破烂，到了网上那可是阿拉江苏人

Exact 匹配结果与词条解释：

- 地域黑 (Region)：即地域歧视，是由对某一地区的认知性偏差和刻板印象导致的社会行为。
- 屌 (Sexism)：男性生殖器的俗称。

Semantic 匹配结果与词条解释：

- 江浙人种 (Region)：讽刺部分江浙人对经济欠发达地区存在地域歧视。
- 南宋人 (Region)：地域相关贬义称呼。
- 江北八市 (Region)：地域相关称谓。
- 南大人 (Region)：讽刺南方群体高高在上的姿态。

补全作用证据：

- 江浙人种连接了 `江苏`、`浙江`、`地域`。
- 南宋人、江北八市 连接了 `江苏/江苏人`。

匹配复核：

- 完整方法：hard_tp=1, soft_tp=2, soft_only_tp=1
- 全局检索基线：hard_tp=0, soft_tp=1, soft_only_tp=1
- 单样本 SRAG：hard_tp=0, soft_tp=1, soft_only_tp=1
- 零样本：hard_tp=0, soft_tp=2, soft_only_tp=2

GT：

```text
浙江人 | 钻到钱眼里 | Region | hate
江苏 | 地域黑十斗，江苏人独占八斗 | Region | hate
```

完整方法输出：

```text
浙江人 | 钻到钱眼里去了 | Region | hate
江苏 | 地域黑十斗，江苏人独占八斗 | Region | hate
```

未 hard 命中基线的实际输出与失效说明：

```text
全局检索基线（soft-only）：
浙江人 | 不屑于地域黑 | Region | hate
江苏 | 地域黑十斗 | Region | hate
  偏差：第一条抽错评价片段；第二条漏掉“江苏人独占八斗”。

单样本 SRAG（soft-only）：
江苏人 | 地域黑十斗，江苏人独占八斗 | Region | hate
  偏差：target 应为“江苏”，多抽为“江苏人”；且漏掉“浙江人”相关输出。

零样本（soft-only）：
浙江人 | 钻到钱眼里去了 | Region | hate
江苏人 | 地域黑十斗 | Region | hate
  偏差：第一条为 soft-only，第二条 target 多抽“人”且 argument 漏掉“江苏人独占八斗”。
```

解释：Region 词条注入帮助模型区分长句中的两个地域目标，并完整保留 `江苏 | 地域黑十斗，江苏人独占八斗` 这一条 hard 命中的边界。

## Case 8: id=1276

原句：那果子狸是男的吗？给这个人点赞的都是男的嘛？😄知乎b站微博里面的女拳师都是男的？

Exact 匹配结果与词条解释：

- 果子狸 (Sexism)：指微博平台一知名女权博主，常发表极端言论，煽动性别对抗。
- 女拳 (Sexism)：`女拳` 是 `女权` 的谐音，常用于对中国女权主义者的贬义表达。
- 拳师 (Sexism)：常用于对中国女权主义者的贬义表达。

Semantic 匹配结果与词条解释：

- 田园女拳 (Sexism)：以女权为借口追求女性收益最大化的群体。
- 田园婊 (Sexism)：对女权/女性群体的贬义称呼。
- 索玛女王 (Sexism)：指常发表极端言论、煽动性别对抗的女权博主。

补全作用证据：

- Exact 直接命中 `果子狸`、`女拳`、`拳师`。
- Semantic 词条进一步连接 `女拳`、`微博` 等线索，帮助模型保留长目标 `知乎b站微博里面的女拳师`。

匹配复核：

- 完整方法：hard_tp=2, soft_tp=2, soft_only_tp=0
- 全局检索基线：hard_tp=0, soft_tp=0, soft_only_tp=0
- 单样本 SRAG：hard_tp=0, soft_tp=0, soft_only_tp=0
- 零样本：hard_tp=0, soft_tp=0, soft_only_tp=0

GT：

```text
果子狸 | 是男的吗？ | Sexism | hate
知乎b站微博里面的女拳师 | 都是男的？ | Sexism | hate
```

完整方法输出：

```text
果子狸 | 是男的吗？ | Sexism | hate
给这个人点赞的 | 都是男的嘛？ | Sexism | hate
知乎b站微博里面的女拳师 | 都是男的？ | Sexism | hate
```

未 hard 命中基线的实际输出与失效说明：

```text
全局检索基线（soft 未命中）：女拳师 | 都是男的？ | Sexism | hate
  偏差：漏掉“果子狸”这一条；第二条 target 左边界漏掉“知乎b站微博里面的”。

单样本 SRAG（soft 未命中）：女拳师 | 都是男的？ | Sexism | hate
  偏差：同上，只抽到短 target，未覆盖完整平台限定语。

零样本（soft 未命中）：女拳师 | 都是男的？ | Sexism | hate
  偏差：同上，漏掉第一条并截短第二条 target。
```

解释：词典注入明确了 `果子狸` 与 `女拳师` 的 Sexism 背景，使完整方法同时 hard 命中两个 GT；三个基线都只保留了短目标 `女拳师`，且漏掉 `果子狸`。

## Case 9: id=1615

原句：在古代男性向外族播种是征服者的体现。女性被播种怀上外族子嗣是种族被其他种族征服的体现，区别还是很大的。

Exact 匹配结果与词条解释：无。

Semantic 匹配结果与词条解释：

- 亩人 (Sexism)：女性相关贬义称呼。
- 中华田园犬 (Sexism)：女性/男性相关性别对立语境中的贬义称呼。
- 田园女拳 (Sexism)：以女权为借口追求女性收益最大化的群体。
- 田园女 (Sexism)：中华田园女权的简称。

补全作用证据：

- 亩人 连接了 `女性`。
- 中华田园犬、田园女拳、田园女 连接了 `女性`、`男性`，强化该样本的 Sexism 而非 Racism 主类别。

匹配复核：

- 完整方法：hard_tp=1, soft_tp=1, soft_only_tp=0
- 全局检索基线：hard_tp=0, soft_tp=0, soft_only_tp=0
- 单样本 SRAG：hard_tp=0, soft_tp=0, soft_only_tp=0
- 零样本：hard_tp=0, soft_tp=0, soft_only_tp=0

GT：

```text
男性 | 向外族播种是征服者的体现 | Sexism | hate
女性 | 被播种怀上外族子嗣是种族被其他种族征服的体现 | Sexism | hate
```

完整方法输出：

```text
女性 | 被播种怀上外族子嗣是种族被其他种族征服的体现 | Sexism | hate
```

未 hard 命中基线的实际输出与失效说明：

```text
全局检索基线（soft 未命中）：
女性 | 被播种怀上外族子嗣是种族被其他种族征服的体现 | Sexism, Racism | hate
  偏差：跨度正确但 targeted_group 多抽 Racism，导致 hard/soft 均未命中。

单样本 SRAG（soft 未命中）：
女性 | 被播种怀上外族子嗣是种族被其他种族征服的体现 | Sexism, Racism | hate
  偏差：同上，类别边界过宽。

零样本（soft 未命中）：
女性 | 被播种怀上外族子嗣是种族被其他种族征服的体现 | Sexism, Racism | hate
  偏差：同上，混入 Racism 类别。
```

解释：三个基线其实抽到了同一文本跨度，但类别边界过宽，把跨族叙述误并入 Racism。semantic 性别词条使完整方法将该片段限定为 Sexism，从而 hard 命中第二条 GT。

## Case 10: id=692

原句：真是勇敢的小仙女！

Exact 匹配结果与词条解释：

- 仙女 (Sexism)：在某些情况下被用来指代自私自利、双标、让人厌恶的女性。
- 小仙女 (Sexism)：在某些情况下被用来指代自私自利、双标、让人厌恶的女性。

Semantic 匹配结果与词条解释：

- 小蚬屡 (Sexism)：`小仙女` 的谐音/变体。
- 小酰钕 (Sexism)：`小仙女` 的谐音/变体。
- 小藓女 (Sexism)：`小仙女` 的谐音/变体。

补全作用证据：

- Exact 直接命中 `小仙女`。
- Semantic 变体词条均连接 `小仙女`，强化其 Sexism/hate 用法。

匹配复核：

- 完整方法：hard_tp=1, soft_tp=1, soft_only_tp=0
- 全局检索基线：hard_tp=0, soft_tp=0, soft_only_tp=0
- 单样本 SRAG：hard_tp=0, soft_tp=0, soft_only_tp=0
- 零样本：hard_tp=0, soft_tp=0, soft_only_tp=0

GT：

```text
小仙女 | 勇敢 | Sexism | hate
```

完整方法输出：

```text
小仙女 | 勇敢 | Sexism | hate
```

未 hard 命中基线的实际输出与失效说明：

```text
全局检索基线（soft 未命中）：小仙女 | 真是勇敢的小仙女！ | non-hate | non-hate
  偏差：argument 过长，且误判为 non-hate。

单样本 SRAG（soft 未命中）：小仙女 | 勇敢 | non-hate | non-hate
  偏差：文本边界正确，但 hate/targeted_group 判断错误。

零样本（soft 未命中）：小仙女 | 勇敢 | non-hate | non-hate
  偏差：同上，未识别反讽贬义。
```

解释：`小仙女` 在短句中容易被基线当作普通称呼；词典注入提供了贬义变体和 Sexism 背景，使完整方法识别出反讽语境并 hard 命中。

## Case 11: id=6406

原句：上次遇到一个坦克，直接跟说自己这么漂亮怎么可能没谈过对象我就知道是什么成分了

Exact 匹配结果与词条解释：

- 坦克 (Sexism)：对体型较胖女性的侮辱性称呼。
- 什么成分 (others)：网络用语，常用于质疑或贬低对方身份/立场。

Semantic 匹配结果与词条解释：

- 开坦克 (Sexism)：指与身材偏胖的女生发生关系的行为。
- 巨坦 (Sexism)：与 `坦克` 相关的体型侮辱性称呼。
- 公交车 (Sexism)：对女性的侮辱性称呼。
- 丑逼 (Sexism, others)：调侃或讽刺长相普通/不好看的人，含贬义。

补全作用证据：

- 坦克 exact 命中目标。
- 开坦克 的定义/词面连接了 `坦克`，补强 Sexism 背景。

匹配复核：

- 完整方法：hard_tp=1, soft_tp=1, soft_only_tp=0
- 全局检索基线：hard_tp=0, soft_tp=0, soft_only_tp=0
- 单样本 SRAG：hard_tp=0, soft_tp=0, soft_only_tp=0
- 零样本：hard_tp=0, soft_tp=0, soft_only_tp=0

GT：

```text
坦克 | 什么成分 | Sexism | hate
```

完整方法输出：

```text
坦克 | 什么成分 | Sexism | hate
```

未 hard 命中基线的实际输出与失效说明：

```text
全局检索基线（soft 未命中）：
坦克 | 直接跟说自己这么漂亮怎么可能没谈过对象我就知道是什么成分了 | Sexism | hate
  偏差：argument 过长，把叙事背景全部合并进混淆跨度。

单样本 SRAG（soft 未命中）：
坦克 | 直接跟说自己这么漂亮怎么可能没谈过对象 | Sexism | hate
  偏差：argument 抽成中间叙述，漏掉核心评价“什么成分”。

零样本（soft 未命中）：
坦克 | 直接跟说自己这么漂亮怎么可能没谈过对象 | Sexism | hate
  偏差：同上，未抽取真正的评价片段。
```

解释：Exact 词条识别 `坦克` 为 Sexism 目标，semantic 近邻进一步强化相关贬义语境，使完整方法将 argument 收束到真正的评价词 `什么成分`，而三个基线都抽取了过长或错误的叙事片段。
