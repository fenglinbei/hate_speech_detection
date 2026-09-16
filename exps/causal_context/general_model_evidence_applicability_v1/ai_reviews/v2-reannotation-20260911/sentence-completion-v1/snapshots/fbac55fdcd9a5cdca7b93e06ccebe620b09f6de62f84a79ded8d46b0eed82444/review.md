# 句子部分收尾复核

共270句；本轮原涉及71句，政策补充后共73句。已有数值严重度 270/270；原9条历史类别复核已处理 9/9，补充复核已处理 2 条。

当前表按严重度映射 hate。字段分别标明用户裁决、AI补标和规则映射；未回答的建议不视为确认。旧人工裁决及其政策保留在历史层。

[完整270句 JSON](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/current-sentences.json) · [完整270句 CSV](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/current-sentences.csv) · [本轮 JSON](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/review.json)

## 待讨论

本轮疑问项已处理完毕。

## 本轮记录

### #1035

> 女性到底应不应该有权利呢？

non-hate / 0 / Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：提问女性权利，没有给出否定权利或贬损命题。


前值（历史保留）：attack_severity=待定

### #1242

> 我认为的偏见是指在还不了解真实情况或者了解不全的的时候就对某一件事情带有明显的情绪化和极端的观点，例如山西都是挖煤的，上海男人说话都是娘娘腔这种就是很明显的地域偏见，我认为你的好恶只是说你不喜欢txl，可我没说你是偏见。你的好恶是自由的，但偏见不是。

non-hate / 0 / Region、LGBTQ、Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：将刻板印象列为偏见并区分好恶，没有认可列举的地域或性别污名。


前值（历史保留）：attack_severity=待定

### #127

> 这问题让我明白为什么基佬在唐山不会被打。基佬染艾原来是为了防身！

hate / 2 / LGBTQ

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：以基佬染艾防身作笑料，认可群体疾病污名；没有认可殴打。


前值（历史保留）：attack_severity=待定

### #1294

> 歧视这个词不太好，太过了。但确实，很多txl给人的观感并不好，我觉得，这个网络也对人有太多影响，短视频里那些娘里娘气的、装疯卖傻的，让人反感，给孩子们不好的教育示范。

hate / 2 / LGBTQ

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：虽先说歧视太过，仍以娘里娘气、装疯卖傻和负面教育示范污名化同性恋。


前值（历史保留）：attack_severity=待定

### #1336

> 十几年前我刚上网的时候网上东北的名声还行，最起码“东北曾经是发达地区只是现在衰落了”这种常识哪怕是再黑东北的人也不会否认，但是这几年不知道怎么了，可能是智能手机便宜了流量便宜了，很多地域黑居然张口就是东北是贫困山区一样的地方😓

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：反驳针对东北经济的妖魔化，地区经济描述不自动变为居民攻击。


前值（历史保留）：attack_severity=待定

### #1376

> 小管家还会威胁你⁉⁉⁉匿名也能逮到啊⁉⁉

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：询问他人遭遇的威胁，不是作者威胁或贬损。


前值（历史保留）：attack_severity=待定

### #1466

> 如果那个黑人对中国友好，那么我倒可以忍受，就像快手上有个嫁黑人的，那个黑人对中国很友好，我倒也可以忍受。虽然还是不喜欢，但只要是与中国友好的黑人，哪怕不喜欢，我也可以忍受

hate / 1 / Racism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：按黑人身份表达不喜欢并使用忍受的居高临下措辞，取轻度贬评；未主张权利排斥。


前值（历史保留）：attack_severity=待定

### #1509

> EG都不是找那种老g吗。优秀的别人看不上

hate / 2 / Sexism

字段来源：hate=rule_mapping_from_human_score；group=user_discussion；attack_severity=user_discussion

AI依据：用户补定2级：可见择偶贬损构成明确性别污名。老g的具体义仍未确认，不因这项词义不明继续搁置已有依据支持的严重度；原hate和Sexism人工字段保留。

- 用户 attack_severity：2；原答：2 级：明确性别污名

前值（历史保留）：attack_severity=待定；hate=待定

### #1653

> 偶然看到，就评论一下。都是人类，性取向就是萝卜白菜，各有所爱，没有什么高低贵贱。只是想到男男，生理上超级排斥，也就是感觉恶心

non-hate / 0 / LGBTQ

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：先否定身份高低贵贱，再描述对男男亲密行为的自身生理反应。参照已确认快照 #2430，不直接推成群体低等或排斥。


前值（历史保留）：attack_severity=待定

### #1660

> 不是，討厭女性，不想跟女性相處，這不是歧視，你隻是gay了

hate / 1 / LGBTQ、Sexism

字段来源：hate=rule_mapping_from_human_score；group=ai_note；attack_severity=user_discussion

AI依据：用户将厌女与gay相联的说法校准为轻度调侃1级；不按AI最初建议升到一般污名2级。group仍为AI提议LGBTQ、Sexism。

- 用户 attack_severity：1；原答：1 级：轻度调侃

前值（历史保留）：attack_severity=待定

### #1786

> 个人觉得吧，挑头的有错，但是跟后头闹事儿的就无辜？

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：追问跟随闹事者是否有责任，只评价行为。


前值（历史保留）：attack_severity=待定

### #1833

> 雄性基因是排外，雌性基因是择优，如果思想上升就不必去骂黑人，但在这个世界大染缸里，清醒的人毕竟是少数，大部分人靠本能去抉择，即使我明白我也一定排黑的，同族群能容我，而异族必不容我

hate / 2 / Racism、Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：认可按族群排斥，并以性别基因概括行为；参照 #924 的一般污名档，不因基因词自动升至3。


前值（历史保留）：attack_severity=待定

### #191

> 先自己调查下好吧，南方系更多好像说的是广东，而且这也是实话，南方系媒体系统性BBC操作还少吗？你一个南方老百姓跟着他们冲锋陷阵啥？大家都是普通老百姓没必要。

non-hate / 0 / Region、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：质疑媒体操作及普通民众跟随行为；BBC在此有媒体语境，不套用 #1915 的特定隐语裁决。


前值（历史保留）：attack_severity=待定

### #1986

> 真的一上知乎全都是骂的，从女权骂到了女性，然而在现实生活中根本见不着，是我层次太低？

non-hate / 0 / Sexism、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：报告网上辱骂女性的情况并自问现实差异，没有认可所述攻击。


前值（历史保留）：attack_severity=待定

### #2004

> 这个吧里都是**有个女的喷我然后让我教育了自己外地的去新一线然后瞧不起我大连土著说我是18线小县城的房子不稀罕殊不知她连户口都没落自称当地人

hate / 2 / Region、others

字段来源：hate=rule_mapping_from_human_score；group=user_discussion；attack_severity=user_discussion

AI依据：用户补定明确辱骂2级；结合已有星号为联想写法和地域争执的裁决，不补造隐藏词。原hate、Region和others人工字段保留。

- 用户 attack_severity：2；原答：2 级：明确辱骂

前值（历史保留）：attack_severity=待定；hate=待定

### #2025

> 已经完全失去了对中国年轻女性的信任（35以下），反正我现在看任何一个我不是很熟的年轻女的都先贴上两个标签 女拳王 媚黑，就像女的说男的都是潜在强奸犯一样，互相戴帽子呗，两性之间的信任会越来越少，直至完全对立🤔

hate / 2 / Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：明确按女性群体预贴女拳王、媚黑标签，承认互相戴帽子也未撤回污名。


前值（历史保留）：attack_severity=待定

### #2054

> 爱马仕不香么？

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：品牌偏好疑问，没有人的攻击对象或具体贬损命题。


前值（历史保留）：attack_severity=待定

### #2075

> 原来女性的地位由男同决定?你这个思想属实搞笑，对于男同仇女确实有，但是我觉得大多数不会，还有你不会以为抵制男同就没有男同了吧，男同事基因，不是后天学习，打压男同只会有更多的男同因为社会压力骗婚，这道理都不懂。

non-hate / 0 / LGBTQ、Sexism

字段来源：hate=rule_mapping_from_human_score；group=user_discussion；attack_severity=user_discussion

AI依据：用户认定只批评论点，严重度0；不将这道理都不懂独立转为个人能力辱骂，不新增others。hate由0映射为non-hate。

- 用户 group：LGBTQ、Sexism；原答：0／LGBTQ、Sexism：只批评论点
- 用户 attack_severity：0；原答：0／LGBTQ、Sexism：只批评论点

前值（历史保留）：attack_severity=待定；group=LGBTQ、Sexism

### #2091

> 女性在非洲地位很低

non-hate / 0 / Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：描述女性在非洲的社会地位，不表示赞成这种处境。


前值（历史保留）：attack_severity=待定

### #2117

> 这话说的，好人就该被枪指着。骗人的不管被骗的有罪✌️

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：通过反问反驳责备被骗者的逻辑；枪指着是被反对的情形。


前值（历史保留）：attack_severity=待定

### #2120

> 其实，爱滋在男同里的比例很低，反倒是异性恋得爱滋的比例高，不信去上网查查。我国对同性恋采取不反对、不提倡。

non-hate / 0 / LGBTQ、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：以其理解的疾病比例和政策反驳偏见，没有对人群攻击；此处不判断陈述的事实准确性。


前值（历史保留）：attack_severity=待定

### #2132

> 嫁娶我看都一样，都是一半基因。黑男黄女和黑女黄男生的孩子都一样黑

non-hate / 0 / Racism、Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：比较婚配与遗传肤色，没有附加肤色高低优劣。


前值（历史保留）：attack_severity=待定

### #2218

> 没见过，只是他说“朋友说新疆人拿刀砍人很正常”我就回了这么一句

non-hate / 0 / Region

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：明确标示别人说过的暴力描述，未表达作者认可。


前值（历史保留）：attack_severity=待定

### #229

> 度度又在刷屏啊，找没找到我是小号的证据啊

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：质询刷屏和小号指控的证据，只谈行为。


前值（历史保留）：attack_severity=待定

### #2349

> 你气炸了的部分原因也有物化女性的成分吧？认为中国女性是中国男性的伴侣，但是这不是印度，女性也能顶半边天。女性和黑人和老外谈恋爱我觉得不要去管那么多，但是如果存在烂交群P等事情，发现一个公布一个，毕竟违法了。

non-hate / 0 / Racism、Region、Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：反对物化和干涉婚恋，后段批评特定行为；没有据身份泛化攻击。法律断言不作为标注事实依据。


前值（历史保留）：attack_severity=待定

### #235

> 大东北现在也不差，主要是网络媒体很多带节奏的别有用心的人搞事情。各项指标：GDP总量、人均GDP、人均可支配收入、人类发展指数、高等教育受教育人数比例等等，都在全国中上游，辽宁还排在前十名之内。

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：用地区指标反驳带节奏者的叙述，别有用心在此针对操纵信息行为。


前值（历史保留）：attack_severity=待定

### #2351

> 这是打算开始通过歧视光棍来试图提高结婚率了？

non-hate / 0 / others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：质疑歧视光棍的做法，没有认可歧视。


前值（历史保留）：attack_severity=待定

### #2416

> 黑东北是某年之后爆发性出现的，大概是2013前后。你可以发现很多吐槽东北的号都是小号的感觉，除了黑东北之外都没有别的使用痕迹了。东北经济大不如前，所以这里的民风肯定算不上第一梯队，但也没有这几年网上妖魔化那么严重。

non-hate / 0 / Region

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：整体解释并反对妖魔化东北，没有把地区民风描述为群体本性低等。


前值（历史保留）：attack_severity=待定

### #243

> 同性恋大多是先天的，只是为了生存不得不一开始学会伪装。还有，同性恋也没法传染。异性恋身上也没有同性恋的基因。

non-hate / 0 / LGBTQ、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：解释性取向与伪装、传染说法，不带攻击；不以本次标注确认其生物学说法。


前值（历史保留）：attack_severity=待定

### #2445

> 不好意思，你们的言论我看不懂，因为我从来没有靠过男人（除了我爸爸），也没有诋毁过任何一个正常男性 你们说的那些人跟我也不是一个圈子的，我也无法理解，所以，我也只是从我的角度说出我的想法，意见不一样很正常，我也接受任何的批评

non-hate / 0 / Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：说明自身经历和意见差异，没有指认某类男性不正常。


前值（历史保留）：attack_severity=待定

### #2496

> 可以理解，你身边估计没有这样的人，你对同性恋的了解可能仅存于网上耽美小说，真正的爱情无论是男是女都是可以打动一个人的，我身边就有这样一对情侣让我彻底改变了对同性恋的看法，真正接触到才会发现同性恋就是普普通通的恋爱呀，和异性恋没有什么区别，那些同性恋者很勇敢，他们可以在常人的异样眼光下勇敢的表达爱情，真正接触了才知道原来同性恋不是下流恶心的，而是和异性恋一样纯洁高贵的爱情

non-hate / 0 / LGBTQ、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：通过亲身接触反驳同性恋污名，负面词被明确否定。


前值（历史保留）：attack_severity=待定

### #25

> 起码在冯丽这件事情上，是这几个博主到冯丽微博下留言加辱骂河南人的，到处蹭热度，我都怀疑这些人是不是有病呀！你骂他他就截屏，说你网暴他，你不骂他马上就跑到话题下继续地域黑，我去过西安，是一座美丽的城市，文化很厚重，但华夏把中原说做陕西，黄帝故里是西安等等常识性错误的言论，还不让人说洛阳是十三朝古都，不是洛阳是古都会影响西安什么吗？无语😲😲😲😲

hate / 1 / Region、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：总体反对地域黑，但有病呀另带对具体博主的轻度个人贬评。原AI仅Region是在旧口径下，现按个人辱骂规则补others。


前值（历史保留）：attack_severity=待定；group=Region

### #2517

> “应该”是一个很复杂的词，智力不足的人在讨论严肃的问题时使用该词会造成歧义，比如现在，读者就无法理解你究竟想表达“不排斥就该被惩罚”还是“不排斥对自身或许会有不好的影响”，这是不好的。

hate / 1 / others

字段来源：hate=rule_mapping_from_human_score；group=user_discussion；attack_severity=user_discussion

AI依据：用户确认智力不足的人为轻度能力贬评，定1级、others；与#309轻度人格贬评锚点相容。

- 用户 group：others；原答：1／others：轻度能力贬评
- 用户 attack_severity：1；原答：1／others：轻度能力贬评

前值（历史保留）：attack_severity=待定；group=[]

### #2531

> 黑g还要考驾照，供非送的钱太多了吧，太恐怖了，要有猩司机了

hate / 3 / Racism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：以猩司机称黑人，把人直接动物化。


前值（历史保留）：attack_severity=待定

### #255

> 相比较来说我觉得普信男还可以一不危害社会，还能自强不息。相反对社会危害最大的是那些既当婊子有立牌坊的人，这可不是什么荡妇羞辱，相反我对一些光明坦荡的性开放是持欣赏态度，现在某些群体又当又立的行为一直到了颠倒黑白混淆丑美的地步了。

hate / 2 / Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：既当婊子又立牌坊及又当又立构成明确侮辱；反对荡妇羞辱的自我声明未消除实际贬称。


前值（历史保留）：attack_severity=待定

### #2554

> 犹太人凯乐奇计划就是要中国引入其他种族，制造新的种族，灭绝中华民族、中华文化。

hate / 2 / Racism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：不同于 #1292 仅提及计划名，本句明确指控犹太人以混合种族灭绝中华民族，构成群体污名；是恶意指控而非作者认可灭绝。


前值（历史保留）：attack_severity=待定

### #2593

> jiabao很难说平等，由于体型和道德要求的区别谁占优势还很难说，只能说是不合情合理，像以前jiabaosiren男的会定过失致人siwangzui，女的就直接是故意伤人了

non-hate / 0 / Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：讨论其理解的家暴及司法差异，没有认可伤害，也不据此确认法律陈述。


前值（历史保留）：attack_severity=待定

### #2655

> 很难不赞同，不过补充一点，杨靖宇将军是河南的，本名马尚德，驻马店的骄傲🍉

non-hate / 0 / Region

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：对河南人物作正面介绍，没有攻击。


前值（历史保留）：attack_severity=待定

### #2682

> 男的比女的多几千万呢！这些光棍当然要找外国女人，找中国的没有了啊。这叫双标吗？

non-hate / 0 / Region、Sexism、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：以婚配人数回应双标指控，没有进一步贬损婚姻身份。


前值（历史保留）：attack_severity=待定

### #2685

> 所以当前突出问题也包括女汉奸的问题，这点这个🐶的贴已经说明了

hate / 2 / Sexism、others

字段来源：hate=rule_mapping_from_human_score；group=ai_note；attack_severity=user_discussion

AI依据：用户将女汉奸及🐶用法判为一般明确侮辱2级，与#4810的狗东西2级一致；不因单个动物表情自动升到3级。

- 用户 attack_severity：2；原答：2 级：一般明确侮辱

前值（历史保留）：attack_severity=待定

### #2747

> 她得艾滋了你别去碰她不就好了，得个艾滋而已总好过去非洲送死吧。再说了，她可以嫁给男艾滋病人嘛，解决中国男艾滋病人的结婚需求不是挺好？中国同胞干嘛便宜非洲佬！

hate / 2 / Region、Sexism、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：借疾病、嫁娶和便宜非洲佬表达轻蔑并将女性当婚配资源；未赞成身体伤害。


前值（历史保留）：attack_severity=待定

### #2750

> 当你已经沦落到与乐色为伍，还把乐色举高过自己的头顶的时候，你已经连乐色都不如了。那天我去买塑料袋。有黄色的白色的透明的还有黑色的，我问老板，哪一种颜色的最便宜？老板说黑色的最便宜。我问为什么黑色的最便宜？老板答，因为黑色的是垃圾材料。

hate / 3 / Racism、others

字段来源：hate=rule_mapping_from_human_score；group=ai_note；attack_severity=user_discussion

AI依据：个人受话者与黑色垃圾材料所影射的黑人群体不同，未证明受话者为黑人；按新同一对象条件保留others。旧人工group保持相同值，此次适用性判断为AI复核。


前值（历史保留）：group=Racism、others

### #2759

> 九几年的故事会就有一个故事——黑人富商，女大学生，怀孕，黑人离开，接盘侠很开心😆

hate / 1 / Racism、Sexism

字段来源：hate=rule_mapping_from_human_score；group=ai_note；attack_severity=user_discussion

AI依据：用户确认借故事情节轻度挖苦，取1级；并非中性转述。group仍为原AI的Racism、Sexism，未把此次分数回复当作人工group确认。

- 用户 attack_severity：1；原答：1 级：借故事轻度挖苦

前值（历史保留）：attack_severity=待定

### #2787

> 那倒也没有，我觉得女的接受度更高。毕竟百合向大多数读者也是女性。就算不看，我看她们很多也对同性恋博主持友好态度。也有接受不了双标的，相对来说吧。不过这种一般会被人批评😅

non-hate / 0 / LGBTQ、Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：讨论不同受众对同性恋内容的接受度，亦提及批评双标；未见攻击。


前值（历史保留）：attack_severity=待定

### #2792

> 河南有一些问题，很多不好的印象其实并不来自于新闻，而是来自人们的亲身体验。至少在80年代，河南的名声就不太好了。另外建议河南对于歧视的反驳，不要集中在历史悠久名人众多产粮大省啥的，你说人遇到两个河南骗子，形成了刻板印象，你给他说河南历史悠久，你觉得有用没？当然现在河南已经好多了，我已经很久没有碰到人歧视我了，希望以后会越来越好。当然在西北地区我就不奢望了。

non-hate / 0 / Region

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：分析负面印象和反驳方法，表达希望歧视减少；未将河南身份判为低等。


前值（历史保留）：attack_severity=待定

### #2824

> 不懂，回族有什么问题？

non-hate / 0 / Racism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：追问回族有什么问题，没有提供负面答案。


前值（历史保留）：attack_severity=待定

### #2840

> 这个事情就是信誉受损问题，男的也知道不是全部女生都会找黑人，但是信誉受损以后 修复时间很长。

hate / 2 / Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：将部分女性与黑人交往归为全体女性信誉受损，认可性别污名。


前值（历史保留）：attack_severity=待定

### #2945

> 这也能遇到你

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：偶遇感叹，没有攻击内容。


前值（历史保留）：attack_severity=待定

### #2994

> 我的意思是为什么大家不骂黑人，不骂那些为了钱去找黑人的女的，而是要骂所以女的？难不成天底下下所有人都喜欢黑人？在座的各位的妈妈也喜欢黑人？哈哈哈，搞笑

hate / 2 / Racism、Sexism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：虽反对骂所有女性，但明确认可转而骂黑人及特定择偶女性，含种族性别污名。


前值（历史保留）：attack_severity=待定

### #3081

> 请问你是自己见过，还是经历过？

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：询问信息来源或经历，没有贬损。


前值（历史保留）：attack_severity=待定

### #3177

> 还理性讨论，还自由？扯。我就说了三从四德是糟粕，反女权不要抗旧社会大旗，立马骂我并禁言了

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：批评旧规范与禁言辱骂行为，未把制度观点变成人群身份辱骂。


前值（历史保留）：attack_severity=待定

### #3212

> 一直就有大把女人反对女拳，现实中女拳并不多，大多混迹在网络

hate / 1 / Sexism

字段来源：hate=rule_mapping_from_human_score；group=ai_note；attack_severity=user_discussion

AI依据：用户认定混迹一词已提供轻度贬损依据，定1级；区别于#4189只批评假冒行为和#6074只谈资格梗。

- 用户 attack_severity：1；原答：1 级：混迹一词带有轻度贬损

前值（历史保留）：attack_severity=待定

### #3252

> 弱势群体为什么不能反击？不是每个弱势群体的人都应该因为弱势而就懦弱或者不能以眼还眼。

non-hate / 0 / others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：主张弱势者可以反击；以眼还眼未指定身体伤害，不据关键词推成4级。


前值（历史保留）：attack_severity=待定

### #3372

> 女权怼了女拳？这事儿女拳知道吗？?

non-hate / 0 / Sexism

字段来源：hate=rule_mapping_from_human_score；group=ai_note；attack_severity=user_discussion

AI依据：用户认定仅谈梗，没有具体攻击命题，定0级，与#6074一致。当前hate映射为non-hate；旧hate是AI旧判断，未覆盖任何旧人工hate裁决。

- 用户 attack_severity：0；原答：0 级：谈梗，没有具体攻击命题

前值（历史保留）：attack_severity=待定；hate=hate

### #3448

> 把自己小号暴露了

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：指出小号暴露，只评价行为。


前值（历史保留）：attack_severity=待定

### #3456

> 不是你们先来碰瓷的？????

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：反问对方先碰瓷，只批评具体行为。


前值（历史保留）：attack_severity=待定

### #3463

> 所谓“混血狗杂更好看”的论调，就是典型文化极度自卑的奴才心理。混杂后代智商低下，面貌丑陋所占比例更大。（黄白乱杂交后代依然是白人，黑皮能够同化全世界所有人类！）而鼓吹混血优越论调的，不是汉奸，就是文化自卑到骨子里的奴才，总觉得外国月亮比较圆。相同人种混血的问题不大，历史上入侵中国的外族几乎全都是黄种人，这就是华夏文明能够延续几千年的最根本，最直接原因！而四大文明古国其它三个，早就换了无数人种。南非、马达加斯加、南北美洲的广大地区，原本都是黄种人的地盘。现在百分之九十都被白黑两种生物占领。欧洲圣母鼓吹所谓“开放包容，接受移民难民”，结果现在变得又黑又绿，动荡不安，早已无回天之力！中国人再不警惕起来，下一个就是中国！

hate / 3 / Racism、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：汉奸、移民难民及拥护混血的政治立场人群有独立身份评价，others不只来自个人辱骂。


前值（历史保留）：group=Racism、others

### #3502

> 异性恋一样有骗婚，一样有仇女，直男的大男子主义盛行更导致女性地位下降??

non-hate / 0 / Sexism、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：指出异性恋也有骗婚、仇女及大男子主义问题，是针对做法的反驳，不是所有成员的本性概括。


前值（历史保留）：attack_severity=待定

### #3517

> 弱势与强势都是相对而言的，在美国社会由于白人站了主体，它们属于少数民族，所以只能说相对来说它们属于弱势群体。不过它们只是相对于白人来说属于弱势而已。在美国对比其他族裔它们还是强势的，因为人口。亚裔中东裔对于它们来说它们就是强势而亚裔与中东裔就是弱势群体。

non-hate / 0 / Racism

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：通篇比较美国族群相对处境，没有认可身份高低；重复它们不足以单独证明动物化意图。


前值（历史保留）：attack_severity=待定

### #3519

> 和女生相关违法问题不是女生问题是TM谁的问题？主语都有问题还发个鸡毛声，发什么声？÷生？你一个男，怎么还性别歧视？管你你是什么玩意儿

hate / 2 / Sexism、others

字段来源：hate=rule_mapping_from_human_score；group=ai_note；attack_severity=user_discussion

AI依据：用户将÷生及其余贬称按明确个人侮辱定2级，修正AI最初以动物词自动取3的建议。此例不支持仅凭畜生谐写就断定达到3级；区别于已确认的具体身份动物化案例，不能外推所有动物骂词一律同档。

- 用户 attack_severity：2；原答：2 级：明确个人侮辱

前值（历史保留）：attack_severity=待定

### #3529

> 哈哈估计自身条件非常烂只能靠出卖同性获得幕刃的关注了

hate / 2 / Sexism、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：贬评具体人的条件并使用幕刃辱女，构成明确个人及性别污名。


前值（历史保留）：attack_severity=待定

### #3582

> 泼妇男，你贴出我拉黑你？你怎么不贴出你持续不断满篇脏话的撒泼言语呢？嗯？那帖子还能正常回吗？嗯？下作的剑人，哈哈哈[表情]

hate / 2 / Sexism、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：泼妇男、下作的剑人直接作性别化个人侮辱，没有另外的身份剥夺或身体伤害主张。


前值（历史保留）：attack_severity=待定

### #3585

> 这种东西都能照做的不知道是没脊椎还是没脑子，不怕一万就怕万一，万一老婆是个仙女....

hate / 2 / Sexism、others

字段来源：hate=rule_mapping_from_human_score；group=ai_note；attack_severity=user_discussion

AI依据：用户认为没脊椎没脑子已达到明确个人侮辱，取2级；区别于#309与#2517的轻度人格或能力贬评1级。

- 用户 attack_severity：2；原答：2 级：明确个人侮辱

前值（历史保留）：attack_severity=待定

### #3602

> 😆哈哈哈哈答主为啥不回你呢

non-hate / 0 / []

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：笑声与追问为何不回，没有足够具体的能力人格贬损命题。


前值（历史保留）：attack_severity=待定

### #3742

> 网上喷黑喷同胞的成万上亿，现实中敢和黑人硬碰硬的寥寥无几，看来还是上网喷同胞安全又不犯法😆

hate / 1 / Racism、others

字段来源：hate=rule_mapping_from_human_score；group=user_discussion；attack_severity=user_discussion

AI依据：用户保留Racism、others，并明确同一受话者既是黑人又是个人辱骂对象才合并；本句受话者不是黑人，个人攻击独立。只复核group，原人工1级保留。

- 用户 group：Racism、others；原答：保留 此处的受话者不是黑人，因此需要独立other，当受话者是黑人且个人时，“种族＋个人优先仅种族”才适用

前值（历史保留）：group=Racism、others

### #4392

> debuff叠满了，本人西安上学维吾尔族极左普信男粉红爱国蛆😄

hate / 1 / Racism、Sexism、others

字段来源：hate=rule_mapping_from_human_score；group=user_discussion；attack_severity=user_discussion

AI依据：用户保留Racism、Sexism、others，政治身份独立贡献others。只重开group，原1级不变。

- 用户 group：Racism、Sexism、others；原答：保留 Racism、Sexism、others：政治身份独立

前值（历史保留）：group=Racism、Sexism、others

### #4810

> 女权背后打拳替美国佬办事的狗东西可不少，把黑人娶进来的咋不说呢

hate / 2 / Racism、Sexism、others

字段来源：hate=rule_mapping_from_human_score；group=user_discussion；attack_severity=user_discussion

AI依据：用户明确保留完整旧group，说明有独立身份依据；不再因种族与个人骂词共现去掉others。原2级不变。

- 用户 group：Racism、Sexism、others；原答：保留 Racism、Sexism、others：有独立身份依据

前值（历史保留）：group=Racism、Sexism、others

### #5734

> 还有现在连某些央视都有嘿人节目，甚至嘿人评委，看着那血盆大嘴在那一张一张的心里就难受的慌，关键评论里没一个反感他们的。只要是老外，不管来自哪里长得如何，下面都是一股脑儿得说帅。眼瞎了么！但有没有发现东南亚的黄皮老外不在此列，相反是看不起越南缅甸柬埔寨等那旮瘩的老外的，感情它们喜欢的是外种？？？

hate / 2 / Racism、Region、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：眼瞎了么针对评论者，未证明这些受话者是黑人；与受辱黑人评委等不是同一对象，按新条件保留others。


前值（历史保留）：group=Racism、Region、others

### #771

> 兄弟们我上条冲错了，牛头怪是自己人，其他三个是真默🐶，冲另外三个默孝子

hate / 3 / Racism、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：默孝子与账号辱骂未证明被辱账号本身是黑人；不得凭支持黑人猜测其种族，按同一对象条件保留others。


前值（历史保留）：group=Racism、others

### #7927

> 你个loser，自己生活失败，就天天黑人怎么怎么样，人家惹你了吗

hate / 2 / Racism、others

字段来源：hate=rule_mapping_from_ai_score；group=user_discussion；attack_severity=ai_note

AI依据：用户确认维护黑人群体与辱骂受话者是不同对象，保留Racism、others。原AI严重度2不改，也不把本次group回复计为人工评分。

- 用户 group：Racism、others；原答：Racism、others：此处个人攻击对象独立

前值（历史保留）：group=Racism、others

### #867

> 倒贴找黑老丑的我都隔应，不自重自爱，就要被唾弃。

hate / 2 / Racism、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：黑老丑把年龄等作为被评价身份，others有独立年龄依据，保留。


前值（历史保留）：group=Racism、others

### #1746

> 关键是那么丑那么穷的h人，某些人还跪舔倒贴，这就是人的问题了，我们男的又不是sb！

hate / 2 / Racism、Sexism、others

字段来源：hate=rule_mapping_from_ai_score；group=ai_note；attack_severity=ai_note

AI依据：旧AI因种族与个人同句共现去掉others过宽。跪舔倒贴所贬评的是另一些人，没有证明他们是黑人；按新条件恢复others。原始AI标签与旧合并视图均保留。


前值（历史保留）：group=Racism、Sexism

### #5423

> 专业护默开小号急的不行

hate / 2 / Racism、others

字段来源：hate=rule_mapping_from_human_score；group=user_discussion；attack_severity=user_discussion

AI依据：用户在一致性问题中明确修订group为Racism、others：护默者没有被证明是黑人，不符合个人与种族指向同一黑人受辱者的合并条件。旧人工仅Racism字段保留为被明确修订的历史，原2级保留。

- 用户 group：Racism、others；原答：改为 Racism、others：受辱者未被证明是黑人

前值（历史保留）：group=Racism

