# WP3 G3 公开形式证据：来源口径修订与候选页面登记

> 状态：`PROPOSED REVISION / NOT YET APPLIED TO SOURCE CATALOG`
>
> 记录日期：2026-08-30（Asia/Shanghai）
>
> 范围：`development-only / form-only / label-free / non-lexicon`

## 1. 文档目的与当前实现的关系

本文记录 2026-08-30 讨论确认的来源选择口径、已检索页面和需要在其他网络环境下载的页面，供主对话正式修订 G3 source catalog、同步器和审核队列时使用。

本文不是已生效的 catalog。当前实现仍是 `wp3-g3-fixed-public-priors/v1` 的七来源闭集，并拒绝 HTTPS 到 HTTP 的降级。主对话采纳本文后，应显式提升 catalog/policy 版本、重新冻结来源快照与 hash，并运行对应测试；不得只修改现有 artifact 或绕过同步/审核生命周期。

## 2. 已确认的新来源口径

### 2.1 协议、机构与可访问性分别判断

1. 不得仅因页面使用 HTTP、从 HTTPS 跳转到同站 HTTP，或缺少 HTTPS 版本而拒绝页面。
2. 政府、公共机构和高校网站的发布主体可信度不因 HTTP 自动下降；传输协议只作为获取元数据和安全风险记录。
3. 当前机器无法访问不等于来源无效。超时、403、412、502、循环跳转或代理拦截均记为 `fetch_status`，可由用户在其他网络环境下载完整页面后回传。
4. 接纳来源的核心标准是内容是否能为形式关系提供有用、可重放的证据，而不是当前机器是否能在线抓取。
5. 来源不必全部属于权威机构。内容合理、关系清楚的论文、媒体、数据集、行业页面或词条汇总可以进入候选池；优先选择发布时间较新、覆盖更丰富、能明确解释形式关系的材料。

### 2.2 证据范围不变

只提取公开材料明确支持的形式关系：

- 拼音首字母或字母缩写；
- 谐音、近音、外语音译；
- 字形、同形或正字替代；
- 已被材料明确并列的书写变体。

不得因页面给出含义就把纯语义俚语自动转成 G3 form relation。definition、流行度、类别、态度、冒犯性、ABC/R、任务标签和 gold 均不得进入正式 form reference。

例如，页面明确写出“`xswl` 是‘笑死我了’的拼音首字母缩写”，可以作为形式证据；只解释某个梗的语用含义，但没有支持 surface 与 canonical 的形式生成关系，则只能作背景或流行度材料。

### 2.3 三类来源角色

| 角色 | 用途 | 接纳方式 |
| --- | --- | --- |
| 直接形式证据 | 页面明确陈述 `surface ↔ canonical` 及缩写、谐音或书写变体关系 | 冻结原文证据后逐条人工审核 |
| 候选池 | 数据量大、较新或覆盖丰富，但发布者一般、解释质量不一或包含语义词条 | 只能产生待审候选，不得批量自动导入 |
| 流行度/分类材料 | 年度榜单、监管分类、讨论文章，只证明表达存在或某类机制存在 | 不得单独证明 canonical 映射 |

对于权威或学术来源中的清晰原文关系，可在一次人工审核后接纳。对于动态词典、商业媒体、自媒体或存在明显错误的页面，建议至少满足“第二独立来源佐证”或由审核人逐条确认并记录判断理由。

### 2.4 下载与冻结要求

用户从其他网络环境回传页面时，优先格式如下：

1. 原始 WARC 或 MHTML；
2. 浏览器“网页，全部”（HTML 加资源目录）；
3. 页面自身提供的原始 PDF；
4. 浏览器打印 PDF，作为无法保存 HTML 时的替代。

不建议只回传截图，因为截图不便进行精确文本重放和正文 hash。每份回传文件应尽量附带：请求 URL、最终跳转 URL、获取时间、页面标题、文件格式和 SHA-256。HTTP 页面原样保存，不需要人为改写为 HTTPS。

正式纳入时仍须：固定 URL 或 revision、保存本地快照、正文归一化后计算 SHA、逐条引用本地证据片段，并由人工 form review 决定。不可访问页面的手工下载是合法输入渠道，但不能成为跳过 snapshot/hash/review gate 的理由。

## 3. 当前固定七来源

以下来源已经出现在 `config/stage1/wp3_g3_public_source_catalog_v1.json` 中。此处保留登记，便于下一版 catalog 统一迁移。

| 来源 | URL | 2026-08-30 观察 | 建议角色 |
| --- | --- | --- | --- |
| 教育部：专家呼吁让网络语言留在网络 | <https://www.moe.gov.cn/jyb_xwfb/xw_ft/moe_46/moe_1055/tnull_11014.html> | 当前环境无法完整取得，待下载 | 直接形式证据 |
| 教育部：网络语言对汉语的影响和前景分析 | <https://www.moe.gov.cn/jyb_xwfb/xw_ft/moe_46/moe_1055/tnull_11604.html> | 当前环境无法完整取得，待下载 | 直接形式证据 |
| 维基百科：中国大陆网络用语列表 | <https://zh.wikipedia.org/wiki/中国大陆网络用语列表> | 可访问；必须固定具体 revision/oldid | 候选池/补充证据 |
| 维基百科：互联网用语 | <https://zh.wikipedia.org/wiki/互联网用语> | 可访问；必须固定具体 revision/oldid | 候选池/补充证据 |
| China Daily：八大网络用语 | <https://language.chinadaily.com.cn/a/202408/08/WS66b491a7a3104e74fddb91d7.html> | 可访问 | 直接形式证据 |
| China Daily：雨女无瓜/AWSL | <https://language.chinadaily.com.cn/a/201908/23/WS5d5f57a8a310cf3e3556787c.html> | 可访问 | 直接形式证据 |
| China Daily：YYDS/NBCS/HHH | <https://language.chinadaily.com.cn/a/202108/06/WS610cd28da310efa1bd667316.html> | 可访问 | 直接形式证据 |

网页访问状态只是 2026-08-30 当前环境中的观察，不是永久状态，也不是来源质量结论。

## 4. 优先新增的高价值来源

### 4.1 CHIME 中文网络梗数据集（候选池优先级最高）

- 项目：<https://github.com/yuboxie/chime>
- 原始数据：<https://raw.githubusercontent.com/yuboxie/chime/main/data/chime_full.json>
- EMNLP 2025 论文：<https://aclanthology.org/2025.emnlp-main.863/>
- 许可：仓库标注 MIT；正式使用前仍应冻结具体 commit 和 license 文件。
- 数据规模：约 1,458 条中文网络梗，其中约 133 条标为谐音、52 条标为缩写。
- 可用字段包括表达、解释、来源、例句和类型。

建议先按谐音/缩写类型筛出约 185 条候选，再由人工从原始解释中确认精确 `surface / canonical / family / evidence`。不得把 meaning、origin 以外的不相关元数据、profanity/offense 标签或自动推断的 canonical 写入正式 reference。由于部分条目缺少充分来源，CHIME 应是机器可读的候选池，不是整库自动接纳证据。

### 4.2 《浅析网络流行词中的谐音词》（2025）

- PDF：<https://www.hanspub.org/downLoad/page_download?filename=ml2025131_192914293.pdf>
- DOI：`10.12677/ml.2025.131019`
- 观察：可访问；论文标注 CC BY 4.0。
- 建议角色：直接形式证据。

明确举例包括：`表→不要`、`酱→这样`、`真不戳→真不错`、`布吉岛→不知道`、`镁铝→美女`、`康康→看看`、`小盆友→小朋友`、`阔以→可以`、`帅锅→帅哥`、`耗子尾汁→好自为之`、`蚌埠住了→绷不住了`、`猴嗨森→好开心`、`9494→就是就是`、`520→我爱你`、`555→呜呜呜`、`7456→气死我了`、`886→再见`、`额滴神呀→我的神呀`、`有木有→有没有`、`方了→慌了`、`芜湖→呜呼`、`河蟹→和谐`、`夺笋→多损`、`稀饭→喜欢`、`宣→喜欢`、`辣么→那么`。

`芭比Q`、`栓Q` 等条目需要区分“语音来源关系”和后续语义，不得把纯语义解释误写成 form canonical。

### 4.3 《网络隐语的流变、生成逻辑与问题辨析》（武汉大学，2025）

- PDF：<https://xwcbpl.whu.edu.cn/e/public/DownFile/?classid=9&id=497>
- 观察：可访问。
- 建议角色：直接形式证据及机制背景。

可复核的例子包括 `斑竹→版主`、`瘟酒屋→Windows 95`、`7456→气死我了`、`侽喷叐→男朋友`、`YYDS→永远的神`、`海龟→海归`、`海带→海待`，以及 ASCII 表情和 `orz` 等历史书写形式。只接纳文中明确支持的形式对；文章中的纯语义隐语不进入 reference。

### 4.4 国家语言资源监测与研究中心材料

1. 华中师范大学中心 2012—2024 年年度网络用语汇总：
   <https://nlp.ccnu.edu.cn/conference/15>
   
   约 130 个年度词条，适合证明表达的年度流行度和存在性。多数条目没有直接 canonical 解释，不宜单独作为形式映射证据。

2. 厦门大学中心网络字母词及书写变体专题：
   <https://ncl.xmu.edu.cn/info/1013/1867.htm>
   
   虽发布时间较早，但形式关系密集，包括 `RMB→人民币`、`LP→老婆`、`FB→腐败`、`MM→美眉`、`GG→哥哥`、`JJ→姐姐`、`TJ→太监`、`BT→变态`，以及 `MP3/mp3`、`SOHO/soho`、`E-mail/Email`、`HIP-HOP/HIPHOP`、`B2B/BtoB/BTOB` 等书写变体。对存在多义性的缩写必须保留多个候选或人工拒绝，不得自动裁义。

### 4.5 网信部门、政府及官方媒体页面

| 来源 | URL | 可复核内容/用途 |
| --- | --- | --- |
| 中国网信网：网络用语演进，不滥用不恶俗是前提 | <https://www.cac.gov.cn/2019-12/13/c_1577773266664736.htm> | `awsl→啊我死了`、`xswl→笑死我了`、`zqsg→真情实感`、`sk→生日快乐/生快`、`ssfd→瑟瑟发抖` |
| 中国网信网：2019年度十大网络用语 | <https://www.cac.gov.cn/2019-12/02/c_1576821710392379.htm> | `雨女无瓜→与你无关`；“狼人/狠人”的形式演变 |
| 中国网信网 2017 官方杂志 PDF | <https://www.cac.gov.cn/wxb_pdf/zazhiwlcb201701.pdf> | `666→溜`、`方了→慌了`、`猴赛雷/猴腮雷→好厉害` |
| 中国网信网：2026 网络生态治理分类材料 | <https://www.cac.gov.cn/2026-01/23/c_1770728781060093.htm> | 明确提到谐音梗、缩写词、拆解字、图文结合；只作机制分类，不证明具体映射 |
| 河南举报网/法治频道：近期网络表达 | <https://fazhi.henanjubao.com/2025/05-30/65981.html> | `YYDS→永远的神`、`DDDD→懂得都懂`、`丸辣→完了`、`尊嘟假嘟→真的假的`、`雨女无瓜→与你无关` |
| 上海市妇儿工委相关页面 | <https://fegw.sh.gov.cn/ywxx/20250317/4bb18e9b75a74c6d8d0111611b2d98b5.html> | `尊嘟假嘟→真的假的`、`雨女无瓜→与你无关`、栓Q/Thank you、XSWL、YYDS、2333、7456 等 |
| China Daily：网络表达是不是非梗不可（2025） | <https://cnews.chinadaily.com.cn/a/202505/14/WS6823e785a310205377032f8a.html> | `nsdd→你说得对`、`DDDD→懂的都懂` |
| 人民日报：为语言拉好文明的缰绳（2025） | <https://paper.people.com.cn/rmrb/pc/content/202508/05/content_30093242.html> | 用于存在性和社会讨论；明确形式对较少 |
| 上海市英文门户转载 China Daily | <https://english.shanghai.gov.cn/en-LearnChinese/20240918/78e8049ddc2044fab5f4134ee4bf7ea4.html> | XSWL、YYDS、ZQSG、WML、NSDD、栓Q、BDJW、芭比Q；与 China Daily 内容有重复，需去重 evidence |

## 5. 内容丰富但应作为候选池的页面

### 5.1 爱翻译网络用语大全

- 页面：<https://www.aifanyi.com/meme>
- 2026-08-30 再检索时可读取，页面自称共 341 条，并按谐音、缩写等标签分类。
- 直接候选包括 `YYDS→永远的神`、`xswl→笑死我了`、`awsl→啊我死了`、`nsdd→你说得对`、`u1s1→有一说一`、`蚌埠住了→绷不住了`、`dbq→对不起`、`kswl→磕死我了`、`gkd→搞快点`。
- 发布和校对机制不明，应视为丰富候选池。页面内容可能动态变化，必须立即保存快照和 hash；不能只记录当前 URL。

### 5.2 CTgoodjobs 2024 内地网络用语

- 页面：<https://resources.ctgoodjobs.hk/article/34721/%E6%BD%AE%E8%AA%9E2024%E2%94%82%E6%BD%AE%E8%AA%9E%E5%AD%97%E5%85%B8%E2%94%82%E5%85%A7%E5%9C%B0%E6%BD%AE%E8%AA%9E-2024%E5%A4%A7%E9%99%B8%E6%B5%81%E8%A1%8C%E7%94%A8%E8%AA%9E%E5%90%88%E9%9B%86-yyds%E9%BB%9E%E8%A7%A3-nsdd%E6%84%8F%E6%80%9D%E4%BF%82%E5%92%A9-%E7%9B%A4%E9%BB%9E20-%E5%80%8B%E5%85%A7%E5%9C%B0%E7%B6%B2%E7%B5%A1%E7%94%A8%E8%AA%9E-2024%E5%B9%B48%E6%9C%88%E6%9B%B4%E6%96%B0>
- 页面列出 20 多个表达，明确解释 YYDS、NSDD、XSWL、ZQSG、PLGG、PLJJ、BDJW、栓Q、辣鸡、炒鸡、狗带、尊嘟假嘟等形式关系。
- 商业媒体且含粤语对照，适合作候选或第二证据；不要把语义性词条一并导入。

### 5.3 其他低门槛候选来源

| 来源 | URL | 质量备注 |
| --- | --- | --- |
| LingoAce 网络流行语整理 | <https://www.lingoace.com/zh/blog/the-most-complete-dictionary-of-chinese-internet-buzzwords-in-2024-cn/> | 约 30 条；标题、发布年和页面所称年份可能不一致，仅作候选/流行度材料 |
| Dadupi 网络缩写整理（2026） | <https://www.dadupi.cn/article/2741> | 覆盖 xswl、yyds、nsdd、bdjw、u1s1、awsl、nbcs、zqsg、plmm、xjj、sk、ssfd 等；`gnps` 在同页出现互相冲突的释义，证明不可自动采信 |
| Gengtu 新网络流行语整理（2025） | <https://gengtu.net/memes/latest-chinese-internet-slang-collection-20250828/> | 有 `润←run`、`欠栓了→欠揍了`、`滚出克→滚出去` 等较新候选；作者与校对机制不清 |
| Zhargon 动态网络俚语词典 | <https://www.zhargon.com/> | 更新快但多数为语义解释，应只筛选明确的形式关系 |

## 6. 当前环境无法完整取得、待用户下载的页面

下表中的“待下载”只描述当前环境的获取状态，不代表页面不可信或不接纳。优先级根据预期形式关系密度、发布主体和与现有来源的增量排序。

### P0：优先下载

1. 教育部《高校网络媒体BBS用字用语调查》  
   <https://www.moe.gov.cn/s78/A19/s8358/moe_815/tnull_17999.html>  
   检索摘要显示内容非常丰富，包括 `GG/gg→哥哥`、`39/3x/3Q/3Ks/SNQ/THX→感谢`、`88/886/3166/C/CU/CUL/BBN/BFN/TTYL→再见`，以及“版主/班主/斑竹/版竹/版猪/板猪”等书写变体。当前环境超时。

2. 教育部《网络语言对汉语的影响和前景分析》  
   <https://www.moe.gov.cn/jyb_xwfb/xw_ft/moe_46/moe_1055/tnull_11604.html>  
   当前固定 catalog 来源；当前环境无法取得完整正文。

3. 教育部《专家呼吁让网络语言留在网络》  
   <https://www.moe.gov.cn/jyb_xwfb/xw_ft/moe_46/moe_1055/tnull_11014.html>  
   当前固定 catalog 来源；当前环境无法取得完整正文。

4. 教育部《专家在线畅谈网络语言的是是非非》  
   <https://www.moe.gov.cn/jyb_xwfb/xw_ft/moe_47/s3573/201004/t20100416_83071.html>  
   检索摘要显示含 `7456`、“偶稀饭”、“温都死”及数字谐音、字母缩略讨论。当前环境超时。

5. 教育部 2019 年度十大网络用语  
   <https://www.moe.gov.cn/jyb_xwfb/gzdt_gzdt/s5987/201912/t20191202_410477.html>  
   检索摘要明确 `雨女无瓜→与你无关`；当前环境出现循环跳转。

6. 教育部 2020 年度十大网络用语  
   <https://www.moe.gov.cn/jyb_xwfb/gzdt_gzdt/s5987/202012/t20201204_503560.html>  
   检索摘要明确 `集美→姐妹`、`52.0→我爱你`，并涉及“奥利给/给力噢”；当前环境出现循环跳转。

### P1：建议下载

7. 《社会语言学视角下网络热词缩写现象的研究》（2025）  
   <https://www.sci-open.net/index.php/JE/article/view/6956>  
   DOI：`10.63887/je.2025.1.10.4`。检索摘要含 YYDS、XSWL、NBCS、U1S1、DDDD、尊嘟假嘟、栓Q、雨女无瓜、鸡你太美、报一丝、2333、520、1314、7456、886。当前访问间歇性 502；文中也有语义再解释和可疑例子，须逐条审证。

8. 《字母型网络缩略语的分析及研究》（2025）  
   <https://m.fx361.com/news/2025/0209/25308190.html>  
   检索摘要显示可能包含 YYDS、DBQ、PYQ、GG、MM 等；当前环境返回 403。

9. 人民网传媒频道《对微博新词汇的研究》  
   <https://media.people.com.cn/n/2014/0826/c388272-25541239.html>  
   检索摘要显示形式映射较丰富；当前环境超时。

10. 全国政协网络表达相关文章（2025）  
    <https://www.cppcc.gov.cn/zxww/2025/05/27/ARTI1748319981854377.shtml>  
    检索摘要含 `丸辣→完了`、`nsdd→你说得对`、`鸡你太美→只因你太美`；当前环境 502。

### P2：可选下载或补充核验

11. 教育部英文站 2020 网络用语榜单  
    <https://en.moe.gov.cn/news/press_releases/202012/t20201229_508056.html>  
    可用于交叉核验 `集美/姐妹` 等关系；当前环境循环跳转，与中文页面可能高度重复。

12. 北京邮电大学项目《少说点黑话吧——网络流行语词典及数据统计工具》  
    <https://win.bupt.edu.cn/program.do?id=5769>  
    当前环境返回 412。项目介绍本身未必包含可提取词典；只有下载后发现正文或附件确有词条时才继续使用。

13. CTgoodjobs 页面若浏览器只能渲染、同步器不能取得完整正文，可保存 MHTML：  
    <https://resources.ctgoodjobs.hk/article/34721/%E6%BD%AE%E8%AA%9E2024%E2%94%82%E6%BD%AE%E8%AA%9E%E5%AD%97%E5%85%B8%E2%94%82%E5%85%A7%E5%9C%B0%E6%BD%AE%E8%AA%9E-2024%E5%A4%A7%E9%99%B8%E6%B5%81%E8%A1%8C%E7%94%A8%E8%AA%9E%E5%90%88%E9%9B%86-yyds%E9%BB%9E%E8%A7%A3-nsdd%E6%84%8F%E6%80%9D%E4%BF%82%E5%92%A9-%E7%9B%A4%E9%BB%9E20-%E5%80%8B%E5%85%A7%E5%9C%B0%E7%B6%B2%E7%B5%A1%E7%94%A8%E8%AA%9E-2024%E5%B9%B48%E6%9C%88%E6%9B%B4%E6%96%B0>

已存在可访问重复转载时，不必优先下载内容完全相同的镜像。例如河南网信页面若被 403 阻断，可以先使用已可访问的河南举报网/法治频道版本，但仍应比较发布主体、正文和时间后决定是否保留两个独立 evidence ID。

## 7. 已检索但不建议直接用于 form reference 的资源

| 资源 | URL | 原因 |
| --- | --- | --- |
| THUOCL | <https://github.com/thunlp/THUOCL> | 通用领域词表，没有稳定的网络变体到 canonical 关系；可用于存在性检查，不能直接生成 form mapping |
| Chinese abbreviation dataset | <https://github.com/lancopku/Chinese-abbreviation-dataset> | 面向一般中文缩略语预测，并非网络语言形式证据；如使用需另立目标和审查范围 |
| 对应论文 | <https://arxiv.org/abs/1712.06289> | 同上，适合方法参考而非本轮 G3 reference |

## 8. 建议主对话实施的正式修订

1. 将“HTTPS only”改为“HTTP/HTTPS 均可，但必须固定最终 URL、站点身份、正文快照与 hash”；HTTP 获取应留下明确安全/传输记录，而非自动失败。
2. 增加 `acquisition_mode`，至少区分 `direct_fetch`、`user_supplied_archive`、`publisher_pdf`；三者进入相同的 snapshot/hash/replay 验证。
3. 为来源增加 `source_role`：`direct_evidence`、`candidate_pool`、`prevalence_or_taxonomy`。
4. 将“当前网络无法访问”表示为可恢复的同步状态，不得因此永久排除来源；但没有任何本地快照时仍不能生成可执行的 form review item。
5. 新 catalog 先纳入已经可冻结的高价值来源；待下载页面在文件回传、hash 和正文验证完成后再加入。不要用空页面或搜索摘要代替正式证据。
6. 候选池可批量提出 relation candidates，但每个 relation 必须指向本地证据并逐条人工确认；不得把整个数据集或动态词典直接变成 reference。
7. 保留 closed allowlist 和禁止自动跟链的原则：本次扩展应通过一次显式版本修订完成，后续新增仍需再修订 catalog，避免不可审计的开放爬取。

## 9. 接收待下载文件后的检查清单

- 文件能否离线打开，正文是否完整；
- 请求 URL、最终 URL、标题和发布主体是否匹配；
- 获取时间、文件大小和 SHA-256 是否记录；
- HTML/MHTML 是否包含动态渲染后的正文；
- 是否存在登录页、错误页、验证码页或代理替换页；
- 搜索摘要中的关键形式例子是否能在完整正文逐字重放；
- 每个候选是否属于允许的四类 form relation；
- 弱来源中的映射是否有第二来源或明确的人工审核理由；
- 页面漂移或内容冲突是否产生独立 evidence/revision，而不是静默覆盖；
- 正式 reference 是否继续排除 definition、ABC/R、gold、任务标签和冒犯性判断。

