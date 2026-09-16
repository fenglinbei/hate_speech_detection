# 裁决规则与偏好表

当前收录 84 个已裁决案例、24 条规则／归纳。明确字段记录：hate 36、group 64、严重度 61。

“已确认规则”有明确的规则级授权；“个案归纳”是 AI 对已确认案例的总结，遇到新边界仍需复核。表中的案例值只展示用户实际裁定的字段，未裁定字段留空。

每轮先检索相近案例，再比较实际对象、攻击命题、作者立场、可见上下文和当时政策。疑似冲突会提示前后案例与具体字段；在用户说明前保留两边记录，不自动覆盖，也不把个案变成通则。

[完整 JSON](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/preferences/snapshots/cf158ae3d22aadf1bff45ff5398c3de415c16fc02963f18d0fdc7918f3aa5ed1/preferences.json) · [案例 CSV](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/preferences/snapshots/cf158ae3d22aadf1bff45ff5398c3de415c16fc02963f18d0fdc7918f3aa5ed1/cases.csv) · [维护说明](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/preferences/README.md)

## 规则速查

| 编号／情形 | 当前口径 | 依据 | 适用边界 |
| --- | --- | --- | --- |
| P01 · 严重度与 hate 的当前映射<br>已确认规则 | 暂按 0→non-hate，1–4→hate；严重度待定时当前 hate 也待定。 | [规则确认](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/policies/severity-hate-default-mapping-v1.json)；[#4392](#case-demo-4392)：hate=hate | 映射产生的 hate 是规则派生值；旧 hate 保留原政策和来源。接受映射不是逐条人工确认。 |
| P02 · 个人辱骂与 others<br>已确认规则 | 已成立的个人辱骂通常计入 others；与种族攻击指向同一黑人受辱者时，个人辱骂这一项按新优先级 P24 处理。 | [规则确认](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/policies/personal-insult-others-v1.json)；[#1517](#case-demo-1517)：group=LGBTQ、others；[#6989](#case-demo-6989)：group=[]；[#7959](#case-demo-7959)：group=others | 先确认具体个人辱骂；行为批评、单纯提人或纯群体攻击不自动加 others。旧增补自分块 03 起生效，种族／个人合并优先级随后由 P24 修订。 |
| P03 · 严重度量表及待定<br>已确认规则 | 使用 0–4 有序量表；按作者实际认可的攻击命题取最高有据等级。确实不能判时记 null。 | [规则确认](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/policies/attack-severity-v1.json)；[#309](#case-demo-309)：attack_severity=1；[#2750](#case-demo-2750)：attack_severity=3；[#2800](#case-demo-2800)：attack_severity=3 | 不是出现某个词就机械升级，也不是对标注者冒犯感打分。量表原文的独立 hate 条款已由 P01 的临时映射覆盖。 |
| P04 · 缺失上下文但文字无攻击<br>个案归纳 | 仅有上下文不全，不足以补出歧视或攻击；可见句子没有贬损命题时可判 0／non-hate。 | [#43](#case-demo-43)：attack_severity=0，group=Region；[#4744](#case-demo-4744)：attack_severity=0，group=[]；[#3194](#case-demo-3194)：hate=non-hate，group=[]；[#1160](#case-query-1160)：attack_severity=0，group=[] | 区别于已出现攻击而关键所指或程度无法确定；后者可保留待定。不能以缺失上下文替已明确辱称编造无害解释。 |
| P05 · 行为批评与人格贬损<br>个案归纳 | 区分“这项行为／说法有问题”与直接贬评人的人格或能力；前者可为 0，后者即使轻也可能为 1。 | [#6631](#case-demo-6631)：attack_severity=0，group=Racism、Region；[#6989](#case-demo-6989)：attack_severity=0，group=[]；[#309](#case-demo-309)：attack_severity=1；[#5919](#case-demo-5919)：attack_severity=1；[#1160](#case-query-1160)：attack_severity=0；[#4189](#case-query-4189)：attack_severity=0，group=Sexism；[#2075](#case-demo-2075)：attack_severity=0，group=LGBTQ、Sexism；[#2517](#case-demo-2517)：attack_severity=1，group=others | “素质”“这道理都不懂”等词不独立定性。#6989 只评行为，#2075 只批评论点而为0；#309 轻度人格贬评、#2517 智力不足的个人能力贬评均为1。须看句内实际指向。 |
| P06 · 身份只作地名、背景或类比<br>个案归纳 | 身份类别需实际进入本句的评价／讨论单位；地点背景或行为选择类比不自动增类。 | [#1456](#case-demo-1456)：hate=non-hate，group=[]；[#1224](#case-demo-1224)：group=Sexism；[#4810](#case-demo-4810)：group=Racism、Sexism、others；[#7959](#case-demo-7959)：group=others；[#2011](#case-query-2011)：group=Racism | #1224 的学历职业类比不另加 others；#4810 美国佬按政治代理背景处理。不能仅因有国家、性别词便扩大 group。 |
| P07 · 抽象政治、资本与制度<br>个案归纳 | 仅谈制度、资本或抽象政治机制，未实际评价具体人群／个人时可为 0／[]。 | [#2598](#case-demo-2598)：hate=non-hate，group=[]；[#2701](#case-demo-2701)：hate=non-hate，group=[]；[#5865](#case-demo-5865)：attack_severity=0，group=[] | #2701 犹太财阀、#5865 犹资在该句按抽象对象处理；并不意味着任何涉及犹太人的指控均排除 Racism。 |
| P08 · 中性身份讨论与普通提问<br>个案归纳 | 性取向、性别、地域等可以是中性讨论对象；没有攻击命题时取 0，不因保留 group 就判 hate。 | [#1180](#case-demo-1180)：hate=non-hate，group=LGBTQ；[#159](#case-demo-159)：hate=non-hate，group=Region；[#1859](#case-demo-1859)：hate=non-hate，group=Sexism；[#5365](#case-demo-5365)：attack_severity=0，group=Sexism、others；[#6074](#case-demo-6074)：attack_severity=0；[#857](#case-demo-857)：attack_severity=0，group=LGBTQ；[#5900](#case-demo-5900)：attack_severity=0；[#6263](#case-demo-6263)：attack_severity=0；[#996](#case-demo-996)：attack_severity=0；[#1292](#case-query-1292)：attack_severity=0，group=Racism；[#4137](#case-query-4137)：attack_severity=0 | 反过来 non-hate 也不要求 group=[]；是否保留身份类别另按对象范围判断。 |
| P09 · 报道、转述及反驳偏见<br>个案归纳 | 判断作者对转述内容的立场；客观报告、罗列并反感偏见、整体反驳偏见可为 0。 | [#5113](#case-demo-5113)：attack_severity=0，group=LGBTQ、Sexism、others；[#7232](#case-demo-7232)：attack_severity=0；[#7545](#case-demo-7545)：attack_severity=0；[#3882](#case-demo-3882)：attack_severity=0；[#986](#case-demo-986)：attack_severity=2；[#996](#case-demo-996)：attack_severity=0；[#4080](#case-query-4080)：attack_severity=0 | 报告他人骂法后若又认可负面概括，仍计攻击（#986＝2）；不能抽出被反驳的偏见充作作者立场。反歧视表述中的独立个人辱骂也需另计。 |
| P10 · 轻度讽刺也可构成攻击<br>个案归纳 | 确认有轻蔑、挖苦或轻度贬义后，可进入轻度攻击；未要求出现粗口。 | [#100](#case-demo-100)：hate=hate，group=Sexism；[#3281](#case-demo-3281)：hate=hate，group=others；[#3492](#case-demo-3492)：hate=hate，group=Sexism、others；[#6469](#case-demo-6469)：attack_severity=1；[#7189](#case-demo-7189)：attack_severity=1，group=LGBTQ、Sexism、others；[#7910](#case-demo-7910)：attack_severity=1；[#1160](#case-query-1160)：attack_severity=0；[#1660](#case-demo-1660)：attack_severity=1；[#2759](#case-demo-2759)：attack_severity=1；[#3212](#case-demo-3212)：attack_severity=1 | 表达强硬、表情、笑话格式本身不足以确定攻击；须能指出实际挖苦命题。#3281／#3492 只列已确认 hate/group，不把提案中的示例分数算成人工评分。 |
| P11 · 隐语与缩写按本句解释<br>个案归纳 | 已有文本用法支持辱称时，不凭未经支持的其他释义把攻击消解；仍按本句证据确定所指。 | [#1158](#case-demo-1158)：hate=hate，group=Sexism；[#1509](#case-demo-1509)：hate=hate，group=Sexism；[#1729](#case-demo-1729)：hate=hate，group=Sexism；[#1815](#case-demo-1815)：hate=hate，group=Sexism；[#1915](#case-demo-1915)：hate=hate，group=Racism；[#5998](#case-demo-5998)：attack_severity=1，group=Sexism、others；[#7959](#case-demo-7959)：group=others；[#857](#case-demo-857)：group=LGBTQ；[#2297](#case-query-2297)：group=LGBTQ；[#4189](#case-query-4189)：attack_severity=0，group=Sexism；[#61](#case-query-61)：attack_severity=2，group=Sexism | 这是个案取义倾向，不是“无法证明无害就一律仇恨”的通则。#5998 原先仅确认个人能力讥讽，后经用户确认两处阿娜采用辱女义，并明确将 group 修订为 Sexism、others，保留 1 级；#7959 接盘侠仍缺少婚恋性别依据。 |
| P12 · 没有足够依据时不增种族类别<br>个案归纳 | 地域、国籍、基因或不明暗指，并不自动成立独立种族判断；按实际所指决定是否增 Racism。 | [#2298](#case-demo-2298)：group=Region、others；[#924](#case-demo-924)：attack_severity=2，group=Region、Sexism；[#5998](#case-demo-5998)：group=Sexism、others；[#61](#case-query-61)：group=Sexism | #924 明确裁定一般污名 2 / Region、Sexism；若新句直接提出种族层级或本性低等，需重新比较命题。 |
| P13 · 句内隐含性别身份<br>个案归纳 | 性别不必总以“男／女”明写；句内含义能确定评价对象时可计 Sexism。 | [#3652](#case-demo-3652)：attack_severity=1，group=Racism、Sexism；[#3660](#case-demo-3660)：attack_severity=2，group=Racism、Sexism；[#7959](#case-demo-7959)：group=others；[#3898](#case-query-3898)：group=Racism、Sexism；[#61](#case-query-61)：group=Sexism | 不把词典常见义或缺失图片当成句内证据。仅有婚恋联想的 #7959 不增 Sexism。 |
| P14 · 1 级：轻度人格贬评、优越感<br>个案归纳 | 轻度人格／能力贬评、挖苦或轻度优越感可以定 1；明确指向人格也不自动升级到 2。 | [#309](#case-demo-309)：attack_severity=1；[#3742](#case-demo-3742)：attack_severity=1，group=Racism、others；[#4392](#case-demo-4392)：attack_severity=1，group=Racism、Sexism、others；[#5919](#case-demo-5919)：attack_severity=1；[#7910](#case-demo-7910)：attack_severity=1；[#877](#case-demo-877)：attack_severity=1；[#7244](#case-query-7244)：attack_severity=1；[#7646](#case-query-7646)：attack_severity=1；[#2517](#case-demo-2517)：attack_severity=1，group=others | #877 优势地位、#2517 智力不足的人在各自语境均定1；#3585 没脊椎没脑子定2。轻度人格/能力评价与明确侮辱须比较具体用法，不能仅靠类别词。 |
| P15 · 2 级：明确侮辱、一般污名与疾病类比<br>个案归纳 | 明确个人侮辱、疾病污名、一般身份负面概括、借归谬实际表达污名可定2；仍须按具体表达校准。 | [#5416](#case-demo-5416)：attack_severity=2，group=LGBTQ、others；[#7464](#case-demo-7464)：attack_severity=2；[#924](#case-demo-924)：attack_severity=2，group=Region、Sexism；[#5615](#case-demo-5615)：attack_severity=2；[#4810](#case-demo-4810)：attack_severity=2；[#986](#case-demo-986)：attack_severity=2；[#2011](#case-query-2011)：attack_severity=2；[#5423](#case-query-5423)：attack_severity=2；[#1509](#case-demo-1509)：attack_severity=2；[#2004](#case-demo-2004)：attack_severity=2；[#2685](#case-demo-2685)：attack_severity=2；[#3519](#case-demo-3519)：attack_severity=2；[#3585](#case-demo-3585)：attack_severity=2 | “正常人”的对照或“基因”的措辞，不单独把程度升到 3；需比较是否已认定本性低等或严重排斥。 |
| P16 · 3 级：本性低等、非人化与严重辱称<br>个案归纳 | 垃圾原材料影射群体本身低等、用动物性别称谓指人等，已有 3 级裁决锚点。 | [#2750](#case-demo-2750)：attack_severity=3，group=Racism、others；[#3845](#case-demo-3845)：attack_severity=3，group=Racism；[#528](#case-demo-528)：attack_severity=3，group=Region、Sexism；[#2011](#case-query-2011)：attack_severity=2；[#2297](#case-query-2297)：attack_severity=3，group=LGBTQ；[#2685](#case-demo-2685)：attack_severity=2；[#3519](#case-demo-3519)：attack_severity=2 | #2685 的🐶、#3519 的÷生均被用户定2，说明动物字样或谐写不自动等于3级。#3845 的具体身份动物称谓、#2750 的原材料低等影射仍为3。#2011 智力/开化比较为2；上述差异按案例保存，不外推所有同词文本。 |
| P17 · 身体伤害认可与讽刺、冷漠<br>个案归纳 | 出现杀死、诛之、死亡等词不自动为 4；需判断是否真正认可身体伤害。 | [#2180](#case-demo-2180)：attack_severity=2；[#2800](#case-demo-2800)：attack_severity=3；[#5394](#case-demo-5394)：attack_severity=3；[#2773](#case-demo-2773)：hate=hate，group=Racism、Region；[#3898](#case-query-3898)：attack_severity=3 | #2180=2、#2800=3 的数值由用户确认，具体解释部分仍为 AI 归纳；#5394 明确理由是漠视而未认可伤害。#2773 只确认了 hate/group，未直接确认严重度。 |
| P18 · 多类别与个人辱骂并存<br>个案归纳 | 分别记录实际参与评价的身份类别；个人辱骂产生的 others 一般并列，但与种族攻击指向同一黑人受辱者时遵循 P24 的优先级。 | [#1517](#case-demo-1517)：group=LGBTQ、others；[#2101](#case-demo-2101)：group=LGBTQ、Sexism、others；[#2750](#case-demo-2750)：group=Racism、others；[#3492](#case-demo-3492)：group=Sexism、others；[#4810](#case-demo-4810)：group=Racism、Sexism、others；[#7189](#case-demo-7189)：group=LGBTQ、Sexism、others | 不能把整句最高严重度分配给所有被提及身份；不能因为受话者属于某群体便断定每句均有独立个人辱骂。 |
| P19 · 谈词语与实际使用辱称<br>个案归纳 | 区分纯谈网络梗／生活选择，与用词对人实施贬损；词的冒犯性和使用方式共同判断。 | [#5365](#case-demo-5365)：attack_severity=0，group=Sexism、others；[#4392](#case-demo-4392)：attack_severity=1，hate=hate，group=Racism、Sexism、others；[#528](#case-demo-528)：attack_severity=3，group=Region、Sexism；[#4080](#case-query-4080)：attack_severity=0；[#7244](#case-query-7244)：attack_severity=1；[#3212](#case-demo-3212)：attack_severity=1；[#3372](#case-demo-3372)：attack_severity=0；[#4137](#case-query-4137)：attack_severity=0 | #3212 混迹在网络被认定有轻度贬损1；#3372 与#6074只谈梗无具体攻击命题为0。#4392自称仍有实际冒犯词为1，不能一律将自称或谈梗视为无害。 #4137 在词义补充后仍明确判 0：仅描述成员构成。词条注明贬义不能替代对具体攻击命题的判断。 |
| P20 · 词义不全与严重度能否独立判断<br>个案归纳 | 不补造未知隐语或星号的内容；若其余可见表达已足以判断攻击强度，可以据此定级。严重度先前待定时保留历史hate/group。 | [#1509](#case-demo-1509)：hate=hate，group=Sexism，attack_severity=2；[#2004](#case-demo-2004)：hate=hate，group=Region、others，attack_severity=2 | #1509、#2004 经本轮明确回复均补为2级；前者老g义仍未查明，后者不得还原不存在的隐藏词。不能把未知词自动当无害，也不能用旧hate标签替代分级依据。 |
| P21 · 网络社群及平台使用者<br>个案归纳 | 若实际贬评社群成员或平台使用者，可归 others；有性别评价时另外保留 Sexism。 | [#179](#case-demo-179)：hate=hate，group=others；[#3281](#case-demo-3281)：hate=hate，group=others；[#3492](#case-demo-3492)：hate=hate，group=Sexism、others | 机构／平台名字本身不等于人群攻击；必须能定位对使用者的评价。 |
| P22 · 保留个案及来源限制<br>个案归纳 | 没有解释的个案裁决原样保留；不能从单条标签反推出未获确认的新总规则。 | [#5615](#case-demo-5615)：hate=hate，attack_severity=2，group=Racism；[#1240](#case-demo-1240)：hate=non-hate，group=[]；[#5423](#case-query-5423)：group=Racism | #5615/#5423早期只保留个案；后续合并优先级经同一对象条件补充，#5423已明确修订。不得把后来解释倒写为早期理由；显式修订与无解释冲突分开记录。 |
| P23 · 仇恨相关术语的提及<br>个案归纳 | 术语有阴谋论或冒犯背景，不足以单独确定本句正在实施攻击；仍判断可见句子的用法。 | [#1292](#case-query-1292)：group=Racism，attack_severity=0；[#4137](#case-query-4137)：attack_severity=0 | #1292 明确判 0／Racism，理由是仅提及、无明确攻击；不能据此认证阴谋真实，也不能把该个案推广为所有同类指控无害。 |
| P24 · 种族与个人辱骂：同一黑人受辱对象才合并<br>已确认规则 | 用户已补充：受话者本身是黑人，且种族与个人辱骂指向同一人时，个人辱骂这一项才并入Racism；不同受辱对象须保留独立others。 | [规则确认](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/policies/racism-personal-same-target-v1.json)；[#5615](#case-demo-5615)：group=Racism；[#5423](#case-query-5423)：group=Racism、others；[#3742](#case-demo-3742)：group=Racism、others；[#4392](#case-demo-4392)：group=Racism、Sexism、others；[#4810](#case-demo-4810)：group=Racism、Sexism、others；[#7927](#case-demo-7927)：group=Racism、others | 仅同句共现不适用。#3742、#7927个人攻击对象独立，均保留others；#5423受辱者未被证明是黑人，经明确复核改为Racism、others。#4392政治身份独立，#4810有独立身份依据，也保留others。不根据“护黑”补受话者种族，不自动外推所有族群或改变严重度。 |

## 一致性复核记录

以下判断属于 AI 复核记录；“可能差异”不代表已认定用户前后矛盾。

- **C01 · 已知政策变化**：#1240 早期采用 non-hate / []；后续当前映射可由 AI 严重度 1 得到 hate。 差异来自用户后来批准的映射，不能写成用户重新裁定 #1240。旧 group 也保留原政策身份。
- **C02 · 用户补充同一对象条件，并明确修订旧例**：用户先提出种族＋个人优先仅种族；在#3742复核中明确仅当受话者是黑人且个人时适用。#3742、#7927个人攻击对象独立；#4392政治身份独立；#4810另有身份依据，因此均保留others。 已提示#5423旧仅Racism与新条件可能冲突，用户明确修订为Racism、others，原2级不变。旧人工group与新值均保留并显式关联；#1746过宽AI合并撤回。#2750等无同一黑人受话者证据的AI复核保留others，不冒充新人工裁决。
- **C03 · 可区分的命题边界**：#924 含基因表述定 2；#2011 有智力、开化比较也明确判 hate／2／Racism；#2750 垃圾原材料影射定 3；#877 优势地位定 1。 不以基因、智商、开化或优势等单词机械升级。#2011 未给解释，因此与“本性低等”的细分边界仍属 AI 归纳；遇到更接近的句子继续对照，不自行改写旧分数。
- **C04 · 用户已区分的自我描述**：#4392 自称包含实际冒犯词被判 1；#4080 自称河南回族处于互联网最底层，被明确判为自嘲处境、未认可身份贬低（0）。 按实际用词与所表达的立场区别，不按第一人称统一归 0 或 1。#4080 的 group 仍为 AI 草稿。
- **C05 · 用户已区分的作者立场**：#986 的末句被明确判为作者认可群体污名（2）；#996 被明确判为描述社会处境（0）。此前 #7232 是罗列并反感偏见（0）。 按各句作者实际立场区分，未发现需要覆盖旧决定的冲突。仅给分数的 #986／#996 不额外计作 group 裁决。
- **C06 · 本轮校准了词语使用边界**：#3212 的混迹在网络被用户定1；#3372只谈梗定0，与#6074、#4189的无具体攻击/行为批评边界一致。 #3372 当前由严重度0映射non-hate；旧hate来自AI，未覆盖旧人工hate。差别有具体措辞与命题依据，不列为用户自相矛盾。
- **C07 · 一般动物骂词与3级的细分边界**：#2685的🐶、#3519的÷生被明确定为2级个人侮辱；#3845的身份动物称谓和#2750垃圾原材料影射仍是3。 修正AI按单个动物词直接升3的倾向；一般个人骂词与身份低等影射的区分是当前案例归纳，未获所有动物用词统一规则。遇到更接近的例句继续提示边界。
- **C08 · 已明确补齐长期待定严重度**：两例原先已有人工hate/group，严重度因未知词或星号而待定；用户现均明确补定2。 保存旧null和词义未明的记录，当前严重度与hate映射已完整；不得把新评分解释成老g释义已获查证。
- **C09 · 批评论点与能力辱骂分界**：#2075只批评论点为0；#2517明确轻度能力贬评为1，与#309同档；#3585没脊椎没脑子为2。 避免见到不懂就追加others或见到能力词就定2；按语境区分论点批评、轻度贬评、明确个人侮辱。
- **C10 · 新用法证据后的明确类别修订**：用户在命中复核中确认 #5998 两处阿娜均采用辱女义，随后明确改为 Sexism、others；保留严重度 1 及 hate。 修订依据为具体用法确认及独立 group 回复，不把定义成立自动等同句子改判。旧仅 others 裁决作为被替代事件保留；新事件见 sentence-completion-v1/discussions/user-decisions-05.json。
- **C11 · 词义补充后的句子强度已校准**：用户要求女圈单列词条，匹配语境中指女权且带贬损；随后明确 #4137 仅描述成员构成，无具体攻击命题，严重度为 0。旧 0 级来自 AI 且依赖中性女性圈子的解释。 保留数值 0 并将严重度来源改为明确人工作答；改正 AI 词义理由。LGBTQ、Sexism 仍为 AI 类别，hate 由 0 映射为 non-hate。新词条补充独立保存，不把词典贬义自动当成句子攻击。
- **明确修订**：demo:3742 的 group 已按本轮明确回复复核；旧值保留在下方字段历史，不作为未解释的裁决冲突。
- **明确修订**：demo:4392 的 group 已按本轮明确回复复核；旧值保留在下方字段历史，不作为未解释的裁决冲突。
- **明确修订**：demo:4810 的 group 已按本轮明确回复复核；旧值保留在下方字段历史，不作为未解释的裁决冲突。
- **明确修订**：query:5423 的 group 已按本轮明确回复复核；旧值保留在下方字段历史，不作为未解释的裁决冲突。
- **明确修订**：demo:5998 的 group 已按本轮明确回复复核；旧值保留在下方字段历史，不作为未解释的裁决冲突。

## 可追溯的案例索引

“未直接裁定”不等于空类别、0 级或 non-hate。历史 hate 不自动替换成当前严重度映射；例如 #1240 的政策变化见一致性记录。

<a id="case-demo-43"></a>

### #43

> 吧主是重庆的啊，你们那儿也算是重灾区了吧

hate：未直接裁定／存在多值；group：Region；attack_severity：0

- group＝Region；理由：上下文不全，未包含贬损其实证据（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-04/user-decisions-01.json)
- attack_severity＝0；理由：上下文不全，未包含贬损其实证据（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-04/discussions/user-decisions-01.json)

<a id="case-query-61"></a>

### #61

> 因为种族优势，中国人普遍都很小

hate：未直接裁定／存在多值；group：Sexism；attack_severity：2

- group＝Sexism；理由：2 / Sexism 基本可判定为生殖器污名（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-09/user-decisions-04.json)
- attack_severity＝2；理由：2 / Sexism 基本可判定为生殖器污名（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-04.json)

<a id="case-demo-100"></a>

### #100

> 水文是水文，我又不是没把国内女权的德行放出来。你也不用来扣帽子。😆

hate：hate；group：Sexism；attack_severity：未直接裁定／存在多值

- group＝Sexism；理由：”国内女权的德行“即带有贬损义，虽然程度不高，但足以纳入（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝hate；理由：”国内女权的德行“即带有贬损义，虽然程度不高，但足以纳入（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-demo-159"></a>

### #159

> 还有我们🐶“江浙人种最高贵”（看不懂这句话语气是自黑还是嘲讽还是自以为是的就要来评论对枪的直接拉黑）

hate：non-hate；group：Region；attack_severity：未直接裁定／存在多值

- group＝Region；理由：没证据表明存在地域歧视（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝non-hate；理由：没证据表明存在地域歧视（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-demo-179"></a>

### #179

> 圆圈吧出了名的鸡窝都不知道？里面都是卖的还有陪玩的，好多在里明卖。

hate：hate；group：others；attack_severity：未直接裁定／存在多值

- group＝others；理由：采纳建议：对贴吧社群成员作性污名泛化。（accepted_ai_suggestion）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝hate；理由：采纳建议：对贴吧社群成员作性污名泛化。（accepted_ai_suggestion）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-demo-309"></a>

### #309

> 跟你一样的想法，感觉这个答主故作清高，自以为是。总以为自己的眼界就是世界，十分狭隘。还来一句骂人无所谓。

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：1

- attack_severity＝1；理由：1 级：轻度人格贬评（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/pilot-01/discussions/user-decisions-01.json)

<a id="case-demo-528"></a>

### #528

> 骂棒畜是一回事，骂幕刃又是另一回事

hate：hate；group：Region、Sexism；attack_severity：3

- group＝Region、Sexism；理由：使用明确辱称，且没证据表明是无害讨论（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- hate＝hate；理由：使用明确辱称，且没证据表明是无害讨论（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- attack_severity＝3；理由：使用明确辱称，且没证据表明是无害讨论（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-05/discussions/user-decisions-01.json)

<a id="case-demo-857"></a>

### #857

> 卡菲勒是同性恋🤭

hate：未直接裁定／存在多值；group：LGBTQ；attack_severity：0

- group＝LGBTQ；理由：0 / LGBTQ：仅陈述性取向，卡菲勒作为称呼（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-08/user-decisions-01.json)
- attack_severity＝0；理由：0 / LGBTQ：仅陈述性取向，卡菲勒作为称呼（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-08/discussions/user-decisions-01.json)

<a id="case-demo-877"></a>

### #877

> 泼图的意义？以诙谐、从容的形式让同性恋们认知到我们处于优势地位的事实

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：1

- attack_severity＝1；理由：1 级：仅有轻度优越感和挖苦（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-08/discussions/user-decisions-01.json)

<a id="case-demo-924"></a>

### #924

> 中国男人超强的繁殖欲，也许我们的基因就决定了我们不团结，所以才需要高喊团结的口号，而国外大多数人泽不需要这些毒鸡汤就能看破实物的本质

hate：未直接裁定／存在多值；group：Region、Sexism；attack_severity：2

- group＝Region、Sexism；理由：2 / Region、Sexism：一般污名，不另作种族判断（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-08/user-decisions-01.json)
- attack_severity＝2；理由：2 / Region、Sexism：一般污名，不另作种族判断（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-08/discussions/user-decisions-01.json)

<a id="case-demo-986"></a>

### #986

> 浙江人在东北不被歧视吧，浙江卫视台倒是全国各地都在骂，不知道你们浙江人骂不骂，在北京浙江卫视都被骂成韩国卫视了，比韩国人都能抄

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：2

- attack_severity＝2；理由：2 级：作者也认可了群体污名（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-01.json)

<a id="case-demo-996"></a>

### #996

> 关键在国外主流国家，中国男人性吸引力都是最低的，尤其在欧美国家，不光中国人，亚裔地位都最低😅😅

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：0

- attack_severity＝0；理由：0 级：描述社会处境，没有主观攻击（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-01.json)

<a id="case-demo-1158"></a>

### #1158

> 女拳真是典中典?

hate：hate；group：Sexism；attack_severity：未直接裁定／存在多值

- group＝Sexism；理由：无论是上下文还是用词，都没有任何显然证据表明此处讨论的是”女性拳击“（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝hate；理由：无论是上下文还是用词，都没有任何显然证据表明此处讨论的是”女性拳击“（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-query-1160"></a>

### #1160

> 为什么不找外星优质男人呢，人家有太空战舰呢

hate：未直接裁定／存在多值；group：[]；attack_severity：0

- group＝[]；理由：0 / []（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-09/user-decisions-01.json)
- attack_severity＝0；理由：0 / []（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-01.json)

<a id="case-demo-1180"></a>

### #1180

> 这就是你们txl的好队友吗？

hate：non-hate；group：LGBTQ；attack_severity：未直接裁定／存在多值

- group＝LGBTQ；理由：没有明显攻击意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝non-hate；理由：没有明显攻击意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-demo-1224"></a>

### #1224

> 内卷是有底线的，你本科毕业再怎么卷，即使在家待业也不会去扫大街。那个男的接这盘，一辈子抬不起头。

hate：hate；group：Sexism；attack_severity：未直接裁定／存在多值

- group＝Sexism；理由：仅 Sexism：前半句只作行为选择的类比，不另标身份类别（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-02.json)
- hate＝hate；理由：仅 Sexism：前半句只作行为选择的类比，不另标身份类别（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-02.json)

<a id="case-demo-1240"></a>

### #1240

> 这货删帖了，不好发挥啊

hate：non-hate；group：[]；attack_severity：未直接裁定／存在多值

- group＝[]；理由：采纳建议：轻度个人贬称，不足以判 hate；无群体身份对象。（accepted_ai_suggestion）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝non-hate；理由：采纳建议：轻度个人贬称，不足以判 hate；无群体身份对象。（accepted_ai_suggestion）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-query-1292"></a>

### #1292

> 凯勒奇计划，犹太人推动

hate：未直接裁定／存在多值；group：Racism；attack_severity：0

- group＝Racism；理由：0／Racism：仅提及，没有明确攻击（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-09/user-decisions-01.json)
- attack_severity＝0；理由：0／Racism：仅提及，没有明确攻击（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-01.json)

<a id="case-demo-1456"></a>

### #1456

> 上广东待几天就看到了

hate：non-hate；group：[]；attack_severity：未直接裁定／存在多值

- group＝[]；理由：广东仅作地名（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝non-hate；理由：广东仅作地名（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-demo-1509"></a>

### #1509

> EG都不是找那种老g吗。优秀的别人看不上

hate：hate；group：Sexism；attack_severity：2

- group＝Sexism；理由：虽老g意义暂时不明，但是没有证据显示这句话是对eg的非污名化讨论，可以检索一下老g的含义（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝hate；理由：虽老g意义暂时不明，但是没有证据显示这句话是对eg的非污名化讨论，可以检索一下老g的含义（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- attack_severity＝2；理由：2 级：明确性别污名（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-01.json)

<a id="case-demo-1517"></a>

### #1517

> 是啊，我上次遇到一个恶臭直男，真的，他一直三句话不离同志是变态，最后在我把很多论据给他看的时候，他还嘴硬，这些论据都是假的宁愿自己看几千年前的小破书，也不愿看看现在的论据。眼看说不过，就发各种表情来阴阳怪气。🤣大半夜的，差点没把我笑死。

hate：hate；group：LGBTQ、others；attack_severity：未直接裁定／存在多值

- group＝LGBTQ、others；理由：提到同性恋的同时包含对个人的辱骂（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝hate；理由：提到同性恋的同时包含对个人的辱骂（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-demo-1660"></a>

### #1660

> 不是，討厭女性，不想跟女性相處，這不是歧視，你隻是gay了

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：1

- attack_severity＝1；理由：1 级：轻度调侃（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-01.json)

<a id="case-demo-1729"></a>

### #1729

> 主页整天搁那和不同的白楠yy，就算你真的那么有钱也不妨碍你lkd

hate：hate；group：Sexism；attack_severity：未直接裁定／存在多值

- group＝Sexism；理由：白楠-白男 yy-意淫 lkd-烂裤裆，用于辱骂女性私生活不检点（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝hate；理由：白楠-白男 yy-意淫 lkd-烂裤裆，用于辱骂女性私生活不检点（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-demo-1815"></a>

### #1815

> 中国尺寸是真的小

hate：hate；group：Sexism；attack_severity：未直接裁定／存在多值

- group＝Sexism；理由：没有证据表明讨论的是男性生殖器以外的事物（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝hate；理由：没有证据表明讨论的是男性生殖器以外的事物（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-demo-1859"></a>

### #1859

> 那要知道这些女人男人塑造的，不过是男人怕其他男人变成舔狗罢了

hate：non-hate；group：Sexism；attack_severity：未直接裁定／存在多值

- group＝Sexism；理由：仅为话题讨论，不包含攻击（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝non-hate；理由：仅为话题讨论，不包含攻击（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-demo-1915"></a>

### #1915

> 所谓批判作用不大的失足妇女和所谓的婚恋中介都是逆向民族主义者，自愿投向BBC的也谈不上受害者，图片来自百度图片

hate：hate；group：Racism；attack_severity：未直接裁定／存在多值

- group＝Racism；理由：BBC在此处没有任何证据表明指向英国广播公司以及其他专有名词，因此此处只能是对黑人群体的侮辱代称，直接导致仇恨+种族（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝hate；理由：BBC在此处没有任何证据表明指向英国广播公司以及其他专有名词，因此此处只能是对黑人群体的侮辱代称，直接导致仇恨+种族（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)

<a id="case-demo-2004"></a>

### #2004

> 这个吧里都是**有个女的喷我然后让我教育了自己外地的去新一线然后瞧不起我大连土著说我是18线小县城的房子不稀罕殊不知她连户口都没落自称当地人

hate：hate；group：Region、others；attack_severity：2

- group＝Region、others；理由：**并非被屏蔽，只是一种让人联想的写法，参考后面语句，没有证据表明此处不是辱骂意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- hate＝hate；理由：**并非被屏蔽，只是一种让人联想的写法，参考后面语句，没有证据表明此处不是辱骂意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-01/user-decisions-01.json)
- attack_severity＝2；理由：2 级：明确辱骂（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-01.json)

<a id="case-query-2011"></a>

### #2011

> 你说的这个例子不一样吧，毕竟不是黑人。至少，金发碧眼一般是西方白人，比黑人智商更高，就是更开化一点什么的。皮肤白色，很多中国的美女也是皮肤白色的。但是非洲，则一般智商平均水平偏低，并且黑色的皮肤也往往不受人喜欢。

hate：hate；group：Racism；attack_severity：2

- group＝Racism；理由：hate / 2 /  种族（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-09/user-decisions-01.json)
- hate＝hate；理由：hate / 2 /  种族（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-09/user-decisions-01.json)
- attack_severity＝2；理由：hate / 2 /  种族（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-01.json)

<a id="case-demo-2075"></a>

### #2075

> 原来女性的地位由男同决定?你这个思想属实搞笑，对于男同仇女确实有，但是我觉得大多数不会，还有你不会以为抵制男同就没有男同了吧，男同事基因，不是后天学习，打压男同只会有更多的男同因为社会压力骗婚，这道理都不懂。

hate：未直接裁定／存在多值；group：LGBTQ、Sexism；attack_severity：0

- attack_severity＝0；理由：0／LGBTQ、Sexism：只批评论点（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-01.json)
- group＝LGBTQ、Sexism；理由：0／LGBTQ、Sexism：只批评论点（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-01.json)

<a id="case-demo-2101"></a>

### #2101

> 你说的很对物已类聚，人已群分。相差很多的两个群体没必要产生多少联系，我觉得那些撑同的人尤其是女性，就是站着说话不腰疼。既然她们喜欢那就不可能真正的公正看待。。

hate：hate；group：LGBTQ、Sexism、others；attack_severity：未直接裁定／存在多值

- group＝LGBTQ、Sexism、others；理由：2101 仇恨 / 性别、性少数、其他（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)
- hate＝hate；理由：2101 仇恨 / 性别、性少数、其他（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)

<a id="case-demo-2180"></a>

### #2180

> 罪不至此，女人趴在我们身上我们也要可以无规则杀死

hate：hate；group：Sexism；attack_severity：2

- group＝Sexism；理由：可能是借由某件杀夫案讽刺男人也可以这么做，借机贬损女性（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)
- hate＝hate；理由：可能是借由某件杀夫案讽刺男人也可以这么做，借机贬损女性（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)
- attack_severity＝2；理由：2180 2  2800 3（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/pilot-01/discussions/user-decisions-01.json)

<a id="case-query-2297"></a>

### #2297

> 本质上根本就没有什么这个恋那个恋的，西方为了用身份政治来控制人民，以保证自己的剥削能够继续，才搞出了这些闹剧，把一群过街老鼠的怪癖放上大庭广众之下，让人们自发的结成两派，左手打右手，就没有合力去对付高高在上的资本财团，妙啊

hate：hate；group：LGBTQ；attack_severity：3

- group＝LGBTQ；理由：hate / 3 / LGBTQ（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-09/user-decisions-02.json)
- hate＝hate；理由：hate / 3 / LGBTQ（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-09/user-decisions-02.json)
- attack_severity＝3；理由：hate / 3 / LGBTQ（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-02.json)

<a id="case-demo-2298"></a>

### #2298

> 你不后悔就好，如果他确实洁身自好品学兼优问题不大。不过你既然说出来了，少不得要有人评论，比如我思想上就接受不了这种事（虽然这是种族歧视，但是我确实歧视有狐臭的人，价值观定型了改不了的，仍然庆幸自己没被公知洗脑觉得外国更好，宁可做一个种族歧视者😅）你不用管评论里说什么，没有狐臭长得好品学兼优的外国人是可以谈恋爱的，但社会到了一个关键节点，迫切需要破除迷信权威，就比如今天拜登拉稀这件事，真假不重要，让美利坚成为笑话，挖苦他在金毛面前标记领地的行为，可以破除人们思想上的枷锁。如果不是激烈的反对劣质外国人，公知的声音就会占领舆论😆你要谈个小李子或者孔雀那个颜值的外国人，评论区将铺满酸柠檬并祝贺你为国争光，

hate：hate；group：Region、others；attack_severity：未直接裁定／存在多值

- group＝Region、others；理由：是否为种族类别缺乏上下文，无法判断的话就不加入种族（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)
- hate＝hate；理由：是否为种族类别缺乏上下文，无法判断的话就不加入种族（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)

<a id="case-demo-2517"></a>

### #2517

> “应该”是一个很复杂的词，智力不足的人在讨论严肃的问题时使用该词会造成歧义，比如现在，读者就无法理解你究竟想表达“不排斥就该被惩罚”还是“不排斥对自身或许会有不好的影响”，这是不好的。

hate：未直接裁定／存在多值；group：others；attack_severity：1

- attack_severity＝1；理由：1／others：轻度能力贬评（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-01.json)
- group＝others；理由：1／others：轻度能力贬评（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-01.json)

<a id="case-demo-2598"></a>

### #2598

> 本质上是我们不识字，没胆量，才被人当了殖民地。。资本说白了就是一只蚂蚁而已，想捏死就捏死，连个p都不算。第二天早上一起来，资本可能就被没收全部财产了。

hate：non-hate；group：[]；attack_severity：未直接裁定／存在多值

- group＝[]；理由：讨论制度资本等话题不在本任务范围内，且无证据说明攻击的是具体的群体/人（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)
- hate＝non-hate；理由：讨论制度资本等话题不在本任务范围内，且无证据说明攻击的是具体的群体/人（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)

<a id="case-demo-2685"></a>

### #2685

> 所以当前突出问题也包括女汉奸的问题，这点这个🐶的贴已经说明了

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：2

- attack_severity＝2；理由：2 级：一般明确侮辱（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-02.json)

<a id="case-demo-2701"></a>

### #2701

> 犹太财阀支持的

hate：non-hate；group：[]；attack_severity：未直接裁定／存在多值

- group＝[]；理由：犹太财阀不构成群体评价单位，与上一例类似（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)
- hate＝non-hate；理由：犹太财阀不构成群体评价单位，与上一例类似（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)

<a id="case-demo-2750"></a>

### #2750

> 当你已经沦落到与乐色为伍，还把乐色举高过自己的头顶的时候，你已经连乐色都不如了。那天我去买塑料袋。有黄色的白色的透明的还有黑色的，我问老板，哪一种颜色的最便宜？老板说黑色的最便宜。我问为什么黑色的最便宜？老板答，因为黑色的是垃圾材料。

hate：hate；group：Racism、others；attack_severity：3

- group＝Racism、others；理由：出现对个人辱骂以及垃圾材料的类比让人联想到种族影射（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)
- hate＝hate；理由：出现对个人辱骂以及垃圾材料的类比让人联想到种族影射（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-02/user-decisions-01.json)
- attack_severity＝3；理由：3 级：影射群体本身低等（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/pilot-01/discussions/user-decisions-01.json)

<a id="case-demo-2759"></a>

### #2759

> 九几年的故事会就有一个故事——黑人富商，女大学生，怀孕，黑人离开，接盘侠很开心😆

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：1

- attack_severity＝1；理由：1 级：借故事轻度挖苦（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-02.json)

<a id="case-demo-2773"></a>

### #2773

> 囧南方人揍黑人白人也很多好吧 你什么时候看见南方人不敢揍了 南方人就低人一等了？

hate：hate；group：Racism、Region；attack_severity：未直接裁定／存在多值

- group＝Racism、Region；理由：整体内涵为”南方人也会揍黑人白人“并觉得此行为为正常，包含了仇恨意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-03/user-decisions-01.json)
- hate＝hate；理由：整体内涵为”南方人也会揍黑人白人“并觉得此行为为正常，包含了仇恨意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-03/user-decisions-01.json)

<a id="case-demo-2800"></a>

### #2800

> 对高调的同性恋真的无法理解，你做的明明是为世所不能容的事，为什么不能低调点，我用回你所举的例子，一个纹身的人去当兵或报考公务员，有关机构说明规定拒绝通过，然后这个人还认为这是自己的私人问题，旁人不该抱以歧视，你本来就是少数派，是异类，你低调点别人有可能还睁一只眼闭一只眼，现在还要跳出来说纹身无罪，甚至还鼓励其他人，这不是公开找骂？我个人认为，这群人的行为首先是对天地造化，阴阳平衡的亵渎，是为不敬；其二，父母赐予生命血肉，本应薪火相传，生生不息，如此看来，是为不孝；其三，中国社会面临人口断崖危机，这类群体不但不能出一份力甚至还以同性为榜样，是为不仁；其四，他们的行为有可能给自己的授业恩师或有地位有身份的朋友亲戚带来不必要的麻烦，此为陷人于不义。这样不敬不孝不仁不义的人，人人得而诛之，痛斥其罪，已经是对他们最大的宽恕。

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：3

- attack_severity＝3；理由：2180 2  2800 3（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/pilot-01/discussions/user-decisions-01.json)

<a id="case-demo-3194"></a>

### #3194

> 你的截图说的很有道理

hate：non-hate；group：[]；attack_severity：未直接裁定／存在多值

- group＝[]；理由：仅靠文字判断即可，该句无任何攻击意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-03/user-decisions-01.json)
- hate＝non-hate；理由：仅靠文字判断即可，该句无任何攻击意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-03/user-decisions-01.json)

<a id="case-demo-3212"></a>

### #3212

> 一直就有大把女人反对女拳，现实中女拳并不多，大多混迹在网络

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：1

- attack_severity＝1；理由：1 级：混迹一词带有轻度贬损（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-02.json)

<a id="case-demo-3281"></a>

### #3281

> 编乎有几个真的，不全是cjb

hate：hate；group：others；attack_severity：未直接裁定／存在多值

- group＝others；理由：整体对”编乎“（即国内知乎平台的玩笑性代称）说里面的人都是CJB（某事某物靠吹起来），整体仇恨程度不高，若用严格判定则为hate + others否则为non-hate+ others（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-03/user-decisions-01.json)
- hate＝hate；理由：整体对”编乎“（即国内知乎平台的玩笑性代称）说里面的人都是CJB（某事某物靠吹起来），整体仇恨程度不高，若用严格判定则为hate + others否则为non-hate+ others（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-03/user-decisions-01.json)

<a id="case-demo-3372"></a>

### #3372

> 女权怼了女拳？这事儿女拳知道吗？?

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：0

- attack_severity＝0；理由：0 级：谈梗，没有具体攻击命题（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-02.json)

<a id="case-demo-3492"></a>

### #3492

> 男性对女性敌意大？微博厕所白叫的吗。除了营销号有几个男的还微博说话。除非什么重大事情的时候才会去微博看看官媒发的东西

hate：hate；group：Sexism、others；attack_severity：未直接裁定／存在多值

- group＝Sexism、others；理由：借由”微博厕所“这一说法抨击其主要使用的女性群体，同时也会误伤正常使用的男女群体，符合仇恨判定，但是攻击性不重（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-03/user-decisions-01.json)
- hate＝hate；理由：借由”微博厕所“这一说法抨击其主要使用的女性群体，同时也会误伤正常使用的男女群体，符合仇恨判定，但是攻击性不重（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-03/user-decisions-01.json)

<a id="case-demo-3519"></a>

### #3519

> 和女生相关违法问题不是女生问题是TM谁的问题？主语都有问题还发个鸡毛声，发什么声？÷生？你一个男，怎么还性别歧视？管你你是什么玩意儿

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：2

- attack_severity＝2；理由：2 级：明确个人侮辱（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-03.json)

<a id="case-demo-3585"></a>

### #3585

> 这种东西都能照做的不知道是没脊椎还是没脑子，不怕一万就怕万一，万一老婆是个仙女....

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：2

- attack_severity＝2；理由：2 级：明确个人侮辱（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-03.json)

<a id="case-demo-3652"></a>

### #3652

> 姐姐你的圈子有点小啊，我就见过黑白在学校门口扭打（女的撕打，嘿人被打），女的带着哭腔质问“选我还是选她”？。。。平时不知道用啥语言沟通，这句倒是字正腔圆的京片子。

hate：未直接裁定／存在多值；group：Racism、Sexism；attack_severity：1

- group＝Racism、Sexism；理由：带有一定讥讽性质（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-04/user-decisions-01.json)
- attack_severity＝1；理由：带有一定讥讽性质（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-04/discussions/user-decisions-01.json)

<a id="case-demo-3660"></a>

### #3660

> 主要是被嘿嘿玩过的，那不是一般的思想，那得多么的。。。

hate：未直接裁定／存在多值；group：Racism、Sexism；attack_severity：2

- group＝Racism、Sexism；理由：句内隐含女性身份（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-04/user-decisions-01.json)
- attack_severity＝2；理由：句内隐含女性身份（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-04/discussions/user-decisions-01.json)

<a id="case-demo-3742"></a>

### #3742

> 网上喷黑喷同胞的成万上亿，现实中敢和黑人硬碰硬的寥寥无几，看来还是上网喷同胞安全又不犯法😆

hate：未直接裁定／存在多值；group：Racism、others；attack_severity：1

- group＝Racism、others（历史值，已由明确复核取代）；理由：轻微讥讽意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-04/user-decisions-01.json)
- attack_severity＝1；理由：轻微讥讽意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-04/discussions/user-decisions-01.json)
- group＝Racism、others；理由：保留 此处的受话者不是黑人，因此需要独立other，当受话者是黑人且个人时，“种族＋个人优先仅种族”才适用（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-04.json)

<a id="case-demo-3845"></a>

### #3845

> 短视频不少公的黑人跟中国女孩交往的人啊爆出来的新闻也很多啊

hate：未直接裁定／存在多值；group：Racism；attack_severity：3

- group＝Racism；理由：动物性别称谓用于人，已支持侮辱（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-04/user-decisions-01.json)
- attack_severity＝3；理由：动物性别称谓用于人，已支持侮辱（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-04/discussions/user-decisions-01.json)

<a id="case-demo-3882"></a>

### #3882

> 😆见义勇为：南京小张。😠嫌疑人：黑龙江张某。（在南京居然还敢见义勇为也是很勇了）

hate：未直接裁定／存在多值；group：Region；attack_severity：0

- group＝Region；理由：反歧视，不含主观攻击意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-04/user-decisions-01.json)
- attack_severity＝0；理由：反歧视，不含主观攻击意（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-04/discussions/user-decisions-01.json)

<a id="case-query-3898"></a>

### #3898

> 不是已经入狱了吗，话说回来，我心中反黑领导人是快手发展黑人祝福起家，领导一群肌肉黑人喃小弟的壮硕中国男人。当地黑社会一霸，并且每个黑人工资一个月只有一千块人民币。真正实现了心理生理双重奴役黑人喃。

hate：hate；group：Racism、Sexism；attack_severity：3

- group＝Racism、Sexism；理由：hate / 3 / Racism、Sexism（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-09/user-decisions-02.json)
- hate＝hate；理由：hate / 3 / Racism、Sexism（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-09/user-decisions-02.json)
- attack_severity＝3；理由：hate / 3 / Racism、Sexism（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-02.json)

<a id="case-query-4080"></a>

### #4080

> 我是河南的回族，简直互联网最底层🤣

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：0

- attack_severity＝0；理由：0 级：自嘲处境，没有认可身份贬低（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-02.json)

<a id="case-query-4137"></a>

### #4137

> 女圈有不少女同

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：0

- attack_severity＝0；理由：0 级：仅描述成员构成，没有具体攻击命题（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-06.json)

<a id="case-query-4189"></a>

### #4189

> 😓早就听闻不少男的假冒女拳败坏我国大部分女性形象，今日一见，果真如此

hate：未直接裁定／存在多值；group：Sexism；attack_severity：0

- group＝Sexism；理由：0／Sexism（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-09/user-decisions-03.json)
- attack_severity＝0；理由：0／Sexism（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-03.json)

<a id="case-demo-4392"></a>

### #4392

> debuff叠满了，本人西安上学维吾尔族极左普信男粉红爱国蛆😄

hate：hate；group：Racism、Sexism、others；attack_severity：1

- group＝Racism、Sexism、others（历史值，已由明确复核取代）；理由：词义本身即有冒犯意，且不是对词本身的讨论（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-04/user-decisions-01.json)
- attack_severity＝1；理由：词义本身即有冒犯意，且不是对词本身的讨论（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-04/discussions/user-decisions-01.json)
- hate＝hate；理由：改为hate（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-04/user-decisions-02.json)
- group＝Racism、Sexism、others；理由：保留 Racism、Sexism、others：政治身份独立（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-04.json)

<a id="case-demo-4744"></a>

### #4744

> 可以说，以上的地域歧视只能说是略有夸大。但是几乎全部都有事实依据。要真是空穴来风根本不可能流传开来。

hate：未直接裁定／存在多值；group：[]；attack_severity：0

- group＝[]；理由：上下文不全，但是句子本身不含贬损歧视义，若有上下文全的其他案例可改判（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-04/user-decisions-01.json)
- attack_severity＝0；理由：上下文不全，但是句子本身不含贬损歧视义，若有上下文全的其他案例可改判（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-04/discussions/user-decisions-01.json)

<a id="case-demo-4810"></a>

### #4810

> 女权背后打拳替美国佬办事的狗东西可不少，把黑人娶进来的咋不说呢

hate：hate；group：Racism、Sexism、others；attack_severity：2

- group＝Racism、Sexism、others（历史值，已由明确复核取代）；理由：美国佬为政治代理指控不参与讨论（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- hate＝hate；理由：美国佬为政治代理指控不参与讨论（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- attack_severity＝2；理由：美国佬为政治代理指控不参与讨论（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-05/discussions/user-decisions-01.json)
- group＝Racism、Sexism、others；理由：保留 Racism、Sexism、others：有独立身份依据（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-04.json)

<a id="case-demo-5113"></a>

### #5113

> 很多男的无感，我老公就是对同志无感，而且他还有一个同事一个战友是同，他们玩的都挺好的。但是他说他在部队，有个男的加他，要给他k，他不同意，然后那个男的说闭上眼睛享受就行了，结果我老公把他拉黑了。今年我不知道啥事问他，他才告诉我的，他说就是很奇怪，为啥会有男的要给他k🤔，神经病么。。。他们直男都不了解为什么同会这样，都觉得很奇怪。其实很多直男对同都是无感的

hate：non-hate；group：LGBTQ、Sexism、others；attack_severity：0

- group＝LGBTQ、Sexism、others；理由：仅为客观描述事件，主体是那个”男的“，并未直接攻击同性恋群体（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- hate＝non-hate；理由：仅为客观描述事件，主体是那个”男的“，并未直接攻击同性恋群体（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- attack_severity＝0；理由：仅为客观描述事件，主体是那个”男的“，并未直接攻击同性恋群体（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-05/discussions/user-decisions-01.json)

<a id="case-demo-5365"></a>

### #5365

> 又是应了那句老话，认真你就输了。人类真是复读机，租鸡青一了解一下？（开玩笑的）

hate：non-hate；group：Sexism、others；attack_severity：0

- group＝Sexism、others；理由：只在谈网络梗和生活选择（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- hate＝non-hate；理由：只在谈网络梗和生活选择（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- attack_severity＝0；理由：只在谈网络梗和生活选择（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-05/discussions/user-decisions-01.json)

<a id="case-demo-5394"></a>

### #5394

> 我感觉西方的教会拒绝甚至诅咒同性恋是对他们信仰的维护。我也不容我的信仰被玷污。当然，西方炼铜的，该怎么死怎么死。在我的生活中，只要txl不干扰他人并强迫接受，那可以理解。当然，不诅咒已经是最大的克制了。不祝福，不喜欢。

hate：hate；group：LGBTQ、others；attack_severity：3

- group＝LGBTQ、others；理由：只是表达漠视、未认可伤害，没到达预定程度（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- hate＝hate；理由：只是表达漠视、未认可伤害，没到达预定程度（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- attack_severity＝3；理由：只是表达漠视、未认可伤害，没到达预定程度（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-05/discussions/user-decisions-01.json)

<a id="case-demo-5416"></a>

### #5416

> 这么说的话最不会传播艾滋病的是女同，女同应该打压男同跟异性恋，让世界上只有女同！

hate：hate；group：LGBTQ、others；attack_severity：2

- group＝LGBTQ、others；理由：归谬得到”男同最会传播艾滋病“带有污名化意，且整体带有阴阳怪气（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- hate＝hate；理由：归谬得到”男同最会传播艾滋病“带有污名化意，且整体带有阴阳怪气（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- attack_severity＝2；理由：归谬得到”男同最会传播艾滋病“带有污名化意，且整体带有阴阳怪气（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-05/discussions/user-decisions-01.json)

<a id="case-query-5423"></a>

### #5423

> 专业护默开小号急的不行

hate：未直接裁定／存在多值；group：Racism、others；attack_severity：2

- group＝Racism（历史值，已由明确复核取代）；理由：2 / Racism（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-09/user-decisions-03.json)
- attack_severity＝2；理由：2 / Racism（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-03.json)
- group＝Racism、others；理由：改为 Racism、others：受辱者未被证明是黑人（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-04.json)

<a id="case-demo-5615"></a>

### #5615

> 嘿人删我贴就是**子！

hate：hate；group：Racism；attack_severity：2

- group＝Racism；理由：未提供（not_provided）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- hate＝hate；理由：未提供（not_provided）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-05/user-decisions-01.json)
- attack_severity＝2；理由：未提供（not_provided）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-05/discussions/user-decisions-01.json)

<a id="case-demo-5865"></a>

### #5865

> 本质上就是被身份政治洗脑，转移矛盾的傀儡罢了，西方犹资的老套路了，不能只看到表象，要知道幕后黑手是谁

hate：未直接裁定／存在多值；group：[]；attack_severity：0

- group＝[]；理由：0 / []：整句仍只谈抽象政治、资本或制度（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-06/user-decisions-01.json)
- attack_severity＝0；理由：0 / []：整句仍只谈抽象政治、资本或制度（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-06/discussions/user-decisions-01.json)

<a id="case-demo-5900"></a>

### #5900

> 东南亚小国，人家肤色都跟我们一样！说的不好听点，好多小国以前还被中国统治过～这能一样吗？混淆概念呢

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：0

- attack_severity＝0；理由：0 级：当前文字没有明确贬损（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-06/discussions/user-decisions-01.json)

<a id="case-demo-5919"></a>

### #5919

> 道德绑架够溜 你没有反对我们不喜欢同性恋的权利 正如你没有反对地球是圆的权利 这些必须就这样 不然你就是双标

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：1

- attack_severity＝1；理由：1 级：轻度人格贬评和强硬争辩（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-06/discussions/user-decisions-01.json)

<a id="case-demo-5998"></a>

### #5998

> 是被阿娜夺舍了吗我寻思着就阿娜这个水平应该也学不会盗号啊

hate：未直接裁定／存在多值；group：Sexism、others；attack_severity：1

- group＝others（历史值，已由明确复核取代）；理由：1 / others：只确认对具体人的能力讥讽（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-06/user-decisions-01.json)
- attack_severity＝1；理由：1 / others：只确认对具体人的能力讥讽（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-06/discussions/user-decisions-01.json)
- group＝Sexism、others；理由：改为 Sexism、others；保留 1 级及 hate（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-05.json)

<a id="case-demo-6074"></a>

### #6074

> 啊这不给他们开除女权

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：0

- attack_severity＝0；理由：0 级：当前没有具体攻击命题（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-06/discussions/user-decisions-01.json)

<a id="case-demo-6263"></a>

### #6263

> 八国洋人在天津消灭了九成天津人，天津貌似没有被毁灭啊

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：0

- attack_severity＝0；理由：0 级：只是描述或区分概念，没有认可攻击（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-06/discussions/user-decisions-01.json)

<a id="case-demo-6469"></a>

### #6469

> 批评干嘛，求仁得仁又何求！求黑得黑，与黑长存嘛

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：1

- attack_severity＝1；理由：1 级：只确认轻度挖苦（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-07/discussions/user-decisions-01.json)

<a id="case-demo-6631"></a>

### #6631

> 现实中不敢跟黑人硬碰硬，网络上还收拾不了本国人？😆

hate：未直接裁定／存在多值；group：Racism、Region；attack_severity：0

- group＝Racism、Region；理由：0 / Racism、Region：只是批评行为（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-07/user-decisions-01.json)
- attack_severity＝0；理由：0 / Racism、Region：只是批评行为（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-07/discussions/user-decisions-01.json)

<a id="case-demo-6989"></a>

### #6989

> 我工作时被河南信阳女客户的素质大跌眼镜。。无语住了。

hate：未直接裁定／存在多值；group：[]；attack_severity：0

- group＝[]；理由：0 / []：仅评价具体行为，没有人格贬损（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-07/user-decisions-01.json)
- attack_severity＝0；理由：0 / []：仅评价具体行为，没有人格贬损（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-07/discussions/user-decisions-01.json)

<a id="case-demo-7189"></a>

### #7189

> 希望你们都能花钱娶女同

hate：未直接裁定／存在多值；group：LGBTQ、Sexism、others；attack_severity：1

- group＝LGBTQ、Sexism、others；理由：1 / LGBTQ、Sexism、others：借婚配挖苦受话者（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-07/user-decisions-01.json)
- attack_severity＝1；理由：1 / LGBTQ、Sexism、others：借婚配挖苦受话者（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-07/discussions/user-decisions-01.json)

<a id="case-demo-7232"></a>

### #7232

> 魔都+帝都从人人清北复交，到著名的排外说，再到男人没男子气概，女人跪舔老外，再到小市民精打细算小家子气斤斤计较，我真的不想列举了

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：0

- attack_severity＝0；理由：0 级：罗列、反感这些偏见（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-07/discussions/user-decisions-01.json)

<a id="case-query-7244"></a>

### #7244

> 知足吧，爱丁堡不比井盖、山越猴、京巴、九头鸟、东百太君、白完、胡建、陕蛋蛋、响马听起来洋气？

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：1

- attack_severity＝1；理由：1 级：借称呼轻度调侃地域身份（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-04.json)

<a id="case-demo-7464"></a>

### #7464

> 基佬的艾正常人也不会要，还谈什么勾引，真艾只属于基佬

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：2

- attack_severity＝2；理由：2 级：明确疾病污名，尚未达到严重排斥（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-08/discussions/user-decisions-01.json)

<a id="case-demo-7545"></a>

### #7545

> 按皮肤归纳人种本来就不对，黑人里面的人种特别多，天差地别。而且黑人不聪明很多是教育的原因。但是黑人数目那么大，比例少但是数量绝对不少。没必要说哪种人比哪种人聪明，更开化什么的。我就在网上看到过一些黑人精英，而且能到中国做生意的黑人，很多时候也是非洲比较聪明厉害有行动力的。就文化我们和他们处不来，但是做生意，尊重一些还是能做到的。

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：0

- attack_severity＝0；理由：0 级：整体在反驳种族智力偏见（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-08/discussions/user-decisions-01.json)

<a id="case-query-7646"></a>

### #7646

> 我的好几个贴，都被删除了，会不会是这玩意举报的？

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：1

- attack_severity＝1；理由：1 级：轻度个人贬称（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-09/discussions/user-decisions-04.json)

<a id="case-demo-7910"></a>

### #7910

> 这不就是中国人口吧那群人吗？老大聪明了

hate：未直接裁定／存在多值；group：未直接裁定／存在多值；attack_severity：1

- attack_severity＝1；理由：1 级：这里已是轻度反讽（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/severity/batch-08/discussions/user-decisions-01.json)

<a id="case-demo-7927"></a>

### #7927

> 你个loser，自己生活失败，就天天黑人怎么怎么样，人家惹你了吗

hate：未直接裁定／存在多值；group：Racism、others；attack_severity：未直接裁定／存在多值

- group＝Racism、others；理由：Racism、others：此处个人攻击对象独立（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/discussions/user-decisions-04.json)

<a id="case-demo-7959"></a>

### #7959

> 一位优秀的接盘侠👍👍

hate：未直接裁定／存在多值；group：others；attack_severity：未直接裁定／存在多值

- group＝others；理由：仅 others：缺少婚恋或性别依据（user_verbatim）。[字段记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/batch-08/user-decisions-01.json)

