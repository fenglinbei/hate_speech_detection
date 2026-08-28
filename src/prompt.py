QWEN2_DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
CHATGLM3_DEFAULT_SYSTEM_PROMPT = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully."
DEFAULT_SYSTEM_PTOMPT_ZH = "你是一名专业的语言学者以及仇恨言论分析人员，请你严格且正确地遵从用户给出的指令。"
DEFAULT_SYSTEM_PTOMPT_EN = "You are a professional linguist and hate speech analyst. Please strictly and correctly follow the user's instructions."

COLD_BINARY_SYSTEM_PROMPT = "你是一位中文内容安全审核专家，请严格按要求完成 COLD 仇恨言论二分类。"

COLD_BINARY_PROMPT_USER = """请判断下面文本是否属于仇恨言论二分类任务中的 hate 或 non-hate。

标签定义：
- hate：针对种族、地域、性别、性取向等群体或身份的贬低、侮辱、污名化、排斥、恶意泛化或煽动敌意。
- non-hate：不包含上述仇恨言论；包括事实描述、反歧视讨论、引用或反驳歧视、普通负面情绪但未针对群体身份攻击。

输出要求：
只输出一个标签：hate 或 non-hate。不要输出解释或标点。

文本：{text}
标签："""

COLD_BINARY_RAG_PROMPT_USER = """请判断下面文本是否属于仇恨言论二分类任务中的 hate 或 non-hate。

标签定义：
- hate：针对种族、地域、性别、性取向等群体或身份的贬低、侮辱、污名化、排斥、恶意泛化或煽动敌意。
- non-hate：不包含上述仇恨言论；包括事实描述、反歧视讨论、引用或反驳歧视、普通负面情绪但未针对群体身份攻击。

背景知识：
{lexicons}

示例：
{examples}

输出要求：
只输出一个标签：hate 或 non-hate。不要输出解释或标点。

文本：{text}
标签："""

COLD_BINARY_RAG_PROMPT_USER_V2 = """请判断下面文本是否属于仇恨言论二分类任务中的 hate 或 non-hate。

标签定义：
- hate：针对种族、地域、性别、性取向等群体或身份的贬低、侮辱、污名化、排斥、恶意泛化或煽动敌意。
- non-hate：不包含上述仇恨言论；包括事实描述、反歧视讨论、引用或反驳歧视、普通负面情绪但未针对群体身份攻击。

边界规则：
- 单纯提到身份、地域、种族、性别、性取向等词，不等于 hate。
- 反歧视讨论、引用或反驳歧视、事实描述，默认判为 non-hate。
- 只针对个人行为或个人品质的负面评价，默认判为 non-hate；只有上升到群体身份攻击时才判 hate。
- 对群体进行恶意泛化、污名化、排斥、驱逐、贬低或煽动敌意，即使表达委婉、反讽或使用谐音，也判为 hate。

背景知识：
{lexicons}

示例：
{examples}

输出要求：
只输出一个标签：hate 或 non-hate。不要输出解释或标点。

文本：{text}
标签："""

COLD_BINARY_EXPLICIT_COT_RAG_PROMPT_USER = """请判断下面文本是否属于仇恨言论二分类任务中的 hate 或 non-hate。

标签定义：
- hate：针对种族、地域、性别、性取向等群体或身份的贬低、侮辱、污名化、排斥、恶意泛化或煽动敌意。
- non-hate：不包含上述仇恨言论；包括事实描述、反歧视讨论、引用或反驳歧视、普通负面情绪但未针对群体身份攻击。

边界规则：
- 单纯提到身份、地域、种族、性别、性取向等词，不等于 hate。
- 反歧视讨论、引用或反驳歧视、事实描述，默认判为 non-hate。
- 只针对个人行为或个人品质的负面评价，默认判为 non-hate；只有上升到群体身份攻击时才判 hate。
- 对群体进行恶意泛化、污名化、排斥、驱逐、贬低或煽动敌意，即使表达委婉、反讽或使用谐音，也判为 hate。

背景知识：
{lexicons}

示例：
{examples}

请先用一到三句话进行简短分析，然后在“最终标签：”后只输出一个标签：hate 或 non-hate。

文本：{text}

分析：
最终标签："""

COLD_BINARY_RAG_PROMPT_USER_V2_WO_EXAMPLES = """请判断下面文本是否属于仇恨言论二分类任务中的 hate 或 non-hate。

标签定义：
- hate：针对种族、地域、性别、性取向等群体或身份的贬低、侮辱、污名化、排斥、恶意泛化或煽动敌意。
- non-hate：不包含上述仇恨言论；包括事实描述、反歧视讨论、引用或反驳歧视、普通负面情绪但未针对群体身份攻击。

边界规则：
- 单纯提到身份、地域、种族、性别、性取向等词，不等于 hate。
- 反歧视讨论、引用或反驳歧视、事实描述，默认判为 non-hate。
- 只针对个人行为或个人品质的负面评价，默认判为 non-hate；只有上升到群体身份攻击时才判 hate。
- 对群体进行恶意泛化、污名化、排斥、驱逐、贬低或煽动敌意，即使表达委婉、反讽或使用谐音，也判为 hate。

背景知识：
{lexicons}

输出要求：
只输出一个标签：hate 或 non-hate。不要输出解释或标点。

文本：{text}
标签："""

COLD_BINARY_RAG_PROMPT_USER_V2_WO_LEX = """请判断下面文本是否属于仇恨言论二分类任务中的 hate 或 non-hate。

标签定义：
- hate：针对种族、地域、性别、性取向等群体或身份的贬低、侮辱、污名化、排斥、恶意泛化或煽动敌意。
- non-hate：不包含上述仇恨言论；包括事实描述、反歧视讨论、引用或反驳歧视、普通负面情绪但未针对群体身份攻击。

边界规则：
- 单纯提到身份、地域、种族、性别、性取向等词，不等于 hate。
- 反歧视讨论、引用或反驳歧视、事实描述，默认判为 non-hate。
- 只针对个人行为或个人品质的负面评价，默认判为 non-hate；只有上升到群体身份攻击时才判 hate。
- 对群体进行恶意泛化、污名化、排斥、驱逐、贬低或煽动敌意，即使表达委婉、反讽或使用谐音，也判为 hate。

示例：
{examples}

输出要求：
只输出一个标签：hate 或 non-hate。不要输出解释或标点。

文本：{text}
标签："""

COLD_BINARY_PROMPT_USER_V2 = """请判断下面文本是否属于仇恨言论二分类任务中的 hate 或 non-hate。

标签定义：
- hate：针对种族、地域、性别、性取向等群体或身份的贬低、侮辱、污名化、排斥、恶意泛化或煽动敌意。
- non-hate：不包含上述仇恨言论；包括事实描述、反歧视讨论、引用或反驳歧视、普通负面情绪但未针对群体身份攻击。

边界规则：
- 单纯提到身份、地域、种族、性别、性取向等词，不等于 hate。
- 反歧视讨论、引用或反驳歧视、事实描述，默认判为 non-hate。
- 只针对个人行为或个人品质的负面评价，默认判为 non-hate；只有上升到群体身份攻击时才判 hate。
- 对群体进行恶意泛化、污名化、排斥、驱逐、贬低或煽动敌意，即使表达委婉、反讽或使用谐音，也判为 hate。

输出要求：
只输出一个标签：hate 或 non-hate。不要输出解释或标点。

文本：{text}
标签："""

COLD_BINARY_EXAMPLE_PROMPT = """文本：{retrieve_content}
标签：{retrieve_output}
"""

HATEXPLAIN_SYSTEM_PROMPT = "You are a professional hate speech analyst. Return only valid JSON that follows the requested schema."

HATEXPLAIN_RAG_PROMPT_USER = """Classify the text for the HateXplain task and identify target groups and rationales.

Labels:
- hatespeech: hateful content targeting a protected or identity-based group.
- offensive: insulting, abusive, or profane content that is not clearly hate speech.
- normal: content that is neither hate speech nor offensive.

Background lexicon:
{lexicons}

Examples:
{examples}

Rules:
- The output must be one valid JSON object and nothing else.
- Use exactly these keys: "label", "target_groups", "rationales".
- "label" must be one of: "hatespeech", "offensive", "normal".
- "target_groups" must be a list of target communities. Use [] when there is no target group.
- "rationales" must be a list of short text spans from the input that support the label. Use [] when no span is needed.
- A lexicon match is only background knowledge; do not classify as hate or offensive solely because a listed term appears.

Text:
{text}

JSON:"""

HATEXPLAIN_EXPLICIT_COT_RAG_PROMPT_USER = """Classify the text for the HateXplain task and identify target groups and rationales.

Labels:
- hatespeech: hateful content targeting a protected or identity-based group.
- offensive: insulting, abusive, or profane content that is not clearly hate speech.
- normal: content that is neither hate speech nor offensive.

Background lexicon:
{lexicons}

Examples:
{examples}

Rules:
- First provide a brief analysis in one to three sentences.
- Then write FINAL_JSON: followed by one valid JSON object and nothing else after it.
- The JSON must use exactly these keys: "label", "target_groups", "rationales".
- "label" must be one of: "hatespeech", "offensive", "normal".
- "target_groups" must be a list of target communities. Use [] when there is no target group.
- "rationales" must be a list of short text spans from the input that support the label. Use [] when no span is needed.
- A lexicon match is only background knowledge; do not classify as hate or offensive solely because a listed term appears.

Text:
{text}

Analysis:
FINAL_JSON:"""

HATEXPLAIN_RAG_PROMPT_USER_WO_EXAMPLES = """Classify the text for the HateXplain task and identify target groups and rationales.

Labels:
- hatespeech: hateful content targeting a protected or identity-based group.
- offensive: insulting, abusive, or profane content that is not clearly hate speech.
- normal: content that is neither hate speech nor offensive.

Background lexicon:
{lexicons}

Rules:
- The output must be one valid JSON object and nothing else.
- Use exactly these keys: "label", "target_groups", "rationales".
- "label" must be one of: "hatespeech", "offensive", "normal".
- "target_groups" must be a list of target communities. Use [] when there is no target group.
- "rationales" must be a list of short text spans from the input that support the label. Use [] when no span is needed.
- A lexicon match is only background knowledge; do not classify as hate or offensive solely because a listed term appears.

Text:
{text}

JSON:"""

HATEXPLAIN_RAG_PROMPT_USER_WO_LEX = """Classify the text for the HateXplain task and identify target groups and rationales.

Labels:
- hatespeech: hateful content targeting a protected or identity-based group.
- offensive: insulting, abusive, or profane content that is not clearly hate speech.
- normal: content that is neither hate speech nor offensive.

Examples:
{examples}

Rules:
- The output must be one valid JSON object and nothing else.
- Use exactly these keys: "label", "target_groups", "rationales".
- "label" must be one of: "hatespeech", "offensive", "normal".
- "target_groups" must be a list of target communities. Use [] when there is no target group.
- "rationales" must be a list of short text spans from the input that support the label. Use [] when no span is needed.

Text:
{text}

JSON:"""

HATEXPLAIN_PROMPT_USER = """Classify the text for the HateXplain task and identify target groups and rationales.

Labels:
- hatespeech: hateful content targeting a protected or identity-based group.
- offensive: insulting, abusive, or profane content that is not clearly hate speech.
- normal: content that is neither hate speech nor offensive.

Rules:
- The output must be one valid JSON object and nothing else.
- Use exactly these keys: "label", "target_groups", "rationales".
- "label" must be one of: "hatespeech", "offensive", "normal".
- "target_groups" must be a list of target communities. Use [] when there is no target group.
- "rationales" must be a list of short text spans from the input that support the label. Use [] when no span is needed.

Text:
{text}

JSON:"""

HATEXPLAIN_EXAMPLE_PROMPT = """Text:
{retrieve_content}
JSON:
{retrieve_output}
"""

TRAIN_PROMPT_FEW_SHOT_V1 = """请分析以下文本，识别其中的评论对象、论点、是否仇恨和仇恨类别。

其中仇恨类别包含以下标签：
    'Racism': 种族歧视
    'Region': 地域
    'LGBTQ': 'LGBTQ'
    'Sexism': '性别'
    'others': '其他'
    'non_hate': 不包含仇恨言论

是否仇恨包含以下标签：
    'hate': 包含仇恨言论
    'non_hate': 不包含仇恨言论

请以以下格式给出回答,若有多个评论对象, 请给出多行回答，若无评论对象或论点，请在对应位置输入NULL， 当<是否仇恨>为hate时，仇恨类别可以包含多个类别，并使用逗号分隔:

<评论对象1> | <论点1> | <仇恨类别1> | <是否仇恨1>
<评论对象2> | <论点2> | <仇恨类别2_1, 仇恨类别2_2> | <是否仇恨2>
...

例子：{shots}
---------------------------------------------------------
文本: {text}
"""

SHOT_PROMPT_V1 = """文本: {text}
{answer}"""

TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V1 = """你是一位中文内容安全审核专家"""

TRAIN_PROMPT_ZERO_SHOT_V1 = """请分析以下文本，识别其中的评论对象、论点、是否仇恨和仇恨类别。

其中仇恨类别包含以下标签：
    'Racism': 种族歧视
    'Region': 地域
    'LGBTQ': 'LGBTQ'
    'Sexism': '性别'
    'others': '其他'
    'non_hate': 不包含仇恨言论

是否仇恨包含以下标签：
    'hate': 包含仇恨言论
    'non_hate': 不包含仇恨言论

请以以下格式给出回答,若有多个评论对象, 请给出多行回答，若无评论对象或论点，请在对应位置输入NULL:

<评论对象1> | <论点1> | <仇恨类别1> | <是否仇恨1>
<评论对象2> | <论点2> | <仇恨类别2> | <是否仇恨2>
...

---------------------------------------------------------
文本: {text}
"""

TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V2 = """请分析以下文本，识别其中的评论对象、论点、仇恨类别以及是否为仇恨文本。"""

TRAIN_PROMPT_ZERO_SHOT_V2 = """ {text}"""

TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V3 = """你是一名专业的语言学者以及仇恨言论分析人员，请你严格且正确地分析我给出的句子，从中提取出一个或者多个包含主体、评论以及仇恨类别的三元组。
其中，仇恨类别请在以下选项中选择：
'Racism': 种族歧视
'Region': 地域
'LGBTQ': 'LGBTQ'
'Sexism': '性别'
'others': '其他'
'non_hate': 不包含仇恨言论

### 句子：
{text}
### 三元组："""

TRAIN_PROMPT_ZERO_SHOT_V3 = """句子：{text}
三元组："""

TRAIN_PROMPT_ZERO_SHOT_V4 = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组:
{text}"""

TRAIN_PROMPT_STEP_1_SYSTEM_V1 = """请分析以下文本，识别其中包含的所有评论对象以及论点，并以以下格式给出回答:
<评论对象1> | <论点1>
<评论对象2> | <论点2>
..."""

TRAIN_PROMPT_STEP_1_V1 = """{system_prompt}{shots}文本: {text}
输出:"""

SHOT_PROMPT_STEP_1_V1 = """文本: {text}
输出: {answer}"""

TRAIN_PROMPT_STEP_2_SYSTEM_V1 = """请根据给出的文本中的评论对象和论点，判断该评论对象和论点是否构成冒犯，并以以下方式给出回答：
<是否冒犯> | <冒犯类型>

其中，是否冒犯选项有：
A: 构成冒犯
B: 不构成冒犯

冒犯类型选项有：
A: 种族歧视
B: 地域歧视
C: LGBTQ歧视
D: 性别歧视
E: 其他冒犯
F: 不构成冒犯
"""

TRAIN_PROMPT_STEP_2_V1 = """{system_prompt}{shots}文本: {text}
评论对象: {target}
论点: {argument}
输出:"""

SHOT_PROMPT_STEP_2_V1 = """文本: {text}
评论对象: {target}
论点: {argument}
输出:{answer}
"""

TRAIN_PROMPT_SYSTEM_V5 = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个四元组"""
TRAIN_PROMPT_USER_V5 = '''请从下面的文本抽取一个或多个四元组，每一个四元组输出格式为评论对象|对象观点|仇恨群体|是否仇恨，评论对象可以为"NULL", 对象观点尽量简洁,仇恨群体只包括(LGBTQ、Region、Sexism、Racism、others、non-hate)，同一四元组可能涉及多个仇恨群体，是否仇恨标签为(hate、non-hate),多个四元组之间用[SEP]分隔,最后一个四元组后面加[END]。
            仇恨样本输入："谁喜欢小黑啊，黑不拉几的看一眼都嫌脏，也就某些连汽车排气口都不放过的男人会喜欢黑妞"
            仇恨样本输出："小黑|黑不拉几的看一眼都嫌脏|Racism|hate [SEP] 某些连汽车排气口都不放过的男人|喜欢黑妞|Sexism, Racism|hate [END]"
            非仇恨样本输入："由同性伴侣抚养长大的孩子，在学校表现比异性伴侣抚养的孩子更好，并且比异性伴侣的孩子高中毕业率高出4.8%。"
            非仇恨样本输出："由同性伴侣抚养长大的孩子|在学校表现比异性伴侣抚养的孩子更好|non-hate|non-hate [END]"
            提取出句子中包含的所有四元组:"{text}"'''

RAG_PROMPT_USER_V1 = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组:
示例：
### 句子：
{retrieve_content}
### 三元组：
{retrieve_output}
### 句子：
{text}
### 三元组：
"""

RAG_PROMPT_USER_V2 = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组:
背景知识：
{lexicons}

示例：
{examples}
### 句子：
{text}
### 三元组：
"""

RAG_PROMPT_USER_V4 = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组:
示例：
{examples}
### 句子：
{text}
### 三元组：
"""

RAG_PROMPT_USER_V3 = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组:
### 句子：
{text}
### 三元组：
"""

RAG_PROMPT_USER_V5 = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组:
示例：
{examples}
### 句子：
{text}
### 三元组：
"""

RAG_PROMPT_USER_V6 = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组。
其中，仇恨类别请在以下选项中选择：
'Racism': 种族歧视
'Region': 地域
'LGBTQ': 'LGBTQ'
'Sexism': '性别'
'others': '其他'
'non_hate': 不包含仇恨言论

背景知识：
{lexicons}

示例：
{examples}
### 句子：
{text}
### 三元组："""

STRUCTURED_EXPLICIT_COT_RAG_PROMPT_USER = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组。
其中，仇恨类别请在以下选项中选择：
'Racism': 种族歧视
'Region': 地域
'LGBTQ': 'LGBTQ'
'Sexism': '性别'
'others': '其他'
'non-hate': 不包含仇恨言论

背景知识：
{lexicons}

示例：
{examples}
### 句子：
{text}

请先用一到三句话进行简短分析，然后在“### 最终三元组：”后只输出三元组。
三元组格式为：target | argument | targeted_group，多个三元组用 [SEP] 分隔，最后用 [END] 结束。

### 分析：
### 最终三元组："""

RAG_PROMPT_EXAMPLE_V2 = """### 句子：
{retrieve_content}
### 三元组：
{retrieve_output}
"""

RAG_PROMPT_EXAMPLE_V3 = """### 句子：
{retrieve_content}
### 错例：
{retrieve_wrong_exp}
### 三元组：
{retrieve_output}
"""

LEXICON_RAG_PROMPT = """### 
关键词：{word}
类别：{category}
定义：{definition}
"""

HATEBASE_LEXICON_RAG_PROMPT = """###
Term: {word}
Category: {category}
Target type(s): {categories}
Hateful meaning: {definition}
Non-hateful meaning: {nonhateful_meaning}
Offensiveness: {average_offensiveness}
"""


STEP_ONE_RAG_PROMPT_USER_V1 = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个评论对象和论点:
示例：
{examples}
### 句子：
{text}
### 二元组：
""" 

STEP_ONE_PROMPT_USER_V1 = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个评论对象和论点:
### 句子：
{text}
### 二元组：
""" 

STEP_ONE_RAG_PROMPT_EXAMPLE_V1 = """### 句子：
{retrieve_content}
### 二元组：
{retrieve_output}"""

RAG_PROMPT_USER_WO_NSHOT = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组:
背景知识：
{lexicons}

### 句子：
{text}
### 三元组：
"""

RAG_PROMPT_USER_WO_LEX = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组:
示例：
{examples}
### 句子：
{text}
### 三元组：
"""


# Stage 1 canonical-quad-json/v1 prompts.  These constants are intentionally
# separate from the legacy triple/pipe prompts above.
STAGE1_QUAD_JSON_SYSTEM_PROMPT_V1 = """你是中文仇恨言论四元组抽取器。
你的回答必须且只能是一个符合 canonical-quad-json/v1 的 JSON 数组；不要输出分析、解释、Markdown 代码块或任何前后缀文字。
数组中每个对象必须且只能按以下顺序包含四个键：target、argument、targeted_group、hateful。
target 和 argument 必须是去除首尾空白后的字符串；字段缺失时使用 JSON null，禁止使用字符串 \"NULL\"。
targeted_group 必须是非空字符串数组，标签只能来自 Racism、Region、LGBTQ、Sexism、others、non-hate；多个标签按该固定顺序排列、不得重复，non-hate 不得与其他标签共存。
hateful 只能是 hate 或 non-hate。不得推断或改写不确定的人工标注。
没有可抽取四元组时输出空数组 []。"""


STAGE1_QUAD_JSON_RAG_PROMPT_USER_V1 = """请根据背景知识和示例，从待分析句子中抽取一个或多个仇恨言论四元组。

背景知识：
{lexicons}

示例：
{examples}

待分析句子：
{text}

只输出 canonical-quad-json/v1 JSON 数组："""


STAGE1_QUAD_JSON_SYSTEM_PROMPT_V2 = STAGE1_QUAD_JSON_SYSTEM_PROMPT_V1 + """
必须依据当前句子独立判断 targeted_group 与 hateful；群体类别或词条命中不能直接推出 hate，词义涉及某群体也不等于句子实际针对该群体。"""


STAGE1_QUAD_JSON_EVIDENCE_PROMPT_USER_V2 = """请从待分析句子抽取仇恨言论四元组。先按完整语境形成判断，再用术语理解信息核验词义。

术语理解信息（不含任务类别；只是可由 verifier 按当前语境修正或拒绝的理解参考，不是分类结论）：
{lexicons}

约束：词条命中或通常带贬义都不等于当前句为 hate；须结合实际指向及引用、自称、否定、反讽、讨论、反仇恨语境，独立判断 targeted_group 与 hateful；忽略不相关词条。

示例：
{examples}

待分析句子：
{text}

只输出 canonical-quad-json/v1 JSON 数组："""


STAGE1_QUAD_JSON_EXAMPLE_PROMPT_V1 = """待分析句子：
{retrieve_content}
canonical-quad-json/v1 JSON 数组：
{retrieve_output}
"""


# Short aliases keep configuration names compact without creating a second
# prompt definition.
QUAD_JSON_SYSTEM_PROMPT_V1 = STAGE1_QUAD_JSON_SYSTEM_PROMPT_V1
QUAD_JSON_RAG_PROMPT_USER_V1 = STAGE1_QUAD_JSON_RAG_PROMPT_USER_V1
QUAD_JSON_EXAMPLE_PROMPT_V1 = STAGE1_QUAD_JSON_EXAMPLE_PROMPT_V1
QUAD_JSON_SYSTEM_PROMPT_V2 = STAGE1_QUAD_JSON_SYSTEM_PROMPT_V2
QUAD_JSON_EVIDENCE_PROMPT_USER_V2 = STAGE1_QUAD_JSON_EVIDENCE_PROMPT_USER_V2
