QWEN2_DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

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
A: 种族歧视
B: 地域歧视
C: LGBTQ歧视
D: 性别歧视
E: 其他冒犯
F: 不构成冒犯"""

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
输出:{answer}"""

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

RAG_PROMPT_EXAMPLE_V2 = """### 句子：
{retrieve_content}
### 三元组：
{retrieve_output}"""

LEXICON_RAG_PROMPT = """关键词：{word}
类别：{category}
定义：{definition}"""
