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
