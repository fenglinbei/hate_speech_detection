import sys
import json
import random

from typing import Optional

sys.path.append(".")

from prompt import *

def get_shots(shot_datas: list[dict], shot_num: int, seed: int = 23333333) -> list[dict]:
    """
    随机采样指定数量的样本
    
    json_data_path: 解析后的json格式测试数据路径
    shot_num: 例子数量
    seed: 随机数种子，用于复现
    """

    random.seed(seed)
    return random.sample(shot_datas, k=min(shot_num, len(shot_datas)))

def build_prompt(
        shots: list[dict],
        prompt_template: str,
        shot_prompt_template: str) -> str:
    """
    构建包含示例的完整prompt

    shots: 随机挑选的测试样本
    """
    examples = []
    for shot in shots:
        # 生成答案部分
        answer_lines = []
        for q in shot["quadruples"]:
            target = (q.get("target", "") or "NULL").strip()
            argument = (q.get("argument", "") or "NULL").strip()
            category = q.get("targeted_group", "non_hate")
            hate_flag = q.get("hateful", "non_hate")
            answer_lines.append(f"{target} | {argument} | {category} | {hate_flag}")
        
        # 构建单个示例
        examples.append(
            shot_prompt_template.format(
                text=shot["content"],
                answer="\n".join(answer_lines)
            )
        )
    
    # 组合所有示例并生成最终prompt
    return prompt_template.format(
        text = "{text}",
        shot_prompt="\n\n".join(examples)
    )

def build_two_step_prompt(
        shots: list[dict],
        prompt_template: str,
        shot_prompt_template: str) -> str:
    """
    构建包含示例的完整prompt

    shots: 随机挑选的测试样本
    """
    examples = []
    for shot in shots:
        # 生成答案部分
        answer_lines = []
        for q in shot["quadruples"]:
            target = (q.get("target", "") or "NULL").strip()
            argument = (q.get("argument", "") or "NULL").strip()
            category = q.get("targeted_group", "non_hate")
            hate_flag = q.get("hateful", "non_hate")
            answer_lines.append(f"{target} | {argument} | {category} | {hate_flag}")
        
        # 构建单个示例
        examples.append(
            shot_prompt_template.format(
                text=shot["content"],
                answer="\n".join(answer_lines)
            )
        )
    
    # 组合所有示例并生成最终prompt
    return prompt_template.format(
        text = "{text}",
        shot_prompt="\n\n".join(examples)
    )


if __name__ == "__main__":
    # test_text = "大家在黑人吧宣传黑人有害的同时，应该在别的贴吧，或者任何平台，或者任何能宣传的地方也宣传反黑人，宣传黑人的危害，这样咱们反黑队伍就越来越大了，反黑的人也越来越多了。我就是看到大家宣传黑人的危害的信息加入反黑队伍的，大家一起努力宣传反黑人的信息"
    # # 获取示例样本
    # sample_shots = get_shots("./temp_test_data.json", shot_num=3)

    # # 构建完整prompt
    # prompt_template = build_few_shot_prompt(sample_shots)

    # # 实际使用时填充文本
    # final_prompt = prompt_template.format(text=test_text)

    # print(final_prompt)

    seed = 23333333
    # shot_num = 5
    # build_all_prompt("./data/temp_train_data.json", "./data/temp_test_data.json", f"./few_shot/data/few_shot_test_{seed}_{shot_num}_prompts.json", shot_num=shot_num, seed=seed)