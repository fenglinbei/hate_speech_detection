from prompt import *
from tools.build_prompt import  get_shots

def build_two_step_shot_prompt(
        shots: list[dict],
        prompt_template: str,
        shot_prompt_template: str) -> str:
    """
    构建prompt

    shots: 随机挑选的测试样本
    """
    examples = []
    for shot in shots:
        # 生成答案部分
        answer_lines = []
        for q in shot["quadruples"]:
            target = (q.get("target", "") or "NULL").strip()
            argument = (q.get("argument", "") or "NULL").strip()
            answer_lines.append(f"{target} | {argument}")
        
        # 构建单个示例
        examples.append(
            shot_prompt_template.format(
                text=shot["content"],
                answer="\n".join(answer_lines)
            )
        )
    
    # 组合所有示例并生成包含示例的prompt模板
    return prompt_template.format(
        text = "{text}",
        shot_prompt="\n\n".join(examples)
    )
