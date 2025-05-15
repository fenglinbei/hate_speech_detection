import sys
import json

sys.path.append(".")

from prompt import *

category_mapping = {
    'Racism': 'A',
    'Region': 'B',
    'LGBTQ': 'C',
    'Sexism': 'D',
    'others': 'E',
    'non-hate': 'F'}
hate_flag_mapping = { 
    'hate': 'A',
    'non-hate': 'B'
}

def unpack_data(datas: list[dict]) -> list[dict]:
    unpack_datas: list[dict] = []
    for data in datas:
        quadruples = data["quadruples"]
        for quadruple in quadruples:
            unpack_datas.append(
                {
                    "id": data["id"],
                    "content": data["content"],
                    "target": quadruple["target"],
                    "argument": quadruple["argument"],
                    "gt_targeted_group": quadruple["targeted_group"],
                    "gt_hateful": quadruple["hateful"]
                }
            )

    return unpack_datas

def convert_to_training_format(sample):
    """转换为LLM训练格式"""
    instruction = TRAIN_PROMPT_STEP_2_V1
    input_str = instruction.format(text=sample['content'], system_prompt='', shots='', target=sample["target"], argument=sample['argument'])
    
    # 构建期望输出
    output_lines = []
    lines = []
    lines.append(hate_flag_mapping[sample['gt_hateful']])
    lines.append(category_mapping[sample['gt_targeted_group'].split(", ")[0]])
    output_lines.append(" | ".join(lines))
    
    return {"messages": [{"role": "system", "content": TRAIN_PROMPT_STEP_2_SYSTEM_V1}, {"role": "user", "content": input_str}, {"role": "assistant", "content": "\n".join(output_lines)}]}

def convert_to_train_json(raw_data_path: str, save_path: str):
    # 加载原始数据
    with open(raw_data_path, encoding="utf-8") as f:
        samples = json.load(f)

    samples = unpack_data(samples)
    
    print(f"原始数据共{len(samples)}条")
    
    # 处理所有样本
    processed_data = [convert_to_training_format(s) for s in samples]
    
    # 保存为JSONL格式
    with open(save_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert_to_train_json("/mnt/i/project/hateSpeechDetection/data/train_data.json", "./two_step/step2/fine_tune/dataset/4000_train_data.jsonl")
