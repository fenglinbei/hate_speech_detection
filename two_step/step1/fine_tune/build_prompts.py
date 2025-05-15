import sys
import json

sys.path.append(".")

from prompt import *

def convert_to_training_format(sample):
    """转换为LLM训练格式"""
    instruction = TRAIN_PROMPT_STEP_1_V1
    input_str = instruction.format(text=sample['content'], system_prompt='', shots='')
    
    # 构建期望输出
    output_lines = []

    quad = sample.get("quadruples", [])[0] if sample.get("quadruples", []) else None

    # line = []
    # line.append(quad['target'] if quad['target'] else "NULL")
    # line.append(quad['argument'] if quad['argument'] else "NULL")
    # output_lines.append(" | ".join(line))

    for quad in sample.get("quadruples", []):
        line = []
        line.append(quad['target'] if quad['target'] else "NULL")
        line.append(quad['argument'] if quad['argument'] else "NULL")
        output_lines.append(" | ".join(line))
    
    return {"messages": [{"role": "system", "content": TRAIN_PROMPT_STEP_1_SYSTEM_V1}, {"role": "user", "content": input_str}, {"role": "assistant", "content": "\n".join(output_lines)}]}

def convert_to_train_json(raw_data_path: str, save_path: str):
    # 加载原始数据
    with open(raw_data_path, encoding="utf-8") as f:
        samples = json.load(f)
    
    print(f"原始数据共{len(samples)}条")
    
    # 处理所有样本
    processed_data = [convert_to_training_format(s) for s in samples]
    
    # 保存为JSONL格式
    with open(save_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert_to_train_json("./data/temp_train_data.json", "./two_step/step1/fine_tune/dataset/3600_train_data.jsonl")
