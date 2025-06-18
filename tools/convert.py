# 该脚本用于转换原始数据为LLM使用的prompt
import sys
import json
import random
from typing import Any

sys.path.append(".")

from prompt import TRAIN_PROMPT_FEW_SHOT_V1, TRAIN_PROMPT_ZERO_SHOT_V2, TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V2

# 定义目标群体映射关系（英文标签转中文）
TARGET_GROUP_MAPPING = {
    "Racism": "种族",
    "Region": "地域",
    "others": "其他",
    "LGBTQ": "LGBTQ",
    "Gender": "性别"
}

def parse_output(output_str: str):
    """解析output字符串为结构化四元组列表"""
    quadruples = []
    # 分割多个四元组
    for quad in output_str.split("[SEP]"):
        quad = quad.strip()
        for sub_quad in quad.split("[END]"):
            if not sub_quad:
                continue
            
            # 分割字段并去除空格
            try:
                # 分割四个字段并去除空格
                parts = [part.strip() for part in sub_quad.split('|')]
                if len(parts) != 4:
                    raise ValueError(f"字段数量错误，应为4个，实际得到{len(parts)}个")
                    
                target, argument, group, hate = parts
                
                # 处理特殊值
                target = None if target == "NULL" else target
                argument = None if argument == "NULL" else argument
                
                quadruples.append({
                    "target": target,
                    "argument": argument,
                    "targeted_group": group,
                    "hateful": hate
                })
            except Exception as e:
                print(f"解析错误 - {str(e)}")
                continue
            
    return quadruples

def convert_to_training_format(sample):
    """转换为LLM训练格式"""
    instruction = TRAIN_PROMPT_ZERO_SHOT_V2
    input_str = instruction.format(text=sample['content'])
    
    # 构建期望输出
    output_lines = []
    for quad in sample["quadruples"]:
        line = []
        line.append(quad['target'] if quad['target'] else "NULL")
        line.append(quad['argument'] if quad['argument'] else "NULL")
        line.append(quad['targeted_group'])
        line.append(quad['hateful'])
        output_lines.append(" | ".join(line))
    
    return {"messages": [{"role": "system", "content": TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V2}, {"role": "user", "content": input_str}, {"role": "assistant", "content": "\n".join(output_lines)}]}

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

    print(f"成功处理{len(processed_data)}条样本，保存为{save_path}")

def convert_to_json(raw_data_path: str, save_path: str, is_test_data: bool = False):
    # 将原始数据转换成更易读的类型

    # 加载原始数据
    with open(raw_data_path, encoding="utf-8") as f:
        samples = json.load(f)
    
    print(f"原始数据共{len(samples)}条")

    # 处理所有样本
    if not is_test_data:
        processed_data = [{"id": s["id"], "content": s["content"], "quadruples": parse_output(s["output"])} for s in samples]
    else:
        processed_data = [{"id": s["id"], "content": s["content"]} for s in samples]

    # 保存为JSON格式
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(processed_data, ensure_ascii=False,indent=2))

    print(f"成功处理{len(processed_data)}条样本，保存为{save_path}")

def partition(processed_data: list[dict], ratio: float = 0.1, seed: int = 23333333) -> dict[str, list[dict]]:
    """
    将数据集打乱后划分成训练集以测试集
    processed_data: 完整的数据集
    ratio: 测试集所占比率
    """
    # 创建原数据的浅拷贝以避免修改原始数据
    data = list(processed_data)
    # 打乱数据顺序
    random.seed(seed)
    random.shuffle(data)
    # 计算测试集的大小
    test_size = int(len(data) * ratio)
    # 分割数据集
    test_set = data[:test_size]
    train_set = data[test_size:]
    # 返回包含训练集和测试集的字典
    return {'train': train_set, 'test': test_set}

def partition_to_json(raw_json_data_path: str, train_set_save_path: str, test_set_save_path: str, ratio: float = 0.1):
    """
    将解析后的json数据集分割并分别保存为json文件
    
    raw_json_data_path: 解析后的json格式数据集
    train_set_save_path: 分割后的训练集的保存路径
    test_set_save_path: 分割后的测试集的保存路径
    ratio: 测试集所占比率"""

    # 读取原始JSON数据
    with open(raw_json_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # 划分数据集
    partitioned = partition(raw_data, ratio)
    
    # 保存训练集
    with open(train_set_save_path, "w", encoding="utf-8") as f:
        json.dump(partitioned["train"], f, ensure_ascii=False, indent=2)
    
    # 保存测试集
    with open(test_set_save_path, "w", encoding="utf-8") as f:
        json.dump(partitioned["test"], f, ensure_ascii=False, indent=2)

def convert_to_raw_txt(output_json_data_path: str, raw_test_data_path: str):
    # 读取原始JSON数据

    with open(raw_test_data_path, "r") as f:
        raw_test_datas: list[dict] = json.load(f)

    with open(output_json_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    results: list[dict] = raw_data["results"]
    results_dict: dict[str, str] = {}

    output_str_list: list[str] = []

    for result in results:
        outputs: list[str] = []

        if result["status"] != "success":
            results_dict[str(result['id'])] = "[END]"
            continue
        
        parsed_quadruples: list[dict] = result["parsed_quadruples"]
        for parsed_quadruple in parsed_quadruples:
            outputs.append(f"{parsed_quadruple['target']} | {parsed_quadruple['argument']} | {parsed_quadruple['targeted_group']} | {parsed_quadruple['hateful']}")
        
        results_dict[str(result['id'])] = " [SEP] ".join(outputs) + " [END]"
    
    # print(results_dict)

    for raw_test_data in raw_test_datas:
        print(str(raw_test_data['id']))

        output_str_list.append(results_dict[str(raw_test_data['id'])])
    
    with open(output_json_data_path[:-5] + ".txt", "w") as f:
        f.write("\n".join(output_str_list))

def convert_to_std_format(input_file_path: str, output_file_path: str):
    # convert data to std format

    with open(file=input_file_path, mode="r") as input_file:
        input_datas: list[dict[str, Any]] = json.load(fp=input_file)

    std_datas: list[dict] = []

    for input_data in input_datas:
        new_entry = {
            "id": input_data["id"],
            "content": input_data["content"],
            "quadruples": []
        }

        # 收集所有Qn前缀（如Q1, Q2等）
        prefixes = set()
        for key in input_data:
            if key.startswith("Q") and " Target" in key:
                prefix = key.split(" ")[0]  # 提取Qn前缀
                prefixes.add(prefix)
        
        # 处理每个Qn前缀并生成四元组
        for prefix in sorted(prefixes):
            target = input_data.get(f"{prefix} Target", "")
            argument = input_data.get(f"{prefix} Argument", "")
            group = input_data.get(f"{prefix} Group", "")
            hateful = input_data.get(f"{prefix} hateful", "")
            
            if target and argument and group and hateful:
                new_entry["quadruples"].append({
                    "target": target,
                    "argument": argument,
                    "targeted_group": group,
                    "hateful": hateful
                })

        std_datas.append(new_entry)

    with open(output_file_path, "w") as output_file:
        json.dump(std_datas, output_file, indent=2, ensure_ascii=False)

def output2triple(text):
    triple = ''
    seqs = text.split(' [SEP] ')
    for seq in seqs:
        parts = seq.split(' | ')
        triple += f'{parts[0]} | {parts[1]} | {parts[2]} [SEP] '
    return triple[:-7] + ' [END]'

def parsed_quad_to_raw_quad(parsed_quads: list[dict]):
    quads = []
    for parsed_quad in parsed_quads:
        quads.append(f"{parsed_quad['target']} | {parsed_quad['argument']} | {parsed_quad['targeted_group']} | {parsed_quad['hateful']}")
    
    return " [SEP] ".join(quads) + " [END]"

if __name__ == "__main__":
    # convert_to_json("./data/raw/test.json", "./data/test_data_parsed.json", is_test_data=True)
    # convert_to_train_json("./data/train_data_parsed.json", "train_data.jsonl")
    # convert_to_json("./data/raw/train.json", "./data/train_data_parsed.json")
    # partition_to_json("./data/train_data_parsed.json", "./data/temp_train_data.json", "./data/temp_test_data.json", 0.1)
    # convert_to_raw_txt(output_json_data_path="./data/result/output_qwen2.5-7b-instruct-ft-202504222112-4336_0_23333333.json", raw_test_data="./data/raw/test.json")

    # convert_to_std_format("data/full/raw/train.json", "data/full/std/train.json")
    convert_to_std_format("data/full/raw/test.json", "data/full/std/test.json")