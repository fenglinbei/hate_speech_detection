import pandas as pd
from transformers.models.auto.tokenization_auto import AutoTokenizer


MAX_LENGTH = 2048
tokenizer = AutoTokenizer.from_pretrained(
        "models/Qwen3-1.7B", 
        use_fast=False, 
        trust_remote_code=True
    )

def build_messages(example: pd.Series) -> list[dict]:
    if example["instruction"]:
        messages = [{'content': example["instruction"], 'role': 'system'}, {'content': example["input"], 'role': 'user'}]
    else:
        messages = [{'content': example["input"], 'role': 'user'}]
    return messages

def process_func(example):
    # 空输入处理
    if not example.get("input", "").strip() or not example.get("output", "").strip():
        return {
            "input_ids": [tokenizer.pad_token_id],
            "attention_mask": [0],
            "labels": [-100]
        }
    
    messages = build_messages(example)
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    
    # Tokenize指令部分
    instruction = tokenizer(text, add_special_tokens=False)
    inst_len = len(instruction["input_ids"])
    
    # Tokenize回答部分
    response = tokenizer(example['output'], add_special_tokens=False)
    
    # 添加EOS（如果不存在）
    if tokenizer.eos_token_id not in response["input_ids"][-1:]:
        response["input_ids"].append(tokenizer.eos_token_id)
        response["attention_mask"].append(1) if "attention_mask" in response else None
    
    resp_len = len(response["input_ids"])
    total_len = inst_len + resp_len
    
    # 长度超限处理（优先保证完整指令）
    if total_len > MAX_LENGTH:
        resp_len = MAX_LENGTH - inst_len - 1
        if resp_len > 0:
            response["input_ids"] = response["input_ids"][:resp_len] 
            response["input_ids"][-1] = tokenizer.eos_token_id  # 确保以EOS结尾
        else:
            # 仅保留指令
            response = {"input_ids": []}
            inst_len = min(inst_len, MAX_LENGTH)
            instruction["input_ids"] = instruction["input_ids"][:inst_len]
    
    # 构建最终数据
    input_ids = instruction["input_ids"] + response.get("input_ids", [])
    labels = [-100]*len(instruction["input_ids"]) + response.get("input_ids", [])
    
    return {
        "input_ids": input_ids,
        "attention_mask": [1]*len(input_ids),  # 统一重建mask
        "labels": labels
    }

def old_process_func(example):
    """
    将数据集进行预处理
    """ 
    input_ids, attention_mask, labels = [], [], []
    messages = build_messages(example)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    instruction = tokenizer(
        text,
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

if __name__ == "__main__":
    # Example usage
    example = pd.Series({
        "instruction": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "input": "The capital of France is",
        "output": "Paris."
    })
    old_results = old_process_func(example)
    new_results = process_func(example)
    print("Old Process Function Results:", old_results)
    print("New Process Function Results:", new_results)
    assert old_results == new_results, "The results of the old and new process functions do not match."
    print("The old and new process functions produce the same results.")