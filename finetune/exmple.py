import os
import json
import torch
import random
import swanlab
import pandas as pd

from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

from prompt import *
from utils.log import init_logger
logger = init_logger(level="DEBUG", show_console=True)

random.seed("23333333")
os.environ["SWANLAB_PROJECT"]="qwen3-sft-hsd"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
device_map = {
    'model.embed_tokens': "cuda:0",
    'model.layers.0': "cuda:0",
    'model.layers.1': "cuda:0",
    'model.layers.2': "cuda:0",
    'model.layers.3': "cuda:0",
    'model.layers.4': "cuda:0",
    'model.layers.5': "cuda:0",
    'model.layers.6': "cuda:0",
    'model.layers.7': "cuda:0",
    'model.layers.8': "cuda:0",
    'model.layers.9': "cuda:1",
    'model.layers.10': "cuda:1",
    'model.layers.11': "cuda:1",
    'model.layers.12': "cuda:1",
    'model.layers.13': "cuda:1",
    'model.layers.14': "cuda:1",
    'model.layers.15': "cuda:1",
    'model.layers.16': "cuda:1",
    'model.layers.17': "cuda:1",
    'model.layers.18': "cuda:2",
    'model.layers.19': "cuda:2",
    'model.layers.20': "cuda:2",
    'model.layers.21': "cuda:2",
    'model.layers.22': "cuda:2",
    'model.layers.23': "cuda:2",
    'model.layers.24': "cuda:2",
    'model.layers.25': "cuda:2",
    'model.layers.26': "cuda:2",
    'model.layers.27': "cuda:3",
    'model.layers.28': "cuda:3",
    'model.layers.29': "cuda:3",
    'model.layers.30': "cuda:3",
    'model.layers.31': "cuda:3",
    'model.layers.32': "cuda:3",
    'model.layers.33': "cuda:3",
    'model.layers.34': "cuda:3",
    'model.layers.35': "cuda:3",
    'model.norm': "cuda:3",
    'lm_head': "cuda:3"
}

device_map = {
    'model.embed_tokens': "cuda:0",
    'model.layers.0': "cuda:0",
    'model.layers.1': "cuda:0",
    'model.layers.2': "cuda:0",
    'model.layers.3': "cuda:0",
    'model.layers.4': "cuda:0",
    'model.layers.5': "cuda:0",
    'model.layers.6': "cuda:0",
    'model.layers.7': "cuda:0",
    'model.layers.8': "cuda:0",
    'model.layers.9': "cuda:0",
    'model.layers.10': "cuda:0",
    'model.layers.11': "cuda:0",
    'model.layers.12': "cuda:0",
    'model.layers.13': "cuda:0",
    'model.layers.14': "cuda:0",
    'model.layers.15': "cuda:0",
    'model.layers.16': "cuda:0",
    'model.layers.17': "cuda:0",
    'model.layers.18': "cuda:0",
    'model.layers.19': "cuda:0",
    'model.layers.20': "cuda:0",
    'model.layers.21': "cuda:0",
    'model.layers.22': "cuda:0",
    'model.layers.23': "cuda:0",
    'model.layers.24': "cuda:0",
    'model.layers.25': "cuda:0",
    'model.layers.26': "cuda:0",
    'model.layers.27': "cuda:0",
    'model.norm': "cuda:0",
    'lm_head': "cuda:0"
}

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
MAX_LENGTH = 512

label_map = {
    "Racism": "A",
    "Region": "B",
    "LGBTQ": "C",
    "Sexism": "D",
    "others": "E",
    "non-hate": "F"
    }

swanlab.config.update({
    "model": "Qwen/Qwen3-1.7B",
    "prompt": TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V3,
    "data_max_length": MAX_LENGTH,
    })



def dataset_transfer_no_think_test(raw_data_path: str, test_output_path: str):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    for raw_data in raw_datas:
        triples = []

        for quadruple in raw_data["quadruples"]:
            raw_labels = quadruple["targeted_group"].split(", ")
            label = ", ".join([label_map[raw_label.strip()] for raw_label in raw_labels])
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")


        input = TRAIN_PROMPT_ZERO_SHOT_V3.format(text=raw_data["content"])
        output = " [SEP] ".join(triples) + " [END]"
        message = {
            "instruction": TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V3,
            "input": f"{input}",
            "output": output,
        }
        messages.append(message)

    # 保存重构后的JSONL文件
    with open(test_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def dataset_transfer_no_think(raw_data_path: str, train_output_path: str, val_output_path: str):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    for raw_data in raw_datas:
        triples = []

        for quadruple in raw_data["quadruples"]:
            raw_labels = quadruple["targeted_group"].split(", ")
            label = ", ".join([label_map[raw_label.strip()] for raw_label in raw_labels])
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")


        input = TRAIN_PROMPT_ZERO_SHOT_V3.format(text=raw_data["content"])
        output = " [SEP] ".join(triples) + " [END]"
        message = {
            "instruction": TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V3,
            "input": f"{input}",
            "output": output,
        }
        messages.append(message)

    split_idx = int(len(messages) * 0.9)
    train_datas = messages[:split_idx]
    val_datas = messages[split_idx:]

    # 保存重构后的JSONL文件
    with open(train_output_path, "w", encoding="utf-8") as file:
        for message in train_datas:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in val_datas:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")



def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def run():
    # 在modelscope上下载Qwen模型到本地目录下
    model_dir = snapshot_download("Qwen/Qwen3-1.7B", cache_dir="models/", revision="master")

    # Transformers加载模型权重
    try:
        # tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-8B", use_fast=False, trust_remote_code=True)
        # model = AutoModelForCausalLM.from_pretrained("models/Qwen3-8B", torch_dtype=torch.bfloat16, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained("models/Qwen/Qwen3-1.7B", use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("models/Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16, device_map=device_map)
        model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    except Exception as err:
        logger.exception(err)
        exit()

    def process_func(example):
        """
        将数据集进行预处理
        """ 
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(
            f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        response = tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
        )
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   

    # 加载、处理数据集和测试集
    train_jsonl_path = "finetune/data/train.jsonl"
    val_jsonl_path = "finetune/data/val.jsonl"

    # 得到训练集
    train_df = pd.read_json(train_jsonl_path, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

    # 得到验证集
    eval_df = pd.read_json(val_jsonl_path, lines=True)
    eval_ds = Dataset.from_pandas(eval_df)
    eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

    args = TrainingArguments(
        output_dir="models/Qwen3-1.7B-sft-hsd/",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=20,
        logging_steps=20,
        num_train_epochs=4,
        save_steps=200,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="swanlab",
        run_name="qwen3-1.7B-hsd-sft",
        # fsdp="full_shard",  # 添加FSDP支持
        # fsdp_config={"fsdp_transformer_layer_cls_to_wrap": ["QwenBlock"]},  # 根据实际模型结构调整
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    

    test_jsonl_path = "finetune/data/test.jsonl"

    # 用测试集的前3条，主观看模型
    test_df = pd.read_json(test_jsonl_path, lines=True)[:20]

    test_text_list = []

    for index, row in test_df.iterrows():
        instruction = row['instruction']
        input_value = row['input']

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]

        response = predict(messages, model, tokenizer)

        response_text = f"""
        Question: {input_value}

        LLM:{response}
        """
        
        test_text_list.append(swanlab.Text(response_text))
        print(response_text)

    swanlab.log({"Prediction": test_text_list})

    swanlab.finish()

def run_lora():
    # 在modelscope上下载Qwen模型到本地目录下
    model_dir = snapshot_download("models/Qwen3-8B", cache_dir="models/", revision="master")

    # Transformers加载模型权重
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-8B", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("models/Qwen3-8B", device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    def process_func(example):
        """
        将数据集进行预处理
        """ 
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(
            f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        response = tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
        )
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   

    # 加载、处理数据集和测试集
    train_jsonl_path = "finetune/data/train.jsonl"
    val_jsonl_path = "finetune/data/val.jsonl"

    # 得到训练集
    train_df = pd.read_json(train_jsonl_path, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

    # 得到验证集
    eval_df = pd.read_json(val_jsonl_path, lines=True)
    eval_ds = Dataset.from_pandas(eval_df)
    eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

    # 配置LoRA参数
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言模型
        inference_mode=False,
        r=8,              # LoRA秩
        lora_alpha=32,    # 缩放因子
        lora_dropout=0.1, # Dropout概率
        target_modules=["q_proj", "v_proj"]  # 针对Qwen的注意力层
    )

    # 将基础模型转换为LoRA模型
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # 打印可训练参数数量（应为原模型的0.1%-1%）

    args = TrainingArguments(
        output_dir="finetune/output_models/Qwen3-1.7B",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=400,
        learning_rate=2e-5,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="swanlab",
        run_name="qwen3-1.7B",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    
    # 保存训练后的LoRA权重
    model.save_pretrained(args.output_dir)
    
    # 测试时加载基础模型+LoRA权重
    base_model = AutoModelForCausalLM.from_pretrained(
        "models/Qwen/Qwen3-1.7B", 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.output_dir)
    model = model.merge_and_unload()  # 合并权重便于推理

    test_jsonl_path = "finetune/data/test.jsonl"

    # 用测试集的前3条，主观看模型
    test_df = pd.read_json(test_jsonl_path, lines=True)[:20]

    test_text_list = []

    for index, row in test_df.iterrows():
        instruction = row['instruction']
        input_value = row['input']

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]

        response = predict(messages, model, tokenizer)

        response_text = f"""
        Question: {input_value}

        LLM:{response}
        """
        
        test_text_list.append(swanlab.Text(response_text))
        print(response_text)

    swanlab.log({"Prediction": test_text_list})

    swanlab.finish()

if __name__ == "__main__":
    # dataset_transfer_no_think_test("data/full/std/train.json", "finetune/data/test.jsonl")
    # dataset_transfer_no_think("data/full/std/train.json", "finetune/data/train.jsonl", "finetune/data/val.jsonl")
    run()