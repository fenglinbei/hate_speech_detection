import os
import re
import json
import torch
import random
import swanlab
import argparse
import datetime
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

from prompt import *
from utils.log import init_logger
from metrics.metric_llm import LLMmetrics
from utils.parser import parse_llm_output_trip, validate_quadruples
logger = init_logger(level="INFO", show_console=True)

def load_config(config_path):
    """从指定路径加载JSON配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_device_map(config):
    """从配置中构建设备映射，如未提供则返回None使用自动分配"""
    return config.get('device_map', "auto")

# device_map = {
#     'model.embed_tokens': "cuda:0",
#     'model.layers.0': "cuda:0",
#     'model.layers.1': "cuda:0",
#     'model.layers.2': "cuda:0",
#     'model.layers.3': "cuda:0",
#     'model.layers.4': "cuda:0",
#     'model.layers.5': "cuda:0",
#     'model.layers.6': "cuda:0",
#     'model.layers.7': "cuda:0",
#     'model.layers.8': "cuda:0",
#     'model.layers.9': "cuda:0",
#     'model.layers.10': "cuda:0",
#     'model.layers.11': "cuda:1",
#     'model.layers.12': "cuda:1",
#     'model.layers.13': "cuda:1",
#     'model.layers.14': "cuda:1",
#     'model.layers.15': "cuda:1",
#     'model.layers.16': "cuda:1",
#     'model.layers.17': "cuda:1",
#     'model.layers.18': "cuda:1",
#     'model.layers.19': "cuda:1",
#     'model.layers.20': "cuda:1",
#     'model.layers.21': "cuda:1",
#     'model.layers.22': "cuda:1",
#     'model.layers.23': "cuda:1",
#     'model.layers.24': "cuda:1",
#     'model.layers.25': "cuda:1",
#     'model.layers.26': "cuda:1",
#     'model.layers.27': "cuda:1",
#     'model.norm': "cuda:0",
#     'lm_head': "cuda:0"
# }

def dataset_transfer_no_think_test(raw_data_path: str, test_output_path: str, prompt_template: str):
    """转换测试集数据格式"""
    messages = []
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    for raw_data in raw_datas:
        triples = []
        for quadruple in raw_data["quadruples"]:
            label = quadruple["targeted_group"]
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")
        
        input = prompt_template.format(text=raw_data["content"])
        answer = " [SEP] ".join(triples) + " [END]"
        message = {"instruction": "", "input": f"{input}", "output": answer}
        messages.append(message)
    
    with open(test_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def dataset_transfer_no_think(raw_data_path: str, train_output_path: str, val_output_path: str, prompt_template: str):
    """转换训练/验证集数据格式"""
    messages = []
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    for raw_data in raw_datas:
        triples = []
        for quadruple in raw_data["quadruples"]:
            label = quadruple["targeted_group"]
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")
        
        input = prompt_template.format(text=raw_data["content"])
        answer = " [SEP] ".join(triples) + " [END]"
        message = {"instruction": "", "input": f"{input}", "output": answer}
        messages.append(message)
    
    split_idx = int(len(messages) * 0.9)
    with open(train_output_path, "w", encoding="utf-8") as file:
        for message in messages[:split_idx]:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in messages[split_idx:]:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def prompt_to_text(prompt: str, prompt_template: str) -> str:
    """从提示模板中提取原始文本"""
    placeholder = "{text}"
    return prompt.split(placeholder)[0] if placeholder in prompt else prompt

def build_messages(example: pd.Series) -> list[dict]:
    if example["instruction"]
        messages = [{'content': example["instruction"], 'role': 'system'}, {'content': example["input"], 'role': 'user'}]
    else:
        messages = [{'content': example["input"], 'role': 'user'}]
    return messages
       

class CustomTrainer(Trainer):
    """自定义Trainer类增加评估指标"""
    def __init__(self, 
                 *args, 
                 eval_tokenizer,
                 eval_config: dict,
                 llm_metrics: LLMmetrics, 
                 eval_raw_dataset=None, 
                 max_retries: int = 0,
                 eval_num: int = 100,
                 prompt_template: str = "",
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.eval_config = eval_config
        self.eval_raw_dataset = eval_raw_dataset
        self.eval_tokenizer = eval_tokenizer
        self.llm_metrics = llm_metrics
        self.max_retries = max_retries
        self.eval_num = eval_num
        self.prompt_template = prompt_template
        
    def evaluate(self, **kwargs):
        """自定义评估逻辑"""
        metrics = super().evaluate(**kwargs)
        logger.debug(metrics)
        custom_metrics = self.evaluate_custom()
        metrics.update(custom_metrics)
        self.log(metrics)
        swanlab.log(metrics)
        return metrics
        
    def evaluate_custom(self):
        """生成回复并计算四元组指标"""
        results = []
        model = self.model.eval()

        progress_bar = tqdm(
            total=min(self.eval_num, len(self.eval_raw_dataset)),
            desc="Evaluating custom metrics",
            dynamic_ncols=True
        )
        
        test_text_list = []
        for idx, example in enumerate(self.eval_raw_dataset[:self.eval_num]):
            item_id = idx
            final_status = "success"
            try:
                messages = build_messages(example)
                response = ""
                for attempt in range(self.max_retries + 1):
                    response = predict(messages, model, self.eval_tokenizer, config=self.eval_config)
                    response_text = f"Question: {example['input']}\nLLM:{response}"
                    test_text_list.append(swanlab.Text(response_text))
                    
                    pred_quads = parse_llm_output_trip(response)
                    gt_quads = parse_llm_output_trip(example['output'])
                    
                    if validate_quadruples(pred_quads):
                        final_status = "success"
                        break
                    else:
                        logger.debug(f"Validation failed (attempt:{attempt+1})")
                        final_status = "invalid"
            except Exception as e:
                logger.error(f"Evaluation error: {str(e)}")
                pred_quads, gt_quads = [], []
                final_status = "fail"
                
            results.append({
                "id": item_id,
                "content": prompt_to_text(example["input"], self.prompt_template),
                "prompt": example["input"],
                "llm_output": response,
                "gt_quadruples": gt_quads,
                "pred_quadruples": pred_quads,
                "status": final_status,
                "attempts": self.max_retries + 1
            })

            success_count = len([r for r in results if r['status']=='success'])
            progress_bar.set_postfix({
                "success": f"{success_count}/{len(results)}",
                "rate": f"{success_count/len(results):.1%}" if len(results) else "0%"
            })
            progress_bar.update(1)

        progress_bar.close()
        swanlab.log({"Prediction": test_text_list})

        # 保存评估结果
        output_dir = Path("finetune/eval_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        current_step = self.state.global_step
        filename = output_dir / f"eval_results_step_{current_step}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "step": current_step,
                "timestamp": str(datetime.datetime.now()),
                "eval_num": len(results),
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        return self.llm_metrics.run(results)


def predict(messages, model, tokenizer, config):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    attention_mask = model_inputs['attention_mask']

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=config.get("max_length", 512),
        repetition_penalty=config.get("repetition_penalty", 1.15),
        temperature=config.get("temperature"),
        top_p=config.get("top_p"),
        top_k=config.get("top_k"),
        min_p=config.get("min_p"),
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def run(config: dict):
    MAX_LENGTH = config.get('max_length', 512)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.get('cuda_devices', [0,1,2,3])))

    swanlab.config.update({ # type: ignore
        "model": config['model_name'],
        "prompt": config['prompt_template'],
        "data_max_length": MAX_LENGTH,
        "use_bf16": config['training'].get('bf16', False)
    })

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config['model_path'], 
            use_fast=False, 
            trust_remote_code=True
        )

        torch_dtype = torch.bfloat16 if config['training'].get('bf16', False) else torch.float32
        device_map = build_device_map(config)

        model = AutoModelForCausalLM.from_pretrained(
            config['model_path'], 
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        model.enable_input_require_grads()
    except Exception as err:
        logger.exception(err)
        exit()

    llm_metrics = LLMmetrics()

    def process_func(example):
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
    
    # 数据准备
    data_config = config['data']

    # 加载数据集
    train_df = pd.read_json(data_config['train_data_path'], lines=True)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

    eval_df = pd.read_json(data_config['val_data_path'], lines=True)
    eval_raw = [row for _, row in eval_df.iterrows()]
    eval_ds = Dataset.from_pandas(eval_df)
    eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

    # 训练参数配置
    training_args = TrainingArguments(
        **config['training']
    )

    # 训练器初始化
    trainer = CustomTrainer(
        model=model,
        eval_tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        eval_raw_dataset=eval_raw,
        llm_metrics=llm_metrics,
        max_retries=config['eval'].get('max_retries', 0),
        eval_num=config['eval'].get('eval_num', 100),
        eval_config=config["eval"],
        prompt_template=config['prompt_template'],
        callbacks=[SwanLabCallback(
            project=os.environ["SWANLAB_PROJECT"],
            experiment_name=config['exp_name'],
        )]
    )

    trainer.train()
    swanlab.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Fine-tuning Script')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault('transfer_data', True)
    config.setdefault('exp_name', 'default-exp')

    # 设置随机种子
    if 'random_seed' in config:
        random.seed(config['random_seed'])
        torch.manual_seed(config['random_seed'])

    # dataset_transfer_no_think_test("data/full/std/train.json", "finetune/data/test.jsonl")
    # dataset_transfer_no_think("data/full/std/train.json", "finetune/data/train.jsonl", "finetune/data/val.jsonl")
    # run()
    run(config)
