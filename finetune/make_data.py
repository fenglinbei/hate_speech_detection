import json
from tqdm import tqdm
from typing import Optional

from prompt import *
from rag.core import Retriever
from tools.convert import output2triple

def dataset_transfer_no_think_test(raw_data_path: str, test_output_path: str, prompt_template: str, system_prompt: Optional[str] = None):
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
        message = {"instruction": system_prompt if system_prompt else "", "input": f"{input}", "output": answer, "content": raw_data["content"]}
        messages.append(message)
    
    with open(test_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def dataset_transfer_no_think(
        raw_data_path: str, 
        train_output_path: str, 
        val_output_path: str, 
        prompt_template: str, 
        system_prompt: Optional[str] = None
        ):
    """转换训练/验证集数据格式"""

    messages = []
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    split_idx = int(len(raw_datas) * 0.9)
    retriever = Retriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5")
    retriever.create_embeddings(raw_datas[:split_idx])
    
    pbar = tqdm(
            total=len(raw_datas),
            desc=f"Preprocessing datas",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )

    for raw_data in raw_datas:
        triples = []
        for quadruple in raw_data["quadruples"]:
            label = quadruple["targeted_group"]
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")
        
        if not retriever:
            input = prompt_template.format(text=raw_data["content"])
        else:
            assert isinstance(retriever, Retriever)
            retrieve_contents, retrieve_outputs = retriever.retrieve(raw_data['content'])
            input = prompt_template.replace("{retrieve_content}", retrieve_contents[0]).\
            replace("{retrieve_output}", output2triple(retrieve_outputs[0])).\
            replace("{text}", raw_data["content"])

        answer = " [SEP] ".join(triples) + " [END]"
        message = {"instruction": system_prompt if system_prompt else "", "input": f"{input}", "output": answer, "content": raw_data["content"]}
        messages.append(message)
        pbar.update(1)
    
    with open(train_output_path, "w", encoding="utf-8") as file:
        for message in messages[:split_idx]:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in messages[split_idx:]:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    dataset_transfer_no_think("data/full/std/train.json", "finetune/data/train.jsonl", "finetune/data/val.jsonl", RAG_PROMPT_USER_V1)