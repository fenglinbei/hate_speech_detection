import os
import json
from typing import Optional

from tqdm import tqdm
from prompt import *
from rag.core import StepOneRetriever


def build_step_one_prompt(
        datas: list,
        srag_retriever: StepOneRetriever,
        prompt_template: str,
        no_rag_prompt_template: str,
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        srag_threshold: float = 0,
        is_test_data: bool = False
        ):
    """构建相似词典检索的提示模板"""

    pbar = tqdm(
            total=len(datas),
            desc=f"Preprocessing datas",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )
    messages = []
    for raw_data in datas:
        tuples = []
        for quadruple in raw_data["quadruples"]:
            tuples.append(f"{quadruple['target']} | {quadruple['argument']}")

        retrieve_contents, retrieve_outputs = srag_retriever.retrieve(raw_data['content'], srag_top_k, threshold=srag_threshold)
        examples = []
        for retrieve_content, retrieve_output in zip(retrieve_contents, retrieve_outputs):
            example_prompt = example_template.replace("{retrieve_content}", retrieve_content).\
                                              replace("{retrieve_output}", retrieve_output)
            examples.append(example_prompt)
    
        
        if not examples:
            prompt = no_rag_prompt_template.replace("{text}", raw_data["content"])
        else:
            prompt = prompt_template.replace("{text}", raw_data["content"]).replace("{examples}", "\n".join(examples))


        answer = " [SEP] ".join(tuples) + " [END]"
        message = {
            "id": raw_data["id"],
            "instruction": system_prompt if system_prompt else "", 
            "input": f"{prompt}", 
            "output": answer, 
            "content": raw_data["content"],
            "gt_quadruples": raw_data["quadruples"] if  is_test_data else ""
            }
        messages.append(message)
        pbar.update(1)

    return messages

def make_step_one_data(
        raw_data_path: str, 
        test_data_path: str,
        output_dir: str,
        prompt_template: str, 
        no_rag_prompt_template: str,
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        srag_threshold: float = 0
        ):
    """转换训练/验证集数据格式"""

    os.makedirs(output_dir, exist_ok=True)

    messages = []
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    split_idx = int(len(raw_datas) * 0.9)
    srag_retriever = StepOneRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5")
    
    srag_retriever.create_embeddings(raw_datas[:split_idx])

    messages = build_step_one_prompt(
        datas=raw_datas[:split_idx],
        srag_retriever=srag_retriever,
        prompt_template=prompt_template,
        no_rag_prompt_template=no_rag_prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        srag_threshold=srag_threshold,
        is_test_data=False
    )

    with open(f"{output_dir}/train.jsonl", "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    srag_retriever.create_embeddings(raw_datas)

    messages = build_step_one_prompt(
        datas=raw_datas[split_idx:],
        srag_retriever=srag_retriever,
        prompt_template=prompt_template,
        no_rag_prompt_template=no_rag_prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        srag_threshold=srag_threshold,
        is_test_data=False
    )

    with open(f"{output_dir}/val.jsonl", "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    with open(test_data_path, "r") as file:
        test_datas = json.load(file)

    messages = build_step_one_prompt(
        datas=test_datas,
        srag_retriever=srag_retriever,
        prompt_template=prompt_template,
        no_rag_prompt_template=no_rag_prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        srag_threshold=srag_threshold,
        is_test_data=True
    )

    with open(f"{output_dir}/test.json", "w", encoding="utf-8") as file:
        json.dump([{
                "id": message["id"], 
                "content": message["content"], 
                "gt_quadruples": message.get("gt_quadruples", []), 
                "messages_list": [[{'content': system_prompt, 'role': 'system'}, {'content': message["input"], 'role': 'user'}]],
            } for message in messages], file, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    make_step_one_data(
        raw_data_path="./data/full/std/train.json",
        test_data_path="./data/full/std/test.json",
        output_dir="finetune/data/step_one_no_rag",
        prompt_template=STEP_ONE_RAG_PROMPT_USER_V1,
        no_rag_prompt_template=STEP_ONE_PROMPT_USER_V1,
        example_template=STEP_ONE_RAG_PROMPT_EXAMPLE_V1,
        system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
        srag_top_k=0,
        srag_threshold=0
    )
