import json
import random
from tqdm import tqdm
from typing import Optional
from modelscope import AutoTokenizer

from prompt import *
from rag.core import Retriever, LexiconRetriever, MultiClassRetriever, MultiClassWrongExpRetriever
from tools.convert import output2triple

def get_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False, 
        trust_remote_code=True
    )
    return tokenizer

def is_overlength(tokenizer, text, max_length):
    input_ids = tokenizer.encode(text, return_tensors="pt")[0]
    return len(input_ids) > max_length

def build_prompt(
        datas: list,
        prompt_template: str,
        use_srag: bool = False,
        use_lex: bool = False,
        srag_retriever: Optional[MultiClassRetriever] = None,
        lex_retriever: Optional[LexiconRetriever] = None,
        example_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        srag_threshold: float = 0,
        lex_top_k: int = -1,
        lex_sim_top_k: int = -1,
        lex_sim_threshold: float = 0,
        weights: Optional[dict] = None,
        weights_reverse: bool = False,
        auto_length: bool = False,
        model_path: Optional[str] = None,
        max_length: int = 2048,
        is_test_data: bool = False
        ):
    """构建相似词典检索的提示模板"""

    def build_prompt(
            raw_data: dict, 
            use_srag: bool,
            use_lex: bool,
            srag_retriever: MultiClassRetriever, 
            lex_retriever: LexiconRetriever, 
            prompt_template: str, 
            example_template: str, 
            srag_top_k: int, 
            srag_threshold: float, 
            lex_top_k: int, 
            lex_sim_top_k: int, 
            lex_sim_threshold: float, 
            weights: Optional[dict], 
            weights_reverse: bool):
        
        if use_srag:
            retrieve_contents, retrieve_outputs = srag_retriever.retrieve(raw_data['content'], srag_top_k, threshold=srag_threshold, weights=weights, weights_reverse=weights_reverse)
            examples = []
            for retrieve_content, retrieve_output in zip(retrieve_contents, retrieve_outputs):
                example_prompt = example_template.replace("{retrieve_content}", retrieve_content).\
                                                    replace("{retrieve_output}", output2triple(retrieve_output))
                examples.append(example_prompt)
        else:
            examples = []
        
        if use_lex:
            lex_contents = lex_retriever.including_retrieve(raw_data['content'], lex_top_k)
            simlex_contents = lex_retriever.similarity_retrieve(raw_data['content'], lex_sim_top_k, deduplicate=True, threshold=lex_sim_threshold)
            for simlex_content in simlex_contents:
                if simlex_content not in lex_contents:
                    lex_contents.append(simlex_content)
        else:
            lex_contents = []
        
        prompt = prompt_template.replace("{examples}", "\n".join(examples)).\
                                replace("{lexicons}", "\n".join(lex_contents)).\
                                replace("{text}", raw_data["content"])
        
        return prompt, examples, lex_contents

    if auto_length and model_path is not None:
        tokenizer = get_tokenizer(model_path)
    else:
        tokenizer = None

    pbar = tqdm(
            total=len(datas),
            desc=f"Preprocessing datas",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )
    messages = []
    srag_examples_nums = 0
    for raw_data in datas:
        triples = []
        for quadruple in raw_data["quadruples"]:
            label = quadruple["targeted_group"]
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")
        
        
        prompt, examples, lex_contents = build_prompt(
            raw_data,
            use_srag,
            use_lex,
            srag_retriever,
            lex_retriever,
            prompt_template,
            example_template,
            srag_top_k,
            srag_threshold,
            lex_top_k,
            lex_sim_top_k,
            lex_sim_threshold,
            weights,
            weights_reverse
        )

        i = 1
        while auto_length and tokenizer is not None and is_overlength(tokenizer, prompt, max_length):
            print(f"Over length: {len(tokenizer(prompt)['input_ids'])} > {max_length}, reduce srag examples and rebuild prompt.")
            prompt, examples, lex_contents = build_prompt(
                raw_data,
                srag_retriever,
                lex_retriever,
                prompt_template,
                example_template,
                srag_top_k - i,
                srag_threshold,
                lex_top_k,
                lex_sim_top_k,
                lex_sim_threshold,
                weights,
                weights_reverse
            )
            i += 1

        srag_examples_nums += len(examples)

        answer = " [SEP] ".join(triples) + " [END]"
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
    if len(datas) > 0:
        print(f"SRAG avg examples nums: {srag_examples_nums / len(datas)}")

    return messages

def make_data(
        raw_data_path: str, 
        test_data_path: str,
        train_output_path: str, 
        val_output_path: str, 
        test_output_path: str,
        prompt_template: str, 
        example_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_srag: bool = False,
        srag_top_k: int = 1,
        srag_threshold: float = 0,
        use_lex: bool = False,
        lex_top_k: int = -1,
        lex_sim_top_k: int = -1,
        lex_sim_threshold: float = 0,
        weights: Optional[dict] = None,
        weights_reverse: bool = False,
        auto_length=False,
        model_path: Optional[str] = None,
        max_length: int = 2048,
        ):
    """转换训练/验证集数据格式"""

    messages = []
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    split_idx = int(len(raw_datas) * 0.9)

    if use_srag:
        srag_retriever = MultiClassRetriever(model_path="./models/base/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5")
        srag_retriever.load_datas(data_list=raw_datas[:split_idx])
        srag_retriever.build_retrievers()
    else:
        srag_retriever = None

    if use_lex:
        lex_retriever = LexiconRetriever(model_path="./models/base/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/lexicon/annotated_lexicon.json")
    else:
        lex_retriever = None
    
    

    messages = build_prompt(
        datas=raw_datas[:split_idx],
        prompt_template=prompt_template,
        use_srag=use_srag,
        use_lex=use_lex,
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        srag_threshold=srag_threshold,
        lex_top_k=lex_top_k,
        lex_sim_top_k=lex_sim_top_k,
        lex_sim_threshold=lex_sim_threshold,
        weights=weights,
        weights_reverse=weights_reverse,
        auto_length=auto_length,
        model_path=model_path,
        max_length=max_length,
    )

    
    with open(train_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    if use_srag:
        srag_retriever.load_datas(data_list=raw_datas)
        srag_retriever.build_retrievers()

    messages = build_prompt(
        datas=raw_datas[split_idx:],
        prompt_template=prompt_template,
        use_srag=use_srag,
        use_lex=use_lex,
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        srag_threshold=srag_threshold,
        lex_top_k=lex_top_k,
        lex_sim_top_k=lex_sim_top_k,
        lex_sim_threshold=lex_sim_threshold,
        weights=weights,
        weights_reverse=weights_reverse,
        auto_length=auto_length,
        model_path=model_path,
        max_length=max_length,
    )

    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    with open(test_data_path, "r") as file:
        test_datas = json.load(file)

    messages = build_prompt(
        datas=test_datas,
        prompt_template=prompt_template,
        use_srag=use_srag,
        use_lex=use_lex,
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        srag_threshold=srag_threshold,
        lex_top_k=lex_top_k,
        lex_sim_top_k=lex_sim_top_k,
        lex_sim_threshold=lex_sim_threshold,
        weights=weights,
        weights_reverse=weights_reverse,
        auto_length=auto_length,
        model_path=model_path,
        max_length=max_length,
        is_test_data=True
    )

    with open(test_output_path, "w", encoding="utf-8") as file:
        json.dump([{
                "id": message["id"], 
                "content": message["content"], 
                "gt_quadruples": message.get("gt_quadruples", []), 
                "messages_list": [[{'content': system_prompt, 'role': 'system'}, {'content': message["input"], 'role': 'user'}]],
            } for message in messages], file, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    make_data(
        raw_data_path="data/full/std/train.json", 
        test_data_path="data/full/std/test.json",
        train_output_path="data/exp_data/after/train.jsonl", 
        val_output_path="data/exp_data/after/val.jsonl",
        test_output_path="data/exp_data/after/test.json",
        prompt_template=RAG_PROMPT_USER_V2,
        use_srag=True,
        srag_top_k=11,
        example_template=RAG_PROMPT_EXAMPLE_V2,
        use_lex=True,
        lex_top_k=5,
        lex_sim_top_k=5,
        auto_length=True,
        model_path="models/base/Qwen2.5-7B-Instruct",
        max_length=1280,
        system_prompt=DEFAULT_SYSTEM_PTOMPT_EN)