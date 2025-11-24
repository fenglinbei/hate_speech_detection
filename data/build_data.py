import json
import random
from tqdm import tqdm
from typing import Optional
from transformers import AutoTokenizer

from prompt import *
from data.config import Config
from rag.core import Retriever, LexiconRetriever, MultiClassRetriever, MultiClassWrongExpRetriever
from tools.convert import output2triple

def get_tokenizer(model_path: str):
    """获取tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=True, 
        trust_remote_code=True
    )
    return tokenizer

def is_overlength(tokenizer, text, max_length):
    """检查文本是否超过最大长度"""
    input_ids = tokenizer.encode(text, return_tensors="pt")[0]
    return len(input_ids) > max_length

def build_prompt(
        datas: list,
        config: Config,
        srag_retriever: Optional[MultiClassRetriever] = None,
        lex_retriever: Optional[LexiconRetriever] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        is_test_data: bool = False
        ):
    """构建相似词典检索的提示模板[3](@ref)"""

    def build_single_prompt(
            raw_data: dict, 
            srag_retriever: Optional[MultiClassRetriever], 
            lex_retriever: Optional[LexiconRetriever]
        ):
        """构建单个数据的提示"""
        
        if config.use_srag and srag_retriever is not None and config.example_template is not None:
            retrieve_contents, retrieve_outputs = srag_retriever.retrieve(
                raw_data['content'], 
                config.srag_top_k, 
                threshold=config.srag_threshold, 
                weights=config.weights, 
                weights_reverse=config.weights_reverse
            )
            examples = []
            for retrieve_content, retrieve_output in zip(retrieve_contents, retrieve_outputs):
                example_prompt = config.example_template.replace("{retrieve_content}", retrieve_content).\
                                                    replace("{retrieve_output}", output2triple(retrieve_output))
                examples.append(example_prompt)
        else:
            examples = []
        
        if config.use_lex and lex_retriever is not None:
            lex_contents = lex_retriever.including_retrieve(raw_data['content'], config.lex_top_k)
            simlex_contents = lex_retriever.similarity_retrieve(
                raw_data['content'], 
                config.lex_sim_top_k, 
                deduplicate=True, 
                threshold=config.lex_sim_threshold
            )
            for simlex_content in simlex_contents:
                if simlex_content not in lex_contents:
                    lex_contents.append(simlex_content)
        else:
            lex_contents = []
        
        prompt = config.prompt_template.replace("{examples}", "\n".join(examples)).\
                                      replace("{lexicons}", "\n".join(lex_contents)).\
                                      replace("{text}", raw_data["content"])
        
        return prompt, examples, lex_contents

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
        
        prompt, examples, lex_contents = build_single_prompt(
            raw_data=raw_data,
            srag_retriever=srag_retriever,
            lex_retriever=lex_retriever
        )

        # 自动长度调整
        i = 1
        while config.auto_length and tokenizer is not None and is_overlength(tokenizer, prompt, config.max_length):
            print(f"Over length: {len(tokenizer(prompt)['input_ids'])} > {config.max_length}, reduce srag examples and rebuild prompt.")
            # 临时减少srag_top_k
            original_top_k = config.srag_top_k
            config.srag_top_k = original_top_k - i
            prompt, examples, lex_contents = build_single_prompt(
                raw_data=raw_data,
                srag_retriever=srag_retriever,
                lex_retriever=lex_retriever
            )
            i += 1
            config.srag_top_k = original_top_k  # 恢复原值

        srag_examples_nums += len(examples)

        answer = " [SEP] ".join(triples) + " [END]"
        message = {
            "id": raw_data["id"],
            "instruction": config.system_prompt if config.system_prompt else "", 
            "input": f"{prompt}", 
            "output": answer, 
            "content": raw_data["content"],
            "gt_quadruples": raw_data["quadruples"] if is_test_data else ""
            }
        messages.append(message)
        pbar.update(1)
    
    if len(datas) > 0:
        print(f"SRAG avg examples nums: {srag_examples_nums / len(datas)}")

    return messages

def make_data(config: Config):
    """转换训练/验证集数据格式[4](@ref)"""

    messages = []
    with open(config.raw_data_path, "r") as file:
        raw_datas = json.load(file)

    split_idx = int(len(raw_datas) * config.split_ratio)

    if config.use_srag:
        srag_retriever = MultiClassRetriever(
            model_path=config.srag_model_path, 
            model_name="bge-large-zh-v1.5"
        )
        srag_retriever.load_datas(data_list=raw_datas[:split_idx])
        srag_retriever.build_retrievers()
    else:
        srag_retriever = None

    if config.use_lex:
        lex_retriever = LexiconRetriever(
            model_path=config.lexicon_model_path, 
            model_name="bge-large-zh-v1.5", 
            data_path=config.lexicon_data_path
        )
    else:
        lex_retriever = None
    
    tokenizer = None
    if config.auto_length and config.tokenizer_path is not None:
        tokenizer = get_tokenizer(config.tokenizer_path)

    # 处理训练数据
    messages = build_prompt(
        datas=raw_datas[:split_idx],
        config=config,
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        tokenizer=tokenizer,
    )

    with open(config.train_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    # 更新检索器用于验证数据
    if config.use_srag and srag_retriever is not None:
        srag_retriever.load_datas(data_list=raw_datas)
        srag_retriever.build_retrievers()

    # 处理验证数据
    messages = build_prompt(
        datas=raw_datas[split_idx:],
        config=config,
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        tokenizer=tokenizer,
    )

    with open(config.val_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    # 处理测试数据
    with open(config.test_data_path, "r") as file:
        test_datas = json.load(file)

    messages = build_prompt(
        datas=test_datas,
        config=config,
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        tokenizer=tokenizer,
        is_test_data=True
    )

    with open(config.test_output_path, "w", encoding="utf-8") as file:
        json.dump([{
                "id": message["id"], 
                "content": message["content"], 
                "gt_quadruples": message.get("gt_quadruples", []), 
                "messages_list": [[
                    {'content': config.system_prompt, 'role': 'system'}, 
                    {'content': message["input"], 'role': 'user'}
                ]],
            } for message in messages], file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Build training and validation data')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    config = Config(args.config)
    make_data(config)