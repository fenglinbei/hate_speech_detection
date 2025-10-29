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
    retriever.create_embeddings(raw_datas)
    
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
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in messages[split_idx:]:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def build_rag_prompt(
        datas: list,
        srag_retriever: Retriever,
        prompt_template: str,
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        ):
    """构建RAG提示"""
    pbar = tqdm(
            total=len(datas),
            desc=f"Preprocessing datas",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )

    messages = []
    for raw_data in datas:
        triples = []
        for quadruple in raw_data["quadruples"]:
            label = quadruple["targeted_group"]
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")
        

        retrieve_contents, retrieve_outputs = srag_retriever.retrieve(raw_data['content'], srag_top_k)
        examples = []
        for retrieve_content, retrieve_output in zip(retrieve_contents, retrieve_outputs):
            example_prompt = example_template.replace("{retrieve_content}", retrieve_content).\
                                              replace("{retrieve_output}", output2triple(retrieve_output))
            examples.append(example_prompt)

            
        input = prompt_template.replace("{examples}", "\n".join(examples)).\
                                replace("{text}", raw_data["content"])

        answer = " [SEP] ".join(triples) + " [END]"
        message = {
            "id": raw_data["id"],
            "instruction": system_prompt if system_prompt else "", 
            "input": f"{input}", 
            "output": answer, 
            "content": raw_data["content"],
            "gt_quadruples": raw_data["quadruples"]
            }
        messages.append(message)
        pbar.update(1)
    return messages

def make_rag_data(
        raw_data_path: str, 
        test_data_path: str,
        train_output_path: str, 
        val_output_path: str, 
        test_output_path: str,
        prompt_template: str, 
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1
        ):
    """转换训练/验证集数据格式"""

    messages = []
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    split_idx = int(len(raw_datas) * 0.9)
    srag_retriever = Retriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5")
    
    srag_retriever.create_embeddings(raw_datas[:split_idx])


    messages = build_rag_prompt(
        datas=raw_datas[:split_idx],
        srag_retriever=srag_retriever,
        prompt_template=prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k
    )

    with open(train_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps({"instruction": system_prompt if system_prompt else "", "input": message["input"], "output": message["output"], "content": message["content"]}, ensure_ascii=False) + "\n")
    
    srag_retriever.create_embeddings(raw_datas)

    messages = build_rag_prompt(
        datas=raw_datas[split_idx:],
        srag_retriever=srag_retriever,
        prompt_template=prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k
    )
    
    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    with open(test_data_path, "r") as file:
        test_datas = json.load(file)

    messages = build_rag_prompt(
        datas=test_datas,
        srag_retriever=srag_retriever,
        prompt_template=prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k
    )

    with open(test_output_path, "w", encoding="utf-8") as file:
        json.dump([{
                "id": message["id"], 
                "content": message["content"], 
                "gt_quadruples": message.get("gt_quadruples", []), 
                "messages_list": [[{'content': system_prompt, 'role': 'system'}, {'content': message["input"], 'role': 'user'}]],
            } for message in messages], file, ensure_ascii=False, indent=4)

def build_lex_rag_prompt(
        datas: list,
        srag_retriever: Retriever,
        lex_retriever: LexiconRetriever,
        prompt_template: str,
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        lex_top_k: int = -1,
        is_test_data: bool = False
        ):
    """构建RAG提示"""
    pbar = tqdm(
            total=len(datas),
            desc=f"Preprocessing datas",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )

    messages = []

    for raw_data in datas:
        triples = []
        for quadruple in raw_data["quadruples"]:
            label = quadruple["targeted_group"]
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")
        

        retrieve_contents, retrieve_outputs = srag_retriever.retrieve(raw_data['content'], srag_top_k)
        examples = []
        for retrieve_content, retrieve_output in zip(retrieve_contents, retrieve_outputs):
            example_prompt = example_template.replace("{retrieve_content}", retrieve_content).\
                                              replace("{retrieve_output}", output2triple(retrieve_output))
            examples.append(example_prompt)

        lex_contents = lex_retriever.including_retrieve(raw_data['content'], lex_top_k)
            
        input = prompt_template.replace("{examples}", "\n".join(examples)).\
                                replace("{lexicons}", "\n".join(lex_contents)).\
                                replace("{text}", raw_data["content"])

        answer = " [SEP] ".join(triples) + " [END]"
        message = {
            "id": raw_data["id"],
            "instruction": system_prompt if system_prompt else "", 
            "input": f"{input}", 
            "output": answer, 
            "content": raw_data["content"],
            "gt_quadruples": raw_data["quadruples"] if  is_test_data else ""
            }
        messages.append(message)
        pbar.update(1)
    return messages

def make_lexcion_rag_data(
        raw_data_path: str, 
        test_data_path: str,
        train_output_path: str, 
        val_output_path: str, 
        test_output_path: str,
        prompt_template: str, 
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        lex_top_k: int = -1
        ):
    """转换训练/验证集数据格式"""

    messages = []
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    split_idx = int(len(raw_datas) * 0.9)
    srag_retriever = Retriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5")
    lex_retriever = LexiconRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/lexicon/annotated_lexicon.json")
    
    srag_retriever.create_embeddings(raw_datas[:split_idx])
 
    messages = build_lex_rag_prompt(
        datas=raw_datas[:split_idx],
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        prompt_template=prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        lex_top_k=lex_top_k
    )
    
    with open(train_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    messages = build_lex_rag_prompt(
        datas=raw_datas[split_idx:],
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        prompt_template=prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        lex_top_k=lex_top_k
    )

    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    with open(test_data_path, "r") as file:
        test_datas = json.load(file)

    messages = build_lex_rag_prompt(
        datas=test_datas,
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        prompt_template=prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        lex_top_k=lex_top_k,
        is_test_data=True
    )

    with open(test_output_path, "w", encoding="utf-8") as file:
        json.dump([{
                "id": message["id"], 
                "content": message["content"], 
                "gt_quadruples": message.get("gt_quadruples", []), 
                "messages_list": [[{'content': system_prompt, 'role': 'system'}, {'content': message["input"], 'role': 'user'}]],
            } for message in messages], file, ensure_ascii=False, indent=4)

def build_simlex_rag_prompt(
        datas: list,
        srag_retriever: Retriever,
        lex_retriever: LexiconRetriever,
        prompt_template: str,
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        lex_top_k: int = -1,
        lex_sim_top_k: int = -1,
        is_test_data: bool = False
        ):
    """构建相似词典检索的RAG提示模板"""
    pbar = tqdm(
            total=len(datas),
            desc=f"Preprocessing datas",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )
    messages = []
    for raw_data in datas:
        triples = []
        for quadruple in raw_data["quadruples"]:
            label = quadruple["targeted_group"]
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")
        

        retrieve_contents, retrieve_outputs = srag_retriever.retrieve(raw_data['content'], srag_top_k)
        examples = []
        for retrieve_content, retrieve_output in zip(retrieve_contents, retrieve_outputs):
            example_prompt = example_template.replace("{retrieve_content}", retrieve_content).\
                                              replace("{retrieve_output}", output2triple(retrieve_output))
            examples.append(example_prompt)

        lex_contents = lex_retriever.including_retrieve(raw_data['content'], lex_top_k)
        simlex_contents = lex_retriever.similarity_retrieve(raw_data['content'], lex_sim_top_k, deduplicate=True)
        for simlex_content in simlex_contents:
            if simlex_content not in lex_contents:
                lex_contents.append(simlex_content)
            
        input = prompt_template.replace("{examples}", "\n".join(examples)).\
                                replace("{lexicons}", "\n".join(lex_contents)).\
                                replace("{text}", raw_data["content"])

        answer = " [SEP] ".join(triples) + " [END]"
        message = {
            "id": raw_data["id"],
            "instruction": system_prompt if system_prompt else "", 
            "input": f"{input}", 
            "output": answer, 
            "content": raw_data["content"],
            "gt_quadruples": raw_data["quadruples"] if  is_test_data else ""
            }
        messages.append(message)
        pbar.update(1)

    return messages

def make_sim_lexcion_rag_data(
        raw_data_path: str, 
        test_data_path: str,
        train_output_path: str, 
        val_output_path: str, 
        test_output_path: str,
        prompt_template: str, 
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        lex_top_k: int = -1,
        lex_sim_top_k: int = -1
        ):
    """转换训练/验证集数据格式"""

    messages = []
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    split_idx = int(len(raw_datas) * 0.9)
    srag_retriever = Retriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5")
    lex_retriever = LexiconRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/lexicon/annotated_lexicon.json")
    
    srag_retriever.create_embeddings(raw_datas[:split_idx])
    
    messages = build_simlex_rag_prompt(
        datas=raw_datas[:split_idx],
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        prompt_template=prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        lex_top_k=lex_top_k,
        lex_sim_top_k=lex_sim_top_k
    )

    with open(train_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    srag_retriever.create_embeddings(raw_datas)

    messages = build_simlex_rag_prompt(
        datas=raw_datas[split_idx:],
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        prompt_template=prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        lex_top_k=lex_top_k,
        lex_sim_top_k=lex_sim_top_k
    )
    
    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    with open(test_data_path, "r") as file:
        test_datas = json.load(file)

    messages = build_simlex_rag_prompt(
        datas=test_datas,
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        prompt_template=prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        lex_top_k=lex_top_k,
        lex_sim_top_k=lex_sim_top_k
    )

    with open(test_output_path, "w", encoding="utf-8") as file:
        json.dump([{
                "id": message["id"], 
                "content": message["content"], 
                "gt_quadruples": message.get("gt_quadruples", []), 
                "messages_list": [[{'content': system_prompt, 'role': 'system'}, {'content': message["input"], 'role': 'user'}]],
            } for message in messages], file, ensure_ascii=False, indent=4)

def make_no_rag_data(
        train_data_path: str, 
        test_data_path: str, 
        train_output_path: str, 
        val_output_path: str, 
        test_output_path: str,
        prompt_template: str, 
        system_prompt: Optional[str] = None
        ):
    
    messages = []
    with open(train_data_path, "r") as file:
        train_datas = json.load(file)

    split_idx = int(len(train_datas) * 0.9)
    
    pbar = tqdm(
            total=len(train_datas),
            desc=f"Preprocessing train datas",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )

    for train_data in train_datas:
        triples = []
        for quadruple in train_data["quadruples"]:
            label = quadruple["targeted_group"]
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")

        input = prompt_template.format(text=train_data["content"])
        

        answer = " [SEP] ".join(triples) + " [END]"
        message = {"instruction": system_prompt if system_prompt else "", "input": f"{input}", "output": answer, "content": train_data["content"]}
        messages.append(message)
        pbar.update(1)
    
    with open(train_output_path, "w", encoding="utf-8") as file:
        for message in messages[:split_idx]:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in messages[split_idx:]:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    test_messages = []
    with open(test_data_path, "r") as file:
        test_datas = json.load(file)
    
    pbar = tqdm(
            total=len(test_datas),
            desc=f"Preprocessing test datas",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )

    for test_data in test_datas:
        triples = []
        for quadruple in test_data["quadruples"]:
            label = quadruple["targeted_group"]
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")

        prompt = prompt_template.format(text=test_data["content"])
        

        answer = " [SEP] ".join(triples) + " [END]"
        message = {
            "id": test_data["id"],
            "instruction": system_prompt if system_prompt else "", 
            "input": f"{prompt}", 
            "output": answer, 
            "content": test_data["content"],
            "gt_quadruples": test_data["quadruples"]
            }
        test_messages.append(message)
        pbar.update(1)
    
    with open(test_output_path, "w", encoding="utf-8") as file:
        json.dump([{
                "id": test_message["id"], 
                "content": test_message["content"], 
                "gt_quadruples": test_message.get("gt_quadruples", []), 
                "messages_list": [[{'content': system_prompt, 'role': 'system'}, {'content': test_message["input"], 'role': 'user'}]],
            } for test_message in test_messages], file, ensure_ascii=False, indent=4)

def build_sim_lexcion_threshold_prompt(
        datas: list,
        srag_retriever: Retriever,
        lex_retriever: LexiconRetriever,
        prompt_template: str,
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        srag_threshold: float = 0,
        lex_top_k: int = -1,
        lex_sim_top_k: int = -1,
        lex_sim_threshold: float = 0,
        rerank: bool = False,
        resort: bool = False,
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
    srag_examples_nums = 0
    for raw_data in datas:
        triples = []
        for quadruple in raw_data["quadruples"]:
            label = quadruple["targeted_group"]
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")
        

        retrieve_contents, retrieve_outputs = srag_retriever.retrieve(raw_data['content'], srag_top_k, threshold=srag_threshold, rerank=rerank, resort=resort)
        print(len(retrieve_contents))
        examples = []
        for retrieve_content, retrieve_output in zip(retrieve_contents, retrieve_outputs):
            example_prompt = example_template.replace("{retrieve_content}", retrieve_content).\
                                              replace("{retrieve_output}", output2triple(retrieve_output))
            examples.append(example_prompt)

        lex_contents = lex_retriever.including_retrieve(raw_data['content'], lex_top_k)
        simlex_contents = lex_retriever.similarity_retrieve(raw_data['content'], lex_sim_top_k, deduplicate=True, threshold=lex_sim_threshold)
        for simlex_content in simlex_contents:
            if simlex_content not in lex_contents:
                lex_contents.append(simlex_content)
        
        srag_examples_nums += len(examples)
        prompt = prompt_template.replace("{examples}", "\n".join(examples)).\
                                replace("{lexicons}", "\n".join(lex_contents)).\
                                replace("{text}", raw_data["content"])
        
        if not examples:
            prompt.replace("示例：\n\n", "")

        if not lex_contents:
            prompt = prompt.replace("背景知识：\n\n", "")

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
    
    print(f"SRAG avg examples nums: {srag_examples_nums / len(datas)}")

    return messages

def make_sim_lexcion_threshold_rag_data(
        raw_data_path: str, 
        test_data_path: str,
        train_output_path: str, 
        val_output_path: str, 
        test_output_path: str,
        prompt_template: str, 
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        srag_threshold: float = 0,
        lex_top_k: int = -1,
        lex_sim_top_k: int = -1,
        lex_sim_threshold: float = 0,
        rerank: bool = False,
        resort: bool = False
        ):
    """转换训练/验证集数据格式"""

    messages = []
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    split_idx = int(len(raw_datas) * 0.9)
    srag_retriever = Retriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", reranker_model_path="./models/Qwen3-Reranker-0.6B" if rerank else None)
    lex_retriever = LexiconRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/lexicon/annotated_lexicon.json")
    
    srag_retriever.create_embeddings(raw_datas[:split_idx])

    messages = build_sim_lexcion_threshold_prompt(
        raw_datas[:split_idx],
        srag_retriever,
        lex_retriever,
        prompt_template,
        example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        srag_threshold=srag_threshold,
        lex_top_k=lex_top_k,
        lex_sim_top_k=lex_sim_top_k,
        lex_sim_threshold=lex_sim_threshold,
        rerank=rerank,
        resort=resort
    )

    # examples = random.sample(messages, k=int(len(messages) * 0.01))
    # for i in examples:
    #     print(i['input'])
    
    with open(train_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    srag_retriever.create_embeddings(raw_datas)

    messages = build_sim_lexcion_threshold_prompt(
        raw_datas[split_idx:],
        srag_retriever,
        lex_retriever,
        prompt_template,
        example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        srag_threshold=srag_threshold,
        lex_top_k=lex_top_k,
        lex_sim_top_k=lex_sim_top_k,
        lex_sim_threshold=lex_sim_threshold,
        rerank=rerank,
        resort=resort
    )

    # examples = random.sample(messages, k=10)
    # for i in examples:
    #     print(i['input'])

    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    with open(test_data_path, "r") as file:
        test_datas = json.load(file)

    messages = build_sim_lexcion_threshold_prompt(
        datas=test_datas,
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        prompt_template=prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        lex_top_k=lex_top_k,
        lex_sim_top_k=lex_sim_top_k,
        is_test_data=True,
        rerank=rerank,
        resort=resort
    )

    with open(test_output_path, "w", encoding="utf-8") as file:
        json.dump([{
                "id": message["id"], 
                "content": message["content"], 
                "gt_quadruples": message.get("gt_quadruples", []), 
                "messages_list": [[{'content': system_prompt, 'role': 'system'}, {'content': message["input"], 'role': 'user'}]],
            } for message in messages], file, ensure_ascii=False, indent=4)

def build_multi_class_sim_lexcion_threshold_prompt(
        datas: list,
        srag_retriever: MultiClassRetriever,
        lex_retriever: LexiconRetriever,
        prompt_template: str,
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        srag_threshold: float = 0,
        lex_top_k: int = -1,
        lex_sim_top_k: int = -1,
        lex_sim_threshold: float = 0,
        weights: Optional[dict] = None,
        weights_reverse: bool = False,
        auto_length: bool = True,
        model_path: Optional[str] = None,
        max_length: int = 2048,
        is_test_data: bool = False
        ):
    """构建相似词典检索的提示模板"""

    def build_prompt(
            raw_data, 
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
            weights_reverse):
        
        retrieve_contents, retrieve_outputs = srag_retriever.retrieve(raw_data['content'], srag_top_k, threshold=srag_threshold, weights=weights, weights_reverse=weights_reverse)
        examples = []
        for retrieve_content, retrieve_output in zip(retrieve_contents, retrieve_outputs):
            example_prompt = example_template.replace("{retrieve_content}", retrieve_content).\
                                                replace("{retrieve_output}", output2triple(retrieve_output))
            examples.append(example_prompt)

        lex_contents = lex_retriever.including_retrieve(raw_data['content'], lex_top_k)
        simlex_contents = lex_retriever.similarity_retrieve(raw_data['content'], lex_sim_top_k, deduplicate=True, threshold=lex_sim_threshold)
        for simlex_content in simlex_contents:
            if simlex_content not in lex_contents:
                lex_contents.append(simlex_content)
        
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

def make_multi_class_sim_lexcion_threshold_rag_data(
        raw_data_path: str, 
        test_data_path: str,
        train_output_path: str, 
        val_output_path: str, 
        test_output_path: str,
        prompt_template: str, 
        example_template: str,
        system_prompt: Optional[str] = None,
        srag_top_k: int = 1,
        srag_threshold: float = 0,
        lex_top_k: int = -1,
        lex_sim_top_k: int = -1,
        lex_sim_threshold: float = 0,
        weights: Optional[dict] = None,
        weights_reverse: bool = False,
        auto_length=True,
        model_path: Optional[str] = None,
        max_length: int = 2048,
        ):
    """转换训练/验证集数据格式"""

    messages = []
    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)

    split_idx = int(len(raw_datas) * 0.9)
    srag_retriever = MultiClassRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5")
    lex_retriever = LexiconRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/lexicon/annotated_lexicon.json")
    
    srag_retriever.load_datas(data_list=raw_datas[:split_idx])
    srag_retriever.build_retrievers()

    messages = build_multi_class_sim_lexcion_threshold_prompt(
        raw_datas[:split_idx],
        srag_retriever,
        lex_retriever,
        prompt_template,
        example_template,
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

    # examples = random.sample(messages, k=int(len(messages) * 0.01))
    # for i in examples:
    #     print(i['input'])
    
    with open(train_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    srag_retriever.load_datas(data_list=raw_datas)
    srag_retriever.build_retrievers()

    messages = build_multi_class_sim_lexcion_threshold_prompt(
        raw_datas[split_idx:],
        srag_retriever,
        lex_retriever,
        prompt_template,
        example_template,
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

    # examples = random.sample(messages, k=10)
    # for i in examples:
    #     print(i['input'])

    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    with open(test_data_path, "r") as file:
        test_datas = json.load(file)

    messages = build_multi_class_sim_lexcion_threshold_prompt(
        datas=test_datas,
        srag_retriever=srag_retriever,
        lex_retriever=lex_retriever,
        prompt_template=prompt_template,
        example_template=example_template,
        system_prompt=system_prompt,
        srag_top_k=srag_top_k,
        lex_top_k=lex_top_k,
        lex_sim_top_k=lex_sim_top_k,
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
        
def build_n_step_multi_class_sim_lexcion_threshold_prompt(
        datas: list,
        srag_retriever: MultiClassWrongExpRetriever,
        lex_retriever: LexiconRetriever,
        prompt_template: str,
        example_template: str,
        wrong_exp_template: str,
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
            raw_data, 
            srag_retriever, 
            lex_retriever, 
            prompt_template, 
            example_template, 
            wrong_exp_template, 
            srag_top_k, 
            srag_threshold, 
            lex_top_k, 
            lex_sim_top_k, 
            lex_sim_threshold, 
            weights, 
            weights_reverse):
        
        retrieve_contents, retrieve_outputs, wrong_exps = srag_retriever.retrieve(raw_data['content'], srag_top_k, threshold=srag_threshold, weights=weights, weights_reverse=weights_reverse)
        examples = []
        for retrieve_content, retrieve_output, wrong_exp in zip(retrieve_contents, retrieve_outputs, wrong_exps):
            if wrong_exp:
                example_prompt = wrong_exp_template.replace("{retrieve_content}", retrieve_content).\
                                                    replace("{retrieve_output}", output2triple(retrieve_output)).\
                                                    replace("{retrieve_wrong_exp}", wrong_exp)
            else:
                example_prompt = example_template.replace("{retrieve_content}", retrieve_content).\
                                                replace("{retrieve_output}", output2triple(retrieve_output))
            examples.append(example_prompt)

        lex_contents = lex_retriever.including_retrieve(raw_data['content'], lex_top_k)
        simlex_contents = lex_retriever.similarity_retrieve(raw_data['content'], lex_sim_top_k, deduplicate=True, threshold=lex_sim_threshold)
        for simlex_content in simlex_contents:
            if simlex_content not in lex_contents:
                lex_contents.append(simlex_content)
        
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
            srag_retriever,
            lex_retriever,
            prompt_template,
            example_template,
            wrong_exp_template,
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
                wrong_exp_template,
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

def make_n_step_multi_class_sim_lexcion_threshold_rag_data(
        raw_data_path: str, 
        test_data_path: str,
        train_output_path: str, 
        val_output_path: str, 
        test_output_path: str,
        prompt_template: str, 
        example_template: str,
        wrong_exp_template: str,
        system_prompt: Optional[str] = None,
        step: int = 1,
        total_step: int = 2,
        full_data: bool = False,
        test_data: bool = False,
        last_output_data_path_list: Optional[list[str]] = None,
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
        ):
    """转换训练/验证集数据格式"""

    with open(raw_data_path, "r") as file:
        raw_datas = json.load(file)
    
    total_length = len(raw_datas)
    data_list = []
    start_index = end_index = 0
    if not full_data:
        start_index = int((step - 1) * total_length / total_step)
        end_index = int(step * total_length / total_step)
        data_list = raw_datas[start_index:end_index] # 训练与验证集数据
    elif full_data:
        start_index = 0
        end_index = total_length
        data_list = raw_datas[start_index:end_index] # 训练与验证集数据

    if step == 1 and not full_data:
        result_data_list = []
    else:
        assert last_output_data_path_list is not None, "last_output_data_path must be provided for step > 1"
        result_data_list = []
        for last_output_data_path in last_output_data_path_list:
            with open(last_output_data_path, "r") as file:
                result_data_list.extend(json.load(file)["results"])

    split_idx = int(len(data_list) * 0.9) # 训练与验证集划分点
    train_data_list = data_list[:split_idx]
    val_data_list = data_list[split_idx:]

    srag_data_list = data_list[:split_idx] + raw_datas[:start_index] # srag可见范围为当前训练集+之前所有数据
    srag_result_data_list = result_data_list # 错例范围为先前所有验证集推理数据
    srag_retriever = MultiClassWrongExpRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_list=srag_data_list, result_data_list=srag_result_data_list)
    lex_retriever = LexiconRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/lexicon/annotated_lexicon.json")

    total_sentence_length = 0

    train_messages = build_n_step_multi_class_sim_lexcion_threshold_prompt(
        train_data_list,
        srag_retriever,
        lex_retriever,
        prompt_template,
        example_template,
        wrong_exp_template,
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
        max_length=max_length
    )

    if len(train_messages) > 0:
        total_sentence_length = sum([len(message['input']) for message in train_messages])
        avg_sentence_length = total_sentence_length / len(train_messages)
        print(f"Avg train input sentence length: {avg_sentence_length}")
    
    with open(train_output_path, "w", encoding="utf-8") as file:
        for message in train_messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    val_messages = build_n_step_multi_class_sim_lexcion_threshold_prompt(
        val_data_list,
        srag_retriever,
        lex_retriever,
        prompt_template,
        example_template,
        wrong_exp_template,
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
        max_length=max_length
    )

    with open(val_output_path, "w", encoding="utf-8") as file:
        for message in val_messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    with open(test_data_path, "r") as file:
        test_datas = json.load(file)

    # 测试集srag可见范围为完整整训练集，错例范围为所有验证集推理数据
    if not test_data:
        test_messages = build_n_step_multi_class_sim_lexcion_threshold_prompt(
            val_data_list,
            srag_retriever,
            lex_retriever,
            prompt_template,
            example_template,
            wrong_exp_template,
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
    else:
        srag_retriever = MultiClassWrongExpRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_list=raw_datas, result_data_list=result_data_list)
        test_messages = build_n_step_multi_class_sim_lexcion_threshold_prompt(
            datas=test_datas,
            srag_retriever=srag_retriever,
            lex_retriever=lex_retriever,
            prompt_template=prompt_template,
            example_template=example_template,
            wrong_exp_template=wrong_exp_template,
            system_prompt=system_prompt,
            srag_top_k=srag_top_k,
            lex_top_k=lex_top_k,
            lex_sim_top_k=lex_sim_top_k,
            weights=weights,
            weights_reverse=weights_reverse,
            auto_length=auto_length,
            model_path=model_path,
            max_length=max_length,
            is_test_data=True
        )

    with open(test_output_path, "w", encoding="utf-8") as file:
        json.dump([{
                "id": test_message["id"], 
                "content": test_message["content"], 
                "gt_quadruples": test_message.get("gt_quadruples", []), 
                "messages_list": [[{'content': system_prompt, 'role': 'system'}, {'content': test_message["input"], 'role': 'user'}]],
            } for test_message in test_messages], file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # dataset_transfer_no_think("data/full/std/train.json", "finetune/data/train_full.jsonl", "finetune/data/val.jsonl", RAG_PROMPT_USER_V1, system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT)
    # make_lexcion_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     train_output_path="finetune/data/train_lex_rag_5.jsonl", 
    #     val_output_path="finetune/data/val_lex_rag_5.jsonl",
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     srag_top_k=5,
    #     lex_top_k=-1)

    #     make_no_rag_data(
    #         train_data_path="data/full/std/train.json", 
    #         test_data_path="data/full/std/test.json",
    #         train_output_path="finetune/data/complex_prompt/train.jsonl", 
    #         val_output_path="finetune/data/complex_prompt/val.jsonl",
    #         test_output_path="finetune/data/complex_prompt/test.json",
    #         prompt_template=TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V3,
    #         system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT
    #         )
    
    # make_no_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     train_output_path="finetune/data/train_no_rag_nosys.jsonl", 
    #     val_output_path="finetune/data/val_no_rag_nosys.jsonl",
    #     prompt_template=RAG_PROMPT_USER_V3,
    #     system_prompt=""
    #     )

    #     make_rag_data(
    #         raw_data_path="data/full/std/train.json", 
    #         test_data_path="data/full/std/test.json",
    #         train_output_path="finetune/data/one_shot_prompt/train.jsonl", 
    #         val_output_path="finetune/data/one_shot_prompt/val.jsonl",
    #         test_output_path="finetune/data/one_shot_prompt/test.json",
    #         prompt_template=RAG_PROMPT_USER_V4,
    #         example_template=RAG_PROMPT_EXAMPLE_V2,
    #         system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #         srag_top_k=1)
    
    # make_lexcion_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/lex_rag5/train.jsonl", 
    #     val_output_path="finetune/data/lex_rag5/val.jsonl",
    #     test_output_path="finetune/data/lex_rag5/test.json",
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     srag_top_k=5,
    #     lex_top_k=-1)

    # make_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag1/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag1/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag1/test.json",
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     srag_top_k=1,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0)

    # make_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag5/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag5/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag5/test.json",
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     srag_top_k=5,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0)
    
    # make_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag5_threshold05/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag5_threshold05/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag5_threshold05/test.json",
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     srag_top_k=5,
    #     srag_threshold=0.5,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0.5)
    
    # make_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag3/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag3/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag3/test.json",
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     srag_top_k=3,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0)
    
    # make_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag7/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag7/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag7/test.json",
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     srag_top_k=7,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0)

    # make_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag13/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag13/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag13/test.json",
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     srag_top_k=13,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0)

    # make_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag9_rerank/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag9_rerank/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag9_rerank/test.json",
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     srag_top_k=9,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0,
    #     rerank=True,
    #     resort=False)
    
    # make_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag9_rerank_resort/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag9_rerank_resort/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag9_rerank_resort/test.json",
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     srag_top_k=9,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0,
    #     rerank=True,
    #     resort=True)

    #     make_multi_class_sim_lexcion_threshold_rag_data(
    #         raw_data_path="data/full/std/train.json", 
    #         test_data_path="data/full/std/test.json",
    #         train_output_path="finetune/data/simlex5_rag9_multi_class/train.jsonl", 
    #         val_output_path="finetune/data/simlex5_rag9_multi_class/val.jsonl",
    #         test_output_path="finetune/data/simlex5_rag9_multi_class/test.json",
    #         prompt_template=RAG_PROMPT_USER_V2,
    #         example_template=RAG_PROMPT_EXAMPLE_V2,
    #         system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #         srag_top_k=9,
    #         srag_threshold=0,
    #         lex_top_k=-1,
    #         lex_sim_top_k=5,
    #         lex_sim_threshold=0)
    
    make_multi_class_sim_lexcion_threshold_rag_data(
        raw_data_path="data/full/std/train.json", 
        test_data_path="data/full/std/test.json",
        train_output_path="finetune/data/ours_prompt/train.jsonl", 
        val_output_path="finetune/data/ours_prompt/val.jsonl",
        test_output_path="finetune/data/ours_prompt/test.json",
        prompt_template=RAG_PROMPT_USER_V2,
        example_template=RAG_PROMPT_EXAMPLE_V2,
        system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
        srag_top_k=11,
        srag_threshold=0,
        lex_top_k=-1,
        lex_sim_top_k=5,
        lex_sim_threshold=0,
        auto_length=True,
        max_length=2048,
        model_path="models/Qwen2.5-7B-Instruct")

    # make_n_step_multi_class_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_1_autolength/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_1_autolength/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_1_autolength/test.json",
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     wrong_exp_template=RAG_PROMPT_EXAMPLE_V3,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     step=1,
    #     total_step=2,
    #     srag_top_k=9,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0,
    #     auto_length=True,
    #     model_path="models/Qwen2.5-7B-Instruct",
    #     max_length=1536)
    
    # make_n_step_multi_class_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_2_autolength/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_2_autolength/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_2_autolength/test.json",
    #     last_output_data_path_list=["runner/output/simlex5_rag9_multi_class_nstep2_1_autolength.json"],
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     wrong_exp_template=RAG_PROMPT_EXAMPLE_V3,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     step=2,
    #     total_step=2,
    #     srag_top_k=9,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0,
    #     auto_length=True,
    #     model_path="models/Qwen2.5-7B-Instruct",
    #     max_length=1280)

    # make_n_step_multi_class_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_test/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_test/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_test/test.json",
    #     last_output_data_path_list=["runner/output/simlex5_rag9_multi_class_nstep2_1_autolength.json",
    #                                 "runner/output/simlex5_rag9_multi_class_nstep2_2_autolength.json"],
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     wrong_exp_template=RAG_PROMPT_EXAMPLE_V3,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     step=3,
    #     total_step=2,
    #     srag_top_k=9,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0,
    #     auto_length=True,
    #     model_path="models/Qwen2.5-7B-Instruct",
    #     max_length=1280)
    
    # make_n_step_multi_class_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_full/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_full/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag9_multi_class_nstep2_full/test.json",
    #     last_output_data_path_list=["runner/output/simlex5_rag9_multi_class_nstep2_1_autolength.json",
    #                                 "runner/output/simlex5_rag9_multi_class_nstep2_2_autolength.json"],
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     wrong_exp_template=RAG_PROMPT_EXAMPLE_V3,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     full_data=True,
    #     test_data=True,
    #     srag_top_k=9,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0,
    #     auto_length=True,
    #     model_path="models/Qwen2.5-7B-Instruct",
    #     max_length=1280)

    # make_n_step_multi_class_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag11_multi_class_wexp_loop_1/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag11_multi_class_wexp_loop_1/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag11_multi_class_wexp_loop_1/test.jsonl",
    #     last_output_data_path_list=[],
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     wrong_exp_template=RAG_PROMPT_EXAMPLE_V3,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     full_data=True,
    #     test_data=False,
    #     srag_top_k=11,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0,
    #     auto_length=True,
    #     model_path="models/Qwen2.5-7B-Instruct",
    #     max_length=1280)

    # make_n_step_multi_class_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag11_multi_class_wexp/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag11_multi_class_wexp/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag11_multi_class_wexp/test.jsonl",
    #     last_output_data_path_list=["runner/output/simlex5_rag11_multi_class_wexp_loop_1.json"],
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     wrong_exp_template=RAG_PROMPT_EXAMPLE_V3,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     full_data=True,
    #     test_data=True,
    #     srag_top_k=11,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0,
    #     auto_length=True,
    #     model_path="models/Qwen2.5-7B-Instruct",
    #     max_length=1280)

    # make_n_step_multi_class_sim_lexcion_threshold_rag_data(
    #     raw_data_path="data/full/std/train.json", 
    #     test_data_path="data/full/std/test.json",
    #     train_output_path="finetune/data/simlex5_rag9_multi_class_al/train.jsonl", 
    #     val_output_path="finetune/data/simlex5_rag9_multi_class_al/val.jsonl",
    #     test_output_path="finetune/data/simlex5_rag9_multi_class_al/test.json",
    #     last_output_data_path_list=[],
    #     prompt_template=RAG_PROMPT_USER_V2,
    #     example_template=RAG_PROMPT_EXAMPLE_V2,
    #     wrong_exp_template=RAG_PROMPT_EXAMPLE_V3,
    #     system_prompt=QWEN2_DEFAULT_SYSTEM_PROMPT,
    #     full_data=True,
    #     test_data=True,
    #     srag_top_k=9,
    #     srag_threshold=0,
    #     lex_top_k=-1,
    #     lex_sim_top_k=5,
    #     lex_sim_threshold=0,
    #     auto_length=True,
    #     model_path="models/Qwen2.5-7B-Instruct",
    #     max_length=1280)