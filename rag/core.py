
import math
from loguru import logger
from typing import Optional
from tools.json_tools import load_json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

from prompt import *
from rag.reranker import Reranker
from tools.convert import output2triple, parsed_quad_to_raw_quad, parsed_quad_to_tar_and_arg

TARGETED_GROUPS = ["non-hate", "Region", "Racism", "Sexism", "LGBTQ", "others"]
DEFAULT_WEIGHTS = {
    "non-hate": 36.9,
    "Region": 13.8,
    "Racism": 12.9,
    "Sexism": 17.1,
    "LGBTQ": 6.7,
    "others": 12.6
}

def allocate_class_num(i, weights_dict, reverse: bool = False):
    # 初始化字典用于存储初始分配值和小数部分
    initial_allocation = {}
    fractions = []
    total_integer = 0

    if reverse:
        weights_dict = {k: 1/v for k, v in weights_dict.items()}
        total_weight = sum(weights_dict.values())
        weights_dict = {k: (v / total_weight) * 100 for k, v in weights_dict.items()}
    
    # 遍历权重字典，计算每个类别的理论值、整数部分和小数部分
    for key, weight in weights_dict.items():
        theory_value = i * weight / 100.0
        integer_part = math.floor(theory_value)
        fraction = theory_value - integer_part
        
        initial_allocation[key] = integer_part
        fractions.append((key, fraction))
        total_integer += integer_part
    
    # 计算剩余量（需要分配的额外单位数）
    remaining = i - total_integer
    
    # 根据小数部分降序排序（小数部分相同则按键名字母顺序升序）
    fractions.sort(key=lambda x: (-x[1], x[0]))
    
    # 将剩余量分配给小数部分最大的前 remaining 个类别
    for idx in range(remaining):
        key = fractions[idx][0]
        initial_allocation[key] += 1
    
    return initial_allocation

class Retriever:

    def __init__(
            self, 
            model_path: str, 
            model_name: str, 
            data_path: Optional[str] = None, 
            reranker_model_path: Optional[str] = None,
            device: str = "cuda:0"):

        logger.info(f"Loading model from path: {model_path}")
        self.model = SentenceTransformer(model_path).to(device)
        self.model_name = model_name

        self.reranker = None
        if reranker_model_path:
            self.reranker = Reranker(model_path=reranker_model_path)

        if data_path:
            self.load_datas(data_path)
            self.create_embeddings()

    def create_embeddings(self, datas: Optional[list[dict]] = None):
        logger.info("Processing embedding")
        if not datas:
            texts = self.texts
        else:
            self.texts = [item['content'] for item in datas]
            self.test2item = {}
            for item in datas:
                item['output'] = parsed_quad_to_raw_quad(item['quadruples'])
                self.test2item[item['content']] = item
            texts = self.texts

        corpus_embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        if corpus_embeddings.is_cuda:
            corpus_embeddings = corpus_embeddings.cpu()
        self.corpus_embeddings_np = corpus_embeddings.numpy()

    def load_datas(self, data_path: str):
        data = load_json(data_path)
        self.texts = [item['content'] for item in data]
        self.test2item = {}
        for item in data:
            item['output'] = parsed_quad_to_raw_quad(item['quadruples'])
            self.test2item[item['content']] = item
    
    def retrieve(
            self, 
            query: str, 
            top_k: int = 1, 
            deduplicate: bool = True, 
            threshold: float = 0,
            rerank: bool = False,
            resort: bool = False,
            ) -> tuple[list[str], list[str]]:
        if top_k == 0:
            return [], []
        
        if rerank and self.reranker:
            new_top_k = top_k * 5
            if new_top_k < 10:
                new_top_k = 10
            elif new_top_k > 100:   
                new_top_k = 100
        else:
            new_top_k = top_k

        query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        
        if query_embedding.is_cuda:
            query_embedding = query_embedding.cpu()
            
        query_embedding_np = query_embedding.numpy().reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding_np, self.corpus_embeddings_np)[0]
        
        # 获取所有索引并按相似度排序
        sorted_indices = np.argsort(similarities)[::-1]
        
        unique_texts = []
        unique_outputs = []
        seen_contents = set() if not deduplicate else set([query])  # 用于追踪已处理的内容
        
        # 遍历所有排序后的索引
        for idx in sorted_indices:
            content = self.texts[idx]
            sim_score = similarities[idx]  # 获取当前相似度分数
            
            # 阈值过滤：如果相似度低于阈值则跳过
            if sim_score < threshold:
                continue  # 跳过低于阈值的结果
                
            # 去重逻辑
            if deduplicate:
                if content in seen_contents:
                    continue  # 已处理过相同内容，跳过
                seen_contents.add(content)
            
            unique_texts.append(content)
            unique_outputs.append(self.test2item[content]['output'])
            
            # 达到需要的 top_k 数量时停止
            if len(unique_texts) >= new_top_k:
                break

        if rerank and self.reranker:
            scores = self.reranker.rerank(query, unique_texts)
            sorted_indices = np.argsort(scores)[::-1][:top_k]
            unique_texts = [unique_texts[i] for i in sorted_indices]
            unique_outputs = [unique_outputs[i] for i in sorted_indices]

        if resort:
            resorted_indices = [0] * len(unique_texts)
            resorted_indices = [0] * len(unique_texts)
            l = 0
            r = len(unique_texts) - 1
            for i in range(len(unique_texts)):
                if i % 2 == 0:
                    resorted_indices[l] = i
                    l += 1
                else:
                    resorted_indices[r] = i
                    r -= 1
                    
            unique_texts = [unique_texts[i] for i in resorted_indices]
            unique_outputs = [unique_outputs[i] for i in resorted_indices]

        return unique_texts, unique_outputs
    
class LexiconRetriever:

    def __init__(self, model_path: str, model_name: str, data_path: str | None = None, device: str = "cuda:0"):
        logger.info(f"Loading model from path: {model_path}")
        self.model = SentenceTransformer(model_path).to(device)
        self.model_name = model_name

        if data_path:
            self.load_datas(data_path)
            self.create_embeddings()

    def create_embeddings(self, datas: Optional[list[dict]] = None):
        logger.info("Processing embedding")
        if not datas:
            texts = self.texts
        else:
            self.texts = [item['content'] for item in datas]
            self.test2item = {}
            for item in datas:
                item['output'] = parsed_quad_to_raw_quad(item['quadruples'])
                self.test2item[item['content']] = item
            texts = self.texts

        corpus_embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        if corpus_embeddings.is_cuda:
            corpus_embeddings = corpus_embeddings.cpu()
        self.corpus_embeddings_np = corpus_embeddings.numpy()

    def load_datas(self, data_path: str):
        datas = load_json(data_path)["terms"]
        self.texts = []
        self.word2item = {}
        for data in datas:
            prompt = LEXICON_RAG_PROMPT.replace("{word}", data["term"]).\
                                        replace("{category}", data["category"]).\
                                        replace("{definition}", data["definition"])
            self.texts.append(prompt)
            self.word2item[data["term"]] = prompt
        
    
    def similarity_retrieve(self, query: str, top_k: int = 1, deduplicate: bool = True, threshold: float = 0) -> list[str]:
        if top_k == 0:
            return []
        query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)

        if query_embedding.is_cuda:
            query_embedding = query_embedding.cpu()

        query_embedding_np = query_embedding.numpy().reshape(1, -1)

        similarities = cosine_similarity(query_embedding_np, self.corpus_embeddings_np)[0]

        # 获取所有索引并按相似度排序
        sorted_indices = np.argsort(similarities)[::-1]

        unique_texts = []
        seen_contents = set() if not deduplicate else set([query])  # 用于追踪已处理的内容
        
        # 遍历所有排序后的索引
        for idx in sorted_indices:
            content = self.texts[idx]
            sim_score = similarities[idx]  # 获取当前相似度分数
            
            # 阈值过滤：如果相似度低于阈值则跳过
            if sim_score < threshold:
                continue  # 跳过低于阈值的结果
            
            # 去重逻辑
            if deduplicate:
                if content in seen_contents:
                    continue  # 已处理过相同内容，跳过
                seen_contents.add(content)
            
            unique_texts.append(content)
            
            # 达到需要的 top_k 数量时停止
            if len(unique_texts) >= top_k:
                break

        return unique_texts
    
    def including_retrieve(self, query: str, top_k: int = -1, deduplicate: bool = True) -> list[str]:
        result = []
        for word in self.word2item.keys():
            if word in query:
                result.append(self.word2item[word])
        
        return result if top_k == -1 else result[:top_k]

class StepOneRetriever:

    def __init__(self, model_path: str, model_name: str, data_path: Optional[str]=None, device: str = "cuda:0"):

        logger.info(f"Loading model from path: {model_path}")
        self.model = SentenceTransformer(model_path).to(device)
        self.model_name = model_name

        if data_path:
            self.load_datas(data_path)
            self.create_embeddings()

    def create_embeddings(self, datas: Optional[list[dict]] = None):
        logger.info("Processing embedding")
        if not datas:
            texts = self.texts
        else:
            self.texts = [item['content'] for item in datas]
            self.test2item = {}
            for item in datas:
                item['output'] = parsed_quad_to_tar_and_arg(item['quadruples'])
                self.test2item[item['content']] = item
            texts = self.texts

        corpus_embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        if corpus_embeddings.is_cuda:
            corpus_embeddings = corpus_embeddings.cpu()
        self.corpus_embeddings_np = corpus_embeddings.numpy()

    def load_datas(self, data_path: str):
        data = load_json(data_path)
        self.texts = [item['content'] for item in data]
        self.test2item = {}
        for item in data:
            item['output'] = parsed_quad_to_tar_and_arg(item['quadruples'])
            self.test2item[item['content']] = item
    
    def retrieve(self, query: str, top_k: int = 1, deduplicate: bool = True, threshold: float = 0) -> tuple[list[str], list[str]]:
        if top_k == 0:
            return [], []
        query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        
        if query_embedding.is_cuda:
            query_embedding = query_embedding.cpu()
            
        query_embedding_np = query_embedding.numpy().reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding_np, self.corpus_embeddings_np)[0]
        
        # 获取所有索引并按相似度排序
        sorted_indices = np.argsort(similarities)[::-1]
        
        unique_texts = []
        unique_outputs = []
        seen_contents = set() if not deduplicate else set([query])  # 用于追踪已处理的内容
        
        # 遍历所有排序后的索引
        for idx in sorted_indices:
            content = self.texts[idx]
            sim_score = similarities[idx]  # 获取当前相似度分数
            
            # 阈值过滤：如果相似度低于阈值则跳过
            if sim_score < threshold:
                continue  # 跳过低于阈值的结果
                
            # 去重逻辑
            if deduplicate:
                if content in seen_contents:
                    continue  # 已处理过相同内容，跳过
                seen_contents.add(content)
            
            unique_texts.append(content)
            unique_outputs.append(self.test2item[content]['output'])
            
            # 达到需要的 top_k 数量时停止
            if len(unique_texts) >= top_k:
                break

        return unique_texts, unique_outputs

class MultiClassRetriever:

    def __init__(
            self, 
            model_path: str, 
            model_name: str, 
            data_path: Optional[str] = None, 
            device: str = "cuda:0"):

        logger.info(f"Loading model from path: {model_path}")
        self.model = SentenceTransformer(model_path).to(device)
        self.model_name = model_name
        self.model_path = model_path
        self.device = device

        self.reranker = None
        if data_path:
            self.load_datas(data_path)
            self.build_retrievers()

    def load_datas(self, data_path: Optional[str] = None, data_list: Optional[list[dict]] = None):
        if data_list is None:
            if data_path is None:
                raise ValueError("Either data_path or data_list must be provided.")
            loaded_data_list: list[dict] = load_json(data_path)
        else:
            loaded_data_list = data_list

        self.class_data_dict = {}
        self.class_texts = {}
        self.class_test2item = {}

        for targeted_group in TARGETED_GROUPS:
            new_data_list = []
            for data in loaded_data_list:
                if targeted_group in [quadruple["targeted_group"] for quadruple in data["quadruples"]]:
                    new_data_list.append(data)
            self.class_data_dict[targeted_group] = new_data_list
    
    def build_retrievers(self):
        self.retrievers = {}
        for class_name in self.class_data_dict.keys():
            retriever = Retriever(model_path=self.model_path, model_name=self.model_name, device=self.device)
            retriever.create_embeddings(self.class_data_dict[class_name])
            self.retrievers[class_name] = retriever

    def retrieve(self, query: str, top_k: int = 1, deduplicate: bool = True, threshold: float = 0, weights: dict[str, float] = DEFAULT_WEIGHTS, reverse: bool = False) -> tuple[list[str], list[str]]:

        allocated_class_top_k = allocate_class_num(top_k, weights, reverse)
        all_texts = []
        all_outputs = []

        for class_name in TARGETED_GROUPS:
            if class_name not in self.retrievers:
                continue
            class_top_k = allocated_class_top_k[class_name]
            if class_top_k == 0:
                continue
            texts, outputs = self.retrievers[class_name].retrieve(query, class_top_k, deduplicate, threshold)
            all_texts.extend(texts)
            all_outputs.extend(outputs)

        return all_texts, all_outputs

if __name__ == "__main__":
    # retriever = LexiconRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/lexicon/annotated_lexicon.json")
    # print(retriever.including_retrieve("那些嫁给默的国女能自愿放弃中国国籍，绝对值得立牌坊。", top_k=-1))
    # retriever = StepOneRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/full/std/train.json")
    # print(retriever.retrieve("那些嫁给默的国女能自愿放弃中国国籍，绝对值得立牌坊。", top_k=5, threshold=0.5))
    import json
    retriever = MultiClassRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/full/std/train.json")
    print(json.dumps(retriever.retrieve("那些嫁给默的国女能自愿放弃中国国籍，绝对值得立牌坊。", top_k=9), ensure_ascii=False, indent=2))