import os
import math
import pickle
import hashlib
from loguru import logger
from typing import Optional, Dict, Any, List, Literal
from tools.json_tools import load_json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

from prompt import *
from rag.reranker import Reranker
from tools.convert import output2triple, parsed_quad_to_raw_quad, parsed_quad_to_tar_and_arg, parsed_quad_to_trip

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

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = "./cache", enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_str = f"{query}_{str(params)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def get(self, query: str, params: Dict[str, Any]):
        """从缓存中获取结果"""
        if not self.enabled:
            return None
            
        cache_key = self._get_cache_key(query, params)
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def set(self, query: str, params: Dict[str, Any], result: Any):
        """将结果存入缓存"""
        if not self.enabled:
            return
            
        cache_key = self._get_cache_key(query, params)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

class Retriever:

    def __init__(
            self, 
            model_path: Optional[str] = None, 
            model_name: Optional[str] = None, 
            model: Optional[SentenceTransformer] = None,
            data_path: Optional[str] = None, 
            reranker_model_path: Optional[str] = None,
            device: str = "cuda:0",
            cache_dir: str = "./cache",
            enable_cache: bool = True):

        logger.info(f"Loading model from path: {model_path}")
        if model:
            self.model = model
            self.model_name = model_name
        else:
            self.model = SentenceTransformer(model_path).to(device)
            self.model_name = model_name

        self.reranker = None
        if reranker_model_path:
            self.reranker = Reranker(model_path=reranker_model_path)

        if data_path:
            self.load_datas(data_path)
            self.create_embeddings()

        # 初始化缓存管理器
        self.cache_manager = CacheManager(cache_dir, enable_cache)

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


    def load_datas(self, data_path: Optional[str] = None, data_list: Optional[list[dict]] = None):
        if data_list is None:
            if data_path is None:
                raise ValueError("Either data_path or data_list must be provided.")
            data: list[dict] = load_json(data_path)
        else:
            data = data_list

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
            use_cache: bool = True,
            **kwargs
            ) -> tuple[list[str], list[str]]:
        
        params = {
            "top_k": top_k,
            "deduplicate": deduplicate,
            "threshold": threshold,
            "rerank": rerank,
            "resort": resort
        }
        
        if use_cache:
            cached_result = self.cache_manager.get(query, params)
            if cached_result is not None:
                # logger.debug(f"Cache hit for query: {query}")
                return cached_result

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

        result = (unique_texts, unique_outputs)
        # 保存到缓存
        if use_cache:
            self.cache_manager.set(query, params, result)

        return result
    
    def batch_retrieve(
            self,
            queries: List[str],
            top_k: int = 1,
            deduplicate: bool = True,
            threshold: float = 0,
            rerank: bool = False,
            resort: bool = False,
            use_cache: bool = True,
            batch_size: int = 32
            ) -> List[tuple[list[str], list[str]]]:
        """批量检索多个查询"""
        
        results = []
        
        # 分批处理查询
        for i in tqdm(range(0, len(queries), batch_size), desc="Batch retrieving"):
            batch_queries = queries[i:i+batch_size]
            batch_results = []
            
            # 批量编码查询
            query_embeddings = self.model.encode(batch_queries, convert_to_tensor=True, show_progress_bar=False)
            
            if query_embeddings.is_cuda:
                query_embeddings = query_embeddings.cpu()
                
            query_embeddings_np = query_embeddings.numpy()
            
            # 批量计算相似度
            similarities = cosine_similarity(query_embeddings_np, self.corpus_embeddings_np)
            
            for j, query in enumerate(batch_queries):
                # 检查缓存
                params = {
                    "top_k": top_k,
                    "deduplicate": deduplicate,
                    "threshold": threshold,
                    "rerank": rerank,
                    "resort": resort
                }
                
                if use_cache:
                    cached_result = self.cache_manager.get(query, params)
                    if cached_result is not None:
                        results.append(cached_result)
                        continue
                
                # 处理单个查询的相似度
                query_similarities = similarities[j]
                
                if rerank and self.reranker:
                    new_top_k = top_k * 5
                    if new_top_k < 10:
                        new_top_k = 10
                    elif new_top_k > 100:   
                        new_top_k = 100
                else:
                    new_top_k = top_k
                
                # 获取所有索引并按相似度排序
                sorted_indices = np.argsort(query_similarities)[::-1]
                
                unique_texts = []
                unique_outputs = []
                seen_contents = set() if not deduplicate else set([query])
                
                # 遍历所有排序后的索引
                for idx in sorted_indices:
                    content = self.texts[idx]
                    sim_score = query_similarities[idx]
                    
                    # 阈值过滤
                    if sim_score < threshold:
                        continue
                        
                    # 去重逻辑
                    if deduplicate:
                        if content in seen_contents:
                            continue
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
                    l = 0
                    r = len(unique_texts) - 1
                    for k in range(len(unique_texts)):
                        if k % 2 == 0:
                            resorted_indices[l] = k
                            l += 1
                        else:
                            resorted_indices[r] = k
                            r -= 1
                    
                    unique_texts = [unique_texts[k] for k in resorted_indices]
                    unique_outputs = [unique_outputs[k] for k in resorted_indices]

                result = (unique_texts, unique_outputs)
                results.append(result)
                
                # 保存到缓存
                if use_cache:
                    self.cache_manager.set(query, params, result)
        
        return results
    
class LexiconRetriever:

    def __init__(
            self, 
            model_path: str, 
            model_name: str, 
            data_path: str | None = None, 
            device: str = "cuda:0",
            cache_dir: str = "./cache",
            enable_cache: bool = True):
        
        logger.info(f"Loading model from path: {model_path}")
        self.model = SentenceTransformer(model_path).to(device)
        self.model_name = model_name
        
        # 初始化缓存管理器
        self.cache_manager = CacheManager(cache_dir, enable_cache)

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
        
    
    def similarity_retrieve(
            self, 
            query: str, 
            top_k: int = 1, 
            deduplicate: bool = True, 
            threshold: float = 0,
            use_cache: bool = True
            ) -> list[str]:
        
        # 检查缓存
        params = {
            "method": "similarity",
            "top_k": top_k,
            "deduplicate": deduplicate,
            "threshold": threshold
        }
        
        if use_cache:
            cached_result = self.cache_manager.get(query, params)
            if cached_result is not None:
                # logger.debug(f"Cache hit for query: {query}")
                return cached_result
        
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

        # 保存到缓存
        if use_cache:
            self.cache_manager.set(query, params, unique_texts)
            
        return unique_texts
    
    def including_retrieve(
            self, 
            query: str, 
            top_k: int = -1, 
            deduplicate: bool = True,
            use_cache: bool = True
            ) -> list[str]:
        
        # 检查缓存
        params = {
            "method": "including",
            "top_k": top_k,
            "deduplicate": deduplicate
        }
        
        if use_cache:
            cached_result = self.cache_manager.get(query, params)
            if cached_result is not None:
                # logger.debug(f"Cache hit for query: {query}")
                return cached_result
        
        result = []
        seen_words = set()
        
        for word in self.word2item.keys():
            if word in query:
                if deduplicate and word in seen_words:
                    continue
                result.append(self.word2item[word])
                seen_words.add(word)
        
        final_result = result if top_k == -1 else result[:top_k]
        
        # 保存到缓存
        if use_cache:
            self.cache_manager.set(query, params, final_result)
            
        return final_result
    
    def batch_similarity_retrieve(
            self,
            queries: List[str],
            top_k: int = 1,
            deduplicate: bool = True,
            threshold: float = 0,
            use_cache: bool = True,
            batch_size: int = 32
            ) -> List[list[str]]:
        """批量相似度检索"""
        
        results = []
        
        # 分批处理查询
        for i in tqdm(range(0, len(queries), batch_size), desc="Batch similarity retrieving"):
            batch_queries = queries[i:i+batch_size]
            
            # 批量编码查询
            query_embeddings = self.model.encode(batch_queries, convert_to_tensor=True, show_progress_bar=False)
            
            if query_embeddings.is_cuda:
                query_embeddings = query_embeddings.cpu()
                
            query_embeddings_np = query_embeddings.numpy()
            
            # 批量计算相似度
            similarities = cosine_similarity(query_embeddings_np, self.corpus_embeddings_np)
            
            for j, query in enumerate(batch_queries):
                # 检查缓存
                params = {
                    "method": "similarity",
                    "top_k": top_k,
                    "deduplicate": deduplicate,
                    "threshold": threshold
                }
                
                if use_cache:
                    cached_result = self.cache_manager.get(query, params)
                    if cached_result is not None:
                        results.append(cached_result)
                        continue
                
                # 处理单个查询的相似度
                query_similarities = similarities[j]
                
                # 获取所有索引并按相似度排序
                sorted_indices = np.argsort(query_similarities)[::-1]

                unique_texts = []
                seen_contents = set() if not deduplicate else set([query])
                
                # 遍历所有排序后的索引
                for idx in sorted_indices:
                    content = self.texts[idx]
                    sim_score = query_similarities[idx]
                    
                    # 阈值过滤
                    if sim_score < threshold:
                        continue
                    
                    # 去重逻辑
                    if deduplicate:
                        if content in seen_contents:
                            continue
                        seen_contents.add(content)
                    
                    unique_texts.append(content)
                    
                    # 达到需要的 top_k 数量时停止
                    if len(unique_texts) >= top_k:
                        break

                results.append(unique_texts)
                
                # 保存到缓存
                if use_cache:
                    self.cache_manager.set(query, params, unique_texts)
        
        return results
    
    def batch_including_retrieve(
            self,
            queries: List[str],
            top_k: int = -1,
            deduplicate: bool = True,
            use_cache: bool = True
            ) -> List[list[str]]:
        """批量包含检索"""
        
        results = []
        
        for query in tqdm(queries, desc="Batch including retrieving"):
            # 检查缓存
            params = {
                "method": "including",
                "top_k": top_k,
                "deduplicate": deduplicate
            }
            
            if use_cache:
                cached_result = self.cache_manager.get(query, params)
                if cached_result is not None:
                    results.append(cached_result)
                    continue
            
            result = []
            seen_words = set()
            
            for word in self.word2item.keys():
                if word in query:
                    if deduplicate and word in seen_words:
                        continue
                    result.append(self.word2item[word])
                    seen_words.add(word)
            
            final_result = result if top_k == -1 else result[:top_k]
            results.append(final_result)
            
            # 保存到缓存
            if use_cache:
                self.cache_manager.set(query, params, final_result)
        
        return results

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
            device: str = "cuda:0",
            model: Optional[SentenceTransformer] = None,
            cache_dir: str = "./cache",
            enable_cache: bool = True):

        if model:
            self.model = model
            self.model_name = model_name
            self.model_path = model_path
            self.device = device
        else:
            logger.info(f"Loading model from path: {model_path}")
            self.model = SentenceTransformer(model_path).to(device)
            self.model_name = model_name
            self.model_path = model_path
            self.device = device

        # 初始化缓存管理器
        self.cache_manager = CacheManager(cache_dir, enable_cache)

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
        self.retrievers: dict[str, Retriever] = {}
        for class_name in self.class_data_dict.keys():
            retriever = Retriever(model=self.model, enable_cache=False)  # 禁用子检索器的缓存，由父级管理
            retriever.create_embeddings(self.class_data_dict[class_name])
            self.retrievers[class_name] = retriever

    def retrieve(
            self, 
            query: str, 
            top_k: int = 1, 
            deduplicate: bool = True, 
            threshold: float = 0, 
            weights: Optional[dict[str, float]] = None, 
            weights_reverse: bool = False,
            use_cache: bool = True,
            **kwargs
            ) -> tuple[list[str], list[str]]:
        
        # 检查缓存
        params = {
            "top_k": top_k,
            "deduplicate": deduplicate,
            "threshold": threshold,
            "weights": weights,
            "weights_reverse": weights_reverse
        }
        
        if use_cache:
            cached_result = self.cache_manager.get(query, params)
            if cached_result is not None:
                # logger.debug(f"Cache hit for query: {query}")
                return cached_result
        
        if weights is None:
            weights = DEFAULT_WEIGHTS

        allocated_class_top_k = allocate_class_num(top_k, weights, weights_reverse)
        all_texts = []
        all_outputs = []

        for class_name in TARGETED_GROUPS:
            if class_name not in self.retrievers:
                continue
            class_top_k = allocated_class_top_k[class_name]
            if class_top_k == 0:
                continue
            texts, outputs = self.retrievers[class_name].retrieve(query, class_top_k, deduplicate, threshold, use_cache=False)
            all_texts.extend(texts)
            all_outputs.extend(outputs)

        result = (all_texts, all_outputs)
        
        # 保存到缓存
        if use_cache:
            self.cache_manager.set(query, params, result)
            
        return result
    
    def batch_retrieve(
            self,
            queries: List[str],
            top_k: int = 1,
            deduplicate: bool = True,
            threshold: float = 0,
            weights: Optional[dict[str, float]] = None,
            weights_reverse: bool = False,
            use_cache: bool = True,
            batch_size: int = 32
            ) -> List[tuple[list[str], list[str]]]:
        """批量检索多个查询"""
        
        results = []
        
        for query in tqdm(queries, desc="Multi-class batch retrieving"):
            # 检查缓存
            params = {
                "top_k": top_k,
                "deduplicate": deduplicate,
                "threshold": threshold,
                "weights": weights,
                "weights_reverse": weights_reverse
            }
            
            if use_cache:
                cached_result = self.cache_manager.get(query, params)
                if cached_result is not None:
                    results.append(cached_result)
                    continue
            
            if weights is None:
                weights = DEFAULT_WEIGHTS

            allocated_class_top_k = allocate_class_num(top_k, weights, weights_reverse)
            all_texts = []
            all_outputs = []

            for class_name in TARGETED_GROUPS:
                if class_name not in self.retrievers:
                    continue
                class_top_k = allocated_class_top_k[class_name]
                if class_top_k == 0:
                    continue
                
                # 使用子检索器的批量检索功能
                class_results = self.retrievers[class_name].batch_retrieve(
                    [query], class_top_k, deduplicate, threshold, use_cache=False, batch_size=1
                )
                if class_results:
                    texts, outputs = class_results[0]
                    all_texts.extend(texts)
                    all_outputs.extend(outputs)

            result = (all_texts, all_outputs)
            results.append(result)
            
            # 保存到缓存
            if use_cache:
                self.cache_manager.set(query, params, result)
        
        return results

class WrongExpRetriever:

    def __init__(
            self, 
            model_path: str, 
            model_name: str, 
            data_list: list[dict], 
            result_data_list: list[dict],
            device: str = "cuda:0",
            model: Optional[SentenceTransformer] = None):

        if model:
            self.model = model
            self.model_name = model_name
            self.model_path = model_path
            self.device = device
        else:
            logger.info(f"Loading model from path: {model_path}")
            self.model = SentenceTransformer(model_path).to(device)
            self.model_name = model_name

        self.load_datas(data_list, result_data_list)
        self.create_embeddings()

    def create_embeddings(self):
        logger.info("Processing embedding")

        corpus_embeddings = self.model.encode(self.texts, convert_to_tensor=True, show_progress_bar=True)
        if corpus_embeddings.is_cuda:
            corpus_embeddings = corpus_embeddings.cpu()
        self.corpus_embeddings_np = corpus_embeddings.numpy()

    def load_datas(self, data_list: list[dict], result_data_list: list[dict]):
        data_list.extend(result_data_list)
        self.texts: list[str] = [item['content'] for item in data_list]
        self.text2item = {}
        self.text2wrong_exp = {}
        for item in data_list:
            quadruples = item.get('quadruples', item.get("gt_quadruples"))
            if not quadruples:
                continue
            item['output'] = parsed_quad_to_trip(quadruples)
            self.text2item[item['content']] = item
            if "llm_output" in item:
                self.text2wrong_exp[item['content']] = item["llm_output"]

    def retrieve(
            self, 
            query: str, 
            top_k: int = 1, 
            deduplicate: bool = True, 
            threshold: float = 0
            ) -> tuple[list[str], list[str], list[Optional[str]]]:
        if top_k == 0:
            return [], [], []

        query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        
        if query_embedding.is_cuda:
            query_embedding = query_embedding.cpu()
            
        query_embedding_np = query_embedding.numpy().reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding_np, self.corpus_embeddings_np)[0]
        
        # 获取所有索引并按相似度排序
        sorted_indices = np.argsort(similarities)[::-1]
        
        unique_texts = []
        unique_outputs = []
        unique_wrong_exps = []
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
            unique_outputs.append(self.text2item[content]['output'])
            unique_wrong_exps.append(self.text2wrong_exp.get(content, None))
            
            # 达到需要的 top_k 数量时停止
            if len(unique_texts) >= top_k:
                break

        return unique_texts, unique_outputs, unique_wrong_exps   

class MultiClassWrongExpRetriever:

    def __init__(
            self, 
            model_path: str, 
            model_name: str, 
            data_list: list[dict], 
            result_data_list: list[dict],
            device: str = "cuda:0"):

        logger.info(f"Loading model from path: {model_path}")
        self.model = SentenceTransformer(model_path).to(device)
        self.model_name = model_name
        self.model_path = model_path
        self.device = device

        self.reranker = None
        self.load_datas(data_list, result_data_list)
        self.build_retrievers()

    def load_datas(self, data_list: list[dict], result_data_list: list[dict]):

        self.class_data_dict = {}
        self.class_result_data_dict = {}

        for targeted_group in TARGETED_GROUPS:
            new_data_list = []
            new_result_data_list = []
            for data in data_list:
                if targeted_group in [quadruple["targeted_group"] for quadruple in data["quadruples"]]:
                    new_data_list.append(data)
            for data in result_data_list:
                if targeted_group in [quadruple["targeted_group"] for quadruple in data["gt_quadruples"]]:
                    new_data_list.append(data)
            self.class_data_dict[targeted_group] = new_data_list
            self.class_result_data_dict[targeted_group] = new_result_data_list
    
    def build_retrievers(self):
        self.retrievers: dict[str, WrongExpRetriever] = {}
        for class_name in self.class_data_dict.keys():
            retriever = WrongExpRetriever(
                model_path=self.model_path, 
                model_name=self.model_name, 
                device=self.device, 
                data_list=self.class_data_dict[class_name], 
                result_data_list=self.class_result_data_dict[class_name],
                model=self.model
                )
            self.retrievers[class_name] = retriever

    def retrieve(self, query: str, top_k: int = 1, deduplicate: bool = True, threshold: float = 0, weights: Optional[dict[str, float]] = None, weights_reverse: bool = False) -> tuple[list[str], list[str], list[Optional[str]]]:
        if weights is None:
            weights = DEFAULT_WEIGHTS

        allocated_class_top_k = allocate_class_num(top_k, weights, weights_reverse)
        all_texts = []
        all_outputs = []
        all_wrong_exps = []

        for class_name in TARGETED_GROUPS:
            if class_name not in self.retrievers:
                continue
            class_top_k = allocated_class_top_k[class_name]
            if class_top_k == 0:
                continue
            texts, outputs, wrong_exps = self.retrievers[class_name].retrieve(query, class_top_k, deduplicate, threshold)
            all_texts.extend(texts)
            all_outputs.extend(outputs)
            all_wrong_exps.extend(wrong_exps)

        return all_texts, all_outputs, all_wrong_exps

from sklearn.cluster import KMeans

class ClusteredRetriever:
    """
    基于聚类的分层 Retriever：
    - 先对全语料做聚类
    - 每个 cluster 视为一层（类似 MultiClassRetriever 中的一个 class）
    - 检索时可以通过 weights / weights_reverse 控制各 cluster 在最终 N-shot 中的比例
    """

    def __init__(
            self,
            model_path: Optional[str] = None,
            model_name: Optional[str] = None,
            model: Optional[SentenceTransformer] = None,
            data_path: Optional[str] = None,
            data_list: Optional[list[dict]] = None,
            n_clusters: int = 8,
            device: str = "cuda:0",
            cluster_model: Optional[KMeans] = None,
            cache_dir: str = "./cache",
            enable_cache: bool = True,
            random_state: int = 42,
    ) -> None:
        """
        参数说明：
        - model / model_path：与原 Retriever 一致
        - data_path / data_list：数据来源，结构与原 Retriever.load_datas 一致
        - n_clusters：聚类个数（如果外部传入 cluster_model，则优先使用外部模型）
        - cluster_model：外部传入的已训练聚类模型（可选）
        - weights / weights_reverse：检索时控制各 cluster 的配额（同 MultiClassRetriever）
        """
        # 复用你现有的编码模型
        if model:
            self.model = model
            self.model_name = model_name
            self.device = device
        else:
            logger.info(f"Loading model from path: {model_path}")
            self.model = SentenceTransformer(model_path).to(device)
            self.model_name = model_name
            self.device = device

        # 缓存管理
        self.cache_manager = CacheManager(cache_dir, enable_cache)

        # 加载数据、构建主 retriever（全部数据）
        self._load_datas(data_path, data_list)
        self._build_global_retriever()

        # 聚类 & 按 cluster 划分数据
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._build_clusters(cluster_model)

        # 构建每个 cluster 对应的子 Retriever
        self._build_cluster_retrievers()

    def _load_datas(self, data_path: Optional[str], data_list: Optional[list[dict]]) -> None:
        """加载原始数据，结构与 Retriever.load_datas 保持一致"""
        if data_list is None:
            if data_path is None:
                raise ValueError("Either data_path or data_list must be provided.")
            data: list[dict] = load_json(data_path)
        else:
            data = data_list

        self.data = data

    def _build_global_retriever(self) -> None:
        """构建包含所有数据的基础 Retriever，用于统一编码 & 聚类"""
        self.global_retriever = Retriever(
            model=self.model,
            model_name=self.model_name,
            enable_cache=False,  # 这里禁用子层缓存，由 ClusteredRetriever 统一管理
        )
        # 使用 Retriever.create_embeddings 的 datas 分支，保证结构与原项目一致
        self.global_retriever.create_embeddings(self.data)
        self.texts = self.global_retriever.texts
        self.test2item = self.global_retriever.test2item
        self.corpus_embeddings_np = self.global_retriever.corpus_embeddings_np

    def _build_clusters(self, cluster_model: Optional[KMeans] = None) -> None:
        """对 embedding 做聚类，构建 cluster -> indices 映射"""

        logger.info("Clustering corpus embeddings for ClusteredRetriever")

        if cluster_model is None:
            # 默认使用 KMeans，你可以根据需要改成 MiniBatchKMeans 或 Agglomerative 等
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init="auto"
            )
            self.cluster_model.fit(self.corpus_embeddings_np)
        else:
            self.cluster_model = cluster_model

        labels = self.cluster_model.labels_
        self.cluster2indices: Dict[int, List[int]] = {}
        for idx, c in enumerate(labels):
            self.cluster2indices.setdefault(int(c), []).append(idx)

        # 为后续权重分配准备 cluster 名称列表（类似 TARGETED_GROUPS）
        self.cluster_ids: List[int] = sorted(self.cluster2indices.keys())

    def _build_cluster_retrievers(self) -> None:
        """
        为每个 cluster 构建一个子 Retriever，接口与原 Retriever 完全一致，
        方便你在其他地方无缝替换/复用。
        """
        self.cluster_retrievers: Dict[int, Retriever] = {}

        for c in self.cluster_ids:
            indices = self.cluster2indices[c]
            cluster_data = [self.data[i] for i in indices]

            retriever = Retriever(
                model=self.model,
                model_name=self.model_name,
                enable_cache=False  # 子 retriever 不再单独使用缓存，由上层统一控制
            )
            retriever.create_embeddings(cluster_data)
            self.cluster_retrievers[c] = retriever

    def _default_cluster_weights(self) -> Dict[int, float]:
        """
        默认权重：可以按 cluster 中样本数比例分配，也可以均匀分配。
        这里演示：按样本数比例。
        """
        counter = {c: len(self.cluster2indices[c]) for c in self.cluster_ids}
        total = sum(counter.values())
        return {c: (counter[c] / total * 100.0) for c in self.cluster_ids}

    def retrieve(
            self,
            query: str,
            top_k: int = 1,
            deduplicate: bool = True,
            threshold: float = 0,
            weights: Optional[Dict[int, float]] = None,
            weights_reverse: bool = False,
            use_cache: bool = True,
            **kwargs
    ) -> tuple[list[str], list[str]]:
        """
        分层检索：
        - top_k：最终希望返回的样本总数
        - weights：cluster_id -> 权重（总和约等于 100），可动态调整
        - weights_reverse：如果为 True，则对权重取倒数后重新归一化（类似你在 MultiClassRetriever 里的少样本放大）
        """
        # 构造缓存 key
        params = {
            "top_k": top_k,
            "deduplicate": deduplicate,
            "threshold": threshold,
            "weights": weights,
            "weights_reverse": weights_reverse,
        }

        if use_cache:
            cached_result = self.cache_manager.get(query, params)
            if cached_result is not None:
                return cached_result

        if top_k == 0:
            return [], []

        if weights is None:
            weights = self._default_cluster_weights()

        # 利用你已有的 allocate_class_num 做“配额分配”
        allocated_cluster_top_k = allocate_class_num(top_k, weights, reverse=weights_reverse)

        all_texts: List[str] = []
        all_outputs: List[Any] = []

        for c in self.cluster_ids:
            if c not in self.cluster_retrievers:
                continue

            cluster_top_k = allocated_cluster_top_k.get(c, 0)
            if cluster_top_k <= 0:
                continue

            texts, outputs = self.cluster_retrievers[c].retrieve(
                query=query,
                top_k=cluster_top_k,
                deduplicate=deduplicate,
                threshold=threshold,
                use_cache=False,   # 子层不再单独缓存
                **kwargs
            )
            all_texts.extend(texts)
            all_outputs.extend(outputs)

        result = (all_texts, all_outputs)

        if use_cache:
            self.cache_manager.set(query, params, result)

        return result

    def batch_retrieve(
            self,
            queries: List[str],
            top_k: int = 1,
            deduplicate: bool = True,
            threshold: float = 0,
            weights: Optional[Dict[int, float]] = None,
            weights_reverse: bool = False,
            use_cache: bool = True,
            batch_size: int = 32,
            **kwargs
    ) -> List[tuple[list[str], list[str]]]:
        """批量分层检索，接口风格与 MultiClassRetriever.batch_retrieve 对齐"""

        results: List[tuple[list[str], list[str]]] = []

        for query in tqdm(queries, desc="Clustered batch retrieving"):
            params = {
                "top_k": top_k,
                "deduplicate": deduplicate,
                "threshold": threshold,
                "weights": weights,
                "weights_reverse": weights_reverse,
            }

            if use_cache:
                cached_result = self.cache_manager.get(query, params)
                if cached_result is not None:
                    results.append(cached_result)
                    continue

            if weights is None:
                weights = self._default_cluster_weights()

            allocated_cluster_top_k = allocate_class_num(top_k, weights, reverse=weights_reverse)

            all_texts: List[str] = []
            all_outputs: List[Any] = []

            for c in self.cluster_ids:
                if c not in self.cluster_retrievers:
                    continue

                cluster_top_k = allocated_cluster_top_k.get(c, 0)
                if cluster_top_k <= 0:
                    continue

                # 这里可以选择是否用 batch_retrieve；为简单和一致起见，直接调用单次 retrieve
                texts, outputs = self.cluster_retrievers[c].retrieve(
                    query=query,
                    top_k=cluster_top_k,
                    deduplicate=deduplicate,
                    threshold=threshold,
                    use_cache=False,
                    **kwargs
                )

                all_texts.extend(texts)
                all_outputs.extend(outputs)

            result = (all_texts, all_outputs)
            results.append(result)

            if use_cache:
                self.cache_manager.set(query, params, result)

        return results
    
class StochasticWeightedRetriever(Retriever):
    """
    在原 Retriever 基础上：
    - 支持对相似度进行形状加权（similarity_alpha）
    - 支持可控随机策略（random_strategy, random_ratio, temperature 等）
    """

    def __init__(
            self,
            random_state: Optional[int] = None,
            model_path: Optional[str] = None, 
            model_name: Optional[str] = None, 
            model: Optional[SentenceTransformer] = None,
            data_path: Optional[str] = None, 
            reranker_model_path: Optional[str] = None,
            device: str = "cuda:0",
            cache_dir: str = "./cache",
            enable_cache: bool = True,
    ):
        super().__init__(
            model_path=model_path, 
            model_name=model_name, 
            model=model,
            data_path=data_path, 
            reranker_model_path=reranker_model_path,
            device=device,
            cache_dir=cache_dir,
            enable_cache=enable_cache
        )
        self.random_state = np.random.RandomState(random_state) if random_state is not None else np.random

    def _transform_similarities(
            self,
            similarities: np.ndarray,
            similarity_alpha: float = 1.0,
    ) -> np.ndarray:
        """
        对相似度进行形状变换：
        - alpha > 1.0：增强高相似度样本的优势（更尖锐）
        - alpha < 1.0：平滑差距，提升低相似度样本的概率（更多样性）
        """
        if similarity_alpha <= 0:
            raise ValueError("similarity_alpha must be > 0")
        if similarity_alpha == 1.0:
            return similarities
        # 为防止出现负数，先做一个简单的平移（保证非负），再幂次变换
        min_sim = similarities.min()
        shifted = similarities - min_sim  # >=0
        transformed = np.power(shifted, similarity_alpha)
        return transformed

    def _sample_indices(
            self,
            scores: np.ndarray,
            candidate_indices: np.ndarray,
            top_k: int,
            random_strategy: Literal["none", "sample", "hybrid"],
            random_ratio: float,
            temperature: float,
    ) -> List[int]:
        """
        根据策略从 candidate_indices 中挑选最终 top_k 的下标。
        返回的是 candidate_indices 的下标（即在 candidate_indices 中的位置，不是语料原始索引）。
        """
        if len(candidate_indices) == 0:
            return []

        random_strategy = random_strategy.lower()
        random_ratio = min(max(random_ratio, 0.0), 1.0)
        temperature = max(temperature, 1e-6)

        # 基础分布：softmax(scores / temperature)
        scores = scores.astype(np.float64)
        # 数值稳定
        max_score = np.max(scores)
        prob = np.exp((scores - max_score) / temperature)
        prob_sum = prob.sum()
        if prob_sum == 0:
            prob = np.ones_like(prob) / len(prob)
        else:
            prob = prob / prob_sum

        n = min(top_k, len(candidate_indices))

        if random_strategy == "none":
            # 完全 deterministic：等价于普通 top-k
            sorted_local_idx = np.argsort(scores)[::-1][:n]
            return sorted_local_idx.tolist()

        elif random_strategy == "sample":
            # 完全基于概率采样，不放回
            chosen = self.random_state.choice(
                len(candidate_indices),
                size=n,
                replace=False,
                p=prob
            )
            return chosen.tolist()

        elif random_strategy == "hybrid":
            # 前 deterministic_ratio 部分使用确定性 top-k，其余使用采样
            deterministic_k = int(round(n * (1.0 - random_ratio)))
            deterministic_k = max(0, min(deterministic_k, n))
            sampled_k = n - deterministic_k

            sorted_local_idx = np.argsort(scores)[::-1]
            deterministic_part = sorted_local_idx[:deterministic_k].tolist()

            if sampled_k <= 0:
                return deterministic_part

            # 剩余部分中按概率采样
            remaining_idx = sorted_local_idx[deterministic_k:]
            if len(remaining_idx) <= sampled_k:
                sampled_part = remaining_idx.tolist()
            else:
                remaining_prob = prob[remaining_idx]
                remaining_prob = remaining_prob / remaining_prob.sum()
                sampled_local = self.random_state.choice(
                    len(remaining_idx),
                    size=sampled_k,
                    replace=False,
                    p=remaining_prob
                )
                sampled_part = remaining_idx[sampled_local].tolist()

            return deterministic_part + sampled_part

        else:
            raise ValueError(f"Unsupported random_strategy: {random_strategy}")

    def retrieve(
            self,
            query: str,
            top_k: int = 1,
            deduplicate: bool = True,
            threshold: float = 0,
            rerank: bool = False,
            resort: bool = False,
            use_cache: bool = True,
            # 新增参数
            similarity_alpha: float = 1.0,
            random_strategy: Literal["none", "sample", "hybrid"] = "none", 
            random_ratio: float = 0.3,
            temperature: float = 1.0,
            candidate_multiplier: float = 3.0,
            **kwargs
    ) -> tuple[list[str], list[str]]:
        """
        新增参数说明：
        - similarity_alpha：相似度形状控制，>1 更集中，<1 更平缓
        - random_strategy："none"（默认，不引入随机）、"sample"、"hybrid"
        - random_ratio：在 hybrid 下，随机部分比例（0~1）
        - temperature：softmax 温度，越大分布越平滑,随机性越高
        - candidate_multiplier：候选池大小 = top_k * candidate_multiplier
        """

        # 需要把新增参数也纳入 cache key，否则缓存失真
        params = {
            "top_k": top_k,
            "deduplicate": deduplicate,
            "threshold": threshold,
            "rerank": rerank,
            "resort": resort,
            "similarity_alpha": similarity_alpha,
            "random_strategy": random_strategy,
            "random_ratio": random_ratio,
            "temperature": temperature,
            "candidate_multiplier": candidate_multiplier,
        }

        if use_cache:
            cached_result = self.cache_manager.get(query, params)
            if cached_result is not None:
                return cached_result

        if top_k == 0:
            return [], []

        # rerank 控制候选池大小
        if rerank and self.reranker:
            base_candidate_k = max(10, min(100, top_k * 5))
        else:
            base_candidate_k = top_k

        # 额外放大候选池，便于做“多样性采样”
        candidate_k = int(max(base_candidate_k, top_k * candidate_multiplier))

        # 1) 编码 query
        query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        if query_embedding.is_cuda:
            query_embedding = query_embedding.cpu()
        query_embedding_np = query_embedding.numpy().reshape(1, -1)

        # 2) 计算 cosine 相似度
        similarities = cosine_similarity(query_embedding_np, self.corpus_embeddings_np)[0]

        # 3) 形状变换（相似度加权）
        transformed_scores = self._transform_similarities(similarities, similarity_alpha)

        # 4) 先按 transformed_scores 做排序，但只取 candidate_k 作为候选
        sorted_indices = np.argsort(transformed_scores)[::-1]

        unique_texts: List[str] = []
        unique_outputs: List[Any] = []
        sim_scores: List[float] = []  # 用于后续 sample

        seen_contents = set() if not deduplicate else set([query])

        for idx in sorted_indices:
            content = self.texts[idx]
            sim_score = similarities[idx]
            score_after_transform = transformed_scores[idx]

            if sim_score < threshold:
                continue

            if deduplicate and content in seen_contents:
                continue

            seen_contents.add(content)
            unique_texts.append(content)
            unique_outputs.append(self.test2item[content]['output'])
            sim_scores.append(score_after_transform)

            if len(unique_texts) >= candidate_k:
                break

        if len(unique_texts) == 0:
            result = ([], [])
            if use_cache:
                self.cache_manager.set(query, params, result)
            return result

        # 5) 如果有 reranker，先在候选集合上 rerank 一次，再作为 scores
        if rerank and self.reranker:
            scores = self.reranker.rerank(query, unique_texts)
            scores = np.array(scores, dtype=np.float64)
        else:
            scores = np.array(sim_scores, dtype=np.float64)

        # 6) 根据 random_strategy，从候选集合中选择最终 top_k
        local_indices = self._sample_indices(
            scores=scores,
            candidate_indices=np.arange(len(unique_texts)),
            top_k=top_k,
            random_strategy=random_strategy,
            random_ratio=random_ratio,
            temperature=temperature,
        )

        unique_texts = [unique_texts[i] for i in local_indices]
        unique_outputs = [unique_outputs[i] for i in local_indices]

        # 7) 若需要 resort，则对选出的结果做“左右穿插重排”，逻辑与原实现一致
        if resort and len(unique_texts) > 1:
            resorted_indices = [0] * len(unique_texts)
            l, r = 0, len(unique_texts) - 1
            for i in range(len(unique_texts)):
                if i % 2 == 0:
                    resorted_indices[l] = i
                    l += 1
                else:
                    resorted_indices[r] = i
                    r -= 1
            unique_texts = [unique_texts[i] for i in resorted_indices]
            unique_outputs = [unique_outputs[i] for i in resorted_indices]

        result = (unique_texts, unique_outputs)
        if use_cache:
            self.cache_manager.set(query, params, result)
        return result

    def batch_retrieve(
            self,
            queries: List[str],
            top_k: int = 1,
            deduplicate: bool = True,
            threshold: float = 0,
            rerank: bool = False,
            resort: bool = False,
            use_cache: bool = True,
            batch_size: int = 32,
            # 新增参数（与单次 retrieve 对齐）
            similarity_alpha: float = 1.0,
            random_strategy: str = "none",
            random_ratio: float = 0.3,
            temperature: float = 1.0,
            candidate_multiplier: float = 3.0,
            **kwargs
    ) -> List[tuple[list[str], list[str]]]:
        """
        批量版本，这里为了简洁直接循环调用单次 retrieve。
        如果你对性能有极致需求，可以再做真正的批量优化（类似你原来的 batch_retrieve）。
        """
        results: List[tuple[list[str], list[str]]] = []
        for q in tqdm(queries, desc="Stochastic batch retrieving"):
            res = self.retrieve(
                query=q,
                top_k=top_k,
                deduplicate=deduplicate,
                threshold=threshold,
                rerank=rerank,
                resort=resort,
                use_cache=use_cache,
                similarity_alpha=similarity_alpha,
                random_strategy=random_strategy,
                random_ratio=random_ratio,
                temperature=temperature,
                candidate_multiplier=candidate_multiplier,
                **kwargs
            )
            results.append(res)
        return results



if __name__ == "__main__":
    # retriever = LexiconRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/lexicon/annotated_lexicon.json")
    # print(retriever.including_retrieve("那些嫁给默的国女能自愿放弃中国国籍，绝对值得立牌坊。", top_k=-1))
    # retriever = StepOneRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/full/std/train.json")
    # print(retriever.retrieve("那些嫁给默的国女能自愿放弃中国国籍，绝对值得立牌坊。", top_k=5, threshold=0.5))
    import json
    # retriever = MultiClassRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/full/std/train.json")
    data_list = load_json("data/full/std/train.json")
    result_data_list = load_json("runner/output/simlex5_rag9_multi_class.json")["results"]
    # retriever = MultiClassWrongExpRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_list=data_list, result_data_list=result_data_list)
    # print(json.dumps(retriever.retrieve("那些嫁给默的国女能自愿放弃中国国籍，绝对值得立牌坊。", top_k=9), ensure_ascii=False, indent=2))

    retriever = StochasticWeightedRetriever(random_state=42, model_path="./models/base/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5")
    retriever.create_embeddings(data_list)
    texts, outputs = retriever.retrieve(
        "那些嫁给默的国女能自愿放弃中国国籍，绝对值得立牌坊。",
        top_k=5,
        similarity_alpha=0.5,
        random_strategy="hybrid",
        random_ratio=0.6,
        temperature=2.0,
        candidate_multiplier=4.0
    )
    print(json.dumps({"texts": texts, "outputs": outputs}, ensure_ascii=False, indent=2))