from loguru import logger
from typing import Optional
from tools.json_tools import load_json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

from prompt import *
from tools.convert import output2triple, parsed_quad_to_raw_quad

class Retriever:

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
    
    def retrieve(self, query: str, top_k: int = 1, deduplicate: bool = True, threshold: float = 0) -> tuple[list[str], list[str]]:
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


if __name__ == "__main__":
    retriever = LexiconRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/lexicon/annotated_lexicon.json")
    print(retriever.including_retrieve("那些嫁给默的国女能自愿放弃中国国籍，绝对值得立牌坊。", top_k=-1))