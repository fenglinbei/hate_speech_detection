
from loguru import logger
from typing import Optional
from tools.json_tools import load_json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

from tools.convert import output2triple

class Retriver:

    def __init__(self, model_path: str, model_name: str, data_path: Optional[str]):

        logger.info(f"Loading model from path: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.model_name = model_name

        if data_path:
            self.load_datas(data_path)
            self.create_embeddings()

    def create_embeddings(self, texts: Optional[list[str]] = None):
        logger.info("Processing embedding")
        if not texts:
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
            item['output'] = output2triple(item['output'])
            self.test2item[item['content']] = item


    def retrieve(self, query: str, top_k: int=1) -> tuple[list[str], list[str]]:
        query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)

        if query_embedding.is_cuda:
            query_embedding = query_embedding.cpu()

        query_embedding_np = query_embedding.numpy().reshape(1, -1)

        similarities = cosine_similarity(query_embedding_np, self.corpus_embeddings_np)[0]

        if len(similarities) <= top_k:
            top_k_indices = np.argsort(similarities)[::-1]
        else:
            top_k_indices = np.argsort(similarities)[-top_k:][::-1] # 从大到小排序

        # 返回最相关的文本
        retrieved_texts = [self.texts[idx] for idx in top_k_indices]
        
        return retrieved_texts, [self.test2item[retrieved_text]['output'] for retrieved_text in retrieved_texts]