import os
import math
import pickle
import hashlib
import json
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

def normalize_target_groups(target_groups: Optional[List[str]] = None) -> List[str]:
    groups = target_groups or TARGETED_GROUPS
    normalized = []
    for group in groups:
        if group and group not in normalized:
            normalized.append(str(group))
    return normalized or list(TARGETED_GROUPS)

def normalize_weights(
        weights: Optional[Dict[str, float]],
        target_groups: List[str]
        ) -> Dict[str, float]:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    normalized = {}
    for group in target_groups:
        value = weights.get(group) if isinstance(weights, dict) else None
        if value is None:
            value = 100.0 / len(target_groups)
        normalized[group] = float(value)

    if sum(normalized.values()) <= 0:
        return {group: 100.0 / len(target_groups) for group in target_groups}

    return normalized

def split_targeted_groups(raw_group: Any) -> List[str]:
    if raw_group is None:
        return []
    if isinstance(raw_group, list):
        parts = raw_group
    else:
        text = str(raw_group).replace(";", ",").replace("/", ",").replace("|", ",")
        parts = text.split(",")
    return [str(part).strip() for part in parts if str(part).strip()]

def quad_has_group(quadruple: dict, targeted_group: str) -> bool:
    return targeted_group in split_targeted_groups(quadruple.get("targeted_group"))

def allocate_class_num(i, weights_dict, reverse: bool = False):
    # ?????????????????????????????
    initial_allocation = {}
    fractions = []
    total_integer = 0

    if reverse:
        weights_dict = {k: 1/v for k, v in weights_dict.items()}
        total_weight = sum(weights_dict.values())
        weights_dict = {k: (v / total_weight) * 100 for k, v in weights_dict.items()}
    
    # ?????????????????????????????????????????
    for key, weight in weights_dict.items():
        theory_value = i * weight / 100.0
        integer_part = math.floor(theory_value)
        fraction = theory_value - integer_part
        
        initial_allocation[key] = integer_part
        fractions.append((key, fraction))
        total_integer += integer_part
    
    # ??????????????????????????
    remaining = i - total_integer
    
    # ??????????????????????????????????????????
    fractions.sort(key=lambda x: (-x[1], x[0]))
    
    for idx in range(remaining):
        key = fractions[idx][0]
        initial_allocation[key] += 1

    return initial_allocation


def _top_k_sorted_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the largest k scores, sorted descending."""
    if scores.size == 0 or k <= 0:
        return np.array([], dtype=np.int32)

    k = min(int(k), scores.size)
    if k >= scores.size:
        candidate_indices = np.arange(scores.size)
    else:
        candidate_indices = np.argpartition(scores, -k)[-k:]

    order = np.argsort(scores[candidate_indices])[::-1]
    return candidate_indices[order].astype(np.int32)

class CacheManager:
    """Cache helper for selection, embedding, and retrieval results."""

    def __init__(self, cache_dir: str = "./cache", enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        os.makedirs(cache_dir, exist_ok=True)
        for stage in ("selection", "embedding", "retrieval"):
            os.makedirs(os.path.join(cache_dir, stage), exist_ok=True)

    # --------- helpers
    @staticmethod
    def _stable_dumps(obj: Any) -> str:
        def _default(o):
            # ????????? json ????????str
            return str(o)
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_default)

    @staticmethod
    def _md5(s: str) -> str:
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    @staticmethod
    def sha1_text(s: str) -> str:
        h = hashlib.sha1()
        h.update(s.encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def texts_signature(texts: List[str]) -> str:
        """Build a stable signature for a corpus."""
        h = hashlib.sha1()
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    def make_key(self, payload: Dict[str, Any]) -> str:
        return self._md5(self._stable_dumps(payload))

    def _path(self, stage: str, key: str, ext: str) -> str:
        return os.path.join(self.cache_dir, stage, f"{key}.{ext}")

    # --------- selection (pickle)
    def get_selection(self, key: str):
        if not self.enabled:
            return None
        p = self._path("selection", key, "pkl")
        if not os.path.exists(p):
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load selection cache: {e}")
            return None

    def set_selection(self, key: str, value: Any):
        if not self.enabled:
            return
        p = self._path("selection", key, "pkl")
        try:
            with open(p, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save selection cache: {e}")

    # --------- embedding (npy)
    def get_embedding(self, key: str):
        if not self.enabled:
            return None
        p = self._path("embedding", key, "npy")
        if not os.path.exists(p):
            return None
        try:
            return np.load(p, allow_pickle=False)
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            return None

    def set_embedding(self, key: str, arr: np.ndarray):
        if not self.enabled:
            return
        p = self._path("embedding", key, "npy")
        try:
            np.save(p, arr)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    # --------- retrieval (npz: indices + sims + optional meta)
    def get_retrieval(self, key: str):
        if not self.enabled:
            return None
        p = self._path("retrieval", key, "npz")
        if not os.path.exists(p):
            return None
        try:
            z = np.load(p, allow_pickle=True)
            indices = z["indices"]
            sims = z["sims"]
            meta = dict(z["meta"].item()) if "meta" in z.files else {}
            return indices, sims, meta
        except Exception as e:
            logger.warning(f"Failed to load retrieval cache: {e}")
            return None

    def set_retrieval(self, key: str, indices: np.ndarray, sims: np.ndarray, meta: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        p = self._path("retrieval", key, "npz")
        try:
            if meta is None:
                np.savez_compressed(p, indices=indices, sims=sims)
            else:
                np.savez_compressed(p, indices=indices, sims=sims, meta=np.array(meta, dtype=object))
        except Exception as e:
            logger.warning(f"Failed to save retrieval cache: {e}")

    # --------- backward-compatible wrappers: treat get/set as selection cache
    def get(self, query: str, params: Dict[str, Any]):
        if not self.enabled:
            return None
        key = self.make_key({"query": query, "params": params})
        return self.get_selection(key)

    def set(self, query: str, params: Dict[str, Any], result: Any):
        if not self.enabled:
            return
        key = self.make_key({"query": query, "params": params})
        self.set_selection(key, result)


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
            self.model_name = model_name or (os.path.basename(model_path) if model_path else 'sentence-transformer')
        else:
            self.model = SentenceTransformer(model_path).to(device)
            self.model_name = model_name

        self.reranker = None
        if reranker_model_path:
            self.reranker = Reranker(model_path=reranker_model_path)

        self.cache_manager = CacheManager(cache_dir, enable_cache)

        if data_path:
            self.load_datas(data_path)
            self.create_embeddings()

    def create_embeddings(self, datas: Optional[list[dict]] = None):
        logger.info("Processing embedding")
        if not datas:
            texts = self.texts
            if not hasattr(self, "text2idx"):
                self.text2idx = {}
                for idx, text in enumerate(texts):
                    self.text2idx.setdefault(text, idx)
        else:
            self.texts = [item['content'] for item in datas]
            self.test2item = {}
            self.text2idx = {}
            for idx, item in enumerate(datas):
                item['output'] = parsed_quad_to_raw_quad(item['quadruples'])
                self.test2item[item['content']] = item
                self.text2idx.setdefault(item['content'], idx)
            texts = self.texts

        self.corpus_sig = CacheManager.texts_signature(texts)

        # corpus embedding cache
        emb_key = None
        if self.cache_manager.enabled:
            emb_key = self.cache_manager.make_key({
                "stage": "corpus_embedding",
                "model": self.model_name,
                "corpus_sig": self.corpus_sig,
            })
            cached = self.cache_manager.get_embedding(emb_key)
            if cached is not None:
                self.corpus_embeddings_np = cached
                return

        corpus_embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        if hasattr(corpus_embeddings, 'is_cuda') and corpus_embeddings.is_cuda:
            corpus_embeddings = corpus_embeddings.cpu()
        self.corpus_embeddings_np = corpus_embeddings.numpy().astype(np.float32)

        if self.cache_manager.enabled and emb_key is not None:
            self.cache_manager.set_embedding(emb_key, self.corpus_embeddings_np)


    def load_datas(self, data_path: Optional[str] = None, data_list: Optional[list[dict]] = None):
        if data_list is None:
            if data_path is None:
                raise ValueError("Either data_path or data_list must be provided.")
            data: list[dict] = load_json(data_path)
        else:
            data = data_list

        self.texts = [item['content'] for item in data]
        self.test2item = {}
        self.text2idx = {}
        for idx, item in enumerate(data):
            item['output'] = parsed_quad_to_raw_quad(item['quadruples'])
            self.test2item[item['content']] = item
            self.text2idx.setdefault(item['content'], idx)
    
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
            "resort": resort,
            "model": self.model_name,
            "corpus_sig": getattr(self, "corpus_sig", None),
        }

        if use_cache:
            cached_result = self.cache_manager.get(query, params)
            if cached_result is not None:
                return cached_result

        if top_k == 0:
            return [], []

        # rerank ??????????
        if rerank and self.reranker:
            new_top_k = top_k * 5
            if new_top_k < 10:
                new_top_k = 10
            elif new_top_k > 100:
                new_top_k = 100
        else:
            new_top_k = top_k

        # 2) query embedding cache?????use_cache ???????????cache ??????
        q_sha1 = CacheManager.sha1_text(query)
        q_key = None
        query_embedding_np = None
        query_idx = getattr(self, "text2idx", {}).get(query)
        if query_idx is not None and hasattr(self, "corpus_embeddings_np"):
            query_embedding_np = self.corpus_embeddings_np[int(query_idx):int(query_idx) + 1]
        elif use_cache and self.cache_manager.enabled:
            q_key = self.cache_manager.make_key({
                "stage": "query_embedding",
                "model": self.model_name,
                "q_sha1": q_sha1,
            })
            query_embedding_np = self.cache_manager.get_embedding(q_key)

        if query_embedding_np is None:
            query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            if hasattr(query_embedding, 'is_cuda') and query_embedding.is_cuda:
                query_embedding = query_embedding.cpu()
            query_embedding_np = query_embedding.numpy().astype(np.float32)
            if query_embedding_np.ndim == 1:
                query_embedding_np = query_embedding_np.reshape(1, -1)
            if use_cache and self.cache_manager.enabled and q_key is not None:
                self.cache_manager.set_embedding(q_key, query_embedding_np)
        else:
            if query_embedding_np.ndim == 1:
                query_embedding_np = query_embedding_np.reshape(1, -1)

        # 3) retrieval cache?????topK indices + sims????????cosine
        corpus_sig = getattr(self, "corpus_sig", None)
        if corpus_sig is None:
            corpus_sig = CacheManager.texts_signature(getattr(self, "texts", []))
            self.corpus_sig = corpus_sig

        retrieval_k = int(max(2048, new_top_k * 10))
        retrieval_k = min(retrieval_k, len(self.texts))

        r_key = None
        cached_retr = None
        if use_cache and self.cache_manager.enabled:
            r_key = self.cache_manager.make_key({
                "stage": "retrieval",
                "model": self.model_name,
                "corpus_sig": corpus_sig,
                "q_sha1": q_sha1,
                "k": retrieval_k,
            })
            cached_retr = self.cache_manager.get_retrieval(r_key)

        if cached_retr is None:
            similarities = cosine_similarity(query_embedding_np, self.corpus_embeddings_np)[0].astype(np.float32)
            min_sim = float(similarities.min())
            sorted_indices = _top_k_sorted_indices(similarities, retrieval_k)
            sorted_sims = similarities[sorted_indices].astype(np.float32)
            if use_cache and self.cache_manager.enabled and r_key is not None:
                self.cache_manager.set_retrieval(r_key, sorted_indices, sorted_sims, meta={"min_sim": min_sim})
        else:
            sorted_indices, sorted_sims, _meta = cached_retr

        # 4) selection??hreshold + dedup + (optional) rerank + resort
        unique_texts: List[str] = []
        unique_outputs: List[Any] = []
        seen_contents = set([query]) if deduplicate else set()

        for idx, sim_score in zip(sorted_indices.tolist(), sorted_sims.tolist()):
            if sim_score < threshold:
                continue
            content = self.texts[int(idx)]
            if deduplicate and content in seen_contents:
                continue
            if deduplicate:
                seen_contents.add(content)

            unique_texts.append(content)
            unique_outputs.append(self.test2item[content]['output'])

            if len(unique_texts) >= new_top_k:
                break

        if rerank and self.reranker and len(unique_texts) > 0:
            scores = self.reranker.rerank(query, unique_texts)
            rr_idx = np.argsort(scores)[::-1][:top_k]
            unique_texts = [unique_texts[i] for i in rr_idx]
            unique_outputs = [unique_outputs[i] for i in rr_idx]
        else:
            unique_texts = unique_texts[:top_k]
            unique_outputs = unique_outputs[:top_k]

        if resort and len(unique_texts) > 1:
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

        if use_cache:
            self.cache_manager.set(query, params, result)

        return result


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
        
        # ????????????
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
            self.text2idx = {}
            for idx, item in enumerate(datas):
                item['output'] = parsed_quad_to_raw_quad(item['quadruples'])
                self.test2item[item['content']] = item
                self.text2idx.setdefault(item['content'], idx)
            texts = self.texts

        self.corpus_sig = CacheManager.texts_signature(texts)
        emb_key = None
        if self.cache_manager.enabled:
            emb_key = self.cache_manager.make_key({"stage":"lex_corpus_embedding","model": self.model_name,"corpus_sig": self.corpus_sig})
            cached = self.cache_manager.get_embedding(emb_key)
            if cached is not None:
                self.corpus_embeddings_np = cached
                return

        corpus_embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        if hasattr(corpus_embeddings, 'is_cuda') and corpus_embeddings.is_cuda:
            corpus_embeddings = corpus_embeddings.cpu()
        self.corpus_embeddings_np = corpus_embeddings.numpy().astype(np.float32)

        if self.cache_manager.enabled and emb_key is not None:
            self.cache_manager.set_embedding(emb_key, self.corpus_embeddings_np)

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

        params = {
            "method": "similarity",
            "top_k": top_k,
            "deduplicate": deduplicate,
            "threshold": threshold,
            "model": self.model_name,
            "corpus_sig": getattr(self, "corpus_sig", None),
        }

        if use_cache:
            cached_result = self.cache_manager.get(query, params)
            if cached_result is not None:
                return cached_result

        if top_k == 0:
            return []

        q_sha1 = CacheManager.sha1_text(query)
        q_key = None
        query_embedding_np = None
        if use_cache and self.cache_manager.enabled:
            q_key = self.cache_manager.make_key({"stage": "lex_query_embedding", "model": self.model_name, "q_sha1": q_sha1})
            query_embedding_np = self.cache_manager.get_embedding(q_key)

        if query_embedding_np is None:
            query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            if hasattr(query_embedding, 'is_cuda') and query_embedding.is_cuda:
                query_embedding = query_embedding.cpu()
            query_embedding_np = query_embedding.numpy().astype(np.float32)
            if query_embedding_np.ndim == 1:
                query_embedding_np = query_embedding_np.reshape(1, -1)
            if use_cache and self.cache_manager.enabled and q_key is not None:
                self.cache_manager.set_embedding(q_key, query_embedding_np)
        else:
            if query_embedding_np.ndim == 1:
                query_embedding_np = query_embedding_np.reshape(1, -1)

        corpus_sig = getattr(self, "corpus_sig", None)
        if corpus_sig is None:
            corpus_sig = CacheManager.texts_signature(getattr(self, "texts", []))
            self.corpus_sig = corpus_sig

        retrieval_k = int(max(1024, top_k * 20))
        retrieval_k = min(retrieval_k, len(self.texts))

        r_key = None
        cached_retr = None
        if use_cache and self.cache_manager.enabled:
            r_key = self.cache_manager.make_key({
                "stage": "lex_retrieval",
                "model": self.model_name,
                "corpus_sig": corpus_sig,
                "q_sha1": q_sha1,
                "k": retrieval_k,
            })
            cached_retr = self.cache_manager.get_retrieval(r_key)

        if cached_retr is None:
            similarities = cosine_similarity(query_embedding_np, self.corpus_embeddings_np)[0].astype(np.float32)
            sorted_indices = _top_k_sorted_indices(similarities, retrieval_k)
            sorted_sims = similarities[sorted_indices].astype(np.float32)
            if use_cache and self.cache_manager.enabled and r_key is not None:
                self.cache_manager.set_retrieval(r_key, sorted_indices, sorted_sims)
        else:
            sorted_indices, sorted_sims, _meta = cached_retr

        unique_texts = []
        seen_contents = set([query]) if deduplicate else set()
        for idx, sim_score in zip(sorted_indices.tolist(), sorted_sims.tolist()):
            if sim_score < threshold:
                continue
            content = self.texts[int(idx)]
            if deduplicate and content in seen_contents:
                continue
            if deduplicate:
                seen_contents.add(content)
            unique_texts.append(content)
            if len(unique_texts) >= top_k:
                break

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
        
        if use_cache:
            self.cache_manager.set(query, params, final_result)
            
        return final_result
    

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
        
        # ????????????????????
        sorted_indices = np.argsort(similarities)[::-1]
        
        unique_texts = []
        unique_outputs = []
        seen_contents = set() if not deduplicate else set([query])  # ???????????????
        
        for idx in sorted_indices:
            content = self.texts[idx]
            sim_score = similarities[idx]  # ??????????????            
            # ??????????????????????????
            if sim_score < threshold:
                continue  # ??????????????
                
            # ??????
            if deduplicate:
                if content in seen_contents:
                    continue
                seen_contents.add(content)
            
            unique_texts.append(content)
            unique_outputs.append(self.test2item[content]['output'])
            
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
            ramdom_strategy: Literal["none", "sample", "hybrid"] = "none",
            random_state: int = 42,
            cache_dir: str = "./cache",
            enable_cache: bool = True,
            target_groups: Optional[List[str]] = None,
            default_weights: Optional[Dict[str, float]] = None):

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

        self.ramdom_strategy = ramdom_strategy
        self.random_state = random_state
        self.target_groups = normalize_target_groups(target_groups)
        self.default_weights = normalize_weights(default_weights, self.target_groups)

        # ????????????
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

        for targeted_group in self.target_groups:
            new_data_list = []
            for data in loaded_data_list:
                if any(quad_has_group(quadruple, targeted_group) for quadruple in data["quadruples"]):
                    new_data_list.append(data)
            self.class_data_dict[targeted_group] = new_data_list
    
    def build_retrievers(self):
        """Build one retriever for each configured target group."""
        self.retrievers: dict[str, Retriever | StochasticWeightedRetriever] = {}

        common = dict(
            model=self.model,
            model_name=self.model_name,
            device=self.device,
            cache_dir=self.cache_manager.cache_dir,
            enable_cache=self.cache_manager.enabled,
        )

        for class_name in self.class_data_dict.keys():
            if not self.class_data_dict[class_name]:
                continue
            if self.ramdom_strategy == "none":
                retriever = Retriever(**common)
            else:
                retriever = StochasticWeightedRetriever(
                    random_state=self.random_state,
                    **common,
                )

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
            similarity_alpha: float = 1.0,
            random_strategy: Literal["none", "sample", "hybrid"] = "none", 
            random_ratio: float = 0.3,
            temperature: float = 1.0,
            candidate_multiplier: float = 3.0,
            use_cache: bool = True,
            **kwargs
            ) -> tuple[list[str], list[str]]:
        
        params = {
            "top_k": top_k,
            "deduplicate": deduplicate,
            "threshold": threshold,
            "weights": weights,
            "weights_reverse": weights_reverse,
            "target_groups": self.target_groups,
            "similarity_alpha": similarity_alpha,
            "random_strategy": random_strategy,
            "random_ratio": random_ratio,
            "temperature": temperature,
            "candidate_multiplier": candidate_multiplier
        }
        
        try:
            parts = []
            for _cls, _ret in getattr(self, 'retrievers', {}).items():
                _sig = getattr(_ret, 'corpus_sig', None)
                if _sig is not None:
                    parts.append(f'{_cls}:{_sig}')
            composite_sig = CacheManager.sha1_text('|'.join(sorted(parts))) if parts else None
        except Exception:
            composite_sig = None

        params.update({
            'model': getattr(self, 'model_name', None),
            'corpus_sig': composite_sig,
        })

        if use_cache:
            cached_result = self.cache_manager.get(query, params)
            if cached_result is not None:
                # logger.debug(f"Cache hit for query: {query}")
                return cached_result
        
        if weights is None:
            weights = self.default_weights
        else:
            weights = normalize_weights(weights, self.target_groups)

        allocated_class_top_k = allocate_class_num(top_k, weights, weights_reverse)
        # logger.debug(f"Allocated top_k per class: {allocated_class_top_k}")
        all_texts = []
        all_outputs = []

        for class_name in self.target_groups:
            if class_name not in self.retrievers:
                continue
            class_top_k = allocated_class_top_k.get(class_name, 0)
            if class_top_k == 0:
                continue
            texts, outputs = self.retrievers[class_name].retrieve(
                query=query, 
                top_k=class_top_k, 
                deduplicate=deduplicate, 
                threshold=threshold, 
                use_cache=False,
                similarity_alpha=similarity_alpha,
                random_strategy=random_strategy,
                random_ratio=random_ratio,
                temperature=temperature,
                candidate_multiplier=candidate_multiplier
                )
            all_texts.extend(texts)
            all_outputs.extend(outputs)

        result = (all_texts, all_outputs)
        
        if use_cache:
            self.cache_manager.set(query, params, result)
            
        return result

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
        
        # ????????????????????
        sorted_indices = np.argsort(similarities)[::-1]
        
        unique_texts = []
        unique_outputs = []
        unique_wrong_exps = []
        seen_contents = set() if not deduplicate else set([query])  # ???????????????
        
        for idx in sorted_indices:
            content = self.texts[idx]
            sim_score = similarities[idx]  # ??????????????            
            # ??????????????????????????
            if sim_score < threshold:
                continue  # ??????????????
                
            # ??????
            if deduplicate:
                if content in seen_contents:
                    continue
                seen_contents.add(content)
            
            unique_texts.append(content)
            unique_outputs.append(self.text2item[content]['output'])
            unique_wrong_exps.append(self.text2wrong_exp.get(content, None))
            
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
            device: str = "cuda:0",
            target_groups: Optional[List[str]] = None,
            default_weights: Optional[Dict[str, float]] = None):

        logger.info(f"Loading model from path: {model_path}")
        self.model = SentenceTransformer(model_path).to(device)
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.target_groups = normalize_target_groups(target_groups)
        self.default_weights = normalize_weights(default_weights, self.target_groups)

        self.reranker = None
        self.load_datas(data_list, result_data_list)
        self.build_retrievers()

    def load_datas(self, data_list: list[dict], result_data_list: list[dict]):

        self.class_data_dict = {}
        self.class_result_data_dict = {}

        for targeted_group in self.target_groups:
            new_data_list = []
            new_result_data_list = []
            for data in data_list:
                if any(quad_has_group(quadruple, targeted_group) for quadruple in data["quadruples"]):
                    new_data_list.append(data)
            for data in result_data_list:
                if any(quad_has_group(quadruple, targeted_group) for quadruple in data["gt_quadruples"]):
                    new_result_data_list.append(data)
            self.class_data_dict[targeted_group] = new_data_list
            self.class_result_data_dict[targeted_group] = new_result_data_list
    
    def build_retrievers(self):
        self.retrievers: dict[str, WrongExpRetriever] = {}
        for class_name in self.class_data_dict.keys():
            if not self.class_data_dict[class_name]:
                continue
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
            weights = self.default_weights
        else:
            weights = normalize_weights(weights, self.target_groups)

        allocated_class_top_k = allocate_class_num(top_k, weights, weights_reverse)
        all_texts = []
        all_outputs = []
        all_wrong_exps = []

        for class_name in self.target_groups:
            if class_name not in self.retrievers:
                continue
            class_top_k = allocated_class_top_k.get(class_name, 0)
            if class_top_k == 0:
                continue
            texts, outputs, wrong_exps = self.retrievers[class_name].retrieve(query, class_top_k, deduplicate, threshold)
            all_texts.extend(texts)
            all_outputs.extend(outputs)
            all_wrong_exps.extend(wrong_exps)

        return all_texts, all_outputs, all_wrong_exps

from sklearn.cluster import KMeans

class ClusteredRetriever:
    """Retriever that allocates demonstrations across learned clusters."""

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
        """Initialize a clustered retriever."""
        # ???????????????
        if model:
            self.model = model
            self.model_name = model_name
            self.device = device
        else:
            logger.info(f"Loading model from path: {model_path}")
            self.model = SentenceTransformer(model_path).to(device)
            self.model_name = model_name
            self.device = device

        # ??????
        self.cache_manager = CacheManager(cache_dir, enable_cache)

        
        self.n_clusters = n_clusters
        self.random_state = random_state

        # ???????????? retriever?????????
        if data_path:
            self._load_datas(data_path)
            self._build_global_retriever()

            # ??? & ??cluster ??????
            self._build_clusters(cluster_model)

            # ?????? cluster ?????? Retriever
            self._build_cluster_retrievers()
    
    def _load_datas(self, data_path: Optional[str] = None, data_list: Optional[list[dict]] = None) -> None:
        """Load corpus data."""
        if data_list is None:
            if data_path is None:
                raise ValueError("Either data_path or data_list must be provided.")
            data: list[dict] = load_json(data_path)
        else:
            data = data_list

        self.data = data

    def _build_global_retriever(self) -> None:
        """Build the shared corpus retriever."""
        self.global_retriever = Retriever(
            model=self.model,
            model_name=self.model_name,
            cache_dir=self.cache_manager.cache_dir,
            enable_cache=self.cache_manager.enabled,
        )
        self.global_retriever.create_embeddings(self.data)
        self.texts = self.global_retriever.texts
        self.test2item = self.global_retriever.test2item
        self.corpus_embeddings_np = self.global_retriever.corpus_embeddings_np

    def _build_clusters(self, cluster_model: Optional[KMeans] = None) -> None:
        """Cluster corpus embeddings and map clusters to sample indices."""

        logger.info("Clustering corpus embeddings for ClusteredRetriever")

        if cluster_model is None:
            self.cluster_model = KMeans(
                n_clusters=min(self.n_clusters, max(1, len(self.data))),
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

        self.cluster_ids: List[int] = sorted(self.cluster2indices.keys())

    def _build_cluster_retrievers(self) -> None:
        """Build a retriever for each cluster."""
        self.cluster_retrievers: Dict[int, Retriever] = {}

        for c in self.cluster_ids:
            indices = self.cluster2indices[c]
            cluster_data = [self.data[i] for i in indices]

            retriever = Retriever(
                model=self.model,
                model_name=self.model_name,
                enable_cache=False  # ??retriever ????????????????????????
            )
            retriever.create_embeddings(cluster_data)
            self.cluster_retrievers[c] = retriever

    def _default_cluster_weights(self) -> Dict[int, float]:
        """Use cluster sizes as default quota weights."""
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
        """Retrieve examples using cluster quota allocation."""
        # ???????key
        params = {
            "top_k": top_k,
            "deduplicate": deduplicate,
            "threshold": threshold,
            "weights": weights,
            "weights_reverse": weights_reverse,
            "n_clusters": self.n_clusters
        }

        if use_cache:
            cached_result = self.cache_manager.get(query, params)
            if cached_result is not None:
                return cached_result

        if top_k == 0:
            return [], []

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

            texts, outputs = self.cluster_retrievers[c].retrieve(
                query=query,
                top_k=cluster_top_k,
                deduplicate=deduplicate,
                threshold=threshold,
                use_cache=False,   # ????????????
                **kwargs
            )
            all_texts.extend(texts)
            all_outputs.extend(outputs)

        result = (all_texts, all_outputs)

        if use_cache:
            self.cache_manager.set(query, params, result)

        return result
    
class StochasticWeightedRetriever(Retriever):
    """Retriever with stochastic similarity-based selection."""

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
        """Transform similarities before sampling."""
        if similarity_alpha <= 0:
            raise ValueError("similarity_alpha must be > 0")
        if similarity_alpha == 1.0:
            return similarities
        # ????????????????????????????????????????????
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
        """Select candidate indices with deterministic or stochastic strategies."""
        if len(candidate_indices) == 0:
            return []

        random_strategy = random_strategy.lower()
        random_ratio = min(max(random_ratio, 0.0), 1.0)
        temperature = max(temperature, 1e-6)

        # ????????oftmax(scores / temperature)
        scores = scores.astype(np.float64)
        max_score = np.max(scores)
        prob = np.exp((scores - max_score) / temperature)
        prob_sum = prob.sum()
        if prob_sum == 0:
            prob = np.ones_like(prob) / len(prob)
        else:
            prob = prob / prob_sum

        n = min(top_k, len(candidate_indices))

        if random_strategy == "none":
            # ??? deterministic??????????top-k
            sorted_local_idx = np.argsort(scores)[::-1][:n]
            return sorted_local_idx.tolist()

        elif random_strategy == "random":
            # ???????????????
            chosen = self.random_state.choice(
                len(candidate_indices),
                size=n,
                replace=False
            )
            return chosen.tolist()

        elif random_strategy == "sample":
            # ??????????????????
            chosen = self.random_state.choice(
                len(candidate_indices),
                size=n,
                replace=False,
                p=prob
            )
            return chosen.tolist()

        elif random_strategy == "hybrid":
            deterministic_k = int(round(n * (1.0 - random_ratio)))
            deterministic_k = max(0, min(deterministic_k, n))
            sampled_k = n - deterministic_k

            sorted_local_idx = np.argsort(scores)[::-1]
            deterministic_part = sorted_local_idx[:deterministic_k].tolist()

            if sampled_k <= 0:
                return deterministic_part

            # ???????????????
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
            # ??????
            similarity_alpha: float = 1.0,
            random_strategy: Literal["none", "sample", "hybrid"] = "none", 
            random_ratio: float = 0.3,
            temperature: float = 1.0,
            candidate_multiplier: float = 3.0,
            **kwargs
    ) -> tuple[list[str], list[str]]:
        """Retrieve examples with optional stochastic candidate selection."""

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
            "model": getattr(self, "model_name", None),
            "corpus_sig": getattr(self, "corpus_sig", None),
        }

        # selection cache?????use_cache=True ??????
        if use_cache:
            cached_result = self.cache_manager.get(query, params)
            if cached_result is not None:
                return cached_result

        if top_k == 0:
            return [], []

        # rerank ???????????
        if rerank and self.reranker:
            base_candidate_k = max(10, min(100, top_k * 5))
        else:
            base_candidate_k = top_k

        candidate_k = int(max(base_candidate_k, top_k * candidate_multiplier))
        candidate_k = min(candidate_k, len(self.texts))

        q_sha1 = CacheManager.sha1_text(query)
        q_key = None
        query_embedding_np = None
        query_idx = getattr(self, "text2idx", {}).get(query)
        if query_idx is not None and hasattr(self, "corpus_embeddings_np"):
            query_embedding_np = self.corpus_embeddings_np[int(query_idx):int(query_idx) + 1]
        elif use_cache and self.cache_manager.enabled:
            q_key = self.cache_manager.make_key({"stage": "query_embedding", "model": self.model_name, "q_sha1": q_sha1})
            query_embedding_np = self.cache_manager.get_embedding(q_key)

        if query_embedding_np is None:
            query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            if hasattr(query_embedding, 'is_cuda') and query_embedding.is_cuda:
                query_embedding = query_embedding.cpu()
            query_embedding_np = query_embedding.numpy().astype(np.float32)
            if query_embedding_np.ndim == 1:
                query_embedding_np = query_embedding_np.reshape(1, -1)
            if use_cache and self.cache_manager.enabled and q_key is not None:
                self.cache_manager.set_embedding(q_key, query_embedding_np)
        else:
            if query_embedding_np.ndim == 1:
                query_embedding_np = query_embedding_np.reshape(1, -1)

        # 2) retrieval cache?????top-k indices/sims + min_sim
        corpus_sig = getattr(self, "corpus_sig", None)
        if corpus_sig is None:
            corpus_sig = CacheManager.texts_signature(getattr(self, "texts", []))
            self.corpus_sig = corpus_sig

        retrieval_k = int(max(2048, candidate_k * 10))
        retrieval_k = min(retrieval_k, len(self.texts))

        r_key = None
        cached_retr = None
        if use_cache and self.cache_manager.enabled:
            r_key = self.cache_manager.make_key({
                "stage": "retrieval",
                "model": self.model_name,
                "corpus_sig": corpus_sig,
                "q_sha1": q_sha1,
                "k": retrieval_k,
            })
            cached_retr = self.cache_manager.get_retrieval(r_key)

        if cached_retr is None:
            similarities = cosine_similarity(query_embedding_np, self.corpus_embeddings_np)[0].astype(np.float32)
            min_sim = float(similarities.min())
            sorted_indices = _top_k_sorted_indices(similarities, retrieval_k)
            sorted_sims = similarities[sorted_indices].astype(np.float32)
            if use_cache and self.cache_manager.enabled and r_key is not None:
                self.cache_manager.set_retrieval(r_key, sorted_indices, sorted_sims, meta={"min_sim": min_sim})
        else:
            sorted_indices, sorted_sims, meta = cached_retr
            if isinstance(meta, dict) and 'min_sim' in meta:
                min_sim = float(meta['min_sim'])
            else:
                min_sim = float(min(sorted_sims.tolist())) if len(sorted_sims) else 0.0

        # 3) selection??hreshold + dedup + scoring?????? shape ?????
        unique_texts: List[str] = []
        unique_outputs: List[Any] = []
        scores_after_transform: List[float] = []

        seen_contents = set([query]) if deduplicate else set()

        for idx, sim_score in zip(sorted_indices.tolist(), sorted_sims.tolist()):
            if sim_score < threshold:
                continue

            content = self.texts[int(idx)]
            if deduplicate and content in seen_contents:
                continue

            if deduplicate:
                seen_contents.add(content)

            unique_texts.append(content)
            unique_outputs.append(self.test2item[content]['output'])

            # transform(similarity) (?????_transform_similarities ????????
            val = sim_score - min_sim
            if val < 0:
                val = 0.0
            scores_after_transform.append(float(val ** similarity_alpha))

            if len(unique_texts) >= candidate_k:
                break

        if len(unique_texts) == 0:
            result = ([], [])
            if use_cache:
                self.cache_manager.set(query, params, result)
            return result

        # 4) rerank???????????????????? scores
        if rerank and self.reranker:
            scores = self.reranker.rerank(query, unique_texts)
            scores = np.array(scores, dtype=np.float64)
        else:
            scores = np.array(scores_after_transform, dtype=np.float64)

        # 5) ??? random_strategy??????????????????top_k
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

        # 6) resort??????
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



if __name__ == "__main__":
    lex_retriever = LexiconRetriever(model_path="./models/base/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/lexicon/annotated_lexicon.json", enable_cache=False)
    # print(retriever.including_retrieve("????????????????????????????????????????, top_k=-1))
    # retriever = StepOneRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/full/std/train.json")
    # print(retriever.retrieve("????????????????????????????????????????, top_k=5, threshold=0.5))
    import json
    retriever = MultiClassRetriever(model_path="./models/base/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_path="data/full/std/train.json")
    data_list = load_json("data/full/std/train.json")
    result_data_list = load_json("runner/output/simlex5_rag9_multi_class.json")["results"]
    # retriever = MultiClassWrongExpRetriever(model_path="./models/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5", data_list=data_list, result_data_list=result_data_list)
    # print(json.dumps(retriever.retrieve("????????????????????????????????????????, top_k=9), ensure_ascii=False, indent=2))

    # retriever = StochasticWeightedRetriever(random_state=42, model_path="./models/base/bge-large-zh-v1.5", model_name="bge-large-zh-v1.5")
    retriever.build_retrievers()
    texts, outputs = retriever.retrieve(
        "debug query",
        top_k=5,
        similarity_alpha=0.5,
        random_strategy="hybrid",
        random_ratio=0.6,
        temperature=2.0,
        candidate_multiplier=4.0
    )
    print(json.dumps({"texts": texts, "outputs": outputs}, ensure_ascii=False, indent=2))

    include_results = lex_retriever.including_retrieve(
        "debug query",
        top_k=5,
        use_cache=False
    )
    print(json.dumps(include_results, ensure_ascii=False, indent=2))

    sim_results = lex_retriever.similarity_retrieve(
        "debug query",
        top_k=5,
        use_cache=False
    )
    print(json.dumps(sim_results, ensure_ascii=False, indent=2))

