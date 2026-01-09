from __future__ import annotations
import os
import json
import faiss
import numpy as np
from typing import List, Tuple

from rag.tools import ensure_dir

class FaissHNSWIndex:
    """
    FAISS IndexHNSWFlat with inner product (for cosine on L2-normalized vectors).
    Stores:
      - index: faiss index
      - id_map: internal_idx -> doc_id (string)
      - vectors are not stored in python if you rely on FAISS only,
        but for MMR you'll want candidate vectors; we'll load from external store or keep in RAM.
    """
    def __init__(self, dim: int, m: int = 32, ef_construction: int = 200, metric: str = "ip"):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction

        if metric == "ip":
            self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError("Only inner-product supported in this implementation (cosine via L2 norm).")

        self.index.hnsw.efConstruction = ef_construction
        self.id_map: List[str] = []   # internal index -> doc_id

    def set_ef_search(self, ef_search: int):
        self.index.hnsw.efSearch = ef_search

    def add(self, vectors: np.ndarray, doc_ids: List[str]):
        assert vectors.dtype == np.float32
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        assert len(doc_ids) == vectors.shape[0]
        self.index.add(vectors)
        self.id_map.extend(doc_ids)

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[List[str], np.ndarray]:
        assert query_vec.dtype == np.float32
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        D, I = self.index.search(query_vec, top_k)
        # D are inner products, I are internal ids
        ids = []
        sims = []
        for internal_id, sim in zip(I[0], D[0]):
            if internal_id < 0:
                continue
            ids.append(self.id_map[int(internal_id)])
            sims.append(float(sim))
        return ids, np.array(sims, dtype=np.float32)

    def save(self, path: str):
        ensure_dir(os.path.dirname(path) or ".")
        faiss.write_index(self.index, path)
        with open(path + ".idmap.json", "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "FaissHNSWIndex":
        index = faiss.read_index(path)
        with open(path + ".idmap.json", "r", encoding="utf-8") as f:
            id_map = json.load(f)
        dim = index.d
        obj = FaissHNSWIndex(dim=dim, m=32, ef_construction=200, metric="ip")
        obj.index = index
        obj.id_map = id_map
        return obj