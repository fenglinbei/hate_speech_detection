# rag_retrieval_pipeline.py
from __future__ import annotations

import os
import json
import numpy as np
from typing import Any, Dict, Optional

from rag.tool import ensure_dir

class CacheManager:
    """
    Simple disk cache:
      - embedding cache: hash(text)->np.float32 vector
      - retrieval cache: query_key->topK ids + sims
      - selection cache: (query_key, params_sig)->selected ids
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)
        self.emb_dir = os.path.join(cache_dir, "emb")
        self.ret_dir = os.path.join(cache_dir, "retrieval")
        self.sel_dir = os.path.join(cache_dir, "selection")
        ensure_dir(self.emb_dir)
        ensure_dir(self.ret_dir)
        ensure_dir(self.sel_dir)

    def _path(self, subdir: str, key: str) -> str:
        return os.path.join(subdir, f"{key}.npz")

    def get_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        p = self._path(self.emb_dir, text_hash)
        if not os.path.exists(p):
            return None
        data = np.load(p)
        return data["vec"].astype(np.float32)

    def set_embedding(self, text_hash: str, vec: np.ndarray):
        p = self._path(self.emb_dir, text_hash)
        np.savez_compressed(p, vec=vec.astype(np.float32))

    def get_retrieval(self, query_key: str) -> Optional[Dict[str, Any]]:
        p = os.path.join(self.ret_dir, f"{query_key}.json")
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def set_retrieval(self, query_key: str, obj: Dict[str, Any]):
        p = os.path.join(self.ret_dir, f"{query_key}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    def get_selection(self, sel_key: str) -> Optional[Dict[str, Any]]:
        p = os.path.join(self.sel_dir, f"{sel_key}.json")
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def set_selection(self, sel_key: str, obj: Dict[str, Any]):
        p = os.path.join(self.sel_dir, f"{sel_key}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)