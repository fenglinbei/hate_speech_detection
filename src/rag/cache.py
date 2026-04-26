from __future__ import annotations

import io
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np

from rag.tool import ensure_dir
from utils.sqlite_kv_cache import SQLiteKVCache


class CacheManager:
    """
    SQLite-backed cache for the MMR retrieval pipeline.

    Stages:
      - embedding: hash(text) -> np.float32 vector
      - retrieval: query_key -> JSON-serializable retrieval object
      - selection: selection key -> JSON-serializable selected shots
    """

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        ensure_dir(cache_dir)
        self.kv = SQLiteKVCache(os.path.join(cache_dir, "retrieval_cache.sqlite3"), enabled=enabled)

    def get_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        value = self.kv.get("embedding", text_hash)
        if value is None:
            return None
        with io.BytesIO(value) as bio:
            data = np.load(bio, allow_pickle=False)
            return data["vec"].astype(np.float32)

    def set_embedding(self, text_hash: str, vec: np.ndarray):
        if not self.enabled:
            return
        with io.BytesIO() as bio:
            np.savez_compressed(bio, vec=vec.astype(np.float32))
            self.kv.set("embedding", text_hash, bio.getvalue())

    def get_retrieval(self, query_key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        value = self.kv.get("retrieval", query_key)
        if value is None:
            return None
        return pickle.loads(value)

    def set_retrieval(self, query_key: str, obj: Dict[str, Any]):
        if not self.enabled:
            return
        self.kv.set("retrieval", query_key, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def get_selection(self, sel_key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        value = self.kv.get("selection", sel_key)
        if value is None:
            return None
        return pickle.loads(value)

    def set_selection(self, sel_key: str, obj: Dict[str, Any]):
        if not self.enabled:
            return
        self.kv.set("selection", sel_key, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def close(self) -> None:
        self.kv.close()
