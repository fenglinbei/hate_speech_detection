from __future__ import annotations

import numpy as np
from typing import Optional, List
from sentence_transformers import SentenceTransformer

from rag.cache import CacheManager
from rag.tools import sha1_text, l2_normalize


class STEmbedder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        batch_size: int = 64,
        cache: Optional[CacheManager] = None,
        normalize: bool = True,
    ):
        self.model = SentenceTransformer(model_path, device=device)
        self.batch_size = batch_size
        self.cache = cache
        self.normalize = normalize

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        # cache-aware batch encoding
        vecs = []
        missing_idx = []
        missing_texts = []

        for i, t in enumerate(texts):
            h = sha1_text(t)
            if self.cache is not None:
                cached = self.cache.get_embedding(h)
                if cached is not None:
                    vecs.append(cached)
                    continue
            vecs.append(None)
            missing_idx.append(i)
            missing_texts.append(t)

        if missing_texts:
            new_vecs = self.model.encode(
                missing_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,   # we normalize ourselves for safety
            ).astype(np.float32)

            if self.normalize:
                new_vecs = l2_normalize(new_vecs)

            # fill back
            for j, i in enumerate(missing_idx):
                vecs[i] = new_vecs[j]
                if self.cache is not None:
                    self.cache.set_embedding(sha1_text(texts[i]), vecs[i])

        vecs = np.vstack(vecs).astype(np.float32)
        if self.normalize:
            vecs = l2_normalize(vecs)
        return vecs

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]