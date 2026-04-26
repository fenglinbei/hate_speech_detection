from __future__ import annotations

import os
import hashlib
import numpy as np

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        x = x[None, :]
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)

def cosine_from_ip(ip: np.ndarray) -> np.ndarray:
    # When vectors are L2-normalized, inner product == cosine similarity.
    return ip