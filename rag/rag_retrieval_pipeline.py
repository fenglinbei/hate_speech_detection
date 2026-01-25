# rag_retrieval_pipeline.py
from __future__ import annotations

import os
import json
import time
import math
import hashlib
import random
from tqdm import tqdm
from loguru import logger
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rag.cache import CacheManager
from rag.embedder import STEmbedder
from rag.faiss_hnsw_index import FaissHNSWIndex
from rag.filters import mmr_select, filter_with_fallback
from rag.config import RetrievalParams
from rag.tool import sha1_text, ensure_dir, l2_normalize
from tools.convert import parsed_quad_to_trip, parsed_quad_to_raw_quad


def params_signature(params: RetrievalParams, extra: Optional[Dict[str, Any]] = None) -> str:
    d = asdict(params)
    if extra:
        d.update(extra)
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# =========================
# 6) Prompt Packer
# =========================

def pack_nshot_prompt(query_text: str, query_label: Optional[str], shots: List[Dict[str, Any]], template_id: str) -> str:
    """
    You can adapt this to your existing prompt style.
    """
    if template_id == "basic":
        # Simple classification instruction
        lines = []
        lines.append("你是一个文本分类器。请根据示例，判断输入文本的标签。")
        lines.append("")
        for i, s in enumerate(shots, 1):
            lines.append(f"示例{i}:")
            lines.append(f"文本: {s['text']}")
            if s.get("label") is not None:
                lines.append(f"标签: {s['label']}")
            lines.append("")
        lines.append("现在请判断：")
        lines.append(f"文本: {query_text}")
        lines.append("标签:")
        return "\n".join(lines)

    # You can extend with rationale, table format, etc.
    return pack_nshot_prompt(query_text, query_label, shots, "basic")


# =========================
# 7) Main Retrieval Pipeline
# =========================

class DiversityRetrievalPipeline:
    """
    End-to-end:
      - embed
      - faiss search
      - filter by sim range
      - MMR selection
      - pack prompt
      - cache + trace
    """

    def __init__(
        self,
        params: RetrievalParams,
        embedder: STEmbedder,
        index: FaissHNSWIndex,
        doc_store: Dict[str, Dict[str, Any]],
        vector_store: Optional[Dict[str, np.ndarray]] = None,
        cache: Optional[CacheManager] = None,
    ):
        self.params = params
        self.embedder = embedder
        self.index = index
        self.doc_store = doc_store
        self.vector_store = vector_store  # doc_id -> embedding
        self.cache = cache or CacheManager(params.cache_dir)

        # ensure efSearch
        self.index.set_ef_search(params.ef_search)

        self.rng = random.Random(params.seed)

        # trace file
        if params.trace_jsonl_path:
            ensure_dir(os.path.dirname(params.trace_jsonl_path) or ".")

    def _write_trace(self, record: Dict[str, Any]):
        if not self.params.trace_jsonl_path:
            return
        with open(self.params.trace_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def retrieve_shots(
            self, query_id: str, 
            query_text: str, 
            query_label: Optional[Any], 
            n_shot: Optional[int],
            mmr_lambda: Optional[float]) -> Dict[str, Any]:
        t0 = time.time()

        if n_shot is not None:
            self.params.n_shot = n_shot

        if mmr_lambda is not None:
            self.params.mmr_lambda = mmr_lambda

        # ---------- cache keys (optional)
        qkey = sha1_text(str(query_id) + "|" + query_text)
        psig = sha1_text(json.dumps(self.params.__dict__, sort_keys=True, ensure_ascii=False))
        sel_key = hashlib.md5((qkey + "|" + psig).encode("utf-8")).hexdigest()

        if self.cache is not None:
            cached = self.cache.get_selection(sel_key)
            if cached is not None:
                shots = cached.get("shots", [])
                prompt = pack_nshot_prompt(query_text, query_label, shots, self.params.template_id)
                return {"shots": shots, "prompt": prompt, "trace": {"cached": True, "sel_key": sel_key}}

        # ---------- embed query
        qvec = self.embedder.encode_one(query_text).astype(np.float32)
        qvec = l2_normalize(qvec).astype(np.float32)  # ensure

        # ---------- retrieve topK
        cand_ids, cand_sims = self.index.search(qvec, self.params.top_k)
        cand_sims = np.array(cand_sims, dtype=np.float32)

        # ---------- filter + fallback (critical insertion point)
        kept_ids, filter_trace, fb_meta = filter_with_fallback(
            query_id=str(query_id),
            query_label=query_label,
            cand_ids=[str(x) for x in cand_ids],            # enforce str id
            cand_sims=cand_sims,
            doc_store=self.doc_store,
            sim_min=self.params.sim_min,
            sim_max=self.params.sim_max,
            n_shot=self.params.n_shot,
            label_policy=self.params.label_policy,
            allow_cross_label_ratio=self.params.allow_cross_label_ratio,
            # (optional) if you later want to re-search larger K:
            # search_more_fn=lambda new_k: self.index.search(qvec, new_k),
            top_k_stages=(0,),
            quantile_q=0.70,
            global_sim_floor=0.10,
            relax_sim_max_to=0.98,
            keep_at_least=1,
        )

        # ---------- build cand_objs
        sim_map = {str(cid): float(s) for cid, s in zip([str(x) for x in cand_ids], cand_sims.tolist())}
        cand_objs: List[Dict[str, Any]] = []
        for cid in kept_ids:
            d = self.doc_store.get(cid)
            if d is None:
                continue
            cand_objs.append({
                "doc_id": cid,
                "text": d["text"],
                "label": d.get("label"),
                "meta": d.get("meta", {}),
                "sim": sim_map.get(cid, 0.0),
            })

        # ---------- final guard: if still empty, use zero-shot or skip sample
        if len(cand_objs) == 0:
            prompt = self._pack_zero_shot_prompt(query_text)
            trace = {
                "query_id": query_id,
                "query_label": query_label,
                "fallback": fb_meta,
                "filter_trace": filter_trace,
                "selected": [],
                "latency_ms": int((time.time() - t0) * 1000),
                "note": "empty_candidates_after_fallback_zero_shot",
            }
            return {"shots": [], "prompt": prompt, "trace": trace}

        # ---------- candidate vectors (for MMR)
        if self.vector_store is not None:
            cand_vecs = np.vstack([self.vector_store[c["doc_id"]] for c in cand_objs]).astype(np.float32)
        else:
            cand_vecs = self.embedder.encode_texts([c["text"] for c in cand_objs]).astype(np.float32)

        cand_vecs = l2_normalize(cand_vecs).astype(np.float32)

        # ---------- MMR select
        sel_indices = mmr_select(
            query_vec=qvec[0],                       # (dim,)
            candidates=cand_objs,
            cand_vecs=cand_vecs,
            n_shot=min(self.params.n_shot, len(cand_objs)),
            lambda_mmr=self.params.mmr_lambda,
            allow_cross_label_ratio=self.params.allow_cross_label_ratio,
            query_label=query_label,
            seed=self.params.seed,
        )
        shots = [cand_objs[i] for i in sel_indices] if sel_indices else cand_objs[:min(self.params.n_shot, len(cand_objs))]

        # ---------- pack prompt
        prompt = pack_nshot_prompt(query_text, query_label, shots, self.params.template_id)

        # ---------- trace
        trace = {
            "query_id": query_id,
            "query_label": query_label,
            "fallback": fb_meta,
            "filter_trace": filter_trace,
            "stats": {
                "retrieved": len(cand_ids),
                "kept_after_fallback": len(cand_objs),
                "selected": len(shots),
                "top_sim": float(np.max(cand_sims)) if cand_sims.size else None,
                "mean_sim": float(np.mean(cand_sims)) if cand_sims.size else None,
                "min_sim": float(np.min(cand_sims)) if cand_sims.size else None,
            },
            "selected": [{"doc_id": s["doc_id"], "sim": s["sim"], "label": s.get("label")} for s in shots],
            "latency_ms": int((time.time() - t0) * 1000),
        }

        # ---------- cache selection
        if self.cache is not None:
            self.cache.set_selection(sel_key, {"shots": shots})

        return {"shots": shots, "prompt": prompt, "trace": trace}


# =========================
# 8) Index Build / Load Helpers
# =========================

def build_faiss_hnsw_index(
    embedder: STEmbedder,
    docs: List[Dict[str, Any]],
    params: RetrievalParams,
    index_path: str,
    store_vectors_in_ram: bool = True,
) -> Tuple[FaissHNSWIndex, Dict[str, Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
    """
    docs: list of dicts with at least { "id": str, "text": str, "label": ..., "meta": ... }
    """
    ensure_dir(os.path.dirname(index_path) or ".")
    cache = embedder.cache

    doc_ids = [str(d["id"]) for d in docs]
    texts = [d["content"] for d in docs]
    vecs = embedder.encode_texts(texts).astype(np.float32)
    vecs = l2_normalize(vecs)

    dim = vecs.shape[1]
    index = FaissHNSWIndex(dim=dim, m=params.hnsw_m, ef_construction=params.ef_construction, metric="ip")
    index.add(vecs, doc_ids)
    index.set_ef_search(params.ef_search)
    index.save(index_path)

    # doc store
    doc_store = {str(d["id"]): {"text": d["content"], "label": d.get("label"), "meta": d.get("meta", {})} for d in docs}

    vector_store = None
    if store_vectors_in_ram:
        vector_store = {doc_id: vecs[i] for i, doc_id in enumerate(doc_ids)}

    return index, doc_store, vector_store


# =========================
# 9) Example: end-to-end usage
# =========================

def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Expect a list of items:
      { "id": "...", "text": "...", "label": "...", "meta": {...} }
    Modify here to adapt your existing json structure.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # If your json is dict or nested, adapt accordingly.
    return data

RETRIEVAL_PARAMS = RetrievalParams(
    cache_dir="./cache_retrieval",
    trace_jsonl_path="./cache_retrieval/selection_trace.jsonl",
    top_k=256,
    n_shot=11,
    sim_min=0.45,
    sim_max=0.90,
    mmr_lambda=0.75,
    label_policy="soft",
    allow_cross_label_ratio=0.15,
    hnsw_m=32,
    ef_construction=200,
    ef_search=2048,
    seed=42,
    template_id="basic",
)

def main_build_index(
        data_path: str, 
        model_path: str, 
        index_path: str = "./cache_retrieval/faiss_hnsw.index",
        docs_path: str = "./cache_retrieval/doc_store.json",
        params: RetrievalParams = RETRIEVAL_PARAMS,
        ):

    cache = CacheManager(params.cache_dir)
    embedder = STEmbedder(model_path=model_path, device="cuda:0", batch_size=64, cache=cache, normalize=True)

    docs = load_json_dataset(data_path)

    index, doc_store, vector_store = build_faiss_hnsw_index(
        embedder=embedder,
        docs=docs,
        params=params,
        index_path=index_path,
        store_vectors_in_ram=True,   # faster MMR
    )

    # Save doc_store for later
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(doc_store, f, ensure_ascii=False)

    print("Index built:", index_path, "docs:", len(doc_store))


class MMRReterever:
    def __init__(
        self,
        data_path: str,
        model_path: str,
        index_path: str,
        docs_path: str,
        params: RetrievalParams = RETRIEVAL_PARAMS,
    ):
        cache = CacheManager(params.cache_dir)
        embedder = STEmbedder(model_path=model_path, device="cuda:0", batch_size=64, cache=cache, normalize=True)

        self.data = load_json_dataset(data_path)
        self.data_dict = {}
        for item in self.data:
            self.data_dict[str(item["id"])] = item

        # load index + doc_store
        index = FaissHNSWIndex.load(index_path)
        with open(docs_path, "r", encoding="utf-8") as f:
            doc_store = json.load(f)

        # optional: load vectors into RAM for faster MMR (recommended if memory allows)
        # If you want to reconstruct from scratch, re-encode doc_store texts.
        # Here we choose to not load (None) to keep demo minimal.
        vector_store = None

        self.pipeline = DiversityRetrievalPipeline(
            params=params,
            embedder=embedder,
            index=index,
            doc_store=doc_store,
            vector_store=vector_store,
            cache=cache,
        )

    def retrieve(
            self, 
            query_id: str, 
            query_text: str, 
            query_label: Optional[Any] = None, 
            n_shot: Optional[int] = None,
            mmr_lambda: Optional[float] = None
            ) -> tuple[list[str], list[str]]:
        shots =  self.pipeline.retrieve_shots(query_id=query_id, query_text=query_text, query_label=query_label, n_shot=n_shot, mmr_lambda=mmr_lambda)["shots"]
        texts = [s["text"] for s in shots]
        labels = [parsed_quad_to_raw_quad(self.data_dict[s["doc_id"]]["quadruples"]) for s in shots]
        return texts, labels

def main_make_nshot_dataset(
        data_path: str = "data/full/classify/train.json",
        model_path: str = "models/base/bge-large-zh-v1.5",
        index_path: str = "./cache_retrieval/faiss_hnsw.index",
        docs_path: str = "./cache_retrieval/doc_store.json",
        params: RetrievalParams = RETRIEVAL_PARAMS,
        ):


    cache = CacheManager(params.cache_dir)
    embedder = STEmbedder(model_path=model_path, device="cuda:0", batch_size=64, cache=cache, normalize=True)

    # load index + doc_store
    index = FaissHNSWIndex.load(index_path)
    with open(docs_path, "r", encoding="utf-8") as f:
        doc_store = json.load(f)

    # optional: load vectors into RAM for faster MMR (recommended if memory allows)
    # If you want to reconstruct from scratch, re-encode doc_store texts.
    # Here we choose to not load (None) to keep demo minimal.
    vector_store = None

    pipeline = DiversityRetrievalPipeline(
        params=params,
        embedder=embedder,
        index=index,
        doc_store=doc_store,
        vector_store=vector_store,
        cache=cache,
    )

    data = load_json_dataset(data_path)
    out_path = "data/exp_data/mmr/train_nshot.jsonl"
    with open(out_path, "w", encoding="utf-8") as w:
        for item in tqdm(data, desc="Making n-shot samples"):
            qid = str(item["id"])
            qtext = item["content"]
            qlabel = item.get("label")
            result = pipeline.retrieve_shots(query_id=qid, query_text=qtext, query_label=qlabel)

            # This is your new fine-tuning sample format; adapt to your trainer.
            # Example: {"prompt":..., "output": label}
            rec = {
                "id": qid,
                "prompt": result["prompt"],
                "label": qlabel,
                "shots": [{"id": s["doc_id"], "label": s.get("label"), "sim": s["sim"]} for s in result["shots"]],
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Saved:", out_path)


if __name__ == "__main__":
    # Choose one:
    # main_build_index(data_path="data/full/classify/train.json", model_path="models/base/bge-large-zh-v1.5")
    # main_make_nshot_dataset()

    retriever = MMRReterever(
        data_path="data/full/std/train.json",
        model_path="models/base/bge-large-zh-v1.5",
        index_path="./cache_retrieval/faiss_hnsw.index",
        docs_path="./cache_retrieval/doc_store.json",
        params=RETRIEVAL_PARAMS,
    )
    results = retriever.retrieve(query_id="test1", query_text="这是一个测试文本，用于分类任务。", n_shot=5)
    print("Retrieved shots:")
    for r in results:
        print(r)
