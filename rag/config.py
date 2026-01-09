from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class RetrievalParams:
    # retrieval
    top_k: int = 100                 # initial recall size
    n_shot: int = 6                  # final selected size
    sim_min: float = 0.45
    sim_max: float = 0.90

    # diversity selection (MMR)
    mmr_lambda: float = 0.75

    # label/meta policies
    label_policy: str = "soft"       # "strict" | "soft" | "none"
    allow_cross_label_ratio: float = 0.15  # in soft mode, max ratio of cross-label samples

    # HNSW
    hnsw_m: int = 32                 # graph degree
    ef_construction: int = 200
    ef_search: int = 128

    # caching / reproducibility
    seed: int = 42
    cache_dir: str = "./cache_retrieval"

    # trace / logging
    trace_jsonl_path: Optional[str] = "./cache_retrieval/selection_trace.jsonl"

    # prompt packing
    template_id: str = "basic"       # choose packer templates