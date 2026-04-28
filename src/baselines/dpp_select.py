from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from typing import Any

sys.path.append(os.getcwd())

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.task_adapter import (
    build_scoring_prompt_no_demo,
    build_scoring_prompt_with_one_demo,
    gold_output,
    normalize_task_type,
    record_classes,
    stable_dumps,
)
from exps.baselines.dpp.two_stage_dpp_hsd import (
    batch_loglik_chat,
    compute_embeddings_sentence_transformers,
    greedy_k_dpp_map,
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def file_sha1(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_messages(user_content: str, task_type: str) -> list[dict[str, str]]:
    system = "You are a professional linguist and hate speech analyst. Follow the user's instructions exactly."
    if normalize_task_type(task_type) == "cold_binary":
        system = "你是一位中文内容安全审核专家，请严格按要求完成 COLD 仇恨言论二分类。"
    if normalize_task_type(task_type) == "hatexplain":
        system = "You are a professional hate speech analyst. Return only valid JSON that follows the requested schema."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def sample_score_set_indices(
    data: list[dict],
    task_type: str,
    exclude: set[int],
    total: int,
    m_per_group: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    group2idx: dict[str, list[int]] = {}
    for idx, record in enumerate(data):
        if idx in exclude:
            continue
        for group in record_classes(record, task_type):
            group2idx.setdefault(group, []).append(idx)

    chosen: list[int] = []
    chosen_set: set[int] = set()
    for group in sorted(group2idx):
        candidates = list(group2idx[group])
        rng.shuffle(candidates)
        for idx in candidates[:m_per_group]:
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
            if len(chosen) >= total:
                return chosen[:total]

    remaining = [idx for idx in range(len(data)) if idx not in exclude and idx not in chosen_set]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, total - len(chosen))])
    return chosen[:total]


def selector_signature(build_config: dict, params: dict[str, Any]) -> dict[str, Any]:
    raw_data_path = build_config.get("data_paths", {}).get("raw_data_path", "")
    return {
        "raw_data_path": raw_data_path,
        "raw_data_sha1": file_sha1(raw_data_path),
        "task_type": normalize_task_type(build_config.get("task_type")),
        "params": params,
        "prompt_template": build_config.get("prompt_templates", {}).get("prompt_template"),
        "example_template": build_config.get("prompt_templates", {}).get("example_template"),
    }


def signature_matches(path: str, expected: dict[str, Any]) -> bool:
    if not os.path.exists(path):
        return False
    try:
        actual = load_json(path)
    except Exception:
        return False
    return stable_dumps(actual) == stable_dumps(expected)


def select_demos(build_config: dict, force: bool = False) -> str:
    baseline = build_config.get("baseline", {})
    dpp_cfg = baseline.get("dpp", {})
    task_type = normalize_task_type(baseline.get("task_type") or build_config.get("task_type"))

    k = int(dpp_cfg.get("k", 10))
    n_sem = int(dpp_cfg.get("n_sem", 200))
    score_total = int(dpp_cfg.get("T", 128))
    m_per_group = int(dpp_cfg.get("m_per_group", 3))
    tau = float(dpp_cfg.get("tau", 1.5))
    seed = int(dpp_cfg.get("seed", 42))
    embed_batch = int(dpp_cfg.get("embed_batch", 64))
    batch_size = int(dpp_cfg.get("batch_size", 6))
    max_length = int(dpp_cfg.get("max_length", 2048))
    dtype_name = str(dpp_cfg.get("dtype", "auto"))

    data_paths = build_config.get("data_paths", {})
    model_settings = build_config.get("model_settings", {})
    raw_data_path = data_paths.get("raw_data_path")
    if not raw_data_path:
        raise ValueError("DPP selector requires data_paths.raw_data_path")

    artifact_dir = dpp_cfg.get("artifact_dir") or os.path.join("artifacts", "dpp")
    cache_dir = dpp_cfg.get("cache_dir") or os.path.join(artifact_dir, "cache")
    demos_path = dpp_cfg.get("demos_path") or os.path.join(artifact_dir, f"demos_k{k}.json")
    prompt_path = dpp_cfg.get("prompt_path") or os.path.join(artifact_dir, f"demos_k{k}_prompt.txt")
    signature_path = dpp_cfg.get("signature_path") or os.path.join(artifact_dir, "selection_signature.json")

    embed_model = dpp_cfg.get("embed_model") or model_settings.get("srag_model_path") or "models/base/bge-large-zh-v1.5"
    llm_model = dpp_cfg.get("llm_model") or data_paths.get("tokenizer_path") or "models/base/Qwen2.5-7B-Instruct"

    params = {
        "k": k,
        "n_sem": n_sem,
        "T": score_total,
        "m_per_group": m_per_group,
        "tau": tau,
        "seed": seed,
        "embed_model": embed_model,
        "llm_model": llm_model,
        "task_type": task_type,
    }
    sig = selector_signature(build_config, params)
    if not force and os.path.exists(demos_path) and signature_matches(signature_path, sig):
        print(f"[DPP] Reusing existing demos: {demos_path}")
        return demos_path

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    data = load_json(raw_data_path)
    if not isinstance(data, list):
        raise ValueError(f"DPP data must be a JSON list: {raw_data_path}")
    if not data:
        raise ValueError("DPP data is empty")

    texts = [str(record.get("content", "")) for record in data]
    emb_path = os.path.join(cache_dir, "embeddings.npy")
    if os.path.exists(emb_path) and not force:
        X = np.load(emb_path)
    else:
        X = compute_embeddings_sentence_transformers(texts, embed_model, batch_size=embed_batch)
        np.save(emb_path, X)

    l_sem_path = os.path.join(cache_dir, "L_sem.npy")
    if os.path.exists(l_sem_path) and not force:
        l_sem = np.load(l_sem_path)
    else:
        l_sem = (X @ X.T).astype(np.float64)
        np.fill_diagonal(l_sem, np.diag(l_sem) + 1e-6)
        np.save(l_sem_path, l_sem)

    idx_sem_path = os.path.join(cache_dir, f"stage1_sem_ids_n{n_sem}.json")
    if os.path.exists(idx_sem_path) and not force:
        idx_sem = load_json(idx_sem_path)
    else:
        idx_sem = greedy_k_dpp_map(l_sem, min(n_sem, len(data)))
        save_json(idx_sem_path, idx_sem)

    idx_sem_set = set(int(idx) for idx in idx_sem)
    score_ids_path = os.path.join(cache_dir, f"score_ids_T{score_total}_m{m_per_group}.json")
    idx_score = []
    if os.path.exists(score_ids_path) and not force:
        idx_score = load_json(score_ids_path)
    if not idx_score:
        score_exclude = idx_sem_set if len(idx_sem_set) < len(data) else set()
        idx_score = sample_score_set_indices(
            data=data,
            task_type=task_type,
            exclude=score_exclude,
            total=min(score_total, max(1, len(data) - len(score_exclude))),
            m_per_group=m_per_group,
            seed=seed,
        )
        save_json(score_ids_path, idx_score)

    sem_items = [data[int(idx)] for idx in idx_sem]
    score_items = [data[int(idx)] for idx in idx_score]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype_name == "auto":
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    elif dtype_name == "bf16":
        dtype = torch.bfloat16
    elif dtype_name == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_path = os.path.join(cache_dir, f"base_ll_T{len(score_items)}.npz")
    if os.path.exists(base_path) and not force:
        base_npz = np.load(base_path)
        base_ll = base_npz["base_ll"].astype(np.float64)
        base_tok = base_npz["base_tok"].astype(np.float64)
    else:
        base_ll_values = []
        base_tok_values = []
        for start in tqdm(range(0, len(score_items), batch_size), desc="DPP base loglik"):
            batch = score_items[start:start + batch_size]
            messages = [build_messages(build_scoring_prompt_no_demo(item.get("content", ""), task_type), task_type) for item in batch]
            gold = [gold_output(item, task_type) for item in batch]
            ll, tok = batch_loglik_chat(
                model=model,
                tokenizer=tokenizer,
                messages_list=messages,
                gold_list=gold,
                max_length=max_length,
                device=model.device if device == "cuda" else device,
            )
            base_ll_values.extend(ll.tolist())
            base_tok_values.extend(tok.tolist())
        base_ll = np.asarray(base_ll_values, dtype=np.float64)
        base_tok = np.asarray(base_tok_values, dtype=np.float64)
        np.savez(base_path, base_ll=base_ll, base_tok=base_tok)

    influence_path = os.path.join(cache_dir, f"I_n{len(sem_items)}_T{len(score_items)}.npy")
    quality_path = os.path.join(cache_dir, f"Q_n{len(sem_items)}_T{len(score_items)}.npy")
    if os.path.exists(influence_path) and os.path.exists(quality_path) and not force:
        influence = np.load(influence_path).astype(np.float64)
        quality = np.load(quality_path).astype(np.float64)
    else:
        influence = np.zeros((len(sem_items), len(score_items)), dtype=np.float64)
        quality = np.zeros((len(sem_items),), dtype=np.float64)
        for i, demo in enumerate(tqdm(sem_items, desc="DPP influence")):
            ll_values = []
            for start in range(0, len(score_items), batch_size):
                batch = score_items[start:start + batch_size]
                messages = [
                    build_messages(build_scoring_prompt_with_one_demo(demo, item.get("content", ""), task_type), task_type)
                    for item in batch
                ]
                gold = [gold_output(item, task_type) for item in batch]
                ll, _tok = batch_loglik_chat(
                    model=model,
                    tokenizer=tokenizer,
                    messages_list=messages,
                    gold_list=gold,
                    max_length=max_length,
                    device=model.device if device == "cuda" else device,
                )
                ll_values.extend(ll.tolist())
            ll_with_demo = np.asarray(ll_values, dtype=np.float64)
            infl = (ll_with_demo - base_ll) / base_tok
            influence[i, :] = infl
            quality[i] = float(np.mean(infl))
        np.save(influence_path, influence)
        np.save(quality_path, quality)

    quality_mean = float(np.mean(quality))
    quality_std = float(np.std(quality) + 1e-6)
    weights = np.exp(((quality - quality_mean) / quality_std) / tau)
    l_inf = (influence @ influence.T).astype(np.float64)
    l_inf = (weights[:, None] * l_inf) * weights[None, :]
    np.fill_diagonal(l_inf, np.diag(l_inf) + 1e-6)

    idx_final_local = greedy_k_dpp_map(l_inf, min(k, len(sem_items)))
    idx_final_local = sorted(idx_final_local, key=lambda idx: quality[idx])
    final_items = [sem_items[int(idx)] for idx in idx_final_local]

    save_json(demos_path, final_items)
    with open(prompt_path, "w", encoding="utf-8") as file:
        for item in final_items:
            file.write(f"示例：\n文本：{item.get('content', '')}\n输出：{gold_output(item, task_type)}\n\n")
    save_json(signature_path, sig)
    print(f"[DPP] Saved demos: {demos_path}")
    return demos_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select self-contained DPP demos for baseline experiments.")
    parser.add_argument("--build-config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    build_config = load_json(args.build_config)
    select_demos(build_config, force=args.force)


if __name__ == "__main__":
    main()
