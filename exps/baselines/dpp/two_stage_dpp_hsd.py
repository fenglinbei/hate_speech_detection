# two_stage_dpp_hsd.py
# Two-Stage DPP demonstration selection for HSD quadruple extraction
# - Stage1: semantic diversity (embed -> L = XX^T -> greedy k-DPP MAP)
# - Stage2: quality + influence diversity (HF loglik -> I, Q -> L = diag(q) (I I^T) diag(q) -> greedy k-DPP MAP)
#
# Requirements:
#   pip install -U torch transformers sentence-transformers numpy tqdm
#
# Example:
#   python two_stage_dpp_hsd.py \
#     --data train.json \
#     --llm_model Qwen/Qwen2.5-7B-Instruct \
#     --embed_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
#     --k 10 --n_sem 200 --T 128 --cache_dir .cache_dpp_hsd \
#     --batch_size 6 --seed 42

import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from exps.utils import data

# --------- 0) Canonicalization for stable log-likelihood ----------

KEY_ORDER = ["target", "argument", "targeted_group", "hateful"]
import re

BASE_GROUPS = ["Racism", "Region", "LGBTQ", "Sexism", "others", "non-hate"]
BASE_SET = set(BASE_GROUPS)

# 可按你的数据再补充一些别名
ALIASES = {
    "lgbt": "LGBTQ",
    "lgbtq+": "LGBTQ",
    "lgbtq": "LGBTQ",
    "queer": "LGBTQ",
    "homosexual": "LGBTQ",
    "nonhate": "non-hate",
    "non_hate": "non-hate",
    "nohate": "non-hate",
    "other": "others",
}

_SPLIT_RE = re.compile(r"[|,/;+\s、和&]+")

def normalize_group_atoms(raw: str) -> list[str]:
    """
    把 raw targeted_group 解析成原子标签列表（去重、固定顺序）。
    支持: 'Racism+Sexism', 'Sexism, Racism', 'Region|others', 'LGBTQ ' 等
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []

    parts = [p.strip() for p in _SPLIT_RE.split(s) if p.strip()]
    atoms = []
    for p in parts:
        p_low = p.lower()
        p2 = ALIASES.get(p_low, p)  # 别名映射
        p2 = p2.strip()
        # 统一大小写（你这套标签本身是 CamelCase/小写混合）
        # 这里用 BASE_GROUPS 的规范形式对齐
        for b in BASE_GROUPS:
            if p2 == b or p2.lower() == b.lower():
                atoms.append(b)
                break
        else:
            # 未知标签：如果你确定不存在，可忽略；否则可映射到 others
            # atoms.append("others")
            pass

    # 去重 + 固定顺序
    atoms = [g for g in BASE_GROUPS if g in set(atoms)]

    # 规则：non-hate 不与其它标签共存（共存时去掉 non-hate）
    if "non-hate" in atoms and len(atoms) > 1:
        atoms = [g for g in atoms if g != "non-hate"]
    return atoms

def normalize_group_combo(raw: str) -> str:
    """
    把 raw targeted_group 归一化为：
      - 单标签：Racism / Sexism / ... / non-hate
      - 组合：Racism, Sexism （按 BASE_GROUPS 顺序）
    """
    atoms = normalize_group_atoms(raw)
    if not atoms:
        # 兜底：若你希望强制落到 others/non-hate，可在这里改
        return "others"
    if len(atoms) == 1:
        return atoms[0]
    return ", ".join(atoms)

def get_group_list_for_prompt(_: list[dict]) -> list[str]:
    # prompt 里不枚举组合，固定写原子标签即可
    return BASE_GROUPS


def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # basic normalization: strip & unify whitespace
    s = s.replace("\u3000", " ").strip()
    # collapse consecutive spaces
    s = " ".join(s.split())
    return s

def canonicalize_quadruples(quads: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Return canonical list: fixed keys + normalized values + sorted list."""
    canon = []
    for q in quads:
        item = {
            "target": _norm_text(q.get("target", "")),
            "argument": _norm_text(q.get("argument", "")),
            "targeted_group": normalize_group_combo(q.get("targeted_group", "")),
            "hateful": _norm_text(q.get("hateful", "")),
        }
        canon.append(item)

    # sort for unique serialization
    canon.sort(key=lambda x: (
        x["targeted_group"], x["hateful"], x["target"], x["argument"]
    ))
    return canon

def quads_to_canonical_json_str(quads: List[Dict[str, Any]]) -> str:
    canon = canonicalize_quadruples(quads)
    # ensure fixed key order by rebuilding dict
    canon2 = [{k: item[k] for k in KEY_ORDER} for item in canon]
    return json.dumps(canon2, ensure_ascii=False, separators=(",", ":"))

# --------- 1) Prompt templates (single-demo scoring + multi-demo inference) ----------

def build_system_prompt() -> str:
    return "你是一个信息抽取模型。"

def build_task_instruction(group_list: List[str]) -> str:
    groups = ", ".join(group_list)
    return (
        "给定一句中文社交媒体文本，请抽取其中所有四元组：\n"
        "- target: 被指向的群体或对象（字符串）\n"
        "- argument: 针对该对象的属性/事件/论断（字符串）\n"
        f"- targeted_group: 由 {{{groups}}} 中一个或多个标签用', '连接组成（例如 Racism, Sexism）；non-hate 仅单独出现\n"
        "- hateful: {hate, non-hate}\n\n"
        "要求：只输出JSON数组，不要输出任何解释或多余文本。"
    )

def build_user_prompt_no_demo(x: str, group_list: List[str]) -> str:
    instr = build_task_instruction(group_list)
    return f"{instr}\n文本：{x}"

def build_user_prompt_with_one_demo(
    demo_x: str, demo_y_json: str,
    x: str, group_list: List[str]
) -> str:
    instr = build_task_instruction(group_list)
    # keep the formatting minimal & stable
    return (
        f"{instr}\n"
        f"示例：\n"
        f"文本：{demo_x}\n"
        f"输出：{demo_y_json}\n\n"
        f"现在请处理：\n"
        f"文本：{x}"
    )

def build_messages(user_content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_content},
    ]

# --------- 2) Embeddings for Stage-1 ----------

def compute_embeddings_sentence_transformers(texts: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers 未安装或导入失败。请先 pip install sentence-transformers") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(model_name, device=device)
    embs = enc.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)

# --------- 3) Greedy k-DPP MAP for L-ensemble DPP ----------

def greedy_k_dpp_map(L: np.ndarray, k: int) -> List[int]:
    """
    Deterministic greedy MAP for k-DPP (L-ensemble).
    Works well in practice and avoids dependency mismatch.
    L must be symmetric PSD-ish. Add jitter outside if needed.
    """
    n = L.shape[0]
    assert L.shape[1] == n
    # Cholesky-like incremental updates (greedy for det)
    selected = []
    # keep "residual" diagonal scores
    diag = np.clip(np.diag(L).copy(), 1e-12, None)
    # store orthogonal components
    C = np.zeros((k, n), dtype=np.float64)  # (t, n)

    for t in range(k):
        i = int(np.argmax(diag))
        if diag[i] <= 1e-12:
            # no further gain
            break
        selected.append(i)

        # update C[t]
        if t == 0:
            C[t, :] = L[i, :] / math.sqrt(diag[i])
        else:
            proj = C[:t, i] @ C[:t, :]  # (n,)
            C[t, :] = (L[i, :] - proj) / math.sqrt(diag[i])

        # update diag: diag[j] -= C[t,j]^2
        diag = np.maximum(diag - C[t, :] ** 2, 1e-12)

        # ensure we don't pick same again
        diag[i] = 1e-12

    return selected

# --------- 4) HF log-likelihood scoring (batched, per-sample) ----------

@torch.no_grad()
def batch_loglik_chat(
    model,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    gold_list: List[str],
    max_length: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return per-sample loglik and token_count for gold tokens only.
    loglik = sum_t log p(y_t | prompt, y_<t)
    """
    assert len(messages_list) == len(gold_list)
    B = len(messages_list)

    prompt_ids_list = []
    gold_ids_list = []

    for msgs, gold in zip(messages_list, gold_list):
        # chat prompt ids (includes assistant generation prompt)
        prompt_ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True
        )
        gold_ids = tokenizer(gold, add_special_tokens=False).input_ids
        prompt_ids_list.append(prompt_ids)
        gold_ids_list.append(gold_ids)

    # build concatenated input & masks
    input_ids_list = []
    mask_full_list = []
    for p_ids, g_ids in zip(prompt_ids_list, gold_ids_list):
        ids = p_ids + g_ids
        if len(ids) > max_length:
            # Hard cut: keep tail of prompt? For your data (short microblogs) usually not needed.
            # Here we truncate from the left of the prompt to preserve the gold region.
            overflow = len(ids) - max_length
            # try to drop from prompt part first
            drop = min(overflow, max(0, len(p_ids) - 32))  # keep at least 32 prompt tokens
            p_ids2 = p_ids[drop:]
            ids = p_ids2 + g_ids
            if len(ids) > max_length:
                ids = ids[-max_length:]  # last resort

        # recompute prompt length after truncation
        # (approx: gold_ids are always at the end)
        g_len = len(g_ids)
        p_len = len(ids) - g_len
        mask_full = [0] * len(ids)
        for t in range(p_len, len(ids)):
            mask_full[t] = 1

        input_ids_list.append(ids)
        mask_full_list.append(mask_full)

    maxL = max(len(ids) for ids in input_ids_list)
    maxL = min(maxL, max_length)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # Qwen often has no pad token set; fall back to eos as pad
        pad_id = tokenizer.eos_token_id

    input_ids = torch.full((B, maxL), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, maxL), dtype=torch.long)
    mask_full = torch.zeros((B, maxL), dtype=torch.float32)

    for i, (ids, m) in enumerate(zip(input_ids_list, mask_full_list)):
        L = min(len(ids), maxL)
        input_ids[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
        attn_mask[i, :L] = 1
        mask_full[i, :L] = torch.tensor(m[:L], dtype=torch.float32)

    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    mask_full = mask_full.to(device)

    # forward
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits  # [B, L, V]

    # shift for causal LM
    logits = logits[:, :-1, :]                 # [B, L-1, V]
    target = input_ids[:, 1:]                  # [B, L-1]
    mask_shift = mask_full[:, 1:]              # [B, L-1]

    log_probs = F.log_softmax(logits, dim=-1)  # [B, L-1, V]
    token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

    # only count gold region
    token_lp = token_lp * mask_shift
    token_count = mask_shift.sum(dim=1).clamp_min(1.0)
    seq_loglik = token_lp.sum(dim=1)

    return seq_loglik.detach().cpu().numpy(), token_count.detach().cpu().numpy()

# --------- 5) Score set construction (stratified by targeted_group) ----------

def sample_score_set_indices(
    data: List[Dict[str, Any]],
    exclude: set,
    T: int,
    m_per_group: int,
    seed: int,
) -> List[int]:
    rng = random.Random(seed)
    # map group -> candidate sample indices
    group2idx = {}
    all_groups = set()

    for i, ex in enumerate(data):
        if i in exclude:
            continue
        groups = set()
        for q in ex.get("quadruples", []):
            groups.update(normalize_group_atoms(q.get("targeted_group", "")))
        if not groups:
            continue
        for g in groups:
            all_groups.add(g)
            group2idx.setdefault(g, []).append(i)

    group_list = sorted(list(all_groups))
    # deterministic shuffle within each group
    for g in group_list:
        rng.shuffle(group2idx[g])

    chosen = []
    chosen_set = set()

    # ensure coverage
    for g in group_list:
        cand = group2idx.get(g, [])
        take = min(m_per_group, len(cand))
        for idx in cand[:take]:
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
            if len(chosen) >= T:
                return chosen[:T]

    # fill remaining randomly from all remaining indices
    remaining = [i for i in range(len(data)) if i not in exclude and i not in chosen_set]
    rng.shuffle(remaining)
    need = T - len(chosen)
    chosen.extend(remaining[:need])
    return chosen[:T]

# --------- 6) Main pipeline ----------

def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    assert isinstance(obj, list)
    return obj

def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get_group_list_from_data(data: List[Dict[str, Any]]) -> List[str]:
    s = set()
    for ex in data:
        for q in ex.get("quadruples", []):
            g = _norm_text(q.get("targeted_group", ""))
            if g:
                s.add(g)
    # Make sure common tokens exist; keep your dataset actual enums
    return sorted(list(s))

@dataclass
class DemoItem:
    idx: int
    ex_id: int
    content: str
    gold_json: str

def build_demo_items(data: List[Dict[str, Any]]) -> List[DemoItem]:
    items = []
    for i, ex in enumerate(data):
        items.append(DemoItem(
            idx=i,
            ex_id=int(ex.get("id", i)),
            content=_norm_text(ex.get("content", "")),
            gold_json=quads_to_canonical_json_str(ex.get("quadruples", [])),
        ))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default= "data/full/std/train.json", help="train.json path (list of dicts)")
    ap.add_argument("--cache_dir", type=str, default="exps/baselines/dpp/cache_dpp_hsd")
    ap.add_argument("--seed", type=int, default=42)

    # selection hyperparams
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n_sem", type=int, default=200)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--m_per_group", type=int, default=3)
    ap.add_argument("--tau", type=float, default=1.5)

    # embedding
    ap.add_argument("--embed_model", type=str, default="models/base/bge-large-zh-v1.5")
    ap.add_argument("--embed_batch", type=int, default=64)

    # LLM scoring
    ap.add_argument("--llm_model", type=str, default="models/base/Qwen2.5-7B-Instruct", help="HF model name/path for scoring")
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.cache_dir, exist_ok=True)

    data = load_json_list(args.data)
    items = build_demo_items(data)
    group_list = get_group_list_for_prompt(data)


    # -------- Stage1: embeddings -> semantic DPP --------
    emb_path = os.path.join(args.cache_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        X = np.load(emb_path)
    else:
        texts = [it.content for it in items]
        X = compute_embeddings_sentence_transformers(texts, args.embed_model, batch_size=args.embed_batch)
        np.save(emb_path, X)

    # L_s = X X^T
    # Use float64 for numerical stability in greedy
    Ls_path = os.path.join(args.cache_dir, "L_sem.npy")
    if os.path.exists(Ls_path):
        L_sem = np.load(Ls_path)
    else:
        L_sem = (X @ X.T).astype(np.float64)
        # jitter for PSD
        np.fill_diagonal(L_sem, np.diag(L_sem) + 1e-6)
        np.save(Ls_path, L_sem)

    sem_ids_path = os.path.join(args.cache_dir, f"stage1_sem_ids_n{args.n_sem}.json")
    if os.path.exists(sem_ids_path):
        idx_sem = json.load(open(sem_ids_path, "r", encoding="utf-8"))
    else:
        idx_sem = greedy_k_dpp_map(L_sem, args.n_sem)
        save_json(sem_ids_path, idx_sem)

    idx_sem_set = set(idx_sem)

    # -------- Score set selection --------
    score_ids_path = os.path.join(args.cache_dir, f"score_ids_T{args.T}_m{args.m_per_group}.json")
    if os.path.exists(score_ids_path):
        idx_score = json.load(open(score_ids_path, "r", encoding="utf-8"))
    else:
        idx_score = sample_score_set_indices(
            data=data, exclude=idx_sem_set,
            T=args.T, m_per_group=args.m_per_group, seed=args.seed
        )
        save_json(score_ids_path, idx_score)

    score_items = [items[i] for i in idx_score]
    sem_items = [items[i] for i in idx_sem]

    # -------- Load LLM for scoring --------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # Ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------- Baseline loglik for score set (no demo) --------
    base_path = os.path.join(args.cache_dir, f"base_ll_T{args.T}.npz")
    if os.path.exists(base_path):
        base_npz = np.load(base_path)
        base_ll = base_npz["base_ll"].astype(np.float64)
        base_tok = base_npz["base_tok"].astype(np.float64)
    else:
        base_ll = []
        base_tok = []
        for b0 in tqdm(range(0, len(score_items), args.batch_size), desc="Base loglik (no demo)"):
            batch = score_items[b0:b0+args.batch_size]
            msg_list = []
            gold_list = []
            for it in batch:
                user = build_user_prompt_no_demo(it.content, group_list)
                msg_list.append(build_messages(user))
                gold_list.append(it.gold_json)
            ll, tok = batch_loglik_chat(
                model=model, tokenizer=tokenizer,
                messages_list=msg_list, gold_list=gold_list,
                max_length=args.max_length, device=model.device if device == "cuda" else device,
            )
            base_ll.extend(ll.tolist())
            base_tok.extend(tok.tolist())
        base_ll = np.asarray(base_ll, dtype=np.float64)
        base_tok = np.asarray(base_tok, dtype=np.float64)
        np.savez(base_path, base_ll=base_ll, base_tok=base_tok)

    # -------- Influence matrix I (N_sem x T) and quality Q --------
    I_path = os.path.join(args.cache_dir, f"I_n{args.n_sem}_T{args.T}.npy")
    Q_path = os.path.join(args.cache_dir, f"Q_n{args.n_sem}_T{args.T}.npy")

    if os.path.exists(I_path) and os.path.exists(Q_path):
        I = np.load(I_path).astype(np.float64)
        Q = np.load(Q_path).astype(np.float64)
    else:
        I = np.zeros((len(sem_items), len(score_items)), dtype=np.float64)
        Q = np.zeros((len(sem_items),), dtype=np.float64)

        for i, demo in enumerate(tqdm(sem_items, desc="Compute influence for each demo")):
            ll_with_all = []

            for b0 in range(0, len(score_items), args.batch_size):
                batch = score_items[b0:b0+args.batch_size]
                msg_list = []
                gold_list = []
                for it in batch:
                    user = build_user_prompt_with_one_demo(
                        demo_x=demo.content,
                        demo_y_json=demo.gold_json,
                        x=it.content,
                        group_list=group_list
                    )
                    msg_list.append(build_messages(user))
                    gold_list.append(it.gold_json)

                ll, tok = batch_loglik_chat(
                    model=model, tokenizer=tokenizer,
                    messages_list=msg_list, gold_list=gold_list,
                    max_length=args.max_length, device=model.device if device == "cuda" else device,
                )
                ll_with_all.extend(ll.tolist())

            ll_with_all = np.asarray(ll_with_all, dtype=np.float64)

            # normalized influence by token_count (gold tokens)
            infl = (ll_with_all - base_ll) / base_tok
            I[i, :] = infl
            Q[i] = float(np.mean(infl))

        np.save(I_path, I)
        np.save(Q_path, Q)

    # -------- Stage2: build L_I and select final k demos --------
    # robust positive weights from Q (avoid negative / scale issues)
    Q_mean = float(np.mean(Q))
    Q_std = float(np.std(Q) + 1e-6)
    z = (Q - Q_mean) / Q_std
    q = np.exp(z / args.tau)  # positive weights

    LI = (I @ I.T).astype(np.float64)
    LI = (q[:, None] * LI) * q[None, :]
    np.fill_diagonal(LI, np.diag(LI) + 1e-6)

    idx_final_local = greedy_k_dpp_map(LI, args.k)  # indices within sem_items
    # order by ascending raw quality (paper-style)
    idx_final_local = sorted(idx_final_local, key=lambda j: Q[j])

    final_items = [sem_items[j] for j in idx_final_local]

    # -------- Export demos --------
    out_path = os.path.join(args.cache_dir, f"demos_k{args.k}.json")
    out = []
    for it in final_items:
        # export in your original schema
        ex = data[it.idx]
        out.append(ex)

    save_json(out_path, out)

    # also export a ready-to-paste prompt block (multi-demo)
    prompt_path = os.path.join(args.cache_dir, f"demos_k{args.k}_prompt.txt")
    demo_blocks = []
    for ex in out:
        y = quads_to_canonical_json_str(ex["quadruples"])
        demo_blocks.append(f"示例：\n文本：{_norm_text(ex['content'])}\n输出：{y}\n")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(demo_blocks))

    print("\n=== Done ===")
    print(f"[Saved] demos json: {out_path}")
    print(f"[Saved] prompt block: {prompt_path}")
    print("Selected demo ids:", [int(e.get("id", -1)) for e in out])

if __name__ == "__main__":
    main()
