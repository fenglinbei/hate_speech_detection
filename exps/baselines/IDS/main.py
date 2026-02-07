import os
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from rag.core import Retriever


# =========================
# 1) vLLM OpenAI-compatible Chat Client
# =========================

class VLLMChatClient:
    """
    适配 vLLM 的 OpenAI-compatible API:
      POST http://localhost:35000/v1/chat/completions
    """
    def __init__(
        self,
        base_url: str = "http://localhost:35000/v1",
        model: str = "Qwen2.5-7B-Instruct",
        api_key: str = "EMPTY",
        timeout: int = 120,
        max_retries: int = 3,
        retry_sleep: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 0.5,
        top_k: int = 20,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        # vLLM 通常支持 top_k，但不同版本字段可能在 extra_body
        # 这里优先放在 payload，若你 vLLM 不认可，可挪到 extra_body
        payload["top_k"] = top_k

        if stop:
            payload["stop"] = stop
        if extra_body:
            payload.update(extra_body)

        last_err = None
        for _ in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                time.sleep(self.retry_sleep)

        raise RuntimeError(f"vLLM chat failed after retries: {last_err}")


# =========================
# 2) IDS 配置
# =========================

@dataclass
class IDSConfig:
    q: int = 3                 # 迭代轮数
    top_k: int = 10             # 每轮检索 demos 数量（对应你的 Ke）
    max_reason_chars: int = 512  # 用于检索的 reason 最多保留多少字符（避免太长）
    temperature: float = 0.7
    top_p: float = 0.5
    top_k_gen: int = 20
    max_tokens_reason: int = 256
    max_tokens_full: int = 512


# =========================
# 3) 简单磁盘 cache（可选）
# =========================

class SimpleDiskCache:
    def __init__(self, cache_dir: str = "./exps/baselines/IDS/cache_ids", enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _key_to_path(cache_dir: str, key: str) -> str:
        return os.path.join(cache_dir, f"{key}.json")

    @staticmethod
    def sha1(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self._key_to_path(self.cache_dir, key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._key_to_path(self.cache_dir, key)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


# =========================
# 4) IDS Retriever（复用你的 Retriever 做 embedding 检索）
# =========================

class IDSRetriever:
    """
    复现 IDS：Reason-driven iterative demonstration selection.

    依赖：
      - base_retriever: 你给的 Retriever（含 .retrieve(query, top_k=...) -> (texts, outputs)）
      - llm: vLLM client（OpenAI-compatible）
      - prompt 模板（Reason-only / Reason+Triples）

    你可以：
      A) 调用 iterative_select：只做 IDS demo selection，返回每轮 demos + reason
      B) 调用 predict_with_ids：完整跑 q 轮推理 + set-voting（适配抽取任务），返回最终三元组
    """
    def __init__(
        self,
        base_retriever: Retriever,
        llm: VLLMChatClient,
        system_prompt: str,
        # 你当前的 prompt（建议为 IDS baseline 单独弄一个含 Reason 段的版本）
        prompt_reason_only_user: str,
        prompt_reason_and_triples_user: str,
        # 将 (demo_texts, demo_outputs) 格式化为 examples 字符串
        format_examples_fn: Optional[Callable[[List[str], List[str]], str]] = None,
        # baseline 默认不注入 lexicons：返回 "" 即可；想对齐你主方法也可换成真实 lexicons
        lexicons_fn: Optional[Callable[[str], str]] = None,
        config: Optional[IDSConfig] = None,
        cache: Optional[SimpleDiskCache] = None,
    ):
        self.base = base_retriever
        self.llm = llm
        self.system_prompt = system_prompt
        self.prompt_reason_only_user = prompt_reason_only_user
        self.prompt_reason_and_triples_user = prompt_reason_and_triples_user
        self.format_examples_fn = format_examples_fn or self._default_format_examples
        self.lexicons_fn = lexicons_fn or (lambda _text: "")
        self.cfg = config or IDSConfig()
        self.cache = cache or SimpleDiskCache(enabled=False)

    # ---------- default formatting ----------
    @staticmethod
    def _default_format_examples(demo_texts: List[str], demo_outputs: List[str]) -> str:
        """
        把检索到的 (content, output) 转成你 prompt 里的 {examples}。
        你可以按你现有示例格式改这块。
        """
        blocks = []
        for t, o in zip(demo_texts, demo_outputs):
            blocks.append(
                f"### 句子：\n{t}\n"
                f"### 三元组：\n{o}\n"
            )
        return "\n".join(blocks).strip()

    # ---------- parsing ----------
    @staticmethod
    def _extract_reason(text: str) -> str:
        """
        从模型输出中抽取 Reason。
        兼容：
          - "Reason: ..."
          - "Reason：..."
          - 没有显式 Reason 时：用前 1~2 行兜底
        """
        if not text:
            return ""
        # 优先找 "Reason"
        for key in ["Reason:", "Reason："]:
            idx = text.find(key)
            if idx != -1:
                # 取该行到行尾
                line = text[idx:].splitlines()[0]
                return line.strip()
        # 兜底：取第一行
        first = text.strip().splitlines()[0] if text.strip().splitlines() else ""
        return first.strip()

    @staticmethod
    def _extract_triples(text: str) -> str:
        """
        抽取三元组输出（适配你现在 prompt 最后是 '### 三元组：' 的形式）
        兼容：
          - 从 "### 三元组：" 之后截取
          - 或从 "Triples:" 之后截取
        """
        if not text:
            return ""
        candidates = ["### 三元组：", "### 三元组:", "Triples:", "Triples："]
        start = -1
        for c in candidates:
            pos = text.find(c)
            if pos != -1:
                start = pos + len(c)
                break
        if start == -1:
            # 没找到 marker：返回全文（交给你现有 parser 再处理）
            out = text.strip()
        else:
            out = text[start:].strip()

        # 截到 [END]（如果有）
        end_pos = out.find("[END]")
        if end_pos != -1:
            out = out[: end_pos + len("[END]")].strip()
        return out

    @staticmethod
    def _normalize_triple_key(triple_str: str) -> str:
        """
        对三元组字符串做轻量规范化，便于投票：
          - 去首尾空白
          - 统一分隔符空白
        你也可以按需要加：lower/全角半角/标点统一等。
        """
        return " ".join(triple_str.strip().split())

    @staticmethod
    def _split_triples(triples_text: str) -> List[str]:
        """
        把输出拆成多个三元组（按 [SEP] 拆）。
        假设你的输出是： t|a|g [SEP] t|a|g [SEP] ... [END]
        """
        if not triples_text:
            return []
        s = triples_text.strip()
        s = s.replace("[END]", "").strip()
        parts = [p.strip() for p in s.split("[SEP]")]
        parts = [p for p in parts if p]
        return parts

    @staticmethod
    def _join_triples(triples: List[str]) -> str:
        if not triples:
            return "[END]"
        return " [SEP] ".join(triples) + " [END]"

    # ---------- LLM calls ----------
    def _llm_reason_only(self, text: str) -> str:
        user = self.prompt_reason_only_user.format(text=text)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]
        out = self.llm.chat(
            messages,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k_gen,
            max_tokens=self.cfg.max_tokens_reason,
        )
        return out

    def _llm_reason_and_triples(self, text: str, examples: str, lexicons: str) -> str:
        user = self.prompt_reason_and_triples_user.format(
            lexicons=lexicons,
            examples=examples,
            text=text,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]
        out = self.llm.chat(
            messages,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k_gen,
            max_tokens=self.cfg.max_tokens_full,
        )
        return out

    # ---------- public APIs ----------
    def iterative_select(
        self,
        text: str,
        top_k: Optional[int] = None,
        q: Optional[int] = None,
        use_cache: bool = True,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        只做 IDS 的 iterative demo selection：
          - 先 zero-shot 得到 R0
          - 每轮用 R{j-1} 检索 demos
          - 用 demos 再生成 Rj（Reason+Triples 的输出里抽取 Reason）
        返回：每轮 (reason, demos) 以及中间原始输出
        """
        top_k = top_k if top_k is not None else self.cfg.top_k
        q = q if q is not None else self.cfg.q
        retriever_kwargs = retriever_kwargs or {}

        cache_key = None
        if use_cache and self.cache.enabled:
            cache_key = self.cache.sha1(f"IDS_SELECT|q={q}|k={top_k}|text={text}")
            hit = self.cache.get(cache_key)
            if hit is not None:
                return hit

        # 0) R0
        raw0 = self._llm_reason_only(text)
        reason = self._extract_reason(raw0)
        reason_for_retrieval = reason[: self.cfg.max_reason_chars]

        rounds = []
        for j in range(1, q + 1):
            demo_texts, demo_outs = self.base.retrieve(
                query=reason_for_retrieval,
                top_k=top_k
            )
            examples = self.format_examples_fn(demo_texts, demo_outs)
            lexicons = self.lexicons_fn(text)

            rawj = self._llm_reason_and_triples(text, examples=examples, lexicons=lexicons)
            reason_j = self._extract_reason(rawj)
            reason_for_retrieval = reason_j[: self.cfg.max_reason_chars]

            rounds.append({
                "iter": j,
                "reason_in": reason,
                "reason_out": reason_j,
                "demo_texts": demo_texts,
                "demo_outputs": demo_outs,
                "raw_output": rawj,
            })
            reason = reason_j

        result = {
            "text": text,
            "q": q,
            "top_k": top_k,
            "zero_shot_raw": raw0,
            "rounds": rounds,
        }

        if use_cache and cache_key and self.cache.enabled:
            self.cache.set(cache_key, result)
        return result

    def predict_with_ids(
        self,
        text: str,
        top_k: Optional[int] = None,
        q: Optional[int] = None,
        use_cache: bool = True,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        完整 IDS baseline（适配三元组抽取）：
          - q 轮：Reason -> 检索 demos -> 生成 (Reason + Triples)
          - 对每轮 Triples 做 set-level majority voting
        返回：final_triples + 每轮 triples + reasons + demos
        """
        top_k = top_k if top_k is not None else self.cfg.top_k
        q = q if q is not None else self.cfg.q
        retriever_kwargs = retriever_kwargs or {}

        cache_key = None
        if use_cache and self.cache.enabled:
            cache_key = self.cache.sha1(f"IDS_PREDICT|q={q}|k={top_k}|text={text}")
            hit = self.cache.get(cache_key)
            if hit is not None:
                return hit

        # 0) R0
        raw0 = self._llm_reason_only(text)
        reason = self._extract_reason(raw0)
        reason_for_retrieval = reason[: self.cfg.max_reason_chars]

        per_round = []
        per_round_triples_lists: List[List[str]] = []

        for j in range(1, q + 1):
            demo_texts, demo_outs = self.base.retrieve(
                query=reason_for_retrieval,
                top_k=top_k,
                **retriever_kwargs
            )
            examples = self.format_examples_fn(demo_texts, demo_outs)
            lexicons = self.lexicons_fn(text)

            rawj = self._llm_reason_and_triples(text, examples=examples, lexicons=lexicons)
            reason_j = self._extract_reason(rawj)
            triples_text = self._extract_triples(rawj)

            triples_list = self._split_triples(triples_text)
            per_round_triples_lists.append(triples_list)

            per_round.append({
                "iter": j,
                "reason_in": reason,
                "reason_out": reason_j,
                "demo_texts": demo_texts,
                "demo_outputs": demo_outs,
                "raw_output": rawj,
                "triples_text": triples_text,
                "triples_list": triples_list,
            })

            reason = reason_j
            reason_for_retrieval = reason_j[: self.cfg.max_reason_chars]

        # ---- set-level majority voting ----
        # 对每个三元组 key 计数：出现次数 >= ceil(q/2) 才保留
        need = (q // 2) + 1
        cnt: Dict[str, int] = {}
        exemplar: Dict[str, str] = {}

        for triples_list in per_round_triples_lists:
            # 一轮内去重（避免同一轮重复输出导致票数膨胀）
            seen_in_round = set()
            for t in triples_list:
                key = self._normalize_triple_key(t)
                if not key or key in seen_in_round:
                    continue
                seen_in_round.add(key)
                cnt[key] = cnt.get(key, 0) + 1
                exemplar.setdefault(key, t)

        voted = [exemplar[k] for k, v in cnt.items() if v >= need]
        # 可选：保持一个稳定顺序（按票数降序 + 字典序）
        voted.sort(key=lambda x: (-cnt[self._normalize_triple_key(x)], x))

        final_triples = self._join_triples(voted)

        result = {
            "text": text,
            "q": q,
            "top_k": top_k,
            "zero_shot_raw": raw0,
            "final_triples": final_triples,
            "rounds": per_round,
            "vote_threshold": need,
            "vote_counts": cnt,
        }

        if use_cache and cache_key and self.cache.enabled:
            self.cache.set(cache_key, result)
        return result


# =========================
# 5) 建议你为 IDS baseline 单独定义两套 prompt（更稳）
# =========================

IDS_PROMPT_REASON_ONLY_USER = """你是一个内容审查专家。
请阅读句子，并给出以“Reason:”开头的一段简短推理（不超过120字），用于指导之后选择类似示例。
不要输出三元组。

### 句子：
{text}
### 分析：
"""

IDS_PROMPT_REASON_AND_TRIPLES_USER = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组:

背景知识：
{lexicons}

示例：
{examples}

### 句子：
{text}

### 分析：
请先输出一行以“Reason:”开头的简短推理（不超过120字）。

### 三元组：
"""


# =========================
# 6) 用法示例（把它接到你现有 Retriever 上）
# =========================

if __name__ == "__main__":
    base = Retriever(model_path="models/base/bge-large-zh-v1.5", data_path="data/full/std/train.json")
    llm = VLLMChatClient(
        base_url="http://localhost:35000/v1",
        model="Qwen2.5-7B-Instruct",
        api_key="EMPTY"
    )

    ids = IDSRetriever(
        base_retriever=base,
        llm=llm,
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        prompt_reason_only_user=IDS_PROMPT_REASON_ONLY_USER,
        prompt_reason_and_triples_user=IDS_PROMPT_REASON_AND_TRIPLES_USER,
        lexicons_fn=lambda _text: "",   # baseline：不注入 lexicons
        config=IDSConfig(q=3, top_k=10), # q=3, top_k=Ke
        cache=SimpleDiskCache("./exps/baselines/IDS/cache_ids", enabled=True),
    )

    sel = ids.iterative_select("说河南人偷井盖的明明是北京人，我一个南方人都知道，东北人会不知道。东北人就会舔北京，然后拉着整个北方对抗南方，搞得像分裂国家一样。")
"""
# 你已有：
# base = Retriever(model_path=..., data_path=..., ...)

llm = VLLMChatClient(
    base_url="http://localhost:35000/v1",
    model="Qwen2.5-7B-Instruct",
    api_key="EMPTY"
)

ids = IDSRetriever(
    base_retriever=base,
    llm=llm,
    system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    prompt_reason_only_user=IDS_PROMPT_REASON_ONLY_USER,
    prompt_reason_and_triples_user=IDS_PROMPT_REASON_AND_TRIPLES_USER,
    lexicons_fn=lambda _text: "",   # baseline：不注入 lexicons
    config=IDSConfig(q=3, top_k=10), # q=3, top_k=Ke
    cache=SimpleDiskCache("./cache_ids", enabled=True),
)

# 只做迭代示例选择：
sel = ids.iterative_select("待分析文本", retriever_kwargs={"deduplicate": True, "threshold": 0.0})

# 完整 baseline（含 set-voting）：
pred = ids.predict_with_ids("待分析文本", retriever_kwargs={"deduplicate": True, "threshold": 0.0})
print(pred["final_triples"])
"""
