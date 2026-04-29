import os
import json
import time
import hashlib
import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from rag.core import Retriever
from utils.parser import parse_llm_output_trip
from tools.convert import output2triple


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
        tokenizer_path: str = "models/base/Qwen2.5-7B-Instruct",
        track_usage: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.tokenizer_path = tokenizer_path
        self.track_usage = track_usage
        self.usage_records: List[Dict[str, Any]] = []
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:
            self._tokenizer = False
            print(f"[WARN] Failed to load tokenizer for usage fallback: {exc}")
        return None if self._tokenizer is False else self._tokenizer

    def _count_prompt_tokens(self, messages: List[Dict[str, str]]) -> Optional[int]:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None
        try:
            return len(tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            ))
        except Exception:
            text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
            return len(tokenizer.encode(text, add_special_tokens=False))

    def _count_completion_tokens(self, text: str) -> Optional[int]:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None
        return len(tokenizer.encode(text or "", add_special_tokens=False))

    @staticmethod
    def _normalize_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
        usage = usage or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        return {
            "prompt_tokens": int(prompt) if prompt is not None else None,
            "completion_tokens": int(completion) if completion is not None else None,
            "total_tokens": int(total) if total is not None else None,
        }

    def _record_usage(
        self,
        messages: List[Dict[str, str]],
        content: str,
        api_usage: Optional[Dict[str, Any]],
        call_type: Optional[str],
    ) -> None:
        if not self.track_usage:
            return

        usage = self._normalize_usage(api_usage)
        source = "api"
        if usage["prompt_tokens"] is None:
            usage["prompt_tokens"] = self._count_prompt_tokens(messages)
            source = "tokenizer"
        if usage["completion_tokens"] is None:
            usage["completion_tokens"] = self._count_completion_tokens(content)
            source = "tokenizer"
        if usage["total_tokens"] is None and usage["prompt_tokens"] is not None and usage["completion_tokens"] is not None:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        self.usage_records.append({
            "call_index": len(self.usage_records),
            "call_type": call_type or "chat",
            "source": source,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "message_count": len(messages),
            "completion_chars": len(content or ""),
        })

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 0.5,
        top_k: int = 20,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        call_type: Optional[str] = None,
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
                content = data["choices"][0]["message"]["content"]
                self._record_usage(messages, content, data.get("usage"), call_type)
                return content
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
        user = self.prompt_reason_only_user.replace("{text}", text)
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
            call_type="reason_only",
        )
        return out

    def _llm_reason_and_triples(self, text: str, examples: str, lexicons: str) -> str:
        user = self.prompt_reason_and_triples_user.replace("{lexicons}", lexicons).replace("{examples}", examples).replace("{text}", text)
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
            call_type="reason_and_triples",
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
            demo_outs = [output2triple(o) for o in demo_outs] 
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
            demo_outs = [output2triple(o) for o in demo_outs]
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

        if not voted:
            # backoff 1：直接用最后一轮（IDS 常用）
            voted = per_round_triples_lists[-1]

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

IDS_PROMPT_REASON_AND_TRIPLES_USER = """你是一个内容审查专家，请你分析我的句子并且从原文中提取出一个或者多个三元组target, argument, target_proup)。
targeted_group 必须且只能从 {Racism, Sexism, LGBTQ, Region, others, non-hate} 中选择，禁止输出其它标签。

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


import traceback

class JSONLProgress:
    """
    用 JSONL 做进度保存与断点续传：
    - 结果逐条 append 到 .jsonl
    - 重跑时读取 .jsonl，收集已完成 key，自动跳过
    - 可随时 materialize 成最终的 .json（list）
    """
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        self.done_keys = set()
        self._load_existing()

    @staticmethod
    def _default_key(item: Dict[str, Any], idx: int) -> str:
        """
        优先使用 item['id']；否则用 content 的 sha1；再否则用 idx。
        """
        if isinstance(item, dict) and "id" in item:
            return str(item["id"])
        content = (item.get("content") if isinstance(item, dict) else None) or ""
        if content:
            return hashlib.sha1(content.encode("utf-8")).hexdigest()
        return str(idx)

    def _load_existing(self) -> None:
        if not os.path.exists(self.jsonl_path):
            return
        # 逐行读取，最后一行若写到一半导致 JSON 解析失败，则直接忽略
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    break
                key = obj.get("_key")
                if key is not None:
                    self.done_keys.add(str(key))

    def is_done(self, key: str) -> bool:
        return str(key) in self.done_keys

    def append(self, record: Dict[str, Any]) -> None:
        """
        追加写入一条记录，并立即 flush。
        """
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        self.done_keys.add(str(record.get("_key")))

    def read_records(self, sort_by_index: bool = True) -> List[Dict[str, Any]]:
        """
        读取当前 jsonl 中所有完整记录。
        """
        records = []
        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        records.append(obj)
                    except Exception:
                        break

        if sort_by_index:
            records.sort(key=lambda x: x.get("_index", 10**18))
        return records

    def materialize_json(
        self,
        json_path: str,
        sort_by_index: bool = True,
        info: Optional[Dict[str, Any]] = None,
        metric: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        将 jsonl 汇总为 json。传入 info 时写成 runner 风格的
        {"info": ..., "results": [...]}，方便和其他结果文件统一统计。
        """
        records = self.read_records(sort_by_index=sort_by_index)

        tmp = json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            if info is None:
                json.dump(records, f, ensure_ascii=False, indent=2)
            else:
                payload = {"info": info, "results": records}
                if metric is not None:
                    payload["metric"] = metric
                json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, json_path)


def _summarize_call_usage(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    missing = 0
    sources = Counter()
    for call in calls:
        sources[call.get("source", "unknown")] += 1
        for key in totals:
            value = call.get(key)
            if value is None:
                missing += 1
            else:
                totals[key] += int(value)
    return {
        **totals,
        "call_count": len(calls),
        "sources": dict(sources),
        "missing_token_fields": missing,
        "calls": calls,
    }


def _aggregate_usage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_calls = 0
    missing = 0
    sources = Counter()
    for record in records:
        usage = record.get("_token_usage") or {}
        for key in totals:
            value = usage.get(key)
            if value is None:
                continue
            totals[key] += int(value)
        total_calls += int(usage.get("call_count") or 0)
        missing += int(usage.get("missing_token_fields") or 0)
        sources.update(usage.get("sources") or {})
    return {
        "usage": totals,
        "total_samples": len(records),
        "status_counts": dict(Counter(record.get("status", "unknown") for record in records)),
        "call_count": total_calls,
        "sources": dict(sources),
        "missing_token_fields": missing,
    }


def _make_info(args: argparse.Namespace, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate = _aggregate_usage(records)
    return {
        "model": args.model,
        "seed": None,
        "shot_num": 0,
        "usage": aggregate["usage"],
        "ids_usage": {
            "total_samples": aggregate["total_samples"],
            "status_counts": aggregate["status_counts"],
            "call_count": aggregate["call_count"],
            "sources": aggregate["sources"],
            "missing_token_fields": aggregate["missing_token_fields"],
        },
        "config": {
            "test_data_file": args.test_data,
            "train_data_file": args.train_data,
            "retriever_model_path": args.retriever_model_path,
            "retriever_device": args.retriever_device,
            "q": args.q,
            "top_k": args.top_k,
            "max_reason_chars": args.max_reason_chars,
            "max_tokens_reason": args.max_tokens_reason,
            "max_tokens_full": args.max_tokens_full,
            "base_url": args.base_url,
            "tokenizer_path": args.tokenizer_path,
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _load_result_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported result JSON shape: {path}")


def _write_results_json(
    path: str,
    records: List[Dict[str, Any]],
    info: Dict[str, Any],
    metric: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {"info": info, "results": records}
    if metric is not None:
        payload["metric"] = metric
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _compute_metrics(records: List[Dict[str, Any]], metrics_json: str) -> Optional[Dict[str, Any]]:
    try:
        from metrics.metric_llm import LLMmetrics
        metrics = LLMmetrics(output_dir=os.path.dirname(metrics_json) or ".")
        metric_result = metrics.run(datas_list=records)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metric_result, f, ensure_ascii=False, indent=2)
        return metric_result
    except Exception as exc:
        print(f"[WARN] Metric computation failed: {exc}")
        return None


def _print_usage_report(records: List[Dict[str, Any]], label: str = "IDS") -> None:
    aggregate = _aggregate_usage(records)
    usage = aggregate["usage"]
    n = aggregate["total_samples"]
    avg_prompt = usage["prompt_tokens"] / n if n else 0
    avg_completion = usage["completion_tokens"] / n if n else 0
    print("method\tN\tprompt_tokens\tcompletion_tokens\ttotal_tokens\tavg_prompt\tavg_completion")
    print(
        f"{label}\t{n}\t{usage['prompt_tokens']}\t{usage['completion_tokens']}\t"
        f"{usage['total_tokens']}\t{avg_prompt:.2f}\t{avg_completion:.2f}"
    )
    print(f"[INFO] status_counts={aggregate['status_counts']} call_count={aggregate['call_count']} sources={aggregate['sources']}")


def run_ids(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    for path in [args.results_jsonl, args.results_json, args.metrics_json]:
        if args.overwrite and path and os.path.exists(path):
            os.remove(path)

    base = Retriever(
        model_path=args.retriever_model_path,
        data_path=args.train_data,
        device=args.retriever_device,
        cache_dir=args.retriever_cache_dir,
    )
    llm = VLLMChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        tokenizer_path=args.tokenizer_path,
        track_usage=True,
    )
    ids = IDSRetriever(
        base_retriever=base,
        llm=llm,
        system_prompt=args.system_prompt,
        prompt_reason_only_user=IDS_PROMPT_REASON_ONLY_USER,
        prompt_reason_and_triples_user=IDS_PROMPT_REASON_AND_TRIPLES_USER,
        lexicons_fn=lambda _text: "",
        config=IDSConfig(
            q=args.q,
            top_k=args.top_k,
            max_reason_chars=args.max_reason_chars,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k_gen=args.top_k_gen,
            max_tokens_reason=args.max_tokens_reason,
            max_tokens_full=args.max_tokens_full,
        ),
        cache=SimpleDiskCache(args.prediction_cache_dir, enabled=args.use_prediction_cache),
    )

    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data: List[Dict[str, Any]] = json.load(f)
    if args.limit is not None:
        test_data = test_data[: args.limit]

    results_jsonl_dir = os.path.dirname(args.results_jsonl)
    if results_jsonl_dir:
        os.makedirs(results_jsonl_dir, exist_ok=True)
    progress = JSONLProgress(args.results_jsonl)
    done_in_scope = sum(1 for idx, item in enumerate(test_data) if progress.is_done(JSONLProgress._default_key(item, idx)))
    pbar = tqdm(total=len(test_data), desc="Running IDS with token usage")
    pbar.update(done_in_scope)

    processed_since_materialize = 0
    for idx, item in enumerate(test_data):
        key = JSONLProgress._default_key(item, idx)
        if progress.is_done(key):
            continue

        text = item.get("content", "")
        gt_quads = item.get("quadruples", item.get("gt_quadruples"))
        usage_start = len(llm.usage_records)
        pred = None
        try:
            pred = ids.predict_with_ids(
                text,
                use_cache=args.use_prediction_cache,
                retriever_kwargs={"deduplicate": True, "threshold": args.retrieve_threshold},
            )
            final_triples = pred["final_triples"]
            pred_quads = parse_llm_output_trip(final_triples)
            calls = llm.usage_records[usage_start:]
            token_usage = _summarize_call_usage(calls)
            record = {
                **{k: v for k, v in item.items() if k != "quadruples"},
                "gt_quadruples": gt_quads,
                "pred_quadruples": pred_quads,
                "status": "success",
                "attempts": int(sum(pred.get("vote_counts", {}).values())) if pred.get("vote_counts") else 0,
                "_llm_calls": token_usage["call_count"],
                "_token_usage": token_usage,
                "_key": key,
                "_index": idx,
            }
            if args.store_trace:
                record["ids_trace"] = pred
        except Exception as exc:
            calls = llm.usage_records[usage_start:]
            token_usage = _summarize_call_usage(calls)
            record = {
                **{k: v for k, v in item.items() if k != "quadruples"},
                "gt_quadruples": gt_quads,
                "pred_quadruples": [],
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "_llm_calls": token_usage["call_count"],
                "_token_usage": token_usage,
                "_key": key,
                "_index": idx,
            }
            if args.stop_on_error:
                progress.append(record)
                raise

        progress.append(record)
        pbar.update(1)
        processed_since_materialize += 1

        if processed_since_materialize >= args.materialize_every:
            records = progress.read_records(sort_by_index=True)
            info = _make_info(args, records)
            progress.materialize_json(args.results_json, sort_by_index=True, info=info)
            processed_since_materialize = 0

    pbar.close()

    records = progress.read_records(sort_by_index=True)
    metric = _compute_metrics(records, args.metrics_json) if args.compute_metric else None
    info = _make_info(args, records)
    _write_results_json(args.results_json, records, info, metric=metric)
    _print_usage_report(records, label="IDS")
    print(f"[DONE] JSONL saved to: {args.results_jsonl}")
    print(f"[DONE] JSON  saved to: {args.results_json}")
    if args.compute_metric:
        print(f"[DONE] metric saved to: {args.metrics_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or report the IDS baseline with token usage tracking.")
    parser.add_argument("--run", action="store_true", help="Run IDS inference and record token usage.")
    parser.add_argument("--report", action="store_true", help="Print token usage from an existing result file.")
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--results-jsonl", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--materialize-every", type=int, default=10)
    parser.add_argument("--compute-metric", action="store_true", default=True)
    parser.add_argument("--no-compute-metric", dest="compute_metric", action="store_false")
    parser.add_argument("--store-trace", action="store_true", help="Store full IDS rounds/demos/raw outputs per sample.")
    parser.add_argument("--stop-on-error", action="store_true")

    parser.add_argument("--train-data", default="data/full/std/train.json")
    parser.add_argument("--test-data", default="data/full/std/test.json")
    parser.add_argument("--retriever-model-path", default="models/base/bge-large-zh-v1.5")
    parser.add_argument("--retriever-device", default=os.getenv("IDS_RETRIEVER_DEVICE", "cuda:0"))
    parser.add_argument("--retriever-cache-dir", default="exps/baselines/IDS/retriever_cache")
    parser.add_argument("--retrieve-threshold", type=float, default=0.0)

    parser.add_argument("--base-url", default=os.getenv("IDS_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:35000/v1")))
    parser.add_argument("--model", default=os.getenv("IDS_MODEL", "Qwen2.5-7B-Instruct"))
    parser.add_argument("--api-key", default=os.getenv("IDS_API_KEY", os.getenv("ALI_VLLM_API_KEY", "EMPTY")))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--tokenizer-path", default="models/base/Qwen2.5-7B-Instruct")
    parser.add_argument("--system-prompt", default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")

    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-reason-chars", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.5)
    parser.add_argument("--top-k-gen", type=int, default=20)
    parser.add_argument("--max-tokens-reason", type=int, default=256)
    parser.add_argument("--max-tokens-full", type=int, default=512)
    parser.add_argument("--use-prediction-cache", action="store_true")
    parser.add_argument("--prediction-cache-dir", default="exps/baselines/IDS/cache_ids_usage")

    args = parser.parse_args()
    if args.run:
        args.results_json = args.results_json or "exps/baselines/IDS/ids_results_with_usage.json"
        args.results_jsonl = args.results_jsonl or "exps/baselines/IDS/ids_results_with_usage.jsonl"
        args.metrics_json = args.metrics_json or "exps/baselines/IDS/ids_metrics_with_usage.json"
    else:
        args.results_json = args.results_json or "exps/baselines/IDS/ids_results.json"
        args.results_jsonl = args.results_jsonl or "exps/baselines/IDS/ids_results.jsonl"
        args.metrics_json = args.metrics_json or "exps/baselines/IDS/ids_metrics.json"
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.run:
        run_ids(args)
    else:
        records = _load_result_records(args.results_json)
        if args.report:
            _print_usage_report(records, label="IDS")
        elif args.compute_metric:
            metric = _compute_metrics(records, args.metrics_json)
            payload_info = None
            try:
                with open(args.results_json, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    payload_info = payload.get("info")
            except Exception:
                payload_info = None
            if payload_info is not None:
                _write_results_json(args.results_json, records, payload_info, metric=metric)
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
