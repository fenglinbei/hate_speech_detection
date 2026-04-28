from __future__ import annotations

import json
import hashlib
import pickle
import os
import random
import re
import time
from dataclasses import replace
from tqdm import tqdm
from typing import Optional, List, Tuple, Any

from prompt import *
from utils.log import init_logger
logger = init_logger(level="INFO", show_console=True)
from data.config import Config
from tools.convert import output2triple
from utils.sqlite_kv_cache import SQLiteKVCache


def _stable_dumps(obj):
    def _default(o):
        return str(o)
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_default)


def _sha1_text(s: str) -> str:
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def _config_signature(cfg: Config) -> str:
    d = {}
    for k, v in vars(cfg).items():
        if k.startswith('_'):
            continue
        if isinstance(v, (str, int, float, bool, type(None))):
            d[k] = v
        elif isinstance(v, (list, dict)):
            d[k] = v
        else:
            d[k] = str(v)
    return _sha1_text(_stable_dumps(d))


def _retriever_signature(r) -> str | None:
    if r is None:
        return None
    # base retriever / lexicon retriever
    sig = getattr(r, 'corpus_sig', None)
    if sig:
        return sig
    # multiclass retriever
    if hasattr(r, 'retrievers'):
        parts = []
        for cls, child in getattr(r, 'retrievers', {}).items():
            csig = getattr(child, 'corpus_sig', None)
            if csig:
                parts.append(f"{cls}:{csig}")
        if parts:
            return _sha1_text('|'.join(sorted(parts)))
    return None


def _file_sha1(path: str) -> str | None:
    """Return sha1 of file content. None if file not exists."""
    if not path:
        return None
    if not os.path.exists(path):
        return None
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _quadruples_to_triples_fallback(quadruples: Any) -> str:
    """A lightweight fallback when tools.convert.output2triple is incompatible."""
    if isinstance(quadruples, str):
        return quadruples
    triples = []
    if isinstance(quadruples, list):
        for q in quadruples:
            if isinstance(q, dict):
                label = q.get("targeted_group", q.get("label", ""))
                triples.append(f"{q.get('target', '')} | {q.get('argument', '')} | {label}")
            else:
                triples.append(str(q))
    return (" [SEP] ".join(triples) + " [END]") if triples else "[END]"


def _is_cold_binary_task(config: Any) -> bool:
    return str(getattr(config, "task_type", "")).strip().lower() == "cold_binary"


def _is_hatexplain_task(config: Any) -> bool:
    return str(getattr(config, "task_type", "")).strip().lower() == "hatexplain"


def _hatexplain_annotation_from_record(record: dict) -> dict:
    annotation = record.get("annotation") or record.get("gt_annotation") or {}
    label = str(annotation.get("label", "")).strip().lower()
    target_groups = annotation.get("target_groups", [])
    rationales = annotation.get("rationales", [])

    if not isinstance(target_groups, list):
        target_groups = [target_groups] if target_groups else []

    rationale_texts: list[str] = []
    if isinstance(rationales, list):
        for rationale in rationales:
            if isinstance(rationale, dict):
                text = str(rationale.get("text", "")).strip()
            else:
                text = str(rationale).strip()
            if text:
                rationale_texts.append(text)

    return {
        "label": label,
        "target_groups": [str(group).strip() for group in target_groups if str(group).strip()],
        "rationales": rationale_texts,
    }


def _format_hatexplain_output(record: dict) -> str:
    return json.dumps(_hatexplain_annotation_from_record(record), ensure_ascii=False, separators=(",", ":"))


def _normalize_binary_label(value: Any) -> str | None:
    if isinstance(value, bool):
        return "hate" if value else "non-hate"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "hate" if int(value) == 1 else "non-hate"

    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"1", "hate", "hateful", "toxic", "offensive", "abusive"}:
        return "hate"
    if text in {"0", "non-hate", "nonhate", "not-hate", "normal", "clean"}:
        return "non-hate"
    return None


def _binary_label_from_quadruples(quadruples: Any) -> str:
    if isinstance(quadruples, dict):
        quadruples = [quadruples]
    if not isinstance(quadruples, list):
        return _binary_label_from_text(quadruples)

    for quad in quadruples:
        if not isinstance(quad, dict):
            continue
        label = _normalize_binary_label(quad.get("hateful"))
        if label == "hate":
            return "hate"
        group = str(quad.get("targeted_group", "")).strip().lower().replace("_", "-")
        if group and group != "non-hate":
            return "hate"
    return "non-hate"


def _binary_label_from_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    label = _normalize_binary_label(text)
    if label:
        return label
    if re.search(r"\b(?:non|not)\s*-?\s*hate(?:ful)?\b", text):
        return "non-hate"

    parts = [part.strip() for part in text.replace("[end]", "").split("|")]
    if len(parts) >= 4:
        label = _normalize_binary_label(parts[3])
        if label:
            return label
    if len(parts) >= 3 and parts[2] and parts[2] != "non-hate":
        return "hate"
    return "non-hate" if "non-hate" in text else "hate"


def _binary_label_from_record(record: dict) -> str:
    if "hateful" in record:
        label = _normalize_binary_label(record.get("hateful"))
        if label:
            return label
    if "label" in record:
        label = _normalize_binary_label(record.get("label"))
        if label:
            return label
    return _binary_label_from_quadruples(record.get("quadruples", record.get("gt_quadruples", [])))


def _binary_label_from_retrieval_output(output: Any) -> str:
    if isinstance(output, list):
        return _binary_label_from_quadruples(output)
    if isinstance(output, dict):
        return _binary_label_from_quadruples(output)
    return _binary_label_from_text(output)


def load_global_demo_examples(
    demos_path: str,
    example_template: str,
    top_k: int = -1,
    shuffle: bool = False,
    seed: int = 42,
    task_type: str = "structured",
) -> tuple[list[str], str | None]:
    """Load a fixed demo file (e.g., demos_k10.json) and build example prompts.

    Returns:
      - examples: List[str], already rendered by example_template
      - demos_sig: sha1 signature of the demo file content (for cache key)
    """
    if not demos_path:
        return [], None
    with open(demos_path, "r", encoding="utf-8") as f:
        demos = json.load(f)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(demos)

    if top_k is not None and int(top_k) > 0:
        demos = demos[: int(top_k)]

    examples: list[str] = []
    for d in demos:
        retrieve_content = d.get("content", "")
        if task_type == "cold_binary":
            retrieve_output_text = _binary_label_from_record(d)
        elif task_type == "hatexplain":
            retrieve_output_text = _format_hatexplain_output(d)
        else:
            retrieve_output = d.get("quadruples", d.get("output", []))
            try:
                retrieve_output_text = output2triple(retrieve_output)
            except Exception:
                retrieve_output_text = _quadruples_to_triples_fallback(retrieve_output)

        ex = (
            example_template.replace("{retrieve_content}", retrieve_content)
            .replace("{retrieve_output}", retrieve_output_text)
        )
        examples.append(ex)

    demos_sig = _file_sha1(demos_path) or _sha1_text(_stable_dumps(demos))
    return examples, demos_sig


class BuildCacheManager:
    """Prompt-build cache keyed by sample/config/retriever/tokenizer signatures."""

    def __init__(
            self,
            cache_dir: str = './cache_build_data',
            enabled: bool = True,
            cache_backend: str = "sqlite"):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.cache_backend = cache_backend
        self.stage = "prompt"
        os.makedirs(cache_dir, exist_ok=True)
        self.kv = SQLiteKVCache(os.path.join(cache_dir, "build_cache.sqlite3"), enabled=enabled)

    def make_key(self, payload: dict) -> str:
        return hashlib.md5(_stable_dumps(payload).encode('utf-8')).hexdigest()

    def get(self, key: str):
        if not self.enabled:
            return None
        try:
            value = self.kv.get(self.stage, key)
            return pickle.loads(value) if value is not None else None
        except Exception:
            return None

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        if not self.enabled:
            return {key: None for key in keys}
        raw_values = self.kv.get_many(self.stage, keys)
        result = {}
        for key, value in raw_values.items():
            if value is None:
                result[key] = None
                continue
            try:
                result[key] = pickle.loads(value)
            except Exception:
                result[key] = None
        return result

    def set(self, key: str, value):
        if not self.enabled:
            return
        try:
            self.kv.set(self.stage, key, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return

    def set_many(self, values: dict[str, Any]) -> None:
        if not self.enabled:
            return
        encoded = {}
        for key, value in values.items():
            try:
                encoded[key] = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                continue
        self.kv.set_many(self.stage, encoded)

    def close(self) -> None:
        self.kv.close()

def get_tokenizer(model_path: str):
    """???tokenizer"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=True, 
        trust_remote_code=True
    )
    return tokenizer

def is_overlength(tokenizer, text, max_length):
    """Return True when text exceeds max_length under tokenizer."""
    return token_length(tokenizer, text) > max_length


def token_length(tokenizer, text) -> int:
    """Return token count without materializing a torch tensor."""
    return len(tokenizer.encode(text, add_special_tokens=True))


def _decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def truncate_prompt_from_tail(tokenizer, text: str, max_length: int) -> tuple[str, int, int, bool]:
    """Keep the prompt prefix and drop tail tokens until it fits max_length."""
    max_length = int(max_length or 0)
    if tokenizer is None or max_length <= 0:
        return text, 0, 0, False

    token_ids = tokenizer.encode(text, add_special_tokens=True)
    original_len = len(token_ids)
    if original_len <= max_length:
        return text, original_len, original_len, False

    keep_len = max_length
    truncated = _decode_token_ids(tokenizer, token_ids[:keep_len])
    truncated_len = token_length(tokenizer, truncated)
    while truncated_len > max_length and keep_len > 0:
        keep_len -= 1
        truncated = _decode_token_ids(tokenizer, token_ids[:keep_len])
        truncated_len = token_length(tokenizer, truncated)

    return truncated, original_len, truncated_len, True

def build_prompt(
        datas: list,
        config: Config,
        srag_retriever: Optional[Any] = None,
        lex_retriever: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        is_test_data: bool = False,
        build_cache: Optional[BuildCacheManager] = None,
        global_examples: Optional[List[str]] = None,
        global_examples_sig: Optional[str] = None
        ):
    """Build prompts for normalized quadruple data."""
    retrieval_cache_enabled = bool(getattr(config, "enable_retrieval_cache", True))
    retrieval_batch_size = int(getattr(config, "retrieval_batch_size", 256) or 256)
    cold_binary = _is_cold_binary_task(config)
    hatexplain = _is_hatexplain_task(config)
    phase = "test" if is_test_data else "train/val"
    logger.info(
        f"[BuildData] Prompt build start: phase={phase}, items={len(datas)}, "
        f"srag={bool(config.use_srag and srag_retriever is not None)}, "
        f"lexicon={bool(config.use_lex and lex_retriever is not None)}, "
        f"retrieval_cache={retrieval_cache_enabled}"
    )

    def render_prompt(raw_data: dict, examples: List[str], lex_contents: List[str]) -> str:
        return config.prompt_template.replace("{examples}", "\n".join(examples)).\
                                      replace("{lexicons}", "\n".join(lex_contents)).\
                                      replace("{text}", raw_data["content"])

    def render_examples(retrieve_contents: list[str], retrieve_outputs: list[Any]) -> list[str]:
        examples = []
        for retrieve_content, retrieve_output in zip(retrieve_contents, retrieve_outputs):
            if cold_binary:
                retrieve_output_text = _binary_label_from_retrieval_output(retrieve_output)
            elif hatexplain:
                if isinstance(retrieve_output, str):
                    retrieve_output_text = retrieve_output
                elif isinstance(retrieve_output, dict):
                    retrieve_output_text = json.dumps(retrieve_output, ensure_ascii=False, separators=(",", ":"))
                else:
                    retrieve_output_text = str(retrieve_output)
            else:
                try:
                    retrieve_output_text = output2triple(retrieve_output)
                except Exception:
                    retrieve_output_text = _quadruples_to_triples_fallback(retrieve_output)
            example_prompt = config.example_template.replace("{retrieve_content}", retrieve_content).\
                                                replace("{retrieve_output}", retrieve_output_text)
            examples.append(example_prompt)
        return examples

    def retrieve_srag_examples_batch(raw_items: list[dict], global_k: Optional[int]) -> list[list[str]]:
        if not raw_items:
            return []

        use_global = bool(getattr(config, "use_global_demos", False)) and bool(global_examples)
        if use_global:
            k = len(global_examples) if global_k is None else max(0, min(int(global_k), len(global_examples)))
            logger.info(f"[BuildData] SRAG batch retrieval skipped: using {k} global demos.")
            return [global_examples[:k] for _ in raw_items]

        if not (config.use_srag and srag_retriever is not None and config.example_template is not None):
            logger.info("[BuildData] SRAG batch retrieval skipped: disabled or retriever unavailable.")
            return [[] for _ in raw_items]

        start_time = time.perf_counter()
        logger.info(
            f"[BuildData] SRAG batch retrieval start: items={len(raw_items)}, "
            f"top_k={config.srag_top_k}, retriever={type(srag_retriever).__name__}"
        )

        if config.mmr and type(srag_retriever).__name__ == "MMRReterever":
            all_examples = []
            with tqdm(
                    raw_items,
                    desc="SRAG batch retrieval",
                    unit="item",
                    dynamic_ncols=True,
                    leave=True,
                    ) as pbar:
                for raw_data in pbar:
                    retrieve_contents, retrieve_outputs = srag_retriever.retrieve(
                        query_id=raw_data['id'],
                        query_text=raw_data['content'],
                        n_shot=config.srag_top_k,
                        mmr_lambda=config.mmr_lambda
                    )
                    all_examples.append(render_examples(retrieve_contents, retrieve_outputs))
                logger.info(
                    f"[BuildData] SRAG batch retrieval done in {time.perf_counter() - start_time:.1f}s."
                )
            return all_examples

        if hasattr(srag_retriever, "retrieve_batch"):
            with tqdm(
                total=1,
                desc="SRAG batch retrieval",
                unit="batch",
                dynamic_ncols=True,
                leave=True,
            ) as pbar:
                retrieval_results = srag_retriever.retrieve_batch(
                    queries=[item["content"] for item in raw_items],
                    top_k=config.srag_top_k,
                    threshold=config.srag_threshold,
                    weights=config.weights,
                    weights_reverse=config.weights_reverse,
                    similarity_alpha=config.similarity_alpha,
                    random_strategy=config.ramdom_strategy,
                    random_ratio=config.random_ratio,
                    temperature=config.random_temperature,
                    candidate_multiplier=config.candidate_multiplier,
                    use_cache=retrieval_cache_enabled,
                    batch_size=retrieval_batch_size,
                )
                pbar.update(1)
            logger.info(f"[BuildData] SRAG batch retrieval done in {time.perf_counter() - start_time:.1f}s.")
            return [render_examples(contents, outputs) for contents, outputs in retrieval_results]

        all_examples = []
        with tqdm(
                raw_items,
                desc="SRAG batch retrieval",
                unit="item",
                dynamic_ncols=True,
                leave=True,
                ) as pbar:
            for raw_data in pbar:
                retrieve_contents, retrieve_outputs = srag_retriever.retrieve(
                    raw_data['content'],
                    config.srag_top_k,
                    threshold=config.srag_threshold,
                    weights=config.weights,
                    weights_reverse=config.weights_reverse,
                    similarity_alpha=config.similarity_alpha,
                    random_strategy=config.ramdom_strategy,
                    random_ratio=config.random_ratio,
                    temperature=config.random_temperature,
                    candidate_multiplier=config.candidate_multiplier,
                    use_cache=retrieval_cache_enabled,
                )
                all_examples.append(render_examples(retrieve_contents, retrieve_outputs))
        logger.info(f"[BuildData] SRAG batch retrieval done in {time.perf_counter() - start_time:.1f}s.")
        return all_examples

    def retrieve_lex_batch(raw_items: list[dict]) -> list[list[str]]:
        if not raw_items:
            return []

        if not (config.use_lex and lex_retriever is not None):
            logger.info("[BuildData] Lexicon retrieval skipped: disabled or retriever unavailable.")
            return [[] for _ in raw_items]

        queries = [item["content"] for item in raw_items]
        start_time = time.perf_counter()
        logger.info(
            f"[BuildData] Lexicon retrieval start: items={len(raw_items)}, "
            f"include_top_k={config.lex_top_k}, similarity_top_k={config.lex_sim_top_k}"
        )

        with tqdm(
            total=2,
            desc="Lexicon retrieval",
            unit="stage",
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            pbar.set_postfix_str("including")
            if hasattr(lex_retriever, "including_retrieve_batch"):
                including_results = lex_retriever.including_retrieve_batch(
                    queries=queries,
                    top_k=config.lex_top_k,
                    use_cache=retrieval_cache_enabled,
                )
            else:
                including_results = [
                    lex_retriever.including_retrieve(query, config.lex_top_k, use_cache=retrieval_cache_enabled)
                    for query in queries
                ]
            pbar.update(1)

            pbar.set_postfix_str("similarity")
            if hasattr(lex_retriever, "similarity_retrieve_batch"):
                similarity_results = lex_retriever.similarity_retrieve_batch(
                    queries=queries,
                    top_k=config.lex_sim_top_k,
                    deduplicate=True,
                    threshold=config.lex_sim_threshold,
                    use_cache=retrieval_cache_enabled,
                    batch_size=retrieval_batch_size,
                )
            else:
                similarity_results = [
                    lex_retriever.similarity_retrieve(
                        query,
                        config.lex_sim_top_k,
                        deduplicate=True,
                        threshold=config.lex_sim_threshold,
                        use_cache=retrieval_cache_enabled,
                    )
                    for query in queries
                ]
            pbar.update(1)

        all_lexicons = []
        for lex_contents, simlex_contents in zip(including_results, similarity_results):
            merged = list(lex_contents)
            for simlex_content in simlex_contents:
                if simlex_content not in merged:
                    merged.append(simlex_content)
            all_lexicons.append(merged)
        logger.info(f"[BuildData] Lexicon retrieval done in {time.perf_counter() - start_time:.1f}s.")
        return all_lexicons

    srag_examples_nums = 0

    # Per-sample prompt build cache.
    if build_cache is None:
        enable_build_cache = getattr(config, "enable_build_cache", True)
        build_cache_dir = getattr(config, "build_cache_dir", "./cache_build_data")
        build_cache = BuildCacheManager(
            cache_dir=build_cache_dir,
            enabled=enable_build_cache,
            cache_backend=getattr(config, "cache_backend", "sqlite"),
        )

    cache_enabled = build_cache is not None and getattr(build_cache, 'enabled', False)
    cache_static_payload = {}
    if cache_enabled:
        cache_static_payload = {
            'config_sig': _config_signature(config),
            'use_global_demos': bool(getattr(config, 'use_global_demos', False)),
            'global_demos_sig': global_examples_sig,
            'global_demos_top_k': getattr(config, 'global_demos_top_k', None),
            'tokenizer': getattr(tokenizer, 'name_or_path', None) if tokenizer is not None else None,
            'max_length': getattr(config, 'max_length', None),
            'srag_sig': _retriever_signature(srag_retriever),
            'lex_sig': _retriever_signature(lex_retriever),
            'retriever_type': type(srag_retriever).__name__ if srag_retriever is not None else None,
        }

    use_global_demos = bool(getattr(config, "use_global_demos", False)) and bool(global_examples)
    default_global_k = None
    if use_global_demos:
        top_k = int(getattr(config, "global_demos_top_k", -1) or -1)
        default_global_k = min(top_k, len(global_examples)) if top_k > 0 else len(global_examples)

    messages: list[dict | None] = [None] * len(datas)
    cache_keys: list[str | None] = [None] * len(datas)
    missing_indices: list[int] = []

    cache_start_time = time.perf_counter()
    if cache_enabled:
        logger.info(f"[BuildData] Prompt cache lookup start: items={len(datas)}")

    for idx, raw_data in enumerate(datas):
        if cache_enabled:
            payload = {
                'id': raw_data.get('id'),
                'content_sha1': _sha1_text(raw_data.get('content', '')),
                'quadruples_sha1': _sha1_text(_stable_dumps(raw_data.get('quadruples', []))),
                'annotation_sha1': _sha1_text(_stable_dumps(raw_data.get('annotation', {}))),
                'metadata_sha1': _sha1_text(_stable_dumps(raw_data.get('metadata', {}))),
                'is_test_data': bool(is_test_data),
                **cache_static_payload,
            }
            _key = build_cache.make_key(payload)
            cache_keys[idx] = _key
        else:
            missing_indices.append(idx)

    if cache_enabled:
        keyed_indices = [idx for idx, key in enumerate(cache_keys) if key is not None]
        cached_values = build_cache.get_many([cache_keys[idx] for idx in keyed_indices])
        for idx in keyed_indices:
            cached = cached_values.get(cache_keys[idx])
            if cached is None:
                missing_indices.append(idx)
            else:
                message, ex_len = cached
                messages[idx] = message
                srag_examples_nums += int(ex_len or 0)

    missing_datas = [datas[idx] for idx in missing_indices]
    cache_hits = len(datas) - len(missing_datas)
    if cache_enabled:
        logger.info(
            f"[BuildData] Prompt cache lookup done in {time.perf_counter() - cache_start_time:.1f}s: "
            f"hits={cache_hits}, misses={len(missing_datas)}"
        )

    batch_examples = retrieve_srag_examples_batch(missing_datas, default_global_k)
    batch_lexicons = retrieve_lex_batch(missing_datas)
    cache_updates = {}

    if missing_datas:
        render_start_time = time.perf_counter()
        logger.info(f"[BuildData] Prompt rendering start: items={len(missing_datas)}")

        with tqdm(
                enumerate(missing_datas),
                total=len(missing_datas),
                desc="Prompt rendering",
                unit="item",
                dynamic_ncols=True,
                leave=True,
                ) as render_pbar:
            for local_idx, raw_data in render_pbar:
                original_idx = missing_indices[local_idx]
                triples = []
                if not cold_binary and not hatexplain:
                    triples = [
                        f"{quadruple['target']} | {quadruple['argument']} | {quadruple['targeted_group']}"
                        for quadruple in raw_data["quadruples"]
                    ]
                global_k = default_global_k
                examples = batch_examples[local_idx]
                lex_contents = batch_lexicons[local_idx]
                prompt = render_prompt(raw_data, examples, lex_contents)
                original_examples = examples

                # When auto_length is enabled, keep the original behavior:
                # reduce demonstrations before rebuilding the prompt.
                i = 1
                cur_len = token_length(tokenizer, prompt) if config.auto_length and tokenizer is not None else 0
                while config.auto_length and tokenizer is not None and cur_len > config.max_length:
                    if use_global_demos:
                        new_k = max(0, int(global_k or 0) - i)
                        logger.debug(
                            f"Over length: {cur_len} > {config.max_length}, "
                            "reduce global demos and rebuild prompt."
                        )
                        examples = global_examples[:new_k]
                        prompt = render_prompt(raw_data, examples, lex_contents)
                        global_k = new_k
                        if new_k <= 0:
                            break
                    else:
                        logger.debug(
                            f"Over length: {cur_len} > {config.max_length}, "
                            "reduce srag examples and rebuild prompt."
                        )
                        new_k = max(0, min(len(original_examples), int(config.srag_top_k) - i))
                        examples = original_examples[:new_k]
                        prompt = render_prompt(raw_data, examples, lex_contents)
                        if new_k <= 0:
                            break
                    i += 1
                    cur_len = token_length(tokenizer, prompt)

                if not config.auto_length and tokenizer is not None:
                    prompt, original_len, truncated_len, was_truncated = truncate_prompt_from_tail(
                        tokenizer,
                        prompt,
                        config.max_length,
                    )
                    if was_truncated:
                        logger.debug(
                            f"Prompt tail-truncated: {original_len} > {config.max_length}, "
                            f"new_len={truncated_len}."
                        )

                srag_examples_nums += len(examples)

                if cold_binary:
                    answer = _binary_label_from_record(raw_data)
                elif hatexplain:
                    answer = _format_hatexplain_output(raw_data)
                else:
                    answer = " [SEP] ".join(triples) + " [END]"

                message = {
                    "id": raw_data["id"],
                    "instruction": config.system_prompt if config.system_prompt else "",
                    "input": f"{prompt}",
                    "output": answer,
                    "content": raw_data["content"],
                    "metadata": raw_data.get("metadata", {}),
                    "gt_quadruples": raw_data.get("quadruples", []) if is_test_data else "",
                }
                if cold_binary:
                    message["gt_label"] = answer
                if hatexplain:
                    message["gt_annotation"] = raw_data.get("annotation", {}) if is_test_data else ""
                messages[original_idx] = message

                # ??????
                if cache_enabled and cache_keys[original_idx] is not None:
                    cache_updates[cache_keys[original_idx]] = (message, len(examples))

        logger.info(f"[BuildData] Prompt rendering done in {time.perf_counter() - render_start_time:.1f}s.")

    if cache_enabled and cache_updates:
        cache_write_start_time = time.perf_counter()
        logger.info(f"[BuildData] Prompt cache write start: items={len(cache_updates)}")
        build_cache.set_many(cache_updates)
        logger.info(f"[BuildData] Prompt cache write done in {time.perf_counter() - cache_write_start_time:.1f}s.")

    if len(datas) > 0:
        print(f"Avg examples nums: {srag_examples_nums / len(datas)}")
    logger.info(f"[BuildData] Prompt build done: items={len(datas)}, cache_hits={cache_hits}")

    return [message for message in messages if message is not None]

def _legacy_make_data(config: Config):
    """Legacy data builder retained for reference."""

    messages = []
    with open(config.raw_data_path, "r") as file:
        raw_datas = json.load(file)

    split_idx = int(len(raw_datas) * config.split_ratio)

    tokenizer = None
    if getattr(config, "tokenizer_path", None):
        tokenizer = get_tokenizer(config.tokenizer_path)

    enable_build_cache = getattr(config, 'enable_build_cache', True)
    build_cache_dir = getattr(config, 'build_cache_dir', './cache_build_data')
    build_cache = BuildCacheManager(
        cache_dir=build_cache_dir,
        enabled=enable_build_cache,
        cache_backend=getattr(config, "cache_backend", "sqlite"),
    )

    # Global fixed demos (optional): bypass SRAG and reuse the same demos for every sample
    global_examples: Optional[List[str]] = None
    global_examples_sig: Optional[str] = None
    use_global_demos = bool(getattr(config, "use_global_demos", False)) and bool(getattr(config, "global_demos_path", None))
    if use_global_demos:
        try:
            global_examples, global_examples_sig = load_global_demo_examples(
                demos_path=config.global_demos_path,
                example_template=config.example_template,
                top_k=getattr(config, "global_demos_top_k", -1),
                shuffle=getattr(config, "global_demos_shuffle", False),
                seed=getattr(config, "global_demos_seed", 42),
                task_type=getattr(config, "task_type", "structured"),
            )
            logger.info(f"[GlobalDemos] Loaded {len(global_examples)} demos from {config.global_demos_path}")
        except Exception as e:
            logger.warning(f"[GlobalDemos] Failed to load demos from {getattr(config, 'global_demos_path', None)}: {e}")
            global_examples, global_examples_sig = [], None
            use_global_demos = False


        if config.use_srag and not use_global_demos:
            if config.clustered:
                srag_retriever = ClusteredRetriever(
                    model_path=config.srag_model_path, 
                    model_name="bge-large-zh-v1.5",
                    n_clusters=config.n_clusters,
                    random_state=config.random_state
                )

                srag_retriever._load_datas(data_list=raw_datas[:split_idx])
                srag_retriever._build_global_retriever()
                srag_retriever._build_clusters()
                srag_retriever._build_cluster_retrievers()

            elif config.stratified:
                srag_retriever = MultiClassRetriever(
                    model_path=config.srag_model_path, 
                    model_name="bge-large-zh-v1.5",
                    ramdom_strategy=config.ramdom_strategy,
                    random_state=config.random_state
                )
                srag_retriever.load_datas(data_list=raw_datas[:split_idx])
                srag_retriever.build_retrievers()
            elif config.mmr:
                srag_retriever = MMRReterever(
                    data_path=config.raw_data_path,
                    model_path="models/base/bge-large-zh-v1.5",
                    index_path="./cache_retrieval/faiss_hnsw.index",
                    docs_path="./cache_retrieval/doc_store.json",
                    params=RETRIEVAL_PARAMS,
                )
            else:
                if config.ramdom_strategy != "none":
                    srag_retriever = StochasticWeightedRetriever(
                    model_path=config.srag_model_path, 
                    model_name="bge-large-zh-v1.5",
                    random_state=config.random_state
                )
                else:
                    srag_retriever = Retriever(
                        model_path=config.srag_model_path, 
                        model_name="bge-large-zh-v1.5"
                    )
                srag_retriever.load_datas(data_list=raw_datas[:split_idx])
                srag_retriever.create_embeddings(raw_datas[:split_idx])
        else:
            srag_retriever = None

        if config.use_lex:
            lex_retriever = LexiconRetriever(
                model_path=config.lexicon_model_path, 
                model_name="bge-large-zh-v1.5", 
                data_path=config.lexicon_data_path
            )
        else:
            lex_retriever = None

        # Write training data.
        messages = build_prompt(
            datas=raw_datas[:split_idx],
            config=config,
            srag_retriever=srag_retriever,
            lex_retriever=lex_retriever,
            tokenizer=tokenizer,
            build_cache=build_cache,
            global_examples=global_examples,
            global_examples_sig=global_examples_sig
        )

        with open(config.train_output_path, "w", encoding="utf-8") as file:
            for message in messages:
                file.write(json.dumps(message, ensure_ascii=False) + "\n")
        
        # Rebuild retriever for validation/test visibility.
        if config.use_srag and srag_retriever is not None:
            if config.clustered:
                srag_retriever = ClusteredRetriever(
                    model_path=config.srag_model_path, 
                    model_name="bge-large-zh-v1.5",
                    n_clusters=config.n_clusters,
                    random_state=config.random_state
                )
                srag_retriever._load_datas(data_list=raw_datas)
                srag_retriever._build_global_retriever()
                srag_retriever._build_clusters()
                srag_retriever._build_cluster_retrievers()
            elif config.stratified:
                srag_retriever = MultiClassRetriever(
                    model_path=config.srag_model_path, 
                    model_name="bge-large-zh-v1.5",
                    ramdom_strategy=config.ramdom_strategy,
                    random_state=config.random_state
                )
                srag_retriever.load_datas(data_list=raw_datas)
                srag_retriever.build_retrievers()
            elif config.mmr:
                srag_retriever = srag_retriever
            else:
                if config.ramdom_strategy != "none":
                    srag_retriever = StochasticWeightedRetriever(
                    model_path=config.srag_model_path, 
                    model_name="bge-large-zh-v1.5",
                    random_state=config.random_state
                )
                else:
                    srag_retriever = Retriever(
                        model_path=config.srag_model_path, 
                        model_name="bge-large-zh-v1.5"
                    )
                srag_retriever.load_datas(data_list=raw_datas)
                srag_retriever.create_embeddings(raw_datas)
            

        # Write validation data.
        messages = build_prompt(
            datas=raw_datas[split_idx:],
            config=config,
            srag_retriever=srag_retriever,
            lex_retriever=lex_retriever,
            tokenizer=tokenizer,
            build_cache=build_cache,
            global_examples=global_examples,
            global_examples_sig=global_examples_sig
        )

        with open(config.val_output_path, "w", encoding="utf-8") as file:
            for message in messages:
                file.write(json.dumps(message, ensure_ascii=False) + "\n")

        # Write test data.
        with open(config.test_data_path, "r") as file:
            test_datas = json.load(file)

        messages = build_prompt(
            datas=test_datas,
            config=config,
            srag_retriever=srag_retriever,
            lex_retriever=lex_retriever,
            tokenizer=tokenizer,
            is_test_data=True,
            build_cache=build_cache,
            global_examples=global_examples,
            global_examples_sig=global_examples_sig
        )

        with open(config.test_output_path, "w", encoding="utf-8") as file:
            json.dump([{
                    "id": message["id"], 
                    "content": message["content"], 
                    "gt_quadruples": message.get("gt_quadruples", []), 
                    "messages_list": [[
                        {'content': config.system_prompt, 'role': 'system'}, 
                        {'content': message["input"], 'role': 'user'}
                    ]],
                } for message in messages], file, ensure_ascii=False, indent=4)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _build_mmr_params(config: Config):
    from rag.rag_retrieval_pipeline import RETRIEVAL_PARAMS

    os.makedirs(config.mmr_cache_dir, exist_ok=True)
    trace_path = os.path.join(config.mmr_cache_dir, "selection_trace.jsonl")
    return replace(
        RETRIEVAL_PARAMS,
        cache_dir=config.mmr_cache_dir,
        trace_jsonl_path=trace_path,
        n_shot=config.srag_top_k,
        mmr_lambda=config.mmr_lambda,
        seed=config.random_state,
    )


def _model_name_from_path(model_name: Optional[str], model_path: Optional[str]) -> str:
    if model_name:
        return model_name
    if model_path:
        return os.path.basename(os.path.normpath(model_path))
    return "sentence-transformer"


def _ensure_local_model_path(model_path: Optional[str]) -> None:
    if not model_path:
        return
    is_local = model_path.startswith(".") or model_path.startswith("/") or os.path.sep in model_path
    if is_local and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Embedding model path does not exist: {model_path}. "
            "Prepare the model locally or override the config path."
        )


def _create_srag_retriever(config: Config, raw_datas: list[dict]) -> Optional[Any]:
    if not config.use_srag or bool(getattr(config, "use_global_demos", False)):
        return None

    from rag.core import ClusteredRetriever, MultiClassRetriever, Retriever, StochasticWeightedRetriever

    target_groups = getattr(config, "target_groups", None)
    default_weights = getattr(config, "default_weights", None)
    model_name = _model_name_from_path(getattr(config, "srag_model_name", None), getattr(config, "srag_model_path", None))
    _ensure_local_model_path(getattr(config, "srag_model_path", None))
    common_task = {
        "task_type": getattr(config, "task_type", "structured"),
        "stratify_field": getattr(config, "stratify_field", "targeted_group"),
        "query_instruction": getattr(config, "srag_query_instruction", ""),
    }
    common_cache = {
        "cache_dir": getattr(config, "retrieval_cache_dir", "./cache_retrieval"),
        "enable_cache": getattr(config, "enable_retrieval_cache", True),
    }

    if config.clustered:
        retriever = ClusteredRetriever(
            model_path=config.srag_model_path,
            model_name=model_name,
            n_clusters=config.n_clusters,
            random_state=config.random_state,
            **common_cache,
            **common_task,
        )
        retriever._load_datas(data_list=raw_datas)
        retriever._build_global_retriever()
        retriever._build_clusters()
        retriever._build_cluster_retrievers()
        return retriever

    if config.stratified:
        retriever = MultiClassRetriever(
            model_path=config.srag_model_path,
            model_name=model_name,
            ramdom_strategy=config.ramdom_strategy,
            random_state=config.random_state,
            target_groups=target_groups,
            default_weights=default_weights,
            **common_cache,
            **common_task,
        )
        retriever.load_datas(data_list=raw_datas)
        retriever.build_retrievers()
        return retriever

    if config.mmr:
        from rag.rag_retrieval_pipeline import MMRReterever, main_build_index

        if not os.path.exists(config.mmr_index_path) or not os.path.exists(config.mmr_docs_path):
            _ensure_parent_dir(config.mmr_index_path)
            _ensure_parent_dir(config.mmr_docs_path)
            main_build_index(
                data_path=config.raw_data_path,
                model_path=config.srag_model_path,
                index_path=config.mmr_index_path,
                docs_path=config.mmr_docs_path,
                params=_build_mmr_params(config),
            )
        return MMRReterever(
            data_path=config.raw_data_path,
            model_path=config.srag_model_path,
            index_path=config.mmr_index_path,
            docs_path=config.mmr_docs_path,
            params=_build_mmr_params(config),
        )

    if config.ramdom_strategy != "none":
        retriever = StochasticWeightedRetriever(
            model_path=config.srag_model_path,
            model_name=model_name,
            random_state=config.random_state,
            **common_cache,
            **common_task,
        )
    else:
        retriever = Retriever(
            model_path=config.srag_model_path,
            model_name=model_name,
            **common_cache,
            **common_task,
        )

    retriever.load_datas(data_list=raw_datas)
    retriever.create_embeddings(raw_datas)
    return retriever


def _create_lex_retriever(config: Config) -> Optional[Any]:
    if not config.use_lex:
        return None
    from rag.core import LexiconRetriever

    _ensure_local_model_path(getattr(config, "lexicon_model_path", None))
    return LexiconRetriever(
        model_path=config.lexicon_model_path,
        model_name=_model_name_from_path(getattr(config, "lexicon_model_name", None), getattr(config, "lexicon_model_path", None)),
        data_path=config.lexicon_data_path,
        cache_dir=getattr(config, "lexicon_cache_dir", "./cache_lexicon"),
        enable_cache=getattr(config, "enable_retrieval_cache", True),
        lexicon_schema=getattr(config, "lexicon_schema", "cold"),
        match_mode=getattr(config, "lexicon_match_mode", "substring"),
        case_sensitive=getattr(config, "lexicon_case_sensitive", True),
        include_variants=getattr(config, "lexicon_include_variants", False),
        query_instruction=getattr(config, "lexicon_query_instruction", ""),
    )


def _write_jsonl(path: str, messages: list[dict]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def _write_runner_test_json(
        path: str,
        messages: list[dict],
        system_prompt: str,
        task_type: str = "structured",
        ) -> None:
    _ensure_parent_dir(path)
    cold_binary = task_type == "cold_binary"
    hatexplain = task_type == "hatexplain"
    records = []
    for message in messages:
        record = {
            "id": message["id"],
            "content": message["content"],
            "metadata": message.get("metadata", {}),
            "gt_quadruples": message.get("gt_quadruples", []),
            "messages_list": [[
                {"content": system_prompt, "role": "system"},
                {"content": message["input"], "role": "user"},
            ]],
        }
        if cold_binary:
            record["gt_label"] = message.get("gt_label") or _binary_label_from_record(message)
        if hatexplain:
            record["gt_annotation"] = message.get("gt_annotation") or {}
        records.append(record)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=4)


def make_data(config: Config):
    with open(config.raw_data_path, "r", encoding="utf-8") as file:
        raw_datas = json.load(file)

    with open(config.test_data_path, "r", encoding="utf-8") as file:
        test_datas = json.load(file)

    if getattr(config, "val_data_path", None) and os.path.exists(config.val_data_path):
        train_datas = raw_datas
        with open(config.val_data_path, "r", encoding="utf-8") as file:
            val_datas = json.load(file)
    else:
        split_idx = int(len(raw_datas) * config.split_ratio)
        train_datas = raw_datas[:split_idx]
        val_datas = raw_datas[split_idx:]

    tokenizer = None
    if getattr(config, "tokenizer_path", None):
        tokenizer = get_tokenizer(config.tokenizer_path)

    build_cache = BuildCacheManager(
        cache_dir=getattr(config, "build_cache_dir", "./cache_build_data"),
        enabled=getattr(config, "enable_build_cache", True),
        cache_backend=getattr(config, "cache_backend", "sqlite"),
    )

    global_examples: Optional[List[str]] = None
    global_examples_sig: Optional[str] = None
    use_global_demos = bool(getattr(config, "use_global_demos", False)) and bool(getattr(config, "global_demos_path", None))
    config.use_global_demos = use_global_demos
    if use_global_demos:
        try:
            global_examples, global_examples_sig = load_global_demo_examples(
                demos_path=config.global_demos_path,
                example_template=config.example_template,
                top_k=getattr(config, "global_demos_top_k", -1),
                shuffle=getattr(config, "global_demos_shuffle", False),
                seed=getattr(config, "global_demos_seed", 42),
                task_type=getattr(config, "task_type", "structured"),
            )
            logger.info(f"[GlobalDemos] Loaded {len(global_examples)} demos from {config.global_demos_path}")
        except Exception as e:
            logger.warning(f"[GlobalDemos] Failed to load demos from {getattr(config, 'global_demos_path', None)}: {e}")
            global_examples, global_examples_sig = [], None
            config.use_global_demos = False

    lex_retriever = _create_lex_retriever(config)

    train_retriever = _create_srag_retriever(config, train_datas)
    train_messages = build_prompt(
        datas=train_datas,
        config=config,
        srag_retriever=train_retriever,
        lex_retriever=lex_retriever,
        tokenizer=tokenizer,
        build_cache=build_cache,
        global_examples=global_examples,
        global_examples_sig=global_examples_sig,
    )
    _write_jsonl(config.train_output_path, train_messages)

    eval_retriever = _create_srag_retriever(config, raw_datas)
    val_messages = build_prompt(
        datas=val_datas,
        config=config,
        srag_retriever=eval_retriever,
        lex_retriever=lex_retriever,
        tokenizer=tokenizer,
        build_cache=build_cache,
        global_examples=global_examples,
        global_examples_sig=global_examples_sig,
    )
    _write_jsonl(config.val_output_path, val_messages)

    val_runner_output_path = getattr(config, "val_runner_output_path", None)
    if val_runner_output_path:
        _write_runner_test_json(
            val_runner_output_path,
            val_messages,
            config.system_prompt,
            task_type=getattr(config, "task_type", "structured"),
        )

    test_messages = build_prompt(
        datas=test_datas,
        config=config,
        srag_retriever=eval_retriever,
        lex_retriever=lex_retriever,
        tokenizer=tokenizer,
        is_test_data=True,
        build_cache=build_cache,
        global_examples=global_examples,
        global_examples_sig=global_examples_sig,
    )
    _write_runner_test_json(
        config.test_output_path,
        test_messages,
        config.system_prompt,
        task_type=getattr(config, "task_type", "structured"),
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Build training and validation data')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    config = Config(args.config)
    make_data(config)

