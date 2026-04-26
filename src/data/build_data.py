import json
import hashlib
import pickle
import os
import random
from dataclasses import replace
from tqdm import tqdm
from typing import Optional, List, Tuple, Any
from transformers import AutoTokenizer

from prompt import *
from utils.log import init_logger
logger = init_logger(level="DEBUG", show_console=True)
from data.config import Config
from rag.core import Retriever, LexiconRetriever, MultiClassRetriever, MultiClassWrongExpRetriever, ClusteredRetriever, StochasticWeightedRetriever
from rag.rag_retrieval_pipeline import MMRReterever, RETRIEVAL_PARAMS, main_build_index
from tools.convert import output2triple


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


def load_global_demo_examples(
    demos_path: str,
    example_template: str,
    top_k: int = -1,
    shuffle: bool = False,
    seed: int = 42,
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

    def __init__(self, cache_dir: str = './cache_build_data', enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def make_key(self, payload: dict) -> str:
        return hashlib.md5(_stable_dumps(payload).encode('utf-8')).hexdigest()

    def get(self, key: str):
        if not self.enabled:
            return None
        p = self._path(key)
        if not os.path.exists(p):
            return None
        try:
            with open(p, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, key: str, value):
        if not self.enabled:
            return
        p = self._path(key)
        try:
            with open(p, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            return

def get_tokenizer(model_path: str):
    """???tokenizer"""
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

def build_prompt(
        datas: list,
        config: Config,
        srag_retriever: Optional[MultiClassRetriever | Retriever | StochasticWeightedRetriever | MMRReterever] = None,
        lex_retriever: Optional[LexiconRetriever] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        is_test_data: bool = False,
        build_cache: Optional[BuildCacheManager] = None,
        global_examples: Optional[List[str]] = None,
        global_examples_sig: Optional[str] = None
        ):
    """Build prompts for normalized quadruple data."""
    retrieval_cache_enabled = bool(getattr(config, "enable_retrieval_cache", True))

    def render_prompt(raw_data: dict, examples: List[str], lex_contents: List[str]) -> str:
        return config.prompt_template.replace("{examples}", "\n".join(examples)).\
                                      replace("{lexicons}", "\n".join(lex_contents)).\
                                      replace("{text}", raw_data["content"])

    def build_single_prompt(
            raw_data: dict,
            srag_retriever: Optional[MultiClassRetriever | Retriever | StochasticWeightedRetriever], 
            lex_retriever: Optional[LexiconRetriever],
            global_examples: Optional[List[str]] = None,
            global_k: Optional[int] = None
        ):
        """Build one prompt for one normalized sample."""
        use_global_demos = bool(getattr(config, "use_global_demos", False)) and bool(global_examples)
        if use_global_demos:
            k = len(global_examples) if global_k is None else max(0, min(int(global_k), len(global_examples)))
            examples = global_examples[:k]
        elif config.use_srag and srag_retriever is not None and config.example_template is not None:
            if config.mmr and isinstance(srag_retriever, MMRReterever):
                retrieve_contents, retrieve_outputs = srag_retriever.retrieve(
                    query_id=raw_data['id'],
                    query_text=raw_data['content'],
                    n_shot=config.srag_top_k,
                    mmr_lambda=config.mmr_lambda
                )
            else:
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
                    use_cache=retrieval_cache_enabled
                )
            examples = []
            for retrieve_content, retrieve_output in zip(retrieve_contents, retrieve_outputs):
                try:
                    retrieve_output_text = output2triple(retrieve_output)
                except Exception:
                    retrieve_output_text = _quadruples_to_triples_fallback(retrieve_output)
                example_prompt = config.example_template.replace("{retrieve_content}", retrieve_content).\
                                                    replace("{retrieve_output}", retrieve_output_text)
                examples.append(example_prompt)
        else:
            examples = []

        if config.use_lex and lex_retriever is not None:
            lex_contents = lex_retriever.including_retrieve(
                raw_data['content'],
                config.lex_top_k,
                use_cache=retrieval_cache_enabled,
            )
            simlex_contents = lex_retriever.similarity_retrieve(
                raw_data['content'],
                config.lex_sim_top_k,
                deduplicate=True,
                threshold=config.lex_sim_threshold,
                use_cache=retrieval_cache_enabled,
            )
            for simlex_content in simlex_contents:
                if simlex_content not in lex_contents:
                    lex_contents.append(simlex_content)
        else:
            lex_contents = []

        prompt = render_prompt(raw_data, examples, lex_contents)

        return prompt, examples, lex_contents

    pbar = tqdm(
            total=len(datas),
            desc=f"Preprocessing datas",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )
    messages = []
    srag_examples_nums = 0

    # Per-sample prompt build cache.
    if build_cache is None:
        enable_build_cache = getattr(config, "enable_build_cache", True)
        build_cache_dir = getattr(config, "build_cache_dir", "./cache_build_data")
        build_cache = BuildCacheManager(cache_dir=build_cache_dir, enabled=enable_build_cache)

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

    for raw_data in datas:
        # Cache key includes config, retriever, tokenizer, and sample signatures.
        if cache_enabled:
            payload = {
                'id': raw_data.get('id'),
                'content_sha1': _sha1_text(raw_data.get('content', '')),
                'quadruples_sha1': _sha1_text(_stable_dumps(raw_data.get('quadruples', []))),
                'is_test_data': bool(is_test_data),
                **cache_static_payload,
            }
            _key = build_cache.make_key(payload)
            cached = build_cache.get(_key)
            if cached is not None:
                message, ex_len = cached
                messages.append(message)
                srag_examples_nums += int(ex_len or 0)
                pbar.update(1)
                continue

        triples = []
        for quadruple in raw_data["quadruples"]:
            label = quadruple["targeted_group"]
            triples.append(f"{quadruple['target']} | {quadruple['argument']} | {label}")
        # global demos (fixed demos for all samples)
        global_k = default_global_k

        prompt, examples, lex_contents = build_single_prompt(
            raw_data=raw_data,
            srag_retriever=srag_retriever,
            lex_retriever=lex_retriever,
            global_examples=global_examples,
            global_k=global_k
        )
        original_examples = examples

        # ?????????
        i = 1
        cur_len = token_length(tokenizer, prompt) if config.auto_length and tokenizer is not None else 0
        while config.auto_length and tokenizer is not None and cur_len > config.max_length:
            if use_global:
                new_k = max(0, int(global_k or 0) - i)
                print(f"Over length: {cur_len} > {config.max_length}, reduce global demos and rebuild prompt.")
                examples = global_examples[:new_k]
                prompt = render_prompt(raw_data, examples, lex_contents)
                global_k = new_k
                if new_k <= 0:
                    break
            else:
                print(f"Over length: {cur_len} > {config.max_length}, reduce srag examples and rebuild prompt.")
                new_k = max(0, min(len(original_examples), int(config.srag_top_k) - i))
                examples = original_examples[:new_k]
                prompt = render_prompt(raw_data, examples, lex_contents)
                if new_k <= 0:
                    break
            i += 1
            cur_len = token_length(tokenizer, prompt)

        srag_examples_nums += len(examples)

        answer = " [SEP] ".join(triples) + " [END]"
        message = {
            "id": raw_data["id"],
            "instruction": config.system_prompt if config.system_prompt else "", 
            "input": f"{prompt}", 
            "output": answer, 
            "content": raw_data["content"],
            "gt_quadruples": raw_data["quadruples"] if is_test_data else ""
            }
        messages.append(message)

        # ??????
        if cache_enabled:
            try:
                build_cache.set(_key, (message, len(examples)))
            except Exception:
                pass

        pbar.update(1)
    
    if len(datas) > 0:
        print(f"Avg examples nums: {srag_examples_nums / len(datas)}")

    return messages

def _legacy_make_data(config: Config):
    """Legacy data builder retained for reference."""

    messages = []
    with open(config.raw_data_path, "r") as file:
        raw_datas = json.load(file)

    split_idx = int(len(raw_datas) * config.split_ratio)

    tokenizer = None
    if config.auto_length and config.tokenizer_path is not None:
        tokenizer = get_tokenizer(config.tokenizer_path)

    enable_build_cache = getattr(config, 'enable_build_cache', True)
    build_cache_dir = getattr(config, 'build_cache_dir', './cache_build_data')
    build_cache = BuildCacheManager(cache_dir=build_cache_dir, enabled=enable_build_cache)

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


def _create_srag_retriever(config: Config, raw_datas: list[dict]) -> Optional[Any]:
    if not config.use_srag or bool(getattr(config, "use_global_demos", False)):
        return None

    target_groups = getattr(config, "target_groups", None)
    default_weights = getattr(config, "default_weights", None)

    if config.clustered:
        retriever = ClusteredRetriever(
            model_path=config.srag_model_path,
            model_name="bge-large-zh-v1.5",
            n_clusters=config.n_clusters,
            random_state=config.random_state,
        )
        retriever._load_datas(data_list=raw_datas)
        retriever._build_global_retriever()
        retriever._build_clusters()
        retriever._build_cluster_retrievers()
        return retriever

    if config.stratified:
        retriever = MultiClassRetriever(
            model_path=config.srag_model_path,
            model_name="bge-large-zh-v1.5",
            ramdom_strategy=config.ramdom_strategy,
            random_state=config.random_state,
            target_groups=target_groups,
            default_weights=default_weights,
        )
        retriever.load_datas(data_list=raw_datas)
        retriever.build_retrievers()
        return retriever

    if config.mmr:
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
            model_name="bge-large-zh-v1.5",
            random_state=config.random_state,
        )
    else:
        retriever = Retriever(
            model_path=config.srag_model_path,
            model_name="bge-large-zh-v1.5",
        )

    retriever.load_datas(data_list=raw_datas)
    retriever.create_embeddings(raw_datas)
    return retriever


def _create_lex_retriever(config: Config) -> Optional[LexiconRetriever]:
    if not config.use_lex:
        return None
    return LexiconRetriever(
        model_path=config.lexicon_model_path,
        model_name="bge-large-zh-v1.5",
        data_path=config.lexicon_data_path,
    )


def _write_jsonl(path: str, messages: list[dict]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def _write_runner_test_json(path: str, messages: list[dict], system_prompt: str) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump([
            {
                "id": message["id"],
                "content": message["content"],
                "gt_quadruples": message.get("gt_quadruples", []),
                "messages_list": [[
                    {"content": system_prompt, "role": "system"},
                    {"content": message["input"], "role": "user"},
                ]],
            }
            for message in messages
        ], file, ensure_ascii=False, indent=4)


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
    if config.auto_length and config.tokenizer_path is not None:
        tokenizer = get_tokenizer(config.tokenizer_path)

    build_cache = BuildCacheManager(
        cache_dir=getattr(config, "build_cache_dir", "./cache_build_data"),
        enabled=getattr(config, "enable_build_cache", True),
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
    _write_runner_test_json(config.test_output_path, test_messages, config.system_prompt)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Build training and validation data')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    config = Config(args.config)
    make_data(config)

