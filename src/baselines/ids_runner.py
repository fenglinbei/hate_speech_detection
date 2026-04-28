from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from baselines.task_adapter import (
    binary_label_from_quadruples,
    binary_label_from_text,
    extract_reason,
    normalize_task_type,
    reason_and_answer_prompt,
    reason_only_prompt,
    result_record,
    structured_output_to_triples,
    task_type_from_configs,
    vote_answers,
)
from metrics.metric_llm import BinaryClassificationMetrics, HateXplainMetrics, LLMmetrics
from prompt import *  # noqa: F401,F403


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def resolve_prompt(value: str) -> str:
    if not isinstance(value, str):
        return str(value)
    return globals().get(value, value)


def make_seeded_output_name(base_name: str, seed: Any) -> str:
    path = Path(base_name)
    suffix = path.suffix if path.suffix else ".json"
    return f"{path.stem}_s{seed}{suffix}"


def make_summary_output_name(base_name: str) -> str:
    path = Path(base_name)
    suffix = path.suffix if path.suffix else ".json"
    return f"{path.stem}_multi_seed{suffix}"


class OpenAICompatibleChatClient:
    def __init__(self, model_name: str, api_base: str, api_key: str = "EMPTY", timeout: int = 120):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], params: dict[str, Any]) -> tuple[str, dict[str, int], int]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.5),
            "top_k": params.get("top_k", 20),
            "max_tokens": params.get("max_tokens", params.get("max_new_tokens", 512)),
            "n": params.get("n", 1),
            "seed": params.get("seed"),
            "chat_template_kwargs": {"enable_thinking": bool(params.get("enable_thinking", False))},
        }
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        status_code = response.status_code
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        answer = data["choices"][0]["message"]["content"]
        return answer, {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }, status_code


def build_retriever(build_config: dict, task_type: str):
    from rag.core import Retriever

    data_paths = build_config.get("data_paths", {})
    retrieval = build_config.get("retrieval_settings", {})
    model_settings = build_config.get("model_settings", {})
    cache_settings = build_config.get("cache_settings", {})
    ids_cache_dir = build_config.get("baseline", {}).get("ids", {}).get("cache_dir")
    model_path = model_settings.get("srag_model_path", "./models/base/bge-large-zh-v1.5")
    model_name = model_settings.get("srag_model_name")
    return Retriever(
        model_path=model_path,
        model_name=model_name,
        data_path=data_paths.get("raw_data_path"),
        task_type=task_type,
        stratify_field=retrieval.get("stratify_field", "targeted_group"),
        query_instruction=model_settings.get("srag_query_instruction", ""),
        cache_dir=ids_cache_dir or cache_settings.get("retrieval_cache_dir", "./cache_retrieval"),
        enable_cache=cache_settings.get("enable_retrieval_cache", True),
    )


def build_examples(
    demo_texts: list[str],
    demo_outputs: list[Any],
    example_template: str,
    task_type: str,
) -> str:
    task_type = normalize_task_type(task_type)
    examples = []
    for text, output in zip(demo_texts, demo_outputs):
        if task_type == "cold_binary":
            if isinstance(output, (dict, list)):
                demo_output = binary_label_from_quadruples(output)
            else:
                demo_output = binary_label_from_text(output)
        elif task_type == "structured":
            demo_output = structured_output_to_triples(output)
        else:
            demo_output = str(output if output is not None else "")
        examples.append(
            example_template.replace("{retrieve_content}", str(text))
            .replace("{retrieve_output}", demo_output)
        )
    return "\n".join(examples).strip()


def run_ids_for_dataset(
    config: dict,
    build_config: dict,
    output_name: str,
    llm_params: dict[str, Any],
) -> dict[str, Any]:
    tester_cfg = config.get("tester", {})
    baseline = config.get("baseline") or build_config.get("baseline") or {}
    ids_cfg = baseline.get("ids", {})
    task_type = task_type_from_configs(build_config, config)

    q = int(ids_cfg.get("q", 3))
    top_k = int(ids_cfg.get("top_k", 10))
    max_reason_chars = int(ids_cfg.get("max_reason_chars", 512))
    max_tokens_reason = int(ids_cfg.get("max_tokens_reason", 256))
    max_tokens_full = int(ids_cfg.get("max_tokens_full", llm_params.get("max_new_tokens", 512)))
    max_retries = int(tester_cfg.get("max_retries", 3))

    example_template = resolve_prompt(tester_cfg.get("prompt_templates", {}).get("example", "RAG_PROMPT_EXAMPLE_V2"))
    system_prompt = resolve_prompt(tester_cfg.get("prompt_templates", {}).get("system", "DEFAULT_SYSTEM_PTOMPT_EN"))
    if task_type == "cold_binary":
        system_prompt = resolve_prompt(tester_cfg.get("prompt_templates", {}).get("system", "COLD_BINARY_SYSTEM_PROMPT"))
    elif task_type == "hatexplain":
        system_prompt = resolve_prompt(tester_cfg.get("prompt_templates", {}).get("system", "HATEXPLAIN_SYSTEM_PROMPT"))

    model_cfg = config.get("model", {}).get("params", {})
    client = OpenAICompatibleChatClient(
        model_name=model_cfg.get("model_name", "qwen2.5"),
        api_base=model_cfg.get("api_base", "http://127.0.0.1:35000/v1/"),
        api_key=model_cfg.get("api_key", "EMPTY"),
        timeout=int(model_cfg.get("timeout", 120)),
    )
    retriever = build_retriever(build_config, task_type)

    test_data = load_json(tester_cfg["test_data_file"])
    results = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for item in tqdm(test_data, desc=f"IDS {task_type}"):
        raw_outputs: list[str] = []
        answers: list[str] = []
        attempts = 0
        try:
            reason_prompt = reason_only_prompt(item.get("content", ""), task_type)
            reason_params = {**llm_params, "max_tokens": max_tokens_reason, "max_new_tokens": max_tokens_reason}
            reason_raw = ""
            for attempt in range(max_retries + 1):
                attempts += 1
                try:
                    reason_raw, usage, _status = client.chat(
                        [{"role": "system", "content": system_prompt}, {"role": "user", "content": reason_prompt}],
                        reason_params,
                    )
                    for key in usage_total:
                        usage_total[key] += usage.get(key, 0)
                    break
                except Exception:
                    if attempt >= max_retries:
                        raise
                    time.sleep(2 ** attempt)

            reason = extract_reason(reason_raw)
            query = reason[:max_reason_chars]
            for _round in range(q):
                demo_texts, demo_outputs = retriever.retrieve(query=query, top_k=top_k)
                examples = build_examples(demo_texts, demo_outputs, example_template, task_type)
                user_prompt = reason_and_answer_prompt(
                    text=item.get("content", ""),
                    examples=examples,
                    lexicons="",
                    task_type=task_type,
                )
                full_params = {**llm_params, "max_tokens": max_tokens_full, "max_new_tokens": max_tokens_full}
                round_raw = ""
                for attempt in range(max_retries + 1):
                    attempts += 1
                    try:
                        round_raw, usage, _status = client.chat(
                            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                            full_params,
                        )
                        for key in usage_total:
                            usage_total[key] += usage.get(key, 0)
                        break
                    except Exception:
                        if attempt >= max_retries:
                            raise
                        time.sleep(2 ** attempt)

                raw_outputs.append(round_raw)
                answers.append(round_raw)
                next_reason = extract_reason(round_raw)
                query = (next_reason or query)[:max_reason_chars]

            final_answer = vote_answers(answers, task_type, q=q)
            result = result_record(item, final_answer, raw_outputs, task_type, attempts=attempts)
        except Exception as exc:
            result = {
                **item,
                "llm_output": None,
                "status": "error",
                "attempts": attempts,
                "error": str(exc),
            }
            if task_type == "cold_binary":
                result["pred_label"] = None
            elif task_type == "hatexplain":
                result["pred_annotation"] = None
                result["pred_quadruples"] = []
            else:
                result["pred_quadruples"] = []
        results.append(result)

    if tester_cfg.get("compute_metric", True):
        if task_type == "cold_binary":
            metric = BinaryClassificationMetrics().run(datas_list=results)
        elif task_type == "hatexplain":
            metric = HateXplainMetrics().run(datas_list=results)
        else:
            metric = LLMmetrics().run(datas_list=results)
    else:
        metric = None

    info = {
        "model": model_cfg.get("model_name", "qwen2.5"),
        "shot_num": tester_cfg.get("shot_num", 0),
        "seed": llm_params.get("seed", tester_cfg.get("seed", 23333333)),
        "usage": usage_total,
        "llm_params": llm_params,
        "config": {
            **tester_cfg,
            "baseline": {
                **baseline,
                "method": "ids",
                "task_type": normalize_task_type(task_type),
                "ids": {
                    "q": q,
                    "top_k": top_k,
                    "max_reason_chars": max_reason_chars,
                },
            },
        },
    }
    output_path = os.path.join(tester_cfg["output_dir"], output_name)
    payload = {"info": info, "results": results, "metric": metric}
    save_json(output_path, payload)
    return payload


def run(config: dict, build_config: dict) -> None:
    tester_cfg = config.get("tester", {})
    run_cfg = tester_cfg.get("run", {})
    base_output_name = config.get("output_name") or "ids.json"
    llm_params = run_cfg.get("llm_params", {})
    seeds_list = run_cfg.get("seeds_list")

    if isinstance(seeds_list, list) and seeds_list:
        per_seed = []
        for seed in seeds_list:
            seed_params = copy.deepcopy(llm_params)
            seed_params["seed"] = seed
            seed_name = make_seeded_output_name(base_output_name, seed)
            payload = run_ids_for_dataset(config, build_config, seed_name, seed_params)
            per_seed.append({"seed": seed, "output_file": seed_name, "metric": payload.get("metric"), "info": payload.get("info")})
        summary = {
            "info": {
                "model": config.get("model", {}).get("params", {}).get("model_name"),
                "seeds_list": seeds_list,
                "base_output_name": base_output_name,
                "generated_at": time.strftime("%Y%m%d_%H%M%S"),
                "baseline": config.get("baseline") or build_config.get("baseline") or {},
            },
            "per_seed": per_seed,
        }
        save_json(os.path.join(tester_cfg["output_dir"], make_summary_output_name(base_output_name)), summary)
        return

    run_ids_for_dataset(config, build_config, base_output_name, llm_params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run self-contained IDS baseline.")
    parser.add_argument("--config", required=True, help="Runner config path")
    parser.add_argument("--build-config", required=True, help="Build config path")
    args = parser.parse_args()
    run(load_json(args.config), load_json(args.build_config))


if __name__ == "__main__":
    main()
