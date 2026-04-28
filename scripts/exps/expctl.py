#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
expctl.py
- gen: 读取 spec(JSON)，枚举 grid，生成独立实验目录，每个实验包含：
  - build_config.json
  - train_config.json
  - runner_config.json
  - manifest.json
"""

import argparse
import copy
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def config_path(path: Any) -> str:
    """Serialize paths with '/' so bash/Unix Python do not treat backslashes as filename characters."""
    return Path(path).as_posix()


def set_by_dotpath(d: dict, dotpath: str, value: Any) -> None:
    """在 dict 中按 'a.b.c' 形式写入 value。"""
    parts = dotpath.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def canonical_hash(payload: dict, length: int = 10) -> str:
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return h[:length]


def cartesian_grid(grid: Dict[str, List[Any]] | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand either a Cartesian grid dict or an explicit list of override dicts."""
    if isinstance(grid, list):
        return [dict(item) for item in grid]

    keys = list(grid.keys())
    values_list = [grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values_list):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def safe_name(value: Any) -> str:
    """Return a filesystem-friendly experiment name segment."""
    raw = str(value).strip()
    chars = []
    last_was_sep = False
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
            last_was_sep = False
        elif not last_was_sep:
            chars.append("_")
            last_was_sep = True
    return "".join(chars).strip("_.-")


def exp_dir_name(exp_id: str, name: Any = None) -> str:
    cleaned = safe_name(name) if name else ""
    return f"exp_{cleaned}_{exp_id}" if cleaned else f"exp_{exp_id}"


def split_meta_overrides(overrides: dict) -> tuple[dict, dict]:
    config_overrides = {}
    meta = {}
    for key, value in overrides.items():
        if key in {"name", "variant"}:
            meta[key] = value
        elif key.startswith("meta."):
            meta[key[len("meta."):]] = value
        else:
            config_overrides[key] = value
    return config_overrides, meta


def apply_overrides(base_build: dict, base_train: dict, base_runner: dict, overrides: dict) -> Tuple[dict, dict, dict, dict]:
    """
    overrides key 形式：
      - build.xxx.yyy
      - train.xxx.yyy
      - runner.xxx.yyy
      - reuse.data_dir
      - reuse.model_checkpoint

    返回：patched_build, patched_train, patched_runner, reuse_meta
    """
    b = copy.deepcopy(base_build)
    t = copy.deepcopy(base_train)
    r = copy.deepcopy(base_runner)

    reuse_meta = {}
    for k, v in overrides.items():
        if k.startswith("build."):
            set_by_dotpath(b, k[len("build."):], v)
        elif k.startswith("train."):
            set_by_dotpath(t, k[len("train."):], v)
        elif k.startswith("runner."):
            set_by_dotpath(r, k[len("runner."):], v)
        elif k.startswith("reuse."):
            reuse_meta[k[len("reuse."):]] = v
        else:
            raise ValueError(f"Unknown override prefix: {k}")

    return b, t, r, reuse_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen"], help="gen experiments from spec")
    ap.add_argument("--spec", required=True, help="path to spec json")
    args = ap.parse_args()

    spec = load_json(args.spec)

    project = spec.get("project", "experiments")
    output_root = Path(spec.get("output_root", f"experiments/{project}"))
    base = spec["base"]
    base_build_path = base["build"]
    base_train_path = base["train"]
    base_runner_path = base["runner"]

    base_build = load_json(base_build_path)
    base_train = load_json(base_train_path)
    base_runner = load_json(base_runner_path)

    # 全局默认 reuse（可被 grid 覆盖）
    reuse_default = spec.get("reuse", {})
    default_data_dir = reuse_default.get("data_dir", None)
    default_model_ckpt = reuse_default.get("model_checkpoint", None)

    grid = spec.get("grid", {})
    combos = cartesian_grid(grid) if grid else [dict()]

    # 端口：port_base + idx
    port_base = int(spec.get("port_base", 35000))

    # vLLM 默认参数写入 manifest（run 脚本读取）
    vllm = spec.get("vllm", {})
    vllm_defaults = {
        "cuda_visible_devices": vllm.get("cuda_visible_devices", "0,1,2,3"),
        "tensor_parallel_size": int(vllm.get("tensor_parallel_size", 4)),
        "max_model_len": int(vllm.get("max_model_len", 8192)),
        "served_model_name": vllm.get("served_model_name", "qwen2.5"),
    }

    # 训练 GPU（可选记录）
    train_cuda = spec.get("train_cuda_visible_devices", "0,1,2,3")

    # 生成
    output_root.mkdir(parents=True, exist_ok=True)
    index = 0
    for overrides in combos:
        # 注入全局默认 reuse（如果 overrides 未覆盖）
        # 注意：reuse.* 不属于 config patch 前缀，因此这里用 overrides 追加 reuse 前缀键
        merged_overrides = dict(overrides)
        if default_data_dir is not None and "reuse.data_dir" not in merged_overrides:
            merged_overrides["reuse.data_dir"] = default_data_dir
        if default_model_ckpt is not None and "reuse.model_checkpoint" not in merged_overrides:
            merged_overrides["reuse.model_checkpoint"] = default_model_ckpt

        config_overrides, meta = split_meta_overrides(merged_overrides)

        # exp_id 用 overrides 的 hash，保证可复现命名
        exp_id = canonical_hash({"project": project, "overrides": merged_overrides}, length=10)
        exp_name = exp_dir_name(exp_id, meta.get("name"))
        exp_dir = output_root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        build_cfg, train_cfg, runner_cfg, reuse_meta = apply_overrides(
            base_build, base_train, base_runner, config_overrides
        )

        # 统一实验内部目录
        exp_data_dir = exp_dir / "data"
        exp_model_dir = exp_dir / "model"
        exp_out_dir = exp_dir / "runner_output"
        exp_progress_dir = exp_dir / "progress"
        exp_prompts_dir = exp_dir / "prompts"
        exp_cache_dir = exp_dir / "cache"
        exp_logs_dir = exp_dir / "logs"

        for d in [exp_data_dir, exp_model_dir, exp_out_dir, exp_progress_dir, exp_prompts_dir, exp_cache_dir, exp_logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # ===== 复用数据（可选） =====
        reuse_data_dir = reuse_meta.get("data_dir", None)
        # 规范化：如果传了相对路径，按 repo cwd 解释；这里写入 manifest 为原样，run 脚本会 resolve
        # 如果复用数据，则 build 阶段可以跳过；但我们仍然生成 build_config 供记录
        if reuse_data_dir:
            train_path = config_path(Path(reuse_data_dir) / "train.jsonl")
            val_path = config_path(Path(reuse_data_dir) / "val.jsonl")
            test_path = config_path(Path(reuse_data_dir) / "test.json")
        else:
            train_path = config_path(exp_data_dir / "train.jsonl")
            val_path = config_path(exp_data_dir / "val.jsonl")
            test_path = config_path(exp_data_dir / "test.json")

        # ===== patch build_config 输出路径（固定写到 exp/data） =====
        # 即使 reuse_data_dir 存在，也让 build 输出写 exp/data（不影响），run 脚本会跳过 build
        build_cfg.setdefault("data_paths", {})
        build_cfg["data_paths"]["train_output_path"] = config_path(exp_data_dir / "train.jsonl")
        build_cfg["data_paths"]["val_output_path"] = config_path(exp_data_dir / "val.jsonl")
        build_cfg["data_paths"]["val_runner_output_path"] = config_path(exp_data_dir / "val_runner.json")
        build_cfg["data_paths"]["test_output_path"] = config_path(exp_data_dir / "test.json")

        # ===== patch train_config 输出路径与数据路径 =====
        train_cfg.setdefault("training", {})
        train_cfg["training"]["output_dir"] = config_path(exp_model_dir)
        # 让 exp_name/run_name 更直观
        train_cfg["exp_name"] = exp_name
        train_cfg.setdefault("project_name", project)

        train_cfg.setdefault("data", {})
        train_cfg["data"]["config_path"] = config_path(exp_dir / "build_config.json")
        train_cfg["data"]["train_data_path"] = train_path
        train_cfg["data"]["val_data_path"] = val_path

        # ===== patch runner_config 输出路径、缓存/进度路径、test 路径 =====
        runner_cfg.setdefault("tester", {})
        runner_cfg["tester"]["output_dir"] = config_path(exp_out_dir)
        runner_cfg["tester"]["progress_dir"] = config_path(exp_progress_dir)
        runner_cfg["tester"]["prompts_save_dir"] = config_path(exp_prompts_dir)
        runner_cfg["tester"]["cache_dir"] = config_path(exp_cache_dir)
        runner_cfg["tester"]["test_data_file"] = test_path

        # api_base 不写死端口，run 脚本会动态 patch
        runner_cfg.setdefault("model", {}).setdefault("params", {})
        runner_cfg["model"]["params"]["api_base"] = "http://127.0.0.1:0/v1/"

        # runner 输出名：用 exp_id，避免覆盖
        runner_cfg["output_name"] = f"exp_{exp_id}.json"

        # ===== manifest =====
        manifest = {
            "project": project,
            "exp_id": exp_id,
            "exp_dir": config_path(exp_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "index": index,
            "port": port_base + index,
            "overrides": merged_overrides,
            "meta": meta,
            "reuse": {
                "data_dir": reuse_data_dir,
                "model_checkpoint": reuse_meta.get("model_checkpoint", None),
            },
            "paths": {
                "build_config": config_path(exp_dir / "build_config.json"),
                "train_config": config_path(exp_dir / "train_config.json"),
                "runner_config": config_path(exp_dir / "runner_config.json"),
                "data_dir": config_path(exp_data_dir),
                "val_runner_file": config_path(exp_data_dir / "val_runner.json"),
                "model_dir": config_path(exp_model_dir),
                "runner_output_dir": config_path(exp_out_dir),
                "logs_dir": config_path(exp_logs_dir),
            },
            "runtime_defaults": {
                "train_cuda_visible_devices": train_cuda,
                "vllm": vllm_defaults,
            },
        }

        dump_json(build_cfg, exp_dir / "build_config.json")
        dump_json(train_cfg, exp_dir / "train_config.json")
        dump_json(runner_cfg, exp_dir / "runner_config.json")
        dump_json(manifest, exp_dir / "manifest.json")

        index += 1

    print(f"[OK] Generated {index} experiments under: {output_root}")


if __name__ == "__main__":
    main()
