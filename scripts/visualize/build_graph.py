#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
import glob
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("请先安装 pandas：pip install pandas") from e


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def safe_get(d: Dict[str, Any], keys: Tuple[str, ...], default=float("nan")) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def find_first_key(obj: Any, key: str) -> Optional[Dict[str, Any]]:
    """
    在嵌套结构中递归寻找第一个包含指定 key 的 dict，返回该 dict[key]（要求也是 dict）。
    """
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
        for v in obj.values():
            got = find_first_key(v, key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for it in obj:
            got = find_first_key(it, key)
            if got is not None:
                return got
    return None


def _is_agg_leaf(d: Any) -> bool:
    """Return True if d looks like an aggregate leaf: {'mean':.., 'std':.., ...}."""
    if not isinstance(d, dict):
        return False
    # allow n to be absent, but require at least one of mean/std
    has_mean = "mean" in d and _is_number(d.get("mean"))
    has_std = "std" in d and _is_number(d.get("std"))
    return has_mean or has_std


def _pick_agg_stat(d: Dict[str, Any], stat: str) -> Any:
    if stat == "std":
        return d.get("std", float("nan"))
    # default: mean
    return d.get("mean", float("nan"))


def metric_from_aggregate(agg: Any, stat: str = "mean") -> Any:
    """Convert an 'aggregate' tree to a metric-like tree by picking mean/std at leaves."""
    if _is_agg_leaf(agg):
        return _pick_agg_stat(agg, stat)
    if isinstance(agg, dict):
        return {k: metric_from_aggregate(v, stat) for k, v in agg.items()}
    if isinstance(agg, list):
        return [metric_from_aggregate(v, stat) for v in agg]
    return agg


def load_metric_block(path: Path, *, agg_stat: str = "mean") -> Dict[str, Any]:
    """Load 'metric' block from a JSON file.

    Supported formats:
      1) Standard single-run output: {'metric': {...}}
      2) Multi-seed aggregated output: {'aggregate': {...}, 'per_seed': [...]}
         - We convert 'aggregate' into a metric-like tree by selecting mean/std.
      3) Fallback: recursively find the first nested 'metric' dict.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    # (A) Multi-seed aggregation file
    if isinstance(data, dict) and "aggregate" in data and isinstance(data["aggregate"], dict):
        metric_like = metric_from_aggregate(data["aggregate"], agg_stat)
        if isinstance(metric_like, dict):
            return metric_like

    # (B) Standard single-run file
    if isinstance(data, dict) and "metric" in data and isinstance(data["metric"], dict):
        return data["metric"]

    # (C) Tolerant fallback: metric not at top level
    found = find_first_key(data, "metric")
    if found is not None:
        return found

    raise KeyError(f"{path.name} 中未找到 'metric' / 'aggregate' 字段（或其不是 dict）")


def extract_key_metrics(metric: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 JSON 的 metric 字段抽取为你要填表的关键指标。
    映射关系：
      f1_hard -> Triple_Hard
      f1_soft -> Triple_Soft
      f1_avg  -> Triple_Avg
      field_metrics.targeted_group.(precision/recall/f1) -> Target Group
      field_metrics.hateful.(precision/recall/f1)        -> Hateful
    """
    out: Dict[str, Any] = {}

    # Triple（来自 metric 顶层）
    out["Triple_Hard"] = safe_get(metric, ("f1_hard",))
    out["Triple_Soft"] = safe_get(metric, ("f1_soft",))
    out["Triple_Avg"]  = safe_get(metric, ("f1_avg",))

    # Target Group / Hateful（来自 field_metrics）
    out["Target_Group_Precision"] = safe_get(metric, ("field_metrics", "targeted_group", "precision"))
    out["Target_Group_Recall"]    = safe_get(metric, ("field_metrics", "targeted_group", "recall"))
    out["Target_Group_F1"]        = safe_get(metric, ("field_metrics", "targeted_group", "f1"))

    out["Hateful_Precision"] = safe_get(metric, ("field_metrics", "hateful", "precision"))
    out["Hateful_Recall"]    = safe_get(metric, ("field_metrics", "hateful", "recall"))
    out["Hateful_F1"]        = safe_get(metric, ("field_metrics", "hateful", "f1"))

    return out


def fmt(x: Any, digits: int) -> Any:
    if _is_number(x):
        return round(float(x), digits)
    return x


def collect_files(input_path: Path, pattern: str, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    if recursive:
        return sorted(input_path.rglob(pattern))
    return sorted(input_path.glob(pattern))



def _expand_path_like(s: str) -> List[Path]:
    """Expand ~ and glob patterns; return candidate Paths."""
    s = os.path.expanduser(s)
    matches = glob.glob(s, recursive=True)
    if matches:
        return [Path(m) for m in matches]
    return [Path(s)]


def collect_files_from_manual(items: List[str], pattern: str, recursive: bool) -> List[Path]:
    """Collect files from an explicit list of path-like strings (files/dirs/globs)."""
    files: List[Path] = []
    for it in items:
        it = it.strip()
        if not it:
            continue
        for p in _expand_path_like(it):
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                files.extend(collect_files(p, pattern, recursive))
            else:
                raise FileNotFoundError(f"路径不存在：{p}")

    # De-duplicate while preserving order
    seen: set = set()
    uniq: List[Path] = []
    for f in files:
        rf = f.resolve()
        if rf not in seen:
            uniq.append(rf)
            seen.add(rf)
    return uniq


def guess_relative_base(files: List[Path], preferred: Optional[Path] = None) -> Path:
    """Pick a base path for --name relative."""
    if preferred is not None:
        return preferred
    if not files:
        return Path(".")
    if len(files) == 1:
        return files[0].parent

    parents = [str(p.resolve().parent) for p in files]
    common = os.path.commonpath(parents)
    return Path(common)

def make_format1(records: Dict[str, Dict[str, Any]], digits: int) -> pd.DataFrame:
    """
    图1：行=指标，列=文件名
    """
    metric_order = [
        "Triple_Hard", "Triple_Soft", "Triple_Avg",
        "Target_Group_Precision", "Target_Group_Recall", "Target_Group_F1",
        "Hateful_Precision", "Hateful_Recall", "Hateful_F1",
    ]

    cols = list(records.keys())
    data = {c: [fmt(records[c].get(m, float("nan")), digits) for m in metric_order] for c in cols}
    df = pd.DataFrame(data, index=metric_order)
    df.index.name = "Metric"
    return df


def make_format2(records: Dict[str, Dict[str, Any]], digits: int) -> pd.DataFrame:
    """
    图2：行=文件名，列按 Target Group / Hateful / Triple 排布（扁平列名）
    """
    cols = [
        "Target_Group_Precision", "Target_Group_Recall", "Target_Group_F1",
        "Hateful_Precision", "Hateful_Recall", "Hateful_F1",
        "Triple_Hard", "Triple_Soft", "Triple_Avg",
    ]
    rows = []
    for name, m in records.items():
        row = {"Configuration": name}
        for c in cols:
            row[c] = fmt(m.get(c, float("nan")), digits)
        rows.append(row)
    return pd.DataFrame(rows)


def write_format2_grouped_csv(df: pd.DataFrame, out_csv: Path) -> None:
    """
    写出“分组两行表头”的 CSV，更接近图2的视觉效果（Excel 打开会更像）。
    第一行：分组名（Target Group / Hateful / Triple）
    第二行：子列名（Precision / Recall / F1 或 Hard / Soft / Avg）
    """
    # 期望列顺序：
    # Configuration,
    # Target_Group_Precision, Target_Group_Recall, Target_Group_F1,
    # Hateful_Precision, Hateful_Recall, Hateful_F1,
    # Triple_Hard, Triple_Soft, Triple_Avg
    col_order = [
        "Configuration",
        "Target_Group_Precision", "Target_Group_Recall", "Target_Group_F1",
        "Hateful_Precision", "Hateful_Recall", "Hateful_F1",
        "Triple_Hard", "Triple_Soft", "Triple_Avg",
    ]
    df2 = df[col_order].copy()

    header_row1 = [
        "Configuration",
        "Target Group", "Target Group", "Target Group",
        "Hateful", "Hateful", "Hateful",
        "Triple", "Triple", "Triple",
    ]
    header_row2 = [
        "Configuration",
        "Precision", "Recall", "F1",
        "Precision", "Recall", "F1",
        "Hard", "Soft", "Avg",
    ]

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header_row1)
        w.writerow(header_row2)
        for _, r in df2.iterrows():
            w.writerow([r[c] for c in col_order])


def main():
    ap = argparse.ArgumentParser(
        description="从一批实验 JSON 中抽取关键指标，输出两种 CSV（图1/图2格式）。"
    )
    ap.add_argument("input", type=str, nargs="?", default=None,
                    help="JSON 文件或目录路径（可选；若使用 --files/--file_list 可省略）")
    ap.add_argument("--files", nargs="+", default=None,
                    help="手动指定一个或多个 JSON 文件/目录（支持通配符，如 runner/output/*.json）")
    ap.add_argument("--file_list", type=str, default=None,
                    help="从文本文件读取 JSON 路径列表（每行一个；支持 # 注释与通配符）")
    ap.add_argument("--pattern", type=str, default="*.json", help="当 input 为目录时的匹配模式（默认 *.json）")
    ap.add_argument("--recursive", action="store_true", help="递归搜索子目录")
    ap.add_argument("--name", choices=["basename", "stem", "relative"], default="stem",
                    help="CSV 表头/第一列使用的文件名形式：basename=含后缀，stem=不含后缀，relative=相对路径")
    ap.add_argument("--out1", type=str, default="metrics_format1.csv", help="图1格式输出 CSV 名称")
    ap.add_argument("--out2", type=str, default="metrics_format2.csv", help="图2格式输出 CSV 名称（两行分组表头）")
    ap.add_argument("--out2_flat", type=str, default="metrics_format2_flat.csv",
                    help="图2格式输出 CSV（单行扁平表头，便于程序读取）")
    ap.add_argument("--digits", type=int, default=4, help="小数保留位数（默认 4）")
    ap.add_argument("--agg_stat", choices=["mean", "std"], default="mean",
                    help="当输入为多种子聚合 JSON（含 aggregate）时，选择读取 mean 或 std（默认 mean）")
    args = ap.parse_args()


    manual_items: List[str] = []
    if args.files:
        manual_items.extend(args.files)
    if args.file_list:
        p_list = Path(args.file_list)
        if not p_list.exists():
            raise FileNotFoundError(f"--file_list 指定的文件不存在：{p_list}")
        for line in p_list.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            manual_items.append(s)

    # 收集 JSON 文件
    preferred_base: Optional[Path] = None
    if args.input is not None:
        ip = Path(args.input)
        if ip.is_dir():
            preferred_base = ip.resolve()

    if manual_items:
        files = collect_files_from_manual(manual_items, args.pattern, args.recursive)
        input_path: Optional[Path] = None
    else:
        if args.input is None:
            raise SystemExit("请提供 input（文件/目录）或使用 --files/--file_list 手动指定 JSON 列表。")
        input_path = Path(args.input)
        files = collect_files(input_path, args.pattern, args.recursive)

    if not files:
        hint = args.input if args.input else "(manual list)"
        raise FileNotFoundError(f"未找到任何 JSON：{hint} / pattern={args.pattern} / recursive={args.recursive}")

    rel_base = guess_relative_base(files, preferred_base)

    # 读入并抽取指标
    records: Dict[str, Dict[str, Any]] = {}
    for fp in files:
        metric_block = load_metric_block(fp, agg_stat=args.agg_stat)
        metrics = extract_key_metrics(metric_block)

        if args.name == "basename":
            key = fp.name
        elif args.name == "relative":
            try:
                key = str(fp.resolve().relative_to(rel_base))
            except Exception:
                key = str(fp)
        else:  # stem
            key = fp.stem

        # 避免重名覆盖
        if key in records:
            key = f"{key}__{fp.name}"
        records[key] = metrics

    # 输出 format1（图1）
    df1 = make_format1(records, args.digits)
    df1.to_csv(args.out1, encoding="utf-8-sig")

    # 输出 format2（图2）
    df2 = make_format2(records, args.digits)
    df2.to_csv(args.out2_flat, index=False, encoding="utf-8-sig")  # 扁平表头版（程序友好）
    write_format2_grouped_csv(df2, Path(args.out2))                 # 两行分组表头版（Excel 更像图2）

    print(f"[OK] 图1格式：{args.out1}")
    print(f"[OK] 图2格式（分组两行表头）：{args.out2}")
    print(f"[OK] 图2格式（扁平单行表头）：{args.out2_flat}")


if __name__ == "__main__":
    main()
