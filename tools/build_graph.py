#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
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


def load_metric_block(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "metric" in data and isinstance(data["metric"], dict):
        return data["metric"]
    # 容错：metric 不在顶层时，尝试递归寻找
    found = find_first_key(data, "metric")
    if found is not None:
        return found
    raise KeyError(f"{path.name} 中未找到 'metric' 字段（或其不是 dict）")


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
    ap.add_argument("input", type=str, help="JSON 文件或目录路径")
    ap.add_argument("--pattern", type=str, default="*.json", help="当 input 为目录时的匹配模式（默认 *.json）")
    ap.add_argument("--recursive", action="store_true", help="递归搜索子目录")
    ap.add_argument("--name", choices=["basename", "stem", "relative"], default="stem",
                    help="CSV 表头/第一列使用的文件名形式：basename=含后缀，stem=不含后缀，relative=相对路径")
    ap.add_argument("--out1", type=str, default="metrics_format1.csv", help="图1格式输出 CSV 名称")
    ap.add_argument("--out2", type=str, default="metrics_format2.csv", help="图2格式输出 CSV 名称（两行分组表头）")
    ap.add_argument("--out2_flat", type=str, default="metrics_format2_flat.csv",
                    help="图2格式输出 CSV（单行扁平表头，便于程序读取）")
    ap.add_argument("--digits", type=int, default=4, help="小数保留位数（默认 4）")
    args = ap.parse_args()

    input_path = Path(args.input)
    files = collect_files(input_path, args.pattern, args.recursive)
    if not files:
        raise FileNotFoundError(f"未找到任何 JSON：{input_path} / pattern={args.pattern} / recursive={args.recursive}")

    # 读入并抽取指标
    records: Dict[str, Dict[str, Any]] = {}
    for fp in files:
        metric_block = load_metric_block(fp)
        metrics = extract_key_metrics(metric_block)

        if args.name == "basename":
            key = fp.name
        elif args.name == "relative":
            key = str(fp.relative_to(input_path if input_path.is_dir() else fp.parent))
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
