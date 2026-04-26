import json
import argparse
from typing import List, Dict, Any, Set

# 允许的标签集合
ALLOWED_GROUPS = {"non-hate", "Region", "Racism", "Sexism", "LGBTQ", "others"}

def parse_targeted_groups(raw_group) -> List[str]:
    """
    将 targeted_group 字段解析成一个 group 列表。
    可能的情况：
      - 单标签字符串: "Sexism"
      - 多标签字符串: "Sexism, Racism" / "Sexism,Racism" / "Sexism、Racism"
      - 已经是列表: ["Sexism", "Racism"]
    只保留在 ALLOWED_GROUPS 中的标签。
    """
    if raw_group is None:
        return []

    # 如果本身就是列表，直接处理
    if isinstance(raw_group, list):
        groups = raw_group
    else:
        # 统一替换分隔符，再按逗号切分
        text = str(raw_group)
        # 兼容中文顿号、多种标点
        text = text.replace("；", ",").replace("、", ",").replace("，", ",")
        groups = [g.strip() for g in text.split(",")]

    cleaned = []
    for g in groups:
        if not g:
            continue
        # 如果标注中大小写不一致，可以统一转化，如 g = g.strip().title()
        if g in ALLOWED_GROUPS:
            cleaned.append(g)
        else:
            # 对于未在 ALLOWED_GROUPS 中的标注，你可以选择：
            # 1) 直接忽略
            # 2) 映射到 others
            # 这里选择忽略，由上层逻辑控制空集合时映射到 others
            pass
    return cleaned


def decide_sentence_label(sample: Dict[str, Any]) -> str:
    """
    根据策略 A 为单句决定一个六分类标签。
    逻辑：
      - 若不存在 hateful == "hate" 的 quadruple -> "non-hate"
      - 否则，从所有 hateful quadruple 的 targeted_group 中收集 group_set
        * 若 group_set 为空 -> "others"
        * 若 len(group_set) == 1 -> 该唯一 group
        * 若 len(group_set) >= 2 -> "others"（复合仇恨文本）
    """
    quads = sample.get("quadruples", [])
    if not isinstance(quads, list):
        return "non-hate"  # 异常情况兜底

    # 1. 过滤出 hateful quadruples
    hateful_quads = [q for q in quads if q.get("hateful") == "hate"]

    # 2. 若没有 hateful quadruples，则句子为 non-hate
    if not hateful_quads:
        return "non-hate"

    # 3. 收集所有 hateful quadruple 的 targeted_group（拆分后）
    group_set: Set[str] = set()
    for q in hateful_quads:
        raw_group = q.get("targeted_group")
        groups = parse_targeted_groups(raw_group)
        for g in groups:
            if g == "non-hate":
                # hateful= "hate" 但 targeted_group = "non-hate" 的情况极少，
                # 为安全起见忽略，让下方逻辑统一处理。
                continue
            group_set.add(g)

    # 4. 根据 group_set 决定句子级标签
    if not group_set:
        # 有 hateful，但没有解析出有效 group -> others
        return "others"

    if len(group_set) == 1:
        return next(iter(group_set))

    # 多个不同 group -> others（复合仇恨文本）
    return "others"


def convert_dataset(
    input_path: str,
    output_path: str,
    output_format: str = "jsonl"
) -> None:
    """
    将原始四元组数据转换为句子级六分类数据。

    输入格式示例（list[dict]）：
      [
        {
          "id": 2448,
          "content": "我的天啊，求求你坦白吧，放过人家",
          "quadruples": [...]
        },
        ...
      ]

    输出格式（每条一条句子级样本）：
      {"id": 2448, "content": "...", "label": "non-hate"}
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_samples = []
    for sample in data:
        sid = sample.get("id")
        text = sample.get("content", "")
        label = decide_sentence_label(sample)

        new_samples.append({
            "id": sid,
            "content": text,
            "label": label
        })

    if output_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for s in new_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    elif output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_samples, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert quadruple-based hate-speech dataset to sentence-level 6-way classification dataset."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to original JSON file.")
    parser.add_argument("--output", "-o", required=True, help="Path to output file.")
    parser.add_argument("--format", "-f", default="jsonl", choices=["jsonl", "json"],
                        help="Output format: jsonl (default) or json.")
    args = parser.parse_args()

    convert_dataset(args.input, args.output, args.format)


if __name__ == "__main__":
    main()
