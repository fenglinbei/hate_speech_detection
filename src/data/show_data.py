import argparse
import json
import os
import sys

#!/usr/bin/env python3
# show_data.py
# 用法:
#   python show_data.py -f path/to/file.jsonl -n 5


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] 无法解析第 {lineno} 行: {e}", file=sys.stderr)

def main():
    p = argparse.ArgumentParser(description="读取 jsonl 文件并打印几条数据")
    p.add_argument("-f", "--file", default="data.jsonl", help="jsonl 文件路径（默认: data.jsonl）")
    p.add_argument("-n", type=int, default=5, help="打印记录数（默认: 5）")
    args = p.parse_args()

    if not os.path.exists(args.file):
        print(f"文件不存在: {args.file}", file=sys.stderr)
        sys.exit(2)

    printed = 0
    for lineno, obj in iter_jsonl(args.file):
        print(f"--- line {lineno} ---")
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        print(f"--- input {lineno} ---")
        print(obj.get("input", ""))
        printed += 1
        if printed >= args.n:
            break

    if printed == 0:
        print("未找到任何记录。", file=sys.stderr)

if __name__ == "__main__":
    main()