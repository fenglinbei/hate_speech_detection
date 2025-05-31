import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 配置路径和指标
DATA_DIR = "few_shot/output"
SAVE_DIR = "few_shot/fig"
METRICS = ["f1_soft", "f1_hard", "f1_avg"]
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 不同模型的颜色
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']  # 不同模型的标记
MODELS = ["qwen3-8b", "qwen3-8b-think"]

os.makedirs(SAVE_DIR, exist_ok=True)

# 数据结构: {model: {shot: {metric: value}}}
results = defaultdict(lambda: defaultdict(dict))

# 解析文件名并提取数据
pattern = re.compile(r"output_(.*?)_(\d+)_\d+\.json$")
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".json"):
        match = pattern.search(filename)
        if match:
            model = match.group(1)
            shot = int(match.group(2))
            filepath = os.path.join(DATA_DIR, filename)
            
            try:
                if model in MODELS:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if "metric" in data:
                            for metric in METRICS:
                                if metric in data["metric"]:
                                    results[model][shot][metric] = data["metric"][metric]
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# 为每个指标创建图表
for metric in METRICS:
    plt.figure(figsize=(10, 6))
    
    # 为每个模型绘制折线
    for i, (model, shot_data) in enumerate(results.items()):
        # 按shot排序并提取数据
        shots_sorted = sorted(shot_data.keys())
        values = [shot_data[shot].get(metric, None) for shot in shots_sorted]
        
        # 跳过没有该指标数据的模型
        if all(v is None for v in values):
            continue
        
        # 绘制折线图
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        plt.plot(shots_sorted, values, 
                 label=model, 
                 color=color, 
                 marker=marker,
                 linestyle='-', 
                 linewidth=2,
                 markersize=8)
    
    # 设置图表属性
    plt.title(f'{metric.upper()} vs Number of Shots', fontsize=14)
    plt.xlabel('Number of Shots', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    
    # 自动调整横坐标刻度
    all_shots = set()
    for shot_data in results.values():
        all_shots.update(shot_data.keys())
    if all_shots:
        plt.xticks(sorted(all_shots))
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{metric}_vs_shots.png'), dpi=300)
    plt.close()

print("图表生成完成！请查看当前目录下的PNG文件")