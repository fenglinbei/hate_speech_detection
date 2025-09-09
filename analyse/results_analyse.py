from metrics.metric_llm import LLMmetrics
from metrics.core import sort_by_hard_example

TARGETED_GROUPS = ["non-hate", "Region", "Racism", "Sexism", "LGBTQ", "others"]
HATEFUL = ["hate", "non-hate"]
LENGTHS = [0, 50, 100, 150, 200, 250, 300]

METRIC = LLMmetrics()

def analyse_target_group(data_list: list[dict]) -> dict:

    results_dict = {}

    for targeted_group in TARGETED_GROUPS:
        new_data_list = []
        for data in data_list:
            if targeted_group in [quadruple["targeted_group"] for quadruple in data["gt_quadruples"]]:
                new_data_list.append(data)
        results_dict[targeted_group] = METRIC.run(new_data_list)
    
    return results_dict

def analyse_hateful(data_list: list[dict]) -> dict:
    results_dict = {}
    for hateful in HATEFUL:
        new_data_list = []
        for data in data_list:
            if hateful in [quadruple["hateful"] for quadruple in data["gt_quadruples"]]:
                new_data_list.append(data)
        results_dict[hateful] = METRIC.run(new_data_list)
    return results_dict

def analyse_length(data_list: list[dict]) -> dict:
    results_dict = {}
    for i in range(len(LENGTHS)-1):
        new_data_list = []
        for data in data_list:
            if LENGTHS[i] <= len(data["content"]) < LENGTHS[i+1]:
                new_data_list.append(data)
        results_dict[f"{LENGTHS[i]}-{LENGTHS[i+1]}"] = METRIC.run(new_data_list)
    # 处理大于最大长度的情况
    new_data_list = []
    for data in data_list:
        if len(data["content"]) >= LENGTHS[-1]:
            new_data_list.append(data)
    results_dict[f"{LENGTHS[-1]}+"] = METRIC.run(new_data_list)
    return results_dict

def get_hard_examples(data_list: list[dict]) -> list[tuple[dict, float, float]]:
    pred_data_dict, gt_data_dict = METRIC._load_data_from_dict(data_list)
    sorted_hard_example_ids, match_scores, label_scores = sort_by_hard_example(list(gt_data_dict.keys()), pred_data_dict, gt_data_dict)
    return [(data_list[i], match_scores[i], label_scores[i]) for i in sorted_hard_example_ids][:10]

def draw_pie(data: dict, title: str, save_path: str):
    import matplotlib.pyplot as plt

    labels = data.keys()
    sizes = [i["total"] for i in data.values()]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True, explode=[0.05]*len(labels))
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(save_path)
    plt.close()

def draw_bar(data: dict, title: str, save_path: str):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = data.keys()
    metrics = ["F1 Hard", "F1 Soft", "F1 Avg"]
    values = []
    for label in labels:
        entry = data[label]
        values.append([entry["f1_hard"], entry["f1_soft"], entry["f1_avg"]])
    
    values = np.array(values)

    # 创建图表
    plt.figure(figsize=(12, 8), dpi=100)
    plt.style.use('ggplot')  # 使用美观的样式

    # 设置位置参数
    x = np.arange(len(labels))  # 标签位置
    width = 0.2  # 每个柱形的宽度

    # 为每个指标绘制柱形
    rects1 = plt.bar(x - width, values[:, 0], width, label=metrics[0], color='#4C72B0')
    rects2 = plt.bar(x, values[:, 1], width, label=metrics[1], color='#55A868')
    rects3 = plt.bar(x + width, values[:, 2], width, label=metrics[2], color='#C44E52')

    # 添加标题和标签
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(x, labels, fontsize=11)
    plt.yticks(fontsize=10)

    # 添加图例
    plt.legend(loc='upper left', fontsize=11, frameon=True)

    # 添加数据标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    # 添加网格线
    plt.grid(axis='y', alpha=0.4, linestyle='--')

    # 调整布局
    plt.tight_layout()
    plt.margins(y=0.1)  # 增加顶部空间

    plt.savefig(save_path)
    plt.close()

def get_results(data_list):
    import json
    targeted_group_results = analyse_target_group(data_list)
    hateful_results = analyse_hateful(data_list)
    length_results = analyse_length(data_list)

    draw_pie(targeted_group_results, "Targeted Group Analysis", "analyse/targeted_group_analysis.png")
    draw_pie(hateful_results, "Hateful Analysis", "analyse/hateful_analysis.png")
    draw_pie(length_results, "Length Analysis", "analyse/length_analysis.png")

    draw_bar(targeted_group_results, "Targeted Group Analysis", "analyse/targeted_group_analysis_bar.png")
    draw_bar(hateful_results, "Hateful Analysis", "analyse/hateful_analysis_bar.png")
    draw_bar(length_results, "Length Analysis", "analyse/length_analysis_bar.png")
    
    final_results = {
        "targeted_group": targeted_group_results,
        "hateful": hateful_results,
        "length": length_results
    }

    with open("analyse/results_analysis.json", "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    import json
    with open("runner/output/simlex5_rag9.json", "r") as f:
        data_list = json.load(f)["results"]

    print(get_hard_examples(data_list))