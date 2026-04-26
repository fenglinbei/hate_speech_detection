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

def get_hard_examples(data_list: list[dict], alpha: float = 0.5, beta: float = 0.5, threshold: float = 0.5) -> list[tuple[dict, float]]:
    pred_data_dict, gt_data_dict = METRIC._load_data_from_dict(data_list)
    sorted_hard_example_ids, match_scores, label_scores = sort_by_hard_example(list(gt_data_dict.keys()), pred_data_dict, gt_data_dict, alpha=0.5, beta=0.5)
    return [(data_list[i], alpha *  match_scores[i] + beta * label_scores[i]) for i in sorted_hard_example_ids if alpha *  match_scores[i] + beta * label_scores[i] < threshold]

def draw_all_plots(targeted_group_results, hateful_results, length_results, save_path="analyse/all_analysis.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    # 创建2行3列的布局
    fig = plt.figure(figsize=(18, 12), dpi=100)
    gs = GridSpec(2, 3, figure=fig)  # 2行3列
    
    # 第一行：三个饼图
    pie_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2])
    ]
    
    # 第二行：三个柱状图
    bar_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2])
    ]
    
    # 绘制饼图（添加引线）
    def draw_pie_on_ax(data, title, ax):
        labels = list(data.keys())
        sizes = [i["total"] for i in data.values()]
        total = sum(sizes)
        percentages = [size/total*100 for size in sizes]
        
        # 设置阈值，小于此值的部分需要引线
        threshold = 5.0  # 小于5%的部分使用引线
        
        # 计算explode值，小部分稍微突出
        explode = [0.05 if p < threshold else 0 for p in percentages]
        
        # 绘制饼图
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=None,  # 不显示默认标签
            autopct=lambda p: f'{p:.1f}%' if p >= threshold else '',  # 只显示大块的百分比
            startangle=140, 
            shadow=True, 
            explode=explode,
            pctdistance=0.85  # 百分比位置
        )
        
        # 设置百分比文本属性
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')
        
        # 添加带引线的标签
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.72, alpha=0.8)
        kw = dict(arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
                  bbox=bbox_props, zorder=0, va="center")
        
        x_pos = [0, 0, 1.2, 1.2, 1.2, 1.2, 1.2]
        y_pos = [0, 0, 1.565, 1.64, 1.58, 1.46, 1.325]
        for i, p in enumerate(percentages):
            if p < threshold:  # 只对小部分添加引线
                ang = (wedges[i].theta2 - wedges[i].theta1)/2. + wedges[i].theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                
                # 根据位置调整水平对齐方式
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                
                # 调整连接点位置
                connectionstyle = f"angle,angleA=0,angleB={ang}"
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                
                # 添加带引线的标签
                print(i, kw)
                ax.annotate(f"{labels[i]}: {p:.1f}%", 
                            xy=(x, y), 
                            xytext=(x_pos[i]*np.sign(x), y_pos[i]*y),
                            horizontalalignment=horizontalalignment,
                            fontsize=9,
                            **kw)
            else:
                # 对大部分添加普通标签
                ang = (wedges[i].theta2 - wedges[i].theta1)/2. + wedges[i].theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                
                # 调整标签位置避免重叠
                label_radius = 1.1
                if i == 1 and labels[i] == "50-100":
                    label_radius = 1.05
                ax.text(label_radius*x, label_radius*y, labels[i], 
                        ha='center', va='center', fontsize=9)
        
        ax.set_title(title, fontsize=14)
        ax.axis('equal')
    
    # 绘制柱状图（保持不变）
    def draw_bar_on_ax(data, title, ax):
        labels = data.keys()
        metrics = ["F1 Hard", "F1 Soft", "F1 Avg"]
        values = []
        for label in labels:
            entry = data[label]
            values.append([entry["f1_hard"], entry["f1_soft"], entry["f1_avg"]])
        
        values = np.array(values)
        x = np.arange(len(labels))
        width = 0.2
        
        rects1 = ax.bar(x - width, values[:, 0], width, label=metrics[0], color='#4C72B0')
        rects2 = ax.bar(x, values[:, 1], width, label=metrics[1], color='#55A868')
        rects3 = ax.bar(x + width, values[:, 2], width, label=metrics[2], color='#C44E52')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Label', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', alpha=0.4, linestyle='--')
        
        # 添加数据标签
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        # add_labels(rects1)
        # add_labels(rects2)
        # add_labels(rects3)
    
    # 绘制三个饼图
    draw_pie_on_ax(targeted_group_results, "Targeted Group Distribution", pie_axes[0])
    draw_pie_on_ax(hateful_results, "Hateful Distribution", pie_axes[1])
    draw_pie_on_ax(length_results, "Length Distribution", pie_axes[2])
    
    # 绘制三个柱状图
    draw_bar_on_ax(targeted_group_results, "Targeted Group Performance", bar_axes[0])
    draw_bar_on_ax(hateful_results, "Hateful Performance", bar_axes[1])
    draw_bar_on_ax(length_results, "Length Performance", bar_axes[2])
    
    # 添加大标题
    fig.suptitle("Comprehensive Analysis Results", fontsize=20, fontweight='bold')
    
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
    plt.savefig(save_path)
    plt.close()

def get_results(data_list):
    import json
    targeted_group_results = analyse_target_group(data_list)
    hateful_results = analyse_hateful(data_list)
    length_results = analyse_length(data_list)
    
    # 使用新的绘图函数
    draw_all_plots(targeted_group_results, hateful_results, length_results)
    
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

    print(json.dumps(get_hard_examples(data_list), indent=2, ensure_ascii=False))
    # get_results(data_list)