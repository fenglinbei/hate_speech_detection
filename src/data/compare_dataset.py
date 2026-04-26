import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from collections import defaultdict

# 准备两组示例数据
group1 = json.load(open("data/mav/train.json", "r", encoding="utf-8")) # 第一组JSON数据（从问题中复制完整数据）
group2 = json.load(open("data/full/std/train.json", "r", encoding="utf-8")) # 第二组JSON数据（从问题中复制完整数据）

# 函数解析第一组数据的标签
def parse_group1(item):
    parts = item['output'].split('|')
    hate_label = parts[-1].strip().split()[0]
    targeted_group = parts[-2].strip()
    return {
        "hate_label": hate_label,
        "targeted_group": targeted_group
    }

# 函数解析第二组数据的标签
def parse_group2(item):
    results = []
    for q in item['quadruples']:
        results.append({
            "hate_label": q['hateful'],
            "targeted_group": q['targeted_group']
        })
    return results

# 数据处理
group1_data = []
group1_contents = set()
for item in group1:
    group1_contents.add(item['content'])
    parsed = parse_group1(item)
    parsed['content'] = item['content']
    group1_data.append(parsed)

group2_data = []
group2_contents = set()
for item in group2:
    group2_contents.add(item['content'])
    parsed_list = parse_group2(item)
    for parsed in parsed_list:
        parsed['content'] = item['content']
        group2_data.append(parsed)

# 1. 基本统计数据
group1_count = len(group1_data)
group2_count = len(group2_data)
common_content_count = len(group1_contents & group2_contents)

# 2. 仇恨言论分布
group1_hate_count = sum(1 for item in group1_data if item['hate_label'] == 'hate')
group1_non_hate_count = group1_count - group1_hate_count
group2_hate_count = sum(1 for item in group2_data if item['hate_label'] == 'hate')
group2_non_hate_count = group2_count - group2_hate_count

# 3. 五种仇恨类型标签分布
hate_categories = ['Racism', 'Sexism', 'Region', 'LGBTQ', 'others', 'non-hate']

def get_hate_category(item):
    if item['hate_label'] == 'non-hate':
        return 'non-hate'
    targeted_group = item['targeted_group']
    if targeted_group in ['Racism', 'Sexism', 'Region', 'LGBTQ', 'others']:
        return targeted_group
    return 'others'  # 其他类型的仇恨归为'others'

# 统计组1的仇恨标签分布
group1_categories = defaultdict(int)
for item in group1_data:
    category = get_hate_category(item)
    group1_categories[category] += 1

# 统计组2的仇恨标签分布
group2_categories = defaultdict(int)
for item in group2_data:
    category = get_hate_category(item)
    group2_categories[category] += 1

# 确保所有分类都被统计，包括计数为0的类别
for cat in hate_categories:
    if cat not in group1_categories:
        group1_categories[cat] = 0
    if cat not in group2_categories:
        group2_categories[cat] = 0

# 按预定顺序排序
group1_counts = [group1_categories[cat] for cat in hate_categories]
group2_counts = [group2_categories[cat] for cat in hate_categories]

# 创建图表
plt.figure(figsize=(18, 12))
plt.suptitle('Hate Speech Dataset Comparison', fontsize=16, fontweight='bold')

# 图表1：数据量对比
plt.subplot(2, 3, 1)
plt.bar(['Group 1', 'Group 2'], [group1_count, group2_count], 
        color=['skyblue', 'lightgreen'], width=0.6)
plt.title('Total Data Items', fontsize=12, fontweight='bold')
plt.ylabel('Count')
for i, v in enumerate([group1_count, group2_count]):
    plt.text(i, v + max(group1_count, group2_count)*0.05, 
             str(v), ha='center', fontweight='bold')

# 图表2：仇恨言论占比（组1）
plt.subplot(2, 3, 2)
sizes1 = [group1_hate_count, group1_non_hate_count]
labels = ['Hate', 'Non-Hate']
colors = ['#ff9999','#66b3ff']
plt.pie(sizes1, labels=labels, autopct='%1.1f%%', startangle=90, 
        colors=colors, shadow=True, explode=(0.05, 0))
plt.title('Group 1: Hate vs Non-Hate', fontsize=12, fontweight='bold')
plt.axis('equal')

# 图表3：仇恨言论占比（组2）
plt.subplot(2, 3, 3)
sizes2 = [group2_hate_count, group2_non_hate_count]
plt.pie(sizes2, labels=labels, autopct='%1.1f%%', startangle=90, 
        colors=colors, shadow=True, explode=(0.05, 0))
plt.title('Group 2: Hate vs Non-Hate', fontsize=12, fontweight='bold')
plt.axis('equal')

# 图表4：仇恨类型分布（组1） - 饼图
plt.subplot(2, 3, 4)
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
explode = [0.05] * len(hate_categories)  # 突出所有部分
plt.pie(group1_counts, labels=hate_categories, autopct='%1.1f%%', 
        startangle=140, colors=colors, explode=explode, 
        shadow=True, textprops={'fontsize': 9})
plt.title('Group 1: Hate Category Distribution', fontsize=12, fontweight='bold')
plt.axis('equal')

# 图表5：仇恨类型分布（组2） - 饼图
plt.subplot(2, 3, 5)
plt.pie(group2_counts, labels=hate_categories, autopct='%1.1f%%', 
        startangle=140, colors=colors, explode=explode, 
        shadow=True, textprops={'fontsize': 9})
plt.title('Group 2: Hate Category Distribution', fontsize=12, fontweight='bold')
plt.axis('equal')

# 图表6：文本内容重复关系
plt.subplot(2, 3, 6)
venn = venn2([group1_contents, group2_contents], 
             set_labels=('Group 1', 'Group 2'), 
             set_colors=('#66b3ff', '#99ff99'),
             alpha=0.7)
plt.title('Content Overlap', fontsize=12, fontweight='bold')

# 添加文本标签
for text in venn.set_labels:
    text.set_fontsize(10)
    text.set_fontweight('bold')
for text in venn.subset_labels:
    if text:
        text.set_fontsize(12)
        text.set_fontweight('bold')

# 添加自定义标签显示数量
plt.text(0.45, -0.2, f'Common Texts: {common_content_count}', 
         ha='center', fontsize=10, fontweight='bold')

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# 保存图片
plt.savefig('data/hate_speech_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("可视化图表已保存为 hate_speech_comparison.png")