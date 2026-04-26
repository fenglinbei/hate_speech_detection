import json
import math
from collections import defaultdict
from typing import Any, Optional
from transformers import AutoTokenizer, BertTokenizerFast # type: ignore

def run(input_file_path: str, output_file_path: str, tokenizer: Optional[AutoTokenizer | BertTokenizerFast] = None):

    def get_text_len(text: str) -> int:
        if isinstance(tokenizer, AutoTokenizer | BertTokenizerFast):
            tokens: list = tokenizer(text)["input_ids"] # type: ignore
            return len(tokens)
        else:
            return len(text)

    with open(file=input_file_path, mode="r") as input_file:
        input_datas: list[dict[str, Any]] = json.load(fp=input_file)


    # 初始化统计容器
    stats = {
        'total_samples': 0,
        'text_length': {
            'all': [],
            'hate': [],
            'non_hate': []
        },
        'label_distribution': {
            'targeted_group': defaultdict(int),
            'hateful': defaultdict(int)
        },
        'quadruple_stats': {
            'per_sample': [],
            'hate_per_sample': [],
            'non_hate_per_sample': []
        },
        'cross_analysis': defaultdict(lambda: defaultdict(int)),
        'complex_labels': defaultdict(int)
    }


    # 数据处理
    for sample in input_datas:
        # 基础统计
        stats['total_samples'] += 1
        text_len = get_text_len(sample['content'])
        stats['text_length']['all'].append(text_len)
        
        # 四元组分析
        quad_count = len(sample['quadruples'])
        stats['quadruple_stats']['per_sample'].append(quad_count)
        
        hate_count = 0
        non_hate_count = 0
        
        for quad in sample['quadruples']:
            # 标签分布
            groups = [g.strip() for g in quad['targeted_group'].split(',') if g.strip()]
            for group in groups:
                if group != 'non-hate':
                    stats['label_distribution']['targeted_group'][group] += 1
                    stats['cross_analysis'][group][quad['hateful']] += 1
            
            # 仇恨标签统计
            stats['label_distribution']['hateful'][quad['hateful']] += 1
            
            # 复合标签统计
            if len(groups) > 1:
                stats['complex_labels'][','.join(sorted(groups))] += 1
            
            # 仇恨相关统计
            if quad['hateful'] == 'hate':
                hate_count += 1
                stats['text_length']['hate'].append(text_len)
            else:
                non_hate_count += 1
                stats['text_length']['non_hate'].append(text_len)
        
        # 四元组数量统计
        stats['quadruple_stats']['hate_per_sample'].append(hate_count)
        stats['quadruple_stats']['non_hate_per_sample'].append(non_hate_count)

    # 统计函数
    def calculate_stats(values):
        if not values:
            return {}
        return {
            'max': max(values),
            'min': min(values),
            'mean': round(sum(values)/len(values), 2),
            'median': sorted(values)[len(values)//2],
            'std_dev': round(math.sqrt(sum((x - sum(values)/len(values))**2 for x in values)/(len(values)-1)), 2) if len(values)>1 else 0
        }

    # 生成最终报告
    report = {
        # 基础统计
        'total_samples': stats['total_samples'],
        'total_quadruples': sum(stats['quadruple_stats']['per_sample']),
        
        # 文本长度分析
        'text_length_stats': {
            'tokenizer': tokenizer.name_or_path if isinstance(tokenizer, AutoTokenizer | BertTokenizerFast) else None,  # type: ignore
            'all': calculate_stats(stats['text_length']['all']),
            'hate': calculate_stats(stats['text_length']['hate']),
            'non_hate': calculate_stats(stats['text_length']['non_hate'])
        },
        
        # 标签分布
        'target_groups': dict(stats['label_distribution']['targeted_group']),
        'hate_ratio': {
            'hate': stats['label_distribution']['hateful']['hate'],
            'non_hate': stats['label_distribution']['hateful']['non-hate']
        },
        
        # 四元组统计
        'quadruple_distribution': {
            'per_sample': calculate_stats(stats['quadruple_stats']['per_sample']),
            'hate_per_sample': calculate_stats(stats['quadruple_stats']['hate_per_sample']),
            'non_hate_per_sample': calculate_stats(stats['quadruple_stats']['non_hate_per_sample'])
        },
        
        # 交叉分析
        'hate_by_group': {k: dict(v) for k, v in stats['cross_analysis'].items()},
        
        # 复杂标签
        'multi_category_labels': dict(stats['complex_labels'])
    }
    
    with open(output_file_path, "w") as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./models/Erlangshen-Roberta-330M-NLI")
    # run("data/full/std/train.json", "data/full/stats/train.json", tokenizer=tokenizer)
    run("data/full/std/test.json", "data/full/stats/test.json", tokenizer=tokenizer)

    # run("data/full/std/train.json", "data/full/stats/train.json")