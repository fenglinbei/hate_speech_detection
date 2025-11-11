import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
label_dict = {
    "Racism": 0,
    "Region": 1,
    "LGBTQ": 2,
    "Sexism": 3,
    "others": 4,
    "non-hate": 5
}

binary_label_dict = {
    "hate": 0,
    "non-hate": 1
}

label_dict_inv = {v: k for k, v in label_dict.items()}
binary_label_dict_inv = {v: k for k, v in binary_label_dict.items()}


def get_data(data_path):
    texts = []
    datas = json.load(open(data_path, "r", encoding="utf-8"))
    labels = []
    for data in datas:
        text = data['content']
        texts.append(text)
        quadruples: list[dict[str, str]] = data['quadruples']
        all_target_groups = set()
        for quadruple in quadruples:
            target, argument, target_groups, is_hate = quadruple.values()
            target_groups = [i.strip() for i in target_groups.split(',')]
            all_target_groups.update(target_groups)
        
        if 'non-hate' not in all_target_groups:
            text_label = all_target_groups.pop()
        else:
            text_label = 'non-hate'
            for tg in all_target_groups:
                if tg != 'non-hate':
                    text_label = tg
                    break
        
        labels.append(label_dict[text_label])
    return texts, labels

def get_data_with_binary_label(data_path):
    texts = []
    datas = json.load(open(data_path, "r", encoding="utf-8"))
    labels = []
    for data in datas:
        text = data['content']
        texts.append(text)
        quadruples: list[dict[str, str]] = data['quadruples']
        
        for quadruple in quadruples:
            target, argument, target_groups, is_hate = quadruple.values()
            if is_hate == 'hate':
                labels.append(binary_label_dict['hate'])
                break
        else:
            labels.append(binary_label_dict['non-hate'])
    return texts, labels