import json

def cut_data():
    with open('finetune/data/lex_rag5/train.jsonl', 'r') as f:
        lines = f.readlines()
    datas = [json.loads(line) for line in lines]
    
    max_length = 2000
    over_idx = []
    for data in datas:
        if len(data['input']) > max_length:
            length = len(data['input'])
            idx = data['id']
            print(f'length: {length}, index: {idx}')
            over_idx.append(idx)
    
    print(f'over length: {len(over_idx)}')
    with open('finetune/data/lex_rag5/new_train.jsonl', 'w') as f:
        for data in datas:
            if data['id'] not in over_idx:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print('done')
    print(f'new data length: {len(datas) - len(over_idx)}')
    print(f'new data path: finetune/data/simlex5_rag5/new_train.json')
    print('check data done')

if __name__ == "__main__":
