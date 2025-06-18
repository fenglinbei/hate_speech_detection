import json

def convert(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        datas = json.load(f)

    with open(output_path, "w", encoding="utf-8") as file:
        for message in datas:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def load_json(path: str):
    return json.load(open(path))

def load_jsonline(path: str) -> list:
    return [json.loads(line) for line in open(path)]

def save_jsonline(datas: list, path: str):
    with open(path, 'w') as fw:
        for data in datas:
            print(json.dumps(data, ensure_ascii=False), file=fw)

def save_json(datas: list, path: str, indent: int = 2):
    with open(path, 'w') as fw:
        print(json.dumps(datas, ensure_ascii=False, indent=indent), file=fw)