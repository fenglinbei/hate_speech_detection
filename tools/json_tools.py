import json

def main(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        datas = json.load(f)

    with open(output_path, "w", encoding="utf-8") as file:
        for message in datas:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

main("finetune/data/train_rag_triple.json", "finetune/data/train_rag_triple_line.json")