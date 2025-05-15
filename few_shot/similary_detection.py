import json

with open("./few_shot/progress.json", 'r') as f:
    datas = json.load(f)

processed_ids = set()
results = []

for result in datas["results"]:
    if result["id"] in processed_ids:
        continue

    processed_ids.add(result["id"])
    results.append(result)

datas["processed_ids"] = list(processed_ids)
datas["results"] = results

with open("./few_shot/progress_new.json", 'w', encoding='utf-8') as f:
    json.dump(datas, f, ensure_ascii=False, indent=2)