import json

def rebuild_status(result_file: str):
    with open(result_file, "r") as f:
        datas = json.load(f)

    new_results = []
    for result in datas["results"]:
        if result["pred_quadruples"]:
            result["status"] = "success"
            result["attempts"] = 1
            if result.get("error"):
                result.pop("error")
        
        new_results.append(result)

    datas["results"] = new_results
    with open(result_file, "w") as f:
        json.dump(datas, f, indent=2, ensure_ascii=False)

def rebuild_label(result_file: str):
    with open(result_file, "r") as f:
        datas = json.load(f)

    new_results = []
    for result in datas["results"]:
        if result["pred_quadruples"]:
            for pred_quadruple in result["pred_quadruples"]:
                if pred_quadruple["targeted_group"] == "non-hate":
                    pred_quadruple["hateful"] = "non-hate"
        
        new_results.append(result)

    datas["results"] = new_results
    with open(result_file, "w") as f:
        json.dump(datas, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    rebuild_status("few_shot/output/output_Qwen3-8B-sft-hsd-v4-cosine-default_shots0_seed23333333.json")