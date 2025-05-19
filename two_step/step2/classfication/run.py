import json
import torch
from loguru import logger
from transformers.pipelines import pipeline

is_hate_pipe = pipeline("text-classification", model="/workspace/two_step/step2/classfication/is_hate_model/checkpoint-3900", tokenizer="IDEA-CCNL/Erlangshen-TCBert-330M-Classification-Chinese")
hate_type_pipe = pipeline("text-classification", model="/workspace/two_step/step2/classfication/hate_type_model/checkpoint-3000", tokenizer="IDEA-CCNL/Erlangshen-TCBert-330M-Classification-Chinese")



is_hate_mapping = {
    'LABEL_1': 'non-hate',
    'LABEL_0': 'hate'
}

hate_type_mapping ={
    'LABEL_1': 'Racism',
    'LABEL_4': 'others',
    'LABEL_2': 'Region',
    'LABEL_0': 'LGBTQ',
    'LABEL_3': 'Sexism',
}

def unpack_step1_data(datas: dict, with_gt: bool=False) -> list[dict]:
    unpack_datas: list[dict] = []
    for data in datas["results"]:

        if not with_gt:
            pred_quadruples = data["parsed_quadruples"]
            for pred_quadruple in pred_quadruples:
                unpack_datas.append(
                    {
                        "id": data["id"],
                        "content": data["content"],
                        "pred_target": pred_quadruple["target"],
                        "pred_argument": pred_quadruple["argument"],
                    }
                )


    return unpack_datas

def unpack_data(datas: list[dict]) -> list[dict]:
    unpack_datas: list[dict] = []
    task_id = 1
    for data in datas:
        quadruples = data["quadruples"]
        for quadruple in quadruples:
            unpack_datas.append(
                {
                    "id": data["id"],
                    "task_id": task_id,
                    "content": data["content"],
                    "target": quadruple["target"],
                    "argument": quadruple["argument"],
                    "gt_targeted_group": quadruple["targeted_group"],
                    "gt_hateful": quadruple["hateful"]
                }
            )
            task_id += 1

    return unpack_datas

def process(test_data_path: str, output_path: str):
    with open(test_data_path, "r") as f:
        # test_datas = unpack_step1_data(json.load(f))
        test_datas = unpack_data(json.load(f))

    results: list[dict] = []

    for test_data in test_datas:
        text = test_data["content"]
        logger.info(f"正在处理文本: {text}")

        is_hate_output: dict = is_hate_pipe(text)
    
        is_hate_label = is_hate_mapping[is_hate_output[0]['label']]
        if is_hate_label == 'hate':
            hate_type_output: dict = hate_type_pipe(text)
            result_dict = {
                **test_data,
                'pred_targeted_group': hate_type_mapping[hate_type_output[0]['label']],
                'pred_hateful': is_hate_label,
                "status": "success"
            }
        else:
             result_dict = {
                **test_data,
                'pred_targeted_group': is_hate_label,
                'pred_hateful': is_hate_label,
                "status": "success"
            }
        
        # logger.debug(json.dumps(result_dict, ensure_ascii=False, indent=2))

        results.append(result_dict)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"info": None, "metric": None, "results": results}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process("/workspace/data/temp_test_data.json",
            "/workspace/two_step/step2/classfication/result/Erlangshen-Roberta-330M-NLI_step3900.json")

