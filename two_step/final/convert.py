import json

def convert_to_raw_txt(step2_output_json_data_path: str, raw_test_data: str):
    # 读取原始JSON数据

    with open(raw_test_data, "r") as f:
        raw_test_datas = json.load(f)

    with open(step2_output_json_data_path, "r", encoding="utf-8") as f:
        step2_output_data = json.load(f)

    step2_results: list[dict] = step2_output_data["results"]
    results_dict: dict[str, list[str]] = {}

    output_str_list: list[str] = []

    for step2_result in step2_results:

        if str(step2_result['id']) not in results_dict:
                results_dict[str(step2_result['id'])] = []

        if step2_result["status"] != "success":
            continue
        
        results_dict[str(step2_result['id'])].append(f"{step2_result['pred_target']} | {step2_result['pred_argument']} | {step2_result['pred_targeted_group']} | {step2_result['pred_hateful']}")

    
    # print(results_dict)

    for raw_test_data in raw_test_datas:
        content_id = str(raw_test_data['id'])
        print(content_id)
        if content_id not in results_dict:
            output_str_list.append("[END]")
        else:
            output_str_list.append(" [SEP] ".join(results_dict[content_id]) + " [END]")
    
    with open("llm_mix_output_sep.txt", "w") as f:
        f.write("\n".join(output_str_list))

if __name__ == "__main__":
    convert_to_raw_txt(
        step2_output_json_data_path="/mnt/i/project/hateSpeechDetection/two_step/step2/result/output_qwen2.5-7b-instruct-ft-202504251158-c4e8_0_23333333_20250425_170543.json",
        raw_test_data="/mnt/i/project/hateSpeechDetection/data/test_data.json")
    