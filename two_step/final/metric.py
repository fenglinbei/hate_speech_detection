import os
import sys
import json

from loguru import logger

sys.path.append(".")

from metrics.core import get_score

class LLMmetrics:

    def __init__(self, pred_data_path: str, gt_data_path: str, output_dir: str = "./metrics/llm/"):
        self.pred_data_path = pred_data_path
        self.gt_data_path = gt_data_path
        self.output_dir = output_dir

        self.total_f1_hard = 0
        self.total_f1_soft = 0
        self.avg_f1 = 0

    def _load_data(self):
        logger.info("正在读取数据...")
        with open(self.pred_data_path, 'r') as f:
            pred_datas = json.load(f)

        with open(self.gt_data_path, 'r') as f:
            gt_datas = json.load(f)
        

        pred_results = pred_datas["results"]
        self.info = pred_datas["info"]

        self.metric_datas = []

        gt_data_dict = {}
        for gt_data in gt_datas:
            gt_data_dict[str(gt_data["id"])] = gt_data["quadruples"]
        
        pred_data_dict = {}
        for pred_data in pred_results:
            content_id = str(pred_data["id"])
            if content_id not in pred_data_dict:
                pred_data_dict[content_id] = [
                    {
                        "target": pred_data.get("pred_target", pred_data.get("target")),
                        "argument": pred_data.get("pred_argument", pred_data.get("argument")),
                        "targeted_group": pred_data.get("pred_targeted_group", pred_data.get("targeted_group")),
                        "hateful": pred_data.get("pred_hateful", pred_data.get("hateful"))
                    }
                ]
            
            else:
                pred_data_dict[content_id].append(
                    {
                        "target": pred_data.get("pred_target", pred_data.get("target")),
                        "argument": pred_data.get("pred_argument", pred_data.get("argument")),
                        "targeted_group": pred_data.get("pred_targeted_group", pred_data.get("targeted_group")),
                        "hateful": pred_data.get("pred_hateful", pred_data.get("hateful"))
                    }
                )


        for content_id in gt_data_dict:
            
            if content_id not in pred_data_dict:
                pred_list = []
            else:
                pred_list = pred_data_dict[content_id]

            gold_list = gt_data_dict[content_id]

            self.metric_datas.append((pred_list, gold_list))

        self.data_len = len(self.metric_datas)
    
    def parse_llm_output(self, llm_output: str):
        quadruples = []
        lines = llm_output.strip().split('\n')
        for line in lines:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) != 4:
                continue
            # 处理 "NULL" 值
            target = parts[0] if parts[0].upper() != 'NULL' else None
            argument = parts[1] if parts[1].upper() != 'NULL' else None
            targeted_group = parts[2] if parts[2].upper() != 'NULL' else None
            hateful = parts[3] if parts[3].upper() != 'NULL' else None
            quadruples.append({
                'target': target,
                'argument': argument,
                'targeted_group': targeted_group,
                'hateful': hateful
            })
        return quadruples

    def _calculate_score(self, pred_list: list[dict[str, str]], gold_list: list[dict[str, str]]):

        f1_hard, f1_soft = get_score(pred_list, gold_list)

        self.total_f1_hard += f1_hard
        self.total_f1_soft += f1_soft
    
    def _save_result(self):
        logger.info("正在保存数据...")
        os.makedirs(self.output_dir, exist_ok=True)

        score_dict = {"f1_hard": self.f1_hard,
                      "f1_soft": self.f1_soft,
                      "f1_avg": self.avg_f1}
        
        if self.info:
            model_name = self.info["model"]
            shot_num = self.info["shot_num"]
            seed = self.info["seed"]
            file_path = os.path.join(self.output_dir, f"metric_{model_name}_{shot_num}_{seed}.json")
        else:
            file_path = os.path.join(self.output_dir, f"metric.json")

        with open(file_path, 'w') as f:
            f.write(json.dumps({"info": self.info, "metrics": score_dict}, ensure_ascii=False, indent=2))
        
        logger.info("数据保存完毕")

    def run(self):
        self._load_data()
        try:
            for idx, (pred_list, gold_list) in enumerate(self.metric_datas, start=1):
                self._calculate_score(pred_list, gold_list)

                if idx % 10 == 0:
                    logger.info(
                                f"进度: {idx}/{self.data_len} "
                                f"({idx/self.data_len:.1%})"
                            )
        except Exception as e:
            print((pred_list, gold_list))
            logger.exception(e)
            logger.error(f"运行时异常: {str(e)}", exc_info=True)
            exit()
        
        self.f1_hard = self.total_f1_hard / self.data_len
        self.f1_soft = self.total_f1_soft / self.data_len
        self.avg_f1 = (self.total_f1_hard + self.total_f1_soft) / 2 / self.data_len

        self._save_result()

if __name__ == "__main__":
    METRIC = LLMmetrics(pred_data_path="/mnt/i/project/hateSpeechDetection/two_step/step2/result/output_qwen2.5-7b-instruct-ft-202504240006-d921_0_23333333_20250424_112230.json",
                        gt_data_path="/mnt/i/project/hateSpeechDetection/data/temp_test_data.json",
                        output_dir="/mnt/i/project/hateSpeechDetection/two_step/final/")
    METRIC.run()
        
    
    

