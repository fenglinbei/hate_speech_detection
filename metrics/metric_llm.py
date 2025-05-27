import os
import json
import time

from tqdm import tqdm
from loguru import logger
from typing import Optional, Tuple

from core import get_score

class LLMmetrics:

    def __init__(self, data_path: str, output_dir: str = "./metrics/llm/"):
        self.data_path = data_path
        self.output_dir = output_dir

        self.init_metric()

    def init_metric(self):
        self.total_f1_hard = 0
        self.total_f1_soft = 0
        self.avg_f1 = 0
        self.success = 0
        self.total = 0

    def _load_data_from_path(self, data_path: str) -> Tuple[list, list]:
        logger.info("正在读取数据...")
        with open(data_path, 'r') as f:
            json_datas = json.load(f)
        
        datas = json_datas["results"]
        return self._load_data(datas)

    def _load_data_from_dict(self, data_list: list[dict]) -> Tuple[list, list]:
        logger.info("正在读取数据...")
        return self._load_data(data_list)

    def _load_data(self, datas: list[dict]) -> Tuple[list, list]:
        pred_list, gold_list = [], []
        for result in datas:
            self.total += 1
            if result["status"] != "success":
                continue
            
            self.success += 1
            gold_quads = result["quadruples"]
            gold_list = [
                {
                    "target": q["target"],
                    "argument": q["argument"],
                    "targeted_group": q["targeted_group"],
                    "hateful": q["hateful"]
                }
                for q in gold_quads
            ]
            # pred_list = self.parse_llm_output(result["llm_output"])
            pred_list = result["parsed_quadruples"]

        return pred_list, gold_list


    
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
    
    def _save_result(self, info_data: dict,score_dict: dict):

        logger.info("正在保存数据...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        model_name = info_data["model"]
        shot_num = info_data["shot_num"]
        seed = info_data["seed"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.output_dir, f"metric_{model_name}_{shot_num}_{seed}_{timestamp}.json")

        with open(file_path, 'w') as f:
            f.write(json.dumps({"info": info_data, "metrics": score_dict}, ensure_ascii=False, indent=2))
        
        logger.info("数据保存完毕")

    def run(
            self, 
            datas_list: Optional[list[dict]] = None, 
            data_path: Optional[str] = None,
            info_data: Optional[dict] = None,
            save_data: bool = False
            ) -> Optional[dict]:
        
        self.init_metric()

        if isinstance(datas_list, list):
            metric_datas = self._load_data_from_dict(datas_list)
        elif isinstance(data_path, str):
            metric_datas = self._load_data_from_path(data_path)
        else:
            raise ValueError(f"Invaild Input, datas_list: {type(datas_list)} expected: list[dict], data_path: {type(data_path)} expected: str")
        
        try:
            data_len = len(metric_datas[0])
            for idx, (pred_list, gold_list) in enumerate(metric_datas, start=1):
                self._calculate_score(pred_list, gold_list)

                if idx % 10 == 0:
                    logger.info(
                                f"进度: {idx}/{data_len} "
                                f"({idx/data_len:.1%})"
                            )
        except Exception as e:
            logger.exception(e)
            logger.error(f"运行时异常: {str(e)}", exc_info=True)
            exit()
        
        self.f1_hard = self.total_f1_hard / data_len
        self.f1_soft = self.total_f1_soft / data_len
        self.avg_f1 = (self.total_f1_hard + self.total_f1_soft) / 2 / data_len

        score_dict = {"f1_hard": self.f1_hard,
                      "f1_soft": self.f1_soft,
                      "f1_avg": self.avg_f1,
                      "success": self.success,
                      "total": self.total,
                      "success_rate": self.success / self.total}

        if save_data:
            if isinstance(info_data, dict):
                self._save_result(info_data, score_dict)
            else:
                raise ValueError(f"Invaild Input, info_data: {type(info_data)} expected: dict")
        else:
            return score_dict

if __name__ == "__main__":
    METRIC = LLMmetrics(data_path="few_shot/output/output_qwen3-8b_0_23333333.json")
    METRIC.run()
        
    
    

