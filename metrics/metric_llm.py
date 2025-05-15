import os
import json

from loguru import logger

from core import get_score

class LLMmetrics:

    def __init__(self, data_path: str, output_dir: str = "./metrics/llm/"):
        self.data_path = data_path
        self.output_dir = output_dir

        self.total_f1_hard = 0
        self.total_f1_soft = 0
        self.avg_f1 = 0
        self.success = 0
        self.total = 0

    def _load_data(self):
        logger.info("正在读取数据...")
        with open(self.data_path, 'r') as f:
            datas = json.load(f)
        
        results = datas["results"]
        self.info = datas["info"]

        self.metric_datas = []

        for result in results:
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
                      "f1_avg": self.avg_f1,
                      "success": self.success,
                      "total": self.total,
                      "success_rate": self.success / self.total}
        
        model_name = self.info["model"]
        shot_num = self.info["shot_num"]
        seed = self.info["seed"]
        file_path = os.path.join(self.output_dir, f"metric_{model_name}_{shot_num}_{seed}.json")

        with open(file_path, 'w') as f:
            f.write(json.dumps({"info": self.info, "metrics": score_dict}, ensure_ascii=False, indent=2))
        
        logger.info("数据保存完毕")

    def run(self):
        self._load_data()
        try:
            pred_list, gold_list = [], []
            for idx, (pred_list, gold_list) in enumerate(self.metric_datas, start=1):
                self._calculate_score(pred_list, gold_list)

                if idx % 10 == 0:
                    logger.info(
                                f"进度: {idx}/{self.data_len} "
                                f"({idx/self.data_len:.1%})"
                            )
        except Exception as e:
            logger.exception(e)
            logger.error(f"运行时异常: {str(e)}", exc_info=True)
            exit()
        
        self.f1_hard = self.total_f1_hard / self.data_len
        self.f1_soft = self.total_f1_soft / self.data_len
        self.avg_f1 = (self.total_f1_hard + self.total_f1_soft) / 2 / self.data_len

        self._save_result()

if __name__ == "__main__":
    METRIC = LLMmetrics(data_path="/workspace/few_shot/output/output_qwen2.5-7b-instruct-ft-202504180934-6766_10_23333333.json")
    METRIC.run()
        
    
    

