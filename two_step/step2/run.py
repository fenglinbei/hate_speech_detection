import concurrent.futures
import os
import sys
import json
import time
import glob
import signal
import random
import requests
import threading
import concurrent
from queue import Queue
from loguru import logger
from typing import Optional, List, Dict, Tuple, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(".")

from metrics.metric_llm import LLMmetrics
from prompt import *
from api.llm import AliyunApiLLMModel
from utils.protocol import UsageInfo

category_mapping = {
    'Racism': 'A',
    'Region': 'B',
    'LGBTQ': 'C',
    'Sexism': 'D',
    'others': 'E',
    'non-hate': 'F'}
hate_flag_mapping = { 
    'hate': 'A',
    'non-hate': 'B'
}

swapped_category_mapping = {v: k for k, v in category_mapping.items()}
swapped_hate_flag_mapping = {v: k for k, v in hate_flag_mapping.items()}


def build_two_step_shot_prompt(
        shots: list[dict],
        prompt_template: str,
        shot_prompt_template: str) -> str:
    """
    构建prompt

    shots: 随机挑选的测试样本
    """

    examples = []
    for shot in shots:
        # 生成答案部分
        category = category_mapping[shot.get("gt_targeted_group", "non-hate").split(", ")[0]]
        hate_flag = hate_flag_mapping[shot.get("gt_hateful", "non-hate")]
        target = shot.get("target", "NULL")
        argument = shot.get("argument", "NULL")
        answer = f"{hate_flag} | {category}"
        
        # 构建单个示例
        examples.append(
            shot_prompt_template.format(
                text=shot["content"],
                target=target,
                argument=argument,
                answer=answer
            )
        )
    
    # 组合所有示例并生成包含示例的prompt模板
    return prompt_template.format(
        text = "{text}",
        target = "{target}",
        argument = "{argument}",
        shots="\n\n".join(examples) + "\n\n"
    )

def get_shots(shot_datas: list[dict], shot_num: int, seed: int = 23333333) -> list[dict]:
    """
    随机采样指定数量的样本
    
    json_data_path: 解析后的json格式测试数据路径
    shot_num: 例子数量
    seed: 随机数种子，用于复现
    """

    random.seed(seed)
    return random.sample(shot_datas, k=min(shot_num, len(shot_datas)))

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

def unpack_step1_data(datas: dict, with_gt: bool=False) -> list[dict]:
    unpack_datas: list[dict] = []
    task_id = 1
    for data in datas["results"]:

        if not with_gt:
            pred_quadruples = data["parsed_quadruples"]
            for pred_quadruple in pred_quadruples:
                unpack_datas.append(
                    {
                        "id": data["id"],
                        "task_id": task_id,
                        "content": data["content"],
                        "pred_target": pred_quadruple["target"],
                        "pred_argument": pred_quadruple["argument"],
                    }
                )
                task_id += 1


    return unpack_datas

# def unpack_data(datas: list[dict]) -> list[dict]:
#     unpack_datas: list[dict] = []
#     task_id = 1
#     for data in datas:
#         gt_quadruples = data["quadruples"]
#         for gt_quadruple in gt_quadruples:
#             unpack_datas.append(
#                 {
#                     "id": data["id"],
#                     "task_id": task_id,
#                     "content": data["content"],
#                     "gt_target": gt_quadruple["target"],
#                     "gt_argument": gt_quadruple["argument"],
#                 }
#             )
#             task_id += 1

#     return unpack_datas


class TwoStepLLMTester:
    def __init__(
        self,
        llm_model: AliyunApiLLMModel,
        output_dir: str,
        shot_dataset_file: Optional[str] = None,
        test_dataset_file: Optional[str] = None,
        shot_num: int = 0,
        seed: int = 23333333,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        concurrency: int = 1,
        max_retries: int = 5,
        base_retry_wait_time: int = 0,
        request_timeout: float = 45.0,
        checkpoint_interval: int = 10,
        progress_dir: str = "./two_step/step2/progress",
        max_progress_files: int = 5,
        input_file: Optional[str] = None,
        step1_result_file: Optional[str] = None
    ):
        """
        FewShot测试执行器

        Args:
            llm_model: 配置好的LLM模型实例
            shot_dataset_file: 例子抽样数据集,
            test_dataset_file: 测试数据集,
            shot_num: 例子抽样个数,
            seed: 随机抽样种子，用于复现结果,
            output_dir: 输出文件保存路径,
            max_tokens: 生成文本的最大token数
            temperature: 采样温度（None表示使用模型默认值）
            concurrency: 并发请求数
            max_retries: 最大重试次数
            request_timeout: 单请求超时时间(秒)
            checkpoint_interval: 检查点保存间隔(处理数量)
            progress_file: 进度文件路径
            input_file: 已经处理好的prompts数据文件（可选）,
        """
        self.llm = llm_model
        self.shot_dataset_file = shot_dataset_file
        self.test_dataset_file = test_dataset_file
        self.step1_result_file = step1_result_file
        self.shot_num = shot_num
        self.seed = seed

        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.base_retry_wait_time = base_retry_wait_time
        self.request_timeout = request_timeout
        self.checkpoint_interval = checkpoint_interval
        self.progress_dir = progress_dir
        self.max_progress_files = max_progress_files
        os.makedirs(self.progress_dir, exist_ok=True)

        # 状态管理
        self.lock = threading.RLock()
        self.processed_ids: Set[int] = set()
        self.results: List[Dict] = []
        self.total_usage = UsageInfo()
        self.last_checkpoint = 0

        self._shutdown_flag = False
        self.executor = None

        # 注册信号处理
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)
    
    def _graceful_shutdown(self, signum, frame):
        """处理中断信号"""
        logger.warning(f"\n捕获信号 {signum}，开始退出...")
        self._shutdown_flag = True
        
        # 关闭执行器
        if self.executor:
            logger.info("正在终止线程池...")
            self.executor.shutdown(wait=True)

        # 保存进度
        with self.lock:
            self._save_progress(force=True)
        
        logger.info("退出进度已保存，可重新运行程序继续处理")

    def _create_datas(
            self,
            shot_data_path: Optional[str] = None, 
            test_data_path: Optional[str] = None, 
            step1_result_file: Optional[str] = None,
            shot_num: int = 5, 
            seed: int = 23333333,
            prompt_template: str = TRAIN_PROMPT_STEP_2_V1,
            shot_prompt_template: str = SHOT_PROMPT_STEP_2_V1,
            system_prompt: Optional[str] = None):
        
        if not step1_result_file:
            assert isinstance(shot_data_path, str) and isinstance(test_data_path, str), "数据类型错误"
        
            with open(shot_data_path, "r") as f:
                self.shot_datas = unpack_data(json.load(f))
            
            with open(test_data_path, "r") as f:
                self.test_datas = unpack_data(json.load(f))
        
        else:
            with open(step1_result_file, "r") as f:
                self.test_datas = unpack_step1_data(json.load(f))

        if system_prompt:
            prompt_template = prompt_template.replace("{system_prompt}", system_prompt)
        else:
            prompt_template = prompt_template.replace("{system_prompt}", "")

        if shot_num:
            sample_shots = get_shots(self.shot_datas, shot_num=shot_num, seed=seed)
            prompt_template = build_two_step_shot_prompt(
                sample_shots, 
                prompt_template, 
                shot_prompt_template)
        else:
            prompt_template = prompt_template.replace("{shots}", "")

        all_datas: list[dict] = []

        for data in self.test_datas:
            if step1_result_file:
                all_datas.append(
                    {   **data,
                        "prompt": prompt_template.format(
                            text=data["content"],
                            target=data["pred_target"],
                            argument=data['pred_argument'])})
            else:
                all_datas.append(
                    {   **data,
                        "prompt": prompt_template.format(
                            text=data["content"],
                            target=data["target"],
                            argument=data['argument'])})
        
        return all_datas

    def _load_progress(self) -> None:
        """加载最新进度文件"""
        try:
            # 获取所有符合命名规则的进度文件
            progress_files = glob.glob(os.path.join(self.progress_dir, "progress_*.json"))
            
            if not progress_files:
                logger.info("未找到历史进度文件")
                return

            # 按修改时间排序（最新在前）
            progress_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = progress_files[0]

            with open(latest_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)

                loaded_ids = {res['task_id'] for res in progress['results']}
                if loaded_ids != set(progress['processed_ids']):
                    logger.error("progress文件id不一致，请重新检查再试...")
                    raise Exception()

                self.processed_ids = set(progress['processed_ids'])
                self.results = progress['results']
                self.total_usage = UsageInfo(**progress['usage'])
                logger.info(f"已从 {os.path.basename(latest_file)} 恢复进度，已处理 {len(self.processed_ids)} 条")

        except Exception as e:
            logger.error(f"进度加载失败: {str(e)}")

    def _save_progress(self, force=False) -> None:
        """保存进度（带时间戳和文件数控制）"""
        # 保存频率检查
        if not force and (len(self.results) - self.last_checkpoint < self.checkpoint_interval):
            return

        # 准备进度数据
        progress = {
            'processed_ids': list(self.processed_ids),
            'results': self.results,
            'usage': {
                'prompt_tokens': self.total_usage.prompt_tokens,
                'completion_tokens': self.total_usage.completion_tokens,
                'total_tokens': self.total_usage.total_tokens
            }
        }

        try:
            # 生成带时间戳的文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"progress_{timestamp}.json"
            filepath = os.path.join(self.progress_dir, filename)
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            logger.debug(f"进度已保存至 {filename}")

            self._clean_old_progress_files()

            self.last_checkpoint = len(self.results)
        except Exception as e:
            logger.error(f"进度保存失败: {str(e)}")
    
    def _clean_old_progress_files(self, keep: Optional[int] = None):
        """清理超出数量限制的旧进度文件"""
        keep = keep if keep is not None else self.max_progress_files
        progress_files = sorted(glob.glob(os.path.join(self.progress_dir, "progress_*.json")), 
                            key=os.path.getmtime)
        remove_count = len(progress_files) - keep
        if remove_count > 0:
            for filepath in progress_files[:remove_count]:
                try:
                    os.remove(filepath)
                    logger.debug(f"清理旧进度文件: {os.path.basename(filepath)}")
                except Exception as e:
                    logger.warning(f"文件清理失败 {filepath}: {str(e)}")

    def _parse_llm_output(self, llm_output: str) -> Optional[Dict]:
        """解析LLM输出并标准化格式"""

        lines = llm_output.strip().split('\n')
        hate_flag = 'non-hate'
        category = 'non-hate'
        if not lines:
            return None
        for line in lines:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) != 2:
                return None

            # 处理 targeted_group 和 hateful
            hate_flag = swapped_hate_flag_mapping.get(parts[0].upper(), None)
            category = swapped_category_mapping.get(parts[1].upper(), None)

            if not hate_flag or not category:
                return None

        return {
                'pred_targeted_group': category,
                'pred_hateful': hate_flag
            }

    def _process_item(self, item: Dict) -> Optional[Dict[str, Any]]:
        """处理单个数据项（含重试机制）"""

        if self._shutdown_flag:
            return None  # 提前终止处理

        item_id = item['task_id']
        prompt = item['prompt']
        results = None
        backoff = 0
        error_code = 1
        response = ""
        last_error = ""
        
        for attempt in range(self.max_retries + 1):
            if self._shutdown_flag:
                logger.debug(f"中断信号已接收，终止处理ID: {item['task_id']}")
                return None
            text = item["content"]
            logger.info(f"正在处理文本: {text}")
            try:
                response, usage, error_code = self.llm.chat(
                    prompt=prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                logger.debug(f"输出: {response}")

                # 更新使用统计
                with self.lock:
                    if usage:
                        self.total_usage.prompt_tokens += usage.prompt_tokens
                        self.total_usage.completion_tokens += usage.completion_tokens
                        self.total_usage.total_tokens += usage.total_tokens

                if error_code == 0:
                    results = self._parse_llm_output(response)
                    if results:
                        return {
                            **item,
                            **results,
                            "llm_output": response,
                            "status": "success",
                            "attempts": attempt + 1
                        }
                    else:
                        last_error = "Validation failed"
                        logger.warning(f"LLM输出验证失败 (ID:{item_id} 尝试:{attempt+1})")
                        backoff = 0
                else:
                    last_error = f"API error: error code {error_code}"
                    logger.warning(f"API错误 (ID:{item_id} 尝试:{attempt+1}) 错误码: {error_code}")
                    if error_code == 429:
                        backoff = 2 ** (attempt + self.base_retry_wait_time + 4)
                    
            except requests.exceptions.Timeout:
                logger.warning(f"请求超时 (ID:{item_id} 尝试:{attempt+1})")
                backoff = 2 ** (attempt + self.base_retry_wait_time)
            except Exception as e:
                logger.exception(e)
                logger.error(f"处理异常 (ID:{item_id}): {str(e)}", exc_info=True)
                backoff = 2 ** (attempt + self.base_retry_wait_time)
                self._shutdown_flag = True

            if attempt < self.max_retries:
                logger.info(f"等待{backoff}s后重试")
                time.sleep(backoff)

        final_status = "invalid" if error_code == 0 else "failed"
        return {
            **item,
            "llm_output": response if error_code == 0 else None,
            "status": final_status,
            "attempts": self.max_retries + 1,
            "error": last_error
        }
    
    def _save_final_results(self, metric: Metrics):
        """保存最终结果并清理"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_name = f"output_{self.llm.model_name}_{self.shot_num}_{self.seed}_{timestamp}.json"
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_name)

        info = {"model": self.llm.model_name,
                "shot_num": self.shot_num,
                "seed": self.seed,
                'usage': self.total_usage.model_dump()}

        if not self.step1_result_file:
            metric_dict = metric.run(self.results)
        else:
            metric_dict = None

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"info": info, "metric": metric_dict, "results": self.results}, f, ensure_ascii=False, indent=2)
            
            self._clean_old_progress_files(keep=0)
            
            logger.info(f"最终结果已保存至 {self.output_dir}")
        except Exception as e:
            logger.error(f"结果保存失败: {str(e)}")
            raise
    
    def run(self, metric: Metrics) -> None:
        """执行分析任务"""
        # 加载进度
        self._load_progress()
        
        # 读取数据
        if not self.step1_result_file:
            dataset = self._create_datas(
                shot_data_path=self.shot_dataset_file, 
                test_data_path=self.test_dataset_file, 
                shot_num=self.shot_num, 
                seed=self.seed, 
                prompt_template=TRAIN_PROMPT_STEP_2_V1, 
                shot_prompt_template=SHOT_PROMPT_STEP_2_V1)
        else:
            dataset = self._create_datas(
                step1_result_file=self.step1_result_file, 
                shot_num=self.shot_num, 
                seed=self.seed, 
                prompt_template=TRAIN_PROMPT_STEP_2_V1, 
                shot_prompt_template=SHOT_PROMPT_STEP_2_V1)


        total_items = len(dataset)
        pending_items = [item for item in dataset if item['task_id'] not in self.processed_ids]

        logger.info(f"开始处理，总数据量: {total_items}，待处理: {len(pending_items)}")

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            self.executor = executor
            futures = {}

            try:
                # 提交初始批次
                for item in pending_items[:self.concurrency * 2]:
                    future = executor.submit(self._process_item, item)
                    futures[future] = item['task_id']

                # 主处理循环
                while futures and not self._shutdown_flag:
                    done, _ = concurrent.futures.wait(
                        futures,
                        timeout=1.0,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    
                    # 处理已完成任务
                    for future in done:
                        item_id = futures.pop(future)
                        result = future.result()

                        if result:
                            with self.lock:
                                logger.debug(json.dumps(result, ensure_ascii=False, indent=2))
                                self.results.append(result)
                                self.processed_ids.add(item_id)

                                self._save_progress()

                            # 打印进度
                            processed = len(self.processed_ids)
                            if processed % 1 == 0:
                                logger.info(
                                    f"进度: {processed}/{total_items} "
                                    f"({processed/total_items:.1%}) | "
                                    f"成功率: {len([r for r in self.results if r['status']=='success'])/processed:.1%}"
                                    )

                    # 提交新任务
                    if not self._shutdown_flag:

                        for item in pending_items:
                            with self.lock:
                                if item['task_id'] in self.processed_ids:
                                    continue

                                # 检查是否已提交至处理队列
                                if item['task_id'] in futures.values():
                                    continue
            
                            if len(futures) >= self.concurrency * 2:
                                break

                            with self.lock:
                                if item['task_id'] in self.processed_ids:
                                    continue

                            if item['task_id'] in futures.values():
                                continue

                            # 提交任务
                            if not self._shutdown_flag:
                                future = executor.submit(self._process_item, item)
                                futures[future] = item['task_id']

            except Exception as e:
                logger.exception(e)
                logger.error(f"运行时异常: {str(e)}")
                self._shutdown_flag = True

            finally:
                # 清理线程池
                self.executor.shutdown(wait=True)
                self.executor = None

        # 最终处理
        if self._shutdown_flag:
            logger.warning("处理被中断")
            with self.lock:
                self._save_progress(force=True)
        else:
            self._save_final_results(metric=metric)
            self._show_summary()

    def _show_summary(self):
        """显示汇总报告"""
        logger.info("\n=== 任务汇总 ===")
        logger.info(f"总处理条目: {len(self.results)}")
        logger.info(f"成功条目: {len([r for r in self.results if r['status']=='success'])}")
        logger.info(f"失败条目: {len([r for r in self.results if r['status']!='success'])}")
        logger.info("\n=== Token消耗 ===")
        logger.info(f"Prompt tokens: {self.total_usage.prompt_tokens}")
        logger.info(f"Completion tokens: {self.total_usage.completion_tokens}")
        logger.info(f"Total tokens: {self.total_usage.total_tokens}")

if __name__ == "__main__" :
    model = AliyunApiLLMModel(
        model_name="qwen2.5-7b-instruct-ft-202504240006-d921",
        api_base="https://dashscope.aliyuncs.com/api/v1",
        api_key="sk-22deaa18dd6b423983d438ccd0aa4a2c",
        use_dashscope=True,
        system_prompt=TRAIN_PROMPT_STEP_2_SYSTEM_V1
    )

    tester = TwoStepLLMTester(
        llm_model=model,
        shot_dataset_file="./data/temp_train_data.json",
        test_dataset_file="./data/temp_test_data.json",
        shot_num=0,
        seed=23333333,
        concurrency=10,
        output_dir="./two_step/step2/result/"
    )

    # tester = TwoStepLLMTester(
    #     llm_model=model,
    #     step1_result_file="/mnt/i/project/hateSpeechDetection/two_step/step1/result/output_qwen2.5-7b-instruct-ft-202504232331-1485_0_23333333_20250424_162731.json",
    #     shot_num=15,
    #     concurrency=3,
    #     output_dir="./two_step/step2/result/"
    # )

    # metric = LLMmetrics()

    tester.run(metric=metric)
    # print(tester._parse_llm_output("B | D"))