from utils.log import init_logger
logger = init_logger()

import concurrent.futures
import os
import sys
import json
import time
import glob
import signal
import requests
import threading
import concurrent
from queue import Queue
from typing import Optional, List, Dict, Set, Any
from concurrent.futures import ThreadPoolExecutor

sys.path.append(".")

from prompt import *
from api.llm import AliyunApiLLMModel, ApiLLMModel
from utils.protocol import UsageInfo
from tools.build_prompt import get_shots

def build_shot_prompt(
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
        answer_lines = []
        for q in shot["quadruples"]:
            target = (q.get("target", "") or "NULL").strip()
            argument = (q.get("argument", "") or "NULL").strip()
            targeted_group = (q.get("targeted_group", "") or "NULL").strip()
            hateful = (q.get("hateful", "") or "NULL").strip()

            answer_lines.append(f"{target} | {argument} | {targeted_group} | {hateful}")
        
        # 构建单个示例
        examples.append(
            shot_prompt_template.format(
                text=shot["content"],
                answer="\n".join(answer_lines)
            )
        )
    
    # 组合所有示例并生成包含示例的prompt模板
    return prompt_template.format(
        text = "{text}",
        shots="\n\n".join(examples)
    )


class FewShotLLMTester:
    def __init__(
        self,
        llm_model: AliyunApiLLMModel | ApiLLMModel,
        shot_dataset_file: str,
        test_dataset_file: str,
        shot_num: int,
        seed: int,
        prompts_save_dir: str,
        output_dir: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        concurrency: int = 1,
        max_retries: int = 3,
        base_retry_wait_time: int = 3,
        request_timeout: float = 45.0,
        checkpoint_interval: int = 10,
        progress_dir: str = "./few_shot/progress",
        max_progress_files: int = 5,
        input_file: Optional[str] = None,
    ):
        """
        FewShot测试执行器

        Args:
            llm_model: 配置好的LLM模型实例
            shot_dataset_file: 例子抽样数据集,
            test_dataset_file: 测试数据集,
            shot_num: 例子抽样个数,
            seed: 随机抽样种子，用于复现结果,
            prompts_save_dir: 生成的prompt文件的保存路径,
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
        self.shot_num = shot_num
        self.seed = seed

        self.prompts_save_dir = prompts_save_dir
        self.input_file = input_file
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

                loaded_ids = {res['id'] for res in progress['results']}
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

    def _create_datas(
            self,
            shot_data_path: Optional[str], 
            test_data_path: str, 
            shot_num: int = 5, 
            seed: int = 23333333,
            prompt_template: str = TRAIN_PROMPT_FEW_SHOT_V1,
            shot_prompt_template: str = SHOT_PROMPT_V1,
            system_prompt: Optional[str] = None):
        
        if shot_data_path:
            with open(shot_data_path, "r") as f:
                self.shot_datas = json.load(f)
        else:
            self.shot_datas = []
        
        with open(test_data_path, "r") as f:
            self.test_datas = json.load(f)

        if system_prompt:
            prompt_template = prompt_template.replace("{system_prompt}", system_prompt)
        else:
            prompt_template = prompt_template.replace("{system_prompt}", "")

        if shot_num:
            sample_shots = get_shots(self.shot_datas, shot_num=shot_num, seed=seed)
            prompt_template = build_shot_prompt(
                sample_shots, 
                prompt_template, 
                shot_prompt_template)
        else:
            prompt_template = prompt_template.replace("{shots}", "")

        all_datas: list[dict] = []

        for data in self.test_datas:
            all_datas.append(
                {
                    "id": data["id"], 
                    "content": data["content"], 
                    "quadruples": data.get("quadruples", None), 
                    "prompt": prompt_template.format(text=data["content"], system_prompt = "")})
        
        return all_datas
    
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

    def _parse_llm_output(self, llm_output: str) -> List[Dict]:
        """解析LLM输出并标准化格式"""
        quadruples = []
        group_mapping = {
            'racism': 'Racism',
            'region': 'Region',
            'lgbtq': 'LGBTQ',
            'sexism': 'Sexism',
            'others': 'Others',
            'non_hate': 'non_hate',  # 特殊处理为全小写
            'non-hate': 'non_hate',
            'nonhate': 'non_hate',   # 兼容无下划线格式
            'nohate': 'non_hate',  # 兼容简写格式
            'hate': 'hate'
        }
        
        lines = llm_output.strip().split('\n')
        for line in lines:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) != 4:
                continue

            # 处理 target 和 argument
            target = parts[0] if parts[0].upper() != 'NULL' else None
            argument = parts[1] if parts[1].upper() != 'NULL' else None

            # 智能处理 targeted_group
            tg_raw = parts[2].strip().lower()
            targeted_groups: list[str] = []
            for tg_raw_sp in tg_raw.split(","):
                tg = group_mapping.get(tg_raw_sp.strip(), None)
                if not tg:
                    continue
                targeted_groups.append(tg)
            targeted_group = ", ".join(targeted_groups)
            
            # 处理 hateful
            hateful = parts[3].strip().lower()
            hateful = group_mapping.get(hateful, None)
            hateful = hateful if hateful in {'hate', 'non_hate'} else None

            if targeted_group and hateful:  # 只有格式合法才保留
                quadruples.append({
                    'target': target,
                    'argument': argument,
                    'targeted_group': targeted_group,
                    'hateful': hateful
                })
        return quadruples

    def _validate_quadruples(self, quadruples: List[Dict]) -> bool:
        """验证四元组格式"""
        if not quadruples:
            return False
            
        # 最终合法值集合（注意大小写）
        valid_targeted_groups = {'Racism', 'Region', 'LGBTQ', 'Sexism', 'Others', 'non_hate'}
        valid_hateful = {'hate', 'non_hate'}
        
        return all(
            (all(s in valid_targeted_groups for s in q['targeted_group'].split(", "))) and q['targeted_group'] and
            (q['hateful'] in valid_hateful) and q['hateful']
            for q in quadruples
        )

    def _process_item(self, item: Dict) -> Optional[Dict[str, Any]]:
        """处理单个数据项（含重试机制）"""

        if self._shutdown_flag:
            return None  # 提前终止处理

        item_id = item['id']
        prompt = item['prompt']
        quadruples = []
        error_code = 1
        response = ""
        last_error = ""
        
        for attempt in range(self.max_retries + 1):
            if self._shutdown_flag:
                logger.debug(f"中断信号已接收，终止处理ID: {item['id']}")
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

                if error_code == 0 and isinstance(response, str):
                    quadruples = self._parse_llm_output(response)
                    if self._validate_quadruples(quadruples):
                        return {
                            **item,
                            "llm_output": response,
                            "parsed_quadruples": quadruples,
                            "status": "success",
                            "attempts": attempt + 1
                        }
                    else:
                        last_error = "Validation failed"
                        logger.warning(f"LLM输出验证失败 (ID:{item_id} 尝试:{attempt+1})")
                else:
                    last_error = f"API error: error code {error_code}"
                    logger.warning(f"API错误 (ID:{item_id} 尝试:{attempt+1}) 错误码: {error_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"请求超时 (ID:{item_id} 尝试:{attempt+1})")
            except Exception as e:
                logger.exception(e)
                logger.error(f"处理异常 (ID:{item_id}): {str(e)}", exc_info=True)

            if attempt < self.max_retries:
                backoff = 2 ** (attempt + self.base_retry_wait_time)
                logger.info(f"等待{backoff}s后重试")
                time.sleep(backoff)

        final_status = "invalid" if error_code == 0 else "failed"
        return {
            **item,
            "llm_output": response if error_code == 0 else None,
            "parsed_quadruples": quadruples,
            "status": final_status,
            "attempts": self.max_retries + 1,
            "error": last_error
        }
    

    def _save_final_results(self):
        """保存最终结果并清理"""

        output_name = f"output_{self.llm.model_name}_{self.shot_num}_{self.seed}.json"
        output_path = os.path.join(self.output_dir, output_name)

        info = {"model": self.llm.model_name,
                "shot_num": self.shot_num,
                "seed": self.seed,
                'usage': self.total_usage.model_dump()}

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"info": info, "results": self.results}, f, ensure_ascii=False, indent=2)
            
            self._clean_old_progress_files(keep=0)
            
            logger.info(f"最终结果已保存至 {self.output_dir}")
        except Exception as e:
            logger.error(f"结果保存失败: {str(e)}")
            raise
    
    def run(self) -> None:
        """执行分析任务"""
        # 加载进度
        self._load_progress()
        
        # 读取数据
        dataset = self._create_datas(
            self.shot_dataset_file, 
            self.test_dataset_file, 
            self.shot_num, 
            self.seed, 
            prompt_template=TRAIN_PROMPT_FEW_SHOT_V1, 
            shot_prompt_template=SHOT_PROMPT_V1)
        
        total_items = len(dataset)
        pending_items = [item for item in dataset if item['id'] not in self.processed_ids]

        logger.info(f"开始处理，总数据量: {total_items}，待处理: {len(pending_items)}")

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            self.executor = executor
            futures = {}

            try:
                # 提交初始批次
                for item in pending_items[:self.concurrency * 2]:
                    future = executor.submit(self._process_item, item)
                    futures[future] = item['id']

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
                                if item['id'] in self.processed_ids:
                                    continue

                                # 检查是否已提交至处理队列
                                if item['id'] in futures.values():
                                    continue
            
                            if len(futures) >= self.concurrency * 2:
                                break

                            with self.lock:
                                if item['id'] in self.processed_ids:
                                    continue

                            if item['id'] in futures.values():
                                continue

                            # 提交任务
                            if not self._shutdown_flag:
                                future = executor.submit(self._process_item, item)
                                futures[future] = item['id']

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
            self._save_final_results()
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
    # model = AliyunApiLLMModel(
    #     model_name="deepseek-v3",
    #     api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    #     api_key="sk-06107e55e13c4b67b7f8a262548b9b53",
    # )

    from prompt import TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V2

    # model = AliyunApiLLMModel(
    #     model_name="qwen2.5-7b-instruct-ft-202504180934-6766",
    #     api_base="https://dashscope.aliyuncs.com/api/v1",
    #     api_key="sk-22deaa18dd6b423983d438ccd0aa4a2c",
    #     use_dashscope=True,
    #     system_prompt=TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V2
    # )

    model = ApiLLMModel(
        model_name="qwen3-8b",
        api_base="http://127.0.0.1:5001/api/v2/",
        api_key='23333333',
        system_prompt=TRAIN_PROMPT_ZERO_SHOT_V2
    )

    # tester = FewShotLLMTester(
    #     llm_model=model,
    #     shot_dataset_file="./data/temp_train_data.json",
    #     test_dataset_file="./data/temp_test_data.json",
    #     shot_num=20,
    #     seed=233,
    #     concurrency=5,
    #     prompts_save_dir="./few_shot/data/",
    #     output_dir="./few_shot/output/"
    # )

    tester = FewShotLLMTester(
        llm_model=model,
        shot_dataset_file="data/full/std/train.json",
        test_dataset_file="data/full/std/test.json",
        shot_num=0,
        seed=23333333,
        concurrency=1,
        prompts_save_dir="./data/prompts/",
        output_dir="few_shot/output/"
    )

    tester.run()
    # print(tester._parse_llm_output("中国人为啥要吹黑人性能力 | 女人真信了谣言去找黑人了又哭天抢地 | Sexism, Racism | hate"))