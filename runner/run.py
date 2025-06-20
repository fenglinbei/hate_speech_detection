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
import argparse
from tqdm import tqdm
from queue import Queue
from functools import partial
from collections import Counter
from typing import Optional, List, Dict, Set, Any, Union
from concurrent.futures import ThreadPoolExecutor

from prompt import *
from rag.core import Retriever
from utils.log import init_logger
from utils.protocol import UsageInfo
from utils.config import ConfigManager
from tools.build_prompt import get_shots
from metrics.metric_llm import LLMmetrics
from engine import AliyunApiLLMModel, ApiLLMModel, VLLM, LLM
from utils.parser import parse_llm_output_quad, validate_quadruples, parse_llm_output_trip
from tools.convert import output2triple



def build_shot_prompt(
        shots: list[dict],
        prompt_template: str,
        shot_prompt_template: str) -> str:
    """构造带有示例的提示"""
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
    
    # 将所有示例组合到提示模板中
    return prompt_template.format(
        text = "{text}",
        shots="\n\n".join(examples)
    )


class LLMTester:
    def __init__(
        self,
        llm_model: Union[AliyunApiLLMModel, ApiLLMModel],
        config: dict,
        metric: Optional[LLMmetrics] = None
    ):
        """
        Few-shot测试执行器
        
        Args:
            llm_model: 配置好的LLM模型实例
            config: 测试配置字典
            metric: 评估指标模块
        """
        self.llm = llm_model
        self.config = config.get('tester', {})
        self.metric = metric
        
        # 确保目录存在
        os.makedirs(self.config.get('progress_dir', './progress'), exist_ok=True)
        os.makedirs(self.config.get('output_dir', './output'), exist_ok=True)
        os.makedirs(self.config.get('prompts_save_dir', './prompts'), exist_ok=True)

        # 状态管理
        self.lock = threading.RLock()
        self._init_state()

        self._shutdown_flag = False
        self.executor = None

        # 信号处理
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)

        self.f1_hard = 0
        self.f1_soft = 0
        self.f1_avg = 0

    def _init_state(self):
        self.processed_ids: Set[int] = set()
        self.results: List[Dict] = []
        self.total_usage = UsageInfo()
        self.last_checkpoint = 0

        self.f1_hard = 0
        self.f1_soft = 0
        self.f1_avg = 0


    def _graceful_shutdown(self, signum, frame):
        """处理中断信号"""
        logger.info(f"\nReceived signal {signum}, initiating shutdown...")
        self._shutdown_flag = True
        
        if self.executor:
            logger.info("Terminating Thread Pool...")
            self.executor.shutdown(wait=True)

        with self.lock:
            self._save_progress(force=True)
        
        logger.info("Progress saved. You can resume by rerunning the program.")

    def _load_progress(self) -> None:
        """加载最新的进度文件"""
        try:
            progress_dir = self.config.get('progress_dir', './progress')
            progress_files = glob.glob(os.path.join(progress_dir, "progress_*.json"))
            
            if not progress_files:
                logger.warning("No history progress files found.")
                return

            # 按修改时间排序（最新在前）
            progress_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = progress_files[0]

            with open(latest_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)

                loaded_ids = {res['id'] for res in progress['results']}
                if loaded_ids != set(progress['processed_ids']):
                    logger.error("ID mismatch in progress file.")
                    raise Exception()

                self.processed_ids = set(progress['processed_ids'])
                self.results = progress['results']
                self.total_usage = UsageInfo(**progress['usage'])
                logger.info(f"Resumed progress from {os.path.basename(latest_file)}, processed {len(self.processed_ids)} items")

        except Exception as e:
            logger.error(f"Progress loading failed: {str(e)}")

    def _save_progress(self, force=False) -> None:
        """保存进度（带时间戳和文件轮换）"""
        if not force and (len(self.results) - self.last_checkpoint < self.config.get('checkpoint_interval', 10)):
            return

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
            progress_dir = self.config.get('progress_dir', './progress')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"progress_{timestamp}.json"
            filepath = os.path.join(progress_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            logger.debug(f"Progress saved to {filename}")

            self._clean_old_progress_files()

            self.last_checkpoint = len(self.results)
        except Exception as e:
            logger.error(f"Progress save failed: {str(e)}")

    def _create_datas(self, use_rag: bool = False, retriever: Optional[Retriever] = None, parallel_num: int = 1) -> List[dict]:
        """创建测试数据集"""
        shot_num = self.config.get('shot_num', 0)
        seed = self.config.get('seed', 23333333)
        
        # 获取模板
        system_prompt = self.config.get('prompt_templates', {}).get('system')
        user_template = self.config.get('prompt_templates', {}).get('train')
        shot_template = self.config.get('prompt_templates', {}).get('shot')
        
        # 加载示例数据
        shot_data_path = self.config.get('shot_dataset_file', "")
        if shot_data_path:
            with open(shot_data_path, "r") as f:
                shot_datas = json.load(f)
        else:
            shot_datas = []
        
        # 加载测试数据
        test_data_path = self.config.get('test_dataset_file')
        with open(test_data_path, "r") as f:
            test_datas = json.load(f)

        # 包含示例的提示
        if shot_num > 0:
            sample_shots = get_shots(shot_datas, shot_num=shot_num, seed=seed)
            user_template = build_shot_prompt(
                sample_shots, 
                user_template, 
                shot_template
            )
        else:
            user_template = user_template.replace("{shots}", "")

        # 构建测试数据

        pbar = tqdm(
            total=len(test_datas),
            desc=f"Preprocessing datas",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )

        all_datas = []
        for data in test_datas:
            prompt_list = []
            if not use_rag:
                user_prompt = user_template.format(text=data["content"])
                prompt_list.append(user_prompt)
            else:
                assert isinstance(retriever, Retriever)
                retrieve_contents, retrieve_outputs = retriever.retrieve(data['content'], top_k=parallel_num)
                for retrieve_content, retrieve_output in zip(retrieve_contents, retrieve_outputs):
                    user_prompt = user_template.replace("{retrieve_content}", retrieve_content).\
                                                replace("{retrieve_output}", output2triple(retrieve_output)).\
                                                replace("{text}", data["content"])
                    prompt_list.append(user_prompt)

            messages_list = []
            if system_prompt:
                for user_prompt in prompt_list:
                    messages = [{'content': system_prompt, 'role': 'system'}, {'content': user_prompt, 'role': 'user'}]
                    messages_list.append(messages)
            else:
                for user_prompt in prompt_list:
                    messages = [{'content': user_prompt, 'role': 'user'}]
                    messages_list.append(messages)

            all_datas.append({
                "id": data["id"], 
                "content": data["content"], 
                "gt_quadruples": data.get("quadruples", []), 
                "messages_list": messages_list
            })
            pbar.update(1)
        
        return all_datas

    def _clean_old_progress_files(self, keep: Optional[int] = None):
        """清理旧的进度文件"""
        keep = keep if keep is not None else self.config.get('max_progress_files', 5)
        progress_dir = self.config.get('progress_dir', './progress')
        progress_files = sorted(glob.glob(os.path.join(progress_dir, "progress_*.json")), 
                            key=os.path.getmtime)
        remove_count = len(progress_files) - keep
        if remove_count > 0:
            for filepath in progress_files[:remove_count]:
                try:
                    os.remove(filepath)
                    logger.debug(f"Cleaned old progress file: {os.path.basename(filepath)}")
                except Exception as e:
                    logger.warning(f"Failed to clean file {filepath}: {str(e)}")

    def _process_item(self, item: dict, llm_params: dict) -> Optional[Dict[str, Any]]:
        """处理单个项目（带重试机制）"""
        if self._shutdown_flag:
            return None

        item_id = item['id']
        quadruples = []
        backoff = 0
        status_code = 500
        response = None
        answer = ""
        last_error = ""

        try:
            max_new_tokens = llm_params["max_new_tokens"]
            n = llm_params["n"]
            top_p = llm_params["top_p"]
            top_k = llm_params["top_k"]
            temperature = llm_params["temperature"]
            enable_thinking = llm_params["enable_thinking"]
        except KeyError as err:
            raise ValueError("Invaild llm_params keys.")

        max_retries = self.config.get('max_retries', 3)
        per_parallel_attempts_num = self.config.get('per_parallel_attempts_num', 1)
        base_wait = self.config.get('base_retry_wait_time', 0)

        text = item["content"]
        logger.debug(f"Processing text: {text}")
        
        pbar = tqdm(
            total=len(item["messages_list"]) * per_parallel_attempts_num,
            desc=f'Processing parallel total: {len(item["messages_list"])}',
            unit="item",
            dynamic_ncols=True,
            leave=True
        )
       
        llm_output_list = []     
        final_status = "failed"
        total_attemps = 0

        for messages in item["messages_list"]:
            logger.debug(f"Processing messages: {messages}")
            for _ in range(per_parallel_attempts_num):
            
                for attempt in range(max_retries + 1):
                    if self._shutdown_flag:
                        logger.debug(f"Shutdown signal received, aborting ID: {item['id']}")
                        return None
                    
                    try:
                        response, usage, status_code = self.llm.chat(
                            messages=messages,
                            max_new_tokens=max_new_tokens,
                            n=n,
                            top_p=top_p,
                            top_k=top_k,
                            temperature=temperature,
                            enable_thinking=enable_thinking
                        )
                        logger.debug(f"LLM Output: {response}")
                        total_attemps += 1
                        if isinstance(response, list):
                            answer = response[0][0]

                            with self.lock:
                                if usage:
                                    self.total_usage.prompt_tokens += usage.prompt_tokens
                                    self.total_usage.completion_tokens += usage.completion_tokens
                                    self.total_usage.total_tokens += usage.total_tokens

                            if status_code == 200:
                                if isinstance(answer, str):
                                    quadruples = parse_llm_output_quad(answer)
                                    if not quadruples:
                                        quadruples = parse_llm_output_trip(answer)
                                    if validate_quadruples(quadruples):
                                        llm_output_list.append(answer)
                                        break
                                    else:
                                        last_error = "Validation failed"
                                        logger.warning(f"LLM output validation failed (ID:{item_id} attempt:{attempt+1})")
                                        backoff = 0
                                else:
                                    last_error = "Invalid Output"
                                    logger.warning(f"Empty LLM output (ID:{item_id} attempt:{attempt+1})")
                                    backoff = 2 ** (attempt + base_wait)
                            else:
                                last_error = f"API error: status code {status_code}"
                                logger.warning(f"API error (ID:{item_id} attempt:{attempt+1}) code: {status_code}")
                                if status_code == 429:
                                    backoff = 2 ** (attempt + base_wait + 4)
                        else:
                            last_error = "Invalid Output"
                            logger.warning(f"Empty LLM output (ID:{item_id} attempt:{attempt+1})")
                            backoff = 2 ** (attempt + base_wait)
                            
                    except requests.exceptions.Timeout:
                        logger.warning(f"Request timeout (ID:{item_id} attempt:{attempt+1})")
                        backoff = 2 ** (attempt + base_wait)
                    except Exception as e:
                        logger.exception(e)
                        logger.error(f"Processing error (ID:{item_id}): {str(e)}", exc_info=True)
                        backoff = 2 ** (attempt + base_wait)
                        self._shutdown_flag = True

                    if attempt < max_retries:
                        logger.debug(f"Retrying after {backoff}s")
                        time.sleep(backoff)

                final_status = "invalid" if status_code == 200 else "failed"
                pbar.update(1)
        
        output_couter = Counter(llm_output_list)

        if len(output_couter) == 0:
            return {
                **item,
                "llm_output": None,
                "pred_quadruples": [],
                "status": final_status,
                "attempts": (max_retries + 1) * len(item["messages_list"]),
                "error": last_error
            }
        
        max_item = output_couter.most_common(1)[0]
        answer = max_item[0]
        quadruples = parse_llm_output_quad(answer)
        if not quadruples:
            quadruples = parse_llm_output_trip(answer)

        return {
                **item,
                "llm_output": answer,
                "frequency": max_item[1],
                "pred_quadruples": quadruples,
                "status": "success",
                "attempts": total_attemps
            }

    

    def _save_final_results(
            self, 
            llm_params: Optional[dict] = None,
            metric_results: Optional[dict] = None,
            output_name: Optional[str] = None):
        """保存最终结果并清理"""
        model_name = self.llm.model_name.replace("/", "_")
        shot_num = self.config.get('shot_num', 0)
        seed = self.config.get('seed', 23333333)
        output_name = f"output_{model_name}_shots{shot_num}_seed{seed}.json" if not output_name else output_name
        output_path = os.path.join(self.config['output_dir'], output_name)

        info = {
            "model": self.llm.model_name,
            "shot_num": shot_num,
            "seed": seed,
            'usage': self.total_usage.model_dump(),
            "llm_params": llm_params,
            "config": self.config
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "info": info, 
                        "results": self.results,
                        "metric": metric_results
                    }, f, ensure_ascii=False, indent=2)
            
            self._clean_old_progress_files(keep=0)
            
            logger.info(f"Final results saved to {output_path}")
        except Exception as e:
            logger.error(f"Result save failed: {str(e)}")
            raise
    
    def run(
            self, 
            llm_params: Dict,
            shot_num: Optional[int] = None,
            retriever: Optional[Retriever] = None,
            parallel_num: int = 1,
            output_name: Optional[str] = None
            ) -> None:
        """执行分析任务"""
        # 更新shot_num配置（如果有）
        if shot_num is not None:
            self.config['shot_num'] = shot_num
            logger.info(f"Override shot_num to: {shot_num}")

        self._init_state()
        self._load_progress()
        if isinstance(retriever, Retriever):
            dataset = self._create_datas(use_rag=True, retriever=retriever, parallel_num=parallel_num)
        else:
            dataset = self._create_datas()
        
        total_items = len(dataset)
        pending_items = [item for item in dataset if item['id'] not in self.processed_ids]

        logger.info(f"LLM Params: {json.dumps(llm_params, ensure_ascii=False, indent=2)}")

        logger.info(f"Starting processing. Total items: {total_items}, Pending: {len(pending_items)}")

        pbar = tqdm(
            total=len(pending_items),
            desc=f"Processing shot: {self.config['shot_num']}",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )

        concurrency = self.config.get('concurrency', 1)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            self.executor = executor
            futures = {}

            try:
                # 提交初始批次
                for item in pending_items[:concurrency * 2]:
                    future = executor.submit(self._process_item, item, llm_params)
                    futures[future] = item['id']

                # 主处理循环
                while futures and not self._shutdown_flag:
                    done, _ = concurrent.futures.wait(
                        futures,
                        timeout=1.0,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    # 处理完成的任务
                    for future in done:
                        item_id = futures.pop(future)
                        result = future.result()

                        if result:
                            with self.lock:
                                logger.debug(json.dumps(result, ensure_ascii=False, indent=2))
                                self.results.append(result)
                                self.processed_ids.add(item_id)

                                self._save_progress()

                            pbar.update(1)
                            success_count = len([r for r in self.results if r['status']=='success'])
                            
                            if len(self.results) % 100 == 0:
                                step_metric = self.metric.run(datas_list=self.results)
                                self.f1_hard = step_metric["f1_hard"]
                                self.f1_soft = step_metric["f1_soft"]
                                self.f1_avg = step_metric["f1_avg"]
                            pbar.set_postfix({
                                "f1_hard": self.f1_hard,
                                "f1_soft": self.f1_soft,
                                "f1_avg": self.f1_avg,
                                "success": f"{success_count}/{len(self.results)}",
                                "rate": f"{success_count/len(self.results):.1%}" if len(self.results) else "0%"
                            })

                    # 提交新任务
                    if not self._shutdown_flag:
                        for item in pending_items:
                            with self.lock:
                                if item['id'] in self.processed_ids:
                                    continue

                                if item['id'] in futures.values():
                                    continue
            
                            if len(futures) >= concurrency * 2:
                                break

                            with self.lock:
                                if item['id'] in self.processed_ids:
                                    continue

                            if item['id'] in futures.values():
                                continue

                            if not self._shutdown_flag:
                                future = executor.submit(self._process_item, item, llm_params)
                                futures[future] = item['id']

            except Exception as e:
                logger.exception(e)
                logger.error(f"Runtime Error: {str(e)}")
                self._shutdown_flag = True

            finally:
                self.executor.shutdown(wait=True)
                self.executor = None

        if self._shutdown_flag:
            logger.warning("Processing interrupted")
            with self.lock:
                self._save_progress(force=True)
        else:
            if self.config.get('compute_metric', False) and self.metric is not None:
                metric_results = self.metric.run(datas_list=self.results)
            else:
                metric_results = None
            self._save_final_results(llm_params, metric_results=metric_results, output_name=output_name)
            self._show_summary()

    def _show_summary(self):
        """显示摘要报告"""
        logger.info("\n=== Summary ===")
        logger.info(f"Total processed: {len(self.results)}")
        logger.info(f"Successful items: {len([r for r in self.results if r['status']=='success'])}")
        logger.info(f"Failed items: {len([r for r in self.results if r['status']!='success'])}")
        logger.info("\n=== Token Usage ===")
        logger.info(f"Prompt tokens: {self.total_usage.prompt_tokens}")
        logger.info(f"Completion tokens: {self.total_usage.completion_tokens}")
        logger.info(f"Total tokens: {self.total_usage.total_tokens}")


def create_model_from_config(model_config: dict) -> Any:
    """根据配置创建模型实例，支持VLLM"""
    model_type = model_config.get('type', 'ApiLLMModel')
    params: dict = model_config.get('params', {})

    if model_type == "LLM":
        return LLM(
            model_path=params.get("model_path"),
            model_name=params.get("model_name")
            )
    elif model_type == 'VLLM':
        # 从参数中提取VLLM配置
        return VLLM(
            model_path=params.get("model_path"),
            model_name=params.get("model_name"),
            tensor_parallel_size=params.get("tensor_parallel_size", 1),
            max_num_seqs=params.get("max_num_seqs", 32),
            max_model_len=params.get("max_model_len", 8192 * 3 // 2),
            gpu_memory_utilization=params.get("gpu_memory_utilization", 0.95),
            seed=params.get("seed", 2024),
        )
    elif model_type == 'AliyunApiLLMModel':
        return AliyunApiLLMModel(**params)
    elif model_type == 'ApiLLMModel':
        params.pop("use_dashscope", None)
        return ApiLLMModel(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
def create_retriever_from_config(retriever_config: dict):
    retriever_type = retriever_config.get('type', 'Retriever')
    params: dict = retriever_config.get('params', {})

    if retriever_type == "Retriever":
        return Retriever(
            model_path=params.get("model_path"),
            model_name=params.get("model_name"),
            data_path=params.get("data_path")
        )


if __name__ == "__main__" :
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='Run few-shot LLM testing')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to configuration file (JSON format)')
    args = parser.parse_args()
    
    # 加载配置
    config = ConfigManager.load_config(args.config)
    
    # 配置日志
    log_config = config.get('log', {})
    logger = init_logger(level=log_config.get('level', 'INFO'), 
                         show_console=log_config.get('show_console', True))
    
    # 创建模型
    model = create_model_from_config(config['model'])

    retriever = None
    if config.get("retriever"):
        retriever_config = config["retriever"]
        retriever = create_retriever_from_config(retriever_config)
    
    # 可选地创建metric
    metric = LLMmetrics() if config['tester'].get('compute_metric', True) else None
    
    # 创建tester
    tester = LLMTester(
        llm_model=model,
        config=config,
        metric=metric
    )
    
    # 运行测试
    run_config = config['tester']['run']
    shots_list = run_config.get('shots_list', [config['tester'].get('shot_num', 5)])
    llm_params = run_config['llm_params']
    output_name = config.get("output_name", None)
    parallel_num = config["tester"].get("parallel_num", 1)

    
    
    for shot_num in shots_list:
        tester.run(llm_params=llm_params, shot_num=shot_num, retriever=retriever, output_name=output_name, parallel_num=parallel_num)