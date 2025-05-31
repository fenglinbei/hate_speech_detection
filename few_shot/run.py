from utils.log import init_logger
logger = init_logger(level="INFO", show_console=True)

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
from tqdm import tqdm
from queue import Queue
from functools import partial
from typing import Optional, List, Dict, Set, Any
from concurrent.futures import ThreadPoolExecutor

from prompt import *
from api.llm import AliyunApiLLMModel, ApiLLMModel
from api.llm import AliyunApiLLMModel, ApiLLMModel
from utils.protocol import UsageInfo
from tools.build_prompt import get_shots
from metrics.metric_llm import LLMmetrics

def build_shot_prompt(
        shots: list[dict],
        prompt_template: str,
        shot_prompt_template: str) -> str:
    """Construct prompt with examples"""

    examples = []
    for shot in shots:
        # Generate answer section
        answer_lines = []
        for q in shot["quadruples"]:
            target = (q.get("target", "") or "NULL").strip()
            argument = (q.get("argument", "") or "NULL").strip()
            targeted_group = (q.get("targeted_group", "") or "NULL").strip()
            hateful = (q.get("hateful", "") or "NULL").strip()

            answer_lines.append(f"{target} | {argument} | {targeted_group} | {hateful}")
        
        # Build single example
        examples.append(
            shot_prompt_template.format(
                text=shot["content"],
                answer="\n".join(answer_lines)
            )
        )
    
    # Combine all examples into the prompt template
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
        prompts_save_dir: str,
        output_dir: str,
        shot_num: int = 0,
        seed: int = 23333333,
        concurrency: int = 1,
        max_retries: int = 3,
        base_retry_wait_time: int = 3,
        request_timeout: float = 45.0,
        checkpoint_interval: int = 10,
        progress_dir: str = "./few_shot/progress",
        max_progress_files: int = 5,
        input_file: Optional[str] = None,
        metric: Optional[LLMmetrics] = None
    ):
        """
        Few-shot test executor

        Args:
            llm_model: Configured LLM model instance
            shot_dataset_file: Dataset for demonstration examples
            test_dataset_file: Test dataset
            shot_num: Number of demonstration examples
            seed: Random seed for reproducibility
            prompts_save_dir: Path to save generated prompts
            output_dir: Output directory for results
            max_tokens: Max tokens for generation
            temperature: Sampling temperature (None for model default)
            concurrency: Concurrent request limit
            max_retries: Max retry attempts
            request_timeout: Request timeout in seconds
            checkpoint_interval: Checkpoint saving interval
            progress_dir: Progress tracking directory
            max_progress_files: Max progress files to retain
            input_file: Pre-generated prompts file (optional)
            metric: Evaluation metric module
        """
        self.llm = llm_model
        self.shot_dataset_file = shot_dataset_file
        self.test_dataset_file = test_dataset_file
        self.shot_num = shot_num
        self.seed = seed

        self.prompts_save_dir = prompts_save_dir
        self.input_file = input_file
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.base_retry_wait_time = base_retry_wait_time
        self.request_timeout = request_timeout
        self.checkpoint_interval = checkpoint_interval
        self.progress_dir = progress_dir
        self.max_progress_files = max_progress_files
        self.metric = metric
        os.makedirs(self.progress_dir, exist_ok=True)

        # State management
        self.lock = threading.RLock()
        self._init_state()

        self._shutdown_flag = False
        self.executor = None

        # Signal handling
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)

    def _init_state(self):
        self.processed_ids: Set[int] = set()
        self.results: List[Dict] = []
        self.total_usage = UsageInfo()
        self.last_checkpoint = 0
    
    def _graceful_shutdown(self, signum, frame):
        """Handle interrupt signals"""

        logger.info(f"\nReceived signal {signum}, initiating shutdown...")
        self._shutdown_flag = True
        
        if self.executor:
            logger.info("Terminating Thread Pool...")
            self.executor.shutdown(wait=True)

        with self.lock:
            self._save_progress(force=True)
        
        logger.info("Progress saved. You can resume by rerunning the program.")

    def _load_progress(self) -> None:
        """Load latest progress file"""

        try:
            progress_files = glob.glob(os.path.join(self.progress_dir, "progress_*.json"))
            
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
        """Save progress with timestamp and file rotation"""

        if not force and (len(self.results) - self.last_checkpoint < self.checkpoint_interval):
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
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"progress_{timestamp}.json"
            filepath = os.path.join(self.progress_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            logger.debug(f"Progress saved to {filename}")

            self._clean_old_progress_files()

            self.last_checkpoint = len(self.results)
        except Exception as e:
            logger.error(f"Progress save failed: {str(e)}")

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
                    "gt_quadruples": data.get("quadruples", None), 
                    "prompt": prompt_template.format(text=data["content"], system_prompt = "")})
        
        return all_datas
    
    def _clean_old_progress_files(self, keep: Optional[int] = None):
        """Clean up old progress files"""

        keep = keep if keep is not None else self.max_progress_files
        progress_files = sorted(glob.glob(os.path.join(self.progress_dir, "progress_*.json")), 
                            key=os.path.getmtime)
        remove_count = len(progress_files) - keep
        if remove_count > 0:
            for filepath in progress_files[:remove_count]:
                try:
                    os.remove(filepath)
                    logger.debug(f"Cleaned old progress file: {os.path.basename(filepath)}")
                except Exception as e:
                    logger.warning(f"Failed to clean file {filepath}: {str(e)}")

    def _parse_llm_output(self, llm_output: str) -> List[Dict]:
        """Parse and standardize LLM output"""

        quadruples = []
        group_mapping = {
            'racism': 'Racism',
            'region': 'Region',
            'lgbtq': 'LGBTQ',
            'sexism': 'Sexism',
            'others': 'Others',
            'non_hate': 'non_hate',
            'non-hate': 'non_hate',
            'nonhate': 'non_hate',
            'nohate': 'non_hate',
            'hate': 'hate'
        }
        
        lines = llm_output.strip().split('\n')
        for line in lines:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) != 4:
                continue

            target = parts[0] if parts[0].upper() != 'NULL' else None
            argument = parts[1] if parts[1].upper() != 'NULL' else None

            tg_raw = parts[2].strip().lower()
            targeted_groups: list[str] = []
            for tg_raw_sp in tg_raw.split(","):
                tg = group_mapping.get(tg_raw_sp.strip(), None)
                if not tg:
                    continue
                targeted_groups.append(tg)
            targeted_group = ", ".join(targeted_groups)
            
            hateful = parts[3].strip().lower()
            hateful = group_mapping.get(hateful, None)
            hateful = hateful if hateful in {'hate', 'non_hate'} else None

            if targeted_group and hateful:
                quadruples.append({
                    'target': target,
                    'argument': argument,
                    'targeted_group': targeted_group,
                    'hateful': hateful
                })
        return quadruples

    def _validate_quadruples(self, quadruples: List[Dict]) -> bool:
        """Validate quadruple format"""

        if not quadruples:
            return False
            
        valid_targeted_groups = {'Racism', 'Region', 'LGBTQ', 'Sexism', 'Others', 'non_hate'}
        valid_hateful = {'hate', 'non_hate'}
        
        return all(
            (all(s in valid_targeted_groups for s in q['targeted_group'].split(", "))) and q['targeted_group'] and
            (q['hateful'] in valid_hateful) and q['hateful']
            for q in quadruples
        )

    def _process_item(
            self, 
            item: dict,
            llm_params: dict) -> Optional[Dict[str, Any]]:
        """Process single item with retry mechanism"""

        if self._shutdown_flag:
            return None

        item_id = item['id']
        prompt = item['prompt']
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

        
        for attempt in range(self.max_retries + 1):
            if self._shutdown_flag:
                logger.debug(f"Shutdown signal received, aborting ID: {item['id']}")
                return None
            text = item["content"]
            logger.debug(f"Processing text: {text}")
            try:
                response, usage, status_code = self.llm.chat(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    n=n,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    enable_thinking=enable_thinking
                )
                logger.debug(f"LLM Output: {response}")
                if isinstance(response, list):
                    answer = response[0][0]

                    with self.lock:
                        if usage:
                            self.total_usage.prompt_tokens += usage.prompt_tokens
                            self.total_usage.completion_tokens += usage.completion_tokens
                            self.total_usage.total_tokens += usage.total_tokens

                    if status_code == 200:
                        
                        if isinstance(answer, str):
                            quadruples = self._parse_llm_output(answer)
                            if self._validate_quadruples(quadruples):
                                return {
                                    **item,
                                    "llm_output": answer,
                                    "pred_quadruples": quadruples,
                                    "status": "success",
                                    "attempts": attempt + 1
                                }
                            else:
                                last_error = "Validation failed"
                                logger.warning(f"LLM output validation failed (ID:{item_id} attempt:{attempt+1})")
                                backoff = 0
                        else:
                            last_error = "Invalid Output"
                            logger.warning(f"Empty LLM output (ID:{item_id} attempt:{attempt+1})")
                            backoff = 2 ** (attempt + self.base_retry_wait_time)
                    else:
                        last_error = f"API error: status code {status_code}"
                        logger.warning(f"API error (ID:{item_id} attempt:{attempt+1}) code: {status_code}")
                        if status_code == 429:
                            backoff = 2 ** (attempt + self.base_retry_wait_time + 4)
                else:
                    last_error = "Invalid Output"
                    logger.warning(f"Empty LLM output (ID:{item_id} attempt:{attempt+1})")
                    backoff = 2 ** (attempt + self.base_retry_wait_time)
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (ID:{item_id} attempt:{attempt+1})")
                backoff = 2 ** (attempt + self.base_retry_wait_time)
            except Exception as e:
                logger.exception(e)
                logger.error(f"Processing error (ID:{item_id}): {str(e)}", exc_info=True)
                backoff = 2 ** (attempt + self.base_retry_wait_time)
                self._shutdown_flag = True

            if attempt < self.max_retries:
                logger.debug(f"Retrying after {backoff}s")
                time.sleep(backoff)

        final_status = "invalid" if status_code == 200 else "failed"
        return {
            **item,
            "llm_output": answer if status_code == 200 else None,
            "parsed_quadruples": quadruples,
            "status": final_status,
            "attempts": self.max_retries + 1,
            "error": last_error
        }
    

    def _save_final_results(
            self, 
            llm_params: Optional[dict] = None,
            metric_results: Optional[dict] = None):
        """Save final results and clean up"""

        output_name = f"output_{self.llm.model_name}_{self.shot_num}_{self.seed}.json"
        output_path = os.path.join(self.output_dir, output_name)

        info = {"model": self.llm.model_name,
                "shot_num": self.shot_num,
                "seed": self.seed,
                'usage': self.total_usage.model_dump(),
                "llm_params": llm_params}

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "info": info, 
                        "results": self.results,
                        "metric": metric_results
                        }, f, ensure_ascii=False, indent=2)
            
            self._clean_old_progress_files(keep=0)
            
            logger.info(f"Final results saved to {self.output_dir}")
        except Exception as e:
            logger.error(f"Result save failed: {str(e)}")
            raise
    
    def run(
            self, 
            llm_params: Dict,
            seed: Optional[int] = None,
            shot_num: Optional[int] = None
            ) -> None:
        """Execute analysis task"""

        self._init_state()

        if seed is not None:
            self.seed = seed
        if shot_num is not None:
            self.shot_num = shot_num

        self._load_progress()
        dataset = self._create_datas(
            self.shot_dataset_file, 
            self.test_dataset_file, 
            self.shot_num, 
            self.seed, 
            prompt_template=TRAIN_PROMPT_FEW_SHOT_V1, 
            shot_prompt_template=SHOT_PROMPT_V1)
        
        total_items = len(dataset)
        pending_items = [item for item in dataset if item['id'] not in self.processed_ids]

        logger.info(f"LLM Params: {json.dumps(llm_params, ensure_ascii=False, indent=2)}")

        logger.info(f"Starting processing. Total items: {total_items}, Pending: {len(pending_items)}")

        pbar = tqdm(
            total=len(pending_items),
            desc=f"Processing shot: {shot_num}",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            self.executor = executor
            futures = {}

            try:
                # Submit initial batch
                for item in pending_items[:self.concurrency * 2]:
                    future = executor.submit(self._process_item, item, llm_params)
                    futures[future] = item['id']

                # Main processing loop
                while futures and not self._shutdown_flag:
                    done, _ = concurrent.futures.wait(
                        futures,
                        timeout=1.0,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    # Process completed tasks
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
                            pbar.set_postfix({
                                "success": f"{success_count}/{len(self.results)}",
                                "rate": f"{success_count/len(self.results):.1%}" if len(self.results) else "0%"
                            })

                    # Submit new tasks
                    if not self._shutdown_flag:

                        for item in pending_items:
                            with self.lock:
                                if item['id'] in self.processed_ids:
                                    continue

                                if item['id'] in futures.values():
                                    continue
            
                            if len(futures) >= self.concurrency * 2:
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
            if isinstance(self.metric, LLMmetrics):
                metric_results = self.metric.run(datas_list=self.results)
            else:
                metric_results = None
            self._save_final_results(llm_params, metric_results=metric_results)
            self._show_summary()

    def _show_summary(self):
        """Display summary report"""
        logger.info("\n=== Summary ===")
        logger.info(f"Total processed: {len(self.results)}")
        logger.info(f"Successful items: {len([r for r in self.results if r['status']=='success'])}")
        logger.info(f"Failed items: {len([r for r in self.results if r['status']!='success'])}")
        logger.info("\n=== Token Usage ===")
        logger.info(f"Prompt tokens: {self.total_usage.prompt_tokens}")
        logger.info(f"Completion tokens: {self.total_usage.completion_tokens}")
        logger.info(f"Total tokens: {self.total_usage.total_tokens}")

if __name__ == "__main__" :
    mertic = LLMmetrics()
    # model = AliyunApiLLMModel(
    #     model_name="qwen2.5-7b-instruct",
    #     api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/",
    #     api_key="sk-06107e55e13c4b67b7f8a262548b9b53",
    # )

    from prompt import TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V2

    # model = AliyunApiLLMModel(
    #     model_name="qwen2.5-7b-instruct",
    #     api_base="https://dashscope.aliyuncs.com/api/v1",
    #     api_key="sk-22deaa18dd6b423983d438ccd0aa4a2c",
    #     use_dashscope=True,
    #     system_prompt=TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V2
    # )

    model = ApiLLMModel(
        model_name="qwen3-8b-think",
        api_base="http://127.0.0.1:5001/v2/",
        api_key='23333333',
        system_prompt=TRAIN_PROMPT_ZERO_SHOT_V2,
        enable_thinking=False,
        timeout=90
    )

    # tester = FewShotLLMTester(
    #     llm_model=model,
    #     shot_dataset_file="./data/v1/temp/temp_train_data.json",
    #     test_dataset_file="./data/v1/temp/temp_test_data.json",
    #     concurrency=5,
    #     prompts_save_dir="./few_shot/data/",
    #     output_dir="./few_shot/output/",
    #     metric=mertic
    # )

    tester = FewShotLLMTester(
        llm_model=model,
        shot_dataset_file="data/full/std/train.json",
        test_dataset_file="data/full/std/test.json",
        shot_num=5,
        base_retry_wait_time=0,
        concurrency=1,
        prompts_save_dir="./data/prompts/",
        output_dir="few_shot/output/",
        metric=mertic
    )

    # for qwen3
    # 对于思考模式，使用 Temperature=0.6，TopP=0.95，TopK=20，以及 MinP=0
    # 对于非思考模式，我们建议使用 Temperature=0.7，TopP=0.8，TopK=20，以及 MinP=0。
    # params = {
    #     "max_new_tokens": 512, 
    #     "n": 1,
    #     "top_p": 0.8,
    #     "top_k": 20,
    #     "min_p": 0,
    #     "temperature": 0.7, 
    #     "enable_thinking": False
    # }

    params = {
        "max_new_tokens": 4096, 
        "n": 1,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "temperature": 0.6, 
        "enable_thinking": True
    }

    for shot_num in range(8, 24, 4):
        tester.run(llm_params=params, shot_num=shot_num)

    # tester.run(llm_params=params, shot_num=10)
    # tester.run(llm_params=params, shot_num=15)
    # tester.run(llm_params=params, shot_num=20)
    # print(tester._validate_quadruples(tester._parse_llm_output("NULL | NULL | non_hate | non_hate")))
    # print(tester._parse_llm_output("中国人为啥要吹黑人性能力 | 女人真信了谣言去找黑人了又哭天抢地 | Sexism, Racism | hate"))
