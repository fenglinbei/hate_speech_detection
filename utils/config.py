import sys
import json
from typing import Optional

from utils.log import init_logger
from prompt import *

logger = init_logger()

class ConfigManager:
    """统一配置管理系统"""
    DEFAULTS = {
        "log": {
            "level": "INFO",
            "show_console": True
        },
        "model": {
            "type": "ApiLLMModel",
            "params": {
                "model_name": "Qwen3-8B-sft-hsd-180-no-think",
                "api_base": "http://127.0.0.1:5001/v2/",
                "api_key": "23333333",
                "system_prompt": "TRAIN_PROMPT_ZERO_SHOT_V2",
                "enable_thinking": False,
                "timeout": 90,
                "use_dashscope": False
            }
        },
        "tester": {
            "shot_dataset_file": "data/full/std/train.json",
            "test_dataset_file": "data/full/std/test.json",
            "shot_num": 5,
            "seed": 23333333,
            "prompts_save_dir": "./data/prompts/",
            "output_dir": "few_shot/output/",
            "concurrency": 1,
            "max_retries": 3,
            "base_retry_wait_time": 0,
            "request_timeout": 45.0,
            "checkpoint_interval": 10,
            "progress_dir": "./few_shot/progress",
            "max_progress_files": 5,
            "input_file": None,
            "compute_metric": True,
            "prompt_templates": {
                "train": "TRAIN_PROMPT_FEW_SHOT_V1",
                "shot": "SHOT_PROMPT_V1"
            },
            "run": {
                "shots_list": [2, 10, 14, 18, 22, 24, 26, 28, 30],
                "llm_params": {
                    "max_new_tokens": 4096,
                    "n": 1,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0,
                    "temperature": 0.6,
                    "enable_thinking": True
                }
            }
        }
    }

    @classmethod
    def load_config(cls, config_file: Optional[str] = None) -> dict:
        """加载配置文件"""
        config = cls.DEFAULTS.copy()
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                cls._deep_merge(config, user_config)
                logger.info(f"Loaded config from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config: {str(e)}")
                sys.exit(1)
        
        # 将字符串映射到实际的prompt变量
        config['model']['params']['system_prompt'] = cls._resolve_prompt(config['model']['params']['system_prompt'])
        config['tester']['prompt_templates']['train'] = cls._resolve_prompt(config['tester']['prompt_templates']['train'])
        config['tester']['prompt_templates']['shot'] = cls._resolve_prompt(config['tester']['prompt_templates']['shot'])
        
        return config

    @staticmethod
    def _deep_merge(target: dict, source: dict) -> dict:
        """深度合并两个字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                ConfigManager._deep_merge(target[key], value)
            else:
                target[key] = value
        return target

    @staticmethod
    def _resolve_prompt(prompt_name: str) -> str:
        """将prompt名称映射到实际变量"""
        prompt_map = {
            "TRAIN_PROMPT_FEW_SHOT_V1": TRAIN_PROMPT_FEW_SHOT_V1,
            "SHOT_PROMPT_V1": SHOT_PROMPT_V1,
            "TRAIN_PROMPT_ZERO_SHOT_V2": TRAIN_PROMPT_ZERO_SHOT_V2,
            "TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V2": TRAIN_PROMPT_ZERO_SHOT_SYSTEM_V2
        }
        
        if prompt_name in prompt_map:
            return prompt_map[prompt_name]
        
        # 如果不是已知的名称，直接返回原始值（可能是自定义的prompt字符串）
        return prompt_name