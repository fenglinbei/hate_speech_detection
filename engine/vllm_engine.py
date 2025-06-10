import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.simplefilter('ignore')

from engine.vllm_engine import LLM as VLLModel, SamplingParams
from typing import List, Dict, Any, Optional
from utils.protocol import UsageInfo

class VLLM:
    """
    用于VLLM推理的模型类，接口与ApiLLMModel一致
    """
    def __init__(
            self, 
            model_path: str, 
            tensor_parallel_size: int = 1,
            max_num_seqs: int = 32, 
            max_model_len: int = 8192 * 3 // 2,
            gpu_memory_utilization: float = 0.95, 
            seed: int = 2024):
        
        self.model_name = model_path
        self.llm = VLLModel(
            model=model_path,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
        )
        self.tokenizer = self.llm.get_tokenizer()
    
    def chat(
            self, 
            messages: str, 
            max_new_tokens: int = 1024, 
            n: int = 1,
            top_p: float = 1.0, 
            top_k: int = -1, 
            temperature: float = 0.0,
            enable_thinking: bool = False
            ) -> tuple[List[List[str]], UsageInfo, int]:
        """
        执行VLLM推理，接口与ApiLLMModel.chat保持一致
        """
        prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        
        # 创建采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k != -1 else None,  # -1表示禁用top_k
            max_tokens=max_new_tokens,
            n=n,
            stop_token_ids=[self.tokenizer.eos_token_id],
        )
        
        try:
            # 执行推理
            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0]  # 只处理一个prompt
            
            # 提取生成的文本
            generated_texts = [[out.text] for out in output.outputs]
            
            # 计算token使用量
            prompt_token_ids = output.prompt_token_ids
            completion_token_ids = output.outputs[0].token_ids
            prompt_tokens = len(prompt_token_ids)
            completion_tokens = len(completion_token_ids)
            
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            return generated_texts, usage, 200
        
        except Exception as e:
            # 错误处理
            error_text = f"VLLM generation error: {str(e)}"
            return [[error_text]], UsageInfo(), 500


# ==================== 修改后处理逻辑 ====================
import re
import difflib
import random

class FewShotLLMTester:
    # ... 保持原有代码不变 ...
    
    def _process_item(self, item: dict, llm_params: dict) -> Optional[Dict[str, Any]]:
        """修改后处理逻辑为知识点标注任务"""
        # ... 保持原有代码直到获取answer变量 ...
        
        # 提取预测结果（知识点）
        prediction = self.extract_knowledge_label(answer)
        # 提取真实标签
        solution = extract_answer_from_dataset(item.get("solution", ""))
        
        # 处理标签匹配
        label_list = self.config.get('label_list', [])
        if prediction and label_list:
            prediction = prediction.strip()
            if prediction in label_list:
                valid_pred = prediction
            else:
                # 尝试模糊匹配
                close_match = difflib.get_close_matches(prediction, label_list, n=1, cutoff=0.5)
                valid_pred = close_match[0] if close_match else random.choice(label_list)
        else:
            valid_pred = random.choice(label_list) if label_list else "Unknown"
        
        # 返回处理结果
        return {
            **item,
            "llm_output": answer,
            "pred_label": valid_pred,
            "true_label": solution,
            "status": "success" if prediction else "invalid"
        }
    
    def extract_knowledge_label(self, text: str) -> Optional[str]:
        """从模型输出中提取知识点标签"""
        if not text:
            return None
        # 尝试匹配boxed内容
        matches = re.findall(r"oxed{(.*?)}", text)
        if matches:
            return matches[-1].strip()
        # 尝试匹配直接给出的标签
        return text.strip().split('\n')[-1].strip()

# ==================== 工具函数 ====================
def extract_answer_from_dataset(text: str) -> str:
    """处理真实标签"""
    return text.strip().replace(',', '') if text else ""