import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
import warnings
warnings.simplefilter('ignore')

from vllm import LLM as VLLModel, SamplingParams
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
        
if __name__ == "__main__":
    model = VLLM(model_path="models/Qwen3-8B-sft-hsd/checkpoint-220", tensor_parallel_size=2)