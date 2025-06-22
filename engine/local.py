import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
import warnings
warnings.simplefilter('ignore')

import torch
from transformers import AutoModel, Qwen2TokenizerFast, Qwen3ForCausalLM, Qwen2ForCausalLM
from typing import List, Dict, Any, Optional
from utils.protocol import UsageInfo

class VLLM:
    """
    用于VLLM推理的模型类，接口与ApiLLMModel一致
    """
    def __init__(
            self, 
            model_path: str, 
            model_name: str,
            
            tensor_parallel_size: int = 1,
            max_num_seqs: int = 32, 
            max_model_len: int = 8192 * 3 // 2,
            gpu_memory_utilization: float = 0.95, 
            seed: int = 2024):
        
        from vllm import LLM as VLLModel, SamplingParams
        
        self.model_path = model_path
        self.model_name = model_name
        print(os.environ["CUDA_VISIBLE_DEVICES"])
        import torch
        torch.cuda.set_device(1) 
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
        
class LLM:

    def __init__(
            self, 
            model_path: str,
            model_name: str,
            device_map: dict | str = "auto",
            **kwarg
            ):
        
        self.model_path = model_path
        self.model_name = model_name
        
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        if "qwen3" in model_name:
            self.model = Qwen3ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map)
        else:
            self.model = Qwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map)
        self.model.eval()
    
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

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 扩展输入以适应n个候选结果
        model_inputs = self.tokenizer([text] * n, return_tensors="pt", padding=True).to("cuda")
        attention_mask = model_inputs['attention_mask']
        
        # 准备generate参数
        generate_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0 or top_p < 1.0 or top_k > 0,  # 当需要采样时启用
            "num_return_sequences": 1  # 因为通过批次维度控制n
        }
        
        # 添加可选采样参数
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        if top_p < 1.0:
            generate_kwargs["top_p"] = top_p
        if top_k > 0:
            generate_kwargs["top_k"] = top_k

        generated_ids = self.model.generate(**generate_kwargs)

        # 分离每个生成的序列
        responses = []
        for i in range(n):
            # 计算每个样本的起始位置
            start_index = i * generate_kwargs["input_ids"].size(0) // n
            output_seq = generated_ids[start_index]
            
            # 跳过输入部分只取生成的token
            input_length = model_inputs.input_ids.size(1)
            gen_tokens = output_seq[input_length:]
            
            # 解码并添加到结果
            response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            responses.append(response)
        
        # 计算token使用量 (基于单个输入)
        prompt_tokens = len(model_inputs.input_ids[0])
        completion_tokens = max(len(r) for r in responses)  # 取最长生成的token数
        
        usage = UsageInfo(
            prompt_tokens=prompt_tokens * n,  # 实际输入token * n
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens * n + completion_tokens
        )
        
        # 包装结果以适应n>1的格式
        return [[r] for r in responses], usage, 200
        
        
        
if __name__ == "__main__":
    model = VLLM(model_path="models/Qwen3-8B-sft-hsd/checkpoint-220", tensor_parallel_size=2)