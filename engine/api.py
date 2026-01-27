import os
import re
import sys
import time
import requests
from loguru import logger
from urllib.parse import urljoin
from typing import Tuple, Optional, Dict, List, Any
from pydantic import validate_call

sys.path.insert(0, ".")
from utils.error import ApiError
from utils.protocol import UsageInfo


def parse_text(text: str) -> Tuple[str, Optional[str]]:
    parts = text.split("</think>")

    if len(parts) == 1:
        return text, None
    
    answer = parts[1].strip()
    thought = parts[0].strip()

    thought = thought.split("<think>")[-1].strip()

    return answer, thought
    

class ApiLLMModel:

    @validate_call
    def __init__(
            self,
            model_name: str, 
            api_base: str, 
            api_key: str, 
            temperature: float=0.8, 
            top_p: float=0.7, 
            top_k: int = 20,
            enable_thinking: bool = False,
            timeout: int = 30,
            http_proxy: Optional[str] = None,
            https_proxy: Optional[str] = None,
            system_prompt: Optional[str] = None) -> None:
        
        if not api_base.startswith(("http://", "https://")):
            raise ValueError("Invalid API base URL")
        
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking
        self.timeout = timeout

        self.usage_count = UsageInfo()

        self.header = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.system_prompt = system_prompt
        self.proxies = self._build_proxies(http_proxy, https_proxy)
    
    def _build_proxies(self, http_proxy, https_proxy):
        proxies = {}
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        return proxies or None

    def _update_usage(self, usage: UsageInfo):
        self.usage_count.prompt_tokens += usage.prompt_tokens
        self.usage_count.completion_tokens += usage.completion_tokens
        self.usage_count.total_tokens+= usage.total_tokens

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def _build_params(
            self, 
            messages: list[dict[str, str]],
            max_new_tokens: int = 1024, 
            n: int = 1,
            top_p: float = 0.8,
            top_k: int = 20,
            temperature: float = 0.7, 
            enable_thinking: bool = False,
            seed: Optional[int] = None
            ) -> dict:
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "top_k": top_k if top_k is not None else self.top_k,
            "max_tokens": max_new_tokens,
            "n": n,
            "chat_template_kwargs": {"enable_thinking": enable_thinking if enable_thinking is not None else self.enable_thinking},
            "seed": seed
        }
        if self.enable_thinking and "openrouter" in self.api_base:
            params["reasoning"] = {"enable": True}
        elif not self.enable_thinking and "openrouter" in self.api_base:
            params["reasoning"] = {"enable": False}
        return params
    
    def _build_url(self) -> str:
        url = urljoin(self.api_base, "chat/completions")
        return url
      
    def _parse_response(self, response_data: dict) -> Tuple[list[Tuple[str, Optional[str]]], UsageInfo]:
        if 'error' in response_data:
            raise ApiError(response_data['error'])
        return (
            [parse_text(choice["message"]["content"]) for choice in response_data["choices"]],
            UsageInfo(**response_data["usage"])
        )

    def chat(
            self, 
            messages: list[dict], 
            max_new_tokens: int, 
            n: int = 1,
            top_p: float = 0.8,
            top_k: int = 20,
            temperature: float = 0.7, 
            enable_thinking: bool = False,
            seed: Optional[int] = None
            ) -> Tuple[Optional[list[Tuple[str, Optional[str]]]], Optional[UsageInfo], int]:
        
        response = None
        status_code = 200

        params = self._build_params(
            messages, 
            temperature=temperature, 
            max_new_tokens=max_new_tokens, 
            enable_thinking=enable_thinking,
            n=n,
            top_p=top_p,
            top_k=top_k,
            seed=seed
            )
        url = self._build_url()

        try:
            response = requests.post(
                url=url, 
                json=params, 
                headers=self.header, 
                proxies=self.proxies,
                timeout=self.timeout)
            response.raise_for_status()
            response_data = response.json()
            status_code = response.status_code
            
            logger.debug(response_data)
            contents, usage = self._parse_response(response_data)
            self._update_usage(usage)

            return contents, usage, status_code
        
        except requests.exceptions.HTTPError as http_err:
            
            status_code = http_err.response.status_code
            response_text = http_err.response.text
            logger.error(f"HTTP Error {status_code}: {response_text}")
            logger.exception(http_err)
            return None, None, status_code
        except requests.exceptions.RequestException as req_err:
            logger.exception(f"Request error occurred: {req_err}")
            status_code = 500
            return None, None, status_code
        except KeyError as key_err:
            response_text = response.text if response else 'No response'
            status_code = response.status_code if response else 500
            logger.exception(...)
            return None, None, status_code
        except Exception as e:
            status_code = getattr(response, 'status_code', 500)
            logger.exception(f"An unexpected error occurred: {e}")
            return None, None, status_code
 
        
class AliyunApiLLMModel(ApiLLMModel):   

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        temperature: float = 0.2,
        top_p: float = 0.1,
        system_prompt: Optional[str] = None,
        use_dashscope: bool = False,
        http_proxy: Optional[str] = None,  # 新增代理参数
        https_proxy: Optional[str] = None
    ) -> None:
        
        super().__init__(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            http_proxy=http_proxy,
            https_proxy=https_proxy,
            system_prompt=system_prompt
        )

        self.use_dashscope = use_dashscope
        self.header = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_messages(self, prompt: str) -> list:
        messages = super()._build_messages(prompt)
        if "deepseek" in self.model_name:
            messages = [msg for msg in messages if msg["role"] != "system"]
        return messages

    def _build_params(
            self, 
            messages: list[dict[str, str]],
            max_new_tokens: int = 1024, 
            n: int = 1,
            top_p: float = 0.8,
            top_k: int = 20,
            temperature: float = 0.7, 
            enable_thinking: bool = False
            ) -> dict:
        
        if not self.use_dashscope:
            params = {
                "model": self.model_name,
                "messages": messages,
                "parameters":
                {
                    "temperature": temperature if temperature is not None else self.temperature,
                    "max_tokens": max_new_tokens,
                    "top_p": top_p if top_p is not None else self.top_p, 
                    "top_k": top_k if top_k is not None else self.top_k,
                    "max_tokens": max_new_tokens,
                    "n": n,
                    "enable_thinking": enable_thinking if enable_thinking is not None else self.enable_thinking
                }
                }
        else:
            params = {
                "model": self.model_name,
                "input": {"messages": messages},
                "parameters":
                {
                    "result_format": "message",
                    "temperature": temperature if temperature is not None else self.temperature,
                    "max_tokens": max_new_tokens,
                    "top_p": top_p if top_p is not None else self.top_p,
                    "top_k": top_k if top_k is not None else self.top_k,
                    "max_tokens": max_new_tokens,
                    "n": n,
                    "enable_thinking": enable_thinking if enable_thinking is not None else self.enable_thinking
                }
                }
        return params
    
    def _build_url(self) -> str:
        if self.use_dashscope:
            url = urljoin(self.api_base, "services/aigc/text-generation/generation")
            return url
        return super()._build_url()
    
    def _parse_response(self, response_data: dict) -> Tuple[str, UsageInfo]:
        if 'code' in response_data and response_data['code'] != 200:
            raise ApiError(response_data['message'])
        
        if self.use_dashscope:
            return (
                response_data["output"]["choices"][0]["message"]["content"],
                UsageInfo(
                    prompt_tokens=response_data["usage"]["input_tokens"],
                    completion_tokens=response_data["usage"]["output_tokens"],
                    total_tokens=response_data["usage"]["total_tokens"]
                )
            )
        return super()._parse_response(response_data)
    

class OpenAIApiLLMModel(ApiLLMModel):
    """
    OpenAI Responses API wrapper (POST /v1/responses).

    - 保持与你现有 ApiLLMModel.chat() 完全一致的签名与返回结构：
      returns (contents: List[Tuple[str, Optional[str]]], usage: UsageInfo|None, status_code: int)

    - Responses API 不支持在一次请求里用 n 生成多个候选；这里通过循环 n 次来兼容。
    """

    @validate_call
    def __init__(
        self,
        model_name: str,
        api_base: str = "https://api.openai.com/v1/",
        api_key: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.7,
        timeout: int = 120,
        http_proxy: Optional[str] = None,
        https_proxy: Optional[str] = None,
        system_prompt: Optional[str] = None,
        store: bool = False,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        reasoning_effort: str = "medium",  # low / medium / high（主要给 gpt-5 / o-series）
    ) -> None:

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Missing OpenAI API key. Provide api_key or set env OPENAI_API_KEY."
            )

        api_base = self._normalize_api_base(api_base)

        # 复用父类的 header / proxies / usage_count 等机制
        super().__init__(
            model_name=model_name,
            api_base=api_base,
            api_key=resolved_key,
            temperature=temperature,
            top_p=top_p,
            top_k=0,  # OpenAI Responses API 无 top_k；占位即可
            enable_thinking=False,  # 父类的 enable_thinking 不直接用；在 build_params 里映射到 reasoning
            timeout=timeout,
            http_proxy=http_proxy,
            https_proxy=https_proxy,
            system_prompt=system_prompt,
        )

        self.store = store
        self.reasoning_effort = reasoning_effort

        # 可选：组织/项目头（OpenAI 平台可用）
        if organization:
            self.header["OpenAI-Organization"] = organization
        if project:
            self.header["OpenAI-Project"] = project

    @staticmethod
    def _normalize_api_base(api_base: str) -> str:
        """
        允许用户传：
        - https://api.openai.com
        - https://api.openai.com/v1
        - https://api.openai.com/v1/
        统一归一到 .../v1/
        """
        base = api_base.rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise ValueError("Invalid API base URL")
        if not base.endswith("/v1"):
            # 若用户已经是 .../v1 以外的兼容服务，请自己显式传完整 api_base
            base = base + "/v1"
        return base + "/"

    @staticmethod
    def _is_reasoning_model(model_name: str) -> bool:
        # OpenAI 官方：reasoning 参数适用于 gpt-5 与 o-series
        return model_name.startswith("gpt-5") or model_name.startswith("o")

    def _build_url(self) -> str:
        # POST {api_base}/responses
        return urljoin(self.api_base, "responses")

    @staticmethod
    def _extract_output_text(resp_json: Dict[str, Any]) -> str:
        """
        尽量稳健地取文本：
        1) 优先使用 resp_json.get("output_text")（SDK 示例用法）
        2) 否则从 output 数组里拼接 message -> content -> output_text
        """
        if isinstance(resp_json.get("output_text"), str) and resp_json["output_text"].strip():
            return resp_json["output_text"].strip()

        outputs = resp_json.get("output", []) or []
        texts: List[str] = []
        for item in outputs:
            if item.get("type") != "message":
                continue
            if item.get("role") != "assistant":
                continue
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text":
                    t = c.get("text", "")
                    if t:
                        texts.append(t)
        return "\n".join(texts).strip()

    def _build_params(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 1024,
        n: int = 1,                # Responses API 不支持一次请求 n；chat() 会循环
        top_p: float = 0.8,
        top_k: int = 20,           # 不支持；保留签名
        temperature: float = 0.7,
        enable_thinking: bool = False,
        seed: Optional[int] = None # 不支持；保留签名
    ) -> dict:

        params: Dict[str, Any] = {
            "model": self.model_name,
            "input": messages,  # Responses API: input 可以是字符串或 message 数组
            "max_output_tokens": max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "store": self.store,
        }

        # 如果传了 system_prompt，并且 messages 里没有 system，就用 instructions 注入（更贴近 Responses API）
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            params["instructions"] = self.system_prompt

        # enable_thinking -> reasoning.effort（仅对 gpt-5 / o-series）
        if enable_thinking and self._is_reasoning_model(self.model_name):
            params["reasoning"] = {"effort": self.reasoning_effort}

        return params

    def _parse_response(self, response_data: dict) -> Tuple[Tuple[str, Optional[str]], UsageInfo]:
        # Responses API response object 里可能有 error 字段（为空或对象）
        if response_data.get("error"):
            raise ApiError(response_data["error"])

        text = self._extract_output_text(response_data)
        if not text:
            # 让上层走重试逻辑
            raise KeyError("Empty output_text")

        usage_raw = response_data.get("usage") or {}
        usage = UsageInfo(
            prompt_tokens=usage_raw.get("input_tokens", 0),
            completion_tokens=usage_raw.get("output_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )
        return parse_text(text), usage

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int,
        n: int = 1,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.7,
        enable_thinking: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[Optional[list[Tuple[str, Optional[str]]]], Optional[UsageInfo], int]:

        url = self._build_url()
        contents_all: list[Tuple[str, Optional[str]]] = []
        merged_usage = UsageInfo()
        status_code = 200

        # Responses API 不支持单次 n；循环模拟 n 次
        loop_n = max(1, int(n))

        for _ in range(loop_n):
            params = self._build_params(
                messages=messages,
                max_new_tokens=max_new_tokens,
                n=1,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                enable_thinking=enable_thinking,
                seed=seed,
            )

            response = None
            try:
                response = requests.post(
                    url=url,
                    json=params,
                    headers=self.header,
                    proxies=self.proxies,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response_data = response.json()
                status_code = response.status_code

                logger.debug(response_data)

                content_one, usage_one = self._parse_response(response_data)
                contents_all.append(content_one)

                self._update_usage(usage_one)
                merged_usage.prompt_tokens += usage_one.prompt_tokens
                merged_usage.completion_tokens += usage_one.completion_tokens
                merged_usage.total_tokens += usage_one.total_tokens

            except requests.exceptions.HTTPError as http_err:
                status_code = http_err.response.status_code
                logger.error(f"HTTP Error {status_code}: {http_err.response.text}")
                logger.exception(http_err)
                return None, None, status_code
            except requests.exceptions.RequestException as req_err:
                logger.exception(f"Request error occurred: {req_err}")
                return None, None, 500
            except Exception as e:
                status_code = getattr(response, "status_code", 500)
                logger.exception(f"An unexpected error occurred: {e}")
                return None, None, status_code

        return contents_all, merged_usage, status_code
