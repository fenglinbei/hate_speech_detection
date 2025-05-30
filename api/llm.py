import sys
import time
import requests
from loguru import logger
from urllib.parse import urljoin
from typing import Tuple, Optional, Dict, List
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
            enable_thinking: bool = False
            ) -> dict:
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "top_k": top_k if top_k is not None else self.top_k,
            "max_tokens": max_new_tokens,
            "n": n,
            "enable_thinking": enable_thinking if enable_thinking is not None else self.enable_thinking
        }
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
            prompt: str, 
            max_new_tokens: int, 
            n: int = 1,
            top_p: float = 0.8,
            top_k: int = 20,
            temperature: float = 0.7, 
            enable_thinking: bool = False
            ) -> Tuple[Optional[list[Tuple[str, Optional[str]]]], Optional[UsageInfo], int]:
        
        response = None
        status_code = 200

        messages = self._build_messages(prompt)
        params = self._build_params(
            messages, 
            temperature=temperature, 
            max_new_tokens=max_new_tokens, 
            enable_thinking=enable_thinking,
            n=n,
            top_p=top_p,
            top_k=top_k
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
        
if __name__ == "__main__":
    # model = AliyunApiLLMModel(
    #     model_name="qwen2.5-7b-instruct",
    #     api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/",
    #     api_key="sk-22deaa18dd6b423983d438ccd0aa4a2c",
    # )
    # model = AliyunApiLLMModel(
    #     model_name="qwen2.5-7b-instruct",
    #     api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/",
    #     api_key="sk-22deaa18dd6b423983d438ccd0aa4a2c",
    # )

    # model = AliyunApiLLMModel(
    #     model_name="qwen2.5-7b-instruct-ft-202504180131-7f77",
    #     api_base="https://dashscope.aliyuncs.com/api/v1/",
    #     api_key="sk-22deaa18dd6b423983d438ccd0aa4a2c",
    #     # use_dashscope=True
    # )

    model = ApiLLMModel(
        model_name="qwen3-4b",
        api_base="http://127.0.0.1:5001/v2/",
        api_key='23333333'
    )

    print(model.chat(
        prompt="你好，你是谁？", 
        n=3,
        max_new_tokens=1024, 
        enable_thinking=True,
        temperature=0.7,
        top_p=0.8,
        top_k=10))
