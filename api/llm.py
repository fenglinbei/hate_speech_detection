import time
import requests
from loguru import logger
from typing import Tuple, Optional
import sys
sys.path.insert(0, ".")
from utils.error import ApiError
from utils.protocol import UsageInfo

class ApiLLMModel:

    def __init__(
            self,
            model_name: str, 
            api_base: str, 
            api_key: str, 
            temperature: float=0.2, 
            top_p: float=0.1, 
            http_proxy: Optional[str]=None,
            https_proxy: Optional[str]=None,
            system_prompt: Optional[str]=None) -> None:
        
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p

        self.usage_count = UsageInfo()

        self.header = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.system_prompt = system_prompt
        if http_proxy or https_proxy:
            self.proxies = {
                'http': http_proxy,
                'https': https_proxy
            }
        else:
            self.proxies = None

    def update_usage(self, usage: UsageInfo):
        self.usage_count.prompt_tokens += usage.prompt_tokens
        self.usage_count.completion_tokens += usage.completion_tokens
        self.usage_count.total_tokens+= usage.total_tokens

    def chat(self, prompt: str, max_new_tokens: int, temperature: Optional[float]=None) -> Tuple[str, UsageInfo, int]:

        if self.system_prompt is not None:
            messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
        else:
            messages = [
                    {"role": "user", "content": prompt}
                ]
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_new_tokens
            }

        try:
            response = requests.post(url=f"{self.api_base}/chat/completions", json=params, headers=self.header, proxies=self.proxies)
            response.raise_for_status()  # 如果响应状态码不是200，则抛出异常
            # logger.debug(f"LLM server response: {response.json()}")
            response = response.json()

            if response['object'] == 'error':
                error_code = response["code"]
                return None, None, error_code
            else:
                error_code = 0

            content = response["choices"][0]["message"]["content"]
            usage = UsageInfo(**response["usage"])
            self.update_usage(usage)
            return content, usage, error_code
        
        except requests.exceptions.HTTPError as http_err:
            logger.exception(f"HTTP error occurred: {http_err}, Response: {http_err.response.text}")
            error_code = response.status_code
            return None, None, error_code
        except requests.exceptions.RequestException as req_err:
            logger.exception(f"Request error occurred: {req_err}")
            error_code = 500
            return None, None, error_code
        except KeyError as key_err:
            logger.exception(f"Response parsing error: missing key {key_err}, Response: {response.text}")
            error_code = response.status_code
            return None, None, error_code
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}")
            error_code = response.status_code
            return None, None, error_code
 
        
class AliyunApiLLMModel(ApiLLMModel):   

    def __init__(
        self,
        model_name: str, 
        api_base: str, 
        api_key: str, 
        temperature: float=0.2, 
        top_p: float=0.1, 
        system_prompt: Optional[str]=None,
        use_dashscope: bool = False
        ) -> None:
        
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p

        self.usage_count = UsageInfo()

        self.header = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.system_prompt = system_prompt
        self.use_dashscope = use_dashscope

    def update_usage(self, usage: UsageInfo):
        self.usage_count.prompt_tokens += usage.prompt_tokens
        self.usage_count.completion_tokens += usage.completion_tokens
        self.usage_count.total_tokens+= usage.total_tokens

    def chat(self, prompt: str, max_new_tokens: int, temperature: Optional[float]=None) -> Tuple[str, UsageInfo, int]:

        if self.system_prompt is not None:
            messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ] if "deepseek" not in self.model_name else [
                    {"role": "user", "content": prompt}
                ]
        else:
            messages = [
                    {"role": "user", "content": prompt}
                ]
        if not self.use_dashscope:
            params = {
                "model": self.model_name,
                "messages": messages,
                "parameters":
                {
                    "temperature": temperature if temperature is not None else self.temperature,
                    "max_tokens": max_new_tokens,
                    "top_p": self.top_p
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
                    "top_p": self.top_p
                }
                }
        logger.debug(params)

        try:
            if not self.use_dashscope:
                response = requests.post(url=f"{self.api_base}/chat/completions", json=params, headers=self.header)
            else:
                response = requests.post(url=f"{self.api_base}/services/aigc/text-generation/generation", json=params, headers=self.header)
            response.raise_for_status()  # 如果响应状态码不是200，则抛出异常
            logger.debug(f"LLM server response: {response.json()}")
            response_data = response.json()
            error_code = 0

            if not self.use_dashscope:
                content = response_data["choices"][0]["message"]["content"]
                usage = UsageInfo(prompt_tokens=response_data["usage"]["prompt_tokens"], completion_tokens=response_data["usage"]["completion_tokens"], total_tokens=response_data["usage"]["total_tokens"])
            else:
                content = response_data["output"]["choices"][0]["message"]["content"]
                usage = UsageInfo(prompt_tokens=response_data["usage"]["input_tokens"], completion_tokens=response_data["usage"]["output_tokens"], total_tokens=response_data["usage"]["total_tokens"])
            self.update_usage(usage)
            return content, usage, error_code
        
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}, Response: {http_err.response.text}")
            error_code = response.status_code
            return None, None, error_code
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error occurred: {req_err}")
            error_code = 500
            return None, None, error_code
        except KeyError as key_err:
            logger.error(f"Response parsing error: missing key {key_err}, Response: {response.text}")
            error_code = response.status_code
            return None, None, error_code
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            error_code = response.status_code
            return None, None, error_code
        
        
if __name__ == "__main__":
    # model = AliyunApiLLMModel(
    #     model_name="qwen2.5-7b-instruct-ft-202504180131-7f77",
    #     api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    #     api_key="sk-22deaa18dd6b423983d438ccd0aa4a2c",
    # )

    model = AliyunApiLLMModel(
        model_name="qwen2.5-7b-instruct-ft-202504180131-7f77",
        api_base="https://dashscope.aliyuncs.com/api/v1",
        api_key="sk-22deaa18dd6b423983d438ccd0aa4a2c",
        use_dashscope=True
    )

    print(model.chat(prompt="你好，你是谁？", max_new_tokens=1000))