import json
from modelscope import AutoTokenizer

def get_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False, 
        trust_remote_code=True
    )
    return tokenizer

def is_overlength(tokenizer, text, max_length):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    return len(input_ids) > max_length

if __name__ == "__main__":
    model_path = "models/Qwen3-1.7B"
    tokenizer = get_tokenizer(model_path)
    text = "谁喜欢小黑啊，黑不拉几的看一眼都嫌脏，也就某些连汽车排气口都不放过的男人会喜欢黑妞"
    print(is_overlength(tokenizer, text, 2048))
