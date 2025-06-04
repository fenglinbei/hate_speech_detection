from transformers import AutoTokenizer, AutoModel, Qwen2TokenizerFast
from transformers import AutoTokenizer, AutoModel, Qwen2TokenizerFast



tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained("models/Qwen3-8B")
print(tokenizer)
messages = [{'content': '你好，你是谁？', 'role': 'system'}, {'content': '你好，你是谁？', 'role': 'user'}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Setting enable_thinking=False disables thinking mode
)
inputs = tokenizer(text).input_ids

print([text])
print(inputs)
# model = AutoModel.from_pretrained("models/Qwen3-8B")

# print(model)
# print(model)