import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import AutoModel, Qwen2TokenizerFast, Qwen3ForCausalLM

tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained("models/Qwen3-8B-sft-hsd/checkpoint-180/")
model: Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained("models/Qwen3-8B-sft-hsd/checkpoint-180/")

# tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained("models/Qwen3-1.7B")
# model: Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained("models/Qwen3-1.7B")

print(tokenizer)
print(model)
messages = [{'content': 'Hello who are you?', 'role': 'user'}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Setting enable_thinking=False disables thinking mode
)
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256,
    repetition_penalty=3.0
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parse thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)