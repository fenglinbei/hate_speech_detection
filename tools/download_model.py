from transformers import AutoTokenizer, AutoModel, Qwen2TokenizerFast

tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained("models/Qwen/Qwen3-1.7B")
model = AutoModel.from_pretrained("models/Qwen/Qwen3-1.7B")
print(tokenizer)
messages = [{'content': '你好，你是谁？', 'role': 'system'}, {'content': '你好，你是谁？', 'role': 'user'}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Setting enable_thinking=False disables thinking mode
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model(
    **inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
output_ids = generated_ids
print(generated_ids[0][len(inputs.input_ids[0]):])
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)