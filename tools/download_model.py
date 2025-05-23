from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", cache_dir="models/")
print(tokenizer)
model = AutoModel.from_pretrained("Qwen/Qwen3-8B", cache_dir="models/")

print(model)