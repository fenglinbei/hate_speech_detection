CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name qwen3 \
--model=./models/exps/before/Qwen3-8B/checkpoint-1446 \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 8192