CUDA_VISIBLE_DEVICES="2,3" python -m vllm.entrypoints.openai.api_server \
--served-model-name qwen2.5 \
--model=./models/exps/k_ablation/k6/checkpoint-1446 \
--trust-remote-code \
--tensor-parallel-size=2 \
--port="35003" \
--max_model_len 8192
