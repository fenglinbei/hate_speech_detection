CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name ds \
--model=./models/model_comparison/before/ds/checkpoint-1446 \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 4096
