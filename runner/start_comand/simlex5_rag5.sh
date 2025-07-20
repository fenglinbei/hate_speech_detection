CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name checkpoint-724 \
--model=./models/qwen2.5-7B-instruct-simlex5-rag5-trip-noalp/checkpoint-720 \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 10000