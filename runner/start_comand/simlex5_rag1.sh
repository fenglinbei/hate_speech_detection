CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name checkpoint-724 \
--model=./models/qwen2.5-7B-instruct-simlex5-rag1-trip-noalp/checkpoint-724 \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 10000