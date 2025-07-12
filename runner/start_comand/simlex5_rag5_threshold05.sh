CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name checkpoint-724 \
--model=./models/qwen2.5-7B-instruct-simlex5_rag5_threshold05-trip-noalp \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 10000