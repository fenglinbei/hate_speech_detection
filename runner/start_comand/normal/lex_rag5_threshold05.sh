CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name checkpoint-722 \
--model=./models/qwen2.5-7B-instruct-lex_rag5_threshold05-trip-noalp/checkpoint-722 \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 10000