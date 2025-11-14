CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name glm3 \
--model=./models/exps/before/ChatGLM3-6B \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 8192