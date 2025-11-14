CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name glm4 \
--model=./models/exps/before/GLM-4-9B-Chat \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 8192