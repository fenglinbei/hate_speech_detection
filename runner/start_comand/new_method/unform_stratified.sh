CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name qwen2.5 \
--model=./models/exps/new_method/unform_stratified/checkpoint-1444 \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 8192