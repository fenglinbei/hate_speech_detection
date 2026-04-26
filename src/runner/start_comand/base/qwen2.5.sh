CUDA_VISIBLE_DEVICES="0,1,2,3" python -m vllm.entrypoints.openai.api_server \
	--served-model-name qwen2.5 \
	--model=models/base/Qwen2.5-7B-Instruct \
	--trust-remote-code \
	--tensor-parallel-size=4 \
	--port="35000" \
	--max_model_len 8192
