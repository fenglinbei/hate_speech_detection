CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
	--served-model-name llama3 \
	--model=/data/liaozijie/hate_speech_detection/models/base/Llama-3-8B-Instruct \
	--trust-remote-code \
	--tensor-parallel-size=1 \
	--port="35003" \
	--max_model_len 8192
