CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name llama3 \
--model=/data/liaozijie/hate_speech_detection/models/llama3-8b-ours-prompt-al1280/checkpoint-1444 \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 8192