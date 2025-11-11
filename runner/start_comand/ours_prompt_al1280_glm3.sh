CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name glm3 \
--model=/data/liaozijie/hate_speech_detection/models/chatglm3_6b_ours_prompt_al1280/checkpoint_1444 \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 8192