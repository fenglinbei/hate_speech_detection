CUDA_VISIBLE_DEVICES="1" python -m vllm.entrypoints.openai.api_server \
--served-model-name glm4 \
--model=/data/liaozijie/hate_speech_detection/models/glm_4_9b_chat_ours_prompt_al1280/checkpoint_1444 \
--trust-remote-code \
--tensor-parallel-size=1 \
--port="35003" \
--max_model_len 10000