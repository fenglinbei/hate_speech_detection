
from loguru import logger
from typing import Optional
from tools.json_tools import load_json
from qwen_gen import QwenGen
from get_prompt import generate_rag_prompt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

from tools.convert import output2triple

class Retriver:

    def __init__(self, model_path: str, model_name: str, data_path: Optional[str]):

        logger.info(f"Loading model from path: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.model_name = model_name

        if data_path:
            self.load_datas(data_path)
            self.create_embeddings()

    def create_embeddings(self, texts: Optional[list[str]] = None):
        logger.info("Processing embedding")
        if not texts:
            texts = self.texts
        corpus_embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        if corpus_embeddings.is_cuda:
            corpus_embeddings = corpus_embeddings.cpu()
        self.corpus_embeddings_np = corpus_embeddings.numpy()

    def load_datas(self, data_path: str):
        data = load_json(data_path)
        self.texts = [item['content'] for item in data]
        self.test2item = {}
        for item in data:
            item['output'] = output2triple(item['output'])
            self.test2item[item['content']] = item


    def retrieve(self, query: str, top_k: int=1) -> tuple[list[str], list[str]]:
        query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)

        if query_embedding.is_cuda:
            query_embedding = query_embedding.cpu()

        query_embedding_np = query_embedding.numpy().reshape(1, -1)

        similarities = cosine_similarity(query_embedding_np, self.corpus_embeddings_np)[0]

        if len(similarities) <= top_k:
            top_k_indices = np.argsort(similarities)[::-1]
        else:
            top_k_indices = np.argsort(similarities)[-top_k:][::-1] # 从大到小排序

        # 返回最相关的文本
        retrieved_texts = [self.texts[idx] for idx in top_k_indices]
        
        return retrieved_texts, [self.test2item[retrieved_text]['output'] for retrieved_text in retrieved_texts]


data = get_json('../data/train.json')
texts = [item['content'] for item in data]




qwen = QwenGen(port=35000, temperature=0.1)
integration_num = 5
threshold = 3

def gen_output(item):
    while True:
        retriver_contents = retriever(texts, item['content'], top_k=integration_num)
        all_responses = []
        for retriver_content in retriver_contents:
            retriver_item = test2item[retriver_content]
            prompt = generate_rag_prompt(item['content'], retriver_item, 'triple')
            result = qwen.response(prompt)
            all_responses.append(result)

        response_counts = Counter(all_responses)
        most_common_list = response_counts.most_common(1)
        actual_most_common_response = most_common_list[0][0]
        count_of_most_common = most_common_list[0][1]

        if count_of_most_common >= threshold:
            return actual_most_common_response


test_data = get_json('../data/test2.json')
with open(f'../data/output/qwen2_7b_instruct_train_rag_triple_integration{integration_num}_{threshold}_t_0.1.txt', 'w', encoding='utf-8') as f:
    for item in tqdm(test_data):
        item['output'] = process_triple(gen_output(item))

        while not check_response(item['output']):
            item['output'] = process_triple(gen_output(item))
            print(item['output'])
        f.write(item['output'] + '\n')



# CUDA_VISIBLE_DEVICES="0,1" python -m vllm.entrypoints.openai.api_server --served-model-name default --model="/data3/zlh/king/CCL2025-final/models/Qwen2___5-7B-Instruct-traindata_train_rag_triple/full/sft" --trust-remote-code --tensor-parallel-size=2 --port="35000" --max_model_len 10000
