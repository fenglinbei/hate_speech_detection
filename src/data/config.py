import json
from prompt import *

class Config:
    """配置类，用于管理所有参数[5,7](@ref)"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """从JSON文件加载配置[1,4](@ref)"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 路径配置
        paths = config_data.get('data_paths', {})
        self.raw_data_path = paths.get('raw_data_path')
        self.val_data_path = paths.get('val_data_path', None)
        self.test_data_path = paths.get('test_data_path')
        self.train_output_path = paths.get('train_output_path')
        self.val_output_path = paths.get('val_output_path')
        self.test_output_path = paths.get('test_output_path')
        self.lexicon_data_path = paths.get('lexicon_data_path')
        self.tokenizer_path = paths.get('tokenizer_path', None)
        
        # 提示模板配置
        templates = config_data.get('prompt_templates', {})
        self.prompt_template = getattr(__import__('prompt'), templates.get('prompt_template', 'RAG_PROMPT_USER_V2'))
        self.example_template = getattr(__import__('prompt'), templates.get('example_template', 'RAG_PROMPT_EXAMPLE_V2'))
        self.system_prompt = getattr(__import__('prompt'), templates.get('system_prompt', 'DEFAULT_SYSTEM_PTOMPT_EN'))
        
        # 检索设置
        retrieval = config_data.get('retrieval_settings', {})
        self.use_srag = retrieval.get('use_srag', False)
        self.stratified = retrieval.get('stratified', True)
        self.srag_top_k = retrieval.get('srag_top_k', 1)
        self.srag_threshold = retrieval.get('srag_threshold', 0)
        self.use_lex = retrieval.get('use_lex', False)
        self.lex_top_k = retrieval.get('lex_top_k', -1)
        self.lex_sim_top_k = retrieval.get('lex_sim_top_k', -1)
        self.lex_sim_threshold = retrieval.get('lex_sim_threshold', 0)
        self.weights = retrieval.get('weights', None)
        self.weights_reverse = retrieval.get('weights_reverse', False)
        self.target_groups = retrieval.get('target_groups', None)
        self.default_weights = retrieval.get('default_weights', None)
        self.clustered = retrieval.get('clustered', False)
        self.n_clusters = retrieval.get('n_clusters', 8)

        self.random_state = retrieval.get('random_state', 42)
        self.ramdom_strategy = retrieval.get('random_strategy', 'none')
        self.random_ratio = retrieval.get('random_ratio', 0.0)
        self.random_temperature = retrieval.get('random_temperature', 1.0)
        self.similarity_alpha = retrieval.get('similarity_alpha', 1)
        self.candidate_multiplier = retrieval.get('candidate_multiplier', 3)

        self.mmr = retrieval.get('mmr', False)
        self.mmr_lambda = retrieval.get('mmr_lambda', 0.75)
        self.mmr_index_path = retrieval.get('mmr_index_path', './cache_retrieval/faiss_hnsw.index')
        self.mmr_docs_path = retrieval.get('mmr_docs_path', './cache_retrieval/doc_store.json')
        self.mmr_cache_dir = retrieval.get('mmr_cache_dir', './cache_retrieval')

        # 全局固定 demos（可选）：所有样本共享同一批示例（例如 demos_k10.json）
        global_demos = config_data.get('global_demo_settings', {})
        self.use_global_demos = global_demos.get('use_global_demos', False)
        self.global_demos_path = global_demos.get('global_demos_path', None)
        self.global_demos_top_k = global_demos.get('global_demos_top_k', -1)
        self.global_demos_shuffle = global_demos.get('global_demos_shuffle', False)
        self.global_demos_seed = global_demos.get('global_demos_seed', 42)
        
        # 训练设置
        training = config_data.get('training_settings', {})
        self.auto_length = training.get('auto_length', False)
        self.max_length = training.get('max_length', 2048)
        self.split_ratio = training.get('split_ratio', 0.9)

        cache = config_data.get('cache_settings', {})
        self.enable_build_cache = cache.get('enable_build_cache', True)
        self.build_cache_dir = cache.get('build_cache_dir', './cache_build_data')
        self.enable_retrieval_cache = cache.get('enable_retrieval_cache', True)
        self.cache_backend = cache.get('cache_backend', 'sqlite')
        self.retrieval_batch_size = cache.get('retrieval_batch_size', 256)
        
        # 模型设置
        models = config_data.get('model_settings', {})
        self.srag_model_path = models.get('srag_model_path', './models/base/bge-large-zh-v1.5')
        self.lexicon_model_path = models.get('lexicon_model_path', './models/base/bge-large-zh-v1.5')
