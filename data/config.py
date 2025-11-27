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
        
        # 训练设置
        training = config_data.get('training_settings', {})
        self.auto_length = training.get('auto_length', False)
        self.max_length = training.get('max_length', 2048)
        self.split_ratio = training.get('split_ratio', 0.9)
        
        # 模型设置
        models = config_data.get('model_settings', {})
        self.srag_model_path = models.get('srag_model_path', './models/base/bge-large-zh-v1.5')
        self.lexicon_model_path = models.get('lexicon_model_path', './models/base/bge-large-zh-v1.5')