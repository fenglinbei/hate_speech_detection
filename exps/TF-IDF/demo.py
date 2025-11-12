import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

from exps.utils.data import *

# 1. 中文文本预处理类
class ChineseTextProcessor:
    def __init__(self, stopwords_path='exps/TF-IDF/stopwords_cn.txt'):
        self.stopwords = self.load_stopwords(stopwords_path)
    
    def load_stopwords(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f])
        except:
            return set()
    
    def clean_text(self, text):
        # 移除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
        return text.strip()
    
    def segment(self, text):
        text = self.clean_text(text)
        words = jieba.cut(text)
        return ' '.join([word for word in words if word not in self.stopwords and len(word) > 1])

# 2. 中文情感分析器（简化版）
class ChineseSentimentAnalyzer:
    def __init__(self, negative_lexicon='chinese_negative_words.txt'):
        self.load_lexicon(negative_lexicon)
    
    def load_lexicon(self, file_path: str):
        raw_dict = load_json(file_path)
        self.categories: list[str] = raw_dict['categories']
        self.dict_datas: list[dict] = raw_dict["terms"]
        self.word2item = {}
        for data in self.dict_datas:
            word = data['term']
            categories = data['category'].split(', ')
            definition = data.get('definition', '')
            self.word2item[word] = {
                'word': word,
                'categories': categories,
                'definition': definition
            }
            for category in categories:
                if category not in self.categories:
                    self.categories.append(category)
    
    def analyze_sentiment(self, text):
        segments = text.split()
        negative_count = sum(1 for word in segments if word in self.negative_words)
        total_words = max(len(segments), 1)  # 避免除零
        return negative_count / total_words  # 返回负面情感强度
    
    def analyze_sentiment(self, text: str) -> list[dict]:
        segments = text.split()
        classified_counter = {}
        total_words = max(len(segments), 1)
        for word in segments:
            if word in self.word2item:
                item = self.word2item[word]
                categories = item['categories']
                for category in categories:
                    if category not in classified_counter:
                        classified_counter[category] = 0
                    classified_counter[category] += 1
        
        analyzed_result = {}
        for category, count in classified_counter.items():
            analyzed_result[category] = count / total_words  # 归一化为频率
        return analyzed_result

# 3. 综合特征提取器
class HateSpeechFeatureExtractor:
    def __init__(self, ngram_range=(1, 3)):
        self.text_processor = ChineseTextProcessor()
        self.sentiment_analyzer = ChineseSentimentAnalyzer()
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=10000,
            min_df=3
        )
    
    def extract_features(self, texts: list[str]):
        # 文本预处理
        processed_texts = [self.text_processor.segment(text) for text in texts]
        
        # 提取TF-IDF特征（N-gram）
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # 提取情感特征
        sentiment_features = [self.sentiment_analyzer.analyze_sentiment(text) for text in processed_texts]
        
        # 合并特征（类似论文中的主题相似性+情感特征）
        import scipy.sparse as sp
        sentiment_features = sp.csr_matrix(sentiment_features).T
        combined_features = sp.hstack([tfidf_features, sentiment_features])
        
        return combined_features

# 4. 主训练流程
def train_hate_speech_detector(train_data_path: str, test_data_path: str):
    train_data = load_json(train_data_path)
    train_texts, train_labels = get_data(train_data_path)

    test_data = load_json(test_data_path)
    test_texts, test_labels = get_data(test_data_path)
    
    # 特征提取
    extractor = HateSpeechFeatureExtractor(ngram_range=(1, 3))
    features = extractor.extract_features(texts)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # 训练模型（使用朴素贝叶斯，与论文一致）
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model, extractor

# 5. 使用示例
if __name__ == "__main__":
    # 示例数据（需要替换为实际仇恨言论数据集）
    example_data = {
        'text': [
            '这个群体应该被彻底清除！',
            '今天的天气真好，适合出游',
            '某些民族就是低劣的种族',
            '欢迎大家和平讨论社会问题'
        ],
        'label': [1, 0, 1, 0]  # 1: 仇恨言论, 0: 正常言论
    }
    
    # 训练检测器
    df = pd.DataFrame(example_data)
    model, extractor = train_hate_speech_detector(df)
    
    # 新文本检测
    test_text = ["这简直是对我们民族的侮辱！"]
    features = extractor.extract_features(test_text)
    prediction = model.predict(features)
    print(f"检测结果: {'仇恨言论' if prediction[0] == 1 else '正常言论'}")