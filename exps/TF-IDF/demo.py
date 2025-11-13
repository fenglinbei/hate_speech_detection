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
    def __init__(self, negative_lexicon='data/lexicon/annotated_lexicon.json'):
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
    
    def _analyze_sentiment(self, text):
        segments = text.split()
        negative_count = sum(1 for word in segments if word in self.negative_words)
        total_words = max(len(segments), 1)  # 避免除零
        return negative_count / total_words  # 返回负面情感强度
    
    def analyze_sentiment(self, text: str, binary: bool = False) -> list[dict]:
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
        
        if binary:
            hate_count = sum(count for category, count in classified_counter.items() if category != 'non-hate')
            non_hate_count = classified_counter.get('non-hate', 0)
            return {
                "hate": hate_count / total_words,
                "non-hate": non_hate_count / total_words
            }
        
        analyzed_result = {
            "Racism": 0,
            "Region": 0,
            "LGBTQ": 0,
            "Sexism": 0,
            "others": 0,
            }
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
    
        self.sentiment_categories = None

    def extract_features(self, texts: list[str], binary: bool = False):
        # 文本预处理
        processed_texts = [self.text_processor.segment(text) for text in texts]
        
        # 提取TF-IDF特征（N-gram）
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # 提取情感特征
        sentiment_features = []
        
        for text in processed_texts:
            # 获取情感分析结果（字典格式）
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(text, binary=binary)
            
            # 如果是第一次处理，初始化情感类别顺序
            if self.sentiment_categories is None:
                self.sentiment_categories = sorted(sentiment_result.keys())
            
            # 将字典转换为固定顺序的数值列表
            feature_vector = [sentiment_result.get(category, 0) for category in self.sentiment_categories]
            sentiment_features.append(feature_vector)
        
        # print(sentiment_features)
        
        # 合并特征（类似论文中的主题相似性+情感特征）
        import scipy.sparse as sp
        sentiment_features = sp.csr_matrix(sentiment_features)
        # print(sentiment_features.shape, tfidf_features.shape)
        combined_features = sp.hstack([tfidf_features, sentiment_features])
        
        return combined_features

# 4. 主训练流程
def train_hate_speech_detector(train_data_path: str, test_data_path: str, binary: bool = False):
    if not binary:
        train_texts, train_labels = get_data(train_data_path)
        test_texts, test_labels = get_data(test_data_path)
    else:
        train_texts, train_labels = get_data_with_binary_label(train_data_path)
        test_texts, test_labels = get_data_with_binary_label(test_data_path)

    test_data_len = len(test_texts)
    all_datas = train_texts + test_texts
    all_labels = train_labels + test_labels

    # 特征提取
    extractor = HateSpeechFeatureExtractor(ngram_range=(1, 3))
    features = extractor.extract_features(all_datas, binary=binary)

    X_train, X_test, y_train, y_test = train_test_split(
        features, all_labels, test_size=test_data_len / len(all_datas), shuffle=False
    )
    
    # 训练模型（使用朴素贝叶斯，与论文一致）
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4, target_names=list((label_dict if not binary else binary_label_dict).keys())))
    
    return model, extractor

# 5. 使用示例
if __name__ == "__main__":
    model, extractor = train_hate_speech_detector("data/full/std/train.json", "data/full/std/test.json", binary=True)