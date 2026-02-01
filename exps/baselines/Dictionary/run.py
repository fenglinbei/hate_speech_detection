import json

from exps.utils.data import *
from exps.utils.calculator import F1Calculator

class DictClassfier:
    def __init__(self, dict_path: str):
        self.categories = []
        self.word2item = {}
        self.load_dict(dict_path)

    def load_dict(self, file_path):
        raw_dict = load_json(file_path)
        self.categories = raw_dict['categories']
        self.dict_datas = raw_dict["terms"]
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

    def find(self, query: str) -> list[str]:
        result = []
        for word in self.word2item.keys():
            if word in query:
                result.append(self.word2item[word])
        return result

    def classify(self, query: str) -> list[dict]:
        find_items = self.find(query)
        classified_counter = {}
        for item in find_items:
            categories = item['categories']
            for category in categories:
                if category not in classified_counter:
                    classified_counter[category] = 0
                classified_counter[category] += 1
        
        return max(classified_counter.items(), key=lambda x: x[1], default=(None, 0))[0]
    
def run(test_data_file: str, classfier: DictClassfier, binary_label=False):
    results = []
    test_data = load_json(test_data_file)
    for sample in test_data:
        text = sample['content']
        classfied_result = classfier.classify(text)
        
        if classfied_result is None:
            predicted_label = label_dict["non-hate"]
        else:
            predicted_label = label_dict[classfied_result]

        if not binary_label:
            results.append({
                "text": text,
                "predicted_label": predicted_label,
                "label_description": label_dict_inv[predicted_label],
                "confidence": 1,
            })
        else:
            if predicted_label in [0, 1, 2, 3, 4]:
                predicted_label = 0
            else:
                predicted_label = 1 

            results.append({
                "text": text,
                "predicted_label": predicted_label,
                "label_description": binary_label_dict_inv[predicted_label],
                "confidence": 1,
                # "all_probabilities": probabilities[j].cpu().numpy()
            })
    return results


def evaluate_target_group_model(classifier: DictClassfier, calculator: F1Calculator,  test_data_path: str, output_path: str="exps/Dictionary/results.json"):
    texts, labels = get_data(test_data_path)
    results = run(test_data_path, classifier, binary_label=False)
    f1_result = calculator.get_f1(results, labels, average='macro', binary_label=False)
    saved_results = calculator.save_results_to_json(f1_result, output_path)
    return saved_results

def evaluate_binary_model(classifier: DictClassfier, calculator: F1Calculator, test_data_path: str, output_path: str="exps/Dictionary/binary_results.json"):
    texts, labels = get_data_with_binary_label(test_data_path)
    results = run(test_data_path, classifier, binary_label=True)
    f1_result = calculator.get_f1(results, labels, average='macro', binary_label=True)
    saved_results = calculator.save_results_to_json(f1_result, output_path)
    return saved_results

if __name__ == "__main__":
    dict_classifier = DictClassfier("data/lexicon/annotated_lexicon.json")
    calculator = F1Calculator()
    evaluate_target_group_model(dict_classifier, calculator, 'data/full/std/test.json', "exps/Dictionary/results.json")
    evaluate_binary_model(dict_classifier, calculator, 'data/full/std/test.json', "exps/Dictionary/binary_results.json")
