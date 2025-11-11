import json

from exps.utils.data import *

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


def evaluate_target_group_model(classifier: DictClassfier, test_data_path: str, output_path: str="exps/bert/results.json"):
    texts, labels = get_data(test_data_path)

    evaluation_results = run(test_data_path, classifier, binary_label=False)
    saved_results = inference.save_results_to_json(evaluation_results, output_path)
    return saved_results

def evaluate_binary_model(classifier: DictClassfier, test_data_path: str, output_path: str="exps/bert/binary_results.json"):
    texts, labels = get_data_with_binary_label(test_data_path)

    print("Binary labels loaded. Sample labels:", labels[:10])

    evaluation_results = inference.evaluate_with_f1(texts, labels, binary_label=True)
    saved_results = inference.save_results_to_json(evaluation_results, output_path)
    return saved_results

if __name__ == "__main__":
    dict_classifier = DictClassfier("data/lexicon/annotated_lexicon.json")
    results = run("data/full/std/test.json", dict_classifier, binary_label=False)
    print(json.dumps(results[:10], ensure_ascii=False, indent=2))
