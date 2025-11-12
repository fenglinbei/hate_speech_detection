import os
import json
import torch
import random
from datetime import datetime
from transformers import BertForSequenceClassification, BertTokenizer

from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

from exps.utils.data import *
from exps.utils.calculator import F1Calculator

random.seed(23333333)
torch.manual_seed(23333333)

class RoBertFusion(nn.Module):
    def __init__(self, Robert_model, gru_hidden_size=128, num_filters=100, kernel_sizes=[3, 4, 5], n_classes=2):
        super(RoBertFusion, self).__init__()
        self.bert = BertModel.from_pretrained(Robert_model)
        
        self.gru = nn.GRU(self.bert.config.hidden_size, 
                          gru_hidden_size, 
                          bidirectional=True, 
                          batch_first=True)
        
        self.textcnn = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, self.bert.config.hidden_size)) 
            for k in kernel_sizes
        ])

        self.fc = nn.Linear(gru_hidden_size * 2 + num_filters * len(kernel_sizes), n_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        gru_outputs, _ = self.gru(last_hidden_state)
        gru_outputs = gru_outputs[:, -1, :]

        x = last_hidden_state.unsqueeze(1)
        cnn_outputs = [torch.relu(conv(x)).squeeze(3) for conv in self.textcnn]
        cnn_outputs = [torch.max(out, dim=2)[0] for out in cnn_outputs]
        cnn_outputs = torch.cat(cnn_outputs, dim=1)

        combined_features = torch.cat((gru_outputs, cnn_outputs), dim=1)

        logits = self.fc(combined_features)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits


class BertInference:
    def __init__(self, model_path, max_length=512, pth_model: bool=False, tokenizer_path: str=None):
        self.pth_model = pth_model
        if pth_model:
            self.model = RoBertFusion(tokenizer_path, n_classes=6)
            pretrained_dict = torch.load(model_path)
            self.model.load_state_dict(pretrained_dict, strict=False)
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        else:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict_single(self, text):
        """对单条文本进行预测"""
        # 文本预处理
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.pth_model:
            inputs.pop("token_type_ids")
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "text": text,
            "predicted_label": predicted_class,
            "label_description": label_dict_inv.get(predicted_class, "未知"),
            "confidence": confidence,
            # "all_probabilities": probabilities.cpu().numpy()[0]
        }
    
    def predict_batch(self, texts, batch_size=32, binary_label=False):
        """批量预测"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 批量编码
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            if self.pth_model:
                inputs.pop("token_type_ids")
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits if not self.pth_model else outputs
                probabilities = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(logits, dim=-1).cpu().numpy()
                confidences = torch.max(probabilities, dim=-1)[0].cpu().numpy()
            
            if not binary_label:
                for j, text in enumerate(batch_texts):
                    results.append({
                        "text": text,
                        "predicted_label": int(predicted_classes[j]),
                        "label_description": label_dict_inv.get(int(predicted_classes[j]), "未知"),
                        "confidence": float(confidences[j]),
                        # "all_probabilities": probabilities[j].cpu().numpy()
                    })
            else:
                for j, text in enumerate(batch_texts):
                    raw_predicted_label = int(predicted_classes[j])
                    if raw_predicted_label in [0, 1, 2, 3, 4]:
                        predicted_label = 0
                    else:
                        predicted_label = 1  # non-hate
                    results.append({
                        "text": text,
                        "predicted_label": predicted_label,
                        "label_description": binary_label_dict_inv.get(int(predicted_classes[j]), "未知"),
                        "confidence": float(confidences[j]),
                        # "all_probabilities": probabilities[j].cpu().numpy()
                    })
        return results
    
def evaluate_target_group_model(inference: BertInference, calculator: F1Calculator, test_data_path: str, output_path: str="exps/bert/results.json"):
    texts, labels = get_data(test_data_path)
    
    predictions = inference.predict_batch(texts, binary_label=False)
    results = calculator.get_f1(predictions, labels, average='macro', binary_label=False)
    saved_results = calculator.save_results_to_json(results, output_path)
    return saved_results

def evaluate_binary_model(inference: BertInference, calculator: F1Calculator, test_data_path: str, output_path: str="exps/bert/binary_results.json"):
    texts, labels = get_data_with_binary_label(test_data_path)

    predictions = inference.predict_batch(texts, binary_label=True)
    results = calculator.get_f1(predictions, labels, average='macro', binary_label=True)
    saved_results = calculator.save_results_to_json(results, output_path)
    return saved_results

# 使用示例
if __name__ == "__main__":
    # 初始化推理器
    # inference = BertInference('./models/bert-finetuned/checkpoint-2409')
    inference = BertInference(model_path="models/roberta-chsd-new/best_model.pth", pth_model=True, tokenizer_path="./models/chinese-roberta-wwm-ext")
    calculator = F1Calculator()

    evaluate_target_group_model(inference, calculator, 'data/full/std/test.json', "exps/RoBERTa-CHSD/results.json")
    evaluate_binary_model(inference, calculator, 'data/full/std/test.json', "exps/RoBERTa-CHSD/binary_results.json")