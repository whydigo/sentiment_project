from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np

class BertSentimentAnalyzer:
    def __init__(self):
        """Инициализация BERT модели для анализа тональности на русском"""
        print("Загрузка модели тональности BERT...")
        
        # Используем rubert-tiny для быстроты, можно заменить на более мощную
        self.model_name = "blanchefort/rubert-base-cased-sentiment"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Создаем pipeline для удобства
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            # Маппинг меток
            self.label_map = {
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative', 
                'NEUTRAL': 'neutral'
            }
            
            print("✅ Модель тональности загружена!")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.model = None
            self.tokenizer = None
            self.classifier = None
    
    def analyze(self, text):
        """Анализ тональности текста"""
        if not text or len(text.strip()) < 3:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'probabilities': {},
                'error': 'Текст слишком короткий'
            }
        
        try:
            # BERT анализ
            result = self.classifier(text[:512])[0]  # Обрезаем до 512 токенов
            
            label = result['label']
            score = result['score']
            
            # Получаем вероятности для всех классов
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Создаем словарь вероятностей
            probabilities = {}
            for i, prob in enumerate(probs[0]):
                class_name = self.model.config.id2label[i]
                mapped_name = self.label_map.get(class_name, class_name.lower())
                probabilities[mapped_name] = float(prob)
            
            return {
                'sentiment': self.label_map.get(label, label.lower()),
                'confidence': float(score),
                'probabilities': probabilities,
                'model': 'ruBERT',
                'raw_label': label
            }
            
        except Exception as e:
            print(f"Ошибка анализа: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'probabilities': {},
                'error': str(e)
            }

# Альтернативный вариант с другой моделью (если нужна)
class TinyBertSentimentAnalyzer(BertSentimentAnalyzer):
    def __init__(self):
        """Используем tiny версию для скорости"""
        self.model_name = "cointegrated/rubert-tiny-sentiment-balanced"
        super().__init__()