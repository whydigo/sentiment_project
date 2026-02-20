from sentiment_model import BertSentimentAnalyzer
from topic_model import BertTopicClassifier

sentiment = BertSentimentAnalyzer()
topic = BertTopicClassifier()

test_texts = [
    "Я обожаю этот фильм, он просто потрясающий!",
    "Ужасная погода сегодня, всё отменяется.",
    "Президент подписал новый закон о налогах."
]

for text in test_texts:
    s = sentiment.analyze(text)
    t = topic.classify(text)
    print(f"Текст: {text[:30]}...")
    print(f"Тональность: {s['sentiment']} ({s['confidence']:.2f})")
    print(f"Тема: {t['topic_name']} ({t['confidence']:.2f})\n")