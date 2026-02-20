from flask import Flask, render_template, request, jsonify
from sentiment_model import BertSentimentAnalyzer
from topic_model import BertTopicClassifier, SimpleTopicClassifier
import time
import torch

app = Flask(__name__)

print("🚀 Загрузка BERT моделей...")

# Инициализация моделей
sentiment_analyzer = BertSentimentAnalyzer()

# Пробуем загрузить продвинутую модель, если не получится - используем простую
try:
    topic_classifier = BertTopicClassifier()
    # Проверяем, работает ли модель
    test_result = topic_classifier.classify("тестовое сообщение")
    if test_result['topic'] == 'other' and test_result['confidence'] < 0.6:
        print("⚠️ Продвинутая модель работает нестабильно, переключаюсь на простую...")
        topic_classifier = SimpleTopicClassifier()
    else:
        print("✅ Используется продвинутая BERT модель для тематики")
except Exception as e:
    print(f"⚠️ Ошибка загрузки BERT модели: {e}")
    print("🔄 Использую простой классификатор на правилах")
    topic_classifier = SimpleTopicClassifier()

print("✅ Все модели загружены!")

@app.route('/')
def index():
    """Главная страница"""
    topics = topic_classifier.get_all_topics()
    return render_template('index.html', topics=topics, model_type=type(topic_classifier).__name__)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """API для анализа текста"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Не указан текст'}), 400
    
    text = data['text'].strip()
    
    if not text:
        return jsonify({'error': 'Пустой текст'}), 400
    
    start_time = time.time()
    
    # Анализ тональности через BERT
    sentiment_result = sentiment_analyzer.analyze(text)
    
    # Анализ тематики
    topic_result = topic_classifier.classify(text)
    
    process_time = time.time() - start_time
    
    # Эмодзи для тональности
    emoji_map = {
        'positive': '😊',
        'negative': '😠', 
        'neutral': '😐'
    }
    
    # Формирование ответа
    response = {
        'text': text,
        'sentiment': {
            'label': sentiment_result['sentiment'],
            'confidence': round(sentiment_result['confidence'], 4),
            'emoji': emoji_map.get(sentiment_result['sentiment'], '🤔'),
            'model': 'ruBERT (тональность)',
            'probabilities': sentiment_result.get('probabilities', {})
        },
        'topic': {
            'label': topic_result['topic'],
            'name': topic_result['topic_name'],
            'confidence': round(topic_result['confidence'], 4),
            'model': topic_result.get('model', 'ruBERT'),
            'all_topics': topic_result.get('all_topics', [])
        },
        'processing_time': round(process_time, 3),
        'models': f"Тональность: BERT, Тематика: {topic_result.get('model', 'BERT')}"
    }
    
    return jsonify(response)

@app.route('/test_topics')
def test_topics():
    """Тестовая страница для проверки определения тем"""
    test_texts = [
        "Сегодня прошел финальный матч чемпионата мира по футболу. Сборная Бразилии одержала победу со счетом 2:1.",
        "Apple представила новый iPhone 15 с улучшенной камерой и процессором A17.",
        "В Госдуме приняли новый закон о цифровых технологиях.",
        "Вышел новый фильм Кристофера Нолана. В главных ролях снялись известные актеры.",
        "Цены на нефть выросли на фоне новостей из Саудовской Аравии.",
        "Ученые обнаружили новую экзопланету в зоне обитаемости.",
        "Врачи рекомендуют пить больше воды и заниматься спортом.",
        "В Эрмитаже открылась выставка картин импрессионистов.",
        "В центре Москвы произошло серьезное ДТП с участием трех автомобилей.",
        "Лучшие отели Турции для семейного отдыха на море."
    ]
    
    results = []
    for text in test_texts:
        topic = topic_classifier.classify(text)
        results.append({
            'text': text[:50] + "...",
            'topic': topic['topic_name'],
            'confidence': topic['confidence'],
            'model': topic.get('model', '')
        })
    
    return jsonify(results)

@app.route('/model_info')
def model_info():
    """Информация об используемых моделях"""
    model_type = type(topic_classifier).__name__
    
    info = {
        'sentiment_model': {
            'name': 'ruBERT для тональности',
            'model': 'blanchefort/rubert-base-cased-sentiment',
            'description': 'BERT модель для анализа тональности русских текстов',
            'classes': ['positive', 'negative', 'neutral']
        },
        'topic_model': {
            'name': model_type,
            'description': 'Модель для классификации тематики',
            'topics': topic_classifier.get_all_topics()
        }
    }
    
    if model_type == 'BertTopicClassifier':
        info['topic_model']['model'] = 'Den4ikAI/ruBert-base-finetuned-russian-topic-classification'
    
    return jsonify(info)

@app.route('/demo_examples')
def demo_examples():
    """Примеры для демонстрации"""
    examples = [
        {
            'text': 'Этот фильм просто великолепен! Актерская игра на высоте, сюжет захватывает с первых минут.',
            'expected_sentiment': 'positive',
            'expected_topic': 'entertainment'
        },
        {
            'text': 'Ужасный матч, наша команда провалилась. Защита никакая, вратарь пропустил три глупых гола.',
            'expected_sentiment': 'negative',
            'expected_topic': 'sports'
        },
        {
            'text': 'В Госдуме обсуждают новый законопроект о цифровых технологиях. Депутаты планируют принять его до конца месяца.',
            'expected_sentiment': 'neutral',
            'expected_topic': 'politics'
        },
        {
            'text': 'Apple представила новый iPhone с потрясающей камерой и невероятной производительностью.',
            'expected_sentiment': 'positive',
            'expected_topic': 'technology'
        },
        {
            'text': 'Ученые из МГУ разработали новый метод лечения рака с помощью наночастиц.',
            'expected_sentiment': 'positive',
            'expected_topic': 'science'
        },
        {
            'text': 'Вчера в центре города произошло серьезное ДТП, три человека пострадали.',
            'expected_sentiment': 'negative',
            'expected_topic': 'incidents'
        },
        {
            'text': 'Рецепт вкусного борща: свекла, капуста, морковь и секретный ингредиент.',
            'expected_sentiment': 'neutral',
            'expected_topic': 'food'
        }
    ]
    return jsonify({'examples': examples})

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 ЗАПУСК BERT-АНАЛИЗАТОРА")
    print("=" * 60)
    print("📊 Модели:")
    print("  - Тональность: ruBERT (blanchefort/rubert-base-cased-sentiment)")
    print(f"  - Тематика: {type(topic_classifier).__name__}")
    if hasattr(topic_classifier, 'model_name'):
        print(f"    Модель: {topic_classifier.model_name}")
    print("=" * 60)
    print("🌐 Откройте в браузере: http://localhost:5000")
    print("📝 Для теста тем: http://localhost:5000/test_topics")
    print("=" * 60)
    
    # Очищаем кэш GPU если есть
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    app.run(debug=True, port=5000, threaded=False)