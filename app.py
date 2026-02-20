from flask import Flask, render_template, request, jsonify, send_file, session
from sentiment_model import BertSentimentAnalyzer
from topic_model import BertTopicClassifier, SimpleTopicClassifier
from excel_processor import ExcelProcessor, BatchAnalyzer
import time
import torch
import os
import uuid
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-2024'  # Для сессий
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB макс размер файла
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOAD_FOLDER'] = 'downloads'

# Создаем папки если их нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

print("🚀 Загрузка BERT моделей...")

# Инициализация моделей
sentiment_analyzer = BertSentimentAnalyzer()

# Пробуем загрузить продвинутую модель для тематики
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

# Инициализация процессора Excel и менеджера заданий
excel_processor = ExcelProcessor(sentiment_analyzer, topic_classifier)
batch_analyzer = BatchAnalyzer(excel_processor)

print("✅ Все модели загружены!")

@app.route('/')
def index():
    """Главная страница с одиночным анализом"""
    topics = topic_classifier.get_all_topics()
    return render_template('index.html', topics=topics, model_type=type(topic_classifier).__name__)

@app.route('/upload')
def upload_page():
    """Страница загрузки Excel"""
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """API для анализа текста (одиночный режим)"""
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

@app.route('/upload_excel', methods=['POST'])
def upload_excel():
    """Загрузка Excel файла"""
    print("=" * 50)
    print("🔄 Начало загрузки файла")
    print("=" * 50)
    
    try:
        # Проверяем наличие файла
        if 'file' not in request.files:
            print("❌ Ошибка: Файл не найден в request.files")
            return jsonify({'error': 'Файл не найден'}), 400
        
        file = request.files['file']
        print(f"📄 Имя файла: {file.filename}")
        
        if file.filename == '':
            print("❌ Ошибка: Имя файла пустое")
            return jsonify({'error': 'Файл не выбран'}), 400
        
        # Проверяем расширение
        if not excel_processor.allowed_file(file.filename):
            print(f"❌ Ошибка: Неподдерживаемый формат {file.filename}")
            return jsonify({'error': 'Неподдерживаемый формат файла. Используйте .xlsx, .xls или .csv'}), 400
        
        # Сохраняем файл
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        print(f"💾 Сохраняем файл в: {filepath}")
        file.save(filepath)
        print("✅ Файл сохранен")
        
        # Проверяем размер файла
        file_size = os.path.getsize(filepath)
        print(f"📊 Размер файла: {file_size} байт")
        
        # Читаем файл для получения информации о колонках
        print("🔄 Читаем файл...")
        try:
            if filename.endswith('.csv'):
                print("📄 Определен как CSV")
                df = pd.read_csv(filepath, encoding='utf-8')
            else:
                print("📄 Определен как Excel")
                # Пробуем разные движки
                try:
                    df = pd.read_excel(filepath, engine='openpyxl')
                except:
                    print("⚠️ openpyxl не сработал, пробуем xlrd")
                    df = pd.read_excel(filepath, engine='xlrd')
            
            print(f"✅ Файл прочитан. Колонок: {len(df.columns)}, строк: {len(df)}")
            print(f"📋 Названия колонок: {df.columns.tolist()}")
            
        except Exception as e:
            print(f"❌ Ошибка чтения файла: {str(e)}")
            return jsonify({'error': f'Ошибка чтения файла: {str(e)}'}), 500
        
        # Сохраняем в сессии
        session['current_file'] = filepath
        session['original_filename'] = filename
        print("✅ Данные сохранены в сессии")
        
        # Создаем список колонок с буквами
        columns_with_letters = []
        for i, col in enumerate(df.columns.tolist()):
            # Преобразуем индекс в букву Excel (0->A, 1->B, 2->C, etc.)
            try:
                letter = excel_processor.get_column_letter(i)
            except:
                # Если функция не работает, используем простой способ
                letter = chr(65 + i) if i < 26 else f"Z{i-25}"  # A, B, C... Z, AA, AB...
            
            columns_with_letters.append({
                'index': i,
                'letter': letter,
                'name': str(col),
                'display': f"Колонка {letter} - {col}"
            })
        
        # Подготавливаем предпросмотр
        preview_rows = []
        for i in range(min(5, len(df))):
            row = []
            for val in df.iloc[i].values:
                # Обрезаем длинные значения
                str_val = str(val) if pd.notna(val) else ""
                if len(str_val) > 50:
                    str_val = str_val[:50] + "..."
                row.append(str_val)
            preview_rows.append(row)
        
        print("✅ Отправляем ответ клиенту")
        return jsonify({
            'success': True,
            'columns': columns_with_letters,
            'preview': {
                'columns': df.columns.tolist(),
                'rows': preview_rows
            }
        })
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/start_batch_analysis', methods=['POST'])
def start_batch_analysis():
    """Запуск пакетного анализа"""
    data = request.get_json()
    
    # Может прийти как индекс, так и буква колонки
    column_value = data.get('column', 0)
    options = data.get('options', {})
    
    # Преобразуем букву в индекс если нужно
    if isinstance(column_value, str) and column_value.isalpha():
        column_index = excel_processor.letter_to_index(column_value.upper())
    else:
        column_index = int(column_value)
    
    filepath = session.get('current_file')
    original_filename = session.get('original_filename')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Файл не найден'}), 400
    
    try:
        # Читаем файл
        df, texts, column_name = excel_processor.read_excel(
            filepath, 
            column_index=column_index
        )
        
        # Получаем букву колонки для информации
        column_letter = excel_processor.get_column_letter(column_index)
        
        # Создаем задание
        job_id = batch_analyzer.create_job(
            filepath, 
            original_filename,
            {
                'column_name': column_name, 
                'column_index': column_index,
                'column_letter': column_letter,
                'options': options
            }
        )
        
        # Запускаем анализ в фоне
        import threading
        thread = threading.Thread(
            target=process_batch_job,
            args=(job_id, df, texts, column_name, options)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'job_id': job_id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_batch_job(job_id, df, texts, column_name, options):
    """Обработка пакетного задания в фоне"""
    try:
        # Обновляем статус
        batch_analyzer.update_job_progress(job_id, 0, len(texts))
        
        # Функция обратного вызова для обновления прогресса
        def progress_callback(current, total):
            batch_analyzer.update_job_progress(job_id, current, total)
        
        # Анализируем тексты
        results = excel_processor.analyze_batch(texts, progress_callback)
        
        # Создаем результирующий DataFrame
        result_df = excel_processor.create_result_dataframe(df, column_name, results, options)
        
        # Сохраняем результат
        output_path, output_filename = excel_processor.save_to_excel(
            result_df, 
            batch_analyzer.get_job(job_id)['original_filename'],
            app.config['DOWNLOAD_FOLDER']
        )
        
        # Завершаем задание
        batch_analyzer.complete_job(job_id, output_path, output_filename)
        
    except Exception as e:
        batch_analyzer.fail_job(job_id, str(e))
        print(f"Ошибка в задании {job_id}: {e}")

@app.route('/job_status/<job_id>')
def job_status(job_id):
    """Получение статуса задания"""
    job = batch_analyzer.get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Задание не найдено'}), 404
    
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'total': job['total'],
        'error': job.get('error'),
        'result_filename': job.get('result_filename')
    })

@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание файла с результатами"""
    filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Файл не найден'}), 404
    
    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

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
        }
    ]
    return jsonify({'examples': examples})

@app.route('/batch_status')
def batch_status():
    """Статус всех заданий"""
    batch_analyzer.cleanup_old_jobs()
    
    return jsonify({
        'active_jobs': len(batch_analyzer.jobs),
        'jobs': [
            {
                'id': j['id'],
                'status': j['status'],
                'progress': f"{j['progress']}/{j['total']}",
                'filename': j['original_filename']
            }
            for j in batch_analyzer.jobs.values()
        ]
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 ЗАПУСК BERT-АНАЛИЗАТОРА С ПАКЕТНОЙ ОБРАБОТКОЙ")
    print("=" * 60)
    print("📊 Модели:")
    print("  - Тональность: ruBERT (blanchefort/rubert-base-cased-sentiment)")
    print(f"  - Тематика: {type(topic_classifier).__name__}")
    print("=" * 60)
    print("🌐 Одиночный анализ: http://localhost:5000")
    print("📊 Загрузка Excel: http://localhost:5000/upload")
    print("📝 Тест тем: http://localhost:5000/test_topics")
    print("=" * 60)
    
    # Очищаем кэш GPU если есть
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    app.run(debug=True, port=5000, threaded=True)