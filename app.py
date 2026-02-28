from flask import Flask, render_template, request, jsonify, send_file, session
from sentiment_model import BertSentimentAnalyzer
from topic_model import BertTopicClassifier, SimpleTopicClassifier
from excel_processor import ExcelProcessor, BatchAnalyzer
import time
import torch
import os
import uuid
import pandas as pd
import threading
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOAD_FOLDER'] = 'downloads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

# Словарь для отслеживания флагов остановки
stop_flags = {}

print("🚀 Загрузка BERT моделей...")

sentiment_analyzer = BertSentimentAnalyzer()

try:
    topic_classifier = BertTopicClassifier()
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

excel_processor = ExcelProcessor(sentiment_analyzer, topic_classifier)
batch_analyzer = BatchAnalyzer(excel_processor)

print("✅ Все модели загружены!")

@app.route('/')
def index():
    topics = topic_classifier.get_all_topics()
    return render_template('index.html', topics=topics, model_type=type(topic_classifier).__name__)

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Не указан текст'}), 400
    
    text = data['text'].strip()
    
    if not text:
        return jsonify({'error': 'Пустой текст'}), 400
    
    start_time = time.time()
    
    sentiment_result = sentiment_analyzer.analyze(text)
    topic_result = topic_classifier.classify(text)
    
    process_time = time.time() - start_time
    
    emoji_map = {
        'positive': '😊',
        'negative': '😠', 
        'neutral': '😐'
    }
    
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
    print("=" * 50)
    print("🔄 Начало загрузки файла")
    print("=" * 50)
    
    try:
        if 'file' not in request.files:
            print("❌ Ошибка: Файл не найден в request.files")
            return jsonify({'error': 'Файл не найден'}), 400
        
        file = request.files['file']
        print(f"📄 Имя файла: {file.filename}")
        
        if file.filename == '':
            print("❌ Ошибка: Имя файла пустое")
            return jsonify({'error': 'Файл не выбран'}), 400
        
        if not excel_processor.allowed_file(file.filename):
            print(f"❌ Ошибка: Неподдерживаемый формат {file.filename}")
            return jsonify({'error': 'Неподдерживаемый формат файла. Используйте .xlsx, .xls или .csv'}), 400
        
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        print(f"💾 Сохраняем файл в: {filepath}")
        file.save(filepath)
        print("✅ Файл сохранен")
        
        file_size = os.path.getsize(filepath)
        print(f"📊 Размер файла: {file_size} байт")
        
        print("🔄 Читаем файл...")
        try:
            if filename.endswith('.csv'):
                print("📄 Определен как CSV")
                df = pd.read_csv(filepath, encoding='utf-8')
            else:
                print("📄 Определен как Excel")
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
        
        session['current_file'] = filepath
        session['original_filename'] = filename
        print("✅ Данные сохранены в сессии")
        
        columns_with_letters = []
        for i, col in enumerate(df.columns.tolist()):
            try:
                letter = excel_processor.get_column_letter(i)
            except:
                if i < 26:
                    letter = chr(65 + i)
                else:
                    first = chr(65 + (i // 26) - 1)
                    second = chr(65 + (i % 26))
                    letter = first + second
            
            columns_with_letters.append({
                'index': i,
                'letter': letter,
                'name': str(col),
                'display': f"Колонка {letter} - {col}"
            })
        
        preview_rows = []
        for i in range(min(5, len(df))):
            row = []
            for val in df.iloc[i].values:
                str_val = str(val) if pd.notna(val) else ""
                if len(str_val) > 50:
                    str_val = str_val[:50] + "..."
                row.append(str_val)
            preview_rows.append(row)
        
        print("✅ Отправляем ответ клиенту")
        return jsonify({
            'success': True,
            'columns': columns_with_letters,
            'total_rows': len(df),
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
    data = request.get_json()
    
    column_value = data.get('column', 0)
    row_limit = data.get('row_limit', 0)
    
    if isinstance(column_value, str) and column_value.isalpha():
        column_index = excel_processor.letter_to_index(column_value.upper())
    else:
        column_index = int(column_value)
    
    filepath = session.get('current_file')
    original_filename = session.get('original_filename')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Файл не найден'}), 400
    
    try:
        df, texts, column_name = excel_processor.read_excel(
            filepath, 
            column_index=column_index
        )
        
        total_rows = len(texts)
        if row_limit > 0 and row_limit < total_rows:
            texts = texts[:row_limit]
            print(f"📊 Анализ ограничен: {row_limit} из {total_rows} строк")
        else:
            print(f"📊 Анализ всех {total_rows} строк")
        
        column_letter = excel_processor.get_column_letter(column_index)
        
        job_id = batch_analyzer.create_job(
            filepath, 
            original_filename,
            {
                'column_name': column_name, 
                'column_index': column_index,
                'column_letter': column_letter,
                'row_limit': row_limit,
                'total_rows': total_rows
            }
        )
        
        thread = threading.Thread(
            target=process_batch_job,
            args=(job_id, df, texts, column_name)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'rows_to_process': len(texts),
            'total_rows': total_rows
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_batch_job(job_id, df, texts, column_name):
    """Обработка пакетного задания в фоне"""
    try:
        batch_analyzer.update_job_progress(job_id, 0, len(texts))
        
        def progress_callback(current, total):
            # Проверяем, не запрошена ли остановка
            if stop_flags.get(job_id, False):
                print(f"🛑 Задание {job_id} остановлено по запросу")
                raise Exception("Analysis stopped by user")
            batch_analyzer.update_job_progress(job_id, current, total)
        
        results = excel_processor.analyze_batch(texts, progress_callback)
        
        # Проверяем остановку перед созданием DataFrame
        if stop_flags.get(job_id, False):
            print(f"🛑 Задание {job_id} остановлено, пропускаем создание результата")
            return
        
        result_df = excel_processor.create_result_dataframe(df, column_name, results)
        
        output_path, output_filename = excel_processor.save_to_excel(
            result_df, 
            batch_analyzer.get_job(job_id)['original_filename'],
            app.config['DOWNLOAD_FOLDER']
        )
        
        # Финальная проверка остановки
        if stop_flags.get(job_id, False):
            print(f"🛑 Задание {job_id} остановлено, удаляем файл результата")
            if os.path.exists(output_path):
                os.remove(output_path)
            return
        
        batch_analyzer.complete_job(job_id, output_path, output_filename)
        print(f"✅ Задание {job_id} завершено. Файл: {output_filename}")
        
    except Exception as e:
        if str(e) == "Analysis stopped by user":
            print(f"🛑 Задание {job_id} остановлено пользователем")
            batch_analyzer.fail_job(job_id, "Остановлено пользователем")
        else:
            batch_analyzer.fail_job(job_id, str(e))
            print(f"❌ Ошибка в задании {job_id}: {e}")
            import traceback
            traceback.print_exc()
    finally:
        # Очищаем флаг остановки
        if job_id in stop_flags:
            del stop_flags[job_id]

@app.route('/stop_analysis/<job_id>', methods=['POST'])
def stop_analysis(job_id):
    """Остановка анализа"""
    job = batch_analyzer.get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Задание не найдено'}), 404
    
    # Устанавливаем флаг остановки
    stop_flags[job_id] = True
    
    # Обновляем статус задания
    job['status'] = 'cancelled'
    job['error'] = 'Анализ остановлен пользователем'
    
    print(f"🛑 Анализ {job_id} остановлен пользователем")
    
    return jsonify({'success': True, 'message': 'Анализ остановлен'})

@app.route('/job_status/<job_id>')
def job_status(job_id):
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
    test_texts = [
        "Сегодня прошел финальный матч чемпионата мира по футболу.",
        "Apple представила новый iPhone 15 с улучшенной камерой.",
        "В Госдуме приняли новый закон о цифровых технологиях.",
        "Вышел новый фильм Кристофера Нолана.",
        "Цены на нефть выросли на фоне новостей из Саудовской Аравии."
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
    
    return jsonify(info)

@app.route('/demo_examples')
def demo_examples():
    examples = [
        {'text': 'Этот фильм просто великолепен!', 'expected_sentiment': 'positive', 'expected_topic': 'entertainment'},
        {'text': 'Ужасный матч, наша команда провалилась.', 'expected_sentiment': 'negative', 'expected_topic': 'sports'},
        {'text': 'В Госдуме обсуждают новый законопроект.', 'expected_sentiment': 'neutral', 'expected_topic': 'politics'}
    ]
    return jsonify({'examples': examples})

@app.route('/batch_status')
def batch_status():
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
    print("🚀 ЗАПУСК BERT-АНАЛИЗАТОРА")
    print("=" * 60)
    print("🌐 Одиночный анализ: http://localhost:5000")
    print("📊 Загрузка Excel: http://localhost:5000/upload")
    print("=" * 60)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    app.run(debug=True, port=5000, threaded=True)