from flask import Flask, render_template, request, jsonify, send_file, session
from sentiment_model import BertSentimentAnalyzer
from topic_model import BertTopicClassifier, SimpleTopicClassifier
from excel_processor import ExcelProcessor, BatchAnalyzer
from vk_parser import VKParser, VKAnalyzer
import time
import torch
import os
import uuid
import pandas as pd
import threading
import json
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

# Словарь для заданий ВК
vk_jobs = {}

print("=" * 60)
print("🚀 ЗАПУСК BERT-АНАЛИЗАТОРА")
print("=" * 60)

print("🔄 Загрузка моделей...")

# Инициализация моделей
sentiment_analyzer = BertSentimentAnalyzer()

# Пробуем загрузить продвинутую модель для тематики
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

# Инициализация процессора Excel и менеджера заданий
excel_processor = ExcelProcessor(sentiment_analyzer, topic_classifier)
batch_analyzer = BatchAnalyzer(excel_processor)

# Инициализация парсера ВК
vk_parser = VKParser()
vk_analyzer = VKAnalyzer(vk_parser, sentiment_analyzer, topic_classifier)

print("✅ Все модели загружены!")
print("=" * 60)
print("🌐 Доступные страницы:")
print("   📝 Одиночный анализ: http://localhost:5000")
print("   📊 Загрузка Excel: http://localhost:5000/upload")
print("   📱 Парсер ВК: http://localhost:5000/vk")
print("=" * 60)

# ==================== ГЛАВНЫЕ СТРАНИЦЫ ====================

@app.route('/')
def index():
    """Главная страница с одиночным анализом"""
    topics = topic_classifier.get_all_topics()
    return render_template('index.html', topics=topics, model_type=type(topic_classifier).__name__)

@app.route('/upload')
def upload_page():
    """Страница загрузки Excel"""
    return render_template('upload.html')

@app.route('/vk')
def vk_page():
    """Страница парсера ВК"""
    return render_template('vk_parser.html')

# ==================== АНАЛИЗ ТЕКСТА ====================

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

# ==================== ПАКЕТНЫЙ АНАЛИЗ EXCEL ====================

@app.route('/upload_excel', methods=['POST'])
def upload_excel():
    """Загрузка Excel файла"""
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
    """Запуск пакетного анализа Excel"""
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
    """Остановка анализа Excel"""
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
    """Получение статуса задания Excel"""
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

# ==================== ПАРСЕР ВК ====================

@app.route('/start_vk_analysis', methods=['POST'])
def start_vk_analysis():
    """Запуск анализа ВК"""
    data = request.get_json()
    
    token = data.get('token')
    days = data.get('days', 7)
    max_posts = data.get('max_posts', 100)
    
    if not token:
        return jsonify({'error': 'Не указан токен ВК'}), 400
    
    try:
        # Создаем парсер с токеном
        parser = VKParser(token)
        analyzer = VKAnalyzer(parser, sentiment_analyzer, topic_classifier)
        
        # Генерируем ID задания
        job_id = f"vk_{int(time.time())}"
        
        # Сохраняем информацию о задании
        vk_jobs[job_id] = {
            'id': job_id,
            'status': 'starting',
            'progress': 0,
            'total': max_posts,
            'created_at': time.time()
        }
        
        print(f"📱 Создано задание ВК: {job_id}")
        
        # Запускаем в фоне
        thread = threading.Thread(
            target=process_vk_job_auto,
            args=(job_id, parser, analyzer, days, max_posts)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started'
        })
        
    except Exception as e:
        print(f"❌ Ошибка при создании задания: {e}")
        return jsonify({'error': str(e)}), 500

def process_vk_job_auto(job_id, parser, analyzer, days, max_posts):
    """Обработка задания ВК в автоматическом режиме"""
    try:
        if job_id not in vk_jobs:
            return
        
        # Этап 1: Сбор данных
        vk_jobs[job_id]['status'] = 'collecting'
        vk_jobs[job_id]['message'] = 'Идет сбор данных с различных источников...'
        print(f"📱 ВК задание {job_id}: начинаем сбор постов")
        
        # Функция обратного вызова для прогресса сбора
        def collect_callback(current, total, source_info):
            if job_id in vk_jobs:
                vk_jobs[job_id]['message'] = f'Обработано источников: {current}/{total} ({source_info})'
        
        # Собираем посты
        df = parser.collect_all_posts(days, max_posts, progress_callback=collect_callback)
        
        # Проверка остановки
        if stop_flags.get(job_id, False):
            vk_jobs[job_id]['status'] = 'cancelled'
            return
        
        if len(df) == 0:
            vk_jobs[job_id]['status'] = 'error'
            vk_jobs[job_id]['error'] = 'Не удалось собрать посты'
            return
        
        # Этап 2: Анализ данных
        vk_jobs[job_id]['status'] = 'analyzing'
        vk_jobs[job_id]['total'] = len(df)
        vk_jobs[job_id]['progress'] = 0
        vk_jobs[job_id]['message'] = f'Начинаем анализ {len(df)} постов...'
        
        def analyze_callback(current, total):
            if job_id in vk_jobs:
                vk_jobs[job_id]['progress'] = current
                vk_jobs[job_id]['message'] = f'Анализ: {current}/{total}'
            if stop_flags.get(job_id, False):
                raise Exception("Analysis stopped by user")
        
        # Анализируем посты
        result_df = analyzer.analyze_posts(df, progress_callback=analyze_callback)
        
        if stop_flags.get(job_id, False):
            vk_jobs[job_id]['status'] = 'cancelled'
            return
        
        # Сохраняем в папку downloads
        filename = analyzer.generate_report(result_df, app.config['DOWNLOAD_FOLDER'])
        
        # Получаем статистику
        stats = analyzer.get_statistics(result_df)
        
        if job_id not in vk_jobs:
            return
        
        # Сохраняем результат
        vk_jobs[job_id].update({
            'status': 'completed',
            'progress': len(df),
            'total': len(df),
            'filename': filename,
            'stats': stats,
            'result_df': result_df.head(20).to_dict('records')  # Показываем 20 постов
        })
        
        print(f"✅ ВК анализ {job_id} завершен. Файл: {filename}")
        
    except Exception as e:
        if str(e) == "Analysis stopped by user":
            print(f"🛑 ВК задание {job_id} остановлено пользователем")
            if job_id in vk_jobs:
                vk_jobs[job_id]['status'] = 'cancelled'
        else:
            print(f"❌ Ошибка в ВК задании {job_id}: {e}")
            import traceback
            traceback.print_exc()
            if job_id in vk_jobs:
                vk_jobs[job_id]['status'] = 'error'
                vk_jobs[job_id]['error'] = str(e)
    finally:
        if job_id in stop_flags:
            del stop_flags[job_id]

@app.route('/vk_status/<job_id>')
def vk_status(job_id):
    """Статус задания ВК"""
    job = vk_jobs.get(job_id)
    
    if not job:
        return jsonify({'error': 'Задание не найдено'}), 404
    
    response = {
        'status': job['status'],
        'progress': job.get('progress', 0),
        'total': job.get('total', 0),
        'error': job.get('error')
    }
    
    if job['status'] == 'completed':
        response.update({
            'filename': job.get('filename'),
            'stats': job.get('stats'),
            'preview': job.get('result_df', [])
        })
    elif job['status'] == 'starting':
        response['message'] = 'Задание создается...'
    elif job['status'] == 'collecting':
        response['message'] = f'Сбор постов... ({job.get("progress", 0)}/{job.get("total", 0)})'
    elif job['status'] == 'analyzing':
        response['message'] = f'Анализ постов... ({job.get("progress", 0)}/{job.get("total", 0)})'
    elif job['status'] == 'cancelled':
        response['message'] = 'Анализ остановлен пользователем'
    
    return jsonify(response)

@app.route('/stop_vk_analysis/<job_id>', methods=['POST'])
def stop_vk_analysis(job_id):
    """Остановка анализа ВК"""
    if job_id in vk_jobs:
        vk_jobs[job_id]['status'] = 'cancelled'
        # Устанавливаем флаг остановки
        stop_flags[job_id] = True
        print(f"🛑 ВК задание {job_id} остановлено пользователем")
        return jsonify({'success': True})
    return jsonify({'error': 'Задание не найдено'}), 404

@app.route('/vk_sources')
def get_vk_sources():
    """Получение популярных источников"""
    return jsonify({'sources': get_default_sources()})

# ==================== ОБЩИЕ МАРШРУТЫ ====================

@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание файла с результатами"""
    # Проверяем в папке downloads
    filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        # Проверяем в текущей папке (для файлов ВК)
        filepath = filename
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
    """Статус всех заданий Excel"""
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

@app.route('/vk_jobs_status')
def vk_jobs_status():
    """Статус всех заданий ВК"""
    # Очищаем старые задания (старше часа)
    current_time = time.time()
    to_delete = []
    
    for job_id, job in vk_jobs.items():
        if current_time - job.get('created_at', 0) > 3600:
            to_delete.append(job_id)
    
    for job_id in to_delete:
        del vk_jobs[job_id]
    
    return jsonify({
        'active_jobs': len(vk_jobs),
        'jobs': [
            {
                'id': j['id'],
                'status': j['status'],
                'progress': f"{j.get('progress', 0)}/{j.get('total', 0)}"
            }
            for j in vk_jobs.values()
        ]
    })

# ==================== ЗАПУСК ====================

if __name__ == '__main__':
    # Очищаем кэш GPU если есть
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    app.run(debug=True, port=5000, threaded=True)