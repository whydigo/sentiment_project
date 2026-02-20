import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
from werkzeug.utils import secure_filename
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter as get_excel_letter
from openpyxl.utils import column_index_from_string

class ExcelProcessor:
    def __init__(self, sentiment_analyzer, topic_classifier):
        self.sentiment_analyzer = sentiment_analyzer
        self.topic_classifier = topic_classifier
        self.allowed_extensions = {'xlsx', 'xls', 'csv'}
        
        # Цвета для тональности (для форматирования)
        self.sentiment_colors = {
            'positive': '92D050',  # Зеленый
            'negative': 'FF6B6B',  # Красный
            'neutral': 'FFD966'     # Желтый
        }
    
    def get_column_letter(self, index):
        """
        Преобразует индекс колонки (0,1,2...) в букву Excel (A, B, C...)
        Пример: 0 -> A, 1 -> B, 25 -> Z, 26 -> AA
        """
        return get_excel_letter(index + 1)  # +1 потому что Excel считает с 1
    
    def letter_to_index(self, letter):
        """
        Преобразует букву Excel (A, B, C, AA, AB...) в индекс (0,1,2...)
        Пример: A -> 0, B -> 1, Z -> 25, AA -> 26
        """
        return column_index_from_string(letter) - 1  # -1 потому что нам нужен индекс с 0
    
    def allowed_file(self, filename):
        """Проверка расширения файла"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def read_excel(self, filepath, sheet_name=0, column_name=None, column_index=0):
        """Чтение Excel файла"""
        try:
            # Определяем тип файла
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath, encoding='utf-8')
            else:
                df = pd.read_excel(filepath, sheet_name=sheet_name, engine='openpyxl')
            
            # Если указано имя колонки
            if column_name and column_name in df.columns:
                texts = df[column_name].dropna().tolist()
                return df, texts, column_name
            # Если указан индекс колонки
            elif column_index < len(df.columns):
                column_name = df.columns[column_index]
                texts = df[column_name].dropna().tolist()
                return df, texts, column_name
            else:
                # Берем первую колонку
                column_name = df.columns[0]
                texts = df[column_name].dropna().tolist()
                return df, texts, column_name
                
        except Exception as e:
            raise Exception(f"Ошибка чтения файла: {str(e)}")
    

    
    def analyze_batch(self, texts, progress_callback=None):
        """Пакетный анализ текстов"""
        results = []
        
        for i, text in enumerate(tqdm(texts, desc="Анализ текстов")):
            try:
                # Приводим к строке
                text = str(text) if pd.notna(text) else ""
                
                if len(text.strip()) < 3:
                    # Пропускаем пустые тексты
                    results.append({
                        'text': text,
                        'sentiment': 'neutral',
                        'sentiment_confidence': 0,
                        'topic': 'other',
                        'topic_name': 'Другое',
                        'topic_confidence': 0,
                        'error': 'Текст слишком короткий'
                    })
                else:
                    # Анализ тональности
                    sentiment_result = self.sentiment_analyzer.analyze(text)
                    
                    # Анализ тематики
                    topic_result = self.topic_classifier.classify(text)
                    
                    results.append({
                        'text': text,
                        'sentiment': sentiment_result['sentiment'],
                        'sentiment_confidence': sentiment_result['confidence'],
                        'topic': topic_result['topic'],
                        'topic_name': topic_result['topic_name'],
                        'topic_confidence': topic_result['confidence'],
                        'sentiment_probs': sentiment_result.get('probabilities', {}),
                        'all_topics': topic_result.get('all_topics', [])
                    })
                
                # Отправляем прогресс
                if progress_callback:
                    progress_callback(i + 1, len(texts))
                    
            except Exception as e:
                print(f"Ошибка анализа текста {i}: {e}")
                results.append({
                    'text': text,
                    'sentiment': 'neutral',
                    'sentiment_confidence': 0,
                    'topic': 'other',
                    'topic_name': 'Другое',
                    'topic_confidence': 0,
                    'error': str(e)
                })
        
        return results
    
    def create_result_dataframe(self, original_df, texts_column, results, options=None):
        """Создание результирующего DataFrame"""
        if options is None:
            options = {}
        
        result_df = original_df.copy()
        
        # Добавляем новые колонки
        result_df['Тональность'] = [r['sentiment'] for r in results]
        
        if options.get('include_confidence', True):
            result_df['Уверенность_тональности'] = [r['sentiment_confidence'] for r in results]
        
        result_df['Тематика'] = [r['topic_name'] for r in results]
        
        if options.get('include_confidence', True):
            result_df['Уверенность_тематики'] = [r['topic_confidence'] for r in results]
        
        # Добавляем эмодзи для наглядности
        if options.get('add_emoji', True):
            emoji_map = {'positive': '😊', 'negative': '😠', 'neutral': '😐'}
            result_df['Эмодзи'] = [emoji_map.get(r['sentiment'], '🤔') for r in results]
        
        # Добавляем альтернативные темы (топ-3)
        if options.get('include_alt_topics', True):
            alt_topics = []
            for r in results:
                if r.get('all_topics'):
                    topics = [f"{t['name']}({t['confidence']:.2f})" for t in r['all_topics'][:3]]
                    alt_topics.append(', '.join(topics))
                else:
                    alt_topics.append('')
            result_df['Альтернативные_темы'] = alt_topics
        
        return result_df
    
    def save_to_excel(self, df, original_filename, output_dir='downloads'):
        """Сохранение результатов в Excel с форматированием"""
        # Создаем папку для загрузок если её нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Генерируем имя файла
        base_name = secure_filename(original_filename)
        name_without_ext = os.path.splitext(base_name)[0]
        output_filename = f"{name_without_ext}_analyzed_{int(time.time())}.xlsx"
        output_path = os.path.join(output_dir, output_filename)
        
        # Сохраняем с форматированием
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Результаты анализа', index=False)
            
            # Получаем workbook и worksheet
            workbook = writer.book
            worksheet = writer.sheets['Результаты анализа']
            
            # Форматирование
            self._format_excel(worksheet, df)
        
        return output_path, output_filename
    
    def _format_excel(self, worksheet, df):
        """Форматирование Excel файла"""
        # Стили
        header_font = Font(bold=True, color="FFFFFF", size=11)
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Форматируем заголовки
        for col_idx, col_name in enumerate(df.columns, 1):
            cell = worksheet.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = border
            
            # Автоподбор ширины
            max_length = len(str(col_name))
            column_letter = openpyxl.utils.get_column_letter(col_idx)
            worksheet.column_dimensions[column_letter].width = min(max_length + 5, 50)
        
        # Форматируем данные
        for row_idx, row in enumerate(worksheet.iter_rows(min_row=2, max_row=worksheet.max_row), 2):
            for col_idx, cell in enumerate(row, 1):
                cell.border = border
                cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
                
                # Цвет для тональности
                if df.columns[col_idx-1] == 'Тональность':
                    sentiment = cell.value
                    if sentiment in self.sentiment_colors:
                        cell.fill = PatternFill(
                            start_color=self.sentiment_colors[sentiment],
                            end_color=self.sentiment_colors[sentiment],
                            fill_type="solid"
                        )
                
                # Автоподбор ширины для длинных текстов
                if col_idx == 1:  # Первая колонка с текстом
                    max_length = min(len(str(cell.value)) if cell.value else 0, 100)
                    current_width = worksheet.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width
                    worksheet.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = max(current_width or 0, max_length + 5)
        
        # Добавляем фильтры
        worksheet.auto_filter.ref = worksheet.dimensions
        
        # Фиксируем шапку
        worksheet.freeze_panes = 'A2'


class BatchAnalyzer:
    """Класс для управления пакетным анализом"""
    
    def __init__(self, excel_processor):
        self.excel_processor = excel_processor
        self.jobs = {}  # Словарь для хранения заданий
        self.job_counter = 0
    
    def create_job(self, filepath, original_filename, column_info):
        """Создание нового задания на анализ"""
        job_id = f"job_{int(time.time())}_{self.job_counter}"
        self.job_counter += 1
        
        self.jobs[job_id] = {
            'id': job_id,
            'filepath': filepath,
            'original_filename': original_filename,
            'column_info': column_info,
            'status': 'pending',  # pending, processing, completed, error
            'progress': 0,
            'total': 0,
            'result_path': None,
            'result_filename': None,
            'error': None,
            'created_at': time.time()
        }
        
        return job_id
    
    def update_job_progress(self, job_id, current, total):
        """Обновление прогресса задания"""
        if job_id in self.jobs:
            self.jobs[job_id]['progress'] = current
            self.jobs[job_id]['total'] = total
            self.jobs[job_id]['status'] = 'processing'
    
    def complete_job(self, job_id, result_path, result_filename):
        """Завершение задания"""
        if job_id in self.jobs:
            self.jobs[job_id]['status'] = 'completed'
            self.jobs[job_id]['progress'] = self.jobs[job_id]['total']
            self.jobs[job_id]['result_path'] = result_path
            self.jobs[job_id]['result_filename'] = result_filename
    
    def fail_job(self, job_id, error):
        """Отметить задание как ошибочное"""
        if job_id in self.jobs:
            self.jobs[job_id]['status'] = 'error'
            self.jobs[job_id]['error'] = str(error)
    
    def get_job(self, job_id):
        """Получить информацию о задании"""
        return self.jobs.get(job_id)
    
    def cleanup_old_jobs(self, max_age=3600):
        """Очистка старых заданий (по умолчанию час)"""
        current_time = time.time()
        to_delete = []
        
        for job_id, job in self.jobs.items():
            if current_time - job['created_at'] > max_age:
                to_delete.append(job_id)
                
                # Удаляем файлы
                if job.get('filepath') and os.path.exists(job['filepath']):
                    try:
                        os.remove(job['filepath'])
                    except:
                        pass
                
                if job.get('result_path') and os.path.exists(job['result_path']):
                    try:
                        os.remove(job['result_path'])
                    except:
                        pass
        
        for job_id in to_delete:
            del self.jobs[job_id]