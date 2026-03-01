import requests
import time
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import os

class VKParser:
    def __init__(self, access_token=None):
        """
        Инициализация парсера ВК
        :param access_token: токен доступа ВК
        """
        self.base_url = "https://api.vk.com/method/"
        self.version = "5.131"
        self.access_token = access_token
        self.session = requests.Session()
        
    def collect_all_posts(self, days=7, max_posts=500, progress_callback=None):
        """
        Сбор постов из всех доступных источников
        :param days: за сколько дней искать
        :param max_posts: максимальное количество постов
        :param progress_callback: функция обратного вызова для прогресса
        :return: DataFrame с постами
        """
        if not self.access_token:
            print("❌ Необходим токен ВК")
            return pd.DataFrame(columns=['text', 'date', 'url'])
        
        # Популярные группы для сбора
        popular_groups = [
            'rt_russian', 'rian_ru', 'tassagency', 'izvestia', 'kommersant',
            'sportweek', 'championat', 'sportexpress', 'eurosport_ru',
            'mdk', 'borshch_info', 'oldlentach', 'the_chor',
            'tproger', 'habr', 'geekcity', 'itmozg',
            'novostiphoto', 'lentach', 'fastmedia', 'meduzaproject',
            'life_ru', 'rbc', 'gazeta_ru', 'mk_ru'
        ]
        
        # Популярные хэштеги
        popular_hashtags = ['новости', 'спорт', 'технологии', 'кино', 'музыка', 'наука', 'политика', 'бизнес']
        
        all_posts = []
        total_sources = len(popular_groups) + len(popular_hashtags)
        current_source = 0
        
        print(f"📊 Начинаем сбор постов из {total_sources} источников...")
        
        # Собираем из групп
        groups_count = min(15, max_posts // 7)
        for group in popular_groups[:groups_count]:
            try:
                current_source += 1
                if progress_callback:
                    progress_callback(current_source, total_sources, f"Группа: {group}")
                
                posts = self._get_group_posts(group, count=min(10, max_posts//15))
                if posts:
                    all_posts.extend(posts)
                time.sleep(0.3)
            except Exception as e:
                print(f"Ошибка при сборе группы {group}: {e}")
                continue
        
        # Собираем по хэштегам
        hashtags_count = min(8, max_posts // 15)
        for hashtag in popular_hashtags[:hashtags_count]:
            try:
                current_source += 1
                if progress_callback:
                    progress_callback(current_source, total_sources, f"Хэштег: #{hashtag}")
                
                posts = self._search_by_hashtag(hashtag, days, count=min(15, max_posts//10))
                if posts:
                    all_posts.extend(posts)
                time.sleep(0.3)
            except Exception as e:
                print(f"Ошибка при сборе хэштега {hashtag}: {e}")
                continue
        
        # Убираем дубликаты
        unique_posts = []
        seen_texts = set()
        
        for post in all_posts:
            if post.get('text') and len(post['text']) > 10:
                if post['text'] not in seen_texts:
                    seen_texts.add(post['text'])
                    unique_posts.append(post)
        
        print(f"✅ Собрано уникальных постов: {len(unique_posts)}")
        
        # Создаем DataFrame
        if unique_posts:
            df = pd.DataFrame(unique_posts)
            df = df[['text', 'date', 'url']]  # Только нужные колонки
        else:
            df = pd.DataFrame(columns=['text', 'date', 'url'])
        
        return df
    
    def _get_group_posts(self, group_id, count=20):
        """Получение постов из группы"""
        # Получаем ID группы
        owner_id = self._get_group_id(group_id)
        if not owner_id:
            return []
        
        params = {
            'access_token': self.access_token,
            'v': self.version,
            'owner_id': owner_id,
            'count': min(count, 100),
            'extended': 1
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}wall.get",
                params=params
            )
            data = response.json()
            
            if 'error' in data:
                return []
            
            posts = data.get('response', {}).get('items', [])
            
            formatted_posts = []
            for post in posts:
                formatted_posts.append(self._format_post(post))
            
            return formatted_posts
            
        except:
            return []
    
    def _search_by_hashtag(self, hashtag, days=7, count=20):
        """Поиск по хэштегу"""
        # Рассчитываем даты
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_time = int(start_date.timestamp())
        end_time = int(end_date.timestamp())
        
        params = {
            'access_token': self.access_token,
            'v': self.version,
            'q': f"#{hashtag}",
            'count': min(count, 200),
            'start_time': start_time,
            'end_time': end_time,
            'extended': 1
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}newsfeed.search",
                params=params
            )
            data = response.json()
            
            if 'error' in data:
                return []
            
            posts = data.get('response', {}).get('items', [])
            
            formatted_posts = []
            for post in posts:
                formatted_posts.append(self._format_post(post))
            
            return formatted_posts
            
        except:
            return []
    
    def _get_group_id(self, screen_name):
        """Получение ID группы по короткому имени"""
        params = {
            'access_token': self.access_token,
            'v': self.version,
            'screen_name': screen_name
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}utils.resolveScreenName",
                params=params
            )
            data = response.json()
            
            if 'response' in data and data['response']:
                return f"-{data['response']['object_id']}"
        except:
            pass
        return None
    
    def _format_post(self, post):
        """Форматирование поста"""
        # Получаем текст
        text = post.get('text', '')
        
        # Если есть репост, добавляем его текст
        if 'copy_history' in post and post['copy_history']:
            copy = post['copy_history'][0]
            if copy.get('text'):
                text += "\n\n[Репост]: " + copy.get('text', '')
        
        # Дата
        date = datetime.fromtimestamp(post.get('date', 0))
        
        return {
            'text': text[:1000] if text else '',  # Ограничиваем длину
            'date': date.strftime('%Y-%m-%d %H:%M'),
            'url': f"https://vk.com/wall{post.get('owner_id')}_{post.get('id')}"
        }


class VKAnalyzer:
    """Класс для анализа постов ВК"""
    
    def __init__(self, vk_parser, sentiment_analyzer, topic_classifier):
        self.vk_parser = vk_parser
        self.sentiment_analyzer = sentiment_analyzer
        self.topic_classifier = topic_classifier
    
    def analyze_posts(self, posts_df, progress_callback=None):
        """
        Анализ тональности и тематики постов
        """
        if len(posts_df) == 0:
            return posts_df
        
        df = posts_df.copy()
        
        # Добавляем колонки для результатов
        df['sentiment'] = ''
        df['topic'] = ''
        
        # Анализируем каждый пост
        for idx, row in df.iterrows():
            try:
                text = str(row['text']) if pd.notna(row['text']) else ""
                
                if len(text.strip()) < 10:
                    df.at[idx, 'sentiment'] = 'neutral'
                    df.at[idx, 'topic'] = 'Другое'
                else:
                    # Анализ тональности
                    sentiment_result = self.sentiment_analyzer.analyze(text)
                    df.at[idx, 'sentiment'] = sentiment_result['sentiment']
                    
                    # Анализ тематики
                    topic_result = self.topic_classifier.classify(text)
                    df.at[idx, 'topic'] = topic_result['topic_name']
                
                # Обновляем прогресс
                if progress_callback:
                    progress_callback(idx + 1, len(df))
                    
            except Exception as e:
                print(f"Ошибка анализа поста {idx}: {e}")
                df.at[idx, 'sentiment'] = 'neutral'
                df.at[idx, 'topic'] = 'Другое'
        
        return df
    
    def get_statistics(self, df):
        """Получение статистики"""
        sentiment_counts = df['sentiment'].value_counts()
        
        stats = {
            'total_posts': len(df),
            'sentiment_distribution': {
                'positive': int(sentiment_counts.get('positive', 0)),
                'negative': int(sentiment_counts.get('negative', 0)),
                'neutral': int(sentiment_counts.get('neutral', 0))
            }
        }
        
        return stats
    
    def generate_report(self, df, output_dir='downloads'):
        """Генерация отчета"""
        # Создаем папку если её нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Добавляем дату анализа
        df['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Сортируем по дате
        if 'date' in df.columns:
            df = df.sort_values('date', ascending=False)
        
        # Сохраняем
        timestamp = int(time.time())
        filename = f'vk_analysis_{timestamp}.xlsx'
        filepath = os.path.join(output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Посты', index=False)
            
            # Статистика
            stats = self.get_statistics(df)
            stats_df = pd.DataFrame([
                {'metric': 'Всего постов', 'value': stats['total_posts']},
                {'metric': 'Позитивных', 'value': stats['sentiment_distribution']['positive']},
                {'metric': 'Негативных', 'value': stats['sentiment_distribution']['negative']},
                {'metric': 'Нейтральных', 'value': stats['sentiment_distribution']['neutral']}
            ])
            stats_df.to_excel(writer, sheet_name='Статистика', index=False)
        
        print(f"✅ Отчет сохранен: {filepath}")
        return filename