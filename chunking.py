"""
Разбиение текста на чанки оптимального размера для эмбеддингов

Этот скрипт загружает группированные статьи из grouped_articles.json и 
разбивает их на чанки оптимального размера для создания эмбеддингов.
"""

import json
import re
from typing import List, Dict

def split_text_into_chunks(text: str, chunk_size: int = 2000) -> List[str]:
    """
    Разбивает текст на чанки оптимального размера.
    
    Args:
        text (str): Исходный текст для разбиения
        chunk_size (int): Максимальный размер чанка в символах
        
    Returns:
        List[str]: Список чанков
    """
    chunks = []
    
    # Пробуем разбить текст по статьям или параграфам
    pattern = r"Art[íi]culo \d+\."
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        # Если не нашли совпадений с шаблоном, используем базовое разбиение
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Если не достигли конца, то ищем последний перенос строки или точку
            if end < len(text):
                # Сначала ищем последний перенос строки в пределах chunk_size
                last_break = text.rfind('\n', start, end)
                if last_break > start + chunk_size // 2:
                    end = last_break + 1
                else:
                    # Если подходящего переноса нет, то ищем последнюю точку
                    last_period = text.rfind('.', start, end)
                    if last_period > start + chunk_size // 2:
                        end = last_period + 1
            
            chunks.append(text[start:end])
            start = end
    else:
        # Если нашли статьи, разбиваем по ним
        for i in range(len(matches)):
            start_idx = matches[i].start()
            
            # Определяем конец текущего фрагмента
            if i < len(matches) - 1:
                end_idx = matches[i+1].start()
            else:
                end_idx = len(text)
            
            # Если фрагмент слишком большой, разбиваем его дополнительно
            fragment = text[start_idx:end_idx]
            if len(fragment) > chunk_size:
                start = 0
                while start < len(fragment):
                    end = min(start + chunk_size, len(fragment))
                    # Ищем последний перенос строки или точку
                    if end < len(fragment):
                        last_break = fragment.rfind('\n', start, end)
                        if last_break > start + chunk_size // 2:
                            end = last_break + 1
                        else:
                            last_period = fragment.rfind('.', start, end)
                            if last_period > start + chunk_size // 2:
                                end = last_period + 1
                    chunks.append(fragment[start:end])
                    start = end
            else:
                chunks.append(fragment)
    
    return chunks

def create_final_chunks(grouped_articles: Dict) -> List[Dict]:
    """
    Создает финальные чанки для эмбеддингов на основе сгруппированных статей.
    
    Args:
        grouped_articles (Dict): Сгруппированные статьи
        
    Returns:
        List[Dict]: Список чанков с метаданными
    """
    final_chunks = []
    
    # Проходим по каждой группе статей
    for group_key, group_data in grouped_articles.items():
        # Получаем текст всей группы
        full_text = group_data["full_text"]
        
        # Разбиваем текст на чанки с учетом структуры текста
        text_chunks = split_text_into_chunks(full_text)
        
        # Определяем, какие статьи входят в каждый чанк
        article_pattern = r'Art[íi]culo (\d+)'
        
        # Создаем словарь с метаданными для каждого чанка
        for i, chunk_text in enumerate(text_chunks):
            # Находим все номера статей в текущем чанке
            article_numbers_in_chunk = re.findall(article_pattern, chunk_text)
            
            # Если статьи в чанке не найдены, используем общие статьи группы
            if not article_numbers_in_chunk:
                article_numbers_in_chunk = group_data["article_numbers"]
            
            chunk = {
                "libro": group_data["libro"],
                "titulo": group_data["titulo"],
                "capitulo": group_data["capitulo"],
                "article_numbers": article_numbers_in_chunk,
                "chunk_index": i,
                "text": chunk_text
            }
            
            final_chunks.append(chunk)
            
    return final_chunks

if __name__ == "__main__":
    # Загружаем сгруппированные статьи
    print("Загрузка сгруппированных статей из grouped_articles.json...")
    try:
        with open('output/grouped_articles.json', 'r', encoding='utf-8') as f:
            grouped_articles = json.load(f)
    except FileNotFoundError:
        with open('grouped_articles.json', 'r', encoding='utf-8') as f:
            grouped_articles = json.load(f)
    
    # Создаем финальные чанки
    print("Создание чанков для эмбеддингов...")
    final_chunks = create_final_chunks(grouped_articles)
    
    # Сохраняем результаты
    output_path = 'output/penal_code_chunks.json'
    print(f"Сохранение {len(final_chunks)} чанков в {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_chunks, f, indent=2, ensure_ascii=False)
    
    print("Разбиение на чанки завершено успешно!")
    print(f"Средний размер чанка: {sum(len(chunk['text']) for chunk in final_chunks) / len(final_chunks):.2f} символов")