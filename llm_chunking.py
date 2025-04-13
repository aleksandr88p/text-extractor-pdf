"""
Скрипт для создания улучшенных чанков с использованием LLM
Он разбивает текст на смысловые фрагменты, отправляя их в LLM для анализа
"""

import os
import json
import time
import re
import requests
from typing import List, Dict, Tuple, Optional
import time

def chunk_with_llm(text_file: str, output_file: str, chunk_size: int = 8000, overlap: int = 500):
    """
    Разбивает текст на смысловые чанки с помощью LLM
    
    Args:
        text_file: путь к файлу с извлеченным текстом
        output_file: путь для сохранения структурированных чанков
        chunk_size: максимальный размер текста для отправки в LLM
        overlap: размер пересечения между частями текста
    """
    # Загружаем текст
    with open(text_file, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Разбиваем текст на большие части для анализа LLM
    parts = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        
        # Если не достигли конца, ищем хорошее место для разрыва
        if end < len(full_text):
            # Ищем ближайший конец параграфа или конец предложения
            paragraph_end = full_text.rfind('\n\n', start, end)
            if paragraph_end > start + chunk_size // 2:
                end = paragraph_end
            else:
                sentence_end = full_text.rfind('. ', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
        
        parts.append(full_text[start:end])
        
        # Добавляем перекрытие для контекста
        start = end - overlap
    
    # Обрабатываем каждую часть с помощью LLM
    structured_chunks = []
    
    for i, part in enumerate(parts):
        print(f"Обработка части {i+1}/{len(parts)}...")
        
        # Анализируем структуру текста с помощью LLM
        analyzed_structure = analyze_text_structure(part)
        
        # Извлекаем структурированные чанки
        chunks = extract_semantic_chunks(part, analyzed_structure)
        
        # Добавляем в общий список
        structured_chunks.extend(chunks)
        
        # Небольшая пауза, чтобы не перегрузить API
        time.sleep(1)
    
    # Удаляем дублирующиеся чанки из-за перекрытий
    deduplicated_chunks = deduplicate_chunks(structured_chunks)
    
    # Сохраняем результаты
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduplicated_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"Создано {len(deduplicated_chunks)} семантических чанков.")
    return deduplicated_chunks

def analyze_text_structure(text: str) -> Dict:
    """
    Отправляет текст в LLM для анализа его структуры
    
    Args:
        text: текст для анализа
        
    Returns:
        словарь с информацией о структуре текста
    """
    # Здесь вы можете использовать API вашей LLM 
    # Пример запроса к LLM API
    prompt = f"""
    Проанализируй следующий фрагмент Уголовного кодекса (Código Penal) и определи его структуру:
    - Определи, к какой книге (LIBRO) относится текст
    - Определи, к какому разделу (TÍTULO) относится текст
    - Определи, к какой главе (CAPÍTULO) относится текст
    - Перечисли номера статей (Artículo), которые встречаются в тексте
    
    Текст:
    {text[:5000]}...
    
    Ответ представь в формате JSON:
    {{
        "libro": "название книги",
        "titulo": "название раздела",
        "capitulo": "название главы",
        "articulos": ["номер1", "номер2", ...]
    }}
    """
    
    # Вместо вызова реального API, эта функция-заглушка
    # В реальном коде здесь будет вызов API вашей LLM
    # response = call_llm_api(prompt)
    
    # Заглушка для примера
    # В реальности, нужно реализовать вызов к API вашей LLM
    # и парсинг JSON из ответа
    
    # Пример выхода:
    structure = {
        "libro": "LIBRO I",
        "titulo": "TÍTULO VII",
        "capitulo": "CAPÍTULO I",
        "articulos": ["130", "131", "132"]
    }
    
    return structure

def extract_semantic_chunks(text: str, structure: Dict) -> List[Dict]:
    """
    Извлекает семантические чанки из текста на основе анализа структуры
    
    Args:
        text: исходный текст
        structure: структура текста, полученная от LLM
        
    Returns:
        список словарей с чанками
    """
    chunks = []
    
    # Находим статьи в тексте
    article_pattern = r'Art[íi]culo (\d+)'
    article_matches = list(re.finditer(article_pattern, text))
    
    # Если статей не нашли, создаем один общий чанк
    if not article_matches:
        chunks.append({
            "libro": structure["libro"],
            "titulo": structure["titulo"],
            "capitulo": structure["capitulo"],
            "article_numbers": structure["articulos"],
            "chunk_index": 0,
            "text": text
        })
        return chunks
    
    # Разбиваем текст по статьям
    for i in range(len(article_matches)):
        start_idx = article_matches[i].start()
        
        # Определяем конец текущего фрагмента
        if i < len(article_matches) - 1:
            end_idx = article_matches[i+1].start()
        else:
            end_idx = len(text)
        
        # Извлекаем текст статьи
        article_text = text[start_idx:end_idx]
        
        # Находим номер статьи
        article_number = re.search(article_pattern, article_text).group(1)
        
        chunks.append({
            "libro": structure["libro"],
            "titulo": structure["titulo"],
            "capitulo": structure["capitulo"],
            "article_numbers": [article_number],
            "chunk_index": i,
            "text": article_text.strip()
        })
    
    return chunks

def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Удаляет дублирующиеся чанки, которые могли появиться из-за перекрытий
    
    Args:
        chunks: список чанков
        
    Returns:
        список уникальных чанков
    """
    unique_chunks = []
    seen_text = set()
    
    for chunk in chunks:
        # Создаем короткую сигнатуру текста (первые 100 символов)
        text_sig = chunk["text"][:100]
        
        if text_sig not in seen_text:
            seen_text.add(text_sig)
            unique_chunks.append(chunk)
    
    return unique_chunks

if __name__ == "__main__":
    # Путь к извлеченному тексту
    text_file = "extracted_text.txt"
    
    # Путь для сохранения чанков
    output_file = "output/llm_chunked_document.json"
    
    # Создаем чанки с помощью LLM
    chunks = chunk_with_llm(text_file, output_file)
    
    print(f"Чанкирование завершено! Создано {len(chunks)} смысловых чанков.")
    print(f"Результаты сохранены в {output_file}")