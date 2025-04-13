"""
Модуль для поиска по эмбеддингам Уголовного кодекса

Этот скрипт позволяет выполнять семантический поиск по тексту Уголовного кодекса
с использованием FAISS индекса и предварительно созданных эмбеддингов.
"""

import faiss
import numpy as np
import json
import os
import re
from sentence_transformers import SentenceTransformer

def search_similar_chunks(
    question: str,
    chunks: list,
    index_path: str = "output/penal_code.index",
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5
) -> list:
    """
    Ищет наиболее похожие чанки текста на основе семантической близости.
    
    Args:
        question (str): Текст запроса
        chunks (list): Список чанков для поиска
        index_path (str): Путь к файлу индекса
        model_name (str): Название модели для создания эмбеддингов
        top_k (int): Количество результатов для возврата
        
    Returns:
        list: Список наиболее похожих чанков с указанием расстояния
    """
    # Предварительная фильтрация по ключевым словам в запросе
    key_terms = extract_key_terms(question.lower())
    
    # Сначала проверяем базовое текстовое совпадение
    text_matches = []
    for idx, chunk in enumerate(chunks):
        chunk_text = chunk["text"].lower()
        score = 0
        for term in key_terms:
            if term in chunk_text:
                score += 1
        
        if score > 0:
            text_matches.append({
                "idx": idx,
                "score": score,
                "chunk": chunk.copy()
            })
    
    # Сортируем по количеству совпадений
    text_matches.sort(key=lambda x: x["score"], reverse=True)
    
    # Если нашли хотя бы несколько совпадений по тексту, берем их
    direct_results = []
    if len(text_matches) >= 2:
        direct_results = [match["chunk"] for match in text_matches[:3]]
    
    # Загружаем FAISS index для векторного поиска
    index = faiss.read_index(index_path)
    
    # Загружаем модель эмбеддингов
    model = SentenceTransformer(model_name)
    
    # Считаем эмбеддинги запроса
    question_embedding = model.encode([question], convert_to_numpy=True)
    
    # Ищем ближайшие K
    distances, indices = index.search(question_embedding, top_k)
    
    # Собираем результаты из векторного поиска
    vector_results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(chunks):
            result = chunks[idx].copy()
            result["distance"] = float(dist)
            vector_results.append(result)
    
    # Комбинируем результаты текстового и векторного поиска
    # Отдаем приоритет текстовым совпадениям
    combined_results = []
    seen_indices = set()
    
    # Добавляем сначала текстовые совпадения
    for result in direct_results:
        key = (result["libro"], result["titulo"], result["capitulo"], result["chunk_index"])
        if key not in seen_indices:
            seen_indices.add(key)
            result["match_type"] = "text_match"
            combined_results.append(result)
    
    # Добавляем векторные совпадения, если их еще нет в результатах
    for result in vector_results:
        key = (result["libro"], result["titulo"], result["capitulo"], result["chunk_index"])
        if key not in seen_indices:
            seen_indices.add(key)
            result["match_type"] = "vector_match"
            combined_results.append(result)
    
    # Ограничиваем общее количество результатов
    return combined_results[:top_k]

def extract_key_terms(question):
    """
    Извлекает ключевые термины для текстового поиска.
    """
    # Список базовых юридических терминов на испанском
    basic_terms = [
        "prescripción", "plazo", "delito", "homicidio", "pena", "robo", 
        "estafa", "hurto", "violencia", "asesinato", "alevosía", 
        "atenuante", "agravante", "responsabilidad", "civil", "penal"
    ]
    
    # Поиск этих терминов в вопросе
    found_terms = []
    for term in basic_terms:
        if term in question:
            found_terms.append(term)
    
    # Дополнительно извлекаем числа и термины вида "artículo X"
    numbers = re.findall(r'\d+', question)
    article_refs = re.findall(r'art[íi]culo\s+\d+', question)
    
    found_terms.extend(numbers)
    found_terms.extend(article_refs)
    
    return found_terms

def search_by_article_number(article_number: str, chunks: list) -> list:
    """
    Ищет конкретную статью по её номеру.
    
    Args:
        article_number (str): Номер статьи
        chunks (list): Список всех чанков
        
    Returns:
        list: Список чанков содержащих указанную статью
    """
    results = []
    
    for chunk in chunks:
        if article_number in chunk["article_numbers"]:
            results.append(chunk)
            
    return results

def format_search_results(results: list) -> str:
    """
    Форматирует результаты поиска для удобного вывода.
    
    Args:
        results (list): Список чанков с результатами поиска
        
    Returns:
        str: Форматированный текст с результатами
    """
    output = []
    
    for i, chunk in enumerate(results, 1):
        match_type = ""
        if "match_type" in chunk:
            if chunk["match_type"] == "text_match":
                match_type = " 🔤 [Текстовое совпадение]"
            elif chunk["match_type"] == "vector_match":
                match_type = " 🔍 [Векторное совпадение]"
        
        output.append(f"\n--- Результат {i}{match_type} {'='*40}")
        output.append(f"📚 Книга: {chunk['libro']}")
        
        if chunk['titulo']:
            output.append(f"📖 Раздел: {chunk['titulo']}")
            
        if chunk['capitulo']:
            output.append(f"📑 Глава: {chunk['capitulo']}")
            
        output.append(f"📋 Статьи: {', '.join(chunk['article_numbers'])}")
            
        output.append(f"\n{chunk['text']}\n")
    
    return "\n".join(output)

def extract_article_number(question: str):
    """
    Извлекает номер статьи из запроса, если есть.
    
    Args:
        question (str): Вопрос пользователя
        
    Returns:
        str: Номер статьи или None
    """
    # Ищем запросы о конкретных статьях
    article_match = re.search(r'(artículo|articulo|статья|статьи|art\.?)\s*(\d+)', question, re.IGNORECASE)
    if article_match:
        return article_match.group(2)
    return None

if __name__ == "__main__":
    # Пути к файлам
    chunks_file = "output/penal_code_chunks.json"
    index_path = "output/penal_code.index"
    
    # Проверка наличия файлов
    if not os.path.exists(chunks_file):
        alternative_path = "penal_code_chunks.json"
        if os.path.exists(alternative_path):
            chunks_file = alternative_path
        else:
            print(f"Ошибка: Файл с чанками не найден ни в {chunks_file}, ни в {alternative_path}")
            exit(1)
            
    if not os.path.exists(index_path):
        print(f"Ошибка: Индексный файл не найден: {index_path}")
        exit(1)
    
    # Загружаем чанки
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print("🔎 Поиск в Уголовном кодексе")
    print(f"Загружено {len(chunks)} чанков текста")
    
    # Интерактивный режим поиска
    while True:
        question = input("\nВведите запрос (или 'q' для выхода): ")
        if question.lower() in ['q', 'quit', 'exit']:
            break
        
        # Проверяем, запрашивается ли конкретная статья
        article_number = extract_article_number(question)
        
        if article_number:
            print(f"Поиск статьи {article_number}...")
            results = search_by_article_number(article_number, chunks)
            if not results:
                print(f"Статья {article_number} не найдена в кодексе.")
                # Пробуем семантический поиск
                results = search_similar_chunks(question, chunks, index_path)
        else:
            # Обычный семантический поиск
            results = search_similar_chunks(question, chunks, index_path)
        
        if results:
            print(format_search_results(results))
        else:
            print("По вашему запросу ничего не найдено.")