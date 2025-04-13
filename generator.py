"""
Модуль для создания эмбеддингов и FAISS индекса для Уголовного кодекса

Этот скрипт создает векторные представления (эмбеддинги) текстовых фрагментов
Уголовного кодекса и сохраняет их в индексе FAISS для быстрого семантического поиска.
"""

import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict
from sentence_transformers import SentenceTransformer

def create_embeddings(
    chunks: List[Dict],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 16,
    show_progress: bool = True
) -> np.ndarray:
    """
    Создает эмбеддинги для чанков текста с использованием модели SentenceTransformers.
    
    Args:
        chunks (List[Dict]): Список чанков текста
        model_name (str): Название модели для создания эмбеддингов
        batch_size (int): Размер батча для создания эмбеддингов
        show_progress (bool): Показывать прогресс-бар
        
    Returns:
        np.ndarray: Массив эмбеддингов
    """
    # Загружаем модель для создания эмбеддингов
    print(f"Загрузка модели эмбеддингов: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Извлекаем тексты из чанков
    texts = [chunk["text"] for chunk in chunks]
    
    # Создаем эмбеддинги
    print("Создание эмбеддингов...")
    if show_progress:
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Создание эмбеддингов"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings)
    else:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    print(f"Создано {len(embeddings)} эмбеддингов размерности {embeddings.shape[1]}")
    return embeddings

def create_faiss_index(embeddings: np.ndarray, output_path: str = "output/penal_code.index") -> None:
    """
    Создает FAISS индекс для быстрого поиска по векторным представлениям.
    
    Args:
        embeddings (np.ndarray): Массив эмбеддингов
        output_path (str): Путь для сохранения индекса
    """
    # Определяем размерность эмбеддингов
    dimension = embeddings.shape[1]
    
    # В зависимости от количества векторов выбираем тип индекса
    n_vectors = embeddings.shape[0]
    
    # Нормализуем векторы для использования косинусного расстояния
    faiss.normalize_L2(embeddings)
    
    # Выбираем лучший тип индекса в зависимости от размера данных
    if n_vectors < 1000:
        # Для маленьких наборов данных используем простой индекс
        index = faiss.IndexFlatIP(dimension)
    else:
        # Для больших наборов данных используем более сложный индекс HNSW
        # HNSW обеспечивает быстрый поиск с небольшой потерей качества
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 - количество соседей
        index.hnsw.efConstruction = 100  # Качество построения (выше = лучше, но медленнее)
        index.hnsw.efSearch = 128  # Качество поиска
    
    # Добавляем векторы в индекс
    print(f"Добавление {n_vectors} векторов в индекс...")
    index.add(embeddings)
    
    # Сохраняем индекс
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Сохранение индекса в {output_path}")
    faiss.write_index(index, output_path)
    
    print("Индекс FAISS успешно создан и сохранен.")

if __name__ == "__main__":
    # Загружаем чанки
    chunks_file = "output/penal_code_chunks.json"
    if not os.path.exists(chunks_file):
        alternative_path = "penal_code_chunks.json"
        if os.path.exists(alternative_path):
            chunks_file = alternative_path
        else:
            print(f"Ошибка: Файл с чанками не найден ни в {chunks_file}, ни в {alternative_path}")
            exit(1)
    
    print(f"Загрузка чанков из {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"Загружено {len(chunks)} чанков.")
    
    # Создаем эмбеддинги - выбираем подходящую модель для испанского языка
    embeddings = create_embeddings(
        chunks,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size=16,
        show_progress=True
    )
    
    # Создаем FAISS индекс
    create_faiss_index(embeddings, output_path="output/penal_code.index")
    
    print("Готово! Теперь вы можете использовать search.py для поиска по Уголовному кодексу.")
    print("Или legal_bot.py для использования юридического ассистента с контекстным поиском.")