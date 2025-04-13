"""
Основной скрипт обработки данных для Уголовного кодекса (Código Penal)

Этот скрипт объединяет все этапы обработки:
1. Загрузка структурированных статей
2. Группировка статей по разделам
3. Разбиение на чанки для эмбеддингов
4. Сохранение результатов
"""

import json
import os
from group import group_articles_by_capitulo
from chunking import create_final_chunks

def main():
    # Создаем каталог для результатов, если он не существует
    os.makedirs('output', exist_ok=True)
    
    print("Запуск основной обработки документа...")
    
    # 1. Загружаем статьи из файла
    print("Загрузка структурированных статей...")
    with open("articles_for_embeddings.json", "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"Загружено {len(articles)} статей")
    
    # 2. Группировка статей по разделам
    print("Группировка статей по структуре libro -> titulo -> capitulo...")
    grouped_articles = group_articles_by_capitulo(articles)
    print(f"Создано {len(grouped_articles)} групп")
    
    # Сохраняем промежуточный результат
    grouped_path = 'output/grouped_articles.json'
    print(f"Сохранение групп в {grouped_path}...")
    with open(grouped_path, 'w', encoding='utf-8') as f:
        json.dump(grouped_articles, f, ensure_ascii=False, indent=2)
    
    # 3. Разбиение на чанки для эмбеддингов
    print("Создание чанков оптимального размера для эмбеддингов...")
    final_chunks = create_final_chunks(grouped_articles)
    print(f"Создано {len(final_chunks)} чанков")
    
    # 4. Сохранение финальных чанков
    final_path = 'output/penal_code_chunks.json'
    print(f"Сохранение финальных чанков в {final_path}...")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)
    
    # 5. Анализ результатов
    avg_chunk_size = sum(len(chunk['text']) for chunk in final_chunks) / len(final_chunks)
    max_chunk_size = max(len(chunk['text']) for chunk in final_chunks)
    min_chunk_size = min(len(chunk['text']) for chunk in final_chunks)
    
    print("\nСтатистика по чанкам:")
    print(f"- Общее количество: {len(final_chunks)}")
    print(f"- Средний размер: {avg_chunk_size:.2f} символов")
    print(f"- Максимальный размер: {max_chunk_size} символов")
    print(f"- Минимальный размер: {min_chunk_size} символов")
    
    print("\n✅ Обработка успешно завершена!")
    print(f"Теперь вы можете использовать файл {final_path} для создания эмбеддингов.")

if __name__ == "__main__":
    main()