"""
Группировка статей по иерархической структуре книги

Этот скрипт загружает статьи из плоского списка articles_for_embeddings.json и группирует их 
по структуре "libro -> titulo -> capitulo" для дальнейшей обработки.
"""

from collections import defaultdict
import json

def group_articles_by_capitulo(articles: list) -> dict:
    """
    Группирует статьи по их структурному расположению в документе.

    Алгоритм:
    1. Создает словарь с вложенной структурой libro -> titulo -> capitulo
    2. Для каждой статьи определяет её место в структуре и добавляет в соответствующую группу
    3. Формирует полный текст каждой группы, объединяя тексты всех её статей

    Args:
        articles (list): Список статей из articles_for_embeddings.json

    Returns:
        dict: Словарь группировок статей по структурным элементам
    """

    # Создаем структуру для хранения сгруппированных статей
    grouped = defaultdict(lambda: {
        "libro": "",
        "titulo": "",
        "capitulo": "",
        "articles": [],              # Список текстов статей
        "article_numbers": [],        # Список номеров статей
        "full_text": ""              # Полный текст всех статей в группе
    })

    # Обрабатываем каждую статью
    for i, item in enumerate(articles):
        print(f"Обработка статьи {i+1} из {len(articles)}")
        
        # Создаем уникальный ключ для группировки
        key = f"{item['libro']} | {item['titulo']} | {item['capitulo']}"
        
        # Заполняем информацию о группе
        grouped[key]["libro"] = item["libro"]
        grouped[key]["titulo"] = item["titulo"]
        grouped[key]["capitulo"] = item["capitulo"]
        
        # Добавляем текст и номер статьи
        grouped[key]["articles"].append(item["text"])
        grouped[key]["article_numbers"].append(item["article_number"])  # Обратите внимание на изменение ключа

    # Собираем полный текст для каждой группы
    for group in grouped.values():
        group["full_text"] = "\n\n".join(group["articles"])

    # Преобразуем defaultdict в обычный словарь для JSON сериализации
    return dict(grouped)


if __name__ == "__main__":
    # Загружаем статьи из файла
    print("Загрузка статей из articles_for_embeddings.json...")
    with open('articles_for_embeddings.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Группируем статьи
    print(f"Начинаю группировку {len(articles)} статей...")
    grouped = group_articles_by_capitulo(articles)
    print(f"Статьи сгруппированы в {len(grouped)} групп")
    
    # Сохраняем результат
    output_path = 'grouped_articles.json'
    print(f"Сохраняю результат в {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(grouped, f, indent=4, ensure_ascii=False)
    
    print("Группировка завершена успешно!")