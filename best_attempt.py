import re
import json
from typing import Dict, List

# Путь к очищенному файлу
cleaned_text_file = 'cleaned_text.txt'

def find_main_sections(input_file: str) -> dict:
    """
    Находит основные секции в тексте закона:
    - TÍTULO PRELIMINAR
    - LIBRO I
    - LIBRO II
    - LIBRO III
    
    Args:
        input_file (str): путь к очищенному файлу с текстом
        
    Returns:
        dict: словарь с найденными секциями и их позициями
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    sections = {}

    # TÍTULO PRELIMINAR
    titulo_prel = re.search(r'TÍTULO PRELIMINAR', text)
    if titulo_prel:
        sections['TÍTULO PRELIMINAR'] = titulo_prel.start()

    # LIBROS
    libro_patterns = [
        r'LIBRO I\b',
        r'LIBRO II\b', 
        r'LIBRO III\b'
    ]
    for pattern in libro_patterns:
        match = re.search(pattern, text)
        if match:
            sections[match.group()] = match.start()
    
    print("\nНайдены следующие секции:")
    for section, pos in sorted(sections.items(), key=lambda x: x[1]):
        print(f"- {section}")
    
    return sections

def extract_sections_text(input_file: str, sections: dict) -> dict:
    """
    Извлекает текст каждой секции, используя найденные позиции
    
    Args:
        input_file (str): путь к файлу с текстом
        sections (dict): словарь с позициями секций
    
    Returns:
        dict: словарь с текстами секций
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    sections_text = {}
    sorted_sections = sorted(sections.items(), key=lambda x: x[1])
    
    for i, (section_name, start_pos) in enumerate(sorted_sections):
        if i == len(sorted_sections) - 1:
            sections_text[section_name] = text[start_pos:]
        else:
            next_section_pos = sorted_sections[i + 1][1]
            sections_text[section_name] = text[start_pos:next_section_pos]
    
    print("\nИзвлечены тексты следующих секций:")
    for section, content in sections_text.items():
        print(f"- {section}: {len(content)} символов")
    
    return sections_text

def find_subsections(sections_text: dict) -> dict:
    """
    Находит подразделы внутри основных секций:
    - Для TÍTULO PRELIMINAR берем весь текст
    - Для LIBRO ищем TÍTULO и внутри них CAPÍTULO
    
    Args:
        sections_text (dict): словарь с текстами основных секций
    
    Returns:
        dict: иерархическая структура всего документа
    """
    structure = {}
    
    for section_name, section_text in sections_text.items():
        print(f"\nОбрабатываю секцию: {section_name}")
        structure[section_name] = {}
        
        # Для TÍTULO PRELIMINAR берем весь текст
        if section_name == 'TÍTULO PRELIMINAR':
            structure[section_name] = {
                "text": section_text,
                "articles": {}  # Здесь будут статьи
            }
        
        # Для LIBRO ищем TÍTULO и CAPÍTULO
        else:
            # Ищем все TÍTULO
            titulos = list(re.finditer(r'TÍTULO\s+(?:PRIMERO|[IVX]+)', section_text))
            
            for i, titulo_match in enumerate(titulos):
                titulo_name = titulo_match.group()
                titulo_start = titulo_match.start()
                
                # Определяем где заканчивается этот TÍTULO
                if i < len(titulos) - 1:
                    titulo_end = titulos[i+1].start()
                else:
                    titulo_end = len(section_text)
                
                titulo_text = section_text[titulo_start:titulo_end]
                print(f"  Titulo found: {titulo_name}")
                
                # Сохраняем TÍTULO и создаем структуру для CAPÍTULO
                structure[section_name][titulo_name] = {
                    "text": titulo_text,
                    "capitulos": {},  # Здесь будут CAPÍTULO
                    "articles": {}  # Статьи на уровне TÍTULO
                }
                
                # Ищем все CAPÍTULO внутри этого TÍTULO
                capitulos = list(re.finditer(r'CAPÍTULO\s+(?:PRIMERO|[IVX]+)', titulo_text))
                
                for j, capitulo_match in enumerate(capitulos):
                    capitulo_name = capitulo_match.group()
                    capitulo_start = capitulo_match.start()
                    
                    # Определяем где заканчивается этот CAPÍTULO
                    if j < len(capitulos) - 1:
                        capitulo_end = capitulos[j+1].start()
                    else:
                        capitulo_end = len(titulo_text)
                    
                    capitulo_text = titulo_text[capitulo_start:capitulo_end]
                    print(f"    Capitulo found: {capitulo_name}")
                    
                    # Сохраняем CAPÍTULO
                    structure[section_name][titulo_name]["capitulos"][capitulo_name] = {
                        "text": capitulo_text,
                        "articles": {}  # Статьи на уровне CAPÍTULO
                    }
    
    return structure

def extract_articles(structure: dict) -> dict:
    """
    Извлекает статьи из всех разделов документа
    
    Args:
        structure (dict): иерархическая структура документа
    
    Returns:
        dict: структура с добавленными статьями
    """
    print("\nИзвлечение статей...")
    article_count = 0
    
    # Паттерн для поиска статей
    article_pattern = r'Artículo\s+(\d+)[^\n]*\n((?:(?!Artículo\s+\d+)[^\n]|\n)*)'
    
    # Для TÍTULO PRELIMINAR
    if 'TÍTULO PRELIMINAR' in structure:
        text = structure['TÍTULO PRELIMINAR']['text']
        articles = extract_articles_from_text(text, article_pattern)
        structure['TÍTULO PRELIMINAR']['articles'] = articles
        article_count += len(articles)
    
    # Для остальных LIBRO
    for section_name, section_data in structure.items():
        if section_name.startswith('LIBRO'):
            for titulo_name, titulo_data in section_data.items():
                # Ищем статьи в тексте TÍTULO
                articles = extract_articles_from_text(titulo_data['text'], article_pattern)
                titulo_data['articles'] = articles
                article_count += len(articles)
                
                # Ищем статьи в каждом CAPÍTULO
                for capitulo_name, capitulo_data in titulo_data['capitulos'].items():
                    articles = extract_articles_from_text(capitulo_data['text'], article_pattern)
                    capitulo_data['articles'] = articles
                    article_count += len(articles)
    
    print(f"Всего найдено статей: {article_count}")
    return structure

def extract_articles_from_text(text: str, pattern: str) -> dict:
    """
    Извлекает статьи из текста
    
    Args:
        text (str): текст для поиска статей
        pattern (str): регулярное выражение для поиска статей
    
    Returns:
        dict: словарь статей (номер: текст)
    """
    articles = {}
    matches = re.finditer(pattern, text, re.MULTILINE)
    
    for match in matches:
        number = match.group(1)
        content = match.group(2).strip()
        articles[number] = content
        
    return articles

def create_flat_structure(structure: dict) -> list:
    """
    Создает плоский список статей с метаданными для эмбеддингов
    
    Args:
        structure (dict): иерархическая структура документа
    
    Returns:
        list: список статей с метаданными
    """
    flat_list = []
    
    # Для TÍTULO PRELIMINAR
    if 'TÍTULO PRELIMINAR' in structure:
        for article_num, article_text in structure['TÍTULO PRELIMINAR']['articles'].items():
            flat_list.append({
                "libro": "TÍTULO PRELIMINAR",
                "titulo": "",
                "capitulo": "",
                "article_number": article_num,
                "text": article_text
            })
    
    # Для LIBRO
    for section_name, section_data in structure.items():
        if section_name.startswith('LIBRO'):
            for titulo_name, titulo_data in section_data.items():
                # Статьи на уровне TÍTULO
                for article_num, article_text in titulo_data['articles'].items():
                    flat_list.append({
                        "libro": section_name,
                        "titulo": titulo_name,
                        "capitulo": "",
                        "article_number": article_num,
                        "text": article_text
                    })
                
                # Статьи в CAPÍTULO
                for capitulo_name, capitulo_data in titulo_data['capitulos'].items():
                    for article_num, article_text in capitulo_data['articles'].items():
                        flat_list.append({
                            "libro": section_name,
                            "titulo": titulo_name,
                            "capitulo": capitulo_name,
                            "article_number": article_num,
                            "text": article_text
                        })
    
    print(f"\nСоздан плоский список из {len(flat_list)} статей")
    return flat_list

def save_to_json(data: dict, output_file: str):
    """Сохраняет данные в JSON файл"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Данные сохранены в {output_file}")

def main():
    print("Запуск обработки текста...")
    
    # Шаг 1: Находим основные секции
    sections = find_main_sections(cleaned_text_file)
    
    # Шаг 2: Извлекаем текст каждой секции
    sections_text = extract_sections_text(cleaned_text_file, sections)
    
    # Шаг 3: Находим подразделы
    structure = find_subsections(sections_text)
    
    # Шаг 4: Извлекаем статьи
    structure = extract_articles(structure)
    
    # Сохраняем полную структуру
    save_to_json(structure, 'structured_document.json')
    
    # Шаг 5: Создаем плоский список для эмбеддингов
    flat_list = create_flat_structure(structure)
    
    # Сохраняем плоский список
    save_to_json(flat_list, 'articles_for_embeddings.json')
    
    print("\nОбработка завершена!")

if __name__ == "__main__":
    main()