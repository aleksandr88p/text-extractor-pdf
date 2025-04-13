import re
import json
from typing import List, Dict

clean_text = 'cleaned_text.txt'
section_names = ["TÍTULO PRELIMINAR", "LIBRO I", "LIBRO II", "LIBRO III"]


def find_main_sections(input_file: str, section_names: List[str]) -> dict:
    """
    находит основные главные разделы в книге
    :param input_file: путь к файлу
    :param section_names: список названий разделов
    :return: словарь с основными разделами и их позициями
    """

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    sections = {}
    for section in section_names:
        # print(f"Searching for section: {section}")
        pattern = re.search(rf'{section}', text)
        if pattern:
            # print(f"Found section: {section}")
            start = pattern.start()
            # print(start)
            sections[section] = start
            # print(f"Found section '{section}' at position {start}")

    return sections





def extract_section_text(input_file: str, sections: dict) -> dict:
    """
    извлекает текст из разделов
    :param input_file: путь к файлу
    :param sections: словарь с основными разделами и их позициями
    :return: словарь с текстом разделов
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    section_texts = {}

    # получаю список секйций и их позиций
    sorted_sections = sorted(sections.items(), key=lambda x: x[1])
    
    for i, (section_name, start_pos) in enumerate(sorted_sections):
        # если последняя секция, то до конца текста
        if i == len(sorted_sections) - 1:
            section_texts[section_name] = text[start_pos:]
        else:
            # следующая секция
            next_section_name, next_start_pos = sorted_sections[i + 1]
            section_texts[section_name] = text[start_pos:next_start_pos]
    print("\nИзвлечены тексты следующих секций:")
    for section, content in section_texts.items():
        print(f"- {section}: {len(content)} символов")
        
    return section_texts


    return section_texts
find_main_sections(clean_text, section_names)
sections = find_main_sections(clean_text, section_names)
extracte_by_sections = extract_section_text(clean_text, sections)




# продолжить с другой функцией, которая будет смотерть уже подразделы в секциях.
# как пример можно взять файл из старого проекта OLD/api/best_attempt

# print(extracte_by_sections.keys())
def find_subsections(sectoinos_text: dict) -> dict:
    """
    для TÍTULO PRELIMINAR беру сразу весь текст, так как там нет TÍTULO и CAPÍTULO
    для LIBRO I, II ищу TÍTULO и CAPÍTULO и делаю вложенные словари с текстом
    args:
        sectoinos_text (dict): словарь с текстом секций
    return:
        dict: иерархическая структура всего документа
    """
    # Создаем структуру для хранения иерархии документа
    structure = {}
    
    # Обрабатываем каждую секцию
    for section_name, section_text in sectoinos_text.items():
        print(f"\nОбрабатываю секцию: {section_name}")
        structure[section_name] = {}
        
        # Если это TÍTULO PRELIMINAR, берем весь текст
        if section_name == 'TÍTULO PRELIMINAR':
            structure[section_name] = {
                "text": section_text,
                "articles": {}  # Здесь будут статьи
            }
        
        # Для LIBRO ищем TÍTULO и CAPÍTULO
        elif section_name.startswith('LIBRO'):
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


