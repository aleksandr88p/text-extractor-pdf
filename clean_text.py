import re

def clean_text_file(input_file, output_file):
    """
    Очищает текстовый файл от маркеров страниц и номеров страниц.
    
    Args:
        input_file (str): Путь к входному файлу
        output_file (str): Путь к выходному файлу
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Удаляем маркеры страниц (=== Страница X ===)
    content = re.sub(r'\n=== Страница \d+ ===\n', '\n', content)
    
    # Удаляем одиночные номера страниц
    content = re.sub(r'\n\d+\n(?=\w)', '\n', content)
    
    # Убираем множественные пустые строки
    content = re.sub(r'\n\s*\n', '\n\n', content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content.strip())

if __name__ == "__main__":
    input_file = "extracted_text.txt"
    output_file = "cleaned_text.txt"
    clean_text_file(input_file, output_file)
    print(f"Текст очищен и сохранен в {output_file}") 