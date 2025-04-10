import fitz

def extract_and_save_text(pdf_path, output_path):
    """
    Извлекает текст из PDF и сохраняет его в текстовый файл.
    Каждая страница отделяется маркером.
    """
    pdf_doc = fitz.open(pdf_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for page_num, page in enumerate(pdf_doc, 1):
            text = page.get_text()
            # Записываем номер страницы и её содержимое
            f.write(f"\n=== Страница {page_num} ===\n")
            f.write(text)
            f.write("\n")
    
    pdf_doc.close()
    print(f"Текст сохранен в файл: {output_path}")

if __name__ == "__main__":
    pdf_path = "codigo_penal.pdf"
    output_path = "extracted_text.txt"
    extract_and_save_text(pdf_path, output_path) 