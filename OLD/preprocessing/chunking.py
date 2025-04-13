import json

with open('grouped_articles.json', 'r') as f:
    grouped_articles = json.load(f)

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # следующий чанк начинается на (chunk_size - overlap) символов позже
        start += chunk_size - overlap

    return chunks



# for key, value in grouped_articles.items():
#     print(key)
#     print(value)
#     break