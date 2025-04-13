import json

from group import group_articles_by_capitulo
from build import build_chunks_with_metadata

# 1. Загрузить статьи из файла
with open("articles_for_embeddings.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# 2. Группировка по capitulos
grouped_articles = group_articles_by_capitulo(articles)

# 3. Построение финальных чанков с метаданными
final_chunks = build_chunks_with_metadata(grouped_articles, chunk_size=1000, overlap=200)

# 4. Сохраняем в JSON
with open("final_chunks.json", "w", encoding="utf-8") as f:
    json.dump(final_chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Сохранено чанков: {len(final_chunks)}")
