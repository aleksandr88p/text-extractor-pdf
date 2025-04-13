import json
from search import search_similar_chunks, expand_chunks_with_neighbors

# Загружаем чанки (те же, что использовали при построении index)
with open("../preprocessing/final_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Вопрос пользователя
question = "¿Qué dice la ley sobre la retroactividad de las normas?"

# Поиск похожих чанков
results = search_similar_chunks(question, chunks, top_k=2)

# Добавляем соседей (только в том же capítulo)
expanded = expand_chunks_with_neighbors(results, chunks, window=1)

# Выводим результат
for i, ch in enumerate(expanded, 1):
    print(f"\n--- chunk {i} ---")
    print(f"{ch['libro']} > {ch['capitulo']} (chunk {ch['chunk_index']})")
    print(ch['text'][:300], "...")
