import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def search_similar_chunks(
    question: str,
    chunks: list,
    index_path: str = "legal.index",
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5
) -> list:
    # Загружаем FAISS index
    index = faiss.read_index(index_path)

    # Загружаем модель эмбедингов
    model = SentenceTransformer(model_name)

    # Считаем эмбединги вопроса
    question_embedding = model.encode([question], convert_to_numpy=True)

    # Ищем ближайшие K
    distances, indices = index.search(question_embedding, top_k)

    # Собираем результаты
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(chunks):  # на всякий случай
            result = chunks[idx].copy()
            result["distance"] = float(dist)
            results.append(result)

    print()

    return results



def expand_chunks_with_neighbors(results: list, all_chunks: list, window: int = 1) -> list:
    """
    Расширяет найденные чанки соседями внутри того же capítulo.
    :param results: список чанков из FAISS поиска (с chunk_index, capitulo, etc.)
    :param all_chunks: полный список чанков (из final_chunks.json)
    :param window: сколько соседей с каждой стороны брать
    :return: список расширенных чанков без повторов
    """
    expanded = []
    seen_ids = set()

    # Группируем все чанки по capítulo
    capitulo_groups = {}
    for ch in all_chunks:
        cap_id = f"{ch['libro']}|{ch['titulo']}|{ch['capitulo']}"
        capitulo_groups.setdefault(cap_id, []).append(ch)

    for res in results:
        cap_id = f"{res['libro']}|{res['titulo']}|{res['capitulo']}"
        chunk_list = capitulo_groups.get(cap_id, [])

        for offset in range(-window, window + 1):
            idx = res["chunk_index"] + offset

            if 0 <= idx < len(chunk_list):
                chunk = chunk_list[idx]
                uid = f"{cap_id}|{chunk['chunk_index']}"  # уникальный ID чанка
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    expanded.append(chunk)

    return expanded
