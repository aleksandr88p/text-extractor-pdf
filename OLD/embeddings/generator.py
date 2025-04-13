from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


def build_faiss_index_from_chunks(chunks: list, model_name: str = "all-MiniLM-L6-v2", index_path: str = "legal.index") -> None:
    model = SentenceTransformer(model_name)

    # Получаем список текстов
    texts = [chunk["text"] for chunk in chunks]

    # Считаем эмбединги (батчом)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Создаём FAISS index (index с L2 расстоянием)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Сохраняем index
    faiss.write_index(index, index_path)
    print(f"✅ FAISS index сохранён в файл: {index_path}")



file_path = "../preprocessing/final_chunks.json"

# Загружаем чанки
with open(file_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Генерим FAISS index
build_faiss_index_from_chunks(chunks, model_name="all-MiniLM-L6-v2", index_path="legal.index")

