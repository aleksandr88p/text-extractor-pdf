

from chunking import split_text_into_chunks

def build_chunks_with_metadata(grouped_articles: dict, chunk_size: int = 1000, overlap: int = 200) -> list:
    all_chunks = []

    for group_key, group_data in grouped_articles.items():
        chunks = split_text_into_chunks(group_data["full_text"], chunk_size=chunk_size, overlap=overlap)

        for idx, chunk in enumerate(chunks):
            chunk_dict = {
                "libro": group_data["libro"],
                "titulo": group_data["titulo"],
                "capitulo": group_data["capitulo"],
                "article_numbers": group_data["article_numbers"],
                "chunk_index": idx,
                "text": chunk
            }
            all_chunks.append(chunk_dict)

    return all_chunks
