from collections import defaultdict
import json


def group_articles_by_capitulo(articles: list) -> dict:
    grouped = defaultdict(lambda: {
        "libro": "",
        "titulo": "",
        "capitulo": "",
        "articles": [],
        "article_numbers": [],
        "full_text": ""
    })

    for i, item in enumerate(articles):
        print(f"Processing article {i+1} of {len(articles)}")
        key = f"{item['libro']} | {item['titulo']} | {item['capitulo']}"
        grouped[key]["libro"] = item["libro"]
        grouped[key]["titulo"] = item["titulo"]
        grouped[key]["capitulo"] = item["capitulo"]
        grouped[key]["articles"].append(item["text"])
        grouped[key]["article_numbers"].append(item["article"])

    # Собираем полный текст
    for group in grouped.values():
        group["full_text"] = "\n\n".join(group["articles"])

    return dict(grouped)


with open('articles_for_embeddings.json', 'r') as f:
    articles = json.load(f)

grouped = group_articles_by_capitulo(articles)

with open('grouped_articles.json', 'w') as f:
    json.dump(grouped, f, indent=4)
