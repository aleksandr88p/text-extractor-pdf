"""
Юридический чат-бот для ответа на вопросы по Уголовному кодексу (Código Penal)

Этот модуль предоставляет функции для создания юридического ассистента,
который отвечает на вопросы по Уголовному кодексу, используя индексы FAISS.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from search import (
    search_similar_chunks, 
    search_by_article_number, 
    expand_chunks_with_neighbors,
    extract_intent,
    calculate_relevance_score
)

class LegalAssistant:
    """
    Юридический ассистент для ответов на вопросы по Уголовному кодексу
    """
    
    def __init__(
        self, 
        chunks_file: str = "output/penal_code_chunks.json",
        index_path: str = "output/penal_code.index",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Инициализирует юридического ассистента.
        
        Args:
            chunks_file (str): Путь к файлу с чанками текста
            index_path (str): Путь к FAISS индексу
            model_name (str): Название модели для эмбеддингов
        """
        # Загрузка чанков
        if not os.path.exists(chunks_file):
            alternative_path = "penal_code_chunks.json"
            if os.path.exists(alternative_path):
                chunks_file = alternative_path
            else:
                raise FileNotFoundError(f"Файл с чанками не найден: {chunks_file}")
        
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
            
        # Проверка индекса
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Индекс не найден: {index_path}")
            
        self.index_path = index_path
        self.model_name = model_name
        
        # Словарь для хранения кэша запросов
        self.query_cache = {}
        
        # Словарь типичных юридических вопросов и их перефразировок для лучшего поиска
        self.legal_questions = {
            "срок давности": [
                "plazo de prescripción", 
                "prescripción de delito",
                "tiempo para procesar"
            ],
            "наказание за": [
                "pena por",
                "sanción aplicable",
                "castigo establecido"
            ],
            "что считается": [
                "definición de",
                "concepto legal de",
                "qué constituye"
            ]
        }
        
        print(f"Юридический ассистент инициализирован. Загружено {len(self.chunks)} чанков текста.")
    
    def answer_question(self, question: str) -> str:
        """
        Отвечает на юридический вопрос, используя данные Уголовного кодекса.
        
        Args:
            question (str): Вопрос пользователя
            
        Returns:
            str: Ответ на вопрос с цитатами из Уголовного кодекса
        """
        # Проверяем кэш запросов
        if question in self.query_cache:
            return self.query_cache[question]
        
        # Определяем тип запроса
        intent, param = extract_intent(question)
        
        # Поиск по статье
        if intent == "article_search" and param:
            results = search_by_article_number(param, self.chunks)
            if not results:
                # Если статья не найдена, пробуем семантический поиск
                results = search_similar_chunks(
                    f"Artículo {param}", 
                    self.chunks, 
                    self.index_path, 
                    self.model_name,
                    top_k=3
                )
        else:
            # Для запросов о сроках давности используем специальные перефразировки
            if intent == "prescription_search":
                expanded_queries = self._expand_legal_query(question, "срок давности")
                # Объединяем результаты из всех запросов
                all_results = []
                for query in expanded_queries:
                    query_results = search_similar_chunks(
                        query, 
                        self.chunks, 
                        self.index_path,
                        self.model_name,
                        top_k=2
                    )
                    all_results.extend(query_results)
                
                # Удаляем дубликаты и сортируем по релевантности
                seen_indices = set()
                results = []
                for res in sorted(all_results, key=lambda x: x.get("distance", float('inf'))):
                    idx = (res["libro"], res["titulo"], res["capitulo"], res["chunk_index"])
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        results.append(res)
                
                results = results[:3]  # Ограничиваем тремя самыми релевантными результатами
            else:
                # Обычный семантический поиск
                results = search_similar_chunks(
                    question, 
                    self.chunks, 
                    self.index_path,
                    self.model_name,
                    top_k=3
                )
        
        # Если нашли результаты, расширяем их контекстом
        if results:
            expanded_results = expand_chunks_with_neighbors(results, self.chunks, window=1)
            answer = self._format_answer(question, expanded_results)
        else:
            answer = "К сожалению, я не нашёл информации по вашему запросу в Уголовном кодексе. " + \
                     "Попробуйте сформулировать вопрос иначе или уточните, что именно вас интересует."
        
        # Сохраняем в кэш
        self.query_cache[question] = answer
        
        return answer
    
    def _expand_legal_query(self, question: str, query_type: str) -> List[str]:
        """
        Расширяет запрос юридическими формулировками для улучшения поиска.
        
        Args:
            question (str): Исходный вопрос
            query_type (str): Тип вопроса (срок давности, наказание и т.д.)
            
        Returns:
            List[str]: Список расширенных запросов
        """
        expanded_queries = [question]
        
        if query_type in self.legal_questions:
            for phrase in self.legal_questions[query_type]:
                expanded_queries.append(f"{question} {phrase}")
        
        return expanded_queries
    
    def _format_answer(self, question: str, results: List[Dict]) -> str:
        """
        Форматирует ответ на основе найденных фрагментов Уголовного кодекса.
        
        Args:
            question (str): Вопрос пользователя
            results (List[Dict]): Найденные фрагменты
            
        Returns:
            str: Отформатированный ответ с цитатами
        """
        answer_parts = []
        
        # Определяем общий заголовок ответа в зависимости от типа вопроса
        intent, param = extract_intent(question)
        
        if intent == "article_search":
            answer_parts.append(f"📚 **По вашему запросу о статье {param} Уголовного кодекса:**\n")
        elif intent == "prescription_search":
            answer_parts.append("📚 **По вашему вопросу о сроках давности:**\n")
        else:
            answer_parts.append("📚 **По вашему запросу я нашел следующую информацию в Уголовном кодексе:**\n")
        
        # Добавляем самые релевантные фрагменты
        seen_texts = set()
        relevance_threshold = 20.0  # Минимальная релевантность для включения в ответ
        
        for i, chunk in enumerate(results[:5], 1):  # Ограничиваем пятью фрагментами
            text = self._clean_text_for_answer(chunk["text"])
            
            # Пропускаем дубликаты текста
            if text in seen_texts:
                continue
                
            # Проверяем релевантность
            relevance = chunk.get("relevance_score", 0)
            if relevance == 0 and "distance" in chunk:
                relevance = calculate_relevance_score(chunk["distance"])
                
            if relevance < relevance_threshold and i > 1:
                continue  # Пропускаем нерелевантные результаты (кроме первого)
                
            seen_texts.add(text)
            
            # Формируем структурированное название раздела
            section_name = []
            if chunk["libro"]:
                section_name.append(chunk["libro"])
            if chunk["titulo"] and chunk["titulo"] != chunk["libro"]:
                section_name.append(chunk["titulo"])
            if chunk["capitulo"] and chunk["capitulo"] not in [chunk["libro"], chunk["titulo"]]:
                section_name.append(chunk["capitulo"])
                
            section_str = " > ".join(section_name)
            
            # Добавляем в ответ
            context_label = " (контекст)" if chunk.get("is_context") else ""
            answer_parts.append(f"### {section_str}{context_label}\n")
            answer_parts.append(f"**Статьи: {', '.join(chunk['article_numbers'])}**\n")
            answer_parts.append(f"{text}\n")
        
        # Добавляем заключение
        answer_parts.append("\n---\n")
        answer_parts.append("Это информация из Уголовного кодекса, которая относится к вашему вопросу. ")
        answer_parts.append("Обратите внимание, что я могу только предоставить текст закона, но не давать юридических консультаций. ")
        answer_parts.append("Для получения официальной консультации обратитесь к юристу.")
        
        return "\n".join(answer_parts)
    
    def _clean_text_for_answer(self, text: str) -> str:
        """
        Очищает и форматирует текст для ответа.
        
        Args:
            text (str): Исходный текст
            
        Returns:
            str: Очищенный и отформатированный текст
        """
        # Удаляем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Восстанавливаем переносы строк в нужных местах
        text = re.sub(r'(\d+\.)\s+', r'\n\1 ', text)
        text = re.sub(r'(Artículo \d+\.?)\s+', r'\n\1 ', text)
        
        return text

if __name__ == "__main__":
    print("Инициализация юридического ассистента...")
    assistant = LegalAssistant()
    
    print("\n🤖 Юридический ассистент по Уголовному кодексу")
    print("Задайте вопрос о законе, статье или юридической ситуации.")
    
    while True:
        question = input("\nВаш вопрос (или 'q' для выхода): ")
        
        if question.lower() in ['q', 'quit', 'exit']:
            break
            
        answer = assistant.answer_question(question)
        print("\n" + answer)