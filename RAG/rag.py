import openai
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

# Устанавливаем ключ API OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


class EmbeddingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if not cls._instance:
            cls._instance = super(EmbeddingService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize_embeddings()
        return cls._instance

    def _initialize_embeddings(self):
        """Инициализация эмбеддингов при создании объекта"""
        # Чтение данных из файла
        text = self.read_data_from_file('data.txt')

        # Разделение текста на чанки с оверлапом
        chunks = self.split_text_into_chunks(text, chunk_size=250, chunk_overlap=50)

        # Конвертируем строки в документы
        self.documents = [Document(page_content=chunk) for chunk in chunks]

        # Генерируем эмбеддинги для всех документов
        self.embeddings = self.get_embeddings_for_documents(self.documents)

        # Создаем BM25 индекс
        tokenized_documents = [doc.page_content.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_documents)

    def read_data_from_file(self, filepath):
        """Чтение всего текста из файла"""
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()

    def split_text_into_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Разделение текста на чанки с оверлапом"""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)

    def get_openai_embedding(self, text, model="text-embedding-3-small"):
        """Получение эмбеддингов через OpenAI API"""
        client = openai.OpenAI()
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding

    def get_embeddings_for_documents(self, documents: List[Document], model: str = "text-embedding-3-small") -> np.ndarray:
        """Получаем эмбеддинги для всех документов"""
        embeddings = []
        for doc in documents:
            embedding = self.get_openai_embedding(doc.page_content, model)
            embeddings.append(embedding)
        return np.array(embeddings)

    def fusion_retrieval(self, query: str, k: int = 28, alpha: float = 0.5) -> List[Document]:
        """
        Комбинированный поиск: семантический (по эмбеддингам) и по ключевым словам (BM25)
        """

        # Step 1: Perform BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Step 2: Perform vector search (semantic search using embeddings)
        query_embedding = self.get_openai_embedding(query)
        vector_scores = np.linalg.norm(self.embeddings - query_embedding, axis=1)

        # Нормализуем векторные оценки
        vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))

        # Нормализуем BM25 оценки
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

        # Step 3: Combine the scores
        combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

        # Step 4: Rank documents based on the combined scores
        sorted_indices = np.argsort(combined_scores)[::-1]

        # Step 5: Return the top k documents
        x = [self.documents[i] for i in sorted_indices[:k]]
        print(x)
        return x


# Пример использования:
if __name__ == "__main__":
    embedding_service = EmbeddingService()  # Создаем или получаем уже созданный экземпляр
    query = "What are the effects of climate change?"

    # Выполняем поиск
    top_docs = embedding_service.fusion_retrieval(query, k=5, alpha=0.5)

    # Выводим результаты
    print("Top documents:")
    for doc in top_docs:
        print(doc.page_content)
