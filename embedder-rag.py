import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query_vector, k=3):
        # FAISS oczekuje macierzy 2D, więc robimy reshape zapytania
        D, I = self.index.search(query_vector, k)
        results = []
        for idx in I[0]:
            if idx != -1: # Sprawdzamy, czy znaleziono indeks
                results.append(self.metadata[idx])
        return results

# Wybór lekkiego i popularnego modelu
embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu", "trust_remote_code": True}

def create_index(documents):
    # Załadowanie modelu embeddingowego
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
    
    # Zakładamy, że 'documents' to lista obiektów/słowników z kluczami 'text' i 'filename'
    texts = [doc['text'] for doc in documents]
    metadata = documents # Przechowujemy całe obiekty jako metadane

    # Generowanie embeddingów dla wszystkich tekstów
    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    # Inicjalizacja indeksu FAISS (L2 to odległość euklidesowa)
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Dodanie wektorów do indeksu
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index_obj, k=3):
    # Załadowanie tego samego modelu co przy tworzeniu indeksu
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
    
    # Zamiana pytania tekstowego na wektor i dostosowanie kształtu do FAISS (1, dimension)
    query_embedding = np.array([embeddings.embed_query(query)]).astype("float32")
    
    # Wywołanie wyszukiwania z klasy FAISSIndex
    results = faiss_index_obj.similarity_search(query_embedding, k)
    return results
