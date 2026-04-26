import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query, k=3):
        D, I = self.index.search(query, k)
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

embed_model_id = ... # nazwa modelu
model_kwargs = {"device": "cpu", "trust_remote_code": True}

def create_index(documents):
    embeddings = ... # załadowanie modelu embeddingowego
    texts = ... # wartości tekstowe wszystkich dokumentów
    metadata = ... # metadane wszystkich dokumentów, czyli słownik {filename:... , text:...}

    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    index = faiss....# ustawienie indeksu przeszukwania
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    embeddings = ... # załadowanie modelu embeddingowego
    query_embedding = ... # embeddowanie zapytania (query)
    results = ... # zwrócenie wyników przeuszkiwania
    return results