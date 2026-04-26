import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import os

# --- KONFIGURACJA ---
st.set_page_config(layout="wide", page_title="Gemini RAG Bot")
st.title("Gemini Chatbot z pamięcią FAISS 🧠")

api_key = st.secrets["API_KEY"]
base_url = st.secrets["BASE_URL"]
selected_model = "gemini-2.5-flash"

# Inicjalizacja modelu embeddingowego (cache, żeby nie ładować go co odświeżenie)
@st.cache_resource
def load_embed_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embed_model = load_embed_model()

# --- KLASA FAISS ---
class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query, k=3):
        query_vec = np.array([embed_model.embed_query(query)]).astype("float32")
        D, I = self.index.search(query_vec, k)
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

# --- FUNKCJE POMOCNICZE ---
def process_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    else:
        text = uploaded_file.getvalue().decode("utf-8")
    
    # Dzielenie tekstu na mniejsze kawałki (chunking)
    # Prosty podział co 1000 znaków dla przykładu
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    documents = [{"text": chunk, "filename": uploaded_file.name} for chunk in chunks]
    return documents

def create_index(documents):
    texts = [doc['text'] for doc in documents]
    embeddings_matrix = np.array([embed_model.embed_query(t) for t in texts]).astype("float32")
    
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_matrix)
    return FAISSIndex(index, documents)

# --- UI I LOGIKA ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Wgraj plik, a przeszukam go za pomocą FAISS!"}]

with st.sidebar:
    st.header("Baza wiedzy")
    uploaded_file = st.file_uploader("Dodaj dokument", type=['txt', 'pdf', 'md'])
    
    if uploaded_file:
        if "vector_index" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
            with st.spinner("Indeksowanie dokumentu..."):
                docs = process_file(uploaded_file)
                st.session_state.vector_index = create_index(docs)
                st.session_state.last_file = uploaded_file.name
            st.success("Zindeksowano pomyślnie!")

# Wyświetlanie czatu
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # WYSZUKIWANIE KONTEKSTU
    context = ""
    if "vector_index" in st.session_state:
        relevant_docs = st.session_state.vector_index.similarity_search(prompt, k=3)
        context = "\n".join([d['text'] for d in relevant_docs])

    # PRZYGOTOWANIE PROMPTU
    full_prompt = prompt
    if context:
        full_prompt = f"Kontekst z dokumentu:\n{context}\n\nPytanie: {prompt}"

    # Komunikacja z Gemini
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Budujemy historię dla API (z kontekstem w ostatniej wiadomości)
    api_msgs = st.session_state.messages[:-1] + [{"role": "user", "content": full_prompt}]

    try:
        response = client.chat.completions.create(model=selected_model, messages=api_msgs)
        answer = response.choices[0].message.content
        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.error(f"Błąd: {e}")
