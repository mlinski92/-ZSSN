import streamlit as st
from chat_openrouter import ChatOpenRouter
import os
from langchain_core.prompts import ChatPromptTemplate
import shutil
from docloader import load_documents_from_folder
from embedder_rag import create_index, retrieve_docs

st.set_page_config(layout="wide", page_title="Rag chatbot app")
st.title("Rag chatbot app")

UPLOAD_FOLDER = "data/uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

template = """
Jesteś pomocnym asystentem. Odpowiedz na pytanie na podstawie dostarczonego kontekstu. Jeśli odpowiedź nie jest w kontekście, powiedz, że nie wiesz.
Question: {question} 
Context: {context} 
Answer:
"""

selected_model = "gemini-2.5-flash"
model = ChatOpenRouter(model_name=selected_model)  # poprawiona nazwa klasy

def answer_question(question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

if "query" not in st.session_state:
    st.session_state.query = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "clear_files" not in st.session_state:
    st.session_state.clear_files = False
if "retrieve_files" not in st.session_state:
    st.session_state.retrieve_files = False
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

uploaded_files = st.sidebar.file_uploader(
    "Ładuj PDF(y)", type=["pdf"], accept_multiple_files=True, key="file_uploader"
)

if st.sidebar.button("Usuń pliki"):
    shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    st.session_state.clear_files = True
    st.session_state.query = ""
    st.session_state.answer = ""
    st.sidebar.success("Pliki wyczyszczone")

if st.session_state.clear_files:
    uploaded_files = None
    st.session_state.clear_files = False
    st.session_state.retrieve_files = False

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.write("Pliki załadowane")

    # 1. Załadowanie dokumentów z folderu
    documents = load_documents_from_folder(UPLOAD_FOLDER)

    # 2. Zbudowanie indeksu FAISS i zapis w sesji
    st.session_state.faiss_index = create_index(documents)

    st.write("Pliki przeliczone")
    st.session_state.retrieve_files = True

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Jak mogę pomóc?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    if st.session_state.retrieve_files:
        # 3. Wyszukanie najlepiej pasujących chunków do pytania
        related_documents = retrieve_docs(st.session_state.faiss_index, question, k=3)

        # 4. Wywołanie modelu z kontekstem z dokumentów
        answer = answer_question(question, related_documents, model)
    else:
        # Odpowiedź bez kontekstu, gdy nie ma wgranych plików
        answer = model.invoke(question)

    st.session_state.messages.append({"role": "assistant", "content": answer.content})
    st.chat_message("assistant").write(answer.content)
