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
...
Question: {question} 
Context: {context} 
Answer:
"""

selected_model = ... # nazwa modelu
model = ... # obiekt wrappera modelu

def answer_question(question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    prompt = ChatPromptTemplate.from_template(template) # prompt template'owy
    chain = prompt | model # chain wywołania odpowiedzi
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

uploaded_files = st.sidebar.file_uploader("Ładuj PDF(y)", type=["pdf"], accept_multiple_files=True, key="file_uploader")

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
    documents = ... # załadowanie dokumentów
    st.session_state.faiss_index = ... # dodanie stanu bazy do sesji
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
        related_documents = ... # wyszukanie najbardziej pasujących do query dokumentów
        answer = ... # wywołanie odpowiedzi na pytanie z dodanymi do kontekstu dokumentami
    else:
        answer = ...
    st.session_state.messages.append({"role": "assistant", "content": answer.content})
    st.chat_message("assistant").write(answer.content)