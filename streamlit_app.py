import streamlit as st
from openai import OpenAI
import os

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
# api_key, base_url = os.getenv("API_KEY")
selected_model = "gemini-2.5-flash"

with st.sidebar:
    st.header("Dodatki")
    uploaded_file = st.file_uploader("Wgraj plik tekstowy", type=['txt', 'py', 'md', 'json'])
    
    if uploaded_file is not None:
        # Odczyt treści pliku
        stringio = uploaded_file.getvalue().decode("utf-8")
        st.success("Plik wgrany pomyślnie!")
        # Opcjonalnie: podgląd pliku w sidebarze
        with st.expander("Podgląd pliku"):
            st.text(stringio)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Cześć! W czym mogę Ci dzisiaj pomóc?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Brak klucza API.")
        st.stop()

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Jeśli plik jest wgrany, doklejamy jego treść do zapytania użytkownika
    full_prompt = prompt
    if uploaded_file is not None:
        full_prompt = f"Kontekst z pliku '{uploaded_file.name}':\n\n{stringio}\n\nPytanie użytkownika: {prompt}"

    # Dodajemy do historii tylko czyste pytanie użytkownika (żeby nie zaśmiecać widoku)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Do API wysyłamy jednak pełny komunikat (z kontekstem pliku)
    # Tworzymy tymczasową listę wiadomości dla API
    api_messages = st.session_state.messages[:-1] + [{"role": "user", "content": full_prompt}]

    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=api_messages
        )

        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    except Exception as e:
        st.error(f"Błąd: {e}")
