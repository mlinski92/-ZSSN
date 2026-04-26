import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import os

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
# api_key, base_url = os.getenv("API_KEY")
selected_model = "gemini-2.5-flash"

with st.sidebar:
    st.header("Dodatki")
    uploaded_file = st.file_uploader(
        "Wgraj plik tekstowy lub PDF", 
        type=['txt', 'py', 'md', 'json', 'pdf']
    )
    
    file_context = ""
    
    if uploaded_file is not None:
        try:
            # Rozpoznawanie typu pliku
            if uploaded_file.name.lower().endswith('.pdf'):
                # Odczyt PDF
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        file_context += text + "\n"
                st.success(f"PDF '{uploaded_file.name}' wczytany!")
            else:
                # Odczyt plików tekstowych
                file_context = uploaded_file.getvalue().decode("utf-8")
                st.success(f"Plik tekstowy '{uploaded_file.name}' wczytany!")
            
            # Podgląd fragmentu treści
            with st.expander("Podgląd treści pliku"):
                st.text(file_context[:1000] + ("..." if len(file_context) > 1000 else ""))
        
        except Exception as e:
            st.error(f"Błąd podczas odczytu pliku: {e}")

# --- LOGIKA CZATU ---

# Inicjalizacja historii wiadomości
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Cześć! Wgraj plik lub zadaj mi pytanie."}]

# Wyświetlanie historii czatu na ekranie
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Reakcja na wpisanie wiadomości przez użytkownika
if prompt := st.chat_input():
    # Inicjalizacja klienta OpenAI (kompatybilnego z Gemini)
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Wyświetlamy samo pytanie użytkownika w oknie czatu
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Przygotowanie promptu dla modelu (z kontekstem z pliku, jeśli istnieje)
    if file_context:
        full_prompt_with_context = (
            f"Poniżej znajduje się treść wgranego pliku:\n"
            f"--- POCZĄTEK PLIKU ---\n{file_context}\n--- KONIEC PLIKU ---\n\n"
            f"Na podstawie powyższych danych odpowiedz na pytanie: {prompt}"
        )
    else:
        full_prompt_with_context = prompt

    # Tworzymy listę wiadomości do wysłania do API (podmieniamy ostatnią wiadomość na taką z kontekstem)
    api_messages = st.session_state.messages[:-1] + [{"role": "user", "content": full_prompt_with_context}]

    # Wywołanie API
    try:
        with st.spinner("Gemini myśli..."):
            response = client.chat.completions.create(
                model=selected_model,
                messages=api_messages
            )

        msg = response.choices[0].message.content
        
        # Zapisujemy i wyświetlamy odpowiedź asystenta
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
        
    except Exception as e:
        st.error(f"Wystąpił błąd podczas komunikacji z API: {e}")
