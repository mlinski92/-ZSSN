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
    # Dodano 'pdf' do dozwolonych typów
    uploaded_file = st.file_uploader("Wgraj plik", type=['txt', 'py', 'md', 'json', 'pdf'])
    
    extracted_content = ""
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'pdf':
                # Logika dla PDF
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    extracted_content += page.extract_text() + "\n"
                st.success("Tekst z PDF został wyodrębniony!")
            else:
                # Logika dla plików tekstowych
                extracted_content = uploaded_file.getvalue().decode("utf-8")
                st.success("Plik tekstowy wczytany!")
            
            with st.expander("Podgląd treści"):
                st.text(extracted_content[:1000] + "..." if len(extracted_content) > 1000 else extracted_content)
        
        except Exception as e:
            st.error(f"Błąd podczas odczytu pliku: {e}")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Cześć! W czym mogę Ci dzisiaj pomóc?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Brak klucza API.")
        st.stop()

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Budowanie promptu z kontekstem
    full_prompt = prompt
    if extracted_content:
        full_prompt = f"Oto treść wgranego pliku:\n{extracted_content}\n\nUżyj powyższego kontekstu, aby odpowiedzieć na pytanie: {prompt}"

    # Wyświetlamy tylko czyste pytanie użytkownika
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Do API wysyłamy zmodyfikowany komunikat (z historią)
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
        st.error(f"Błąd API: {e}")
