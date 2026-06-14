"""
streamlit_app.py – Bielik Legal Chatbot
========================================
Aplikacja RAG do analizy polskich aktów prawnych.
Używa inteligentnego chunkingu (legal_chunker.py) zamiast podziału co N znaków.
"""

import io
from threading import Thread

import faiss
import numpy as np
import streamlit as st
import torch
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# ── Import nowego chunkera ────────────────────────────────────────────────────
from legal_chunker import chunk_legal_document, LegalChunk

# ─────────────────────────────────────────────────────────────────────────────
# Konfiguracja
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "speakleash/Bielik-PL-11B-v3.0-Instruct"
EMBED_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

st.set_page_config(
    layout="wide",
    page_title="Asystent aktów prawnych – Bielik",
    page_icon="⚖️",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS – minimalistyczny, czytelny wygląd
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { max-width: 900px; margin: auto; }
    .source-badge {
        display: inline-block;
        background: #f0f4ff;
        border: 1px solid #c7d2fe;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        color: #3730a3;
        margin: 2px 2px 6px 0;
    }
    .chunk-expander summary { font-size: 0.85rem; color: #6b7280; }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Asystent aktów prawnych")
st.caption("Zadaj pytanie dotyczące wgranego aktu – model odpowie na podstawie jego treści.")

# ─────────────────────────────────────────────────────────────────────────────
# Wczytywanie modeli (cache)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Wczytuję model embeddingów…")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
    )


@st.cache_resource(show_spinner="Wczytuję model językowy Bielik (to może chwilę potrwać)…")
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index oparty na LegalChunk
# ─────────────────────────────────────────────────────────────────────────────

class LegalFAISSIndex:
    """FAISS index przechowujący LegalChunk-i z pełnymi metadanymi."""

    def __init__(self, chunks: list[LegalChunk], embeddings: HuggingFaceEmbeddings):
        self.chunks = chunks
        texts = [c.text for c in chunks]

        with st.spinner(f"Indeksuję {len(chunks)} fragment-ów aktu…"):
            matrix = np.array(
                [embeddings.embed_query(t) for t in texts], dtype="float32"
            )

        self.index = faiss.IndexFlatL2(matrix.shape[1])
        self.index.add(matrix)
        self.embeddings = embeddings

    def search(self, query: str, k: int = 4) -> list[LegalChunk]:
        q_vec = np.array(
            [self.embeddings.embed_query(query)], dtype="float32"
        )
        _, indices = self.index.search(q_vec, k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]


# ─────────────────────────────────────────────────────────────────────────────
# Generowanie odpowiedzi (streaming)
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(messages: list[dict], max_new_tokens: int = 1024):
    tokenizer, model = load_model()
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    Thread(
        target=model.generate,
        kwargs=dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        ),
    ).start()

    for token in streamer:
        yield token


# ─────────────────────────────────────────────────────────────────────────────
# Ekstrakcja tekstu z PDF
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        t = page.extract_text() or ""
        pages.append(t)
    return "\n".join(pages)


# ─────────────────────────────────────────────────────────────────────────────
# Stan sesji
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Dzień dobry! Wgraj akt prawny i zadaj pytanie – postaram się odpowiedzieć na podstawie jego treści."}
    ]
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index: LegalFAISSIndex | None = None
    st.session_state.indexed_filename: str | None = None
    st.session_state.chunks: list[LegalChunk] = []

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – upload i statystyki
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📄 Dokument")
    uploaded_file = st.file_uploader(
        "Wgraj akt prawny (PDF)",
        type=["pdf"],
        help="Obsługiwane formaty: PDF",
    )

    if uploaded_file and uploaded_file.name != st.session_state.indexed_filename:
        file_bytes = uploaded_file.read()
        with st.spinner("Wyodrębniam tekst z PDF…"):
            raw_text = extract_text_from_pdf(file_bytes)

        # ── Inteligentny chunking prawny ─────────────────────────────────────
        chunks = chunk_legal_document(
            text=raw_text,
            source_file=uploaded_file.name,
            max_chars=1800,    # ~400 tokenów – dobry rozmiar dla embeddingu
            overlap_units=1,   # ostatni artykuł powtarza się w kolejnym chunku
        )
        st.session_state.chunks = chunks

        # ── Budowa indeksu ───────────────────────────────────────────────────
        embeddings = get_embeddings()
        st.session_state.faiss_index = LegalFAISSIndex(chunks, embeddings)
        st.session_state.indexed_filename = uploaded_file.name

        st.success(f"Zaindeksowano **{len(chunks)}** fragmentów aktu.")

    # Statystyki dokumentu
    if st.session_state.chunks:
        st.divider()
        st.subheader("📊 Struktura dokumentu")
        chunks = st.session_state.chunks

        unit_types: dict[str, int] = {}
        for c in chunks:
            for u in c.units:
                unit_types[u.unit_type] = unit_types.get(u.unit_type, 0) + 1

        label_map = {
            "article": "Artykuły",
            "paragraph": "Paragrafy (§)",
            "section": "Sekcje",
            "preamble": "Wstęp",
        }
        for utype, count in unit_types.items():
            st.metric(label_map.get(utype, utype), count)
        st.metric("Fragmenty do wyszukiwania", len(chunks))

        # Podgląd chunkingu
        with st.expander("🔍 Podgląd fragmentów", expanded=False):
            for c in chunks[:10]:
                st.markdown(
                    f'<span class="source-badge">{c.display_reference}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(c.text[:200] + ("…" if len(c.text) > 200 else ""))
                st.divider()

    st.divider()
    st.caption("Model: Bielik-PL-11B-v3.0-Instruct (4-bit)")
    st.caption("Embeddingi: paraphrase-multilingual-MiniLM-L12-v2")

# ─────────────────────────────────────────────────────────────────────────────
# Główny obszar czatu
# ─────────────────────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Pokaż źródła jeśli są
        if msg.get("sources"):
            with st.expander("📎 Źródła z dokumentu", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<span class="source-badge">📌 {src["ref"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(src["preview"])

# ─────────────────────────────────────────────────────────────────────────────
# Obsługa pytania
# ─────────────────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Zadaj pytanie dotyczące aktu…"):

    # Walidacja: czy dokument jest zaindeksowany?
    if st.session_state.faiss_index is None:
        st.warning("⚠️ Najpierw wgraj akt prawny w panelu po lewej stronie.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # ── Wyszukiwanie semantyczne ──────────────────────────────────────────────
    relevant_chunks: list[LegalChunk] = st.session_state.faiss_index.search(prompt, k=4)

    # Buduj kontekst z metadanymi odwołań
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(
            f"[{chunk.display_reference}]\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # ── Budowa wiadomości dla modelu ─────────────────────────────────────────
    messages_for_api = [
        {
            "role": "system",
            "content": (
                "Jesteś asystentem prawnym. Odpowiadaj wyłącznie na podstawie "
                "poniższego kontekstu z aktu prawnego. Przy każdej informacji "
                "podawaj odwołanie do artykułu/paragrafu w nawiasie kwadratowym "
                "(np. [Art. 10]). Jeśli informacja nie wynika z kontekstu, "
                "powiedz to wprost.\n\n"
                f"=== KONTEKST Z DOKUMENTU ===\n{context}"
            ),
        }
    ]
    # Dodaj historię rozmowy (bez pierwszego powitania asystenta)
    for m in st.session_state.messages:
        if m.get("sources"):
            messages_for_api.append({"role": m["role"], "content": m["content"]})
        elif m["role"] != "assistant" or len(messages_for_api) > 1:
            messages_for_api.append({"role": m["role"], "content": m["content"]})

    # ── Generowanie i streaming odpowiedzi ───────────────────────────────────
    with st.chat_message("assistant"):
        response_text = st.write_stream(generate_response(messages_for_api))

        # Pokaż źródła pod odpowiedzią
        sources = [
            {
                "ref": c.display_reference,
                "preview": c.text[:180] + "…",
            }
            for c in relevant_chunks
        ]
        with st.expander("📎 Fragmenty użyte do odpowiedzi", expanded=False):
            for src in sources:
                st.markdown(
                    f'<span class="source-badge">📌 {src["ref"]}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(src["preview"])

    # Zapisz odpowiedź z metadanymi źródeł
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources,
    })
