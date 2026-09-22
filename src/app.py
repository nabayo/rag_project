import streamlit as st
from sentence_transformers import CrossEncoder

from chunking import create_chunks
from gen import ask_rag
from ocr import extract_text
from vector_store import VectorStore


# Charger la base et le reranker UNE SEULE FOIS (grâce au cache Streamlit)
@st.cache_resource
def init_rag_system():
    db = VectorStore(
        chemin_persistance="./chroma_db"
    )  # Chargement de ChromaDB
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )  # Modèle de reranking
    return db, reranker


db, reranker = init_rag_system()

st.title("💬 Nabilath's RAG Chatbot Project")

question = st.chat_input("Ask a question ! :)")

if question:
    with st.spinner("Recherche et analyse en cours..."):
        result = ask_rag(db, reranker, question)
        st.chat_message("user").markdown(question)
        st.write(result["answer"])

        for i, src in enumerate(result["sources"]):
            with st.expander(f"Source {i + 1}"):
                st.write(f"Document : {src[0]['source']}, score = {src[1]}")
