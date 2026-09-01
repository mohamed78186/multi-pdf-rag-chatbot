"""
Streamlit UI - Multi-PDF RAG Chatbot (Gemini edition)

Run with:
    streamlit run app.py

Features (covers Part 3 + bonus challenges from the lab):
  - Upload multiple PDFs dynamically (never hard-coded)
  - One shared ChromaDB collection across all uploaded PDFs
  - Checkbox list to choose which PDFs to search
  - Adjustable k (number of retrieved chunks)
  - Chat interface with conversation history
  - Shows the source filename + page number under every answer
"""

import os
import shutil
import tempfile

import streamlit as st

from rag_utils import (
    ask,
    build_rag_chain,
    build_vectorstore,
    format_sources,
    get_api_key,
    get_embeddings,
    get_llm,
    get_retriever,
    load_pdfs,
    split_documents,
)

st.set_page_config(page_title="Multi-PDF RAG Chatbot (Gemini)", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "sources_available" not in st.session_state:
    st.session_state.sources_available = []
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = tempfile.mkdtemp(prefix="rag_chroma_")

# ---------------------------------------------------------------------------
# Sidebar - setup
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    provider = st.radio(
        "Model provider",
        options=["Gemini (cloud)", "Ollama (local)"],
        horizontal=True,
    )
    st.session_state.provider = "gemini" if provider.startswith("Gemini") else "ollama"

    if st.session_state.provider == "gemini":
        api_key_input = st.text_input(
            "Google Gemini API key",
            type="password",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            help="Get a key from https://aistudio.google.com/apikey",
        )
        if api_key_input:
            get_api_key(api_key_input)
    else:
        st.text_input(
            "Ollama chat model",
            value=os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2"),
            key="ollama_chat_model",
            help="Must already be pulled: `ollama pull llama3.2`",
        )
        st.text_input(
            "Ollama embedding model",
            value=os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            key="ollama_embed_model",
            help="Must already be pulled: `ollama pull nomic-embed-text`",
        )
        st.caption("Requires `ollama serve` running locally.")
        os.environ["OLLAMA_CHAT_MODEL"] = st.session_state.ollama_chat_model
        os.environ["OLLAMA_EMBEDDING_MODEL"] = st.session_state.ollama_embed_model

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    chunk_size = st.slider("Chunk size", 200, 2000, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 100, 50)

    process_clicked = st.button("🔄 Process PDFs", use_container_width=True, type="primary")

    st.divider()
    k = st.slider("Retriever k (chunks to retrieve)", 1, 10, 4)

    if st.session_state.sources_available:
        st.subheader("📄 Search only these PDFs")
        selected_sources = []
        for src in st.session_state.sources_available:
            if st.checkbox(src, value=True, key=f"cb_{src}"):
                selected_sources.append(src)
    else:
        selected_sources = []

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Process uploaded PDFs
# ---------------------------------------------------------------------------
if process_clicked:
    if st.session_state.provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        st.sidebar.error("Please enter your Gemini API key first.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        with st.spinner("Loading, chunking, and embedding your PDF(s)..."):
            tmp_dir = tempfile.mkdtemp(prefix="rag_uploads_")
            saved_paths = []
            for f in uploaded_files:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                saved_paths.append(path)

            docs = load_pdfs(saved_paths)
            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # fresh collection each time PDFs are (re)processed
            if os.path.exists(st.session_state.persist_dir):
                shutil.rmtree(st.session_state.persist_dir, ignore_errors=True)
            os.makedirs(st.session_state.persist_dir, exist_ok=True)

            embeddings = get_embeddings(provider=st.session_state.provider)
            vectordb = build_vectorstore(
                chunks,
                embeddings,
                persist_directory=st.session_state.persist_dir,
                collection_name=f"rag_collection_{st.session_state.provider}",
            )
            st.session_state.vectordb = vectordb
            st.session_state.sources_available = sorted({d.metadata["source"] for d in docs})
            st.session_state.messages = []

        st.sidebar.success(
            f"Indexed {len(chunks)} chunks from {len(st.session_state.sources_available)} PDF(s)."
        )
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("📚 Multi-PDF RAG Chatbot")
st.caption(f"LangChain + ChromaDB + {'Gemini' if st.session_state.provider == 'gemini' else 'Ollama (local)'}")

if not st.session_state.vectordb:
    st.info("👈 Upload PDFs and click **Process PDFs** in the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    st.markdown(msg["sources"])

    question = st.chat_input("Ask a question about your PDF(s)...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = get_retriever(
                    st.session_state.vectordb,
                    k=k,
                    sources=selected_sources or None,
                )
                llm = get_llm(provider=st.session_state.provider)
                chain = build_rag_chain(llm)
                answer, docs = ask(retriever, chain, question)
                sources_md = format_sources(docs)

            st.markdown(answer)
            with st.expander("Sources"):
                st.markdown(sources_md)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources_md}
        )