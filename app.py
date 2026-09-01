"""
Streamlit UI - Multi-PDF RAG Chatbot (Gemini edition)

Run with:
    streamlit run app.py

Features:
  - Upload multiple PDFs dynamically
  - One shared ChromaDB collection across all uploaded PDFs
  - Checkbox list to choose which PDFs to search
  - Adjustable k
  - Chat interface with conversation history
  - Shows source filename + page number
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
    get_embeddings,
    get_llm,
    get_retriever,
    load_pdfs,
    split_documents,
)

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "sources_available" not in st.session_state:
    st.session_state.sources_available = []

if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = tempfile.mkdtemp(prefix="rag_chroma_")

# Gemini only
st.session_state.provider = "gemini"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Setup")

    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Gemini API key is not configured.")
    else:
        st.success("Gemini is ready.")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=200,
        max_value=2000,
        value=800,
        step=100,
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=50,
    )

    process_clicked = st.button(
        "🔄 Process PDFs",
        use_container_width=True,
        type="primary",
    )

    st.divider()

    k = st.slider(
        "Retriever k (chunks to retrieve)",
        min_value=1,
        max_value=10,
        value=4,
    )

    if st.session_state.sources_available:
        st.subheader("📄 Search only these PDFs")

        selected_sources = []

        for src in st.session_state.sources_available:
            if st.checkbox(src, value=True, key=f"cb_{src}"):
                selected_sources.append(src)

    else:
        selected_sources = []

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Process PDFs
# ---------------------------------------------------------------------------

if process_clicked:

    if not os.environ.get("GOOGLE_API_KEY"):
        st.sidebar.error(
            "Gemini API key is not configured in Streamlit Secrets."
        )

    elif not uploaded_files:
        st.sidebar.error(
            "Please upload at least one PDF."
        )

    else:
        try:
            with st.spinner(
                "Loading, chunking, and embedding your PDF(s)..."
            ):
                tmp_dir = tempfile.mkdtemp(prefix="rag_uploads_")

                saved_paths = []

                for f in uploaded_files:
                    path = os.path.join(tmp_dir, f.name)

                    with open(path, "wb") as out:
                        out.write(f.getbuffer())

                    saved_paths.append(path)

                # Load PDFs
                docs = load_pdfs(saved_paths)

                # Split into chunks
                chunks = split_documents(
                    docs,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                # Remove previous Chroma collection
                if os.path.exists(st.session_state.persist_dir):
                    shutil.rmtree(
                        st.session_state.persist_dir,
                        ignore_errors=True,
                    )

                os.makedirs(
                    st.session_state.persist_dir,
                    exist_ok=True,
                )

                # Gemini embeddings
                embeddings = get_embeddings(
                    provider="gemini"
                )

                # Build vector store
                vectordb = build_vectorstore(
                    chunks,
                    embeddings,
                    persist_directory=st.session_state.persist_dir,
                    collection_name="rag_collection_gemini",
                )

                st.session_state.vectordb = vectordb

                st.session_state.sources_available = sorted(
                    {
                        d.metadata["source"]
                        for d in docs
                    }
                )

                st.session_state.messages = []

            st.sidebar.success(
                f"Indexed {len(chunks)} chunks from "
                f"{len(st.session_state.sources_available)} PDF(s)."
            )

            st.rerun()

        except Exception as e:
            st.error(f"Error while processing PDFs: {e}")

# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

st.title("📚 Multi-PDF RAG Chatbot")

st.caption(
    "LangChain + ChromaDB + Google Gemini"
)

if not st.session_state.vectordb:

    st.info(
        "👈 Upload PDFs and click **Process PDFs** "
        "in the sidebar to get started."
    )

else:

    # Show previous chat messages
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            if msg.get("sources"):
                with st.expander("Sources"):
                    st.markdown(msg["sources"])

    # User question
    question = st.chat_input(
        "Ask a question about your PDF(s)..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:
                with st.spinner("Thinking..."):

                    retriever = get_retriever(
                        st.session_state.vectordb,
                        k=k,
                        sources=selected_sources or None,
                    )

                    llm = get_llm(
                        provider="gemini"
                    )

                    chain = build_rag_chain(llm)

                    answer, docs = ask(
                        retriever,
                        chain,
                        question,
                    )

                    sources_md = format_sources(docs)

                st.markdown(answer)

                with st.expander("Sources"):
                    st.markdown(sources_md)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_md,
                    }
                )

            except Exception as e:
                st.error(
                    f"Error while generating answer: {e}"
                )
