"""
Streamlit UI - Multi-PDF RAG Chatbot
"""

import os
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


# Streamlit Community Cloud only exposes secrets through st.secrets, it does
# NOT copy them into os.environ automatically. rag_utils reads the key from
# os.environ, so bridge it here once at startup (works both locally with a
# .env/real env var and on Streamlit Cloud with .streamlit/secrets.toml).
if not os.environ.get("GOOGLE_API_KEY"):
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass


MAX_FILE_SIZE_MB = 10
MAX_FILES = 3


# Loading the BGE embedding model and creating the LLM client are the two
# expensive/slow steps. Cache them as resources so every rerun (every click,
# every chat message) doesn't reload the embedding model from disk/HF cache.
@st.cache_resource(show_spinner=False)
def get_cached_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner=False)
def get_cached_llm():
    return get_llm()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "sources_available" not in st.session_state:
    st.session_state.sources_available = []


with st.sidebar:
    st.header("⚙️ Setup")

    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Gemini API key is not configured.")
    else:
        st.success("Gemini is ready.")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum {MAX_FILES} files, {MAX_FILE_SIZE_MB} MB each.",
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=400,
        max_value=1500,
        value=800,
        step=100,
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=300,
        value=80,
        step=20,
    )

    process_clicked = st.button(
        "🔄 Process PDFs",
        use_container_width=True,
        type="primary",
    )

    st.divider()

    if st.session_state.sources_available:
        st.subheader("📄 Search only these PDFs")

        if len(st.session_state.sources_available) > 1:
            st.caption(
                "⚠️ Multiple unrelated PDFs are indexed together. "
                "Uncheck the ones not relevant to your question — "
                "otherwise the answer/Sources may pull in chunks from "
                "the wrong document."
            )

        selected_sources = []

        for src in st.session_state.sources_available:
            if st.checkbox(
                src,
                value=True,
                key=f"cb_{src}",
            ):
                selected_sources.append(src)

    else:
        selected_sources = []

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


if process_clicked:

    if not os.environ.get("GOOGLE_API_KEY"):
        st.sidebar.error(
            "Gemini API key is not configured."
        )

    elif not uploaded_files:
        st.sidebar.error(
            "Please upload at least one PDF."
        )

    elif len(uploaded_files) > MAX_FILES:
        st.sidebar.error(
            f"Please upload at most {MAX_FILES} PDF files."
        )

    else:
        file_too_large = False

        for f in uploaded_files:
            file_size_mb = len(f.getbuffer()) / (1024 * 1024)

            if file_size_mb > MAX_FILE_SIZE_MB:
                st.sidebar.error(
                    f"{f.name} is too large. "
                    f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
                )
                file_too_large = True
                break

        if not file_too_large:

            try:
                with st.spinner(
                    "Loading, chunking, and indexing your PDF(s)..."
                ):
                    tmp_dir = tempfile.mkdtemp(
                        prefix="rag_uploads_"
                    )

                    saved_paths = []

                    for f in uploaded_files:
                        path = os.path.join(
                            tmp_dir,
                            f.name,
                        )

                        with open(path, "wb") as out:
                            out.write(f.getbuffer())

                        saved_paths.append(path)

                    docs = load_pdfs(saved_paths)

                    chunks = split_documents(
                        docs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    embeddings = get_cached_embeddings()

                    vectordb = build_vectorstore(
                        chunks,
                        embeddings,
                        collection_name="rag_collection_bge",
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
                st.error(
                    f"Processing error: {str(e)}"
                )


st.title("📚 Multi-PDF RAG Chatbot")

st.caption(
    "BGE Embeddings + ChromaDB + Gemini"
)


if not st.session_state.vectordb:

    st.info(
        "👈 Upload your PDF files and click "
        "**Process PDFs** to get started."
    )

else:

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            if msg.get("sources"):
                with st.expander("Sources"):
                    st.markdown(msg["sources"])

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

                    question_lower = question.lower()

                    broad_words = [
                        "all",
                        "list",
                        "every",
                        "each",
                        "summarize",
                        "summary",
                        "overview",
                        "compare",
                        "projects",
                        "skills",
                        "experience",
                        "education",
                        "certifications",
                        "certificates",
                        "courses",
                        "technologies",
                    ]

                    is_broad = any(
                        word in question_lower
                        for word in broad_words
                    )

                    # Broad/"list everything" questions need wide, diverse
                    # coverage (MMR); specific factual questions get precise
                    # top-k similarity search.
                    dynamic_k = 6 if is_broad else 4

                    retriever = get_retriever(
                        st.session_state.vectordb,
                        k=dynamic_k,
                        sources=selected_sources or None,
                        search_type="mmr" if is_broad else "similarity_score_threshold",
                    )

                    llm = get_cached_llm()

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
                    f"Error: {str(e)}"
                )
