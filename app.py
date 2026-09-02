"""
Streamlit UI - Multi-PDF RAG Chatbot
"""

import os
import tempfile

import streamlit as st

from rag_utils import (
    ask,
    build_bm25_index,
    build_rag_chain,
    build_vectorstore,
    format_sources,
    get_embeddings,
    get_llm,
    get_reranker,
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


# Loading the BGE embedding model, the cross-encoder reranker, and creating
# the LLM client are the expensive/slow steps. Cache them as resources so
# every rerun (every click, every chat message) doesn't reload them from
# disk/HF cache.
@st.cache_resource(show_spinner=False)
def get_cached_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner=False)
def get_cached_llm():
    return get_llm()


@st.cache_resource(show_spinner=False)
def get_cached_reranker():
    return get_reranker()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None

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

    relevance_strictness = st.slider(
        "Relevance strictness",
        min_value=-12.0,
        max_value=4.0,
        value=-4.0,
        step=0.5,
        help=(
            "Raw cross-encoder reranker score below which a retrieved "
            "chunk is dropped entirely, per chunk (see rag_utils.rerank). "
            "Raise it if you're seeing chunks from an unrelated PDF in "
            "Sources / the answer. Lower it if you're getting "
            "\"I don't know\" for questions the document(s) actually "
            "answer. There's no universal correct value -- it depends on "
            "the reranker model and your documents, so tune it live "
            "against your own questions instead of guessing in code."
        ),
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

                    # BM25 (keyword/sparse) index, built over the same
                    # chunks/chunk_ids as the vector store. Combined with
                    # vectordb at question time for hybrid search — see
                    # rag_utils.hybrid_pool / ask().
                    bm25_index = build_bm25_index(chunks)

                    st.session_state.vectordb = vectordb
                    st.session_state.bm25_index = bm25_index

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
    "BGE Embeddings + BM25 Hybrid Search + Cross-Encoder Rerank + ChromaDB + Gemini"
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
                with st.spinner("Searching and thinking..."):

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

                    # Broad/"list everything" questions keep more chunks
                    # after reranking (they may need several chunks from
                    # the SAME document to be complete) and, for small
                    # document sets, skip top-k retrieval entirely in favor
                    # of reranking every indexed chunk -- see
                    # rag_utils.ask()'s FULL_POOL_CHUNK_LIMIT -- so a
                    # relevant bullet point can't be missed just because it
                    # didn't score in retrieval's initial cut. Specific
                    # factual questions use a tighter, top-k-based pool.
                    # Either way, every chunk is filtered individually
                    # against `relevance_strictness` (the sidebar slider)
                    # before top_k is applied in rag_utils.rerank() -- so a
                    # weak/unrelated chunk from a second PDF loaded in the
                    # same session can't ride along just to fill up top_k
                    # behind one strong match, while a genuinely relevant
                    # chunk that simply isn't the #1 match still gets kept.
                    final_k = 10 if is_broad else 4
                    pool_size = max(final_k * 4, 15)

                    llm = get_cached_llm()
                    reranker = get_cached_reranker()

                    chain = build_rag_chain(llm)

                    answer, docs = ask(
                        st.session_state.vectordb,
                        st.session_state.bm25_index,
                        chain,
                        question,
                        reranker=reranker,
                        sources=selected_sources or None,
                        pool_size=pool_size,
                        final_k=final_k,
                        score_threshold=relevance_strictness,
                        is_broad=is_broad,
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
