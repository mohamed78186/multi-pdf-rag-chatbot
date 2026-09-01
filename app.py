"""
Streamlit UI - Multi-PDF RAG Chatbot
"""

import os
import tempfile

import streamlit as st

from rag_utils import (
    DEFAULT_FETCH_K,
    DEFAULT_TOP_N,
    ask,
    build_hybrid_retriever,
    build_rag_chain,
    build_vectorstore,
    format_sources,
    get_embeddings,
    get_llm,
    get_reranker,
    is_broad_query,
    load_pdfs,
    resolve_retrieval_params,
    split_documents,
)


st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)


MAX_FILE_SIZE_MB = 10
MAX_FILES = 3


# Cache heavyweight resources across reruns/questions instead of reloading
# the embedding model, the reranker, and the LLM client every single time.
@st.cache_resource(show_spinner=False)
def cached_get_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner=False)
def cached_get_reranker():
    return get_reranker()


@st.cache_resource(show_spinner=False)
def cached_get_llm():
    return get_llm()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

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

    with st.expander("🎯 Accuracy settings"):
        top_n = st.slider(
            "Chunks sent to the model (after reranking)",
            min_value=3,
            max_value=10,
            value=DEFAULT_TOP_N,
            help=(
                "More chunks = more coverage but more chance of noise. "
                "Fewer chunks = more focused, precise answers."
            ),
        )

        fetch_k = st.slider(
            "Candidates retrieved before reranking",
            min_value=10,
            max_value=40,
            value=DEFAULT_FETCH_K,
            step=5,
            help=(
                "How many chunks the hybrid (keyword + vector) search pulls "
                "before the cross-encoder reranker narrows them down."
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

                    embeddings = cached_get_embeddings()

                    vectordb = build_vectorstore(
                        chunks,
                        embeddings,
                        collection_name="rag_collection_bge",
                    )

                    st.session_state.vectordb = vectordb
                    st.session_state.chunks = chunks

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

                    effective_fetch_k, effective_top_n = resolve_retrieval_params(
                        question,
                        fetch_k=fetch_k,
                        top_n=top_n,
                    )

                    if is_broad_query(question):
                        st.caption(
                            "🔎 Detected a broad/listing question — "
                            "searching more thoroughly to avoid missing items."
                        )

                    retriever = build_hybrid_retriever(
                        st.session_state.vectordb,
                        st.session_state.chunks,
                        fetch_k=effective_fetch_k,
                        sources=selected_sources or None,
                    )

                    reranker = cached_get_reranker()
                    llm = cached_get_llm()
                    chain = build_rag_chain(llm)

                    answer, docs = ask(
                        retriever,
                        reranker,
                        chain,
                        question,
                        top_n=effective_top_n,
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
