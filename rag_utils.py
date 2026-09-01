"""
rag_utils.py
------------
Multi-PDF RAG using BGE embeddings + ChromaDB + Gemini.

Accuracy-focused pipeline:
PDFs
-> Load + clean text
-> Chunk
-> BGE Embeddings (dense) + BM25 (sparse) hybrid retrieval
-> Cross-Encoder reranking (precision step)
-> Gemini (strict, grounded prompt)
-> Answer + Sources
"""

import os
import re
import time
from typing import List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


CHAT_MODEL = "gemini-3.5-flash-lite"

# Small + fast dense embedding model. Precision is recovered later by the
# cross-encoder reranker, so we don't need a heavy embedding model here.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Lightweight cross-encoder reranker. Cheap enough to run on CPU (Streamlit
# Cloud free tier) while giving a large precision boost over raw vector/BM25
# retrieval, because it actually reads (question, chunk) pairs together
# instead of comparing embeddings.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates to pull from each retriever before reranking, and how
# many of the reranked, highest-scoring chunks are finally sent to the LLM.
DEFAULT_FETCH_K = 20
DEFAULT_TOP_N = 6

# Questions that ask for an enumeration ("list all X", "every Y") need many
# more chunks than a narrow factual question, or items get silently dropped
# because the reranker can only keep a handful of the highest-scoring chunks.
# When one of these words appears, fetch_k/top_n are scaled up automatically.
BROAD_QUERY_WORDS = [
    "all",
    "list",
    "every",
    "each",
    "projects",
    "skills",
    "experience",
    "experiences",
    "education",
    "certifications",
    "certificates",
    "courses",
    "technologies",
    "responsibilities",
    "summarize",
    "summary",
    "overview",
]

BROAD_FETCH_K_MULTIPLIER = 2.0
BROAD_TOP_N_MULTIPLIER = 2.5


def is_broad_query(question: str) -> bool:
    """Heuristic: does this question ask for an enumeration/full listing?"""
    q = question.lower()
    return any(word in q for word in BROAD_QUERY_WORDS)


def resolve_retrieval_params(
    question: str,
    fetch_k: int = DEFAULT_FETCH_K,
    top_n: int = DEFAULT_TOP_N,
):
    """
    Scale fetch_k/top_n up for enumeration-style questions so that items
    spread across many chunks (e.g. several CV projects) aren't truncated
    by the reranker keeping only a small, narrowly "most relevant" set.
    """
    if is_broad_query(question):
        fetch_k = max(fetch_k, int(fetch_k * BROAD_FETCH_K_MULTIPLIER))
        top_n = max(top_n, int(top_n * BROAD_TOP_N_MULTIPLIER))

    return fetch_k, top_n


SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question
using ONLY the context excerpts below, which were retrieved from the user's PDF
document(s).

Rules (follow strictly):
1. Base your answer strictly on the given context. Never use outside knowledge,
   even if you are confident it is correct.
2. If the context does not contain the answer, reply exactly:
   "I don't know based on the provided document(s)."
3. If the context only partially answers the question, answer only the
   supported part, and say explicitly which part is not covered.
4. Copy numbers, dates, names, and technical terms exactly as written in the
   context. Do not paraphrase, round, or "correct" them.
5. If the question asks for a specific count or a list of items, include
   exactly the items found in the context (no more, no fewer), and note if
   the context might be incomplete.
6. Do not merge or infer connections across different documents unless the
   context explicitly makes that connection.
7. After every factual claim, cite where it came from like this:
   (source: <file name>, p.<page>).
8. Be concise, factual, and avoid speculation, filler, or hedging language
   beyond what rule 2/3 require.

Context:
{context}
"""


def get_api_key(explicit_key: Optional[str] = None) -> str:
    key = explicit_key or os.environ.get("GOOGLE_API_KEY")

    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    os.environ["GOOGLE_API_KEY"] = key
    return key


def get_embeddings(model_name: str = EMBEDDING_MODEL):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_reranker(model_name: str = RERANKER_MODEL):
    """Cross-encoder reranker: scores (question, chunk) pairs directly."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=512)


def get_llm(
    temperature: float = 0.0,
    provider: Optional[str] = None,
):
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=temperature,
    )


def _clean_text(text: str) -> str:
    """Normalize whitespace/line breaks left over from PDF extraction."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdfs(pdf_paths: Sequence[str]) -> List[Document]:
    all_docs: List[Document] = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = os.path.basename(
                d.metadata.get("source", path)
            )
            d.page_content = _clean_text(d.page_content)

        all_docs.extend(docs)

    return all_docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 80,
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Drop near-empty chunks (page numbers, running headers, etc.) that add
    # noise to retrieval without adding information.
    return [c for c in chunks if len(c.page_content.strip()) >= 20]


def build_vectorstore(
    chunks: List[Document],
    embeddings,
    persist_directory=None,
    collection_name: str = "rag_collection",
) -> Chroma:
    """
    In-memory ChromaDB.

    No persist_directory is used, so Streamlit Cloud does not need
    to write to a local SQLite database.
    """

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    vectordb.add_documents(chunks)

    return vectordb


def _filter_chunks_by_source(
    chunks: List[Document],
    sources: Optional[List[str]],
) -> List[Document]:
    if not sources:
        return chunks

    filtered = [c for c in chunks if c.metadata.get("source") in sources]
    return filtered or chunks


def build_hybrid_retriever(
    vectordb: Chroma,
    chunks: List[Document],
    fetch_k: int = DEFAULT_FETCH_K,
    sources: Optional[List[str]] = None,
):
    """
    Hybrid retrieval = dense vector search (MMR) + sparse BM25 keyword
    search, combined. BM25 matters a lot for accuracy: exact names, numbers,
    codes, or rare terms are often matched better by keyword overlap than by
    embedding similarity, especially with a small embedding model.
    """

    filtered_chunks = _filter_chunks_by_source(chunks, sources)

    bm25_retriever = BM25Retriever.from_documents(filtered_chunks)
    bm25_retriever.k = fetch_k

    vector_search_kwargs = {
        "k": fetch_k,
        "fetch_k": max(fetch_k * 3, 30),
        "lambda_mult": 0.7,
    }

    if sources:
        vector_search_kwargs["filter"] = {"source": {"$in": sources}}

    vector_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs=vector_search_kwargs,
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []

    for d in docs:
        key = (
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.page_content[:120],
        )

        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    return unique_docs


def rerank_documents(
    reranker,
    question: str,
    docs: List[Document],
    top_n: int = DEFAULT_TOP_N,
) -> List[Document]:
    """Re-score deduplicated candidates with a cross-encoder and keep top_n."""

    unique_docs = _dedupe_docs(docs)

    if not unique_docs:
        return []

    pairs = [[question, d.page_content] for d in unique_docs]
    scores = reranker.predict(pairs)

    scored = sorted(
        zip(scores, unique_docs),
        key=lambda pair: pair[0],
        reverse=True,
    )

    return [doc for _, doc in scored[:top_n]]


def _format_docs(docs: List[Document]) -> str:
    parts = []

    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        page_display = page + 1 if isinstance(page, int) else page

        parts.append(
            f"[Source: {src} | Page: {page_display}]\n"
            f"{d.page_content}"
        )

    return "\n\n---\n\n".join(parts)


def build_rag_chain(llm: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Question: {question}"),
        ]
    )

    return prompt | llm | StrOutputParser()


def _extract_retry_delay(
    error_text: str,
    fallback: float,
) -> float:

    match = re.search(
        r"retry in ([\d.]+)s",
        error_text,
        re.IGNORECASE,
    )

    if match:
        try:
            return float(match.group(1)) + 2
        except ValueError:
            pass

    return fallback


def _is_rate_limit_error(e: Exception) -> bool:
    text = str(e)

    return (
        "RESOURCE_EXHAUSTED" in text
        or "429" in text
        or getattr(e, "code", None) == 429
    )


def _invoke_chain_with_retry(
    chain,
    inputs: dict,
    max_retries: int = 4,
) -> str:

    delay = 5

    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)

        except Exception as e:
            if not _is_rate_limit_error(e):
                raise

            if attempt >= max_retries - 1:
                raise RuntimeError(
                    "Gemini quota has been reached or the service is temporarily busy."
                )

            wait_time = _extract_retry_delay(
                str(e),
                delay,
            )

            time.sleep(wait_time)
            delay = min(delay * 2, 60)

    raise RuntimeError(
        "Unable to get a response from Gemini."
    )


def ask(
    retriever,
    reranker,
    chain,
    question: str,
    top_n: int = DEFAULT_TOP_N,
):
    raw_docs = retriever.invoke(question)
    docs = rerank_documents(reranker, question, raw_docs, top_n=top_n)

    if not docs:
        return (
            "I don't know based on the provided document(s).",
            [],
        )

    context = _format_docs(docs)

    answer = _invoke_chain_with_retry(
        chain,
        {
            "context": context,
            "question": question,
        },
    )

    return answer, docs


def format_sources(docs: List[Document]) -> str:
    seen = set()
    lines = []

    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")

        key = (src, page)

        if key not in seen:
            seen.add(key)

            page_display = (
                page + 1
                if isinstance(page, int)
                else page
            )

            lines.append(
                f"- {src} (page {page_display})"
            )

    return "\n".join(lines) if lines else "- no sources retrieved"
