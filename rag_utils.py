"""
rag_utils.py
------------
Shared building blocks for the Multi-PDF RAG Chatbot (Gemini edition).

This module is imported by both:
  - Multi_PDF_RAG_Lab_Gemini.ipynb  (the lab notebook, Parts 1-3)
  - app.py                          (the Streamlit chat UI)

Pipeline:
    PDF(s) -> PyPDFLoader -> RecursiveCharacterTextSplitter -> Gemini Embeddings
            -> ChromaDB -> Retriever -> Prompt + Context -> Gemini LLM -> Answer + Sources
"""

import os
import re
import time
from typing import List, Optional, Sequence

from google.genai.errors import ClientError
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# Two supported providers:
#   "gemini" - cloud, needs GOOGLE_API_KEY, no local install required
#   "ollama" - local, needs `ollama serve` running + the model pulled
#              (e.g. `ollama pull llama3.2` / `ollama pull nomic-embed-text`)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")  # "gemini" | "ollama"

# Gemini chat model used as the LLM (replaces the Cohere LLM from the original lab)
# Using the "Lite" variant on purpose: Google gives Flash-Lite models a much higher
# free-tier daily request quota than standard Flash models, which matters a lot for
# a lab notebook that fires off many LLM calls while you're testing.
CHAT_MODEL = "gemini-3.5-flash-lite"
# Gemini embedding model (replaces Cohere Embeddings from the original lab)
EMBEDDING_MODEL = "gemini-embedding-001"

# Ollama local models
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
context provided below, which was retrieved from one or more PDF documents.

Rules:
- Answer strictly using the given context. Do not use outside knowledge.
- If the answer is not contained in the context, reply exactly with:
  "I don't know based on the provided document(s)."
- If the question asks for a specific number of items (e.g. "three causes"),
  list EXACTLY that many — the number stated in the source text, not more.
  Never pad the list with related-but-distinct items from elsewhere in the context.
- If the context only partially answers the question, answer only the part
  that is supported and say what's missing, rather than filling gaps with
  outside knowledge or loosely related context.
- If the question requires connecting facts across multiple documents and the
  context does not explicitly state that connection, do not infer or
  synthesize it — say so plainly instead of guessing.
- When useful, mention which document(s) the answer came from.
- Be concise and factual.

Context:
{context}
"""


def get_api_key(explicit_key: Optional[str] = None) -> str:
    """Resolve the Gemini API key from an explicit argument, env var, or prompt."""
    key = explicit_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        import getpass
        key = getpass.getpass("Enter your Google Gemini API key: ")
        os.environ["GOOGLE_API_KEY"] = key
    else:
        os.environ["GOOGLE_API_KEY"] = key
    return key


def get_embeddings(provider: Optional[str] = None):
    """Embeddings client. provider = "gemini" (default) or "ollama"."""
    provider = (provider or LLM_PROVIDER).lower()
    if provider == "ollama":
        return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm(temperature: float = 0.0, provider: Optional[str] = None):
    """Chat model client. provider = "gemini" (default) or "ollama"."""
    provider = (provider or LLM_PROVIDER).lower()
    if provider == "ollama":
        return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=temperature, base_url=OLLAMA_BASE_URL)
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=temperature)


# ---------------------------------------------------------------------------
# Part 1 & 2 & 3: Loading and chunking
# ---------------------------------------------------------------------------

def load_pdfs(pdf_paths: Sequence[str]) -> List[Document]:
    """Load one or many PDFs with PyPDFLoader.

    Each resulting Document keeps page_content plus metadata that already
    includes 'source' (file path) and 'page' (0-indexed page number) -
    this is what lets Part 3 show which PDF/page an answer came from.
    """
    all_docs: List[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # normalize the 'source' metadata to just the filename for nicer display
        for d in docs:
            d.metadata["source"] = os.path.basename(d.metadata.get("source", path))
        all_docs.extend(docs)
    return all_docs


def split_documents(
    docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 100
) -> List[Document]:
    """Split documents into chunks with RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class DailyQuotaExceeded(RuntimeError):
    """Raised when the FREE TIER's per-day request cap is hit (retrying won't help)."""


def _extract_retry_delay(error_text: str, fallback: float) -> float:
    """Pull the server-suggested retry delay (e.g. 'retry in 40.29...s') out of the
    error message when present, otherwise fall back to our own backoff value."""
    match = re.search(r"retry in ([\d.]+)s", error_text)
    if match:
        try:
            return float(match.group(1)) + 2  # small safety margin
        except ValueError:
            pass
    return fallback


def _classify_and_wait(error_text: str, attempt: int, max_retries: int, delay: float) -> float:
    """Inspect a rate-limit error message and either wait (per-minute limit) or
    raise immediately with a clear explanation (per-day limit, since retrying
    within the same day cannot succeed)."""
    if "PerDay" in error_text:
        raise DailyQuotaExceeded(
            "You've hit Gemini's FREE TIER DAILY request cap for this model "
            "(resets ~24h after your first request, not just a few seconds). "
            "Retrying now will not help. Options: (1) wait for the daily reset, "
            "(2) enable billing on your Google Cloud project to raise the limit, "
            "or (3) switch CHAT_MODEL / EMBEDDING_MODEL in rag_utils.py to a model "
            "with a higher free daily quota. See https://ai.google.dev/gemini-api/docs/rate-limits "
            f"for current limits.\n\nOriginal error: {error_text}"
        )
    if attempt >= max_retries - 1:
        raise RuntimeError(f"Rate limited repeatedly and out of retries.\n\n{error_text}")
    wait_time = _extract_retry_delay(error_text, delay)
    print(f"  Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.0f}s before retrying...")
    time.sleep(wait_time)
    return min(delay * 2, 90)


def _is_rate_limit_error(e: Exception) -> bool:
    text = str(e)
    return "RESOURCE_EXHAUSTED" in text or "429" in text or getattr(e, "code", None) == 429


def _add_batch_with_retry(vectordb: Chroma, batch: List[Document], max_retries: int = 6) -> None:
    """Add one batch of chunks to Chroma, backing off and retrying on 429 rate limits."""
    delay = 15
    for attempt in range(max_retries):
        try:
            vectordb.add_documents(batch)
            return
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            delay = _classify_and_wait(str(e), attempt, max_retries, delay)


def build_vectorstore(
    chunks: List[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
    persist_directory: str,
    collection_name: str = "rag_collection",
    batch_size: int = 50,
    delay_between_batches: float = 5.0,
) -> Chroma:
    """Embed chunks with Gemini embeddings and store them in ChromaDB.

    Chunks are added in small batches with a short pause between them, and any
    429 (RESOURCE_EXHAUSTED / rate limit) response is retried with exponential
    backoff. This keeps large documents (e.g. a whole book) from blowing past
    the Gemini free-tier embedding quota (100 requests/minute).
    """
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Embedding chunks {i + 1}-{min(i + batch_size, total)} of {total}...")
        _add_batch_with_retry(vectordb, batch)
        if i + batch_size < total:
            time.sleep(delay_between_batches)

    return vectordb


def get_retriever(
    vectordb: Chroma,
    k: int = 4,
    sources: Optional[List[str]] = None,
    score_threshold: Optional[float] = None,
):
    """Build a retriever, optionally restricted to a subset of source filenames
    (this is what powers the 'select which PDFs to search' bonus feature).

    score_threshold (0-1, higher = stricter) drops chunks that are only weakly
    related to the query instead of always returning exactly k chunks — this is
    what keeps a barely-relevant chunk from showing up in the Sources list for
    an answer that didn't actually use it.
    """
    search_kwargs = {"k": k}
    if sources:
        search_kwargs["filter"] = {"source": {"$in": sources}}

    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
        return vectordb.as_retriever(
            search_type="similarity_score_threshold", search_kwargs=search_kwargs
        )
    return vectordb.as_retriever(search_kwargs=search_kwargs)


# ---------------------------------------------------------------------------
# RAG chain (modern LangChain Expression Language style)
# ---------------------------------------------------------------------------

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[Source: {src} | Page: {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(llm: ChatGoogleGenerativeAI):
    """Return a runnable that turns {context, question} into a plain-text answer."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "Question: {question}")]
    )
    return prompt | llm | StrOutputParser()


def _invoke_chain_with_retry(chain, inputs: dict, max_retries: int = 6) -> str:
    """Call the LLM chain, backing off and retrying on 429 rate limits."""
    delay = 15
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            delay = _classify_and_wait(str(e), attempt, max_retries, delay)


def ask(retriever, chain, question: str):
    """Retrieve relevant chunks, run the chain, and return (answer, source_docs)."""
    docs = retriever.invoke(question)
    context = _format_docs(docs)
    answer = _invoke_chain_with_retry(chain, {"context": context, "question": question})
    return answer, docs


def format_sources(docs: List[Document]) -> str:
    """Human-readable 'source PDF + page number' listing for display under an answer."""
    seen = set()
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            # PyPDFLoader pages are 0-indexed -> show 1-indexed to the user
            page_display = page + 1 if isinstance(page, int) else page
            lines.append(f"- {src} (page {page_display})")
    return "\n".join(lines) if lines else "- no sources retrieved"