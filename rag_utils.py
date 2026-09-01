"""
rag_utils.py
------------
Shared building blocks for the Multi-PDF RAG Chatbot.

Pipeline:
    PDF(s)
    -> PyPDFLoader
    -> RecursiveCharacterTextSplitter
    -> Local HuggingFace Embeddings
    -> ChromaDB
    -> Retriever
    -> Prompt + Context
    -> Gemini LLM
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

CHAT_MODEL = "gemini-3.5-flash-lite"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
context provided below, which was retrieved from one or more PDF documents.

Rules:
- Answer strictly using the given context. Do not use outside knowledge.
- If the answer is not contained in the context, reply exactly with:
  "I don't know based on the provided document(s)."
- If the question asks for a specific number of items (e.g. "three causes"),
  list EXACTLY that many — the number stated in the source text, not more.
- Never pad the list with related-but-distinct items from elsewhere in the context.
- If the context only partially answers the question, answer only the part
  that is supported and say what's missing.
- Do not fill gaps with outside knowledge.
- If the question requires connecting facts across multiple documents and the
  context does not explicitly state that connection, do not infer or synthesize it.
- When useful, mention which document(s) the answer came from.
- Be concise and factual.

Context:
{context}
"""


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def get_api_key(explicit_key: Optional[str] = None) -> str:
    """
    Resolve the Gemini API key from an explicit argument or environment variable.
    """

    key = explicit_key or os.environ.get("GOOGLE_API_KEY")

    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured."
        )

    os.environ["GOOGLE_API_KEY"] = key
    return key


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings(provider: Optional[str] = None):
    """
    Local free embeddings.

    These embeddings run locally on the Streamlit server and do not consume
    Gemini embedding quota.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_llm(
    temperature: float = 0.0,
    provider: Optional[str] = None
):
    """
    Gemini chat model used only for final answer generation.
    """

    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Loading PDFs
# ---------------------------------------------------------------------------

def load_pdfs(
    pdf_paths: Sequence[str]
) -> List[Document]:
    """
    Load one or more PDF files with PyPDFLoader.

    Metadata keeps:
    - source
    - page
    """

    all_docs: List[Document] = []

    for path in pdf_paths:

        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = os.path.basename(
                d.metadata.get("source", path)
            )

        all_docs.extend(docs)

    return all_docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_documents(
    docs: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 80,
) -> List[Document]:
    """
    Split documents into chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def build_vectorstore(
    chunks: List[Document],
    embeddings,
    persist_directory: str,
    collection_name: str = "rag_collection",
) -> Chroma:
    """
    Create a ChromaDB vector store using local HuggingFace embeddings.

    No Gemini embedding API calls are made here.
    """

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    vectordb.add_documents(chunks)

    return vectordb


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def get_retriever(
    vectordb: Chroma,
    k: int = 3,
    sources: Optional[List[str]] = None,
    score_threshold: Optional[float] = None,
):
    """
    Build a retriever.

    Optional:
    - restrict search to selected source PDFs
    - use a similarity threshold
    """

    search_kwargs = {
        "k": k
    }

    if sources:
        search_kwargs["filter"] = {
            "source": {
                "$in": sources
            }
        }

    if score_threshold is not None:

        search_kwargs["score_threshold"] = score_threshold

        return vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs,
        )

    return vectordb.as_retriever(
        search_kwargs=search_kwargs
    )


# ---------------------------------------------------------------------------
# Formatting retrieved documents
# ---------------------------------------------------------------------------

def _format_docs(
    docs: List[Document]
) -> str:
    """
    Format retrieved documents before sending them to Gemini.
    """

    parts = []

    for d in docs:

        src = d.metadata.get(
            "source",
            "unknown"
        )

        page = d.metadata.get(
            "page",
            "?"
        )

        parts.append(
            f"[Source: {src} | Page: {page}]\n"
            f"{d.page_content}"
        )

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

def build_rag_chain(
    llm: ChatGoogleGenerativeAI
):
    """
    Create the prompt -> Gemini -> text output chain.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                "Question: {question}",
            ),
        ]
    )

    return (
        prompt
        | llm
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# Gemini rate limit handling
# ---------------------------------------------------------------------------

class DailyQuotaExceeded(RuntimeError):
    """
    Raised when Gemini daily quota is exhausted.
    """


def _extract_retry_delay(
    error_text: str,
    fallback: float,
) -> float:
    """
    Extract Gemini's recommended retry delay.
    """

    match = re.search(
        r"retry in ([\d.]+)s",
        error_text,
        re.IGNORECASE,
    )

    if match:
        try:
            return float(
                match.group(1)
            ) + 2
        except ValueError:
            pass

    return fallback


def _is_rate_limit_error(
    e: Exception
) -> bool:
    """
    Detect Gemini 429 / quota errors.
    """

    text = str(e)

    return (
        "RESOURCE_EXHAUSTED" in text
        or "429" in text
        or getattr(
            e,
            "code",
            None,
        ) == 429
    )


def _classify_and_wait(
    error_text: str,
    attempt: int,
    max_retries: int,
    delay: float,
) -> float:
    """
    Handle Gemini API rate-limit errors.
    """

    lower_text = error_text.lower()

    if (
        "perday" in lower_text
        or "per day" in lower_text
        or "requestsperday" in lower_text
    ):

        raise DailyQuotaExceeded(
            "Gemini daily quota has been reached. "
            "Please try again later."
        )

    if attempt >= max_retries - 1:

        raise RuntimeError(
            "Gemini is temporarily rate limited."
        )

    wait_time = _extract_retry_delay(
        error_text,
        delay,
    )

    time.sleep(wait_time)

    return min(
        delay * 2,
        60,
    )


def _invoke_chain_with_retry(
    chain,
    inputs: dict,
    max_retries: int = 4,
) -> str:
    """
    Call Gemini with retry handling for temporary 429 errors.
    """

    delay = 5

    for attempt in range(
        max_retries
    ):

        try:
            return chain.invoke(inputs)

        except Exception as e:

            if not _is_rate_limit_error(e):
                raise

            delay = _classify_and_wait(
                str(e),
                attempt,
                max_retries,
                delay,
            )

    raise RuntimeError(
        "Unable to get a response from Gemini."
    )


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

def ask(
    retriever,
    chain,
    question: str,
):
    """
    Retrieve relevant chunks and ask Gemini using only the retrieved context.
    """

    docs = retriever.invoke(
        question
    )

    context = _format_docs(
        docs
    )

    answer = _invoke_chain_with_retry(
        chain,
        {
            "context": context,
            "question": question,
        },
    )

    return answer, docs


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def format_sources(
    docs: List[Document]
) -> str:
    """
    Format source PDF names and page numbers.
    """

    seen = set()
    lines = []

    for d in docs:

        src = d.metadata.get(
            "source",
            "unknown"
        )

        page = d.metadata.get(
            "page",
            "?"
        )

        key = (
            src,
            page,
        )

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

    if lines:
        return "\n".join(lines)

    return "- no sources retrieved"
