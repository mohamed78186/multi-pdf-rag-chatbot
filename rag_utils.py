"""
rag_utils.py
------------
Multi-PDF RAG using TF-IDF retrieval + Gemini generation.

Pipeline:
PDFs
-> Load
-> Chunk
-> TF-IDF index
-> Retrieve top matching chunks
-> Gemini
-> Answer + Sources
"""

import os
import re
import time
from typing import List, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

CHAT_MODEL = "gemini-3.5-flash-lite"


SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
context provided below, which was retrieved from one or more PDF documents.

Rules:
- Answer strictly using the given context. Do not use outside knowledge.
- If the answer is not contained in the context, reply exactly with:
  "I don't know based on the provided document(s)."
- If the question asks for a specific number of items, list EXACTLY that many.
- If the context only partially answers the question, answer only the supported part.
- Do not infer unsupported connections across documents.
- When useful, mention which document(s) the answer came from.
- Be concise and factual.

Context:
{context}
"""


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def get_api_key(explicit_key: Optional[str] = None) -> str:
    key = explicit_key or os.environ.get("GOOGLE_API_KEY")

    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    os.environ["GOOGLE_API_KEY"] = key
    return key


# ---------------------------------------------------------------------------
# Gemini LLM
# ---------------------------------------------------------------------------

def get_llm(
    temperature: float = 0.0,
    provider: Optional[str] = None,
):
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def load_pdfs(pdf_paths: Sequence[str]) -> List[Document]:
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# TF-IDF vector store
# ---------------------------------------------------------------------------

class TfidfVectorStore:
    def __init__(self, chunks: List[Document]):
        self.chunks = chunks

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        texts = [doc.page_content for doc in chunks]

        self.matrix = self.vectorizer.fit_transform(texts)

    def search(
        self,
        query: str,
        k: int = 3,
        sources: Optional[List[str]] = None,
    ) -> List[Document]:

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).flatten()

        ranked_indices = scores.argsort()[::-1]

        results = []

        for idx in ranked_indices:
            doc = self.chunks[idx]

            if sources:
                source = doc.metadata.get("source")
                if source not in sources:
                    continue

            if scores[idx] <= 0:
                continue

            results.append(doc)

            if len(results) >= k:
                break

        return results


def build_vectorstore(
    chunks: List[Document],
    embeddings=None,
    persist_directory=None,
    collection_name: str = "rag_collection",
):
    return TfidfVectorStore(chunks)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TfidfRetriever:
    def __init__(
        self,
        vectorstore: TfidfVectorStore,
        k: int = 3,
        sources: Optional[List[str]] = None,
    ):
        self.vectorstore = vectorstore
        self.k = k
        self.sources = sources

    def invoke(self, question: str) -> List[Document]:
        return self.vectorstore.search(
            query=question,
            k=self.k,
            sources=self.sources,
        )


def get_retriever(
    vectordb,
    k: int = 3,
    sources: Optional[List[str]] = None,
    score_threshold: Optional[float] = None,
):
    return TfidfRetriever(
        vectorstore=vectordb,
        k=k,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# Compatibility function
# ---------------------------------------------------------------------------

def get_embeddings(provider: Optional[str] = None):
    """
    Kept only so app.py does not need major changes.
    TF-IDF does not use embeddings.
    """
    return None


# ---------------------------------------------------------------------------
# Format retrieved docs
# ---------------------------------------------------------------------------

def _format_docs(docs: List[Document]) -> str:
    parts = []

    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")

        parts.append(
            f"[Source: {src} | Page: {page}]\n{d.page_content}"
        )

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

def build_rag_chain(llm: ChatGoogleGenerativeAI):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Question: {question}"),
        ]
    )

    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Gemini rate-limit handling
# ---------------------------------------------------------------------------

class DailyQuotaExceeded(RuntimeError):
    pass


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
                    "Gemini is temporarily unavailable or quota has been reached."
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


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

def ask(
    retriever,
    chain,
    question: str,
):

    docs = retriever.invoke(question)

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


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

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
