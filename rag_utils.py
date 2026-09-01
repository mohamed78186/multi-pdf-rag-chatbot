"""
rag_utils.py
------------
Multi-PDF RAG using BGE embeddings + ChromaDB + Gemini.

Pipeline:
PDFs
-> Load
-> Chunk
-> BGE Embeddings
-> ChromaDB (in-memory)
-> Adaptive Retriever (MMR)
-> Gemini
-> Answer + Sources
"""

import os
import re
import time
from typing import List, Optional, Sequence

# Streamlit Community Cloud ships an old system sqlite3 (< 3.35) which
# chromadb refuses to run on. Swap in the pysqlite3-binary wheel (see
# requirements.txt) *before* chromadb/langchain_chroma is imported anywhere.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


SYSTEM_PROMPT = """You are a precise document-QA assistant. Answer using ONLY the
context below, which was retrieved from one or more PDF documents. Accuracy matters
more than completeness or fluency.

Rules:
- Answer strictly using the given context. Never use outside knowledge, never guess,
  and never fill gaps with something that "sounds right."
- If the answer is not contained in the context, reply exactly with:
  "I don't know based on the provided document(s)."
- If the context only partially answers the question, answer only the supported part
  and explicitly say which part is missing rather than inferring it.
- Copy numbers, dates, names, and figures EXACTLY as written in the context. Do not
  round, recalculate, or paraphrase them.
- If the question asks for a specific number of items, list EXACTLY that many, and if
  the context contains fewer than that number, say so instead of inventing extra ones.
- If different chunks of context conflict with each other, point out the conflict
  instead of silently picking one.
- Do not infer unsupported connections across different documents or sections.
- When useful, mention which document(s) (and page, if shown in the context) the
  answer came from.
- Answer in the same language the question was asked in.
- Be concise and factual; use short bullet points for lists.

Context:
{context}
"""


def get_api_key(explicit_key: Optional[str] = None) -> str:
    key = explicit_key or os.environ.get("GOOGLE_API_KEY")

    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    os.environ["GOOGLE_API_KEY"] = key
    return key


def get_embeddings(provider: Optional[str] = None):
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm(
    temperature: float = 0.0,
    provider: Optional[str] = None,
):
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=temperature,
        # Keep answers deterministic and give enough headroom so longer
        # "list everything" answers don't get cut off mid-sentence.
        max_output_tokens=2048,
        top_p=0.95,
    )


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

    return splitter.split_documents(docs)


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


def get_retriever(
    vectordb: Chroma,
    k: int = 4,
    sources: Optional[List[str]] = None,
    search_type: str = "similarity",
):
    """
    Defaults to plain similarity search: for factual Q&A, pulling the
    top-k *most relevant* chunks is more accurate than MMR, which
    deliberately trades relevance for diversity and can drop a chunk
    that repeats (but confirms) the right answer.

    Pass search_type="mmr" if you specifically want broader, less
    redundant coverage (e.g. for very open-ended "summarize everything"
    style questions).
    """
    if search_type == "mmr":
        search_kwargs = {
            "k": k,
            "fetch_k": max(k * 4, 20),
            "lambda_mult": 0.5,
        }
    else:
        search_kwargs = {"k": k}

    if sources:
        search_kwargs["filter"] = {
            "source": {
                "$in": sources
            }
        }

    return vectordb.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def _format_docs(docs: List[Document]) -> str:
    parts = []

    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")

        parts.append(
            f"[Source: {src} | Page: {page}]\n"
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
