"""
rag_utils.py
------------
Multi-PDF RAG using BGE embeddings + ChromaDB + BM25 + Cross-Encoder rerank + Gemini.

Pipeline:
PDFs
-> Load
-> Chunk (each chunk gets a stable chunk_id)
-> BGE Embeddings -> ChromaDB (in-memory, dense/semantic search)
-> BM25 index (sparse/keyword search, same chunk_id space)
-> Hybrid fusion (Reciprocal Rank Fusion of the two rankings)
-> Cross-Encoder reranker (precise relevance scoring of the fused pool)
-> Score-threshold cut -> Gemini
-> Answer + Sources

Hybrid search matters because dense embeddings are good at "meaning" but can
miss exact keyword/number/name matches (e.g. an ID, a rare term, an acronym);
BM25 is good at exact keyword matches but blind to paraphrasing. Fusing both
rankings (instead of picking one) recovers the correct chunk in more cases
than either alone. The cross-encoder reranker then re-scores that fused
pool with a much more accurate (but slower) query<->chunk relevance model
than raw embedding cosine similarity, which is what actually improves
precision -- it is the single biggest lever for cutting down wrong/irrelevant
context reaching the LLM.
"""

import math
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

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
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Small, fast, CPU-friendly cross-encoder. It only has to re-score a handful
# of already-retrieved chunks per question, so latency stays low even on
# Streamlit Community Cloud's shared CPUs.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Reciprocal Rank Fusion constant. 60 is the standard value from the
# original RRF paper; it just controls how quickly rank position decays --
# not sensitive enough to need tuning here.
RRF_K = 60


SYSTEM_PROMPT = """You are a precise document-QA assistant. Answer using ONLY the
context below, which was retrieved from one or more PDF documents. Accuracy matters
more than completeness or fluency.

Rules:
- Answer strictly using the given context. Never use outside knowledge, never guess,
  and never fill gaps with something that "sounds right."
- If the context contains ANYTHING relevant to the question, even if partial or
  loosely related (e.g. training, courses, or programs when asked about
  "certifications"), answer with that instead of refusing. Only use the fallback
  sentence below when the context has NOTHING relevant at all.
- If the answer is not contained in the context at all, reply with ONLY this
  sentence and nothing else added before or after it:
  "I don't know based on the provided document(s)."
- Never combine that fallback sentence with any other explanation in the same
  answer -- if you can say anything useful from the context, say that instead of
  the fallback sentence, and note what's missing in plain prose (not the fixed
  sentence).
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


def get_reranker() -> CrossEncoder:
    """
    Cross-encoder used to re-score the hybrid (BM25 + vector) candidate pool.

    Unlike embedding cosine similarity (which scores query and chunk
    independently, then compares vectors), a cross-encoder reads the query
    and the chunk together in one forward pass, so it captures actual
    query<->chunk relevance far more accurately. It's slower per pair, which
    is exactly why it only runs on the small fused candidate pool rather
    than the whole collection.
    """
    return CrossEncoder(RERANKER_MODEL, max_length=512)


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

    chunks = splitter.split_documents(docs)

    # A stable, 0-indexed chunk_id shared by the vector store and the BM25
    # index is what lets hybrid_pool() fuse the two rankings by identity
    # instead of by (fragile) text matching.
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    return chunks


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

    collection_metadata forces cosine distance explicitly (Chroma's
    default is L2). BGE embeddings are normalized specifically for
    cosine similarity (see get_embeddings), and hybrid_pool()'s vector
    leg assumes a 0-1, higher-is-better cosine relevance score -- so
    this must stay set for that to be meaningful.
    """

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    vectordb.add_documents(chunks)

    return vectordb


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    # \w+ with re.UNICODE covers Arabic and other non-Latin scripts too;
    # .lower() is a safe no-op on scripts without case.
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """
    Thin wrapper around rank_bm25's BM25Okapi that keeps the chunk list
    alongside the index, so callers never have to juggle the two
    separately. Index position == chunk_id (see split_documents).
    """

    def __init__(self, chunks: List[Document]):
        self.chunks = chunks
        tokenized_corpus = [_tokenize(c.page_content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def top_k(
        self,
        query: str,
        k: int,
        allowed_sources: Optional[List[str]] = None,
    ) -> List[Tuple[int, float]]:
        """Returns [(chunk_id, bm25_score), ...] sorted best-first."""
        scores = self._bm25.get_scores(_tokenize(query))

        candidate_ids = range(len(self.chunks))
        if allowed_sources:
            allowed = set(allowed_sources)
            candidate_ids = [
                i for i in candidate_ids
                if self.chunks[i].metadata.get("source") in allowed
            ]

        ranked = sorted(
            candidate_ids,
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [(i, float(scores[i])) for i in ranked]


def build_bm25_index(chunks: List[Document]) -> BM25Index:
    return BM25Index(chunks)


def hybrid_pool(
    vectordb: Chroma,
    bm25_index: BM25Index,
    query: str,
    k_vector: int = 20,
    k_bm25: int = 20,
    sources: Optional[List[str]] = None,
) -> List[Document]:
    """
    Runs dense (vector/semantic) search and sparse (BM25/keyword) search
    independently, then fuses the two rankings with Reciprocal Rank Fusion
    (RRF): a document's fused score is the sum of 1/(RRF_K + rank) across
    whichever of the two lists it appears in.

    RRF is used instead of e.g. averaging raw scores because cosine
    similarity and BM25 scores live on completely different, uncomparable
    scales -- RRF only needs each ranking's *order*, not its scores, so
    there's nothing to normalize or tune per collection.

    Returns documents sorted best-first; this is a *candidate pool*, not
    the final answer context -- rerank() below narrows it down.
    """
    filter_dict = {"source": {"$in": sources}} if sources else None

    vector_results = vectordb.similarity_search_with_relevance_scores(
        query,
        k=k_vector,
        **({"filter": filter_dict} if filter_dict else {}),
    )
    vector_ranks: Dict[int, int] = {
        doc.metadata["chunk_id"]: rank
        for rank, (doc, _score) in enumerate(vector_results)
    }
    chunk_lookup: Dict[int, Document] = {
        doc.metadata["chunk_id"]: doc for doc, _score in vector_results
    }

    bm25_results = bm25_index.top_k(query, k=k_bm25, allowed_sources=sources)
    bm25_ranks: Dict[int, int] = {
        chunk_id: rank for rank, (chunk_id, _score) in enumerate(bm25_results)
    }
    for chunk_id, _score in bm25_results:
        chunk_lookup.setdefault(chunk_id, bm25_index.chunks[chunk_id])

    all_ids = set(vector_ranks) | set(bm25_ranks)

    fused_scores = {}
    for chunk_id in all_ids:
        score = 0.0
        if chunk_id in vector_ranks:
            score += 1.0 / (RRF_K + vector_ranks[chunk_id] + 1)
        if chunk_id in bm25_ranks:
            score += 1.0 / (RRF_K + bm25_ranks[chunk_id] + 1)
        fused_scores[chunk_id] = score

    ranked_ids = sorted(all_ids, key=lambda i: fused_scores[i], reverse=True)

    return [chunk_lookup[i] for i in ranked_ids]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ms-marco-MiniLM-L-6-v2's raw logits are NOT a calibrated 0-1 probability --
# a genuinely relevant chunk can score well below 0 in raw terms, so a fixed
# absolute cutoff (even after sigmoid) ends up rejecting correct chunks as
# often as it rejects wrong ones. -8 is a "this pair is essentially
# unrelated" floor observed empirically for this model, used only to catch
# the case where NOTHING in the pool is relevant -- not to filter the
# ranked list itself (top_k already does that).
_RERANK_FLOOR = -8.0


def rerank(
    query: str,
    docs: List[Document],
    reranker: CrossEncoder,
    top_k: int = 4,
    floor: Optional[float] = None,
    margin: Optional[float] = None,
) -> List[Document]:
    """
    Re-scores each (query, chunk) pair with the cross-encoder and returns
    the most relevant chunks, best-first, capped at top_k.

    Two separate cutoffs, for two separate jobs:
    - `floor` (defaults to _RERANK_FLOOR): if even the BEST chunk in the
      pool scores below this, nothing in the pool is relevant at all --
      return empty so ask() can say "I don't know".
    - `margin`: once there IS a relevant top match, drop any chunk whose
      score falls more than `margin` below that top score, before applying
      top_k. This is what keeps a second, unrelated document (indexed in
      the same session) from filling up the remaining top_k slots with
      chunks that are merely "somewhat related" -- e.g. when a broad
      question reranks the full pool across multiple PDFs, the wrong
      document's best chunks are usually still well below the right
      document's, and margin prunes them even though top_k alone would not.
    """
    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    raw_scores = reranker.predict(pairs)

    scored = sorted(
        zip(docs, raw_scores),
        key=lambda pair: pair[1],
        reverse=True,
    )

    floor = _RERANK_FLOOR if floor is None else floor
    if not scored or scored[0][1] < floor:
        return []

    if margin is not None:
        top_score = scored[0][1]
        scored = [
            (doc, score) for doc, score in scored
            if score >= top_score - margin
        ]

    return [doc for doc, _score in scored[:top_k]]


def _display_page(page) -> str:
    """
    pypdf's page metadata is 0-indexed (first page = 0). Convert to the
    1-indexed page/slide number a human would actually see, so the number
    the LLM quotes in its answer matches the number shown in the Sources
    expander (format_sources below) instead of being off by one.
    """
    if isinstance(page, int):
        return str(page + 1)
    return str(page) if page is not None else "?"


def _format_docs(docs: List[Document]) -> str:
    parts = []

    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = _display_page(d.metadata.get("page"))

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


# Above this many chunks, "give me everything" questions still go through
# retrieval (a full scan would be too much context/too slow). At or below
# it -- a typical single CV/short document -- broad questions instead skip
# retrieval entirely and rerank the WHOLE indexed set, so nothing (a
# project bullet, a certification line) can get missed just because it
# didn't score in retrieval's initial top-k.
FULL_POOL_CHUNK_LIMIT = 60


def ask(
    vectordb: Chroma,
    bm25_index: BM25Index,
    chain,
    question: str,
    reranker: Optional[CrossEncoder] = None,
    sources: Optional[List[str]] = None,
    pool_size: int = 20,
    final_k: int = 4,
    floor: Optional[float] = None,
    margin: Optional[float] = None,
    is_broad: bool = False,
):
    """
    Full retrieval pipeline for one question:
    hybrid_pool() (BM25 + vector, fused via RRF) -> rerank() (cross-encoder,
    rank-based cut, see rerank()) -> LLM.

    If `reranker` is omitted, falls back to using the hybrid pool's own
    fused order (still hybrid search, just without the precision boost
    from cross-encoder reranking).
    """
    candidate_chunks = [
        c for c in bm25_index.chunks
        if not sources or c.metadata.get("source") in sources
    ]

    if is_broad and len(candidate_chunks) <= FULL_POOL_CHUNK_LIMIT:
        # Small document(s) + a "list/summarize/compare everything" style
        # question: retrieval's top-k can legitimately miss a relevant
        # chunk (e.g. a second, differently-worded project bullet), so
        # just rerank the entire indexed set for these sources instead.
        pool = candidate_chunks
    else:
        pool = hybrid_pool(
            vectordb,
            bm25_index,
            question,
            k_vector=pool_size,
            k_bm25=pool_size,
            sources=sources,
        )

    if reranker is not None:
        docs = rerank(
            question,
            pool,
            reranker,
            top_k=final_k,
            floor=floor,
            margin=margin,
        )
    else:
        docs = pool[:final_k]

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
        raw_page = d.metadata.get("page")
        page_display = _display_page(raw_page)

        key = (src, page_display)

        if key not in seen:
            seen.add(key)

            lines.append(
                f"- {src} (page {page_display})"
            )

    return "\n".join(lines) if lines else "- no sources retrieved"
