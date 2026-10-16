from langchain_core.documents import Document

from src.rag.dense import dense_search
from src.rag.bm25 import bm25_search


# ==================================================
# RRF Configuration
# ==================================================

RRF_K = 60


# ==================================================
# Reciprocal Rank Fusion
# ==================================================

def reciprocal_rank_fusion(
    dense_results: list[Document],
    bm25_results: list[Document],
    k: int = RRF_K,
) -> list[Document]:
    """
    Combine Dense and BM25 rankings using
    Reciprocal Rank Fusion (RRF).
    """

    rrf_scores = {}

    documents = {}

    # ----------------------------------------------
    # Dense Retrieval Results
    # ----------------------------------------------

    for rank, document in enumerate(dense_results, start=1):

        chunk_id = document.metadata["chunk_id"]

        score = 1 / (k + rank)

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0) + score
        )

        documents[chunk_id] = document

    # ----------------------------------------------
    # BM25 Retrieval Results
    # ----------------------------------------------

    for rank, document in enumerate(bm25_results, start=1):

        chunk_id = document.metadata["chunk_id"]

        score = 1 / (k + rank)

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0) + score
        )

        documents[chunk_id] = document

    # ----------------------------------------------
    # Sort by RRF Score
    # ----------------------------------------------

    ranked_chunk_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )

    # ----------------------------------------------
    # Return Documents
    # ----------------------------------------------

    results = [
        documents[chunk_id]
        for chunk_id in ranked_chunk_ids
    ]

    return results


# ==================================================
# Hybrid Search
# ==================================================

def hybrid_search(
    query: str,
    vector_store,
    bm25,
    chunks: list[Document],
    dense_k: int = 20,
    bm25_k: int = 20,
    final_k: int = 10,
) -> list[Document]:
    """
    Retrieve documents using both Dense Retrieval
    and BM25, then combine their rankings using RRF.
    """

    # ----------------------------------------------
    # 1. Dense Retrieval
    # ----------------------------------------------

    dense_results = dense_search(
        vector_store=vector_store,
        query=query,
        k=dense_k,
    )

    # ----------------------------------------------
    # 2. BM25 Retrieval
    # ----------------------------------------------

    bm25_results = bm25_search(
        bm25=bm25,
        chunks=chunks,
        query=query,
        k=bm25_k,
    )

    # ----------------------------------------------
    # 3. Combine Rankings
    # ----------------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
    )

    # ----------------------------------------------
    # 4. Keep Final Top-K
    # ----------------------------------------------

    return hybrid_results[:final_k]