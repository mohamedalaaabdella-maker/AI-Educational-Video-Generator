import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


# ==================================================
# Tokenization
# ==================================================

def tokenize(text: str) -> list[str]:
    """
    Convert text into normalized tokens for BM25.
    """

    tokens = re.findall(
        r"\b[\w.-]+\b",
        text.lower(),
    )

    return tokens


# ==================================================
# Create BM25 Index
# ==================================================

def create_bm25_index(
    chunks: list[Document],
) -> BM25Okapi:
    """
    Create a BM25 index from document chunks.
    """

    tokenized_chunks = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25


# ==================================================
# BM25 Retrieval
# ==================================================

def bm25_search(
    bm25: BM25Okapi,
    chunks: list[Document],
    query: str,
    k: int = 5,
) -> list[Document]:
    """
    Retrieve the top-k most relevant chunks using BM25.
    """

    # Tokenize the query
    tokenized_query = tokenize(query)

    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Get indices of the highest-scoring chunks
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    # Keep only top-k results
    top_indices = ranked_indices[:k]

    # Return original Document objects
    results = [
        chunks[index]
        for index in top_indices
    ]

    return results



