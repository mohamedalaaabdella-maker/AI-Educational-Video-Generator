from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

def create_reranker() -> CrossEncoder:
    """
    Create and load the Cross-Encoder reranker model.
    """

    reranker = CrossEncoder(
        RERANKER_MODEL_NAME
    )

    return reranker

def rerank_documents(
    reranker: CrossEncoder,
    query: str,
    documents: list[Document],
    top_k: int = 10,
) -> list[Document]:
    """
    Re-rank retrieved documents using a Cross-Encoder.
    """

    if not documents :
        return []
    
    # Create query-document pairs
    pairs = [
        [query, document.page_content]
        for document in documents
    ]

    # Calculate relevance scores
    scores = reranker.predict(pairs)

    # Combine documents with their scores
    scored_documents = list(
        zip(documents, scores)
    )

    # Sort by relevance score
    scored_documents = sorted(
        scored_documents,
        key=lambda item: item[1],
        reverse=True,
    )

    # Return top-k documents
    results = [
        document
        for document, score in scored_documents[:top_k]
    ]

    return results    




