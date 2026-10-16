from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.rag.document import DocumentsLoader
from src.rag.dense import (
    create_embedding_model,
    create_vector_store,
    load_vector_store
)
from src.rag.bm25 import create_bm25_index
from src.rag.reranker import (
    create_reranker,
    rerank_documents
)
from src.rag.hybrid import hybrid_search

class RAGPipeline:
    """
    End-to-end hybrid retrieval pipeline.

    Pipeline:
        Document Loading
        -> Embeddings + Chroma
        -> BM25 Index
        -> Hybrid Search (Dense + BM25 + RRF)
        -> Cross-Encoder Reranking
        -> Final Documents
    """


    def __init__(
        self,
        file_path: str | Path,
        dense_k: int = 20,
        bm25_k: int = 20,
        candidate_k: int = 20,
        final_k: int = 5,
        load_existing_vector_store: bool = False,
    ):
        self.file_path = Path(file_path)

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.candidate_k = candidate_k
        self.final_k = final_k

        # 1. Load, clean, chunk, and prepare documents
        self.documents_loader = DocumentsLoader()

        self.chunks: list[Document] = (
            self.documents_loader.load(self.file_path)
        )


        if not self.chunks:
            raise ValueError(
                f"No document chunks were generated from: "
                f"{self.file_path}"
            )

        # 2. Create the embedding model
        self.embeddings: HuggingFaceEmbeddings = (
            create_embedding_model()
        )

        # 3. Create or load the Chroma vector store
        if load_existing_vector_store:
            self.vector_store: Chroma = (
                load_vector_store(self.embeddings)
            )
        else:
            self.vector_store: Chroma = (
                create_vector_store(
                    chunks=self.chunks,
                    embeddings=self.embeddings,
                )
            )

        # 4. Build the BM25 index
        self.bm25: BM25Okapi = create_bm25_index(
            self.chunks
        )

        # 5. Load the Cross-Encoder reranker
        self.reranker: CrossEncoder = create_reranker()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Retrieve candidate documents using hybrid search.

        Dense Retrieval + BM25 + RRF.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query must not be empty."
            )

        candidate_k = (
            top_k if top_k is not None
            else self.candidate_k
        )

        documents = hybrid_search(
            query=query,
            vector_store=self.vector_store,
            bm25=self.bm25,
            chunks=self.chunks,
            dense_k=self.dense_k,
            bm25_k=self.bm25_k,
            final_k=candidate_k,
        )

        return documents

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Retrieve and rerank documents.

        Returns the final top-k relevant documents.
        """

        final_k = (
            top_k if top_k is not None
            else self.final_k
        )

        # 1. Retrieve candidate documents
        candidates = self.retrieve(
            query=query,
            top_k=self.candidate_k,
        )

        # 2. Rerank candidates using Cross-Encoder
        results = rerank_documents(
            reranker=self.reranker,
            query=query,
            documents=candidates,
            top_k=final_k,
        )

        return results

    def get_context(
        self,
        query: str,
        top_k: int | None = None,
    ) -> str:
        """
        Convert retrieved documents into context
        that can be passed to an LLM.
        """

        documents = self.search(
            query=query,
            top_k=top_k,
        )

        context_parts = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            context_parts.append(
                f"[Document {index}]\n"
                f"{document.page_content}"
            )

        return "\n\n".join(context_parts)
   