from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# ==================================================
# Embedding Configuration
# ==================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

VECTORSTORE_DIR = r'C:\Users\Mohammed Gamal\OneDrive\Desktop\DEPI\AI-Educational-Video-Generator\src\rag\Chroma'

# ==================================================
# Create Embedding Model
# ==================================================

def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create the embedding model used for
    document and query embeddings.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    return embeddings


# ==================================================
# Create / Load Chroma Vector Store
# ==================================================

def create_vector_store(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """
    Create a persistent Chroma vector store
    from document chunks.
    """

    chroma_chunks = []

    for chunk in chunks:

        metadata = chunk.metadata.copy()

        # Convert list values to strings
        for key, value in metadata.items():

            if isinstance(value, list):
                metadata[key] = "\n".join(
                    str(item)
                    for item in value
                )

        chroma_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )

    vector_store = Chroma.from_documents(
        documents=chroma_chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name="enterprise_rag",
    )

    return vector_store


# ==================================================
# Load Existing Vector Store
# ==================================================

def load_vector_store(
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """
    Load an existing persistent Chroma vector store.
    """

    vector_store = Chroma(
        collection_name="enterprise_rag",
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    return vector_store


# ==================================================
# Dense Retrieval
# ==================================================

def dense_search(
    vector_store: Chroma,
    query: str,
    k: int = 5,
) -> list[Document]:
    """
    Retrieve the top-k most semantically similar
    document chunks for a query.
    """

    results = vector_store.similarity_search(
        query,
        k=k,
    )

    return results