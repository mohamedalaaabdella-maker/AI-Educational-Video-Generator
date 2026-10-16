from pathlib import Path


class DocumentMetadataBuilder:
    """
    Build metadata for document chunks.

    Supports PDF and DOCX documents and preserves
    the section and page information available
    in each chunk.
    """

    def build(
        self,
        file_path: str | Path,
        chunks: list[dict],
        document_id: str | None = None,
    ) -> list[dict]:
        """
        Add metadata to document chunks.

        Args:
            file_path: Original document path.
            chunks: Chunks produced by DocumentChunker.
            document_id: Optional identifier for the document.
                A new identifier is generated if not provided.

        Returns:
            A list of chunks containing text and metadata.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        file_type = file_path.suffix.lower().lstrip(".")

        if file_type not in {"pdf", "docx"}:
            raise ValueError(
                f"Unsupported document type: .{file_type}"
            )

        documents = []

        for index, chunk in enumerate(chunks, start=1):

            text = chunk.get("text", "").strip()

            if not text:
                continue

            metadata = {
                "chunk_id": f"_chunk_{index:04d}",
                "source": file_path.name,
                "file_type": file_type,
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "subsection": chunk.get("subsection"),
            }

            documents.append(
                {
                    "text": text,
                    "metadata": metadata,
                }
            )

        return documents