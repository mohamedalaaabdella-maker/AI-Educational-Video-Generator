from pathlib import Path

from langchain_core.documents import Document

from src.document.parser import DocumentParser
from src.document.cleaner import DocumentCleaner
from src.document.chunker import DocumentChunker
from src.document.metadata import DocumentMetadataBuilder

class DocumentsLoader:
    """
    Load a PDF or DOCX file and convert it into
    a list of LangChain Documents.

    Pipeline:
        Parse -> Clean -> Chunk -> Build Metadata
        -> Convert to LangChain Documents
    """

    def __init__(self):
        self.parser = DocumentParser()
        self.cleaner = DocumentCleaner()
        self.chunker = DocumentChunker()
        self.metadata_builder = DocumentMetadataBuilder()


    def load(self, file_path: str | Path) -> list[Document]:
        """
        Process a PDF or DOCX file and return
        a list of LangChain Documents.
        """

        file_path = Path(file_path)

        # 1. Parse the document
        parser_ = self.parser.get_parser(file_path)
        parsed_data = parser_.parse(file_path)

        # 2. Clean extracted content
        cleaned_data = self.cleaner.clean(
            file_path,
            parsed_data,
        )

        # 3. Split content into chunks
        chunks = self.chunker.chunk(cleaned_data)

        # 4. Build metadata for each chunk
        documents = self.metadata_builder.build(
            file_path=file_path,
            chunks=chunks,
        )

        # 5. Convert dictionaries into LangChain Documents
        langchain_documents = [
            Document(
                page_content=item["text"],
                metadata=item["metadata"],
            )
            for item in documents
        ]

        return langchain_documents
