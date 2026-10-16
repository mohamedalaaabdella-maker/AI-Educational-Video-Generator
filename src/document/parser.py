from pathlib import Path

from src.document.pdf_parser import PDFParser
from src.document.docx_parser import DOCXParser


class DocumentParser:
    """
    Factory for selecting the appropriate document parser
    based on the file extension.
    """
    @staticmethod
    def get_parser(file_path: str | Path):
        """
        Return the appropriate parser for the given document.

        Args:
            file_path: Path to the document.

        Returns:
            PDFParser or DOCXParser instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is unsupported.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return PDFParser()

        if extension == ".docx":
            return DOCXParser()

        raise ValueError(
            f"Unsupported document type: {extension}"
        )