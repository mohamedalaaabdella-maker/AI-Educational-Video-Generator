
from pathlib import Path
import re


class DocumentCleaner:
    """
    Unified cleaner for parsed PDF and DOCX documents.

    Cleans text and table content while preserving
    document structure and metadata.
    """

    def clean(
        self,
        file_path: str | Path,
        elements: list[dict]
    ) -> list[dict]:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        extension = file_path.suffix.lower()

        if extension not in {".pdf", ".docx"}:
            raise ValueError(
                f"Unsupported document type: {extension}"
            )

        cleaned_elements = []

        for element in elements:
            element = element.copy()

            # PDF: page-level elements
            if extension == ".pdf":

                text = self._clean_text(
                    element.get("text", "")
                )

                if not text:
                    continue

                element["text"] = text

            # DOCX: structured elements
            else:

                element_type = element.get("type")

                if element_type in {
                    "text",
                    "paragraph",
                    "heading"
                }:

                    text = self._clean_text(
                        element.get("text", "")
                    )

                    if not text:
                        continue

                    element["text"] = text

                elif element_type == "table":

                    rows = self._clean_table(
                        element.get("rows", [])
                    )

                    if not rows:
                        continue

                    element["rows"] = rows

            cleaned_elements.append(element)

        return cleaned_elements

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace without changing meaning."""

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def _clean_table(rows: list[list]) -> list[list]:
        """Clean table cells while preserving table structure."""

        cleaned_rows = []

        for row in rows:
            cleaned_row = []

            for cell in row:
                cell = str(cell)
                cell = re.sub(r"\s+", " ", cell)
                cell = cell.strip()

                cleaned_row.append(cell)

            cleaned_rows.append(cleaned_row)

        return cleaned_rows