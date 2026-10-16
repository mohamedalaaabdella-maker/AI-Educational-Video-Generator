from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P


class DOCXParser:
    """
    Extract text and basic structure from a DOCX document.

    The parser preserves:
    - Paragraphs
    - Headings
    - Tables
    - Original document order

    Cleaning, chunking, and advanced metadata processing
    are handled by later components.
    """

    def parse(self, file_path: str | Path) -> list[dict]:
        """
        Parse a DOCX document while preserving the original
        order of paragraphs, headings, and tables.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            A list of document elements.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"DOCX file not found: {file_path}"
            )

        if file_path.suffix.lower() != ".docx":
            raise ValueError(
                f"Expected a DOCX file, got: {file_path.suffix}"
            )

        document = Document(file_path)

        elements = []

        # Iterate through the document body
        # while preserving the original order.
        for child in document.element.body.iterchildren():

            # --------------------------------------------------
            # Paragraph
            # --------------------------------------------------
            if isinstance(child, CT_P):

                paragraph = Paragraph(
                    child,
                    document
                )

                text = paragraph.text.strip()

                if not text:
                    continue

                # Some paragraphs may not have a style.
                style = paragraph.style

                if style is not None:
                    style_name = style.name
                else:
                    style_name = ""

                # Check whether the paragraph is a heading.
                if style_name.startswith("Heading"):

                    level = self._get_heading_level(
                        style_name
                    )

                    elements.append(
                        {
                            "type": "heading",
                            "text": text,
                            "level": level,
                        }
                    )

                else:
                    elements.append(
                        {
                            "type": "paragraph",
                            "text": text,
                        }
                    )

            # --------------------------------------------------
            # Table
            # --------------------------------------------------
            elif isinstance(child, CT_Tbl):

                table = Table(
                    child,
                    document
                )

                rows = []

                for row in table.rows:

                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    rows.append(cells)

                if rows:
                    elements.append(
                        {
                            "type": "table",
                            "rows": rows,
                        }
                    )

        return elements

    @staticmethod
    def _get_heading_level(style_name: str) -> int:
        """
        Extract heading level from a Word style name.

        Examples:
            Heading 1 -> 1
            Heading 2 -> 2
            Heading 3 -> 3
        """

        try:
            return int(style_name.split()[-1])

        except (ValueError, IndexError):
            return 0