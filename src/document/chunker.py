from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Structure-aware chunker for PDF and DOCX documents.

    The chunker:
    - Supports PDF page-level parsed data.
    - Supports DOCX structured elements.
    - Preserves section information when available.
    - Preserves page information for PDF documents.
    - Splits large content into smaller chunks for RAG.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def chunk(self, elements: list[dict]) -> list[dict]:
        """
        Convert parsed/cleaned elements into chunks.

        The method automatically detects whether the input
        looks like PDF page-level data or DOCX structured data.
        """

        if not elements:
            return []

        if self._is_pdf(elements):
            return self._chunk_pdf(elements)

        if self._is_docx(elements):
            return self._chunk_docx(elements)

        raise ValueError(
            "Unsupported document structure."
        )

    # =========================================================
    # PDF
    # =========================================================

    def _chunk_pdf(self, elements: list[dict]) -> list[dict]:
        """
        Chunk PDF content while preserving page information.
        """

        chunks = []

        for element in elements:

            page_number = element.get("page")
            text = element.get("text", "").strip()

            if not text:
                continue

            split_texts = self.splitter.split_text(text)

            for text_chunk in split_texts:

                chunks.append(
                    {
                        "text": text_chunk,
                        "page": page_number,
                    }
                )

        return chunks

    # =========================================================
    # DOCX
    # =========================================================

    def _chunk_docx(self, elements: list[dict]) -> list[dict]:
        """
        Chunk DOCX content while preserving heading hierarchy.
        """

        chunks = []

        current_section = None
        current_subsection = None
        current_content = []

        for element in elements:

            element_type = element.get("type")

            # -------------------------------------------------
            # Heading
            # -------------------------------------------------

            if element_type == "heading":

                if current_content:
                    chunks.extend(
                        self._create_chunks(
                            content=current_content,
                            section=current_section,
                            subsection=current_subsection
                        )
                    )

                    current_content = []

                level = element.get("level", 1)
                heading = element.get("text", "").strip()

                if not heading:
                    continue

                if level == 1 or current_section is None:
                    current_section = heading
                    current_subsection = None

                else:
                    current_subsection = heading

            # -------------------------------------------------
            # Paragraph
            # -------------------------------------------------

            elif element_type == "paragraph":

                text = element.get("text", "").strip()

                if text:
                    current_content.append(text)

            # -------------------------------------------------
            # Table
            # -------------------------------------------------

            elif element_type == "table":

                table_text = self._table_to_text(
                    element.get("rows", [])
                )

                if table_text:
                    current_content.append(table_text)

        # -----------------------------------------------------
        # Remaining content
        # -----------------------------------------------------

        if current_content:
            chunks.extend(
                self._create_chunks(
                    content=current_content,
                    section=current_section,
                    subsection=current_subsection
                )
            )

        return chunks

    # =========================================================
    # Common Chunk Creation
    # =========================================================

    def _create_chunks(
        self,
        content: list[str],
        section: str | None,
        subsection: str | None
    ) -> list[dict]:
        """
        Split accumulated content into smaller chunks.
        """

        full_text = "\n\n".join(content)

        split_texts = self.splitter.split_text(
            full_text
        )

        chunks = []

        for text in split_texts:

            chunks.append(
                {
                    "text": text,
                    "section": section,
                    "subsection": subsection
                }
            )

        return chunks

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _is_pdf(elements: list[dict]) -> bool:
        """
        Detect PDF page-level structure.
        """

        first_element = elements[0]

        return (
            "page" in first_element
            and "text" in first_element
            and "type" not in first_element
        )

    @staticmethod
    def _is_docx(elements: list[dict]) -> bool:
        """
        Detect DOCX structured elements.
        """

        first_element = elements[0]

        return "type" in first_element

    @staticmethod
    def _table_to_text(rows: list[list]) -> str:
        """
        Convert table rows into readable text
        while preserving row structure.
        """

        if not rows:
            return ""

        table_lines = []

        for row in rows:

            cells = [
                str(cell).strip()
                for cell in row
            ]

            row_text = " | ".join(cells)

            if row_text:
                table_lines.append(row_text)

        return "\n".join(table_lines)