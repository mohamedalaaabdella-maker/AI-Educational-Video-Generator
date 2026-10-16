from pathlib import Path
import fitz


class PDFParser:
    """
    Extract text, tables, and basic metadata from a PDF document.
    """

    def parse(self, file_path: str | Path) -> list[dict]:
        """
        Parse a PDF file page by page, extracting both regular text 
        and tables (formatted as Markdown).

        Args:
            file_path: Path to the PDF file.

        Returns:
            A list of dictionaries, where each dictionary represents
            one PDF page with its combined text and tables.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, got: {file_path.suffix}"
            )

        pages = []

        with fitz.open(file_path) as document:
            for page_number, page in enumerate(document, start=1):
             
                text = page.get_text("text")

             
                tables_content = []
                tables = page.find_tables()
                
                if tables.tables:
                    for table in tables:
                 
                        df = table.extract()
                        if df:
                            markdown_table = self._to_markdown(df)
                            tables_content.append(markdown_table)

            
                full_page_text = text
                if tables_content:
                    full_page_text += "\n\n" + "\n\n".join(tables_content)

                pages.append(
                    {
                        "page": page_number,
                        "text": full_page_text.strip(),
                    }
                )

        return pages

    def _to_markdown(self, table_data: list[list]) -> str:
        """
        Helper method to convert extracted table rows/columns into Markdown format.
        """
        if not table_data:
            return ""

        markdown_lines = []
        for i, row in enumerate(table_data):
          
            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
            markdown_lines.append("| " + " | ".join(cleaned_row) + " |")
            
       
            if i == 0:
                markdown_lines.append("| " + " | ".join(["---"] * len(row)) + " |")

        return "\n".join(markdown_lines)