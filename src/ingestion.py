from pypdf import PdfReader
from pathlib import Path


def load_pdf(file_path: str) -> list[dict]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    reader = PdfReader(path)

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text()

        if not text:
            continue

        documents.append(
            {
                "text": text.strip(),
                "metadata": {
                    "source": path.name,
                    "document_id": path.stem,
                    "page": page_number,
                },
            }
        )

    return documents


def load_text(file_path: str) -> list[dict]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    return [
        {
            "text": text,
            "metadata": {
                "source": path.name,
                "document_id": path.stem,
                "page": None,
            },
        }
    ]


def load_document(file_path: str) -> list[dict]:
    """Load a PDF or TXT file into page-level documents."""

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension == ".txt":
        return load_text(file_path)

    if extension == ".pdf":
        return load_pdf(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


class PDFIngester:

    def ingest(self, file_path: str) -> list[dict]:
        return load_document(file_path)