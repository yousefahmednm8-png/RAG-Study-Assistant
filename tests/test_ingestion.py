import pytest

from src.ingestion import (
    PDFIngester,
    load_document,
    load_pdf,
    load_text,
)


def test_load_pdf_raises_for_missing_file():

    with pytest.raises(FileNotFoundError):
        load_pdf("does/not/exist.pdf")


def test_load_text(tmp_path):

    path = tmp_path / "notes.txt"

    path.write_text(
        "Heat is a form of energy.",
        encoding="utf-8",
    )

    documents = load_text(str(path))

    assert len(documents) == 1

    assert documents[0]["text"] == (
        "Heat is a form of energy."
    )

    assert documents[0]["metadata"]["source"] == "notes.txt"

    assert documents[0]["metadata"]["document_id"] == "notes"

    assert documents[0]["metadata"]["page"] is None


def test_load_document_unsupported_extension(tmp_path):

    path = tmp_path / "notes.doc"

    path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(str(path))


def test_pdf_ingester_handles_txt(tmp_path):

    path = tmp_path / "notes.txt"

    path.write_text("Thermal energy.", encoding="utf-8")

    documents = PDFIngester().ingest(str(path))

    assert len(documents) == 1

    assert documents[0]["text"] == "Thermal energy."