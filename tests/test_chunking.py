import pytest

from src.chunking import chunk_text, create_chunks


def test_chunk_text_respects_max_size():

    text = " ".join(["word"] * 200)

    chunks = chunk_text(
        text=text,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) > 1

    assert all(
        len(chunk) <= 100
        for chunk in chunks
    )


def test_chunk_text_splits_oversized_paragraph():

    paragraph = "x" * 500

    text = f"{paragraph}\n\nshort paragraph"

    chunks = chunk_text(
        text=text,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) >= 6

    assert "short paragraph" in chunks[-1]


def test_chunk_text_overlap_shares_content():

    text = "a" * 300

    chunks = chunk_text(
        text=text,
        chunk_size=100,
        chunk_overlap=50,
    )

    assert len(chunks) == 6

    assert chunks[1].startswith(chunks[0][-50:])


def test_chunk_text_validates_args():

    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=0)

    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=100, chunk_overlap=-1)

    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=100, chunk_overlap=100)


def test_create_chunks_preserves_metadata_with_page():

    documents = [
        {
            "text": "First paragraph. " + "filler " * 200,
            "metadata": {
                "source": "physics.pdf",
                "document_id": "physics",
                "page": 4,
            },
        }
    ]

    chunks = create_chunks(
        documents=documents,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        metadata = chunk["metadata"]

        assert metadata["source"] == "physics.pdf"

        assert metadata["document_id"] == "physics"

        assert metadata["chunk_index"] == 0 or (
            metadata["chunk_index"] > 0
        )

        assert metadata["chunk_id"].startswith(
            "physics_page_4_chunk_"
        )


def test_create_chunks_chunk_id_without_page():

    documents = [
        {
            "text": "First paragraph. " + "filler " * 200,
            "metadata": {
                "source": "notes.txt",
                "document_id": "notes",
            },
        }
    ]

    chunks = create_chunks(
        documents=documents,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) > 1

    assert all(
        chunk["metadata"]["chunk_id"].startswith(
            "notes_chunk_"
        )
        for chunk in chunks
    )