import numpy as np
import pytest

from src.vector_store import VectorStore


def test_add_documents():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    documents = [
        {
            "text": "machine learning",
            "metadata": {
                "source": "test.pdf",
                "page": 1,
            },
        },
        {
            "text": "deep learning",
            "metadata": {
                "source": "test.pdf",
                "page": 2,
            },
        },
    ]

    store = VectorStore(dimension=3)

    store.add(embeddings, documents)

    assert store.index.ntotal == 2
    assert len(store.documents) == 2


def test_search_returns_results():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    documents = [
        {
            "text": "machine learning",
            "metadata": {"source": "test.pdf"},
        },
        {
            "text": "deep learning",
            "metadata": {"source": "test.pdf"},
        },
    ]

    store = VectorStore(dimension=3)

    store.add(embeddings, documents)

    query = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    results = store.search(query, top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "machine learning"
    assert "score" in results[0]


def test_save_and_load_roundtrip(tmp_path):

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    documents = [
        {
            "text": "machine learning",
            "metadata": {"source": "test.pdf", "page": 1},
        },
        {
            "text": "deep learning",
            "metadata": {"source": "test.pdf", "page": 2},
        },
    ]

    store = VectorStore(dimension=3)

    store.add(embeddings, documents)

    index_path = tmp_path / "index.faiss"
    documents_path = tmp_path / "documents.json"

    store.save(
        str(index_path),
        str(documents_path),
    )

    loaded = VectorStore(dimension=3)

    loaded.load(
        str(index_path),
        str(documents_path),
    )

    assert loaded.index.ntotal == 2

    assert [d["text"] for d in loaded.documents] == [
        "machine learning",
        "deep learning",
    ]

    assert loaded.documents[0]["metadata"]["page"] == 1


def test_clear_resets_store():

    embeddings = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    store = VectorStore(dimension=3)

    store.add(embeddings, [{"text": "x", "metadata": {}}])

    store.clear()

    assert store.index.ntotal == 0
    assert store.documents == []


def test_add_requires_matching_dimension():

    embeddings = np.array(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(embeddings, [{"text": "x", "metadata": {}}])


def test_search_returns_empty_when_unpopulated():

    store = VectorStore(dimension=3)

    query = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    assert store.search(query, top_k=3) == []