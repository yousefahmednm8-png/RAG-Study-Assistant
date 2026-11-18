import numpy as np

from src.retrieval import Retriever
from src.bm25_retrieval import BM25Retriever


def test_retriever_passes_plain_query_string():

    captured = {}

    class Model:

        def encode_query(self, query):
            captured["query"] = query
            return np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )

    class Store:

        def search(self, query_embedding, top_k=5):
            captured["top_k"] = top_k
            return [
                {
                    "text": "Force is a push or pull.",
                    "metadata": {
                        "chunk_id": "doc_1_chunk_0",
                        "source": "physics.pdf",
                        "page": 1,
                    },
                    "score": 0.9,
                }
            ]

    retriever = Retriever(
        embedding_model=Model(),
        vector_store=Store(),
    )

    results = retriever.retrieve(
        query="What is force?",
        top_k=3,
    )

    assert captured["query"] == "What is force?"

    assert captured["top_k"] == 3

    assert len(results) == 1

    assert results[0]["text"] == (
        "Force is a push or pull."
    )


def test_retriever_does_not_wrap_query_in_list():

    class Model:

        def encode_query(self, query):
            return np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )

    class Store:

        def search(self, query_embedding, top_k=5):
            return []

    retriever = Retriever(
        embedding_model=Model(),
        vector_store=Store(),
    )

    results = retriever.retrieve("hello", top_k=5)

    assert results == []


def test_bm25_build_requires_documents():

    bm25 = BM25Retriever()

    assert bm25.retrieve("anything", top_k=5) == []


def test_bm25_retrieves_relevant_documents():

    bm25 = BM25Retriever()

    bm25.build(
        [
            {
                "text": "centripetal force keeps objects moving in circles",
                "metadata": {"chunk_id": "doc_1_chunk_0"},
            },
            {
                "text": "friction opposes the sliding of surfaces",
                "metadata": {"chunk_id": "doc_1_chunk_1"},
            },
            {
                "text": "viscosity measures the resistance of fluids",
                "metadata": {"chunk_id": "doc_1_chunk_2"},
            },
        ]
    )

    results = bm25.retrieve(
        "centripetal force",
        top_k=5,
    )

    assert len(results) >= 1

    assert results[0]["metadata"]["chunk_id"] == (
        "doc_1_chunk_0"
    )

    assert results[0]["bm25_score"] > 0.0


def test_bm25_returns_empty_for_unrelated_query():

    bm25 = BM25Retriever()

    bm25.build(
        [
            {
                "text": "centripetal force circling motion",
                "metadata": {"chunk_id": "doc_1_chunk_0"},
            },
            {
                "text": "friction opposes sliding surfaces",
                "metadata": {"chunk_id": "doc_1_chunk_1"},
            },
        ]
    )

    results = bm25.retrieve(
        "zzzzqwerty nomatch",
        top_k=5,
    )

    assert results == []