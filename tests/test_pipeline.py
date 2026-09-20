import numpy as np
import pytest

from src.pipeline import RAGPipeline


class FakeEmbeddingModel:

    def __init__(self):
        self.last_query = ""
        self.encode_calls = 0

    def encode_documents(self, texts, batch_size=32):
        return np.array(
            [[1.0, 0.0, 0.0] for _ in texts],
            dtype=np.float32,
        )

    def encode_query(self, query):
        self.last_query = query
        self.encode_calls += 1
        return np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )


class FakeVectorStore:

    def __init__(self, empty=False):
        self.documents = []
        self.empty = empty

    def add(self, embeddings, documents):
        self.documents.extend(documents)

    def clear(self):
        self.documents.clear()

    def search(self, query_embedding, top_k=5):
        if self.empty:
            return []
        return [
            {
                "text": "Supervised learning uses labeled data.",
                "metadata": {
                    "document_id": "ml",
                    "chunk_id": "ml_page_3_chunk_0",
                    "source": "ml.pdf",
                    "page": 3,
                },
                "score": 0.9,
            }
        ][:top_k]


class FakeReranker:

    def __init__(self, model_name="fake"):
        self.model_name = model_name

    def rerank(self, query, documents, top_k=5):
        ranked = sorted(
            documents,
            key=lambda d: float(d.get("score", 0.0)),
            reverse=True,
        )
        for document in ranked:
            document["rerank_score"] = document["score"]
        return ranked[:top_k]


class FakeGenerator:

    def __init__(self):
        self.last_prompt = ""
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        self.last_prompt = prompt
        return "Supervised learning uses labeled data."


def build_pipeline(empty_store=False):
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore(empty=empty_store)
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        ingester=None,
        generator=generator,
        chunker=None,
    )

    pipeline.reranker = FakeReranker()

    return pipeline, embedding_model, generator


@pytest.fixture(autouse=True)
def no_real_reranker(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline.Reranker",
        FakeReranker,
    )


def test_pipeline_query():

    pipeline, embedding_model, generator = build_pipeline()

    documents = [
        {
            "text": "Supervised learning uses labeled data.",
            "metadata": {
                "source": "ml.pdf",
                "document_id": "ml",
                "page": 3,
            },
        }
    ]

    pipeline.index_documents(documents)

    result = pipeline.query(
        "What is supervised learning?",
        top_k=1,
    )

    assert result["answer"] == (
        "Supervised learning uses labeled data."
    )

    assert len(result["retrieved_chunks"]) == 1
    assert len(result["sources"]) == 1

    source = result["sources"][0]

    assert source["source_id"] == "S1"
    assert source["document_id"] == "ml"
    assert source["chunk_id"] == "ml_page_3_chunk_0"
    assert source["source"] == "ml.pdf"
    assert source["page"] == 3

    assert generator.calls == 1


def test_pipeline_embeds_plain_question_string():

    pipeline, embedding_model, _ = build_pipeline()

    documents = [
        {
            "text": "Supervised learning uses labeled data.",
            "metadata": {"source": "ml.pdf", "page": 3},
        }
    ]

    pipeline.index_documents(documents)

    question = "What is supervised learning?"

    pipeline.query(
        question,
        top_k=1,
    )

    assert embedding_model.last_query == question


def test_pipeline_prompt_contains_context_and_sources():

    pipeline, _, generator = build_pipeline()

    chunks = [
        {
            "text": "Supervised learning uses labeled data.",
            "metadata": {
                "document_id": "ml",
                "chunk_id": "ml_page_3_chunk_0",
                "source": "ml.pdf",
                "page": 3,
            },
            "score": 0.9,
        }
    ]

    prompt = pipeline.build_prompt(
        "What is supervised learning?",
        chunks,
    )

    assert "What is supervised learning?" in prompt
    assert "ml.pdf" in prompt
    assert "Page: 3" in prompt
    assert "Supervised learning uses labeled data." in prompt


def test_pipeline_empty_retrieval_skips_generation():

    pipeline, _, generator = build_pipeline(empty_store=True)

    result = pipeline.query(
        "What is supervised learning?",
        top_k=1,
    )

    assert "could not find relevant information" in result["answer"]

    assert result["sources"] == []
    assert result["retrieved_chunks"] == []

    assert generator.calls == 0


def test_index_documents_replaces_not_duplicates():

    pipeline, _, _ = build_pipeline()

    documents = [
        {
            "text": "Supervised learning uses labeled data.",
            "metadata": {"source": "ml.pdf", "page": 3},
        }
    ]

    pipeline.index_documents(documents)
    pipeline.index_documents(documents)

    assert len(pipeline.vector_store.documents) == 1