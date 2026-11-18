import numpy as np

from src.embeddings import EmbeddingModel


class StubSentenceTransformer:

    def __init__(self, model_name):
        self.model_name = model_name
        self.last_inputs = None

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            self.last_inputs = [texts]
        else:
            self.last_inputs = list(texts)
        return np.zeros(
            (len(self.last_inputs), 2),
            dtype=np.float32,
        )

    def get_embedding_dimension(self):
        return 2


def test_document_prefix(monkeypatch):

    monkeypatch.setattr(
        "src.embeddings.SentenceTransformer",
        StubSentenceTransformer,
    )

    model = EmbeddingModel("intfloat/multilingual-e5-base")

    model.encode_documents(["Newton's second law"])

    assert model.model.last_inputs[0].startswith(
        "passage: "
    )


def test_query_prefix(monkeypatch):

    monkeypatch.setattr(
        "src.embeddings.SentenceTransformer",
        StubSentenceTransformer,
    )

    model = EmbeddingModel("intfloat/multilingual-e5-base")

    model.encode_query("What is force?")

    assert model.model.last_inputs[0] == (
        "query: What is force?"
    )


def test_dimension(monkeypatch):

    monkeypatch.setattr(
        "src.embeddings.SentenceTransformer",
        StubSentenceTransformer,
    )

    model = EmbeddingModel("intfloat/multilingual-e5-base")

    assert model.dimension() == 2