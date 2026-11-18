from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
    ):
        texts = [
            f"passage: {text}"
            for text in texts
        ]

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def encode_query(self, query: str):
        return self.model.encode(
            f"query: {query}",
            batch_size=32,
            normalize_embeddings=True,
        )

    def dimension(self) -> int:
        return self.model.get_embedding_dimension()