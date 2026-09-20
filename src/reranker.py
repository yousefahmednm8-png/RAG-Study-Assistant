from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        if not documents:
            return []

        pairs = [
            (query, document["text"])
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        results = []

        for document, score in ranked[:top_k]:

            document = document.copy()

            relevance_score = float(score)

            document["rerank_score"] = relevance_score

            # Keep compatibility with build_sources()
            document["score"] = relevance_score

            results.append(document)

        return results