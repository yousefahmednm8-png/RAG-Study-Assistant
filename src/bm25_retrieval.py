from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):
        self.documents = []
        self.bm25 = None

    def build(self, documents: list[dict]) -> None:
        self.documents = documents

        tokenized_documents = [
            document["text"].lower().split()
            for document in documents
        ]

        if tokenized_documents:
            self.bm25 = BM25Okapi(tokenized_documents)
        else:
            self.bm25 = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:

            if scores[index] <= 0:
                continue

            document = self.documents[index].copy()

            document["bm25_score"] = float(scores[index])

            results.append(document)

        return results