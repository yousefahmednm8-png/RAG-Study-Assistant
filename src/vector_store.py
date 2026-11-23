import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add(self, embeddings: np.ndarray, documents: list[dict]) -> None:
        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents"
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"got {query_embedding.shape[1]}"
            )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            document = self.documents[index].copy()
            document["score"] = float(score)
            results.append(document)

        return results

    def clear(self) -> None:
        self.index.reset()
        self.documents.clear()


    def save(
    self,
    index_path: str,
    documents_path: str,
) -> None:

        Path(index_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Path(documents_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("Saving index:", index_path)
        print("Saving docs :", documents_path)

        faiss.write_index(
            self.index,
            str(index_path),
        )

        with open(
            documents_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.documents,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(
        self,
        index_path: str,
        documents_path: str,
    ) -> None:

        self.index = faiss.read_index(
            str(index_path)
        )

        with open(
            documents_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.documents = json.load(file)