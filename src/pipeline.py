from pathlib import Path

from src.reranker import Reranker
from src.retrieval import Retriever


class RAGPipeline:

    def __init__(
        self,
        embedding_model,
        vector_store,
        ingester,
        generator,
        chunker,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.ingester = ingester
        self.generator = generator
        self.chunker = chunker

        self.retriever = Retriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        self.reranker = Reranker(
            model_name="BAAI/bge-reranker-v2-m3"
        )

    # --------------------------------------------------
    # Retrieval settings
    # --------------------------------------------------

    def choose_top_k(self, question: str) -> int:

        question_lower = question.lower()

        comprehensive_keywords = [
            # English
            "all",
            "list",
            "every",
            "complete",
            "chapters",
            "lessons",
            "topics",
            "table of contents",

            # Arabic
            "كل",
            "جميع",
            "الدروس",
            "الفصول",
            "الموضوعات",
            "المحتويات",
        ]

        if any(
            keyword in question_lower
            for keyword in comprehensive_keywords
        ):
            return 15

        return 5

    # --------------------------------------------------
    # Indexing
    # --------------------------------------------------

    def index_documents(self, documents):
        print("Number of documents:", len(documents))

        self.vector_store.clear()

        texts = [document["text"] for document in documents]

        print("First text:", texts[0][:100] if texts else "EMPTY")

        embeddings = self.embedding_model.encode_documents(texts)

        print("Embeddings type:", type(embeddings))
        print("Embeddings shape:", getattr(embeddings, "shape", None))

        self.vector_store.add(
            embeddings=embeddings,
            documents=documents,
        )
    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def save_index(
        self,
        index_path: str,
        documents_path: str,
    ):
        Path(index_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Path(documents_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("Saving index to:", index_path)
        print("Saving documents to:", documents_path)

        self.vector_store.save(
            index_path=index_path,
            documents_path=documents_path,
        )
    def load_index(
        self,
        index_path: str,
        documents_path: str,
    ):
        self.vector_store.load(
            index_path=index_path,
            documents_path=documents_path,
        )

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int | None = None,
    ):

        if top_k is None:
            top_k = self.choose_top_k(question)

        # --------------------------------------------------
        # Final Retrieval Pipeline
        #
        # Dense Top-15
        #       ↓
        #   Reranker
        #       ↓
        #      Top-K
        # --------------------------------------------------

        dense_candidates = self.retriever.retrieve(
            query=question,
            top_k=15,
        )

        retrieved_chunks = self.reranker.rerank(
            query=question,
            documents=dense_candidates,
            top_k=top_k,
        )

        # --------------------------------------------------
        # Generation
        # --------------------------------------------------

        if not retrieved_chunks:
            return {
                "answer": (
                    "Sorry, I could not find relevant information "
                    "in the knowledge base to answer this question."
                ),
                "sources": [],
                "retrieved_chunks": [],
            }

        prompt = self.build_prompt(
            question,
            retrieved_chunks,
        )

        answer = self.generator.generate(
            prompt
        )

        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        sources = self.build_sources(
            retrieved_chunks
        )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
        }

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> str:

        context_parts = []

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            metadata = chunk.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
                "Unknown",
            )

            page = metadata.get(
                "page",
                "Unknown",
            )

            context_parts.append(
                f"[S{i}]\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(
            context_parts
        )

        prompt = f"""
You are a study assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not invent facts.
- If the context is insufficient, say so clearly.
- Use the retrieved sources when answering.
- Keep the answer focused on the user's question.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

        return prompt

    # --------------------------------------------------
    # Sources
    # --------------------------------------------------

    def build_sources(
        self,
        retrieved_documents: list[dict],
    ) -> list[dict]:

        sources = []

        for i, document in enumerate(
            retrieved_documents,
            start=1,
        ):
            metadata = document["metadata"]

            sources.append(
                {
                    "source_id": f"S{i}",
                    "document_id": metadata["document_id"],
                    "chunk_id": metadata["chunk_id"],
                    "source": metadata["source"],
                    "page": metadata.get("page"),
                    "score": document.get(
                        "score",
                        document.get(
                            "rerank_score",
                            0.0,
                        ),
                    ),
                }
            )

        return sources