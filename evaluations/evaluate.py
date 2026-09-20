import json
from pathlib import Path

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.pipeline import RAGPipeline
from src.bm25_retrieval import BM25Retriever


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INDEX_PATH = (
    PROJECT_DIR
    / "data"
    / "vector_store"
    / "index.faiss"
)

DOCUMENTS_PATH = (
    PROJECT_DIR
    / "data"
    / "vector_store"
    / "documents.json"
)

QUESTIONS_PATH = (
    PROJECT_DIR
    / "evaluations"
    / "questions.json"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "evaluations"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Output Files
# ============================================================

SUMMARY_PATH = (
    RESULTS_DIR
    / "retrieval_ablation_v2_summary.json"
)

DETAILS_PATH = (
    RESULTS_DIR
    / "retrieval_ablation_v2_details.json"
)


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL = (
    "intfloat/multilingual-e5-base"
)

# Evaluation cutoffs
EVAL_KS = [1, 3, 5, 10, 15]

# Dense -> Reranker candidate sizes
DENSE_RERANK_CANDIDATE_KS = [
    15,
    30,
    50,
    100,
]

# Hybrid Dense + BM25 -> RRF
HYBRID_CANDIDATE_K = 15


# ============================================================
# Data Loading
# ============================================================

def load_questions(path: Path) -> list[dict]:
    """Load evaluation questions from JSON."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# Metrics
# ============================================================

def hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> int:
    """Return 1 if at least one relevant chunk appears in top-k."""

    top_k = retrieved_ids[:k]

    return int(
        any(
            chunk_id in relevant_ids
            for chunk_id in top_k
        )
    )


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Recall@K."""

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    found = sum(
        chunk_id in relevant_ids
        for chunk_id in top_k
    )

    return found / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Precision@K."""

    if k == 0:
        return 0.0

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    found = sum(
        chunk_id in relevant_ids
        for chunk_id in top_k
    )

    return found / len(top_k)


def reciprocal_rank_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate MRR@K."""

    top_k = retrieved_ids[:k]

    for rank, chunk_id in enumerate(
        top_k,
        start=1,
    ):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


# ============================================================
# Evaluate One Retrieval Configuration
# ============================================================

def evaluate_retriever(
    retriever_function,
    questions: list[dict],
    ks: list[int],
):
    """Evaluate a retrieval configuration."""

    summary = {}
    details = []

    max_k = max(ks)

    # --------------------------------------------------------
    # Per-question evaluation
    # --------------------------------------------------------

    for item in questions:

        question = item["question"]

        relevant_ids = set(
            item["relevant_chunks"]
        )

        # Retrieve enough results for the largest K
        results = retriever_function(
            question,
            max_k,
        )

        retrieved_ids = [
            document["metadata"]["chunk_id"]
            for document in results
        ]

        # ----------------------------------------------------
        # First Relevant Rank
        # ----------------------------------------------------

        first_relevant_rank = None

        for rank, chunk_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            if chunk_id in relevant_ids:
                first_relevant_rank = rank
                break

        # ----------------------------------------------------
        # Per-question result
        # ----------------------------------------------------

        question_result = {
            "question": question,
            "relevant_chunks": list(
                relevant_ids
            ),
            "retrieved_chunks": retrieved_ids,
            "first_relevant_rank": (
                first_relevant_rank
            ),
        }

        # ----------------------------------------------------
        # Metrics for Every K
        # ----------------------------------------------------

        question_metrics = {}

        for k in ks:

            question_metrics[
                f"hit@{k}"
            ] = hit_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )

            question_metrics[
                f"recall@{k}"
            ] = recall_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )

            question_metrics[
                f"precision@{k}"
            ] = precision_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )

            question_metrics[
                f"mrr@{k}"
            ] = reciprocal_rank_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )

        question_result["metrics"] = (
            question_metrics
        )

        details.append(
            question_result
        )

    # --------------------------------------------------------
    # Aggregate Metrics
    # --------------------------------------------------------

    num_questions = len(details)

    for k in ks:

        if num_questions == 0:

            summary[f"hit@{k}"] = 0.0
            summary[f"recall@{k}"] = 0.0
            summary[f"precision@{k}"] = 0.0
            summary[f"mrr@{k}"] = 0.0

            continue

        summary[f"hit@{k}"] = (
            sum(
                item["metrics"][
                    f"hit@{k}"
                ]
                for item in details
            )
            / num_questions
        )

        summary[f"recall@{k}"] = (
            sum(
                item["metrics"][
                    f"recall@{k}"
                ]
                for item in details
            )
            / num_questions
        )

        summary[f"precision@{k}"] = (
            sum(
                item["metrics"][
                    f"precision@{k}"
                ]
                for item in details
            )
            / num_questions
        )

        summary[f"mrr@{k}"] = (
            sum(
                item["metrics"][
                    f"mrr@{k}"
                ]
                for item in details
            )
            / num_questions
        )

    return summary, details


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("RAG Retrieval Ablation Evaluation v2")
    print("=" * 60)

    # ========================================================
    # Load Evaluation Questions
    # ========================================================

    questions = load_questions(
        QUESTIONS_PATH
    )

    print(
        f"\nNumber of evaluation questions: "
        f"{len(questions)}"
    )

    # ========================================================
    # Load Embedding Model
    # ========================================================

    print(
        f"\nLoading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    embedding_model = EmbeddingModel(
        model_name=EMBEDDING_MODEL
    )

    # ========================================================
    # Load Vector Store
    # ========================================================

    vector_store = VectorStore(
        dimension=embedding_model.dimension()
    )

    print("\nLoading vector store...")

    vector_store.load(
        index_path=str(INDEX_PATH),
        documents_path=str(DOCUMENTS_PATH),
    )

    print(
        f"Loaded documents: "
        f"{len(vector_store.documents)}"
    )

    # ========================================================
    # Build Pipeline
    # ========================================================

    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        ingester=None,
        generator=None,
        chunker=None,
    )

    # ========================================================
    # Build BM25 Retriever
    # ========================================================

    bm25_retriever = BM25Retriever()

    bm25_retriever.build(
        vector_store.documents
    )

    # ========================================================
    # Retrieval Functions
    # ========================================================

    # --------------------------------------------------------
    # 1. Dense Retrieval
    # --------------------------------------------------------

    def dense(question, k):

        return pipeline.retriever.retrieve(
            query=question,
            top_k=k,
        )

    # --------------------------------------------------------
    # 2. BM25 Retrieval
    # --------------------------------------------------------

    def bm25(question, k):

        return bm25_retriever.retrieve(
            query=question,
            top_k=k,
        )

    # --------------------------------------------------------
    # 3. Hybrid Dense + BM25 + RRF
    # --------------------------------------------------------

    def rrf_fuse(question, top_k, candidate_k):

        dense_results = pipeline.retriever.retrieve(
            query=question,
            top_k=candidate_k,
        )

        bm25_results = bm25_retriever.retrieve(
            query=question,
            top_k=candidate_k,
        )

        rrf_k = 60
        rrf_scores = {}

        for rank, document in enumerate(
            dense_results,
            start=1,
        ):
            chunk_id = document["metadata"]["chunk_id"]

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

        for rank, document in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = document["metadata"]["chunk_id"]

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

        ranked_chunk_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )

        by_chunk_id = {}

        for document in dense_results:
            by_chunk_id[
                document["metadata"]["chunk_id"]
            ] = document

        for document in bm25_results:
            by_chunk_id[
                document["metadata"]["chunk_id"]
            ] = document

        results = []

        for chunk_id in ranked_chunk_ids[:top_k]:
            document = by_chunk_id[chunk_id].copy()
            document["rrf_score"] = rrf_scores[chunk_id]
            results.append(document)

        return results

    def hybrid(question, k):

        return rrf_fuse(
            question=question,
            top_k=k,
            candidate_k=HYBRID_CANDIDATE_K,
        )

    # --------------------------------------------------------
    # 4. Hybrid + RRF + Reranker
    # --------------------------------------------------------

    def hybrid_reranked(question, k):

        candidates = rrf_fuse(
            question=question,
            top_k=HYBRID_CANDIDATE_K,
            candidate_k=HYBRID_CANDIDATE_K,
        )

        return pipeline.reranker.rerank(
            query=question,
            documents=candidates,
            top_k=k,
        )

    # ========================================================
    # Register Normal Experiments
    # ========================================================

    retrievers = {
        "dense": dense,
        "bm25": bm25,
        "hybrid_rrf": hybrid,
        "hybrid_rrf_reranker": hybrid_reranked,
    }

    # ========================================================
    # Run Normal Evaluation
    # ========================================================

    all_summary = {}
    all_details = {}

    for name, retriever in retrievers.items():

        print("\n")
        print("=" * 60)
        print(f"Evaluating: {name}")
        print("=" * 60)

        summary, details = evaluate_retriever(
            retriever_function=retriever,
            questions=questions,
            ks=EVAL_KS,
        )

        all_summary[name] = summary
        all_details[name] = details

        print("\nResults:")

        for k in EVAL_KS:

            print(
                f"K={k:2d} | "
                f"Hit={summary[f'hit@{k}']:.4f} | "
                f"Recall={summary[f'recall@{k}']:.4f} | "
                f"Precision={summary[f'precision@{k}']:.4f} | "
                f"MRR={summary[f'mrr@{k}']:.4f}"
            )

    # ========================================================
    # Dense -> Reranker Candidate-K Ablation
    # ========================================================

    print("\n")
    print("=" * 60)
    print("Dense -> Reranker Candidate-K Ablation")
    print("=" * 60)

    all_summary["dense_reranker"] = {}
    all_details["dense_reranker"] = {}

    for candidate_k in DENSE_RERANK_CANDIDATE_KS:

        experiment_name = (
            f"candidate_k_{candidate_k}"
        )

        print("\n")
        print("-" * 60)
        print(
            f"Evaluating: Dense Top-{candidate_k} "
            f"-> Reranker"
        )
        print("-" * 60)

        # ----------------------------------------------------
        # Create Retriever for This Candidate K
        # ----------------------------------------------------

        def dense_reranked(
            question,
            k,
            candidate_k=candidate_k,
        ):

            candidates = (
                pipeline.retriever.retrieve(
                    query=question,
                    top_k=candidate_k,
                )
            )

            return pipeline.reranker.rerank(
                query=question,
                documents=candidates,
                top_k=k,
            )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        summary, details = evaluate_retriever(
            retriever_function=dense_reranked,
            questions=questions,
            ks=EVAL_KS,
        )

        all_summary[
            "dense_reranker"
        ][experiment_name] = summary

        all_details[
            "dense_reranker"
        ][experiment_name] = details

        # ----------------------------------------------------
        # Print Results
        # ----------------------------------------------------

        print("\nResults:")

        for k in EVAL_KS:

            print(
                f"K={k:2d} | "
                f"Hit={summary[f'hit@{k}']:.4f} | "
                f"Recall={summary[f'recall@{k}']:.4f} | "
                f"Precision={summary[f'precision@{k}']:.4f} | "
                f"MRR={summary[f'mrr@{k}']:.4f}"
            )

    # ========================================================
    # Candidate-K Comparison
    # ========================================================

    print("\n")
    print("=" * 60)
    print("Dense -> Reranker Candidate-K Comparison")
    print("=" * 60)

    print(
        "\n"
        "Candidate K | "
        "Hit@5 | "
        "Recall@5 | "
        "Precision@5 | "
        "MRR@5"
    )

    print("-" * 60)

    for candidate_k in DENSE_RERANK_CANDIDATE_KS:

        summary = all_summary[
            "dense_reranker"
        ][
            f"candidate_k_{candidate_k}"
        ]

        print(
            f"{candidate_k:11d} | "
            f"{summary['hit@5']:.4f} | "
            f"{summary['recall@5']:.4f} | "
            f"{summary['precision@5']:.4f} | "
            f"{summary['mrr@5']:.4f}"
        )

    # ========================================================
    # Save Summary
    # ========================================================

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_summary,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # ========================================================
    # Save Detailed Results
    # ========================================================

    with open(
        DETAILS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_details,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # ========================================================
    # Done
    # ========================================================

    print("\n")
    print("=" * 60)
    print("Evaluation completed.")
    print("=" * 60)

    print(
        f"\nSummary saved to:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"\nDetailed per-question results saved to:\n"
        f"{DETAILS_PATH}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()