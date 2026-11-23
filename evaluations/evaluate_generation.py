"""Run the 30-question RAG generation evaluation with Gemini API.

Run from the repository root:
    uv add google-genai
    uv run python evaluations/evaluate_generation.py

.env:
    GEMINI_API_KEY=...
    GEMINI_MODEL=<exact model ID that works with your API key>
    GEMINI_JUDGE_MODEL=<optional; defaults to GEMINI_MODEL>

The script resumes from successful records in the output JSON and checkpoints
progress after each question. It does not use Ollama.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "evaluations" / "generation_eval_dataset.json"
INDEX_PATH = PROJECT_DIR / "data" / "vector_store" / "index.faiss"
DOCUMENTS_PATH = PROJECT_DIR / "data" / "vector_store" / "documents.json"
DEFAULT_OUTPUT = PROJECT_DIR / "evaluations" / "results" / "generation_gemini.json"


def load_env_file(path: Path = PROJECT_DIR / ".env") -> None:
    """Load simple KEY=VALUE entries; no python-dotenv dependency required."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        # Drop optional inline comments (e.g. "gemini-... # note").
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        os.environ.setdefault(key.strip(), value.strip('"').strip("'"))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def parse_judge_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Gemini Judge returned invalid JSON: {text[:400]}")
        result = json.loads(match.group())
    if not isinstance(result, dict):
        raise TypeError("Gemini Judge response must be a JSON object.")
    return result


def _is_transient(exc: Exception) -> bool:
    message = str(exc).upper()
    return any(token in message for token in ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED", "REMOTEPROTOCOL", "DISCONNECT", "PROTOCOL"))


def _retry(call, attempts: int = 5, base_delay: float = 2.0):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(f"  Transient Gemini error (attempt {attempt}/{attempts}); retrying in {delay:.1f}s: {exc}")
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Retry loop exited without a result.")


class GeminiGenerator:
    """Adapter matching the generate(prompt) interface expected by RAGPipeline."""

    def __init__(self, client: Any, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        def _call() -> str:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            if not response.text:
                raise RuntimeError("Gemini returned an empty generation response.")
            return response.text.strip()

        return _retry(_call)


def build_pipeline(generator: GeminiGenerator):
    # Imports stay local so missing project dependencies produce clear errors.
    sys.path.insert(0, str(PROJECT_DIR))
    from src.chunking import Chunker
    from src.embeddings import EmbeddingModel
    from src.ingestion import PDFIngester
    from src.pipeline import RAGPipeline
    from src.vector_store import VectorStore

    if not INDEX_PATH.exists() or not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            "Prebuilt index/documents not found. Expected data/vector_store/"
            "index.faiss and data/vector_store/documents.json."
        )

    embedding_model = EmbeddingModel("intfloat/multilingual-e5-base", device="cpu")
    vector_store = VectorStore(dimension=embedding_model.dimension())
    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        ingester=PDFIngester(),
        generator=generator,
        chunker=Chunker(),
    )
    pipeline.load_index(str(INDEX_PATH), str(DOCUMENTS_PATH))
    return pipeline


def judge_answer(
    client: Any,
    model_name: str,
    item: dict[str, Any],
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    context = "\n\n".join(
        f"[Chunk {i}] {chunk.get('text', '')}"
        for i, chunk in enumerate(retrieved_chunks, start=1)
    )
    criteria = "\n".join(f"- {criterion}" for criterion in item["evaluation_criteria"])
    prompt = f"""
You are a strict evaluator of a Retrieval-Augmented Generation (RAG) study assistant.
Evaluate the generated answer against BOTH the reference answer and the retrieved
context. The reference answer describes expected content; the retrieved context is
the authority for whether claims are grounded. Do not reward unsupported claims.

Return ONLY valid JSON with this schema:
{{
  "correctness": 1,
  "faithfulness": 1,
  "relevance": 1,
  "completeness": 1,
  "overall": 1,
  "failure_category": "none",
  "reason": "brief explanation",
  "unsupported_claims": []
}}
All scores must be integers from 1 (very poor) to 5 (excellent).
Allowed failure_category values: none, retrieval, context_insufficient,
hallucination, incorrect_answer, incomplete_answer, irrelevant_answer.

QUESTION:
{item['question']}

REFERENCE ANSWER:
{item['reference_answer']}

EVALUATION CRITERIA:
{criteria}

RETRIEVED CONTEXT:
{context or '[NO CONTEXT RETRIEVED]'}

GENERATED ANSWER:
{answer}
"""
    def _call() -> str:
        response = client.models.generate_content(model=model_name, contents=prompt)
        if not response.text:
            raise RuntimeError("Gemini Judge returned an empty response.")
        return response.text

    text = _retry(_call)
    return parse_judge_json(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and judge answers for the 30-question RAG dataset using Gemini."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only N pending questions; default is all remaining questions.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Path for checkpointed JSON results.")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between questions (default: 1).")
    args = parser.parse_args()

    load_env_file()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    generation_model = os.getenv("GEMINI_MODEL")
    judge_model = os.getenv("GEMINI_JUDGE_MODEL") or generation_model
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is missing from .env.")
    if not generation_model:
        raise RuntimeError(
            "Set GEMINI_MODEL in .env to the exact Gemini model ID that already works for you."
        )
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Generation dataset missing: {DATASET_PATH}")

    dataset = load_json(DATASET_PATH)
    if not isinstance(dataset, list) or len(dataset) != 30:
        raise ValueError("Expected generation_eval_dataset.json to contain exactly 30 questions.")

    output_path = args.output if args.output.is_absolute() else PROJECT_DIR / args.output
    previous_payload = load_json(output_path) if output_path.exists() else []
    # Support both the per-question checkpoint list and the final report object.
    previous = previous_payload.get("results", []) if isinstance(previous_payload, dict) else previous_payload
    completed_ids = {
        row.get("id") for row in previous
        if row.get("status") == "success" and row.get("answer") and row.get("judge")
    }
    pending = [item for item in dataset if item["id"] not in completed_ids]
    if args.limit is not None:
        pending = pending[:max(0, args.limit)]
    if not pending:
        print("No pending questions. All successful results are already saved.")
        return

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Gemini SDK missing. Install it with: uv add google-genai") from exc

    client = genai.Client(api_key=api_key)
    pipeline = build_pipeline(GeminiGenerator(client, generation_model))
    print(
        f"Loaded RAG index | Gemini generation: {generation_model} | "
        f"Gemini judge: {judge_model} | pending: {len(pending)}"
    )
    results = list(previous)

    for number, item in enumerate(pending, start=1):
        print(f"[{number}/{len(pending)}] {item['id']} — {item['question']}")
        try:
            rag_result = pipeline.query(item["question"])
            answer = rag_result.get("answer", "")
            chunks = rag_result.get("retrieved_chunks", [])
            scores = judge_answer(client,str(judge_model), item, answer, chunks)
            row = {
                **item,
                "answer": answer,
                "sources": rag_result.get("sources", []),
                "retrieved_chunk_ids": [
                    chunk.get("metadata", {}).get("chunk_id") for chunk in chunks
                ],
                "judge": scores,
                "status": "success",
                "generation_model": generation_model,
                "judge_model": judge_model,
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  Overall: {scores.get('overall', 'n/a')}/5")
        except Exception as exc:
            row = {
                "id": item["id"],
                "question": item["question"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  ERROR: {row['error']}")

        results = [existing for existing in results if existing.get("id") != item["id"]]
        results.append(row)
        save_json(output_path, results)  # checkpoint each question
        if args.delay > 0 and number < len(pending):
            time.sleep(args.delay)

    successful = [r for r in results if r.get("status") == "success" and r.get("judge")]
    metric_names = ["correctness", "faithfulness", "relevance", "completeness", "overall"]
    summary = {
        "completed": len(successful),
        "failed": sum(r.get("status") == "error" for r in results),
        "average_scores_out_of_5": {
            metric: round(
                sum(r["judge"][metric] for r in successful
                    if isinstance(r["judge"].get(metric), (int, float))) /
                max(1, sum(isinstance(r["judge"].get(metric), (int, float)) for r in successful)),
                2,
            ) if any(isinstance(r["judge"].get(metric), (int, float)) for r in successful)
            else None
            for metric in metric_names
        },
    }
    report = {
        "metadata": {
            "generation_model": generation_model,
            "judge_model": judge_model,
            "dataset": "evaluations/generation_eval_dataset.json",
            "dataset_size": 30,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "results": results,
    }
    save_json(output_path, report)
    print("\nGeneration evaluation finished.")
    print(f"Saved report: {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
