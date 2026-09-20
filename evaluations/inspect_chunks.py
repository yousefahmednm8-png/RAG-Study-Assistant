import json
from pathlib import Path

from src.vector_store import VectorStore
from src.embeddings import EmbeddingModel


EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = PROJECT_ROOT / "data" / "vector_store" / "index.faiss"
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "vector_store" / "documents.json"


embedding_model = EmbeddingModel(
    model_name=EMBEDDING_MODEL
)

vector_store = VectorStore(
    dimension=embedding_model.dimension()
)

vector_store.load(
    index_path=INDEX_PATH,
    documents_path=DOCUMENTS_PATH,
)


for document in vector_store.documents:

    metadata = document["metadata"]

    print("=" * 80)

    print(
        f"chunk_id: {metadata.get('chunk_id')}"
    )

    print(
        f"source: {metadata.get('source')}"
    )

    print(
        f"page: {metadata.get('page')}"
    )

    print()

    print(
        document["text"][:500]
    )