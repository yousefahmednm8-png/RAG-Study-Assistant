def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk += (
                paragraph
                if not current_chunk
                else "\n\n" + paragraph
            )

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If a paragraph itself is larger than chunk_size,
            # split it using the original character-based approach.
            if len(paragraph) > chunk_size:

                start = 0

                while start < len(paragraph):

                    end = start + chunk_size
                    small_chunk = paragraph[start:end]

                    if small_chunk.strip():
                        chunks.append(small_chunk.strip())

                    start = end - chunk_overlap

                current_chunk = ""

            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def create_chunks(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:

    chunked_documents = []

    for document in documents:

        text = document["text"]

        metadata = document.get("metadata", {})

        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk in enumerate(chunks):

            chunk_metadata = metadata.copy()

            chunk_metadata["chunk_index"] = chunk_index

            document_id = chunk_metadata.get(
                "document_id",
                chunk_metadata.get("source", "document"),
            )

            page = chunk_metadata.get("page")

            if page is not None:
                chunk_metadata["chunk_id"] = (
                    f"{document_id}_page_{page}_chunk_{chunk_index}"
                )
            else:
                chunk_metadata["chunk_id"] = (
                    f"{document_id}_chunk_{chunk_index}"
                )

            chunked_documents.append(
                {
                    "text": chunk,
                    "metadata": chunk_metadata,
                }
            )

    return chunked_documents
class Chunker:

    def chunk(self, documents):
        return create_chunks(documents)