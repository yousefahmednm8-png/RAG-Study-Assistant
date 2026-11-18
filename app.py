from pathlib import Path
import streamlit as st
from src.ingestion import PDFIngester
from src.pipeline import RAGPipeline

from src.chunking import Chunker
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.generation import Generator


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .app-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        font-size: 1.05rem;
        color: #777;
        margin-bottom: 2rem;
    }

    /* Cards */
    .source-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        margin-bottom: 0.8rem;
        background-color: rgba(128, 128, 128, 0.04);
    }

    .source-title {
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    .source-meta {
        font-size: 0.85rem;
        color: #777;
    }

    /* Status */
    .status-ready {
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        background-color: rgba(0, 180, 100, 0.10);
        border: 1px solid rgba(0, 180, 100, 0.25);
    }

    .status-warning {
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        background-color: rgba(255, 170, 0, 0.10);
        border: 1px solid rgba(255, 170, 0, 0.25);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "vector_store"

INDEX_PATH = INDEX_DIR / "index.faiss"
DOCUMENTS_PATH = INDEX_DIR / "documents.json"

UPLOAD_DIR = DATA_DIR / "uploads"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)


# ============================================================
# Session state
# ============================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "index_ready" not in st.session_state:
    st.session_state.index_ready = (
        INDEX_PATH.exists() and DOCUMENTS_PATH.exists()
    )

if "answer" not in st.session_state:
    st.session_state.answer = None

if "sources" not in st.session_state:
    st.session_state.sources = []

if "question" not in st.session_state:
    st.session_state.question = ""


# ============================================================
# Pipeline initialization
# ============================================================

@st.cache_resource
def create_pipeline():

    embedding_model = EmbeddingModel(model_name="intfloat/multilingual-e5-base")

    vector_store = VectorStore(dimension=embedding_model.dimension())

    ingester = PDFIngester()

    chunker = Chunker()

    generator = Generator(
        model_name="gemma4:31b-cloud",
        base_url="http://localhost:11434",
    )

    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        ingester=ingester,
        generator=generator,
        chunker=chunker,
    )

    return pipeline


def get_pipeline():

    if st.session_state.pipeline is None:

        with st.spinner("Loading AI Study Assistant..."):
            st.session_state.pipeline = create_pipeline()

    return st.session_state.pipeline


# ============================================================
# Load existing index
# ============================================================

def load_existing_index():

    pipeline = get_pipeline()

    pipeline.load_index(
        index_path=str(INDEX_PATH),
        documents_path=str(DOCUMENTS_PATH),
    )

    st.session_state.index_ready = True


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="app-title">📚 AI Study Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-subtitle">
    Ask questions about your study materials and get answers grounded in
    the documents you provide.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Knowledge Base")

    st.caption(
        "Upload study materials to build the knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF and TXT",
    )

    if uploaded_files:

        if st.button(
            "Build Knowledge Base",
            type="primary",
            use_container_width=True,
        ):

            pipeline = get_pipeline()

            documents = []

            progress = st.progress(0)

            try:

                for i, uploaded_file in enumerate(uploaded_files):

                    file_path = UPLOAD_DIR / uploaded_file.name

                    file_path.write_bytes(
                        uploaded_file.getbuffer()
                    )

                    # Ingestion
                    ingested = pipeline.ingester.ingest(
                        str(file_path)
                    )

                    # Chunking
                    chunks = pipeline.chunker.chunk(
                        ingested
                    )

                    documents.extend(chunks)

                    progress.progress(
                        (i + 1) / len(uploaded_files)
                    )

                if not documents:
                    st.error(
                        "No text could be extracted from the uploaded files."
                    )

                else:

                    with st.spinner(
                        "Creating embeddings and indexing documents..."
                    ):

                        pipeline.index_documents(documents)

                        pipeline.save_index(
                            index_path=str(INDEX_PATH),
                            documents_path=str(DOCUMENTS_PATH),
                        )

                    st.session_state.index_ready = True

                    st.success(
                        f"Knowledge base ready — "
                        f"{len(documents)} chunks indexed."
                    )

            except Exception as e:

                st.error(
                    f"Indexing failed: {e}"
                )

    st.divider()

    # Index status
    if st.session_state.index_ready:

        st.markdown(
            """
            <div class="status-ready">
            🟢 <b>Knowledge base ready</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="status-warning">
            🟡 <b>No knowledge base loaded</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.caption(
        "Local AI • Ollama"
    )


# ============================================================
# Main content
# ============================================================

if not st.session_state.index_ready:

    st.info(
        "Upload your study materials from the sidebar and build the "
        "knowledge base to start asking questions."
    )

else:

    # Make sure index is loaded
    try:

        pipeline = get_pipeline()

        if not pipeline.vector_store.documents:
            load_existing_index()

    except Exception as e:

        st.error(
            f"Could not load the knowledge base: {e}"
        )
        st.stop()

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    st.subheader("Ask a question")

    question = st.text_area(
        "Question",
        placeholder=(
            "Example: What are the main differences between "
            "supervised and unsupervised learning?"
        ),
        height=110,
        label_visibility="collapsed",
    )

    ask = st.button(
        "Ask AI",
        type="primary",
        use_container_width=False,
    )

    if ask:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching your documents and generating an answer..."
            ):

                try:

                    result = pipeline.query(
                        question=question.strip()
                    )

                    st.session_state.answer = result["answer"]
                    st.session_state.sources = result["sources"]
                    st.session_state.question = question.strip()

                except Exception as e:

                    st.error(
                        f"Something went wrong: {e}"
                    )


    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    if st.session_state.answer:

        st.divider()

        st.subheader("Answer")

        st.markdown(
            st.session_state.answer
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        sources = st.session_state.sources

        if sources:

            st.divider()

            st.subheader(
                f"Sources ({len(sources)})"
            )

            for source in sources:

                source_id = source.get(
                    "source_id",
                    "S?",
                )

                source_name = source.get(
                    "source",
                    "Unknown source",
                )

                page = source.get(
                    "page",
                    "Unknown",
                )

                chunk_id = source.get(
                    "chunk_id",
                    "Unknown",
                )

                score = source.get(
                    "score"
                )

                with st.expander(
                    f"{source_id}  •  {source_name}  •  Page {page}"
                ):

                    col1, col2 = st.columns(2)

                    with col1:
                        st.caption(
                            f"Document: {source.get('document_id', 'Unknown')}"
                        )

                    with col2:
                        st.caption(
                            f"Chunk: {chunk_id}"
                        )

                    if score is not None:

                        st.caption(
                            f"Relevance score: {score:.4f}"
                        )