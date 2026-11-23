import os
from pathlib import Path

import streamlit as st

from src.chunking import Chunker
from src.embeddings import EmbeddingModel
from src.generation import Generator
from src.ingestion import PDFIngester
from src.pipeline import RAGPipeline
from src.vector_store import VectorStore

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .app-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 1.6rem;
    }

    .hero-card {
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.10), rgba(236,72,153,0.08));
        border: 1px solid rgba(99,102,241,0.25);
        box-shadow: 0 8px 24px rgba(99,102,241,0.12);
        margin-bottom: 1.6rem;
    }

    .pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        margin: 0.15rem 0.3rem 0.15rem 0;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a5b4fc;
    }

    .source-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px;
        margin-bottom: 0.8rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.05), rgba(139,92,246,0.04));
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(99,102,241,0.15);
    }

    .source-title { font-weight: 600; margin-bottom: 0.3rem; }
    .source-meta { font-size: 0.85rem; color: #9ca3af; }

    .status-ready, .status-warning {
        padding: 0.7rem 0.9rem;
        border-radius: 10px;
        font-weight: 500;
    }
    .status-ready {
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.30);
        color: #4ade80;
    }
    .status-warning {
        background: rgba(251,191,36,0.10);
        border: 1px solid rgba(251,191,36,0.30);
        color: #fbbf24;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(99,102,241,0.35);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(99,102,241,0.06), rgba(15,15,20,0.0));
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
def create_pipeline(provider: str = "auto"):

    embedding_model = EmbeddingModel(model_name="intfloat/multilingual-e5-base")

    vector_store = VectorStore(dimension=embedding_model.dimension())

    ingester = PDFIngester()

    chunker = Chunker()

    use_gemini = (provider == "gemini") or (
        provider == "auto"
        and bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    )

    if use_gemini:
        from src.generation import GeminiGenerator

        generator = GeminiGenerator(
            model_name=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        )
    else:
        generator = Generator(
            model_name="gemma4:31b-cloud",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        ingester=ingester,
        generator=generator,
        chunker=chunker,
    )

    return pipeline


def get_pipeline(provider: str = "auto"):

    if st.session_state.pipeline is None:

        with st.spinner("Loading AI Study Assistant..."):
            st.session_state.pipeline = create_pipeline(provider)

    return st.session_state.pipeline


# ============================================================
# Load existing index
# ============================================================

def load_existing_index():

    pipeline = get_pipeline(st.session_state.get("provider", "auto"))

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
    <div class="hero-card">
      <div style="font-size:1.15rem;font-weight:600;margin-bottom:0.4rem;">Welcome 👋</div>
      <div style="color:#cbd5e1;font-size:0.95rem;">
        Upload PDFs or TXT files in the sidebar, build the knowledge base,
        then ask anything. Answers are grounded in your documents and cite
        their sources.
      </div>
      <div style="margin-top:0.8rem;">
        <span class="pill">RAG</span>
        <span class="pill">FAISS</span>
        <span class="pill">Ollama / Gemini</span>
        <span class="pill">Reranking</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Settings")

    provider = st.selectbox(
        "AI provider",
        ["auto", "gemini", "ollama"],
        index=0,
        help="auto uses Gemini if GEMINI_API_KEY is set, else Ollama.",
    )

    if st.session_state.get("provider") != provider:
        st.session_state.provider = provider
        st.session_state.pipeline = None

    st.divider()

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

            pipeline = get_pipeline(st.session_state.get("provider", "auto"))

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

        pipeline = get_pipeline(st.session_state.get("provider", "auto"))

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