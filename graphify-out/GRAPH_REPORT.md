# Graph Report - .  (2026-11-24)

## Corpus Check
- Corpus is ~8,082 words - fits in a single context window. You may not need a graph.

## Summary
- 199 nodes · 163 edges · 80 communities detected
- Extraction: 71% EXTRACTED · 29% INFERRED · 0% AMBIGUOUS · INFERRED: 47 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `RAGPipeline` - 16 edges
2. `build_pipeline()` - 10 edges
3. `GeminiGenerator` - 9 edges
4. `VectorStore` - 8 edges
5. `main()` - 7 edges
6. `EmbeddingModel` - 7 edges
7. `FakeVectorStore` - 7 edges
8. `evaluate_retriever()` - 6 edges
9. `FakeEmbeddingModel` - 6 edges
10. `StubSentenceTransformer` - 5 edges

## Surprising Connections (you probably didn't know these)
- `GeminiGenerator` --uses--> `Chunker`  [INFERRED]
  evaluations/evaluate_generation.py → src/chunking.py
- `GeminiGenerator` --uses--> `EmbeddingModel`  [INFERRED]
  evaluations/evaluate_generation.py → src/embeddings.py
- `GeminiGenerator` --uses--> `PDFIngester`  [INFERRED]
  evaluations/evaluate_generation.py → src/ingestion.py
- `GeminiGenerator` --uses--> `RAGPipeline`  [INFERRED]
  evaluations/evaluate_generation.py → src/pipeline.py
- `GeminiGenerator` --uses--> `VectorStore`  [INFERRED]
  evaluations/evaluate_generation.py → src/vector_store.py

## Communities

### Community 0 - "Test"
Cohesion: 0.12
Nodes (10): build_pipeline(), FakeEmbeddingModel, FakeGenerator, FakeReranker, FakeVectorStore, test_index_documents_replaces_not_duplicates(), test_pipeline_embeds_plain_question_string(), test_pipeline_empty_retrieval_skips_generation() (+2 more)

### Community 1 - "Pipeline"
Cohesion: 0.13
Nodes (3): RAGPipeline, Reranker, Retriever

### Community 2 - "Test"
Cohesion: 0.14
Nodes (2): EmbeddingModel, StubSentenceTransformer

### Community 3 - "Evaluate"
Cohesion: 0.28
Nodes (10): build_pipeline(), GeminiGenerator, _is_transient(), judge_answer(), load_env_file(), load_json(), main(), parse_judge_json() (+2 more)

### Community 4 - "Vector"
Cohesion: 0.22
Nodes (2): FAISS, VectorStore

### Community 5 - "Evaluate"
Cohesion: 0.46
Nodes (7): evaluate_retriever(), hit_at_k(), load_questions(), main(), precision_at_k(), recall_at_k(), reciprocal_rank_at_k()

### Community 6 - "Ingestion"
Cohesion: 0.43
Nodes (5): load_document(), load_pdf(), load_text(), PDFIngester, pypdf

### Community 7 - "Test"
Cohesion: 0.29
Nodes (0): 

### Community 8 - "Test"
Cohesion: 0.29
Nodes (0): 

### Community 9 - "Test"
Cohesion: 0.33
Nodes (0): 

### Community 10 - "App"
Cohesion: 0.6
Nodes (4): create_pipeline(), get_pipeline(), load_existing_index(), Streamlit

### Community 11 - "Chunking"
Cohesion: 0.6
Nodes (3): chunk_text(), Chunker, create_chunks()

### Community 12 - "Test"
Cohesion: 0.4
Nodes (0): 

### Community 13 - "Generation"
Cohesion: 0.5
Nodes (1): Generator

### Community 14 - "Test"
Cohesion: 0.67
Nodes (0): 

### Community 15 - "Inspect"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Init"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Readme"
Cohesion: 1.0
Nodes (1): RAG Study Assistant

### Community 18 - "Requirements"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "Yousef"
Cohesion: 1.0
Nodes (1): Yousef Ahmed AI CV (ATS)

### Community 20 - "Ai"
Cohesion: 1.0
Nodes (1): AI in Data Engineering (SB)

### Community 21 - "Doc"
Cohesion: 1.0
Nodes (1): Physics Student Guide (doc-2)

### Community 22 - "Ecommerce"
Cohesion: 1.0
Nodes (1): E-Commerce Sales Mini Project

### Community 23 - "English"
Cohesion: 1.0
Nodes (1): English AI (SB)

### Community 24 - "Arabic"
Cohesion: 1.0
Nodes (1): Exam Vocabulary Guide

### Community 25 - "Ollama"
Cohesion: 1.0
Nodes (1): Ollama

### Community 26 - "Gemini"
Cohesion: 1.0
Nodes (1): Gemini

### Community 27 - "E5"
Cohesion: 1.0
Nodes (1): intfloat/multilingual-e5-base

### Community 28 - "Bge"
Cohesion: 1.0
Nodes (1): BAAI/bge-reranker-v2-m3

### Community 29 - "Rag"
Cohesion: 1.0
Nodes (1): Retrieval-Augmented Generation

### Community 30 - "Dense"
Cohesion: 1.0
Nodes (1): Dense Retrieval

### Community 31 - "Reranking"
Cohesion: 1.0
Nodes (1): Reranking

### Community 32 - "Yousef"
Cohesion: 1.0
Nodes (1): Yousef Ahmed

### Community 33 - "Distilbert"
Cohesion: 1.0
Nodes (1): DistilBERT

### Community 34 - "Pytorch"
Cohesion: 1.0
Nodes (1): PyTorch

### Community 35 - "Tensorflow"
Cohesion: 1.0
Nodes (1): TensorFlow

### Community 36 - "Keras"
Cohesion: 1.0
Nodes (1): Keras

### Community 37 - "Machine"
Cohesion: 1.0
Nodes (1): Machine Learning

### Community 38 - "Deep"
Cohesion: 1.0
Nodes (1): Deep Learning

### Community 39 - "Nlp"
Cohesion: 1.0
Nodes (1): Natural Language Processing

### Community 40 - "Computer"
Cohesion: 1.0
Nodes (1): Computer Vision

### Community 41 - "Banking77"
Cohesion: 1.0
Nodes (1): Banking77 Intent Classification

### Community 42 - "Ag"
Cohesion: 1.0
Nodes (1): AG News Text Classification

### Community 43 - "Imdb"
Cohesion: 1.0
Nodes (1): IMDB Sentiment Analysis

### Community 44 - "Netflix"
Cohesion: 1.0
Nodes (1): Netflix User Churn Prediction

### Community 45 - "Force"
Cohesion: 1.0
Nodes (1): Force

### Community 46 - "Gravitational"
Cohesion: 1.0
Nodes (1): Gravitational Force

### Community 47 - "Centripetal"
Cohesion: 1.0
Nodes (1): Centripetal Force

### Community 48 - "Friction"
Cohesion: 1.0
Nodes (1): Friction Force

### Community 49 - "Viscosity"
Cohesion: 1.0
Nodes (1): Viscosity

### Community 50 - "Lubrication"
Cohesion: 1.0
Nodes (1): Lubrication

### Community 51 - "Heat"
Cohesion: 1.0
Nodes (1): Heat

### Community 52 - "Temperature"
Cohesion: 1.0
Nodes (1): Temperature

### Community 53 - "Thermal"
Cohesion: 1.0
Nodes (1): Thermal Energy

### Community 54 - "Heat"
Cohesion: 1.0
Nodes (1): Heat Transfer

### Community 55 - "Conduction"
Cohesion: 1.0
Nodes (1): Conduction

### Community 56 - "Convection"
Cohesion: 1.0
Nodes (1): Convection

### Community 57 - "Radiation"
Cohesion: 1.0
Nodes (1): Radiation

### Community 58 - "Specific"
Cohesion: 1.0
Nodes (1): Specific Heat

### Community 59 - "Heat"
Cohesion: 1.0
Nodes (1): Heat Capacity

### Community 60 - "Thermometer"
Cohesion: 1.0
Nodes (1): Thermometer

### Community 61 - "Pandas"
Cohesion: 1.0
Nodes (1): Pandas

### Community 62 - "Matplotlib"
Cohesion: 1.0
Nodes (1): Matplotlib

### Community 63 - "Seaborn"
Cohesion: 1.0
Nodes (1): Seaborn

### Community 64 - "Groupby"
Cohesion: 1.0
Nodes (1): GroupBy

### Community 65 - "Data"
Cohesion: 1.0
Nodes (1): Data Cleaning

### Community 66 - "Visualization"
Cohesion: 1.0
Nodes (1): Data Visualization

### Community 67 - "Neural"
Cohesion: 1.0
Nodes (1): Neural Networks

### Community 68 - "Algorithm"
Cohesion: 1.0
Nodes (1): Algorithm

### Community 69 - "Predictive"
Cohesion: 1.0
Nodes (1): Predictive Analytics

### Community 70 - "Automation"
Cohesion: 1.0
Nodes (1): Automation

### Community 71 - "Chatbot"
Cohesion: 1.0
Nodes (1): Chatbot

### Community 72 - "Predictive"
Cohesion: 1.0
Nodes (1): Predictive Maintenance

### Community 73 - "Recruitment"
Cohesion: 1.0
Nodes (1): Recruitment Automation

### Community 74 - "Sustainability"
Cohesion: 1.0
Nodes (1): Sustainability

### Community 75 - "Disaster"
Cohesion: 1.0
Nodes (1): Disaster Preparedness

### Community 76 - "Media"
Cohesion: 1.0
Nodes (1): Media Literacy

### Community 77 - "Career"
Cohesion: 1.0
Nodes (1): Career Skills

### Community 78 - "Ai"
Cohesion: 1.0
Nodes (1): AI in Data Engineering

### Community 79 - "Data"
Cohesion: 1.0
Nodes (1): Data Engineering

## Knowledge Gaps
- **65 isolated node(s):** `RAG Study Assistant`, `Yousef Ahmed AI CV (ATS)`, `AI in Data Engineering (SB)`, `Physics Student Guide (doc-2)`, `E-Commerce Sales Mini Project` (+60 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Inspect`** (1 nodes): `inspect_chunks.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Readme`** (1 nodes): `RAG Study Assistant`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Requirements`** (1 nodes): `requirements.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Yousef`** (1 nodes): `Yousef Ahmed AI CV (ATS)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ai`** (1 nodes): `AI in Data Engineering (SB)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Doc`** (1 nodes): `Physics Student Guide (doc-2)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ecommerce`** (1 nodes): `E-Commerce Sales Mini Project`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `English`** (1 nodes): `English AI (SB)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Arabic`** (1 nodes): `Exam Vocabulary Guide`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ollama`** (1 nodes): `Ollama`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Gemini`** (1 nodes): `Gemini`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `E5`** (1 nodes): `intfloat/multilingual-e5-base`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Bge`** (1 nodes): `BAAI/bge-reranker-v2-m3`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Rag`** (1 nodes): `Retrieval-Augmented Generation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Dense`** (1 nodes): `Dense Retrieval`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Reranking`** (1 nodes): `Reranking`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Yousef`** (1 nodes): `Yousef Ahmed`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Distilbert`** (1 nodes): `DistilBERT`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pytorch`** (1 nodes): `PyTorch`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tensorflow`** (1 nodes): `TensorFlow`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Keras`** (1 nodes): `Keras`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Machine`** (1 nodes): `Machine Learning`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deep`** (1 nodes): `Deep Learning`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Nlp`** (1 nodes): `Natural Language Processing`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Computer`** (1 nodes): `Computer Vision`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Banking77`** (1 nodes): `Banking77 Intent Classification`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ag`** (1 nodes): `AG News Text Classification`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Imdb`** (1 nodes): `IMDB Sentiment Analysis`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Netflix`** (1 nodes): `Netflix User Churn Prediction`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Force`** (1 nodes): `Force`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Gravitational`** (1 nodes): `Gravitational Force`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Centripetal`** (1 nodes): `Centripetal Force`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Friction`** (1 nodes): `Friction Force`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Viscosity`** (1 nodes): `Viscosity`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Lubrication`** (1 nodes): `Lubrication`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Heat`** (1 nodes): `Heat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Temperature`** (1 nodes): `Temperature`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Thermal`** (1 nodes): `Thermal Energy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Heat`** (1 nodes): `Heat Transfer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Conduction`** (1 nodes): `Conduction`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Convection`** (1 nodes): `Convection`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Radiation`** (1 nodes): `Radiation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Specific`** (1 nodes): `Specific Heat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Heat`** (1 nodes): `Heat Capacity`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Thermometer`** (1 nodes): `Thermometer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pandas`** (1 nodes): `Pandas`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Matplotlib`** (1 nodes): `Matplotlib`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Seaborn`** (1 nodes): `Seaborn`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Groupby`** (1 nodes): `GroupBy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data`** (1 nodes): `Data Cleaning`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Visualization`** (1 nodes): `Data Visualization`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Neural`** (1 nodes): `Neural Networks`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Algorithm`** (1 nodes): `Algorithm`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Predictive`** (1 nodes): `Predictive Analytics`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Automation`** (1 nodes): `Automation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Chatbot`** (1 nodes): `Chatbot`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Predictive`** (1 nodes): `Predictive Maintenance`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Recruitment`** (1 nodes): `Recruitment Automation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sustainability`** (1 nodes): `Sustainability`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Disaster`** (1 nodes): `Disaster Preparedness`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Media`** (1 nodes): `Media Literacy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Career`** (1 nodes): `Career Skills`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ai`** (1 nodes): `AI in Data Engineering`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data`** (1 nodes): `Data Engineering`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GeminiGenerator` connect `Evaluate` to `Pipeline`, `Test`, `Vector`, `Ingestion`, `Chunking`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Why does `RAGPipeline` connect `Pipeline` to `Test`, `Evaluate`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `EmbeddingModel` connect `Test` to `Evaluate`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `RAGPipeline` (e.g. with `GeminiGenerator` and `Reranker`) actually correct?**
  _`RAGPipeline` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `build_pipeline()` (e.g. with `FakeEmbeddingModel` and `FakeVectorStore`) actually correct?**
  _`build_pipeline()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `GeminiGenerator` (e.g. with `main()` and `Chunker`) actually correct?**
  _`GeminiGenerator` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `main()` (e.g. with `load_env_file()` and `load_json()`) actually correct?**
  _`main()` has 6 INFERRED edges - model-reasoned connections that need verification._