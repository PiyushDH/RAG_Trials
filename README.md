RAG Trials
==========
Experimentation ground for building a Retrieval-Augmented Generation (RAG) pipeline that will power a future portfolio website with a chat interface. The notebooks here walk through loading documents, chunking them, embedding with SentenceTransformers, and storing/retrieving vectors from Chroma.

Project goals
- Prototype a small-but-complete RAG pipeline before shipping it in a production portfolio/chatbot.
- Keep experiments reproducible: notebooks show each step from ingestion to retrieval.
- Capture learnings about document chunking, metadata, and vector store persistence for later reuse.

Repository layout
- `notebook/pdf_loader.ipynb`: main RAG walkthrough using PDFs, PyMuPDF loading, `RecursiveCharacterTextSplitter`, SentenceTransformer (`all-MiniLM-L6-v2`), persistent Chroma vector store, and a basic `RAGRetriever` for similarity search.
- `notebook/document.ipynb`: introductory ingestion/embedding experiments with text/PDF loaders and vector store basics.
- `notebook/UnderstandingExercise.ipynb`: duplicate/backup of the ingestion + vector store exercise for quick reference.
- `data/pdfs/`: sample corpus (portfolio chatbot guide and resume PDFs).
- `data/text_files/`: toy text files used in early ingestion exercises.
- `data/vector_store/`: persistent Chroma collection produced by the notebooks.
- `main.py`: placeholder CLI entry point.

Current capabilities
- Load PDFs with PyMuPDF for layout-aware text extraction.
- Chunk text with `RecursiveCharacterTextSplitter` (size ~1000, overlap ~200 in the notebook).
- Embed chunks with `all-MiniLM-L6-v2` via `sentence-transformers`.
- Persist vectors to Chroma on disk (`data/vector_store`) and run similarity search through the `RAGRetriever` helper class.
- Inspect retrieved chunks with metadata to see what matched.

Planned next steps (for the future portfolio chatbot)
- Add a generation step (LLM response synthesis with citations) on top of retrieved chunks.
- Improve metadata (page numbers, chunk ids, source paths) and deterministic IDs/upserts to avoid duplication when rebuilding.
- Explore reranking/hybrid search (e.g., BM25 + dense + reranker) and evaluation with a small Q/A set.

Getting started
1) Prerequisites
   - Python 3.13+ (matches `pyproject.toml`)
   - Recommended: virtual environment (`python -m venv .venv`) or `uv venv`

2) Install dependencies (pick one)
- Python + pip
```
python -m venv .venv
.\.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

- uv (faster installs; works with `requirements.txt` or `pyproject.toml`)
```
python -m pip install uv              # once, if uv is not installed
uv init                               # create pyproject + lock if missing
uv python pin 3.13.2                  # align to project Python version
uv add -r requirements.txt            # resolves and records deps in lockfile
uv venv                               # creates .venv
.\.venv\Scripts\activate
uv sync                               # installs from lock into the venv
```

3) Environment variables
- Create a `.env` file with your keys (do not commit secrets):
```
OPENAI_API_KEY=your-key-here
```

Running the notebooks
1) Launch Jupyter or VS Code notebooks with the virtualenv active.
2) Open `notebook/pdf_loader.ipynb` and run cells in order:
   - Load PDFs from `data/pdfs`.
   - Chunk the text.
   - Generate embeddings and persist to Chroma (`data/vector_store`).
   - Use the `RAGRetriever` class cell to issue similarity queries and inspect matches.
3) `notebook/document.ipynb` and `notebook/UnderstandingExercise.ipynb` cover the same concepts with lighter text examples if you want a simpler starting point.

Minimal retrieval example (after running the pdf_loader notebook once)
```python
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

# Load the persistent store
db = Chroma(
    collection_name="pdf_documents",
    persist_directory="data/vector_store",
    embedding_function=SentenceTransformer("all-MiniLM-L6-v2").encode,
)

query = "What are the advantages of Next.js for a portfolio chatbot?"
results = db.similarity_search_with_score(query, k=3)
for doc, score in results:
    print(f"score={score:.4f} | source={doc.metadata.get('source_file')} | preview={doc.page_content[:120]}...")
```

Notes and tips
- Regenerate the vector store after changing chunk sizes or adding documents to keep embeddings in sync.
- If you rerun ingestion, clear or overwrite `data/vector_store` to avoid duplicate entries until deterministic IDs/upserts are added.
- Keep PDF/text sources small while experimenting; bigger corpora will slow local embedding and increase storage.

License
- MIT License
