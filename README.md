# SEC Filing RAG

A multi-document RAG system for querying and comparing SEC 10-K filings 
across multiple companies using natural language. Built with hybrid search, 
citation-grounded generation, and a quantitative evaluation pipeline.

---

## Overview

Annual 10-K filings contain rich, unstructured narrative content that is 
difficult to analyze at scale — particularly across multiple companies. 
This system enables analysts to ask natural language questions that require 
synthesizing information across filings, such as:

- "How do Apple and Microsoft describe their AI strategies differently?"
- "Which company expressed the most concern about tariffs?"
- "Compare risk factors across all five companies"

---

## Setup

```bash
# Clone the repo
git clone https://github.com/ryan-kuntz/sec-filing-rag.git
cd sec-filing-rag

# Create and activate virtual environment
python -m venv venv

# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key to .env
echo "GEMINI_API_KEY=your_key_here" > .env

# Start Qdrant (requires Docker)
docker run -p 6333:6333 qdrant/qdrant

# Run the app
PYTHONPATH=. streamlit run app/app.py
```

> **Note:** You'll need to reproduce the data before running the app. See [Reproducing the Data](#reproducing-the-data) below.

---

## Reproducing the Data

All data is sourced from the free SEC EDGAR API and can be fully regenerated 
from scratch. Data files are excluded from version control.

1. Start Qdrant in a separate terminal:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. Run the ingestion pipeline in order:
```bash
PYTHONPATH=. python ingestion/scraper.py
PYTHONPATH=. python ingestion/parser.py
PYTHONPATH=. python ingestion/xbrl_parser.py
PYTHONPATH=. python ingestion/chunker.py
PYTHONPATH=. python ingestion/embedder.py
PYTHONPATH=. python retrieval/vector_store.py
```

> **Note:** The embedder will download the `BAAI/bge-m3` model (~2GB) on first run. 
> Subsequent runs use the cached model and are significantly faster.

---

## Architecture

### Ingestion Pipeline

**`ingestion/scraper.py`**
Downloads 10-K filings directly from the SEC EDGAR API for a configurable 
list of companies. Uses each company's CIK number to retrieve their most 
recent annual filing. No API key required — EDGAR is a free public resource.

**`ingestion/parser.py`**
Extracts narrative text sections from the raw HTML filings. Strips HTML to 
plain text and uses regex to identify section headers (`Item 1.`, `Item 1A.`, 
etc.) — the standardized numbering required by the SEC. A content-length 
heuristic filters out table-of-contents entries, and duplicate sections are 
resolved by keeping the longer occurrence. Works across all companies without 
per-company tuning.

**`ingestion/xbrl_parser.py`**
Extracts structured financial data from inline XBRL tags (`ix:nonFraction`) 
embedded in the filing HTML. Pulls a curated set of ~25 key metrics (revenue, 
net income, EPS, assets, cash flow, etc.) with their periods and segment 
breakdowns, then converts them into natural language sentences for embedding 
alongside narrative text.

**`ingestion/chunker.py`**
Implements section-aware chunking — respecting section boundaries while 
further splitting large sections into overlapping chunks of ~500 words with 
50-word overlap. Each chunk carries metadata including company name, section, 
and chunk index, enabling precise citation in generated answers.

**`ingestion/embedder.py`**
Generates dense vector embeddings for all chunks using `BAAI/bge-m3`, a 
state-of-the-art open source embedding model with 1024 dimensions. Embeddings 
are normalized for cosine similarity and saved locally.

### Retrieval Pipeline

**`retrieval/vector_store.py`**
Loads chunk embeddings into a Qdrant vector database running locally via 
Docker. Creates a `sec_filings` collection with cosine similarity configured 
for dense retrieval.

**`retrieval/hybrid_search.py`**
Implements hybrid search combining:
- **Dense retrieval** — semantic search via bge-m3 embeddings in Qdrant
- **Sparse retrieval** — keyword search via BM25 (rank-bm25)
- **Reciprocal Rank Fusion (RRF)** — combines both result sets into a single 
  ranked list, improving recall over either method alone

### Generation Pipeline

**`generation/prompts.py`**
Contains the prompt template for the RAG system. Each prompt includes 
retrieved chunks as numbered sources with company and section metadata, 
and instructs the model to cite every claim using source numbers. Keeping 
prompts in a dedicated file makes iteration and experimentation easier 
without touching the core generation logic.

**`generation/synthesizer.py`**
Takes retrieved chunks and generates a cited answer using Google Gemini 2.5 
Flash. Structures the response with source attribution so every claim traces 
back to a specific company and section of a specific 10-K filing.

### Evaluation Pipeline

**`evaluation/test_set.csv`**
A hand-curated set of 10 question/answer pairs covering factual and analytical 
question types across key 10-K sections. Expected answers were derived directly 
from the processed filing text to ensure ground truth accuracy.

**`evaluation/run_evals.py`**
Runs each test question through the full RAG pipeline and evaluates generated 
answers against expected answers using semantic similarity via bge-m3 embeddings. 
Results are saved as timestamped JSON files in `evaluation/results/` with per-question 
scores and a summary breakdown by question type. A similarity threshold of 0.75 
determines pass/fail for each question.

### Frontend

**`app/app.py`**
A Streamlit web application providing a natural language interface for querying 
the RAG system. Features include a sidebar with example questions, expandable 
source citations showing the exact retrieved text, and cached model loading 
for fast repeated queries.

---

## Known Limitations & Future Improvements

- **Gemini free tier rate limits:** The current setup uses Gemini 2.5 Flash on 
  the free tier, which is limited to 5 requests per minute and 25 requests per 
  day. This makes running large eval sets slow and requires rate limiting logic 
  in the eval runner. Upgrading to the paid tier or switching to OpenAI 
  GPT-4o-mini would remove these constraints at minimal cost.


---

## Tech Stack

- **Embeddings:** BAAI/bge-m3 (sentence-transformers)
- **Vector DB:** Qdrant
- **Sparse Search:** BM25 (rank-bm25)
- **HTML Parsing:** BeautifulSoup4
- **LLM:** Google Gemini 2.5 Flash
- **Frontend:** Streamlit
- **Data Source:** SEC EDGAR API (free, no authentication required)