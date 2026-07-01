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
list of companies. Uses each company's CIK number to retrieve the `NUM_FILINGS` 
(default 3) most recent annual filings per company, giving multi-year XBRL 
trend data and several narrative snapshots for comparison. Each filing is saved 
under `data/raw/<company>/<fiscal_year>/`, alongside a `filing_metadata.json` 
sidecar (accession number, filing date, report date, source URL). Already-downloaded 
years are skipped on rerun. No API key required — EDGAR is a free public resource.

**`ingestion/parser.py`**
Extracts narrative text sections from the raw HTML filings. Strips HTML to 
plain text and uses regex to identify section headers (`Item 1.`, `Item 1A.`, 
etc.) — the standardized numbering required by the SEC. A content-length 
heuristic filters out table-of-contents entries, and duplicate sections are 
resolved by keeping the longer occurrence. Only `HIGH_VALUE_SECTIONS` (Business, 
Risk Factors, Cybersecurity, Legal Proceedings, MD&A, Market Risk, Financial 
Statements) are saved — boilerplate sections like Properties or Mine Safety are 
dropped to reduce retrieval noise. Output mirrors the raw data's 
`<company>/<fiscal_year>/` structure. Works across all companies without 
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
50-word overlap. XBRL financial sentences are the exception: since each line 
is already an atomic, self-contained fact, they're chunked one sentence per 
chunk rather than grouped into 500-word blocks. Each chunk carries metadata 
including company name, fiscal year, section, chunk index, and filing 
provenance (accession number, filing/report date, source URL), enabling 
precise citation in generated answers.

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
- **Dense retrieval** — semantic search via bge-m3 embeddings in Qdrant, supports
  an optional `company_filter` to scope results to a single company's chunks
- **Sparse retrieval** — keyword search via BM25 (rank-bm25)
- **Reciprocal Rank Fusion (RRF)** — combines both result sets into a single 
  ranked list, improving recall over either method alone

Note: `company_filter` only applies to dense search — BM25 scores all chunks
regardless of company. See Known Limitations.

### Generation Pipeline

**`generation/prompts.py`**
Contains the prompt template for the RAG system. Each prompt includes 
retrieved chunks as numbered sources with company and section metadata, 
and instructs the model to cite every claim using source numbers. Keeping 
prompts in a dedicated file makes iteration and experimentation easier 
without touching the core generation logic.

**`generation/synthesizer.py`**
Takes retrieved chunks and generates a cited answer. Supports two backends 
via a `use_local` flag — Gemini 2.5 Flash (default, production quality) and 
Ollama/llama3.1:8b (local, free, no quota). The local backend is intended 
for eval iteration runs where Gemini's 25/day quota would otherwise block 
repeated testing. Structures the response with source attribution so every 
claim traces back to a specific company and section of a specific 10-K filing.

### Evaluation Pipeline

**`evaluation/test_set.csv`**
A hand-curated set of 25 question/answer pairs covering factual and analytical
question types across Apple, Microsoft, Google, and cross-company comparative
questions. Columns: `question`, `expected_answer`, `question_type`,
`section_hint` (which 10-K section contains the answer), and `company_hint`
(which company to filter retrieval to — blank for cross-company questions).
Expected answers are derived directly from processed filing text.

**`evaluation/run_evals.py`**
Runs each test question through the full RAG pipeline (retrieval + generation) 
and evaluates generated answers against expected answers using two independent 
scoring signals — both free and local, no additional API cost:
- **Semantic similarity** (bge-m3 cosine similarity ≥ 0.75) — catches meaning/phrasing drift
- **Numeric match** (regex extraction with 1% tolerance) — catches factually wrong figures 
  that cosine similarity alone would miss (e.g. "$391B" vs "$132.4B" with similar phrasing)

A question only passes if both checks pass. Supports `--local` flag to route generation 
through Ollama/llama3.1:8b instead of Gemini — use this for frequent iteration runs to 
avoid burning Gemini quota. Results are saved as timestamped JSON files in 
`evaluation/results/`. This is the **Layer 2** (generation quality) eval — see Notes below.

```bash
python -m evaluation.run_evals          # Gemini (production model, use sparingly)
python -m evaluation.run_evals --local  # Ollama (free, unlimited, for iteration)
```

**`evaluation/eval_retrieval.py`**
Evaluates retrieval in isolation — no generation call, no API cost. For each 
test question, runs `hybrid_search` and checks whether the chunks retrieved 
actually come from the section named in `section_hint`. Reports hit rate@K, 
precision@K, and MRR (mean reciprocal rank), saved to `evaluation/results/`. 
This is the **Layer 1** (retrieval quality) eval, intended to isolate retrieval 
or chunking changes from generation-model variance — see Notes below.

### Frontend

**`app/app.py`**
A Streamlit web application providing a natural language interface for querying
the RAG system. Features include:
- **Company filter** — sidebar dropdown to scope retrieval to a single company
  (Apple, Microsoft, Google) or search across all three. Defaults to all companies;
  set to a specific company for focused single-company research and switch back to
  all companies for cross-company comparative questions.
- Expandable source citations showing the exact retrieved text
- Cached model loading for fast repeated queries

---

## Known Limitations & Future Improvements

- **Gemini free tier rate limits:** The current setup uses Gemini 2.5 Flash on 
  the free tier, which is limited to 5 requests per minute and 25 requests per 
  day. This is mitigated by routing eval iteration runs through a local Ollama 
  model (`--local` flag), reserving Gemini calls for final validation passes only.

- **Similarity threshold is calibrated for Gemini, not Ollama:** The 0.75 cosine 
  similarity threshold was implicitly set for Gemini's output style. Ollama answers 
  tend to be more verbose and phrased differently from expected answers, producing 
  lower similarity scores even when the meaning is correct — baseline Ollama pass 
  rate on the 10-question set was 30% vs an expected ~80%+ with Gemini. Treat 
  Ollama eval results as relative signals for comparing strategy A vs B, not as 
  absolute quality scores.

- **BM25 is not filtered by company:** The `company_filter` in `hybrid_search.py`
  only applies to dense (Qdrant) search. BM25 scores against all chunks regardless
  of company, meaning other companies' chunks can still surface in hybrid results
  even when a filter is active. A future improvement would pre-filter the BM25
  candidate pool to the selected company before scoring.

- **Short, definitional facts can lose to numerically-dense chunks:** Baseline 
  retrieval eval (`eval_retrieval.py`) found one consistent miss — "When does 
  Apple's fiscal year end?" retrieves `Item_8`/XBRL chunks instead of `Item_1`. 
  The correct answer is a single short sentence in Item 1, while Item_8 and the 
  XBRL chunks are saturated with literal fiscal-year-end dates repeated across 
  dozens of facts, pulling both dense and sparse search away from the actual 
  definitional sentence. Worth re-checking this specific question after any 
  chunking or retrieval change.


---

## Tech Stack

- **Embeddings:** BAAI/bge-m3 (sentence-transformers)
- **Vector DB:** Qdrant
- **Sparse Search:** BM25 (rank-bm25)
- **HTML Parsing:** BeautifulSoup4
- **LLM:** Google Gemini 2.5 Flash
- **Frontend:** Streamlit
- **Data Source:** SEC EDGAR API (free, no authentication required)


## Notes: 

When evaluating RAG effectiveness, there are two layers to consider

- Layer 1 — Retrieval quality (no LLM calls needed, completely free)
- Layer 2 — Generation quality (the expensive layer, needs an LLM)

- For comparing retrieval/chunking strategies, the choice of generation LLM used during iteration is mostly a free variable you can optimize for cost (use the free local model for the bulk of testing) — but it's worth one final confirmation pass with the real production model before you declare a winner, since model strength can change how much a retrieval flaw actually matters. The embedding model is the opposite case: it's part of retrieval itself, not a stand-in you can swap freely, so if you ever change it, that's a Layer 1 experiment in its own right, not a constant to hold while testing something else.