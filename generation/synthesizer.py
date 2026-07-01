from time import time

from generation.prompts import build_prompt
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import requests

load_dotenv()

client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def _generate_gemini(prompt: str) -> str:
    response = client_genai.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text


def _generate_ollama(prompt: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]


def synthesize(query: str, chunks: list[dict], use_local: bool = False) -> dict:
    """
    Generate an answer from retrieved chunks.
    use_local=True routes to Ollama (llama3.1:8b) instead of Gemini —
    free and quota-free, intended for eval iteration runs.
    """
    if not chunks:
        return {
            "answer": "No relevant sources found for this query.",
            "sources": []
        }

    prompt = build_prompt(query, chunks)

    if use_local:
        answer = _generate_ollama(prompt)
    else:
        answer = _generate_gemini(prompt)

    # Build source list for attribution
    sources = [
        {
            "source_num": i + 1,
            "company": chunk["company"],
            "section": chunk["section"],
            "chunk_id": chunk["chunk_id"]
        }
        for i, chunk in enumerate(chunks)
    ]

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from rank_bm25 import BM25Okapi
    from retrieval.hybrid_search import hybrid_search, load_all_chunks, build_bm25_index

    print("Loading retrieval components...")
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("BAAI/bge-m3")
    chunks = load_all_chunks()
    bm25 = build_bm25_index(chunks)

    query = "What are Apple's biggest risk factors related to regulation?"
    print(f"\nQuery: {query}\n")

    retrieved = hybrid_search(client, model, bm25, chunks, query)
    result = synthesize(query, retrieved)
    time.sleep(15)

    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES USED:")
    for s in result["sources"]:
        print(f"  [{s['source_num']}] {s['company'].upper()} | {s['section']}")