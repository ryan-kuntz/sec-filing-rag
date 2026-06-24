from time import time

from generation.prompts import build_prompt
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"


def synthesize(query: str, chunks: list[dict]) -> dict:
    """
    Generate an answer from retrieved chunks using Gemini.
    Returns the answer text and the sources used.
    """
    if not chunks:
        return {
            "answer": "No relevant sources found for this query.",
            "sources": []
        }

    prompt = build_prompt(query, chunks)

    response = client_genai.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    answer = response.text

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