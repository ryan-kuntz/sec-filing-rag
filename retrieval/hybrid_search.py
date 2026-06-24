# This is where dense semantic search and BM25 sparse search are combined.

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import json
from pathlib import Path

COLLECTION_NAME = "sec_filings"
MODEL_NAME = "BAAI/bge-m3"
EMBEDDINGS_PATH = Path("data/embeddings")
TOP_K = 5  # number of results to return per query


def get_client() -> QdrantClient:
    return QdrantClient(host="localhost", port=6333)


def load_all_chunks() -> list[dict]:
    """
    Load all chunks from all companies for BM25 index.
    """
    all_chunks = []
    for path in EMBEDDINGS_PATH.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    return all_chunks


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Build a BM25 index from chunk texts.
    """
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def dense_search(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    company_filter: str = None,
    top_k: int = TOP_K
) -> list[dict]:
    """
    Semantic search using bge-m3 embeddings.
    """
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    search_filter = None
    if company_filter:
        search_filter = Filter(
            must=[FieldCondition(
                key="company",
                match=MatchValue(value=company_filter)
            )]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True
    ).points

    return [
        {
            "chunk_id": r.payload["chunk_id"],
            "company": r.payload["company"],
            "section": r.payload["section"],
            "text": r.payload["text"],
            "score": r.score,
            "source": "dense"
        }
        for r in results
    ]


def sparse_search(
    bm25: BM25Okapi,
    chunks: list[dict],
    query: str,
    top_k: int = TOP_K
) -> list[dict]:
    """
    Keyword search using BM25.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "chunk_id": chunks[i]["chunk_id"],
            "company": chunks[i]["company"],
            "section": chunks[i]["section"],
            "text": chunks[i]["text"],
            "score": scores[i],
            "source": "sparse"
        }
        for i in top_indices
    ]


def hybrid_search(
    client: QdrantClient,
    model: SentenceTransformer,
    bm25: BM25Okapi,
    chunks: list[dict],
    query: str,
    top_k: int = TOP_K
) -> list[dict]:
    """
    Combine dense and sparse results using Reciprocal Rank Fusion (RRF).
    """
    dense_results = dense_search(client, model, query, top_k=top_k * 2)
    sparse_results = sparse_search(bm25, chunks, query, top_k=top_k * 2)

    # Reciprocal Rank Fusion
    rrf_scores = {}
    k = 60  # RRF constant

    for rank, result in enumerate(dense_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, {"score": 0, "data": result})
        rrf_scores[cid]["score"] += 1 / (k + rank + 1)

    for rank, result in enumerate(sparse_results):
        cid = result["chunk_id"]
        if cid not in rrf_scores:
            rrf_scores[cid] = {"score": 0, "data": result}
        rrf_scores[cid]["score"] += 1 / (k + rank + 1)

    # Sort by RRF score
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    return [r["data"] for r in sorted_results[:top_k]]


if __name__ == "__main__":
    print("Loading model and index...")
    client = get_client()
    model = SentenceTransformer(MODEL_NAME)
    chunks = load_all_chunks()
    bm25 = build_bm25_index(chunks)

    # Test query
    query = "What are Apple's biggest risk factors?"
    print(f"\nQuery: {query}\n")

    results = hybrid_search(client, model, bm25, chunks, query)

    for i, r in enumerate(results):
        print(f"Result {i+1} [{r['company']} | {r['section']}]")
        print(f"  {r['text'][:200]}...")
        print()