from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import json
from pathlib import Path

EMBEDDINGS_PATH = Path("data/embeddings")
COLLECTION_NAME = "sec_filings"
VECTOR_SIZE = 1024  # bge-m3 dimension


def get_client() -> QdrantClient:
    return QdrantClient(host="localhost", port=6333)


def create_collection(client: QdrantClient):
    """
    Create Qdrant collection if it doesn't already exist.
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        print(f"  Collection '{COLLECTION_NAME}' already exists, skipping creation.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"  Created collection '{COLLECTION_NAME}'")


def load_embeddings(company: str) -> list[dict]:
    path = EMBEDDINGS_PATH / f"{company}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def upsert_embeddings(client: QdrantClient, chunks: list[dict]):
    """
    Upload embeddings to Qdrant with metadata as payload.
    """
    points = []
    for i, chunk in enumerate(chunks):
        points.append(PointStruct(
            id=i,
            vector=chunk["embedding"],
            payload={
                "chunk_id": chunk["chunk_id"],
                "company": chunk["company"],
                "fiscal_year": chunk.get("fiscal_year"),
                "section": chunk["section"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "accession_number": chunk.get("accession_number"),
                "filing_date": chunk.get("filing_date"),
                "report_date": chunk.get("report_date"),
                "source_url": chunk.get("source_url"),
            }
        ))

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"  Upserted {len(points)} points")


def build_vector_store():
    client = get_client()
    create_collection(client)

    for embedding_file in EMBEDDINGS_PATH.glob("*.json"):
        company = embedding_file.stem
        print(f"\nUploading {company.upper()}...")
        chunks = load_embeddings(company)
        upsert_embeddings(client, chunks)

    print("\nVector store ready.")


if __name__ == "__main__":
    build_vector_store()