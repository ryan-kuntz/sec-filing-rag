import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

CHUNKS_PATH = Path("data/chunks")
EMBEDDINGS_PATH = Path("data/embeddings")
EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-m3"


def load_chunks(company: str) -> list[dict]:
    path = CHUNKS_PATH / f"{company}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Generate embeddings for all chunks.
    """
    texts = [chunk["text"] for chunk in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True  # important for cosine similarity
    )
    return embeddings


def save_embeddings(company: str, chunks: list[dict], embeddings: np.ndarray):
    """
    Save embeddings and chunks together as a single file.
    """
    output = []
    for chunk, embedding in zip(chunks, embeddings):
        output.append({
            **chunk,
            "embedding": embedding.tolist()
        })

    output_path = EMBEDDINGS_PATH / f"{company}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False) # no indent — embedding files get large

    print(f"  Saved {len(output)} embeddings to {output_path}")


def embed_all():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    for chunk_file in CHUNKS_PATH.glob("*.json"):
        company = chunk_file.stem
        print(f"\nProcessing {company.upper()}...")
        chunks = load_chunks(company)
        embeddings = embed_chunks(chunks, model)
        save_embeddings(company, chunks, embeddings)

    print("\nDone.")


if __name__ == "__main__":
    embed_all()