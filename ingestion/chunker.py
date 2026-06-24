# Section-aware chunking

# Respect section boundaries when chunking text, but further split large sections into smaller overlapping chunks. Each chunk knows which section and company it came from.

from pathlib import Path
import json
import re

PROCESSED_DATA_PATH = Path("data/processed")
CHUNKS_PATH = Path("data/chunks")
CHUNKS_PATH.mkdir(parents=True, exist_ok=True)

# Tunable parameters — worth mentioning in your README
CHUNK_SIZE = 500        # target words per chunk
CHUNK_OVERLAP = 50      # words of overlap between chunks


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # step forward with overlap

    return chunks


def chunk_company(company: str) -> list[dict]:
    """
    Chunk all sections for a company into metadata-rich chunk objects.
    """
    company_path = PROCESSED_DATA_PATH / company
    all_chunks = []

    for section_file in sorted(company_path.glob("*.txt")):
        section_name = section_file.stem  # e.g. "Item_1A_Risk_Factors"
        text = section_file.read_text(encoding="utf-8")

        if not text.strip():
            continue

        chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{company}_{section_name}_{i}",
                "company": company,
                "section": section_name,
                "chunk_index": i,
                "total_chunks_in_section": len(chunks),
                "text": chunk_text,
                "word_count": len(chunk_text.split())
            })

    return all_chunks


def save_chunks(company: str, chunks: list[dict]):
    """
    Save chunks as a JSON file under data/chunks/<company>.json
    """
    output_path = CHUNKS_PATH / f"{company}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(chunks)} chunks to {output_path}")


def chunk_all():
    for company_dir in PROCESSED_DATA_PATH.iterdir():
        if not company_dir.is_dir():
            continue

        company = company_dir.name
        print(f"\nChunking {company.upper()}...")
        chunks = chunk_company(company)
        save_chunks(company, chunks)
        
        # Print summary stats
        sections = set(c["section"] for c in chunks)
        print(f"  Sections: {len(sections)}")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Avg words per chunk: {sum(c['word_count'] for c in chunks) // len(chunks)}")


if __name__ == "__main__":
    chunk_all()