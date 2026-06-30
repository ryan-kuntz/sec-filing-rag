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


def load_filing_metadata(year_path: Path) -> dict:
    """
    Load the filing_metadata.json sidecar for a company/fiscal_year folder, if present.
    """
    metadata_file = year_path / "filing_metadata.json"
    if not metadata_file.exists():
        return {}

    with open(metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_filing(company: str, fiscal_year: str) -> list[dict]:
    """
    Chunk all sections for one company's filing (a single fiscal year)
    into metadata-rich chunk objects.
    """
    year_path = PROCESSED_DATA_PATH / company / fiscal_year
    filing_metadata = load_filing_metadata(year_path)
    all_chunks = []

    for section_file in sorted(year_path.glob("*.txt")):
        section_name = section_file.stem  # e.g. "Item_1A"
        text = section_file.read_text(encoding="utf-8")

        if not text.strip():
            continue

        # XBRL sentences are atomic self-contained facts — chunk one per line
        # rather than grouping into 500-word blocks like narrative text.
        if section_name == "xbrl_financials":
            chunks = [line for line in text.splitlines() if line.strip()]
        else:
            chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{company}_{fiscal_year}_{section_name}_{i}",
                "company": company,
                "fiscal_year": fiscal_year,
                "section": section_name,
                "chunk_index": i,
                "total_chunks_in_section": len(chunks),
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "accession_number": filing_metadata.get("accession_number"),
                "filing_date": filing_metadata.get("filing_date"),
                "report_date": filing_metadata.get("report_date"),
                "source_url": filing_metadata.get("source_url"),
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

        all_chunks = []
        for year_dir in sorted(company_dir.iterdir()):
            if not year_dir.is_dir():
                continue

            fiscal_year = year_dir.name
            all_chunks.extend(chunk_filing(company, fiscal_year))

        save_chunks(company, all_chunks)

        # Print summary stats
        sections = set(c["section"] for c in all_chunks)
        print(f"  Sections: {len(sections)}")
        print(f"  Total chunks: {len(all_chunks)}")
        if all_chunks:
            print(f"  Avg words per chunk: {sum(c['word_count'] for c in all_chunks) // len(all_chunks)}")


if __name__ == "__main__":
    chunk_all()