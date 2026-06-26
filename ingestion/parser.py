from bs4 import BeautifulSoup
from pathlib import Path
import re

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

ITEM_PATTERN = re.compile(
    r"^\s*(Item\s+\d+[A-C]?)\.",
    re.MULTILINE | re.IGNORECASE
)

MIN_SECTION_LENGTH = 200


def _item_sort_key(label: str) -> tuple:
    match = re.match(r"Item (\d+)([A-C])?", label)
    return (int(match.group(1)), match.group(2) or "")


def parse_10k(html_path: Path) -> dict:
    """
    Parse a 10-K HTML file into clean sections keyed by standardized Item number.
    Works across all companies — uses regex on plain text rather than style-based detection.
    """
    with open(html_path, "rb") as f:
        soup = BeautifulSoup(f, "html.parser")

    for tag in soup(["script", "style", "meta", "noscript"]):
        tag.decompose()

    plain_text = soup.get_text(separator="\n", strip=True)
    plain_text = re.sub(r"\xa0", " ", plain_text)
    plain_text = re.sub(r"\n{3,}", "\n\n", plain_text)

    # Find all "Item X." occurrences, capturing only the item number
    all_matches = []
    for match in ITEM_PATTERN.finditer(plain_text):
        raw_label = match.group(1).strip()
        normalized = re.sub(r"\s+", " ", raw_label).title()
        all_matches.append({"label": normalized, "position": match.start()})

    # Filter out TOC entries — real headers have substantial content before the next match
    real_headers = []
    for i, m in enumerate(all_matches):
        next_pos = all_matches[i + 1]["position"] if i + 1 < len(all_matches) else len(plain_text)
        if next_pos - m["position"] > MIN_SECTION_LENGTH:
            real_headers.append(m)

    # Extract content and deduplicate (keep the occurrence with the most content)
    sections = {}
    for i, h in enumerate(real_headers):
        start = h["position"]
        end = real_headers[i + 1]["position"] if i + 1 < len(real_headers) else len(plain_text)

        content = plain_text[start:end]
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline:].strip()

        label = h["label"]
        if label not in sections or len(content) > len(sections[label]):
            sections[label] = content

    return dict(sorted(sections.items(), key=lambda x: _item_sort_key(x[0])))


def save_sections(company: str, sections: dict):
    """
    Save each section as a separate .txt file under data/processed/<company>/
    """
    company_path = PROCESSED_DATA_PATH / company
    company_path.mkdir(parents=True, exist_ok=True)

    for section_name, text in sections.items():
        clean_name = section_name.replace(" ", "_")
        filepath = company_path / f"{clean_name}.txt"
        filepath.write_text(text, encoding="utf-8")
        print(f"  Saved: {filepath} ({len(text)} chars)")


def parse_all():
    for company_dir in RAW_DATA_PATH.iterdir():
        if not company_dir.is_dir():
            continue

        company = company_dir.name
        print(f"\nParsing {company.upper()}...")

        for html_file in company_dir.glob("*.html"):
            sections = parse_10k(html_file)
            print(f"  Found {len(sections)} sections")
            save_sections(company, sections)


if __name__ == "__main__":
    parse_all()