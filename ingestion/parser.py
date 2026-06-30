from bs4 import BeautifulSoup
from pathlib import Path
import re
import shutil

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

ITEM_PATTERN = re.compile(
    r"^\s*(Item\s+\d+[A-C]?)\.",
    re.MULTILINE | re.IGNORECASE
)

MIN_SECTION_LENGTH = 200

# Sections worth chunking and embedding for financial research.
# Excludes boilerplate/stub sections (Properties, Reserved, Mine Safety,
# governance/exhibits) that add retrieval noise without research value.
HIGH_VALUE_SECTIONS = {
    "Item 1",    # Business
    "Item 1A",   # Risk Factors
    "Item 1C",   # Cybersecurity
    "Item 3",    # Legal Proceedings
    "Item 5",    # Market for Registrant's Common Equity
    "Item 7",    # MD&A
    "Item 7A",   # Quantitative and Qualitative Disclosures About Market Risk
    "Item 8",    # Financial Statements and Supplementary Data
}


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


def save_sections(company: str, fiscal_year: str, sections: dict):
    """
    Save each section as a separate .txt file under data/processed/<company>/<fiscal_year>/
    """
    output_path = PROCESSED_DATA_PATH / company / fiscal_year
    output_path.mkdir(parents=True, exist_ok=True)

    for section_name, text in sections.items():
        if section_name not in HIGH_VALUE_SECTIONS:
            print(f"  Skipped (low-value): {section_name}")
            continue

        clean_name = section_name.replace(" ", "_")
        filepath = output_path / f"{clean_name}.txt"
        filepath.write_text(text, encoding="utf-8")
        print(f"  Saved: {filepath} ({len(text)} chars)")


def parse_all():
    for company_dir in RAW_DATA_PATH.iterdir():
        if not company_dir.is_dir():
            continue

        company = company_dir.name
        print(f"\nParsing {company.upper()}...")

        for year_dir in company_dir.iterdir():
            if not year_dir.is_dir():
                continue

            fiscal_year = year_dir.name

            for html_file in year_dir.glob("*.html"):
                sections = parse_10k(html_file)
                print(f"  Found {len(sections)} sections ({fiscal_year})")
                save_sections(company, fiscal_year, sections)

            metadata_file = year_dir / "filing_metadata.json"
            if metadata_file.exists():
                output_path = PROCESSED_DATA_PATH / company / fiscal_year
                output_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(metadata_file, output_path / "filing_metadata.json")


if __name__ == "__main__":
    parse_all()