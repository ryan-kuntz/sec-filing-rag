from bs4 import BeautifulSoup
from pathlib import Path
import re

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Note: This section relies on Apple HTML formatting and will likely break for other tech companies
def is_section_header(text: str, style: str) -> bool:
    """
    Identify major 10-K section headers by their text and style.
    """
    is_bold = "font-weight:700" in style
    is_9pt = "font-size:9pt" in style
    is_item = re.match(r"^Item\s+\d+[A-Z]?\.", text.strip())
    return is_bold and is_9pt and is_item is not None


def clean_text(text: str) -> str:
    """
    Clean up common artifacts in SEC HTML text.
    """
    text = text.replace("\xa0", " ")  # non-breaking spaces
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()


def parse_10k(html_path: Path) -> dict:
    """
    Parse a 10-K HTML file into clean sections keyed by Item number.
    """
    with open(html_path, "rb") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Remove noise tags
    for tag in soup(["script", "style", "meta", "noscript"]):
        tag.decompose()

    sections = {}
    current_section = None
    current_text = []

    for span in soup.find_all("span"):
        text = span.get_text(separator=" ", strip=True)
        style = span.get("style", "")

        if not text:
            continue

        text = clean_text(text)

        if is_section_header(text, style):
            # Save previous section
            if current_section and current_text:
                sections[current_section] = "\n\n".join(current_text)
            current_section = text
            current_text = []
        else:
            if current_section and len(text) > 30:
                current_text.append(text)

    # Save last section
    if current_section and current_text:
        sections[current_section] = "\n\n".join(current_text)

    return sections


def save_sections(company: str, sections: dict):
    """
    Save each section as a separate .txt file under data/processed/<company>/
    """
    company_path = PROCESSED_DATA_PATH / company
    company_path.mkdir(parents=True, exist_ok=True)

    for section_name, text in sections.items():
        # Create clean filename from section name
        clean_name = re.sub(r"[^\w\s]", "", section_name)
        clean_name = clean_name.replace(" ", "_")[:50]
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