import os

import requests
import time
from pathlib import Path

RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "your-name your-email@example.com")
}

COMPANIES = {
    "apple": "0000320193",
}


def get_latest_10k_url(cik: str) -> str:
    """
    Fetch the most recent 10-K filing document URL for a given CIK.
    """
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(submissions_url, headers=HEADERS)
    data = response.json()

    filings = data["filings"]["recent"]
    forms = filings["form"]
    accession_numbers = filings["accessionNumber"]
    primary_documents = filings["primaryDocument"]

    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accession_numbers[i].replace("-", "")
            primary_doc = primary_documents[i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
            return url

    return None


def download_10k(company: str, cik: str):
    """
    Download the most recent 10-K for a company and save to data/raw.
    """
    print(f"\nFetching 10-K for {company.upper()}...")
    url = get_latest_10k_url(cik)

    if not url:
        print(f"  No 10-K found for {company}")
        return

    print(f"  URL: {url}")
    response = requests.get(url, headers=HEADERS)

    company_path = RAW_DATA_PATH / company
    company_path.mkdir(parents=True, exist_ok=True)

    # Determine file extension
    ext = "html" if "htm" in url else "txt"
    filepath = company_path / f"10k_latest.{ext}"
    filepath.write_bytes(response.content)
    print(f"  Saved: {filepath}")
    time.sleep(1)


def scrape_all():
    for company, cik in COMPANIES.items():
        download_10k(company, cik)


if __name__ == "__main__":
    scrape_all()
