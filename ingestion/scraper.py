import os

import requests
import time
import json
from pathlib import Path

RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "your-name your-email@example.com")
}

COMPANIES = {
    "apple": "0000320193",
    "microsoft": "0000789019", 
    "google": "0001652044"
}


def get_latest_10k_filing(cik: str) -> dict:
    """
    Fetch metadata for the most recent 10-K filing for a given CIK:
    accession number, filing date, report (fiscal period) date, and document URL.
    """
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(submissions_url, headers=HEADERS)
    data = response.json()

    filings = data["filings"]["recent"]
    forms = filings["form"]
    accession_numbers = filings["accessionNumber"]
    primary_documents = filings["primaryDocument"]
    filing_dates = filings["filingDate"]
    report_dates = filings["reportDate"]

    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accession_numbers[i]
            accession_nodash = accession.replace("-", "")
            primary_doc = primary_documents[i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{primary_doc}"
            return {
                "accession_number": accession,
                "filing_date": filing_dates[i],
                "report_date": report_dates[i],
                "source_url": url
            }

    return None


def download_10k(company: str, cik: str):
    """
    Download the most recent 10-K for a company and save it under
    data/raw/<company>/<fiscal_year>/, alongside a filing_metadata.json sidecar.
    """
    print(f"\nFetching 10-K for {company.upper()}...")
    filing = get_latest_10k_filing(cik)

    if not filing:
        print(f"  No 10-K found for {company}")
        return

    url = filing["source_url"]
    fiscal_year = filing["report_date"][:4]
    print(f"  URL: {url}")
    print(f"  Fiscal year: {fiscal_year}")
    response = requests.get(url, headers=HEADERS)

    company_path = RAW_DATA_PATH / company / fiscal_year
    company_path.mkdir(parents=True, exist_ok=True)

    # Determine file extension
    ext = "html" if "htm" in url else "txt"
    filepath = company_path / f"10-K.{ext}"
    filepath.write_bytes(response.content)
    print(f"  Saved: {filepath}")

    metadata_path = company_path / "filing_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(filing, f, indent=2)
    print(f"  Saved: {metadata_path}")

    time.sleep(1)


def scrape_all():
    for company, cik in COMPANIES.items():
        download_10k(company, cik)


if __name__ == "__main__":
    scrape_all()
