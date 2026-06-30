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

# Number of historical 10-K filings to ingest per company.
# 3 gives ~5 years of XBRL trend data (each filing carries 3-year comparatives)
# and 3 distinct narrative snapshots for risk-factor/MD&A diffing.
NUM_FILINGS = 3


def get_10k_filings(cik: str, num_filings: int = NUM_FILINGS) -> list[dict]:
    """
    Fetch metadata for the N most recent 10-K filings for a given CIK,
    in reverse chronological order (most recent first).
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

    results = []
    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accession_numbers[i]
            accession_nodash = accession.replace("-", "")
            primary_doc = primary_documents[i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{primary_doc}"
            results.append({
                "accession_number": accession,
                "filing_date": filing_dates[i],
                "report_date": report_dates[i],
                "source_url": url
            })
            if len(results) == num_filings:
                break

    return results


def download_10k_filings(company: str, cik: str):
    """
    Download the N most recent 10-Ks for a company. Each filing is saved under
    data/raw/<company>/<fiscal_year>/, alongside a filing_metadata.json sidecar.
    Skips a year if it has already been downloaded.
    """
    print(f"\nFetching 10-Ks for {company.upper()}...")
    filings = get_10k_filings(cik)

    if not filings:
        print(f"  No 10-K filings found for {company}")
        return

    for filing in filings:
        fiscal_year = filing["report_date"][:4]
        company_path = RAW_DATA_PATH / company / fiscal_year

        if any(company_path.glob("10-K.*")):
            print(f"  Skipped {fiscal_year} (already downloaded)")
            continue

        url = filing["source_url"]
        print(f"  Downloading {fiscal_year}: {url}")
        response = requests.get(url, headers=HEADERS)

        company_path.mkdir(parents=True, exist_ok=True)

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
        download_10k_filings(company, cik)


if __name__ == "__main__":
    scrape_all()
