from bs4 import BeautifulSoup
from pathlib import Path
import re

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

CURATED_CONCEPTS = {
    # Income statement
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
    "us-gaap:Revenues": "Revenue",
    "us-gaap:CostOfGoodsAndServicesSold": "Cost of Revenue",
    "us-gaap:GrossProfit": "Gross Profit",
    "us-gaap:OperatingIncomeLoss": "Operating Income",
    "us-gaap:NetIncomeLoss": "Net Income",
    "us-gaap:EarningsPerShareBasic": "Earnings Per Share (Basic)",
    "us-gaap:EarningsPerShareDiluted": "Earnings Per Share (Diluted)",
    # Balance sheet
    "us-gaap:Assets": "Total Assets",
    "us-gaap:Liabilities": "Total Liabilities",
    "us-gaap:StockholdersEquity": "Stockholders Equity",
    "us-gaap:CashAndCashEquivalentsAtCarryingValue": "Cash and Cash Equivalents",
    "us-gaap:LongTermDebt": "Long-Term Debt",
    "us-gaap:LongTermDebtNoncurrent": "Long-Term Debt (Non-Current)",
    "us-gaap:ShortTermInvestments": "Short-Term Investments",
    "us-gaap:AccountsReceivableNetCurrent": "Accounts Receivable",
    "us-gaap:Goodwill": "Goodwill",
    # Cash flow
    "us-gaap:NetCashProvidedByUsedInOperatingActivities": "Operating Cash Flow",
    "us-gaap:NetCashProvidedByUsedInInvestingActivities": "Investing Cash Flow",
    "us-gaap:NetCashProvidedByUsedInFinancingActivities": "Financing Cash Flow",
    # Shares
    "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic": "Shares Outstanding (Basic)",
    "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": "Shares Outstanding (Diluted)",
    "us-gaap:CommonStockSharesOutstanding": "Common Shares Outstanding",
    # R&D
    "us-gaap:ResearchAndDevelopmentExpense": "R&D Expense",
}


def _build_context_lookup(soup):
    contexts = {}
    for ctx in soup.find_all("xbrli:context"):
        ctx_id = ctx.get("id")

        period = ctx.find("xbrli:period")
        period_info = {}
        if period:
            instant = period.find("xbrli:instant")
            start = period.find("xbrli:startdate")
            end = period.find("xbrli:enddate")
            if instant:
                period_info = {"type": "instant", "date": instant.text}
            elif start and end:
                period_info = {"type": "duration", "start": start.text, "end": end.text}

        segments = []
        segment_el = ctx.find("xbrli:segment")
        if segment_el:
            for member in segment_el.find_all("xbrldi:explicitmember"):
                segments.append({
                    "dimension": member.get("dimension", ""),
                    "value": member.text,
                })

        contexts[ctx_id] = {"period": period_info, "segments": segments}

    return contexts


def _format_value(numeric_value, label):
    if abs(numeric_value) >= 1e9:
        return f"${numeric_value/1e9:,.1f} billion"
    elif abs(numeric_value) >= 1e6:
        return f"${numeric_value/1e6:,.1f} million"
    elif "Share" in label or "Earnings Per" in label:
        return f"${numeric_value:,.2f}"
    else:
        return f"${numeric_value:,.0f}"


def _format_segment(segments):
    if not segments:
        return ""
    parts = []
    for s in segments:
        raw = s["value"].split(":")[-1]
        clean = raw.replace("Member", "")
        clean = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", clean)
        parts.append(clean)
    return f" ({', '.join(parts)})"


def extract_xbrl(html_path: Path, company: str) -> list[str]:
    with open(html_path, "rb") as f:
        soup = BeautifulSoup(f, "html.parser")

    contexts = _build_context_lookup(soup)
    sentences = []

    for tag in soup.find_all("ix:nonfraction"):
        concept = tag.get("name", "")
        if concept not in CURATED_CONCEPTS:
            continue

        label = CURATED_CONCEPTS[concept]
        raw_value = tag.get_text(strip=True)
        scale = int(tag.get("scale", "0"))
        ctx = contexts.get(tag.get("contextref", ""), {})

        try:
            clean_val = raw_value.replace(",", "").replace("(", "-").replace(")", "")
            if clean_val in ("-", "—", ""):
                continue
            numeric_value = float(clean_val) * (10 ** scale)
        except ValueError:
            continue

        period = ctx.get("period", {})
        if period.get("type") == "instant":
            period_str = f"as of {period['date']}"
        elif period.get("type") == "duration":
            period_str = f"for the period {period['start']} to {period['end']}"
        else:
            continue

        segment_str = _format_segment(ctx.get("segments", []))
        val_str = _format_value(numeric_value, label)

        sentence = f"{company.title()}'s {label}{segment_str} was {val_str} {period_str}."
        sentences.append(sentence)

    return sentences


def save_xbrl(company: str, sentences: list[str]):
    company_path = PROCESSED_DATA_PATH / company
    company_path.mkdir(parents=True, exist_ok=True)

    filepath = company_path / "xbrl_financials.txt"
    filepath.write_text("\n".join(sentences), encoding="utf-8")
    print(f"  Saved: {filepath} ({len(sentences)} sentences)")


def parse_all_xbrl():
    for company_dir in RAW_DATA_PATH.iterdir():
        if not company_dir.is_dir():
            continue

        company = company_dir.name
        print(f"\nExtracting XBRL for {company.upper()}...")

        for html_file in company_dir.glob("*.html"):
            sentences = extract_xbrl(html_file, company)
            print(f"  Found {len(sentences)} financial data points")
            save_xbrl(company, sentences)


if __name__ == "__main__":
    parse_all_xbrl()
