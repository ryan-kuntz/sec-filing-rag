# Retrieval quality (layer 1) evaluation: does retrieval surface the right section, independent
# of what the LLM later does with it? No generation calls, no API cost —
# just comparing retrieved chunk sections against the test set's section_hint.

import re
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from sentence_transformers import SentenceTransformer

from retrieval.hybrid_search import hybrid_search, load_all_chunks, build_bm25_index, get_client

TEST_SET_PATH = Path("evaluation/test_set.csv")
RESULTS_PATH = Path("evaluation/results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

TOP_K = 5

ITEM_PREFIX_PATTERN = re.compile(r"Item_\d+[A-C]?")


def load_test_set() -> list[dict]:
    df = pd.read_csv(TEST_SET_PATH)
    return df.to_dict(orient="records")


def normalize_section(section_hint: str) -> str:
    """
    Reduce a section_hint like 'Item_7_Managements_Discussion_and_Analysis_of_Fina'
    down to 'Item_7' so it can be compared against a chunk's 'section' field.
    """
    match = ITEM_PREFIX_PATTERN.match(section_hint)
    return match.group(0) if match else section_hint


def score_retrieval(retrieved: list[dict], expected_section: str) -> dict:
    """
    Score one question's retrieval against its expected section.
    """
    retrieved_sections = [r["section"] for r in retrieved]
    hits = [i for i, s in enumerate(retrieved_sections) if s == expected_section]

    hit = len(hits) > 0
    precision_at_k = len(hits) / len(retrieved_sections) if retrieved_sections else 0.0
    reciprocal_rank = 1 / (hits[0] + 1) if hit else 0.0

    return {
        "hit": hit,
        "precision_at_k": round(precision_at_k, 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
        "retrieved_sections": retrieved_sections,
    }


def run_retrieval_eval(top_k: int = TOP_K):
    print("Loading model and index...")
    client = get_client()
    model = SentenceTransformer("BAAI/bge-m3")
    chunks = load_all_chunks()
    bm25 = build_bm25_index(chunks)

    test_cases = load_test_set()
    results = []

    print(f"\nRunning {len(test_cases)} retrieval-only eval questions (top_k={top_k})...\n")

    for i, test in enumerate(test_cases):
        question = test["question"]
        expected_section = normalize_section(test["section_hint"])

        print(f"[{i+1}/{len(test_cases)}] {question[:60]}...")

        retrieved = hybrid_search(client, model, bm25, chunks, question, top_k=top_k)
        scores = score_retrieval(retrieved, expected_section)

        results.append({
            "question": question,
            "expected_section": expected_section,
            **scores,
        })

        print(f"  Hit: {scores['hit']} | Precision@{top_k}: {scores['precision_at_k']} | RR: {scores['reciprocal_rank']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_PATH / f"retrieval_eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    hit_rate = sum(r["hit"] for r in results) / len(results)
    avg_precision = sum(r["precision_at_k"] for r in results) / len(results)
    mrr = sum(r["reciprocal_rank"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"RETRIEVAL EVAL SUMMARY (top_k={top_k})")
    print(f"{'='*50}")
    print(f"Total questions: {len(results)}")
    print(f"Hit rate@{top_k}: {hit_rate*100:.1f}%")
    print(f"Avg precision@{top_k}: {avg_precision:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_retrieval_eval()
