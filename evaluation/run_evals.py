import argparse
import pandas as pd
import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from retrieval.hybrid_search import hybrid_search, load_all_chunks, build_bm25_index
from generation.synthesizer import synthesize
from generation.prompts import build_prompt
import csv
from datetime import datetime

RESULTS_PATH = Path("evaluation/results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

TEST_SET_PATH = Path("evaluation/test_set.csv")

SIMILARITY_THRESHOLD = 0.75
NUMERIC_TOLERANCE = 0.01  # 1% relative tolerance, to allow for rounding

NUMBER_PATTERN = re.compile(
    r"\$?\d[\d,]*\.?\d*\s*(trillion|billion|million|%)?",
    re.IGNORECASE
)

MULTIPLIERS = {"trillion": 1e12, "billion": 1e9, "million": 1e6}


def load_test_set() -> list[dict]:
    df = pd.read_csv(TEST_SET_PATH)
    return df.to_dict(orient="records")


def extract_numbers(text: str) -> set[float]:
    """
    Extract normalized numeric values from text (e.g. "$132.4 billion" -> 132_400_000_000.0).
    Percentages are kept as their literal value (e.g. "12%" -> 12.0).
    """
    numbers = set()
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group(0).strip()
        if not raw or not re.search(r"\d", raw):
            continue

        unit = match.group(1)
        digits = re.sub(r"[^\d.]", "", raw)
        if not digits or digits == ".":
            continue

        try:
            value = float(digits)
        except ValueError:
            continue

        if unit and unit.lower() in MULTIPLIERS:
            value *= MULTIPLIERS[unit.lower()]

        # Skip bare 4-digit calendar years (1900–2100) and small day-of-month
        # integers (≤31, no unit) — these appear in date phrases like "fiscal year
        # 2025" or "December 31" and are not financial figures the answer must cite.
        if not unit and value == int(value) and (1900 <= value <= 2100 or value <= 31):
            continue

        numbers.add(value)

    return numbers


def numbers_match(expected_numbers: set[float], generated_numbers: set[float]) -> bool:
    """
    True if every number in expected_numbers has a match in generated_numbers
    within NUMERIC_TOLERANCE (relative). Questions with no numeric claims in
    the expected answer trivially pass this check.
    """
    if not expected_numbers:
        return True

    for expected_val in expected_numbers:
        if not any(
            abs(expected_val - gen_val) <= NUMERIC_TOLERANCE * max(abs(expected_val), 1)
            for gen_val in generated_numbers
        ):
            return False

    return True


def evaluate_answer(generated: str, expected: str, model: SentenceTransformer) -> dict:
    """
    Evaluate a generated answer against expected using two free, local signals:
    - semantic similarity (bge-m3 cosine similarity) for overall meaning/phrasing
    - numeric match (regex extraction) for factual correctness of any figures cited

    A question only passes if both checks pass — semantic similarity alone can't
    tell "$416.2 billion" from "$391.0 billion" if the surrounding phrasing is close.
    """
    gen_embedding = model.encode(generated, normalize_embeddings=True)
    exp_embedding = model.encode(expected, normalize_embeddings=True)
    similarity = float(gen_embedding @ exp_embedding)

    expected_numbers = extract_numbers(expected)
    generated_numbers = extract_numbers(generated)
    numeric_pass = numbers_match(expected_numbers, generated_numbers)

    similarity_pass = similarity >= SIMILARITY_THRESHOLD

    return {
        "semantic_similarity": round(similarity, 4),
        "similarity_pass": similarity_pass,
        "numeric_pass": numeric_pass,
        "expected_numbers": sorted(expected_numbers),
        "pass": similarity_pass and numeric_pass
    }


def run_evals(use_local: bool = False):
    generator = "ollama/llama3.1:8b" if use_local else "gemini-2.5-flash"
    print(f"Loading components... (generator: {generator})")
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("BAAI/bge-m3")
    chunks = load_all_chunks()
    bm25 = build_bm25_index(chunks)

    test_cases = load_test_set()
    results = []

    print(f"\nRunning {len(test_cases)} eval questions...\n")

    for i, test in enumerate(test_cases):
        question = test["question"]
        expected = test["expected_answer"]
        question_type = test["question_type"]

        print(f"[{i+1}/{len(test_cases)}] {question[:60]}...")

        # Retrieve and generate
        retrieved = hybrid_search(client, model, bm25, chunks, question)
        result = synthesize(question, retrieved, use_local=use_local)
        generated = result["answer"]

        # Evaluate
        scores = evaluate_answer(generated, expected, model)

        results.append({
            "question": question,
            "question_type": question_type,
            "expected_answer": expected,
            "generated_answer": generated,
            "semantic_similarity": scores["semantic_similarity"],
            "similarity_pass": scores["similarity_pass"],
            "numeric_pass": scores["numeric_pass"],
            "expected_numbers": scores["expected_numbers"],
            "pass": scores["pass"],
            "sources_used": [s["section"] for s in result["sources"]]
        })

        print(f"  Similarity: {scores['semantic_similarity']} | Numeric: {scores['numeric_pass']} | Pass: {scores['pass']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_PATH / f"eval_results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    passed = sum(1 for r in results if r["pass"])
    numeric_failures = [r for r in results if not r["numeric_pass"]]
    avg_similarity = sum(r["semantic_similarity"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"EVAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total questions: {len(results)}")
    print(f"Passed: {passed}/{len(results)}")
    print(f"Pass rate: {passed/len(results)*100:.1f}%")
    print(f"Avg semantic similarity: {avg_similarity:.4f}")
    print(f"Numeric mismatches: {len(numeric_failures)}/{len(results)}")
    if numeric_failures:
        print("  (caught by numeric check despite high semantic similarity, if any:)")
        for r in numeric_failures:
            if r["similarity_pass"]:
                print(f"  - {r['question'][:60]}... expected numbers {r['expected_numbers']}")
    print(f"\nResults saved to: {output_path}")

    # Breakdown by question type
    print(f"\nBy question type:")
    for qtype in ["factual", "analytical"]:
        type_results = [r for r in results if r["question_type"] == qtype]
        if type_results:
            type_passed = sum(1 for r in type_results if r["pass"])
            type_avg = sum(r["semantic_similarity"] for r in type_results) / len(type_results)
            print(f"  {qtype}: {type_passed}/{len(type_results)} passed | avg similarity: {type_avg:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Use local Ollama model instead of Gemini")
    args = parser.parse_args()
    run_evals(use_local=args.local)