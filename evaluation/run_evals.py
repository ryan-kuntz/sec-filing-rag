import pandas as pd
import json
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


def load_test_set() -> list[dict]:
    df = pd.read_csv(TEST_SET_PATH)
    return df.to_dict(orient="records")


def evaluate_answer(generated: str, expected: str, model: SentenceTransformer) -> dict:
    """
    Evaluate a generated answer against expected using semantic similarity.
    """
    gen_embedding = model.encode(generated, normalize_embeddings=True)
    exp_embedding = model.encode(expected, normalize_embeddings=True)

    # Cosine similarity (dot product of normalized vectors)
    similarity = float(gen_embedding @ exp_embedding)

    return {
        "semantic_similarity": round(similarity, 4),
        "pass": similarity >= 0.75  # threshold for a passing answer
    }


def run_evals():
    print("Loading components...")
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
        result = synthesize(question, retrieved)
        generated = result["answer"]

        # Evaluate
        scores = evaluate_answer(generated, expected, model)

        results.append({
            "question": question,
            "question_type": question_type,
            "expected_answer": expected,
            "generated_answer": generated,
            "semantic_similarity": scores["semantic_similarity"],
            "pass": scores["pass"],
            "sources_used": [s["section"] for s in result["sources"]]
        })

        print(f"  Similarity: {scores['semantic_similarity']} | Pass: {scores['pass']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_PATH / f"eval_results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    passed = sum(1 for r in results if r["pass"])
    avg_similarity = sum(r["semantic_similarity"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"EVAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total questions: {len(results)}")
    print(f"Passed: {passed}/{len(results)}")
    print(f"Pass rate: {passed/len(results)*100:.1f}%")
    print(f"Avg semantic similarity: {avg_similarity:.4f}")
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
    run_evals()