"""Stage 6: End-to-end RAG evaluation.

Evaluates RAGRoute accuracy vs No-RAG baseline on the test split.
Uses question_order_*.json to fix evaluation order for reproducibility.

Modes:
  --mode ragroute  : encode → route → retrieve → LLM
  --mode no_rag    : no retrieval, LLM only
  --mode rag_all   : retrieve from ALL sources (naive baseline)

Usage:
  python scripts/06_evaluate.py --config experiments/mirage_top32.yaml --mode ragroute
  python scripts/06_evaluate.py --config experiments/mirage_top32.yaml --mode no_rag
  python scripts/06_evaluate.py --config experiments/mirage_top32.yaml --mode rag_all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BENCHMARK_DIR,
    DATA_DIR,
    EMBEDDINGS_DIR,
    K_RETRIEVE,
    LLM_CONFIG,
    STATS_DIR,
)
from src.data_source import DataSource
from src.embedding_model import EmbeddingModel
from src.federated_retriever import FederatedRetriever
from src.rag_router import RAGRouter
from src.utils import (
    MEDRAG_SYSTEM_PROMPT,
    check_mirage_answer,
    check_mmlu_answer,
    _chunk_to_text,
)


MEDRAG_CORPORA = ["pubmed", "statpearls", "textbooks", "wikipedia"]
MIRAGE_BENCHMARKS = ["pubmedqa", "medqa", "bioasq", "medmcqa", "mmlu-med"]


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def load_sources_medrag(emb_dir: str, stats_dir: str) -> List[DataSource]:
    return [
        DataSource.from_files(
            source_id=corpus, dataset="medrag",
            index_path=os.path.join(emb_dir, f"{corpus}_index.faiss"),
            chunks_path=os.path.join(emb_dir, f"{corpus}_chunks.json"),
            stats_path=os.path.join(stats_dir, f"{corpus}_stats.json"),
        )
        for corpus in MEDRAG_CORPORA
    ]


def load_sources_mmlu(emb_dir: str, stats_dir: str) -> List[DataSource]:
    return [
        DataSource.from_files(
            source_id=str(c), dataset="wikipedia",
            index_path=os.path.join(emb_dir, f"cluster_{c}_index.faiss"),
            chunks_path=os.path.join(emb_dir, f"cluster_{c}_chunks.json"),
            stats_path=os.path.join(stats_dir, "cluster_stats.json"),
        )
        for c in range(10)
    ]


# ---------------------------------------------------------------------------
# Question loaders
# ---------------------------------------------------------------------------

def load_mirage_questions(split_dict: Dict[str, str]) -> Dict[str, List[Dict]]:
    """Load test-split questions per benchmark, ordered by question_order_*.json."""
    mirage_path = os.path.join(BENCHMARK_DIR, "MIRAGE.json")
    with open(mirage_path, "r") as f:
        mirage = json.load(f)

    result: Dict[str, List[Dict]] = {}
    for benchmark in MIRAGE_BENCHMARKS:
        order_path = os.path.join(DATA_DIR, f"question_order_MIRAGE_{benchmark}.json")
        if not os.path.exists(order_path):
            raise FileNotFoundError(
                f"Missing required evaluation order file: {order_path}. "
                "Copy question_order_*.json from the original RAGRoute repo before evaluation."
            )

        with open(order_path, "r") as f:
            ordered_ids = json.load(f)

        qmap = {item["id"]: item for item in mirage.get(benchmark, [])}
        missing_ids = sorted(set(ordered_ids) - set(qmap))
        extra_ids = sorted(set(qmap) - set(ordered_ids))
        if missing_ids or extra_ids:
            raise ValueError(
                f"question_order file does not match MIRAGE benchmark '{benchmark}': "
                f"{len(missing_ids)} unknown ids in order file, {len(extra_ids)} MIRAGE ids missing from order file"
            )

        questions = [
            qmap[qid] for qid in ordered_ids
            if qid in qmap and split_dict.get(qid) == "test"
        ]
        result[benchmark] = questions
        print(f"  {benchmark}: {len(questions)} test questions")
    return result


def load_mmlu_questions(split_dict: Dict[str, str], emb_dir: str) -> List[Dict]:
    meta_path = os.path.join(emb_dir, "mmlu_query_meta.json")
    with open(meta_path, "r") as f:
        all_meta = json.load(f)
    return [item for item in all_meta if split_dict.get(item["id"]) == "test"]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, llm_cfg: Dict) -> str:
    import ollama
    response = ollama.chat(
        model=llm_cfg["ollama_model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.message.content


def build_user_prompt(question: str, options: Dict[str, str], chunks: List[Any]) -> str:
    if chunks:
        context_parts = [f"[{i+1}] {_chunk_to_text(c)}" for i, c in enumerate(chunks)]
        context = "\n".join(context_parts)
        return (
            f"Here are the relevant documents:\n{context}\n\n"
            f"Here is the question: {question}\n"
            f"Here are the potential choices:\n"
            + "\n".join(f"{k}: {v}" for k, v in options.items())
        )
    else:
        return (
            f"Here is the question: {question}\n"
            f"Here are the potential choices:\n"
            + "\n".join(f"{k}: {v}" for k, v in options.items())
        )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_mirage(
    cfg: Dict,
    mode: str,
    sources: List[DataSource],
    embedding_model: EmbeddingModel,
    router: RAGRouter | None,
    retriever: FederatedRetriever,
    split_dict: Dict[str, str],
    results_dir: str,
) -> None:
    questions_by_benchmark = load_mirage_questions(split_dict)
    llm_cfg = LLM_CONFIG

    for benchmark, questions in questions_by_benchmark.items():
        if not questions:
            continue

        results_path = os.path.join(results_dir, f"{mode}_{benchmark}.json")
        if os.path.exists(results_path):
            print(f"  {benchmark}: results already exist at {results_path}, skipping.")
            continue

        print(f"\n  Evaluating {benchmark} ({len(questions)} questions, mode={mode}) ...")
        records = []
        correct = 0

        for item in questions:
            q_id = item["id"]
            question = item["question"]
            options = item["options"]   # {"A": "...", "B": "...", ...}
            answer = item["answer"]     # "A" / "B" / "C" / "D"

            query_vec = embedding_model.encode_query(question)

            if mode == "no_rag":
                chunks = []
                selected_ids = []
            elif mode == "rag_all":
                chunks = retriever.retrieve(query_vec, sources)
                selected_ids = [s.source_id for s in sources]
            else:  # ragroute
                selected = router.route(query_vec)
                chunks = retriever.retrieve(query_vec, selected)
                selected_ids = [s.source_id for s in selected]

            user_prompt = build_user_prompt(question, options, chunks)
            llm_output = call_llm(MEDRAG_SYSTEM_PROMPT, user_prompt, llm_cfg)
            is_correct = check_mirage_answer(llm_output, answer)
            if is_correct:
                correct += 1

            records.append({
                "q_id": q_id,
                "correct": is_correct,
                "selected_sources": selected_ids,
                "n_chunks": len(chunks),
                "llm_output": llm_output,
            })

        accuracy = correct / len(questions) * 100
        print(f"  {benchmark}: accuracy={accuracy:.2f}% ({correct}/{len(questions)})")

        with open(results_path, "w") as f:
            json.dump({"accuracy": accuracy, "n_correct": correct, "n_total": len(questions), "records": records}, f, indent=2)


def evaluate_mmlu(
    cfg: Dict,
    mode: str,
    sources: List[DataSource],
    embedding_model: EmbeddingModel,
    router: RAGRouter | None,
    retriever: FederatedRetriever,
    split_dict: Dict[str, str],
    results_dir: str,
) -> None:
    emb_dir = os.path.join(EMBEDDINGS_DIR, "mmlu")
    questions = load_mmlu_questions(split_dict, emb_dir)
    llm_cfg = LLM_CONFIG

    results_path = os.path.join(results_dir, f"{mode}_mmlu.json")
    if os.path.exists(results_path):
        print(f"  MMLU results already exist at {results_path}, skipping.")
        return

    print(f"\n  Evaluating MMLU ({len(questions)} questions, mode={mode}) ...")
    records = []
    correct = 0

    for item in questions:
        q_id = item["id"]
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        options = {chr(65 + i): c for i, c in enumerate(choices)}

        formatted_q = question + "\n" + " | ".join(choices)
        query_vec = embedding_model.encode_query(formatted_q)

        if mode == "no_rag":
            chunks = []
            selected_ids = []
        elif mode == "rag_all":
            chunks = retriever.retrieve(query_vec, sources)
            selected_ids = [s.source_id for s in sources]
        else:
            selected = router.route(query_vec)
            chunks = retriever.retrieve(query_vec, selected)
            selected_ids = [s.source_id for s in selected]

        user_prompt = build_user_prompt(question, options, chunks)
        llm_output = call_llm(MEDRAG_SYSTEM_PROMPT, user_prompt, llm_cfg)
        is_correct = check_mmlu_answer(llm_output, answer_idx)
        if is_correct:
            correct += 1

        records.append({
            "q_id": q_id,
            "correct": is_correct,
            "selected_sources": selected_ids,
            "n_chunks": len(chunks),
        })

    accuracy = correct / len(questions) * 100
    print(f"  MMLU: accuracy={accuracy:.2f}% ({correct}/{len(questions)})")

    with open(results_path, "w") as f:
        json.dump({"accuracy": accuracy, "n_correct": correct, "n_total": len(questions), "records": records}, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["ragroute", "no_rag", "rag_all"], default="ragroute")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    k_eval = cfg["k_eval"]
    processed_dir = cfg["paths"]["processed_dir"]
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print(f"=== Stage 6: Evaluate [{dataset}] mode={args.mode} k_eval={k_eval} ===")

    # Load split
    split_path = os.path.join(processed_dir, "train_test_split.json")
    with open(split_path, "r") as f:
        split_dict = json.load(f)

    # Load sources
    if dataset == "medrag":
        emb_dir = os.path.join(EMBEDDINGS_DIR, "mirage")
        sources = load_sources_medrag(emb_dir, STATS_DIR)
    else:
        emb_dir = os.path.join(EMBEDDINGS_DIR, "mmlu")
        sources = load_sources_mmlu(emb_dir, STATS_DIR)

    print(f"  Loaded {len(sources)} sources.")

    # Embedding model
    embedding_model = EmbeddingModel(dataset=dataset)

    # Retriever
    retriever = FederatedRetriever(k_retrieve=K_RETRIEVE, k_global=k_eval)

    # Router (only needed for ragroute mode)
    router = None
    if args.mode == "ragroute":
        model_path  = os.path.join(checkpoint_dir, f"{dataset}_model_best.pth")
        scaler_path = os.path.join(checkpoint_dir, f"{dataset}_scaler.pkl")
        router = RAGRouter.load(
            checkpoint_path=model_path,
            scaler_path=scaler_path,
            sources=sources,
            dataset=dataset,
            threshold=cfg["router"]["threshold"],
        )
        print(f"  Router loaded from {model_path}")

    # Evaluate
    if dataset == "medrag":
        evaluate_mirage(cfg, args.mode, sources, embedding_model, router, retriever, split_dict, results_dir)
    else:
        evaluate_mmlu(cfg, args.mode, sources, embedding_model, router, retriever, split_dict, results_dir)

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
