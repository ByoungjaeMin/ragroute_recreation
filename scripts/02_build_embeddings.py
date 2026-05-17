"""Stage 2: Build article and query embeddings.

medrag:
  - article embeddings: MedCPT-Article-Encoder, CLS pooling, per corpus
  - query embeddings:   MedCPT-Query-Encoder,   CLS pooling, per MIRAGE benchmark
    *** MIRAGE.json question order must be preserved exactly ***

mmlu (wikipedia):
  - query embeddings: DPR-question-encoder, pooler_output
    format: question + "\\n" + " | ".join(choices)
    q_id:   f"question_{enumerate_index}"
  - k-means clustering (n=10) → cluster_{i}_embeddings.npy + chunks.json

Usage:
  python scripts/02_build_embeddings.py --dataset medrag
  python scripts/02_build_embeddings.py --dataset mmlu
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BENCHMARK_DIR,
    DATA_DIR,
    EMBEDDINGS_DIR,
    MMLU_TARGET_SUBJECTS,
    RAW_DIR,
)
from src.embedding_model import EmbeddingModel


# ---------------------------------------------------------------------------
# medrag
# ---------------------------------------------------------------------------

MIRAGE_BENCHMARKS = ["pubmedqa", "medqa", "bioasq", "medmcqa", "mmlu-med"]

MEDRAG_CORPORA = ["pubmed", "statpearls", "textbooks", "wikipedia"]


def build_medrag_embeddings(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model = EmbeddingModel(dataset="medrag")

    # --- Article embeddings (one per corpus) ---
    for corpus in MEDRAG_CORPORA:
        emb_path = os.path.join(out_dir, f"{corpus}_article_embeddings.npy")
        chunks_path = os.path.join(out_dir, f"{corpus}_chunks.json")
        if os.path.exists(emb_path) and os.path.exists(chunks_path):
            print(f"[article] {corpus}: already exists, skipping.")
            continue

        corpus_dir = os.path.join(RAW_DIR, "mirage", corpus)
        chunks, texts = _load_medrag_corpus(corpus_dir)

        print(f"[article] {corpus}: encoding {len(texts)} chunks ...")
        embeddings = model.encode_articles(texts, batch_size=128)
        assert embeddings.shape == (len(chunks), 768), f"Unexpected shape {embeddings.shape}"

        np.save(emb_path, embeddings)
        with open(chunks_path, "w") as f:
            json.dump(chunks, f)
        print(f"[article] {corpus}: saved {embeddings.shape}")

    # --- Query embeddings (per MIRAGE benchmark, MUST preserve MIRAGE.json order) ---
    mirage_path = os.path.join(BENCHMARK_DIR, "MIRAGE.json")
    with open(mirage_path, "r") as f:
        mirage = json.load(f)

    for benchmark in MIRAGE_BENCHMARKS:
        emb_path = os.path.join(out_dir, f"{benchmark}_query_embeddings.npy")
        qids_path = os.path.join(out_dir, f"{benchmark}_query_ids.json")
        if os.path.exists(emb_path) and os.path.exists(qids_path):
            print(f"[query]   {benchmark}: already exists, skipping.")
            continue

        questions = mirage.get(benchmark, [])
        if not questions:
            print(f"[query]   {benchmark}: no questions found in MIRAGE.json, skipping.")
            continue

        # Preserve exact MIRAGE.json order — alignment with downstream stages depends on this
        q_ids = [item["id"] for item in questions]
        q_texts = [item["question"] for item in questions]

        print(f"[query]   {benchmark}: encoding {len(q_texts)} queries ...")
        embeddings = model.encode_batch(q_texts, batch_size=128)
        assert embeddings.shape == (len(q_texts), 768)

        np.save(emb_path, embeddings)
        with open(qids_path, "w") as f:
            json.dump(q_ids, f)
        print(f"[query]   {benchmark}: saved {embeddings.shape}")


def _load_medrag_corpus(corpus_dir: str):
    """Load all chunk JSON files from a MedRAG corpus directory.

    Returns (chunks, texts) where texts are the strings to encode.
    MedRAG chunk format: {"id": ..., "title": ..., "content": ...}
    """
    import glob

    chunk_files = sorted(glob.glob(os.path.join(corpus_dir, "*.json")) +
                         glob.glob(os.path.join(corpus_dir, "*.jsonl")))
    if not chunk_files:
        raise FileNotFoundError(f"No JSON/JSONL chunk files found in {corpus_dir}")

    chunks = []
    texts = []
    for fpath in tqdm(chunk_files, desc=f"Loading {os.path.basename(corpus_dir)}", leave=False):
        with open(fpath, "r") as f:
            if fpath.endswith(".jsonl"):
                items = [json.loads(l) for l in f if l.strip()]
            else:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
        for item in items:
            chunks.append(item)
            texts.append(f"{item.get('title', '')} {item.get('content', '')}".strip())

    return chunks, texts


# ---------------------------------------------------------------------------
# mmlu (wikipedia)
# ---------------------------------------------------------------------------

def build_mmlu_embeddings(out_dir: str) -> None:
    from sklearn.cluster import MiniBatchKMeans

    os.makedirs(out_dir, exist_ok=True)
    model = EmbeddingModel(dataset="wikipedia")

    # --- Query embeddings for MMLU target subjects ---
    q_emb_path = os.path.join(out_dir, "mmlu_query_embeddings.npy")
    q_ids_path = os.path.join(out_dir, "mmlu_query_ids.json")
    q_meta_path = os.path.join(out_dir, "mmlu_query_meta.json")

    if not (os.path.exists(q_emb_path) and os.path.exists(q_ids_path)):
        print("[query] Loading MMLU dataset ...")
        from datasets import load_dataset as hf_load
        ds = hf_load("cais/mmlu", "all", split="test")

        q_ids, q_texts, q_meta = [], [], []
        for i, item in enumerate(ds):
            if item["subject"] not in MMLU_TARGET_SUBJECTS:
                continue
            q_id = f"question_{i}"
            # MMLU query format: question + "\n" + " | ".join(choices)
            formatted = item["question"] + "\n" + " | ".join(item["choices"])
            q_ids.append(q_id)
            q_texts.append(formatted)
            q_meta.append({
                "id": q_id,
                "question": item["question"],
                "choices": item["choices"],
                "answer": int(item["answer"]),
                "subject": item["subject"],
            })

        print(f"[query] Encoding {len(q_texts)} MMLU queries ...")
        embeddings = model.encode_batch(q_texts, batch_size=128)
        np.save(q_emb_path, embeddings)
        with open(q_ids_path, "w") as f:
            json.dump(q_ids, f)
        with open(q_meta_path, "w") as f:
            json.dump(q_meta, f)
        print(f"[query] Saved {embeddings.shape}")
    else:
        print("[query] MMLU query embeddings already exist, skipping.")

    # --- Wikipedia article embeddings ---
    wiki_emb_path = os.path.join(out_dir, "wikipedia_all_embeddings.npy")
    wiki_chunks_path = os.path.join(out_dir, "wikipedia_all_chunks.json")

    if not os.path.exists(wiki_emb_path):
        snippets_path = os.path.join(DATA_DIR, "raw", "wikipedia_1m", "snippets.jsonl")
        if not os.path.exists(snippets_path):
            raise FileNotFoundError(
                f"Wikipedia snippets not found at {snippets_path}. "
                "Run scripts/01_download_data.sh first."
            )

        print("[article] Loading Wikipedia snippets ...")
        all_chunks = []
        all_texts = []
        with open(snippets_path, "r") as f:
            for line in tqdm(f, desc="Loading snippets"):
                item = json.loads(line)
                all_chunks.append((item["title"], item["text"]))
                all_texts.append(item["text"])

        print(f"[article] Encoding {len(all_texts)} Wikipedia chunks ...")
        # DEVIATION FROM ORIGINAL: original uses Cohere Embed V3 (proprietary, unavailable).
        # Original: cohere.embed(texts, model="embed-multilingual-v3.0")
        # This impl: DPR question encoder for both queries and passages (asymmetric use).
        #            Ideally DPRContextEncoder should be used for passages, but since
        #            the embedding space must match the query encoder at search time,
        #            and we cannot replicate Cohere, both sides use the same DPR encoder.
        embeddings = model.encode_batch(all_texts, batch_size=256)
        np.save(wiki_emb_path, embeddings)
        with open(wiki_chunks_path, "w") as f:
            json.dump(all_chunks, f)
        print(f"[article] Saved {embeddings.shape}")
    else:
        print("[article] Wikipedia embeddings already exist, skipping.")
        embeddings = np.load(wiki_emb_path)
        with open(wiki_chunks_path, "r") as f:
            all_chunks = json.load(f)

    # --- k-means clustering into 10 clusters ---
    cluster_done = all(
        os.path.exists(os.path.join(out_dir, f"cluster_{i}_embeddings.npy"))
        for i in range(10)
    )
    if cluster_done:
        print("[kmeans] Cluster files already exist, skipping.")
        return

    print("[kmeans] Fitting MiniBatchKMeans (n_clusters=10) ...")
    km = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=10000, n_init=3)
    labels = km.fit_predict(embeddings.astype(np.float32))

    for c in range(10):
        mask = labels == c
        c_emb = embeddings[mask]
        c_chunks = [all_chunks[i] for i in np.where(mask)[0]]
        np.save(os.path.join(out_dir, f"cluster_{c}_embeddings.npy"), c_emb.astype(np.float32))
        with open(os.path.join(out_dir, f"cluster_{c}_chunks.json"), "w") as f:
            json.dump(c_chunks, f)
        print(f"  cluster {c}: {c_emb.shape[0]} snippets")

    sizes = [(labels == c).sum() for c in range(10)]
    print(f"[kmeans] Cluster sizes: min={min(sizes)}, max={max(sizes)}")
    if min(sizes) < 49_000 or max(sizes) > 149_000:
        print("  WARNING: cluster sizes outside expected range [49k, 149k]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["medrag", "mmlu"])
    args = parser.parse_args()

    if args.dataset == "medrag":
        out_dir = os.path.join(EMBEDDINGS_DIR, "mirage")
        print(f"=== Building medrag embeddings → {out_dir} ===")
        build_medrag_embeddings(out_dir)
    else:
        out_dir = os.path.join(EMBEDDINGS_DIR, "mmlu")
        print(f"=== Building mmlu embeddings → {out_dir} ===")
        build_mmlu_embeddings(out_dir)

    print("\nDone. Next: python scripts/03_build_index.py --dataset", args.dataset)


if __name__ == "__main__":
    main()
