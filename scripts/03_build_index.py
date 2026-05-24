"""Stage 3: Build FAISS indices and stats files.

medrag:
  - IndexFlatL2 per corpus
  - stats: {corpus}_stats.json = {"centroid": [...], "num_documents": N}

mmlu (wikipedia):
  - normalize_L2 then IndexFlatIP per cluster  ← NOT IndexFlatL2
  - stats: cluster_stats.json = [{"centroid": [...]}, ...]  (list, index=cluster_id)

Usage:
  python scripts/03_build_index.py --dataset medrag
  python scripts/03_build_index.py --dataset mmlu
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_SOURCES, EMBEDDINGS_DIR, STATS_DIR


MEDRAG_CORPORA = DATA_SOURCES["medrag"]


# ---------------------------------------------------------------------------
# medrag
# ---------------------------------------------------------------------------

def build_medrag_index(emb_dir: str, stats_dir: str) -> None:
    os.makedirs(stats_dir, exist_ok=True)

    for corpus in MEDRAG_CORPORA:
        emb_path = os.path.join(emb_dir, f"{corpus}_article_embeddings.npy")
        index_path = os.path.join(emb_dir, f"{corpus}_index.faiss")
        stats_path = os.path.join(stats_dir, f"{corpus}_stats.json")

        if os.path.exists(index_path) and os.path.exists(stats_path):
            print(f"[medrag] {corpus}: index + stats already exist, skipping.")
            continue

        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"{emb_path} not found. Run 02_build_embeddings.py --dataset medrag first."
            )

        embs = np.load(emb_path).astype(np.float32)
        print(f"[medrag] {corpus}: building IndexFlatL2 for {embs.shape} ...")

        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(embs)
        faiss.write_index(index, index_path)

        centroid = embs.mean(axis=0).tolist()
        stats = {"centroid": centroid, "num_documents": int(embs.shape[0])}
        with open(stats_path, "w") as f:
            json.dump(stats, f)

        print(f"[medrag] {corpus}: index={index.ntotal} vectors, stats saved.")


# ---------------------------------------------------------------------------
# mmlu (wikipedia)
# ---------------------------------------------------------------------------

def build_mmlu_index(emb_dir: str, stats_dir: str) -> None:
    os.makedirs(stats_dir, exist_ok=True)

    cluster_stats = []

    for c in range(10):
        emb_path = os.path.join(emb_dir, f"cluster_{c}_embeddings.npy")
        index_path = os.path.join(emb_dir, f"cluster_{c}_index.faiss")

        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"{emb_path} not found. Run 02_build_embeddings.py --dataset mmlu first."
            )

        embs = np.load(emb_path).astype(np.float32)
        centroid = embs.mean(axis=0)

        if not os.path.exists(index_path):
            print(f"[mmlu] cluster {c}: normalizing + building IndexFlatIP for {embs.shape} ...")
            embs_copy = embs.copy()
            faiss.normalize_L2(embs_copy)
            index = faiss.IndexFlatIP(embs_copy.shape[1])
            index.add(embs_copy)
            faiss.write_index(index, index_path)
            print(f"[mmlu] cluster {c}: index={index.ntotal} vectors.")
        else:
            print(f"[mmlu] cluster {c}: index already exists, skipping build.")

        # centroid is computed from UN-normalized embeddings (same as medrag)
        cluster_stats.append({"centroid": centroid.tolist()})

    # Single list file — index = cluster_id
    stats_path = os.path.join(stats_dir, "cluster_stats.json")
    with open(stats_path, "w") as f:
        json.dump(cluster_stats, f)
    print(f"[mmlu] cluster_stats.json saved ({len(cluster_stats)} entries).")


# ---------------------------------------------------------------------------
# arxiv (STEM extension experiment)
# ---------------------------------------------------------------------------

def build_arxiv_index(emb_dir: str, stats_dir: str) -> None:
    """Mirror of build_mmlu_index() for the arXiv dataset.

    Uses normalize_L2 + IndexFlatIP (same as wikipedia).
    Stats saved as arxiv_cluster_stats.json (separate from cluster_stats.json).
    """
    os.makedirs(stats_dir, exist_ok=True)

    cluster_stats = []

    for c in range(10):
        emb_path = os.path.join(emb_dir, f"cluster_{c}_embeddings.npy")
        index_path = os.path.join(emb_dir, f"cluster_{c}_index.faiss")

        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"{emb_path} not found. Run 02_build_embeddings.py --dataset arxiv first."
            )

        embs = np.load(emb_path).astype(np.float32)
        centroid = embs.mean(axis=0)

        if not os.path.exists(index_path):
            print(f"[arxiv] cluster {c}: normalizing + building IndexFlatIP for {embs.shape} ...")
            embs_copy = embs.copy()
            faiss.normalize_L2(embs_copy)
            index = faiss.IndexFlatIP(embs_copy.shape[1])
            index.add(embs_copy)
            faiss.write_index(index, index_path)
            print(f"[arxiv] cluster {c}: index={index.ntotal} vectors.")
        else:
            print(f"[arxiv] cluster {c}: index already exists, skipping build.")

        cluster_stats.append({"centroid": centroid.tolist()})

    stats_path = os.path.join(stats_dir, "arxiv_cluster_stats.json")
    with open(stats_path, "w") as f:
        json.dump(cluster_stats, f)
    print(f"[arxiv] arxiv_cluster_stats.json saved ({len(cluster_stats)} entries).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["medrag", "mmlu", "arxiv"])
    args = parser.parse_args()

    if args.dataset == "medrag":
        emb_dir = os.path.join(EMBEDDINGS_DIR, "mirage")
        print(f"=== Building medrag indices → {emb_dir} ===")
        build_medrag_index(emb_dir, STATS_DIR)
    elif args.dataset == "mmlu":
        emb_dir = os.path.join(EMBEDDINGS_DIR, "mmlu")
        print(f"=== Building mmlu indices → {emb_dir} ===")
        build_mmlu_index(emb_dir, STATS_DIR)
    else:
        emb_dir = os.path.join(EMBEDDINGS_DIR, "arxiv")
        print(f"=== Building arxiv indices → {emb_dir} ===")
        build_arxiv_index(emb_dir, STATS_DIR)

    print("\nDone. Next: python scripts/04_generate_train_data.py --config experiments/...")


if __name__ == "__main__":
    main()
