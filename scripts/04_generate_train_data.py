"""Stage 4: Generate router training data (features + labels) and question-level split.

For each (query, source) pair:
  1. Search all sources → collect (score, source_id, local_idx)
  2. Sort by dataset-specific direction → global top-LABEL_K source set
  3. label = 1 if source in top-k else 0
  4. feature = [query_vec | centroid | one-hot]

Split is question-level (NOT row-level):
  test  = 60%  of all questions
  val   = 10%  of remaining train questions
  train = rest (~36% of all)

Usage:
  python scripts/04_generate_train_data.py --config experiments/mirage_top32.yaml
  python scripts/04_generate_train_data.py --config experiments/mmlu_top10.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_SOURCES,
    EMBEDDINGS_DIR,
    LABEL_K,
    STATS_DIR,
)
from src.data_source import DataSource
from src.router_trainer import RouterTrainer


MEDRAG_CORPORA = DATA_SOURCES["medrag"]


def _load_query_embeddings(emb_path: str, ids_path: str) -> tuple[np.ndarray, list[str]]:
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing query embeddings: {emb_path}")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"Missing query id file: {ids_path}")

    vecs = np.load(emb_path).astype(np.float32)
    with open(ids_path, "r") as f:
        ids = json.load(f)

    if vecs.ndim != 2 or vecs.shape[1] != 768:
        raise ValueError(f"{emb_path} must have shape (N, 768), got {vecs.shape}")
    if len(ids) != vecs.shape[0]:
        raise ValueError(
            f"Query id count mismatch for {emb_path}: {len(ids)} ids vs {vecs.shape[0]} embeddings"
        )

    return vecs, ids


def load_sources_medrag(emb_dir: str, stats_dir: str) -> list[DataSource]:
    sources = []
    for corpus in MEDRAG_CORPORA:
        sources.append(DataSource.from_files(
            source_id=corpus,
            dataset="medrag",
            index_path=os.path.join(emb_dir, f"{corpus}_index.faiss"),
            chunks_path=os.path.join(emb_dir, f"{corpus}_chunks.json"),
            stats_path=os.path.join(stats_dir, f"{corpus}_stats.json"),
        ))
    return sources


def load_sources_mmlu(emb_dir: str, stats_dir: str) -> list[DataSource]:
    sources = []
    for c in range(10):
        sources.append(DataSource.from_files(
            source_id=str(c),
            dataset="wikipedia",
            index_path=os.path.join(emb_dir, f"cluster_{c}_index.faiss"),
            chunks_path=os.path.join(emb_dir, f"cluster_{c}_chunks.json"),
            stats_path=os.path.join(stats_dir, "cluster_stats.json"),
        ))
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]                 # "medrag" or "wikipedia"
    out_dir = cfg["paths"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Stage 4: Generate training data [{dataset}] ===")

    # ------------------------------------------------------------------
    # Load sources
    # ------------------------------------------------------------------
    if dataset == "medrag":
        emb_dir = os.path.join(EMBEDDINGS_DIR, "mirage")
        sources = load_sources_medrag(emb_dir, STATS_DIR)
        benchmark_list = ["pubmedqa", "medqa", "bioasq", "medmcqa", "mmlu-med"]
    else:
        emb_dir = os.path.join(EMBEDDINGS_DIR, "mmlu")
        sources = load_sources_mmlu(emb_dir, STATS_DIR)
        benchmark_list = ["mmlu"]

    print(f"  Loaded {len(sources)} sources: {[s.source_id for s in sources]}")

    # ------------------------------------------------------------------
    # Load query embeddings and IDs
    # ------------------------------------------------------------------
    all_q_vecs = []
    all_q_ids = []

    if dataset == "medrag":
        for benchmark in benchmark_list:
            emb_path = os.path.join(emb_dir, f"{benchmark}_query_embeddings.npy")
            ids_path = os.path.join(emb_dir, f"{benchmark}_query_ids.json")
            vecs, ids = _load_query_embeddings(emb_path, ids_path)
            all_q_vecs.append(vecs)
            all_q_ids.extend(ids)
            print(f"  {benchmark}: {len(ids)} queries")
        all_q_vecs = np.concatenate(all_q_vecs, axis=0)
    else:
        emb_path = os.path.join(emb_dir, "mmlu_query_embeddings.npy")
        ids_path = os.path.join(emb_dir, "mmlu_query_ids.json")
        all_q_vecs, all_q_ids = _load_query_embeddings(emb_path, ids_path)
        print(f"  mmlu: {len(all_q_ids)} queries")

    print(f"  Total queries: {len(all_q_ids)}, embedding shape: {all_q_vecs.shape}")

    # ------------------------------------------------------------------
    # Question-level split
    # ------------------------------------------------------------------
    split_path = os.path.join(out_dir, "train_test_split.json")
    trainer = RouterTrainer()

    if os.path.exists(split_path):
        print(f"  Split file already exists: {split_path}")
        with open(split_path, "r") as f:
            split_dict = json.load(f)
    else:
        print("  Creating question-level split ...")
        split_dict = trainer.split_questions(all_q_ids, dataset, split_path)

    n_train = sum(1 for v in split_dict.values() if v == "train")
    n_val   = sum(1 for v in split_dict.values() if v == "val")
    n_test  = sum(1 for v in split_dict.values() if v == "test")
    print(f"  Split: train={n_train}, val={n_val}, test={n_test}")

    # ------------------------------------------------------------------
    # Generate labels (full dataset, then split)
    # ------------------------------------------------------------------
    all_npy = os.path.join(out_dir, "X_all.npy")
    all_y_npy = os.path.join(out_dir, "y_all.npy")
    all_qids_json = os.path.join(out_dir, "row_qids.json")

    cached_paths = [all_npy, all_y_npy, all_qids_json]
    existing_cached_paths = [p for p in cached_paths if os.path.exists(p)]
    if existing_cached_paths and len(existing_cached_paths) != len(cached_paths):
        missing = sorted(set(cached_paths) - set(existing_cached_paths))
        raise FileNotFoundError(
            "Incomplete pre-computed training data cache. "
            f"Existing={existing_cached_paths}, missing={missing}"
        )

    if len(existing_cached_paths) == len(cached_paths):
        print("  Loading pre-computed X/y ...")
        X_all = np.load(all_npy)
        y_all = np.load(all_y_npy)
        with open(all_qids_json, "r") as f:
            row_qids = json.load(f)
    else:
        print(f"  Generating labels (LABEL_K={LABEL_K}) ...")
        X_all, y_all, row_qids = trainer.generate_labels(
            all_q_vecs, all_q_ids, sources, dataset, k=LABEL_K
        )
        np.save(all_npy, X_all)
        np.save(all_y_npy, y_all)
        with open(all_qids_json, "w") as f:
            json.dump(row_qids, f)
        print(f"  X_all={X_all.shape}, y_all={y_all.shape}")

    expected_rows = len(all_q_ids) * len(sources)
    if X_all.shape[0] != expected_rows or y_all.shape[0] != expected_rows or len(row_qids) != expected_rows:
        raise ValueError(
            "Training data row count mismatch: "
            f"expected {expected_rows}, got X={X_all.shape[0]}, y={y_all.shape[0]}, row_qids={len(row_qids)}"
        )

    # Sanity check: labels must contain both 0 and 1
    unique_labels = np.unique(y_all)
    if len(unique_labels) < 2:
        raise RuntimeError(
            f"Label array contains only {unique_labels} — label generation bug. "
            "Check sort direction (medrag=ascending, wikipedia=descending)."
        )
    pos_rate = y_all.mean()
    print(f"  Label distribution: pos_rate={pos_rate:.4f} ({y_all.sum():.0f} / {len(y_all)})")

    # ------------------------------------------------------------------
    # Split X/y by question
    # ------------------------------------------------------------------
    row_qids_arr = np.array(row_qids)
    train_mask = np.array([split_dict.get(qid, "test") == "train" for qid in row_qids_arr])
    val_mask   = np.array([split_dict.get(qid, "test") == "val"   for qid in row_qids_arr])
    test_mask  = np.array([split_dict.get(qid, "test") == "test"  for qid in row_qids_arr])

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    print(f"  X_train={X_train.shape}  y_train pos={y_train.mean():.4f}")
    print(f"  X_val={X_val.shape}    y_val pos={y_val.mean():.4f}")
    print(f"  X_test={X_test.shape}   y_test pos={y_test.mean():.4f}")

    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(out_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(out_dir, "y_test.npy"),  y_test)

    print(f"\nDone. Files saved to {out_dir}")
    print("Next: python scripts/05_train_router.py --config", args.config)


if __name__ == "__main__":
    main()
