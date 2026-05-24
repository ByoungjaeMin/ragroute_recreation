from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Tuple

import faiss
import numpy as np


@dataclass
class DataSource:
    source_id: str
    dataset: str          # "medrag" or "wikipedia"
    chunks: List[Any]     # medrag: List[dict] / wikipedia: List[(title, text)]
    index: faiss.Index
    centroid: np.ndarray  # float32, shape (768,)
    size: int

    def search(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) each shape (k,).

        medrag:    scores = L2 distances  (smaller is better)
        wikipedia: scores = inner products (larger is better)
        """
        if self.index is None:
            raise RuntimeError(f"DataSource '{self.source_id}': index not loaded.")
        if query_vec.dtype != np.float32:
            raise TypeError(
                f"DataSource '{self.source_id}': query_vec must be float32, got {query_vec.dtype}"
            )

        if self.dataset == "medrag":
            q = query_vec.reshape(1, -1)
            scores, indices = self.index.search(q, k)
            return scores[0], indices[0]

        elif self.dataset in ("wikipedia", "arxiv"):
            # DEVIATION FROM ORIGINAL: normalize in-place on a copy to avoid mutating caller's array
            # Original: same logic, explicit copy required before faiss.normalize_L2
            # This impl: reshape + copy, then normalize, then search
            query_copy = query_vec.reshape(1, -1).copy()
            faiss.normalize_L2(query_copy)
            scores, indices = self.index.search(query_copy, k)
            return scores[0], indices[0]

        else:
            raise ValueError(f"Unknown dataset '{self.dataset}'. Expected 'medrag', 'wikipedia', or 'arxiv'.")

    @classmethod
    def from_files(
        cls,
        source_id: str,
        dataset: str,
        index_path: str,
        chunks_path: str,
        stats_path: str,
    ) -> "DataSource":
        """Load a DataSource from pre-built index, chunks, and stats files.

        medrag stats file:    {"centroid": [...], "num_documents": N}
        wikipedia stats file: [{"centroid": [...]}, ...]  (list, index = cluster_id)
        """
        index = faiss.read_index(index_path)

        with open(chunks_path, "r") as f:
            chunks = json.load(f)

        with open(stats_path, "r") as f:
            raw = json.load(f)

        if dataset == "medrag":
            centroid = np.array(raw["centroid"], dtype=np.float32)
            size = raw["num_documents"]

        elif dataset in ("wikipedia", "arxiv"):
            # raw is a list; index into it by cluster id
            cluster_id = int(source_id)
            centroid = np.array(raw[cluster_id]["centroid"], dtype=np.float32)
            size = len(chunks)

        else:
            raise ValueError(f"Unknown dataset '{dataset}'. Expected 'medrag', 'wikipedia', or 'arxiv'.")

        return cls(
            source_id=source_id,
            dataset=dataset,
            chunks=chunks,
            index=index,
            centroid=centroid,
            size=size,
        )
