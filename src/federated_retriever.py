from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.data_source import DataSource


class FederatedRetriever:
    """Queries selected data sources and merges results into a global top-k list.

    Merge sort direction differs by dataset (CRITICAL — wrong direction corrupts results):
      medrag:    L2 distances  → ascending  (smaller = more similar)
      wikipedia: inner product → descending (larger  = more similar)
    """

    def __init__(self, k_retrieve: int = 50, k_global: int = 32):
        self.k_retrieve = k_retrieve
        self.k_global = k_global

    def retrieve(
        self, query_vec: np.ndarray, selected_sources: List[DataSource]
    ) -> List[Any]:
        """Return global top-k_global chunks from selected sources.

        Returns an empty list if selected_sources is empty (no fallback).
        """
        if not selected_sources:
            return []

        all_results: List[Tuple[float, DataSource, int]] = []

        for src in selected_sources:
            scores, indices = src.search(query_vec, self.k_retrieve)
            for score, idx in zip(scores, indices):
                if idx >= 0:  # FAISS returns -1 for padding when index has < k items
                    all_results.append((float(score), src, int(idx)))

        dataset = selected_sources[0].dataset
        all_results = self._sort(all_results, dataset)

        top = all_results[: self.k_global]
        return [src.chunks[idx] for _, src, idx in top]

    def retrieve_with_stats(
        self,
        query_vec: np.ndarray,
        selected_sources: List[DataSource],
        all_sources: List[DataSource],
    ) -> Dict[str, Any]:
        """Retrieve and also report efficiency statistics.

        Returns:
            chunks:              retrieved chunk list
            n_queries:           number of sources actually queried
            n_queries_naive:     total number of sources (naive baseline)
            query_reduction_pct: percentage reduction vs naive
        """
        chunks = self.retrieve(query_vec, selected_sources)
        n_queries = len(selected_sources)
        n_queries_naive = len(all_sources)
        reduction = (
            (1.0 - n_queries / n_queries_naive) * 100.0 if n_queries_naive > 0 else 0.0
        )
        return {
            "chunks": chunks,
            "n_queries": n_queries,
            "n_queries_naive": n_queries_naive,
            "query_reduction_pct": reduction,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sort(
        results: List[Tuple[float, DataSource, int]], dataset: str
    ) -> List[Tuple[float, DataSource, int]]:
        if dataset == "medrag":
            return sorted(results, key=lambda x: x[0], reverse=False)
        elif dataset in ("wikipedia", "arxiv"):
            return sorted(results, key=lambda x: x[0], reverse=True)
        else:
            raise ValueError(f"Unknown dataset '{dataset}'.")
