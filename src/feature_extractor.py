from __future__ import annotations

from typing import List

import numpy as np

from src.config import INPUT_DIM, MEDRAG_SOURCE_TO_ID


class RouterFeatureExtractor:
    """Builds the fixed-dimension feature vector fed to CorpusRoutingNN.

    Feature layout (from actual code, NOT the paper description):
      [0 : 768]     query embedding
      [768 : 1536]  corpus centroid embedding
      [1536 : ]     one-hot vector over source ids

    Resulting dim:
      medrag:    768 + 768 + 4  = 1540
      wikipedia: 768 + 768 + 10 = 1546
    """

    def extract(
        self,
        query_vec: np.ndarray,
        centroid: np.ndarray,
        source_id: str,
        dataset: str,
        all_source_ids: List[str],
    ) -> np.ndarray:
        """Return feature vector as float32 array of shape (INPUT_DIM[dataset],)."""
        if query_vec.shape != (768,):
            raise ValueError(f"query_vec must be shape (768,), got {query_vec.shape}")
        if centroid.shape != (768,):
            raise ValueError(f"centroid must be shape (768,), got {centroid.shape}")

        n_sources = len(all_source_ids)
        expected_dim = INPUT_DIM[dataset]
        if 768 + 768 + n_sources != expected_dim:
            raise ValueError(
                f"all_source_ids length {n_sources} inconsistent with "
                f"INPUT_DIM['{dataset}'] = {expected_dim}"
            )

        if dataset == "medrag":
            source_idx = MEDRAG_SOURCE_TO_ID[source_id]
        elif dataset == "wikipedia":
            source_idx = int(source_id)
        else:
            raise ValueError(f"Unknown dataset '{dataset}'.")

        one_hot = np.zeros(n_sources, dtype=np.float32)
        one_hot[source_idx] = 1.0

        feature = np.concatenate([
            query_vec.astype(np.float32),
            centroid.astype(np.float32),
            one_hot,
        ])
        return feature
