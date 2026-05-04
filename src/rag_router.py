from __future__ import annotations

import pickle
from typing import List

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.config import INPUT_DIM
from src.data_source import DataSource
from src.feature_extractor import RouterFeatureExtractor
from src.router_model import CorpusRoutingNN


class RAGRouter:
    """Inference-time router: given a query embedding, selects relevant sources.

    All sources are evaluated in a single batched forward pass for sub-millisecond
    selection latency (encode_query time is excluded from this measurement).
    """

    def __init__(
        self,
        model: CorpusRoutingNN,
        scaler: StandardScaler,
        sources: List[DataSource],
        dataset: str,
        threshold: float = 0.5,
    ):
        self.model = model
        self.scaler = scaler
        self.sources = sources
        self.dataset = dataset
        self.threshold = threshold
        self.extractor = RouterFeatureExtractor()

        self._source_ids = [src.source_id for src in sources]
        self._device = next(model.parameters()).device

    def route(self, query_vec: np.ndarray) -> List[DataSource]:
        """Return sources predicted relevant for query_vec.

        May return an empty list — no fallback to all_sources.
        """
        if query_vec.dtype != np.float32:
            raise TypeError(f"query_vec must be float32, got {query_vec.dtype}")

        features = np.stack([
            self.extractor.extract(
                query_vec, src.centroid, src.source_id, self.dataset, self._source_ids
            )
            for src in self.sources
        ]).astype(np.float32)   # (N_sources, INPUT_DIM)

        features_scaled = self.scaler.transform(features).astype(np.float32)

        X = torch.tensor(features_scaled, dtype=torch.float32).to(self._device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X).squeeze(1)          # (N_sources,)
            probs = torch.sigmoid(logits).cpu().numpy()

        return [src for src, prob in zip(self.sources, probs) if prob > self.threshold]

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        scaler_path: str,
        sources: List[DataSource],
        dataset: str,
        threshold: float = 0.5,
    ) -> "RAGRouter":
        """Load router from saved checkpoint and scaler."""
        input_dim = INPUT_DIM[dataset]
        model = CorpusRoutingNN(input_dim)

        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        return cls(
            model=model,
            scaler=scaler,
            sources=sources,
            dataset=dataset,
            threshold=threshold,
        )
