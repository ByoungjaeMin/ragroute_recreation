"""Integration tests — verifies the full training and inference pipeline.

No real models or internet access required.
Uses synthetic FAISS indices with controlled data.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile

import faiss
import numpy as np
import pytest
import torch

np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_source(source_id: str, dataset: str, n: int = 200):
    from src.data_source import DataSource

    rng = np.random.default_rng(int(source_id) if source_id.isdigit() else ord(source_id[0]))
    embs = rng.standard_normal((n, 768)).astype(np.float32)
    centroid = embs.mean(axis=0).astype(np.float32)
    chunks = [{"title": f"doc_{source_id}_{i}", "content": f"text {i}"} for i in range(n)]

    if dataset == "medrag":
        index = faiss.IndexFlatL2(768)
        index.add(embs)
    else:
        embs_copy = embs.copy()
        faiss.normalize_L2(embs_copy)
        index = faiss.IndexFlatIP(768)
        index.add(embs_copy)

    return DataSource(
        source_id=source_id,
        dataset=dataset,
        chunks=chunks,
        index=index,
        centroid=centroid,
        size=n,
    )


def make_medrag_sources():
    return [_make_source(sid, "medrag") for sid in ["pubmed", "statpearls", "textbooks", "wikipedia"]]


def make_wikipedia_sources():
    return [_make_source(str(i), "wikipedia") for i in range(10)]


def make_query_vecs(n: int = 100) -> np.ndarray:
    return np.random.randn(n, 768).astype(np.float32)


# ---------------------------------------------------------------------------
# test_ragroute_end_to_end (medrag)
# ---------------------------------------------------------------------------

class TestEndToEndMedrag:
    def test_feature_shape(self):
        from src.config import INPUT_DIM
        from src.feature_extractor import RouterFeatureExtractor
        fe = RouterFeatureExtractor()
        q = np.random.randn(768).astype(np.float32)
        c = np.random.randn(768).astype(np.float32)
        all_ids = ["pubmed", "statpearls", "textbooks", "wikipedia"]
        feat = fe.extract(q, c, "pubmed", "medrag", all_ids)
        assert feat.shape == (INPUT_DIM["medrag"],)

    def test_generate_labels_medrag(self):
        """Labels must contain both 0 and 1; X shape must match INPUT_DIM."""
        from src.config import INPUT_DIM, LABEL_K
        from src.router_trainer import RouterTrainer
        sources = make_medrag_sources()
        q_vecs = make_query_vecs(30)
        q_ids = [f"q_{i}" for i in range(30)]

        trainer = RouterTrainer()
        X, y, row_qids = trainer.generate_labels(q_vecs, q_ids, sources, "medrag", k=LABEL_K)

        assert X.shape == (30 * 4, INPUT_DIM["medrag"])
        assert y.shape == (30 * 4,)
        assert X.dtype == np.float32
        assert y.dtype == np.float32
        assert len(np.unique(y)) == 2, "Labels must contain both 0 and 1"
        assert len(row_qids) == 30 * 4

    def test_question_level_split_medrag(self):
        from src.router_trainer import RouterTrainer
        q_ids = [f"q_{i}" for i in range(100)]
        with tempfile.TemporaryDirectory() as tmp:
            split_path = os.path.join(tmp, "split.json")
            trainer = RouterTrainer()
            split = trainer.split_questions(q_ids, "medrag", split_path)

        assert set(split.values()) == {"train", "val", "test"}
        n_test = sum(1 for v in split.values() if v == "test")
        assert abs(n_test - 60) <= 3   # ~60%

        with open(split_path) as f:
            loaded = json.load(f)
        assert loaded == split

    def test_model_fc1_input_dim_medrag(self):
        from src.config import INPUT_DIM
        from src.router_model import CorpusRoutingNN
        model = CorpusRoutingNN(INPUT_DIM["medrag"])
        assert model.fc1[0].in_features == INPUT_DIM["medrag"]

    def test_router_route_returns_list_medrag(self):
        """route() must return List[DataSource], empty list is valid."""
        from src.config import INPUT_DIM
        from src.router_model import CorpusRoutingNN
        from src.rag_router import RAGRouter
        from sklearn.preprocessing import StandardScaler

        sources = make_medrag_sources()
        model = CorpusRoutingNN(INPUT_DIM["medrag"])
        scaler = StandardScaler()
        # Fit scaler on dummy data
        dummy = np.random.randn(20, INPUT_DIM["medrag"]).astype(np.float32)
        scaler.fit(dummy)

        router = RAGRouter(model=model, scaler=scaler, sources=sources,
                           dataset="medrag", threshold=0.5)
        q = np.random.randn(768).astype(np.float32)
        selected = router.route(q)

        assert isinstance(selected, list)
        # Each element must be a DataSource
        from src.data_source import DataSource
        for s in selected:
            assert isinstance(s, DataSource)

    def test_retriever_returns_k_global_medrag(self):
        from src.federated_retriever import FederatedRetriever
        sources = make_medrag_sources()
        ret = FederatedRetriever(k_retrieve=50, k_global=32)
        q = np.random.randn(768).astype(np.float32)
        chunks = ret.retrieve(q, sources)
        assert len(chunks) == 32

    def test_retriever_empty_sources_no_fallback(self):
        from src.federated_retriever import FederatedRetriever
        ret = FederatedRetriever(k_retrieve=50, k_global=32)
        q = np.random.randn(768).astype(np.float32)
        chunks = ret.retrieve(q, [])
        assert chunks == [], "Must return empty list, no fallback to all sources"

    def test_merge_sort_ascending_medrag(self):
        """medrag retrieve must sort by L2 ascending (smaller distance first)."""
        from src.federated_retriever import FederatedRetriever
        sources = make_medrag_sources()[:2]

        results = [
            (3.0, sources[0], 0),
            (1.0, sources[1], 1),
            (2.0, sources[0], 2),
        ]

        sorted_results = FederatedRetriever._sort(results, "medrag")
        assert [score for score, _, _ in sorted_results] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# test_ragroute_end_to_end (wikipedia)
# ---------------------------------------------------------------------------

class TestEndToEndWikipedia:
    def test_feature_shape(self):
        from src.config import INPUT_DIM
        from src.feature_extractor import RouterFeatureExtractor
        fe = RouterFeatureExtractor()
        q = np.random.randn(768).astype(np.float32)
        c = np.random.randn(768).astype(np.float32)
        all_ids = [str(i) for i in range(10)]
        feat = fe.extract(q, c, "3", "wikipedia", all_ids)
        assert feat.shape == (INPUT_DIM["wikipedia"],)

    def test_generate_labels_wikipedia(self):
        """Labels must contain both 0 and 1."""
        from src.config import INPUT_DIM, LABEL_K
        from src.router_trainer import RouterTrainer
        sources = make_wikipedia_sources()
        q_vecs = make_query_vecs(20)
        q_ids = [f"question_{i}" for i in range(20)]

        trainer = RouterTrainer()
        X, y, row_qids = trainer.generate_labels(q_vecs, q_ids, sources, "wikipedia", k=LABEL_K)

        assert X.shape == (20 * 10, INPUT_DIM["wikipedia"])
        assert y.shape == (20 * 10,)
        assert len(np.unique(y)) == 2, "Labels must contain both 0 and 1"

    def test_model_fc1_input_dim_wikipedia(self):
        from src.config import INPUT_DIM
        from src.router_model import CorpusRoutingNN
        model = CorpusRoutingNN(INPUT_DIM["wikipedia"])
        assert model.fc1[0].in_features == INPUT_DIM["wikipedia"]

    def test_retriever_returns_k_global_wikipedia(self):
        from src.federated_retriever import FederatedRetriever
        sources = make_wikipedia_sources()
        ret = FederatedRetriever(k_retrieve=50, k_global=10)
        q = np.random.randn(768).astype(np.float32)
        chunks = ret.retrieve(q, sources)
        assert len(chunks) == 10

    def test_merge_sort_descending_wikipedia(self):
        """wikipedia retrieve must sort by IP descending (larger score first)."""
        from src.federated_retriever import FederatedRetriever
        sources = make_wikipedia_sources()[:3]

        results = [
            (0.10, sources[0], 0),
            (0.95, sources[1], 1),
            (0.40, sources[2], 2),
        ]

        sorted_results = FederatedRetriever._sort(results, "wikipedia")
        assert [score for score, _, _ in sorted_results] == [0.95, 0.40, 0.10]

    def test_router_route_wikipedia(self):
        from src.config import INPUT_DIM
        from src.router_model import CorpusRoutingNN
        from src.rag_router import RAGRouter
        from sklearn.preprocessing import StandardScaler

        sources = make_wikipedia_sources()
        model = CorpusRoutingNN(INPUT_DIM["wikipedia"])
        scaler = StandardScaler()
        scaler.fit(np.random.randn(20, INPUT_DIM["wikipedia"]).astype(np.float32))

        router = RAGRouter(model=model, scaler=scaler, sources=sources,
                           dataset="wikipedia", threshold=0.5)
        q = np.random.randn(768).astype(np.float32)
        selected = router.route(q)
        assert isinstance(selected, list)


# ---------------------------------------------------------------------------
# train pipeline smoke test (fast — tiny data, 3 epochs)
# ---------------------------------------------------------------------------

class TestTrainSmoke:
    def test_train_medrag_smoke(self):
        """Run 3 epochs on tiny synthetic data — should not raise."""
        from src.config import INPUT_DIM, TRAIN_CONFIG
        from src.router_trainer import RouterTrainer

        rng = np.random.default_rng(0)
        dim = INPUT_DIM["medrag"]
        X = rng.standard_normal((200, dim)).astype(np.float32)
        y = rng.integers(0, 2, 200).astype(np.float32)
        X_val = rng.standard_normal((40, dim)).astype(np.float32)
        y_val = rng.integers(0, 2, 40).astype(np.float32)

        # Patch epochs to 3 for speed
        original_epochs = TRAIN_CONFIG["medrag"]["epochs"]
        TRAIN_CONFIG["medrag"]["epochs"] = 3
        try:
            trainer = RouterTrainer()
            with tempfile.TemporaryDirectory() as tmp:
                model, scaler = trainer.train(X, y, X_val, y_val, "medrag")
        finally:
            TRAIN_CONFIG["medrag"]["epochs"] = original_epochs

        assert model.fc1[0].in_features == dim
        from sklearn.preprocessing import StandardScaler as SK_SS
        assert isinstance(scaler, SK_SS)

    def test_train_wikipedia_smoke(self):
        from src.config import INPUT_DIM, TRAIN_CONFIG
        from src.router_trainer import RouterTrainer

        rng = np.random.default_rng(1)
        dim = INPUT_DIM["wikipedia"]
        X = rng.standard_normal((200, dim)).astype(np.float32)
        y = rng.integers(0, 2, 200).astype(np.float32)
        X_val = rng.standard_normal((40, dim)).astype(np.float32)
        y_val = rng.integers(0, 2, 40).astype(np.float32)

        original_epochs = TRAIN_CONFIG["wikipedia"]["epochs"]
        TRAIN_CONFIG["wikipedia"]["epochs"] = 3
        try:
            trainer = RouterTrainer()
            with tempfile.TemporaryDirectory() as tmp:
                model, scaler = trainer.train(X, y, X_val, y_val, "wikipedia")
        finally:
            TRAIN_CONFIG["wikipedia"]["epochs"] = original_epochs

        assert model.fc1[0].in_features == dim

    def test_scaler_not_fit_on_val(self):
        """StandardScaler must be fit on train only — val uses transform."""
        from src.config import INPUT_DIM, TRAIN_CONFIG
        from src.router_trainer import RouterTrainer

        rng = np.random.default_rng(2)
        dim = INPUT_DIM["medrag"]
        # Make val distribution deliberately different to detect leakage
        X_train = rng.standard_normal((100, dim)).astype(np.float32)
        y_train = rng.integers(0, 2, 100).astype(np.float32)
        X_val = (rng.standard_normal((20, dim)) * 100 + 50).astype(np.float32)
        y_val = rng.integers(0, 2, 20).astype(np.float32)

        original_epochs = TRAIN_CONFIG["medrag"]["epochs"]
        TRAIN_CONFIG["medrag"]["epochs"] = 2
        try:
            trainer = RouterTrainer()
            model, scaler = trainer.train(X_train, y_train, X_val, y_val, "medrag")
        finally:
            TRAIN_CONFIG["medrag"]["epochs"] = original_epochs

        # Scaler mean should reflect train distribution (~0), not val (~50)
        assert abs(scaler.mean_.mean()) < 5.0, "Scaler appears to be fit on val data"

    def test_router_load_save(self):
        """Save and reload router; verify it produces same outputs."""
        from src.config import INPUT_DIM, TRAIN_CONFIG
        from src.router_model import CorpusRoutingNN
        from src.rag_router import RAGRouter
        from sklearn.preprocessing import StandardScaler

        sources = make_medrag_sources()
        dim = INPUT_DIM["medrag"]
        model = CorpusRoutingNN(dim)
        scaler = StandardScaler()
        scaler.fit(np.random.randn(20, dim).astype(np.float32))

        router = RAGRouter(model=model, scaler=scaler, sources=sources,
                           dataset="medrag", threshold=0.5)
        q = np.random.randn(768).astype(np.float32)
        selected_before = [s.source_id for s in router.route(q)]

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "model.pth")
            scaler_path = os.path.join(tmp, "scaler.pkl")
            torch.save(model.state_dict(), ckpt)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            router2 = RAGRouter.load(ckpt, scaler_path, sources, "medrag", threshold=0.5)

        selected_after = [s.source_id for s in router2.route(q)]
        assert selected_before == selected_after
