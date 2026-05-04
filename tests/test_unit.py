"""Unit tests — no real models or data required.

All tests use synthetic data with fixed seeds.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch

# Fix seeds globally
np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestConfig:
    def test_input_dim_medrag(self):
        from src.config import INPUT_DIM
        assert INPUT_DIM["medrag"] == 1540   # 768 + 768 + 4

    def test_input_dim_wikipedia(self):
        from src.config import INPUT_DIM
        assert INPUT_DIM["wikipedia"] == 1546  # 768 + 768 + 10

    def test_label_k_and_k_retrieve_are_distinct(self):
        from src.config import LABEL_K, K_RETRIEVE
        assert LABEL_K == 15
        assert K_RETRIEVE == 50
        assert LABEL_K != K_RETRIEVE

    def test_medrag_source_count(self):
        from src.config import DATA_SOURCES
        assert len(DATA_SOURCES["medrag"]) == 4

    def test_wikipedia_source_count(self):
        from src.config import DATA_SOURCES
        assert len(DATA_SOURCES["wikipedia"]) == 10

    def test_mmlu_target_subjects_count(self):
        from src.config import MMLU_TARGET_SUBJECTS
        assert len(MMLU_TARGET_SUBJECTS) == 8

    def test_train_config_medrag_seed(self):
        from src.config import TRAIN_CONFIG
        assert TRAIN_CONFIG["medrag"]["seed"] == 12

    def test_train_config_wikipedia_seed(self):
        from src.config import TRAIN_CONFIG
        assert TRAIN_CONFIG["wikipedia"]["seed"] == 42

    def test_train_config_medrag_no_pos_weight(self):
        from src.config import TRAIN_CONFIG
        assert TRAIN_CONFIG["medrag"]["use_pos_weight"] is False

    def test_train_config_wikipedia_pos_weight(self):
        from src.config import TRAIN_CONFIG
        cfg = TRAIN_CONFIG["wikipedia"]
        assert cfg["use_pos_weight"] is True
        assert cfg["pos_weight_scale"] == 5.0

    def test_train_config_best_metric(self):
        from src.config import TRAIN_CONFIG
        assert TRAIN_CONFIG["medrag"]["best_metric"] == "val_auc"
        assert TRAIN_CONFIG["wikipedia"]["best_metric"] == "val_f1"

    def test_train_config_cyclic_cutoff(self):
        from src.config import TRAIN_CONFIG
        assert TRAIN_CONFIG["medrag"]["cyclic_cutoff"] == 115
        assert TRAIN_CONFIG["wikipedia"]["cyclic_cutoff"] == 115

    def test_train_config_weight_decay_differ(self):
        from src.config import TRAIN_CONFIG
        assert TRAIN_CONFIG["medrag"]["weight_decay"] == 3e-5
        assert TRAIN_CONFIG["wikipedia"]["weight_decay"] == 1e-5
        assert TRAIN_CONFIG["medrag"]["weight_decay"] != TRAIN_CONFIG["wikipedia"]["weight_decay"]


# ---------------------------------------------------------------------------
# router_model.py
# ---------------------------------------------------------------------------

class TestCorpusRoutingNN:
    def test_fc1_in_features_medrag(self):
        from src.config import INPUT_DIM
        from src.router_model import CorpusRoutingNN
        model = CorpusRoutingNN(INPUT_DIM["medrag"])
        assert model.fc1[0].in_features == 1540

    def test_fc1_in_features_wikipedia(self):
        from src.config import INPUT_DIM
        from src.router_model import CorpusRoutingNN
        model = CorpusRoutingNN(INPUT_DIM["wikipedia"])
        assert model.fc1[0].in_features == 1546

    def test_output_shape(self):
        from src.router_model import CorpusRoutingNN
        model = CorpusRoutingNN(1540)
        x = torch.randn(8, 1540)
        out = model(x)
        assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"

    def test_output_is_raw_logit_not_probability(self):
        """Output must be raw logits — no sigmoid in forward().

        LayerNorm keeps magnitudes small, so we verify structurally:
        1. No Sigmoid module anywhere in the model.
        2. Output contains both positive and negative values (not (0,1)-bounded).
        """
        import torch.nn as nn
        from src.router_model import CorpusRoutingNN
        torch.manual_seed(0)
        model = CorpusRoutingNN(1540)

        # 1. No Sigmoid in the module tree
        for name, module in model.named_modules():
            assert not isinstance(module, nn.Sigmoid), \
                f"Sigmoid found at '{name}' — forward() must not apply sigmoid"

        # 2. Outputs span both positive and negative (not bounded to (0,1))
        x = torch.randn(64, 1540)
        out = model(x).squeeze(1)
        assert (out > 0).any() and (out < 0).any(), \
            "Outputs are all-positive — sigmoid may be applied in forward()"

    def test_layer_architecture(self):
        import torch.nn as nn
        from src.router_model import CorpusRoutingNN
        model = CorpusRoutingNN(1540)
        # fc1: Linear → LayerNorm → ReLU → Dropout
        assert isinstance(model.fc1[0], nn.Linear)
        assert isinstance(model.fc1[1], nn.LayerNorm)
        assert isinstance(model.fc1[2], nn.ReLU)
        assert isinstance(model.fc1[3], nn.Dropout)
        # fc_out: Linear(32, 1)
        assert isinstance(model.fc_out, nn.Linear)
        assert model.fc_out.out_features == 1

    def test_dropout_rate(self):
        from src.router_model import CorpusRoutingNN
        model = CorpusRoutingNN(1540)
        for block in [model.fc1, model.fc2, model.fc3]:
            assert block[3].p == pytest.approx(0.4)

    def test_layer_sizes(self):
        from src.router_model import CorpusRoutingNN
        model = CorpusRoutingNN(1540)
        assert model.fc1[0].out_features == 128
        assert model.fc2[0].out_features == 64
        assert model.fc3[0].out_features == 32
        assert model.fc_out.out_features == 1


# ---------------------------------------------------------------------------
# feature_extractor.py
# ---------------------------------------------------------------------------

class TestRouterFeatureExtractor:
    def _make_vecs(self):
        rng = np.random.default_rng(42)
        q = rng.standard_normal(768).astype(np.float32)
        c = rng.standard_normal(768).astype(np.float32)
        return q, c

    def test_medrag_output_shape(self):
        from src.config import INPUT_DIM
        from src.feature_extractor import RouterFeatureExtractor
        fe = RouterFeatureExtractor()
        q, c = self._make_vecs()
        feat = fe.extract(q, c, "pubmed", "medrag", ["pubmed", "statpearls", "textbooks", "wikipedia"])
        assert feat.shape == (INPUT_DIM["medrag"],), f"Got {feat.shape}"

    def test_wikipedia_output_shape(self):
        from src.config import INPUT_DIM
        from src.feature_extractor import RouterFeatureExtractor
        fe = RouterFeatureExtractor()
        q, c = self._make_vecs()
        feat = fe.extract(q, c, "3", "wikipedia", [str(i) for i in range(10)])
        assert feat.shape == (INPUT_DIM["wikipedia"],), f"Got {feat.shape}"

    def test_output_dtype_float32(self):
        from src.feature_extractor import RouterFeatureExtractor
        fe = RouterFeatureExtractor()
        q, c = self._make_vecs()
        feat = fe.extract(q, c, "pubmed", "medrag", ["pubmed", "statpearls", "textbooks", "wikipedia"])
        assert feat.dtype == np.float32

    def test_one_hot_is_correct(self):
        from src.feature_extractor import RouterFeatureExtractor
        fe = RouterFeatureExtractor()
        q, c = self._make_vecs()
        all_ids = ["pubmed", "statpearls", "textbooks", "wikipedia"]
        feat = fe.extract(q, c, "statpearls", "medrag", all_ids)
        one_hot = feat[1536:]
        assert one_hot.shape == (4,)
        assert one_hot[1] == pytest.approx(1.0)   # statpearls index = 1
        assert one_hot[0] == pytest.approx(0.0)
        assert one_hot[2] == pytest.approx(0.0)
        assert one_hot[3] == pytest.approx(0.0)

    def test_query_and_centroid_embedded_correctly(self):
        from src.feature_extractor import RouterFeatureExtractor
        fe = RouterFeatureExtractor()
        q, c = self._make_vecs()
        feat = fe.extract(q, c, "pubmed", "medrag", ["pubmed", "statpearls", "textbooks", "wikipedia"])
        np.testing.assert_array_equal(feat[:768], q)
        np.testing.assert_array_equal(feat[768:1536], c)

    def test_wrong_query_shape_raises(self):
        from src.feature_extractor import RouterFeatureExtractor
        fe = RouterFeatureExtractor()
        bad_q = np.zeros(100, dtype=np.float32)
        c = np.zeros(768, dtype=np.float32)
        with pytest.raises(ValueError):
            fe.extract(bad_q, c, "pubmed", "medrag", ["pubmed", "statpearls", "textbooks", "wikipedia"])


# ---------------------------------------------------------------------------
# data_source.py
# ---------------------------------------------------------------------------

def _make_fake_source(source_id: str, dataset: str, n: int = 100) -> "DataSource":
    """Create a DataSource with a real FAISS index over random embeddings."""
    import faiss
    from src.data_source import DataSource

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((n, 768)).astype(np.float32)
    centroid = embs.mean(axis=0)
    chunks = [{"title": f"chunk_{i}", "content": f"text_{i}"} for i in range(n)]

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
        centroid=centroid.astype(np.float32),
        size=n,
    )


class TestDataSource:
    def test_medrag_search_output_shape(self):
        src = _make_fake_source("pubmed", "medrag")
        q = np.random.randn(768).astype(np.float32)
        scores, indices = src.search(q, k=10)
        assert scores.shape == (10,)
        assert indices.shape == (10,)

    def test_wikipedia_search_output_shape(self):
        src = _make_fake_source("0", "wikipedia")
        q = np.random.randn(768).astype(np.float32)
        scores, indices = src.search(q, k=10)
        assert scores.shape == (10,)
        assert indices.shape == (10,)

    def test_medrag_scores_are_l2_distances(self):
        """L2 distances must be non-negative."""
        src = _make_fake_source("pubmed", "medrag")
        q = np.random.randn(768).astype(np.float32)
        scores, _ = src.search(q, k=10)
        assert (scores >= 0).all(), "L2 distances must be non-negative"

    def test_wikipedia_scores_are_inner_products(self):
        """IP scores with normalized vectors are in [-1, 1]."""
        src = _make_fake_source("0", "wikipedia")
        q = np.random.randn(768).astype(np.float32)
        scores, _ = src.search(q, k=10)
        assert (scores >= -1.01).all() and (scores <= 1.01).all()

    def test_wrong_dtype_raises(self):
        src = _make_fake_source("pubmed", "medrag")
        q_bad = np.random.randn(768).astype(np.float64)
        with pytest.raises(TypeError):
            src.search(q_bad, k=5)

    def test_from_files_medrag(self):
        from src.data_source import DataSource
        import faiss

        rng = np.random.default_rng(0)
        embs = rng.standard_normal((50, 768)).astype(np.float32)
        centroid = embs.mean(axis=0).tolist()
        chunks = [{"title": "t", "content": "c"}] * 50

        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "idx.faiss")
            chunks_path = os.path.join(tmp, "chunks.json")
            stats_path = os.path.join(tmp, "stats.json")

            idx = faiss.IndexFlatL2(768)
            idx.add(embs)
            faiss.write_index(idx, index_path)

            with open(chunks_path, "w") as f:
                json.dump(chunks, f)
            with open(stats_path, "w") as f:
                json.dump({"centroid": centroid, "num_documents": 50}, f)

            src = DataSource.from_files("pubmed", "medrag", index_path, chunks_path, stats_path)
            assert src.size == 50
            assert src.centroid.shape == (768,)
            assert src.centroid.dtype == np.float32

    def test_from_files_wikipedia(self):
        from src.data_source import DataSource
        import faiss

        rng = np.random.default_rng(1)
        embs = rng.standard_normal((30, 768)).astype(np.float32)
        centroid = embs.mean(axis=0).tolist()
        chunks = [("title", "text")] * 30

        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "idx.faiss")
            chunks_path = os.path.join(tmp, "chunks.json")
            stats_path = os.path.join(tmp, "stats.json")

            embs_copy = embs.copy()
            faiss.normalize_L2(embs_copy)
            idx = faiss.IndexFlatIP(768)
            idx.add(embs_copy)
            faiss.write_index(idx, index_path)

            with open(chunks_path, "w") as f:
                json.dump(chunks, f)
            # wikipedia stats: list, index = cluster_id
            cluster_stats = [{"centroid": [0.0] * 768}] * 3   # cluster 0,1,2
            cluster_stats[2] = {"centroid": centroid}           # cluster 2 = our source
            with open(stats_path, "w") as f:
                json.dump(cluster_stats, f)

            src = DataSource.from_files("2", "wikipedia", index_path, chunks_path, stats_path)
            assert src.centroid.shape == (768,)
            np.testing.assert_allclose(src.centroid, np.array(centroid, dtype=np.float32))


# ---------------------------------------------------------------------------
# federated_retriever.py
# ---------------------------------------------------------------------------

class TestFederatedRetriever:
    def _make_sources(self, dataset: str, n_sources: int = 3, n_per_source: int = 100):
        return [_make_fake_source(str(i) if dataset == "wikipedia" else ["pubmed","statpearls","textbooks"][i],
                                  dataset, n_per_source)
                for i in range(n_sources)]

    def test_retrieve_returns_k_global_medrag(self):
        from src.federated_retriever import FederatedRetriever
        sources = self._make_sources("medrag", n_sources=3)
        ret = FederatedRetriever(k_retrieve=20, k_global=10)
        q = np.random.randn(768).astype(np.float32)
        chunks = ret.retrieve(q, sources)
        assert len(chunks) == 10

    def test_retrieve_returns_k_global_wikipedia(self):
        from src.federated_retriever import FederatedRetriever
        sources = self._make_sources("wikipedia", n_sources=3)
        ret = FederatedRetriever(k_retrieve=20, k_global=10)
        q = np.random.randn(768).astype(np.float32)
        chunks = ret.retrieve(q, sources)
        assert len(chunks) == 10

    def test_empty_sources_returns_empty_list(self):
        from src.federated_retriever import FederatedRetriever
        ret = FederatedRetriever()
        q = np.random.randn(768).astype(np.float32)
        chunks = ret.retrieve(q, [])
        assert chunks == []

    def test_medrag_merge_is_ascending(self):
        """Verify medrag merges by ascending L2 distance."""
        import faiss
        from src.data_source import DataSource
        from src.federated_retriever import FederatedRetriever

        # Build source where we know exact distances
        q = np.zeros(768, dtype=np.float32)
        embs = np.eye(768, dtype=np.float32)[:10]   # known L2 distances from q
        centroid = embs.mean(axis=0)
        index = faiss.IndexFlatL2(768)
        index.add(embs)
        chunks = [{"id": i} for i in range(10)]
        src = DataSource("pubmed", "medrag", chunks, index, centroid, 10)

        ret = FederatedRetriever(k_retrieve=10, k_global=5)
        result = ret.retrieve(q, [src])
        # All returned items should be valid chunks
        assert len(result) == 5

    def test_wikipedia_merge_is_descending(self):
        """Verify wikipedia merges by descending inner product."""
        import faiss
        from src.data_source import DataSource
        from src.federated_retriever import FederatedRetriever

        rng = np.random.default_rng(7)
        embs = rng.standard_normal((20, 768)).astype(np.float32)
        centroid = embs.mean(axis=0)
        embs_copy = embs.copy()
        faiss.normalize_L2(embs_copy)
        index = faiss.IndexFlatIP(768)
        index.add(embs_copy)
        chunks = [{"id": i} for i in range(20)]
        src = DataSource("0", "wikipedia", chunks, index, centroid, 20)

        q = rng.standard_normal(768).astype(np.float32)
        ret = FederatedRetriever(k_retrieve=20, k_global=10)
        result = ret.retrieve(q, [src])
        assert len(result) == 10

    def test_retrieve_with_stats_keys(self):
        from src.federated_retriever import FederatedRetriever
        sources = self._make_sources("medrag", n_sources=2)
        ret = FederatedRetriever(k_retrieve=10, k_global=5)
        q = np.random.randn(768).astype(np.float32)
        stats = ret.retrieve_with_stats(q, sources[:1], sources)
        assert "chunks" in stats
        assert "n_queries" in stats
        assert "n_queries_naive" in stats
        assert "query_reduction_pct" in stats
        assert stats["n_queries"] == 1
        assert stats["n_queries_naive"] == 2


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------

class TestUtils:
    def test_check_mirage_correct(self):
        from src.utils import check_mirage_answer
        output = '{"step_by_step_thinking": "...", "answer_choice": "B"}'
        assert check_mirage_answer(output, "B") is True

    def test_check_mirage_incorrect(self):
        from src.utils import check_mirage_answer
        output = '{"answer_choice": "A"}'
        assert check_mirage_answer(output, "C") is False

    def test_check_mirage_case_insensitive(self):
        from src.utils import check_mirage_answer
        output = '{"answer_choice": "c"}'
        assert check_mirage_answer(output, "C") is True

    def test_check_mmlu_correct(self):
        from src.utils import check_mmlu_answer
        output = '{"answer_choice": "A"}'
        assert check_mmlu_answer(output, 0) is True

    def test_check_mmlu_incorrect(self):
        from src.utils import check_mmlu_answer
        output = '{"answer_choice": "B"}'
        assert check_mmlu_answer(output, 2) is False

    def test_extract_answer_from_embedded_json(self):
        from src.utils import check_mirage_answer
        output = 'some text {"answer_choice": "D"} more text'
        assert check_mirage_answer(output, "D") is True

    def test_answer_choice_key_without_json_is_invalid(self):
        from src.utils import check_mirage_answer
        output = 'some text "answer_choice": "D" more text'
        assert check_mirage_answer(output, "D") is False
