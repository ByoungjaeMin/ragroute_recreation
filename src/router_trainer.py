from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CyclicLR, StepLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import INPUT_DIM, LABEL_K, TRAIN_CONFIG
from src.data_source import DataSource
from src.feature_extractor import RouterFeatureExtractor
from src.router_model import CorpusRoutingNN


class RouterTrainer:
    """Handles label generation, data splitting, and model training for RAGRoute."""

    def __init__(self):
        self.extractor = RouterFeatureExtractor()

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def generate_labels(
        self,
        query_vecs: np.ndarray,
        q_ids: List[str],
        sources: List[DataSource],
        dataset: str,
        k: int = LABEL_K,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate binary relevance labels for (query, source) pairs.

        For each query, search all sources, merge results with dataset-specific
        sort direction, assign label=1 to sources appearing in global top-k.

        Sort direction (CRITICAL):
          medrag:    L2 ascending  (reverse=False)
          wikipedia: IP descending (reverse=True)

        Returns:
            X       (Q * N_sources, INPUT_DIM[dataset]) float32
            y       (Q * N_sources,) float32
            row_qids list of q_id repeated N_sources times (for split masking)
        """
        if dataset not in ("medrag", "wikipedia", "arxiv"):
            raise ValueError(f"Unknown dataset '{dataset}'.")

        reverse = dataset in ("wikipedia", "arxiv")
        all_source_ids = [src.source_id for src in sources]

        X_rows: List[np.ndarray] = []
        y_rows: List[float] = []
        row_qids: List[str] = []

        for q_id, q_vec in tqdm(zip(q_ids, query_vecs), total=len(q_ids), desc="Generating labels"):
            q_vec = q_vec.astype(np.float32)

            # Collect (score, source_id, local_idx) across all sources
            all_results: List[Tuple[float, str, int]] = []
            for src in sources:
                scores, indices = src.search(q_vec, k)
                for score, idx in zip(scores, indices):
                    if idx >= 0:
                        all_results.append((float(score), src.source_id, int(idx)))

            all_results.sort(key=lambda x: x[0], reverse=reverse)
            top_k_source_ids = {sid for _, sid, _ in all_results[:k]}

            for src in sources:
                feat = self.extractor.extract(q_vec, src.centroid, src.source_id, dataset, all_source_ids)
                label = 1.0 if src.source_id in top_k_source_ids else 0.0
                X_rows.append(feat)
                y_rows.append(label)
                row_qids.append(q_id)

        X = np.stack(X_rows).astype(np.float32)
        y = np.array(y_rows, dtype=np.float32)
        return X, y, row_qids

    # ------------------------------------------------------------------
    # Question-level split
    # ------------------------------------------------------------------

    def split_questions(
        self,
        all_question_ids: List[str],
        dataset: str,
        save_path: str,
    ) -> Dict[str, str]:
        """Split question IDs into train/val/test at the question level.

        Split sizes (from actual code, not the paper's stated 30/10/60):
          test  = 60% of all questions
          val   = 10% of remaining train questions (~4% of all)
          train = remaining (~36% of all)

        Saves {q_id: "train"/"val"/"test"} JSON to save_path.
        """
        seed = TRAIN_CONFIG[dataset]["seed"]

        train_qs, test_qs = train_test_split(
            all_question_ids, test_size=0.6, random_state=seed
        )
        train_qs, val_qs = train_test_split(
            train_qs, test_size=0.1, random_state=seed
        )

        split_dict: Dict[str, str] = {}
        for q_id in train_qs:
            split_dict[q_id] = "train"
        for q_id in val_qs:
            split_dict[q_id] = "val"
        for q_id in test_qs:
            split_dict[q_id] = "test"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(split_dict, f)

        return split_dict

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        dataset: str,
    ) -> Tuple[CorpusRoutingNN, StandardScaler]:
        """Train CorpusRoutingNN and return (best_model, fitted_scaler).

        StandardScaler is fit on X_train only; val set uses transform only.
        LR schedule: CyclicLR for epochs 0-114, StepLR from epoch 115 onward.
        Best model selection: medrag=val_auc, wikipedia=val_f1.
        """
        cfg = TRAIN_CONFIG[dataset]
        self._set_seed(cfg["seed"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)

        # DataLoaders
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train_scaled),
                torch.tensor(y_train).unsqueeze(1),
            ),
            batch_size=cfg["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val_scaled),
                torch.tensor(y_val).unsqueeze(1),
            ),
            batch_size=cfg["batch_size"],
            shuffle=False,
        )

        # Model
        input_dim = INPUT_DIM[dataset]
        model = CorpusRoutingNN(input_dim).to(device)

        # Loss
        if cfg["use_pos_weight"]:
            n_pos = float(y_train.sum())
            n_neg = float((y_train == 0).sum())
            if n_pos == 0:
                raise RuntimeError("Training labels contain no positive examples.")
            pw = torch.tensor([cfg["pos_weight_scale"] * n_neg / n_pos], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr_base"],
            weight_decay=cfg["weight_decay"],
        )

        # Schedulers
        cyclic_scheduler = CyclicLR(
            optimizer,
            base_lr=cfg["lr_base"],
            max_lr=cfg["lr_max"],
            step_size_up=cfg["cyclic_step_up"],
            mode=cfg["cyclic_mode"],
            cycle_momentum=False,
        )
        step_scheduler = StepLR(
            optimizer,
            step_size=cfg["step_lr_step"],
            gamma=cfg["step_lr_gamma"],
        )

        best_score = -1.0
        best_state: Optional[dict] = None

        for epoch in range(cfg["epochs"]):
            # Train
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

            if epoch < cfg["cyclic_cutoff"]:
                cyclic_scheduler.step()
            else:
                step_scheduler.step()

            # Validate
            score = self._evaluate(model, val_loader, device, cfg["best_metric"])

            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1:3d}/{cfg['epochs']}  {cfg['best_metric']}={score:.4f}  best={best_score:.4f}")

        # Restore best weights
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()

        return model, scaler

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate(
        model: CorpusRoutingNN,
        loader: DataLoader,
        device: torch.device,
        metric: str,
    ) -> float:
        model.eval()
        all_logits: List[float] = []
        all_labels: List[float] = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch).squeeze(1).cpu().tolist()
                all_logits.extend(logits)
                all_labels.extend(y_batch.squeeze(1).tolist())

        labels = np.array(all_labels)
        probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
        preds = (probs >= 0.5).astype(int)

        if metric == "val_auc":
            if len(np.unique(labels)) < 2:
                return 0.0
            return float(roc_auc_score(labels, probs))
        elif metric == "val_f1":
            return float(f1_score(labels, preds, zero_division=0))
        else:
            raise ValueError(f"Unknown best_metric '{metric}'.")

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
