"""Stage 5: Train the CorpusRoutingNN router.

Loads X_train/y_train/X_val/y_val from Stage 4 output.
Saves best model checkpoint and fitted scaler.

Usage:
  python scripts/05_train_router.py --config experiments/mirage_top32.yaml
  python scripts/05_train_router.py --config experiments/mmlu_top10.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.router_trainer import RouterTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    processed_dir = cfg["paths"]["processed_dir"]
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"=== Stage 5: Train router [{dataset}] ===")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(processed_dir, "y_val.npy"))

    print(f"  X_train={X_train.shape}  pos={y_train.mean():.4f}")
    print(f"  X_val  ={X_val.shape}    pos={y_val.mean():.4f}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = RouterTrainer()
    model, scaler = trainer.train(X_train, y_train, X_val, y_val, dataset=dataset)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    model_path  = os.path.join(checkpoint_dir, f"{dataset}_model_best.pth")
    scaler_path = os.path.join(checkpoint_dir, f"{dataset}_scaler.pkl")

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n  Model saved  → {model_path}")
    print(f"  Scaler saved → {scaler_path}")

    # ------------------------------------------------------------------
    # Quick eval on test set
    # ------------------------------------------------------------------
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test_scaled).to(device)).squeeze(1).cpu().numpy()

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    acc  = accuracy_score(y_test, preds)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    auc  = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")

    print(f"\n  Test set results:")
    print(f"    Accuracy={acc*100:.2f}%  Recall={rec*100:.2f}%  F1={f1*100:.2f}%  AUC={auc*100:.2f}%")
    print(f"\nDone. Next: python scripts/06_evaluate.py --config {args.config}")


if __name__ == "__main__":
    main()
