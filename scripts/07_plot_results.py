"""Stage 7: Generate report figures from evaluation results.

Reads results/*.json produced by 06_evaluate.py and generates:
  Figure 1 — MIRAGE accuracy: No-RAG vs RAG-all vs RAGRoute (per benchmark + avg)
  Figure 2 — MMLU accuracy:   No-RAG vs RAG-all vs RAGRoute
  Figure 3 — Router classification metrics (Acc / Recall / F1 / AUC)
  Figure 4 — Query reduction: selected-source rate per corpus (medrag)

Usage:
  python scripts/07_plot_results.py

Outputs saved to: figures/fig{1-4}.png  (300 dpi)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    CHECKPOINT_DIR,
    DATA_SOURCES,
    EMBEDDINGS_DIR,
    INPUT_DIM,
    PROCESSED_DIR,
    RESULTS_DIR,
    STATS_DIR,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIRAGE_BENCHMARKS = ["pubmedqa", "medqa", "bioasq", "medmcqa", "mmlu-med"]
MODES = ["no_rag", "rag_all", "ragroute"]
MODE_LABELS = {"no_rag": "No-RAG", "rag_all": "RAG-all", "ragroute": "RAGRoute"}
MODE_COLORS = {"no_rag": "#6c757d", "rag_all": "#4c72b0", "ragroute": "#dd8452"}

PAPER_TARGETS_MIRAGE = {
    "no_rag":   {"pubmedqa": 66.0,  "medqa": 60.6,  "bioasq": 78.0,  "medmcqa": 62.7, "mmlu-med": 68.9},
    "rag_all":  {"pubmedqa": 73.0,  "medqa": 64.0,  "bioasq": 82.5,  "medmcqa": 69.3, "mmlu-med": 72.4},
    "ragroute": {"pubmedqa": 73.2,  "medqa": 64.1,  "bioasq": 82.5,  "medmcqa": 69.3, "mmlu-med": 72.3},
}
PAPER_TARGETS_MMLU = {"no_rag": 66.67, "rag_all": 68.18, "ragroute": 70.45}

MEDRAG_CORPORA = DATA_SOURCES["medrag"]

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_result(results_dir: str, mode: str, benchmark: str) -> Optional[Dict]:
    path = os.path.join(results_dir, f"{mode}_{benchmark}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _avg_accuracy(results: Dict[str, Optional[Dict]]) -> Optional[float]:
    vals = [r["accuracy"] for r in results.values() if r is not None]
    return float(np.mean(vals)) if vals else None


def _bar_group(
    ax: plt.Axes,
    categories: List[str],
    data: Dict[str, List[Optional[float]]],
    ylabel: str,
    title: str,
    paper_refs: Optional[Dict[str, List[Optional[float]]]] = None,
    ylim: Tuple[float, float] = (50, 90),
) -> None:
    n_cat = len(categories)
    n_mode = len(MODES)
    width = 0.22
    x = np.arange(n_cat)

    for i, mode in enumerate(MODES):
        vals = data[mode]
        offset = (i - (n_mode - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
            alpha=0.88,
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333",
                )

    if paper_refs is not None:
        for i, mode in enumerate(MODES):
            refs = paper_refs[mode]
            offset = (i - (n_mode - 1) / 2) * width
            for j, ref in enumerate(refs):
                if ref is not None:
                    ax.plot(
                        x[j] + offset, ref, marker="_",
                        color="black", markersize=10, markeredgewidth=1.5,
                        zorder=4,
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Figure 1 — MIRAGE accuracy
# ---------------------------------------------------------------------------

def plot_mirage_accuracy(results_dir: str, out_path: str) -> None:
    categories = [b.upper() if b != "mmlu-med" else "MMLU-med" for b in MIRAGE_BENCHMARKS]
    categories.append("Average")
    benchmarks_plus_avg = MIRAGE_BENCHMARKS + ["avg"]

    data: Dict[str, List[Optional[float]]] = {m: [] for m in MODES}
    paper: Dict[str, List[Optional[float]]] = {m: [] for m in MODES}

    any_result = False
    for mode in MODES:
        per_bench: Dict[str, Optional[Dict]] = {
            b: _load_result(results_dir, mode, b) for b in MIRAGE_BENCHMARKS
        }
        for b in MIRAGE_BENCHMARKS:
            r = per_bench[b]
            data[mode].append(r["accuracy"] if r else None)
            paper[mode].append(PAPER_TARGETS_MIRAGE[mode].get(b))
            if r:
                any_result = True
        avg = _avg_accuracy(per_bench)
        data[mode].append(avg)
        paper[mode].append(
            float(np.mean(list(PAPER_TARGETS_MIRAGE[mode].values())))
        )

    if not any_result:
        print("[Fig 1] 결과 파일 없음 — 06_evaluate.py 실행 후 재시도하세요.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    _bar_group(
        ax, categories, data,
        ylabel="Accuracy (%)",
        title="Figure 1 — MIRAGE Benchmark Accuracy (Top-32)",
        paper_refs=paper,
        ylim=(50, 92),
    )
    ax.plot([], [], marker="_", color="black", linestyle="none",
            markersize=10, label="Paper target")
    ax.legend(fontsize=8.5, ncol=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Fig 1] saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — MMLU accuracy
# ---------------------------------------------------------------------------

def plot_mmlu_accuracy(results_dir: str, out_path: str) -> None:
    data: Dict[str, Optional[float]] = {}
    any_result = False
    for mode in MODES:
        r = _load_result(results_dir, mode, "mmlu")
        data[mode] = r["accuracy"] if r else None
        if r:
            any_result = True

    if not any_result:
        print("[Fig 2] 결과 파일 없음 — 06_evaluate.py (mmlu) 실행 후 재시도하세요.")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.arange(len(MODES))
    width = 0.5

    for i, mode in enumerate(MODES):
        val = data[mode]
        ref = PAPER_TARGETS_MMLU[mode]
        bar = ax.bar(i, val if val is not None else 0, width,
                     color=MODE_COLORS[mode], alpha=0.88, zorder=3,
                     label=MODE_LABELS[mode])
        if val is not None:
            ax.text(i, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.plot(i, ref, marker="_", color="black",
                markersize=18, markeredgewidth=2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Figure 2 — MMLU Accuracy (Top-10)", fontsize=11, fontweight="bold")
    ax.set_ylim(55, 80)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.plot([], [], marker="_", color="black", linestyle="none",
            markersize=12, label="Paper target")
    ax.legend(fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Fig 2] saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Router classification metrics
# ---------------------------------------------------------------------------

def plot_classification_metrics(out_path: str) -> None:
    from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
    import torch

    try:
        from src.router_model import CorpusRoutingNN
    except ImportError:
        print("[Fig 3] src.router_model import 실패.")
        return

    experiments = [
        ("MIRAGE Top-32", "medrag",    "data/processed/mirage", "medrag"),
        ("MIRAGE Top-10", "medrag",    "data/processed/mirage", "medrag"),
        ("MMLU Top-10",   "wikipedia", "data/processed/mmlu",   "wikipedia"),
    ]
    paper_targets = {
        "MIRAGE Top-32": (85.63, 85.47, 85.79, 92.60),
        "MIRAGE Top-10": (87.30, 88.32, 85.43, 93.67),
        "MMLU Top-10":   (90.06, 76.23, 78.29, 92.88),
    }
    metric_names = ["Accuracy", "Recall", "F1", "AUC"]
    metric_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    results = {}
    any_result = False
    for label, dataset, proc_dir, ckpt in experiments:
        X_path = os.path.join(proc_dir, "X_test.npy")
        y_path = os.path.join(proc_dir, "y_test.npy")
        m_path = os.path.join(CHECKPOINT_DIR, f"{ckpt}_model_best.pth")
        s_path = os.path.join(CHECKPOINT_DIR, f"{ckpt}_scaler.pkl")
        if not all(os.path.exists(p) for p in [X_path, y_path, m_path, s_path]):
            results[label] = None
            continue

        X_test = np.load(X_path).astype(np.float32)
        y_test = np.load(y_path)
        model = CorpusRoutingNN(INPUT_DIM[dataset])
        model.load_state_dict(torch.load(m_path, map_location="cpu", weights_only=True))
        model.eval()
        with open(s_path, "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X_test).astype(np.float32)
        with torch.no_grad():
            logits = model(torch.tensor(X_scaled)).squeeze(1).numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        results[label] = (
            accuracy_score(y_test, preds) * 100,
            recall_score(y_test, preds, zero_division=0) * 100,
            f1_score(y_test, preds, zero_division=0) * 100,
            roc_auc_score(y_test, probs) * 100 if len(np.unique(y_test)) > 1 else float("nan"),
        )
        any_result = True

    if not any_result:
        print("[Fig 3] 체크포인트 없음 — 05_train_router.py 실행 후 재시도하세요.")
        return

    exp_labels = [e[0] for e in experiments]
    n_exp = len(exp_labels)
    n_met = len(metric_names)
    x = np.arange(n_exp)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (metric, color) in enumerate(zip(metric_names, metric_colors)):
        vals = [
            results[lbl][j] if results[lbl] is not None else None
            for lbl in exp_labels
        ]
        refs = [paper_targets[lbl][j] for lbl in exp_labels]
        offset = (j - (n_met - 1) / 2) * width
        bars = ax.bar(x + offset, [v if v else 0 for v in vals],
                      width, label=metric, color=color, alpha=0.85, zorder=3)
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)
        for k, ref in enumerate(refs):
            ax.plot(x[k] + offset, ref, marker="_", color="black",
                    markersize=9, markeredgewidth=1.5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_title("Figure 3 — Router Classification Metrics", fontsize=11, fontweight="bold")
    ax.set_ylim(60, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.plot([], [], marker="_", color="black", linestyle="none",
            markersize=9, label="Paper target")
    ax.legend(fontsize=8.5, ncol=5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Fig 3] saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Query reduction (source selection rate per corpus)
# ---------------------------------------------------------------------------

def plot_query_reduction(results_dir: str, out_path: str) -> None:
    ragroute_results = {
        b: _load_result(results_dir, "ragroute", b) for b in MIRAGE_BENCHMARKS
    }
    if not any(r is not None for r in ragroute_results.values()):
        print("[Fig 4] ragroute 결과 없음 — 06_evaluate.py --mode ragroute 실행 후 재시도하세요.")
        return

    # Count per-corpus selection frequency from records
    corpus_selected = {c: 0 for c in MEDRAG_CORPORA}
    n_total = 0

    for benchmark, result in ragroute_results.items():
        if result is None:
            continue
        for rec in result.get("records", []):
            n_total += 1
            for src in rec.get("selected_sources", []):
                if src in corpus_selected:
                    corpus_selected[src] += 1

    if n_total == 0:
        print("[Fig 4] records 데이터 없음.")
        return

    # Selection rate (%) per corpus
    rates = [corpus_selected[c] / n_total * 100 for c in MEDRAG_CORPORA]
    # Naive baseline: always select all → 100%
    naive = [100.0] * len(MEDRAG_CORPORA)

    x = np.arange(len(MEDRAG_CORPORA))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_naive = ax.bar(x - width / 2, naive, width, label="RAG-all (100%)",
                        color="#4c72b0", alpha=0.5, zorder=3)
    bars_route = ax.bar(x + width / 2, rates, width, label="RAGRoute",
                        color="#dd8452", alpha=0.88, zorder=3)

    for bar, rate in zip(bars_route, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    overall_reduction = (1 - sum(corpus_selected.values()) / (n_total * len(MEDRAG_CORPORA))) * 100
    ax.set_xlabel("Corpus", fontsize=10)
    ax.set_ylabel("Query Selection Rate (%)", fontsize=10)
    ax.set_title(
        f"Figure 4 — Per-Corpus Query Selection Rate\n"
        f"(overall query reduction: {overall_reduction:.1f}%)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in MEDRAG_CORPORA], fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Fig 4] saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"결과 디렉터리: {RESULTS_DIR}")
    print(f"그래프 저장:   {FIGURES_DIR}\n")

    plot_mirage_accuracy(RESULTS_DIR, os.path.join(FIGURES_DIR, "fig1_mirage_accuracy.png"))
    plot_mmlu_accuracy(RESULTS_DIR, os.path.join(FIGURES_DIR, "fig2_mmlu_accuracy.png"))
    plot_classification_metrics(os.path.join(FIGURES_DIR, "fig3_classification_metrics.png"))
    plot_query_reduction(RESULTS_DIR, os.path.join(FIGURES_DIR, "fig4_query_reduction.png"))

    print("\n완료. figures/ 디렉터리를 확인하세요.")


if __name__ == "__main__":
    main()
