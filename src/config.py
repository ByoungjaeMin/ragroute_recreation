import os

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

# DEVIATION FROM ORIGINAL: Colab resource constraints
# Original: ["pubmed", "statpearls", "textbooks", "wikipedia"] — 4 corpora, INPUT_DIM=1540
# This impl: ["textbooks", "statpearls"] — 2 corpora, INPUT_DIM=1538 (matches sacs-epfl Colab setup)
DATA_SOURCES = {
    "medrag":    ["textbooks", "statpearls"],
    "wikipedia": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
}

# Maps corpus name → one-hot index (medrag only)
MEDRAG_SOURCE_TO_ID = {
    "textbooks": 0, "statpearls": 1,
}

# Router input dim = 768 (query) + 768 (centroid) + n_sources (one-hot)
INPUT_DIM = {
    "medrag":    1538,   # 768 + 768 + 2
    "wikipedia": 1546,   # 768 + 768 + 10
}

# ---------------------------------------------------------------------------
# K values — keep these distinct, do NOT mix them up
# ---------------------------------------------------------------------------

LABEL_K    = 15   # top-k used for generating binary router labels (training)
K_RETRIEVE = 50   # per-source top-k during retrieval at inference time

# ---------------------------------------------------------------------------
# Training hyper-parameters (one dict per dataset)
# ---------------------------------------------------------------------------

TRAIN_CONFIG = {
    "medrag": {
        "seed":           12,
        "batch_size":     128,
        "epochs":         150,
        "lr_base":        1e-3,
        "lr_max":         5e-3,
        "weight_decay":   3e-5,
        "grad_clip":      1.0,
        "cyclic_step_up": 10,
        "cyclic_mode":    "triangular2",
        "cyclic_cutoff":  115,        # switch to StepLR from this epoch
        "step_lr_step":   50,
        "step_lr_gamma":  0.05,
        "best_metric":    "val_auc",  # NOT val_accuracy
        "use_pos_weight": False,      # BCEWithLogitsLoss without pos_weight
    },
    "wikipedia": {
        "seed":             42,
        "batch_size":       128,
        "epochs":           150,
        "lr_base":          1e-3,
        "lr_max":           5e-3,
        "weight_decay":     1e-5,
        "grad_clip":        1.0,
        "cyclic_step_up":   10,
        "cyclic_mode":      "triangular2",
        "cyclic_cutoff":    115,
        "step_lr_step":     50,
        "step_lr_gamma":    0.05,
        "best_metric":      "val_f1",  # NOT val_accuracy / val_auc
        "use_pos_weight":   True,
        "pos_weight_scale": 5.0,       # pos_weight = 5 * (neg / pos)
    },
}

# ---------------------------------------------------------------------------
# MMLU target subjects (8, not 10 as stated in the paper)
# ---------------------------------------------------------------------------

MMLU_TARGET_SUBJECTS = {
    "high_school_microeconomics",
    "international_law",
    "college_biology",
    "miscellaneous",
    "prehistory",
    "philosophy",
    "professional_psychology",
    "high_school_mathematics",
}

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

# DEVIATION FROM ORIGINAL: Ollama → vLLM (OpenAI-compatible API)
# Original: ollama.chat() at localhost:11434
# This impl: openai.OpenAI() at localhost:8000/v1 (vLLM server, matches sacs-epfl Colab setup)
LLM_CONFIG = {
    "vllm_model":          "unsloth/Meta-Llama-3.1-8B-Instruct",
    "hf_name":             "unsloth/Meta-Llama-3.1-8B-Instruct",
    "docs_context_length": 128000,
    "base_url":            "http://localhost:8000/v1",
    "api_key":             "dummy",
    "disable_rerank":      True,
}

# ---------------------------------------------------------------------------
# Paths (all local, relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")

BENCHMARK_DIR  = os.path.join(DATA_DIR, "benchmark")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")
STATS_DIR      = os.path.join(DATA_DIR, "stats")
RAW_DIR        = os.path.join(DATA_DIR, "raw")

# ---------------------------------------------------------------------------
# Embedding model names
# ---------------------------------------------------------------------------

MEDCPT_QUERY_ENCODER   = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
DPR_QUESTION_ENCODER   = "facebook/dpr-question_encoder-single-nq-base"
