#!/usr/bin/env bash
# Stage 1: Download raw data and copy benchmark files.
#
# Prerequisites:
#   - git
#   - huggingface_hub (pip install huggingface_hub)
#   - HF_TOKEN env var set (for gated repos)
#   - Original ragroute repo already cloned (for benchmark/question_order files)
#
# Usage:
#   bash scripts/01_download_data.sh [--ragroute-repo /path/to/ragroute]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT
DATA_DIR="$PROJECT_ROOT/data"
RAW_DIR="$DATA_DIR/raw/mirage"
BENCHMARK_DIR="$DATA_DIR/benchmark"

RAGROUTE_REPO=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --ragroute-repo) RAGROUTE_REPO="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Stage 1: Download data ==="

# ---------------------------------------------------------------------------
# 1. Copy benchmark files from original ragroute repo (required for reproducibility)
# ---------------------------------------------------------------------------
if [[ -n "$RAGROUTE_REPO" ]]; then
    echo "[1/3] Copying benchmark files from $RAGROUTE_REPO ..."
    mkdir -p "$BENCHMARK_DIR"
    cp "$RAGROUTE_REPO/data/benchmark/MIRAGE.json" "$BENCHMARK_DIR/"
    cp "$RAGROUTE_REPO"/data/question_order_*.json "$DATA_DIR/"
    echo "      Copied MIRAGE.json and question_order_*.json"
else
    echo "[1/3] --ragroute-repo not provided. Skipping benchmark file copy."
    echo "      You must manually place the following files:"
    echo "        data/benchmark/MIRAGE.json"
    echo "        data/question_order_MIRAGE_bioasq.json"
    echo "        data/question_order_MIRAGE_medmcqa.json"
    echo "        data/question_order_MIRAGE_medqa.json"
    echo "        data/question_order_MIRAGE_mmlu-med.json"
    echo "        data/question_order_MIRAGE_pubmedqa.json"
fi

# ---------------------------------------------------------------------------
# 2. Clone MedRAG (provides corpora: pubmed, statpearls, textbooks, wikipedia)
# ---------------------------------------------------------------------------
echo "[2/3] Cloning MedRAG toolkit ..."
MEDRAG_DIR="$PROJECT_ROOT/external/MedRAG"
if [[ -d "$MEDRAG_DIR" ]]; then
    echo "      MedRAG already cloned at $MEDRAG_DIR"
else
    mkdir -p "$PROJECT_ROOT/external"
    git clone https://github.com/Teddy-XiongGZ/MedRAG.git "$MEDRAG_DIR"
    echo "      Cloned to $MEDRAG_DIR"
fi

# Symlink or copy corpus chunks into data/raw/mirage/
echo "      Linking MedRAG corpora into data/raw/mirage/ ..."
mkdir -p "$RAW_DIR"
for corpus in pubmed statpearls textbooks wikipedia; do
    SRC="$MEDRAG_DIR/src/data/$corpus"
    DST="$RAW_DIR/$corpus"
    if [[ -d "$SRC" && ! -e "$DST" ]]; then
        ln -s "$SRC" "$DST"
        echo "        linked $corpus"
    elif [[ ! -d "$SRC" ]]; then
        echo "        WARNING: $SRC not found — MedRAG may need its own setup step"
    fi
done

# ---------------------------------------------------------------------------
# 3. Download Wikipedia subset for MMLU (1M snippets via HuggingFace datasets)
# ---------------------------------------------------------------------------
echo "[3/3] Downloading Wikipedia subset for MMLU ..."
python3 - <<'PYEOF'
import os, sys
try:
    from datasets import load_dataset
except ImportError:
    print("  ERROR: 'datasets' not installed. Run: pip install datasets")
    sys.exit(1)

save_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "data", "raw", "wikipedia_1m")
if os.path.exists(save_path):
    print(f"  Wikipedia subset already exists at {save_path}")
    sys.exit(0)

print("  Loading wikimedia/wikipedia (20231101.en) — this may take a while ...")
# Cohere embed v3 Wikipedia subset used in the paper is not publicly released.
# We use the raw Wikipedia dump and embed ourselves.
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

os.makedirs(save_path, exist_ok=True)
import json
count = 0
target = 1_000_000
out_file = open(os.path.join(save_path, "snippets.jsonl"), "w")
for article in ds:
    # Split article text into ~100-word chunks
    words = article["text"].split()
    for i in range(0, len(words), 100):
        chunk = " ".join(words[i:i+100])
        out_file.write(json.dumps({"title": article["title"], "text": chunk}) + "\n")
        count += 1
        if count >= target:
            break
    if count >= target:
        break
out_file.close()
print(f"  Saved {count} snippets to {save_path}/snippets.jsonl")
PYEOF

echo ""
echo "=== Stage 1 complete ==="
echo "Next: python scripts/02_build_embeddings.py --dataset medrag"
echo "      python scripts/02_build_embeddings.py --dataset mmlu"
