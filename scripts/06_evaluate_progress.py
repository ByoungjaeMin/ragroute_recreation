"""Thin wrapper around 06_evaluate.py that adds tqdm progress bar.

Patches call_llm_batch before calling main(). 06_evaluate.py is untouched.

Usage (same args as 06_evaluate.py):
  python scripts/06_evaluate_progress.py --config experiments/mirage_top32.yaml --mode no_rag
"""
from __future__ import annotations

import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

spec = importlib.util.spec_from_file_location(
    "evaluate", os.path.join(os.path.dirname(__file__), "06_evaluate.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _batch_with_progress(prompts, llm_cfg):
    with ThreadPoolExecutor(max_workers=mod.LLM_CONCURRENCY) as ex:
        fut = {ex.submit(mod.call_llm, s, u, llm_cfg): i for i, (s, u) in enumerate(prompts)}
        out = [""] * len(prompts)
        with tqdm(total=len(prompts), unit="q", ncols=80) as bar:
            for f in as_completed(fut):
                out[fut[f]] = f.result()
                bar.update(1)
    return out


mod.call_llm_batch = _batch_with_progress
mod.main()
