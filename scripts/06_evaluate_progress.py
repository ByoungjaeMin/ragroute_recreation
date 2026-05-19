"""Thin wrapper around 06_evaluate.py that adds tqdm progress bar and fixes.

Patches applied (06_evaluate.py is untouched):
  1. call_llm        — max_tokens 256→512; handles BadRequestError (context too long)
  2. call_llm_batch  — tqdm progress bar
  3. _extract_answer_choice — lenient parsing: accepts "B: text", "Answer: B", etc.

Usage (same args as 06_evaluate.py):
  python scripts/06_evaluate_progress.py --config experiments/mirage_top32.yaml --mode no_rag
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

spec = importlib.util.spec_from_file_location(
    "evaluate", os.path.join(os.path.dirname(__file__), "06_evaluate.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ---------------------------------------------------------------------------
# Patch 1: call_llm — max_tokens 256→512, handle context-too-long error
# ---------------------------------------------------------------------------

def _call_llm_patched(system_prompt: str, user_prompt: str, llm_cfg: dict) -> str:
    from openai import OpenAI, BadRequestError
    client = OpenAI(base_url=llm_cfg["base_url"], api_key=llm_cfg["api_key"])

    def _create(prompt: str) -> str:
        response = client.chat.completions.create(
            model=llm_cfg["vllm_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content

    try:
        return _create(user_prompt)
    except BadRequestError as e:
        if "maximum context length" not in str(e):
            raise
        # Truncate context: keep first 2000 chars (question + first few chunks)
        truncated = user_prompt[:4000]
        return _create(truncated)


mod.call_llm = _call_llm_patched

# ---------------------------------------------------------------------------
# Patch 2: call_llm_batch — tqdm progress bar
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Patch 3: _extract_answer_choice — lenient parsing
# ---------------------------------------------------------------------------
import src.utils as _utils

_orig_extract = _utils._extract_answer_choice


def _extract_lenient(llm_output: str):
    # Try original strict parser first
    result = _orig_extract(llm_output)
    if result is not None:
        return result

    # Fallback: find any A/B/C/D in the answer_choice JSON value
    parsed = _utils._parse_json_object(llm_output)
    if isinstance(parsed, dict):
        ans = parsed.get("answer_choice", "")
        if isinstance(ans, str):
            m = re.search(r'\b([A-Da-d])\b', ans)
            if m:
                return m.group(1).upper()

    # Regex directly on raw output for JSON key
    m = re.search(r'"answer_choice"\s*:\s*"([A-Da-d])', llm_output)
    if m:
        return m.group(1).upper()

    # Free-text patterns: "The answer is B", "Answer: B", "(B)", "**B**"
    for pattern in [
        r'[Tt]he\s+(?:correct\s+)?answer\s+is\s+["\']?([A-Da-d])["\']?',
        r'[Aa]nswer\s*[:\-]\s*["\']?([A-Da-d])["\']?',
        r'\*{1,2}([A-Da-d])\*{1,2}',
        r'\(([A-Da-d])\)',
        r'^([A-Da-d])[).\s]',
    ]:
        m = re.search(pattern, llm_output, re.MULTILINE)
        if m:
            return m.group(1).upper()

    return None


_utils._extract_answer_choice = _extract_lenient

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

mod.main()
