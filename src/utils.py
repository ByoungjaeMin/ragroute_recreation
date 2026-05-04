from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def check_mirage_answer(llm_output: str, correct_answer: str) -> bool:
    """Parse LLM JSON output and compare answer to ground truth."""
    predicted = _extract_answer_choice(llm_output)
    if predicted is None:
        return False
    return predicted.strip().upper() == correct_answer.strip().upper()


def check_mmlu_answer(llm_output: str, correct_idx: int) -> bool:
    """Parse LLM JSON output and compare answer to ground-truth option index (0-based)."""
    predicted = _extract_answer_choice(llm_output)
    if predicted is None:
        return False
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    return mapping.get(predicted.strip().upper()) == correct_idx


# ---------------------------------------------------------------------------
# LLM prompt builders
# ---------------------------------------------------------------------------

MEDRAG_SYSTEM_PROMPT = (
    "You are a helpful medical expert, and your task is to answer a multi-choice "
    "medical question using the relevant documents. Please first think step-by-step "
    "and then choose the answer from the provided options. "
    'Output JSON: {"step_by_step_thinking": "...", "answer_choice": "A/B/C/D"}'
)


def build_medrag_user_prompt(question: str, options: Dict[str, str], chunks: List[Any]) -> str:
    context = _format_chunks_medrag(chunks)
    options_str = "\n".join(f"{k}: {v}" for k, v in options.items())
    return (
        f"Here are the relevant documents:\n{context}\n\n"
        f"Here is the question: {question}\n"
        f"Here are the potential choices:\n{options_str}"
    )


def build_no_rag_user_prompt(question: str, options: Dict[str, str]) -> str:
    options_str = "\n".join(f"{k}: {v}" for k, v in options.items())
    return (
        f"Here is the question: {question}\n"
        f"Here are the potential choices:\n{options_str}"
    )


# ---------------------------------------------------------------------------
# Context truncation
# ---------------------------------------------------------------------------

def truncate_context(chunks: List[Any], tokenizer: Any, max_tokens: int) -> str:
    """Join chunks and truncate to max_tokens using the provided tokenizer."""
    texts = [_chunk_to_text(c) for c in chunks]
    joined = "\n".join(texts)
    encoded = tokenizer.encode(joined)[:max_tokens]
    return tokenizer.decode(encoded)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_answer_choice(llm_output: str) -> Optional[str]:
    """Extract answer letter from LLM JSON output string."""
    parsed = _parse_json_object(llm_output)
    if not isinstance(parsed, dict):
        return None

    ans = parsed.get("answer_choice")
    if not isinstance(ans, str):
        return None

    ans = ans.strip().upper()
    if re.fullmatch(r"[A-D]", ans):
        return ans

    return None


def _parse_json_object(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _format_chunks_medrag(chunks: List[Any]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        text = _chunk_to_text(chunk)
        parts.append(f"[{i}] {text}")
    return "\n".join(parts)


def _chunk_to_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        title = chunk.get("title", "")
        content = chunk.get("content", chunk.get("text", ""))
        return f"{title}: {content}" if title else content
    if isinstance(chunk, (list, tuple)) and len(chunk) == 2:
        return f"{chunk[0]}: {chunk[1]}"
    return str(chunk)
