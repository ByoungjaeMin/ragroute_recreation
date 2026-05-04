from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.config import LLM_CONFIG
from src.embedding_model import EmbeddingModel
from src.federated_retriever import FederatedRetriever
from src.rag_router import RAGRouter
from src.utils import (
    MEDRAG_SYSTEM_PROMPT,
    build_no_rag_user_prompt,
    truncate_context,
)


class RAGPipeline:
    """End-to-end pipeline: encode → route → retrieve → LLM → answer.

    Supports both RAGRoute mode (router selects sources) and No-RAG baseline
    (context=[], router skipped).
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        router: RAGRouter,
        retriever: FederatedRetriever,
        llm_config: Optional[Dict] = None,
    ):
        self.embedding_model = embedding_model
        self.router = router
        self.retriever = retriever
        self.llm_config = llm_config or LLM_CONFIG

        self._tokenizer: Optional[Any] = None  # lazy-loaded for context truncation

    def answer(
        self,
        question: str,
        options: Dict[str, str],
        no_rag: bool = False,
    ) -> Dict[str, Any]:
        """Run full pipeline for a single question.

        Returns:
            llm_output:          raw LLM response string
            selected_sources:    list of source_ids chosen by router
            n_chunks:            number of retrieved chunks
        """
        if no_rag:
            user_prompt = build_no_rag_user_prompt(question, options)
            llm_output = self._call_llm(MEDRAG_SYSTEM_PROMPT, user_prompt)
            return {
                "llm_output": llm_output,
                "selected_sources": [],
                "n_chunks": 0,
            }

        query_vec = self.embedding_model.encode_query(question)

        selected_sources = self.router.route(query_vec)

        chunks = self.retriever.retrieve(query_vec, selected_sources)

        context = self._build_context(chunks)
        if context:
            user_prompt = (
                f"Here are the relevant documents:\n{context}\n\n"
                f"Here is the question: {question}\n"
                f"Here are the potential choices:\n"
                + "\n".join(f"{k}: {v}" for k, v in options.items())
            )
        else:
            user_prompt = build_no_rag_user_prompt(question, options)

        llm_output = self._call_llm(MEDRAG_SYSTEM_PROMPT, user_prompt)

        return {
            "llm_output": llm_output,
            "selected_sources": [src.source_id for src in selected_sources],
            "n_chunks": len(chunks),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, chunks: List[Any]) -> str:
        if not chunks:
            return ""
        tokenizer = self._get_tokenizer()
        max_tokens = self.llm_config["docs_context_length"]
        return truncate_context(chunks, tokenizer, max_tokens)

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            hf_name = self.llm_config["hf_name"]
            self._tokenizer = AutoTokenizer.from_pretrained(hf_name)
        return self._tokenizer

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model=self.llm_config["ollama_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.message.content
