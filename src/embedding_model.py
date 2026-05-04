from __future__ import annotations

from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

from src.config import (
    DPR_QUESTION_ENCODER,
    MEDCPT_ARTICLE_ENCODER,
    MEDCPT_QUERY_ENCODER,
)


class EmbeddingModel:
    """Wraps MedCPT (medrag) and DPR (wikipedia) encoders with CLS / pooler pooling.

    medrag query:   MedCPT-Query-Encoder   + CLS pooling
    medrag article: MedCPT-Article-Encoder + CLS pooling
    wikipedia:      DPR-question-encoder   + pooler_output

    DEVIATION FROM ORIGINAL: original uses CustomizeSentenceTransformer.
    This impl uses HuggingFace AutoModel + explicit CLS token extraction.
    Behaviour (CLS pooling, float32 output) is identical.
    """

    def __init__(self, dataset: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        if dataset not in ("medrag", "wikipedia"):
            raise ValueError(f"Unknown dataset '{dataset}'. Expected 'medrag' or 'wikipedia'.")

        self.dataset = dataset
        self.device = device

        if dataset == "medrag":
            self._query_tokenizer = AutoTokenizer.from_pretrained(MEDCPT_QUERY_ENCODER)
            self._query_model = AutoModel.from_pretrained(MEDCPT_QUERY_ENCODER).to(device).eval()

            self._article_tokenizer = AutoTokenizer.from_pretrained(MEDCPT_ARTICLE_ENCODER)
            self._article_model = AutoModel.from_pretrained(MEDCPT_ARTICLE_ENCODER).to(device).eval()

        else:  # wikipedia
            self._query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(DPR_QUESTION_ENCODER)
            self._query_model = DPRQuestionEncoder.from_pretrained(DPR_QUESTION_ENCODER).to(device).eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query string. Returns float32 array of shape (768,)."""
        return self._encode_texts([text], mode="query")[0]

    def encode_batch(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Encode a list of texts. Returns float32 array of shape (N, 768)."""
        return self._encode_texts(texts, mode="query", batch_size=batch_size)

    def encode_articles(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Encode corpus article chunks (medrag only). Returns (N, 768) float32."""
        if self.dataset != "medrag":
            raise RuntimeError("encode_articles() is only available for dataset='medrag'.")
        return self._encode_texts(texts, mode="article", batch_size=batch_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_texts(
        self, texts: List[str], mode: str, batch_size: int = 128
    ) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            emb = self._encode_batch_single(batch, mode)
            all_embeddings.append(emb)

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def _encode_batch_single(self, texts: List[str], mode: str) -> np.ndarray:
        if self.dataset == "medrag":
            if mode == "query":
                tokenizer = self._query_tokenizer
                model = self._query_model
                max_length = 512
            else:
                tokenizer = self._article_tokenizer
                model = self._article_model
                max_length = 512

            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**encoded)

            # CLS pooling: first token of last hidden state
            embeddings = outputs.last_hidden_state[:, 0, :]

        else:  # wikipedia, mode is always "query"
            encoded = self._query_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._query_model(**encoded)

            # DPR uses pooler_output (not last_hidden_state CLS)
            embeddings = outputs.pooler_output

        return embeddings.cpu().float().numpy()
