from __future__ import annotations

from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.config import (
    BGE_ENCODER,
    MEDCPT_ARTICLE_ENCODER,
    MEDCPT_QUERY_ENCODER,
)


class EmbeddingModel:
    """Wraps MedCPT (medrag) and BGE (wikipedia) encoders with CLS pooling.

    medrag query:   MedCPT-Query-Encoder   + CLS pooling
    medrag article: MedCPT-Article-Encoder + CLS pooling
    wikipedia:      BAAI/bge-large-en-v1.5 + CLS pooling (single encoder)

    DEVIATION FROM ORIGINAL: original uses CustomizeSentenceTransformer + Cohere Embed V3.
    This impl uses HuggingFace AutoModel + CLS pooling.
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
            self._query_tokenizer = AutoTokenizer.from_pretrained(BGE_ENCODER)
            self._query_model = AutoModel.from_pretrained(
                BGE_ENCODER, torch_dtype=torch.float16
            ).to(device).eval()

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

        else:  # wikipedia — BGE, single encoder for both queries and passages
            encoded = self._query_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._query_model(**encoded)

            # BGE uses CLS pooling (first token of last hidden state)
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings.cpu().float().numpy()
