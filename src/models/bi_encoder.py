"""Bi-encoder model built on HuggingFace Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class BiEncoderConfig:
    """Configuration for the BiEncoder."""

    model_name: str
    normalize: bool = True
    device: Optional[str] = None


class BiEncoder:
    """Bi-encoder for dense retrieval using separate query and passage encoders.

    This implementation uses CLS pooling on top of a HuggingFace Transformer
    backbone, with optional L2 normalization of the output embeddings.
    """

    def __init__(self, model_name: str, normalize: bool = True, device: Optional[str] = None) -> None:
        """Initialize tokenizer and encoders.

        Args:
            model_name: Name or path of a HuggingFace Transformer model.
            normalize: If True, L2-normalize embeddings.
            device: Explicit device string (e.g. "cpu", "cuda", "cuda:0").
                If None, uses "cuda" when available, otherwise "cpu".
        """
        self.config = BiEncoderConfig(model_name=model_name, normalize=normalize, device=device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.doc_encoder = AutoModel.from_pretrained(model_name)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.query_encoder.to(self.device)
        self.doc_encoder.to(self.device)
        self.query_encoder.eval()
        self.doc_encoder.eval()

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return int(self.query_encoder.config.hidden_size)

    def _encode(
        self,
        texts: List[str],
        batch_size: int,
        max_len: int,
        is_query: bool,
    ) -> np.ndarray:
        """Shared encoding logic for queries and passages."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        model = self.query_encoder if is_query else self.doc_encoder
        all_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                outputs = model(**encoded)
                # CLS pooling
                token_embeddings = outputs.last_hidden_state  # (B, L, H)
                cls_embeddings = token_embeddings[:, 0, :]  # (B, H)

                if self.config.normalize:
                    cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=-1)

                all_embeddings.append(cls_embeddings.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(self, texts: List[str], batch_size: int, max_len: int) -> np.ndarray:
        """Encode a batch of query texts into dense vectors.

        Args:
            texts: List of queries.
            batch_size: Batch size for encoding.
            max_len: Maximum sequence length for tokenization.

        Returns:
            NumPy array of shape (len(texts), dim) with dtype float32.
        """
        return self._encode(texts=texts, batch_size=batch_size, max_len=max_len, is_query=True)

    def encode_passages(self, texts: List[str], batch_size: int, max_len: int) -> np.ndarray:
        """Encode a batch of passage texts into dense vectors.

        Args:
            texts: List of passages.
            batch_size: Batch size for encoding.
            max_len: Maximum sequence length for tokenization.

        Returns:
            NumPy array of shape (len(texts), dim) with dtype float32.
        """
        return self._encode(texts=texts, batch_size=batch_size, max_len=max_len, is_query=False)

