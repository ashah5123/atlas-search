"""Cross-encoder reranker using sentence_transformers.CrossEncoder."""

from __future__ import annotations

from typing import List

import torch

try:
    from sentence_transformers import CrossEncoder
except ImportError as exc:
    raise ImportError(
        "CrossEncoderReranker requires 'sentence-transformers'. "
        "Install with: pip install sentence-transformers"
    ) from exc


def _select_device(device: str | None) -> str:
    """Prefer mps > cuda > cpu when device is not specified."""
    if device is not None:
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class CrossEncoderReranker:
    """Rerank candidates using a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device_str = _select_device(device)
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=self.device_str)

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        text_key: str = "passage",
    ) -> List[dict]:
        """Score (query, passage) pairs in batches, add cross_score, return sorted by score descending."""
        if not candidates:
            return []

        texts = [c.get(text_key, "") for c in candidates]
        pairs = [(query, t) for t in texts]

        all_scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            scores = self._model.predict(batch, convert_to_numpy=True)
            if scores.ndim == 0:
                scores = [float(scores)]
            else:
                scores = scores.tolist()
            all_scores.extend(scores)

        out = []
        for cand, score in zip(candidates, all_scores):
            c = dict(cand)
            c["cross_score"] = float(score)
            out.append(c)

        out.sort(key=lambda x: x["cross_score"], reverse=True)
        return out
