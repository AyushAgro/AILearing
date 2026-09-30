"""Image embedding index, recall@k, and ColPali-style late interaction (MaxSim)."""
from __future__ import annotations

import numpy as np


def normalise(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)


class ExactIndex:
    """Brute-force cosine index. The oracle every approximate index is measured against."""

    def __init__(self, vectors: np.ndarray, labels: list[str] | None = None):
        self.vectors = normalise(vectors.astype(np.float32))          # (N, D)
        self.labels = labels or [str(i) for i in range(len(vectors))]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = normalise(queries.astype(np.float32)) @ self.vectors.T   # (Q, N)
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, 1), axis=1)
        ids = np.take_along_axis(top, order, 1)
        return ids, np.take_along_axis(scores, ids, 1)


def recall_at_k(ids: np.ndarray, relevant: list[list[int]], k: int) -> float:
    """Fraction of queries with at least one relevant item in the top k."""
    hits = [bool(set(row[:k]) & set(rel)) for row, rel in zip(ids, relevant)]
    return float(np.mean(hits))


def maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """Late interaction: each query vector takes its best-matching document vector,
    and the scores are summed. query (Q, D), doc (T, D), both L2-normalised."""
    return float((query_vecs @ doc_vecs.T).max(axis=1).sum())
