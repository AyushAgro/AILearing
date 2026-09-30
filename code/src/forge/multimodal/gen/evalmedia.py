"""Metrics for generated media: sample-based Fréchet distance, k-NN precision and
recall, CLIPScore, Bradley-Terry preference fits and word error rate."""

from __future__ import annotations

import numpy as np

from forge.multimodal.gen.metrics import frechet_gaussian


def frechet_from_samples(a: np.ndarray, b: np.ndarray) -> float:
    return frechet_gaussian(
        a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False)
    )


def knn_precision_recall(
    real: np.ndarray, fake: np.ndarray, k: int = 3
) -> tuple[float, float]:
    """Kynkäänniemi et al. (2019). Precision: share of fakes inside some real point's
    k-NN ball. Recall: share of reals inside some fake point's k-NN ball."""

    def radii(x):
        d = np.linalg.norm(x[:, None] - x[None], axis=-1)
        return np.sort(d, axis=1)[:, k]  # column 0 is self

    def inside(points, centres, r):
        d = np.linalg.norm(points[:, None] - centres[None], axis=-1)
        return float(np.mean((d <= r[None, :]).any(axis=1)))

    return inside(fake, real, radii(real)), inside(real, fake, radii(fake))


def clip_score(
    image_emb: np.ndarray, text_emb: np.ndarray, w: float = 2.5
) -> np.ndarray:
    """Hessel et al. (2021): w * max(cos(E_I, E_T), 0) per pair. torchmetrics: w=100."""
    a = image_emb / np.linalg.norm(image_emb, axis=-1, keepdims=True)
    b = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)
    return w * np.maximum((a * b).sum(-1), 0.0)


def bradley_terry(wins: np.ndarray, iters: int = 200) -> np.ndarray:
    """MM fit of Bradley-Terry strengths from a win matrix wins[i, j] = i beat j.
    Returns log-strengths centred at 0."""
    n = len(wins)
    p = np.ones(n)
    games = wins + wins.T
    for _ in range(iters):
        denom = np.array(
            [
                sum(games[i, j] / (p[i] + p[j]) for j in range(n) if j != i)
                for i in range(n)
            ]
        )
        p = wins.sum(1) / np.maximum(denom, 1e-12)
        p /= np.exp(np.mean(np.log(p)))
    return np.log(p)


def wer(ref: str, hyp: str) -> float:
    """Word error rate: word-level Levenshtein distance / reference length."""
    r, h = ref.split(), hyp.split()
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d = d, np.empty_like(d)
        d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[-1] / max(len(r), 1))
