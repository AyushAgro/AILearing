"""2-D toy distributions and the sample-quality probes used before any image exists."""

from __future__ import annotations

import numpy as np


def eight_gaussians(
    rng: np.random.Generator, n: int, radius: float = 2.0, std: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Eight modes on a ring. Returns points (N, 2) and the mode label of each."""
    labels = rng.integers(0, 8, size=n)
    angle = labels * (2.0 * np.pi / 8.0)
    centres = radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    return centres + std * rng.normal(size=(n, 2)), labels


def mode_centres(radius: float = 2.0) -> np.ndarray:
    angle = np.arange(8) * (2.0 * np.pi / 8.0)
    return radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def mode_report(samples: np.ndarray, radius: float = 2.0, std: float = 0.15) -> dict:
    """Precision: share of samples within 3 std of a mode. Coverage: modes with
    at least 2% of samples. Balance: KL(mode histogram || uniform) in nats."""
    centres = mode_centres(radius)
    dist = np.linalg.norm(samples[:, None, :] - centres[None, :, :], axis=-1)  # (N, 8)
    nearest = dist.argmin(axis=1)
    good = dist.min(axis=1) < 3.0 * std
    counts = np.bincount(nearest[good], minlength=8)
    freq = counts / max(counts.sum(), 1)
    kl = float(
        np.sum(np.where(freq > 0, freq * np.log(np.maximum(freq, 1e-12) * 8), 0))
    )
    return {
        "precision": float(good.mean()),
        "modes": int((freq >= 0.02).sum()),
        "kl_uniform": kl,
    }


def mmd_rbf(x: np.ndarray, y: np.ndarray, scales=(0.1, 0.5, 2.0)) -> float:
    """Biased MMD^2 with a sum of RBF kernels. Zero iff the samples match in law."""

    def k(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return sum(np.exp(-d2 / (2 * s * s)) for s in scales)

    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())
