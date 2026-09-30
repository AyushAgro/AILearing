"""Fréchet distance, cosine score, latent autoencoder, vector quantiser, SDEdit."""
from __future__ import annotations

import numpy as np

from forge.multimodal.nn import MLP


def frechet_gaussian(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray,
                     cov2: np.ndarray) -> float:
    """Closed-form Fréchet distance between two Gaussians.

    FID is this distance on Inception features. Here the inputs are the
    Gaussians themselves, so the number is exact and small.
    """
    diff = mu1 - mu2
    svals, svecs = np.linalg.eigh(cov1)
    sqrt_c1 = (svecs * np.sqrt(np.maximum(svals, 0.0))) @ svecs.T
    mid = sqrt_c1 @ cov2 @ sqrt_c1
    mvals, mvecs = np.linalg.eigh(mid)
    root = (mvecs * np.sqrt(np.maximum(mvals, 0.0))) @ mvecs.T
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * root))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def train_autoencoder(rng: np.random.Generator, latent: int, *, steps: int = 300,
                      lr: float = 1e-2) -> tuple[float, float]:
    """Data live on a 2-D subspace of R^8. Compare a latent of `latent` dims."""
    basis = rng.normal(size=(8, 2))
    z = rng.normal(size=(512, 2))
    data = z @ basis.T
    encoder = MLP(rng, [8, 16, latent])
    decoder = MLP(rng, [latent, 16, 8])
    final = 0.0
    for _ in range(steps):
        idx = rng.integers(0, len(data), size=64)
        batch = data[idx]
        code, enc_caches = encoder.forward(batch)
        recon, dec_caches = decoder.forward(code)
        err = recon - batch
        final = float(np.mean(err ** 2))
        grad_code = decoder.backward(dec_caches, 2.0 * err, lr)
        encoder.backward(enc_caches, grad_code, lr)
    code, _ = encoder.forward(data[:128])
    recon, _ = decoder.forward(code)
    return final, float(np.mean((recon - data[:128]) ** 2))


def vector_quantise(frames: np.ndarray, codebook: np.ndarray) -> tuple[np.ndarray, float]:
    # frames (N, D), codebook (K, D)
    distances = ((frames[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=-1)
    idx = distances.argmin(axis=1)
    recon = codebook[idx]
    mse = float(np.mean((recon - frames) ** 2))
    return idx, mse


def sdedit_correlation(x: np.ndarray, alpha_bar_t: float, rng: np.random.Generator) -> float:
    """Correlation between x and a noised copy at noise level alpha_bar_t."""
    noise = rng.normal(size=x.shape)
    xt = np.sqrt(alpha_bar_t) * x + np.sqrt(1.0 - alpha_bar_t) * noise
    return cosine(x.ravel(), xt.ravel())
