"""Video generation probes: temporally correlated noise, flicker and identity metrics,
and the spacetime-token arithmetic that sets the bill."""

from __future__ import annotations

import numpy as np


def correlated_noise(
    rng: np.random.Generator, frames: int, n: int, dim: int, rho: float
) -> np.ndarray:
    """AR(1) noise across frames. Each frame is still N(0, I); neighbours correlate rho.
    Shape (frames, n, dim)."""
    z = np.empty((frames, n, dim))
    z[0] = rng.normal(size=(n, dim))
    for f in range(1, frames):
        z[f] = rho * z[f - 1] + np.sqrt(1 - rho**2) * rng.normal(size=(n, dim))
    return z


def flicker(clip: np.ndarray) -> float:
    """Mean RMS change between consecutive frames. clip: (frames, n, pixels)."""
    return float(np.sqrt(np.mean((clip[1:] - clip[:-1]) ** 2)))


def identity_consistency(labels_per_frame: np.ndarray) -> float:
    """Share of frames whose predicted identity matches frame 0. (frames, n)."""
    return float(np.mean(labels_per_frame == labels_per_frame[0:1]))


def spacetime_tokens(
    seconds: float,
    fps: int,
    height: int,
    width: int,
    *,
    vae_t: int = 4,
    vae_s: int = 8,
    patch: int = 2,
) -> int:
    """Tokens a video DiT attends over after a (vae_t x vae_s x vae_s) autoencoder
    and (1 x patch x patch) patchify. Illustrative factors, not any product's."""
    frames = int(seconds * fps)
    lt = 1 + (frames - 1) // vae_t
    return lt * (height // vae_s // patch) * (width // vae_s // patch)


def attention_flops(tokens: int, d_model: int, layers: int) -> float:
    """Forward FLOPs of attention scores and value products only: 4 N^2 d per layer."""
    return 4.0 * tokens**2 * d_model * layers
