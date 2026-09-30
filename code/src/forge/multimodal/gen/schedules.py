"""Noise schedules shared by DDPM and DDIM."""
from __future__ import annotations

import numpy as np


def linear_schedule(steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    """The DDPM linear beta schedule. Needs hundreds of steps to reach pure noise."""
    betas = np.linspace(beta_start, beta_end, steps)
    alphas = 1.0 - betas
    return {"betas": betas, "alphas": alphas, "alpha_bar": np.cumprod(alphas)}


def toy_schedule(steps: int) -> dict:
    """Short grid whose cumulative noise still reaches alpha_bar = 0.02."""
    alpha_bar = np.linspace(0.99, 0.02, steps)
    alphas = np.empty(steps)
    alphas[0] = alpha_bar[0]
    alphas[1:] = alpha_bar[1:] / alpha_bar[:-1]
    betas = 1.0 - alphas
    return {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}


def q_sample(x0: np.ndarray, t: np.ndarray, alpha_bar: np.ndarray,
             noise: np.ndarray) -> np.ndarray:
    a = alpha_bar[t][:, None]
    return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * noise


def time_embed(t: np.ndarray, dim: int) -> np.ndarray:
    half = dim // 2
    freqs = np.exp(-np.log(10_000.0) * np.arange(half) / half)
    angles = t.astype(np.float64)[:, None] * freqs[None, :]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1).astype(np.float64)
