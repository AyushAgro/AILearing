"""DDPM training, ancestral sampling, DDIM and classifier-free guidance, from scratch.

Conventions follow Ho et al. (2020): t = 0 .. T-1, alpha_bar decreasing, the network
predicts the noise eps that was mixed into x_0. Time is fed to the net as t / T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forge.multimodal.gen.nets import EMA, MLP, features


@dataclass
class Schedule:
    betas: np.ndarray
    alphas: np.ndarray
    alpha_bar: np.ndarray

    @property
    def T(self) -> int:
        return len(self.betas)


def linear_schedule(
    T: int = 1000, beta_1: float = 1e-4, beta_T: float = 0.02
) -> Schedule:
    betas = np.linspace(beta_1, beta_T, T)
    alphas = 1.0 - betas
    return Schedule(betas, alphas, np.cumprod(alphas))


def cosine_schedule(T: int = 1000, s: float = 0.008) -> Schedule:
    """Nichol and Dhariwal (2021): alpha_bar follows a squared cosine."""
    steps = np.arange(T + 1) / T
    f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f[1:] / f[0]
    betas = np.clip(1.0 - alpha_bar / np.concatenate([[1.0], alpha_bar[:-1]]), 0, 0.999)
    alphas = 1.0 - betas
    return Schedule(betas, alphas, np.cumprod(alphas))


def q_sample(
    x0: np.ndarray, t: np.ndarray, sched: Schedule, eps: np.ndarray
) -> np.ndarray:
    """Closed-form forward process: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps."""
    ab = sched.alpha_bar[t][:, None]
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps


def train_ddpm(
    data: np.ndarray,
    sched: Schedule,
    rng: np.random.Generator,
    *,
    steps: int = 3000,
    batch: int = 256,
    hidden: int = 128,
    labels: np.ndarray | None = None,
    n_classes: int = 0,
    p_uncond: float = 0.1,
    lr: float = 2e-3,
) -> tuple[MLP, list[float]]:
    """Train an eps-prediction MLP. With labels, drop them with prob p_uncond (CFG)."""
    dim = data.shape[1]
    net = MLP(
        [dim + 16 + (n_classes + 1 if n_classes else 0), hidden, hidden, hidden, dim],
        rng,
        lr=lr,
    )
    ema = EMA(net)
    losses = []
    for _ in range(steps):
        idx = rng.integers(0, len(data), size=batch)
        x0 = data[idx]
        t = rng.integers(0, sched.T, size=batch)
        eps = rng.normal(size=x0.shape)
        xt = q_sample(x0, t, sched, eps)
        y = None
        if n_classes:
            y = labels[idx].copy()
            y[rng.random(batch) < p_uncond] = n_classes  # the null condition
        losses.append(net.mse_step(features(xt, t / sched.T, y, n_classes), eps))
        ema.update(net)
    return ema.net, losses


def predict_eps(
    net: MLP,
    x: np.ndarray,
    t: int,
    T: int,
    labels=None,
    n_classes: int = 0,
    guidance: float = 1.0,
) -> np.ndarray:
    """eps prediction, with classifier-free guidance batched into one forward."""
    n = len(x)
    tt = np.full(n, t / T)
    if labels is None or guidance == 1.0:
        return net(features(x, tt, labels, n_classes))
    null = np.full(n, n_classes)
    both = features(
        np.concatenate([x, x]),
        np.concatenate([tt, tt]),
        np.concatenate([labels, null]),
        n_classes,
    )
    eps_c, eps_u = np.split(net(both), 2)
    return eps_u + guidance * (eps_c - eps_u)


def ddpm_sample(
    net: MLP,
    sched: Schedule,
    rng: np.random.Generator,
    n: int,
    dim: int = 2,
    labels=None,
    n_classes: int = 0,
    guidance: float = 1.0,
) -> np.ndarray:
    """Ancestral sampling: one net call per training step, fresh noise each step."""
    x = rng.normal(size=(n, dim))
    for t in range(sched.T - 1, -1, -1):
        eps = predict_eps(net, x, t, sched.T, labels, n_classes, guidance)
        coef = sched.betas[t] / np.sqrt(1.0 - sched.alpha_bar[t])
        mean = (x - coef * eps) / np.sqrt(sched.alphas[t])
        x = mean + (
            np.sqrt(sched.betas[t]) * rng.normal(size=x.shape) if t > 0 else 0.0
        )
    return x


def ddim_sample(
    net: MLP,
    sched: Schedule,
    rng: np.random.Generator,
    n: int,
    steps: int,
    dim: int = 2,
    eta: float = 0.0,
    labels=None,
    n_classes: int = 0,
    guidance: float = 1.0,
    x_T: np.ndarray | None = None,
    t_start: int | None = None,
) -> np.ndarray:
    """DDIM (Song et al., 2020) on a sub-grid of `steps` timesteps.

    eta = 0 is deterministic; eta = 1 recovers DDPM-like stochasticity.
    t_start < T starts from a partially noised x_T (used by SDEdit in Chapter 7).
    """
    top = sched.T - 1 if t_start is None else t_start
    grid = np.unique(np.linspace(0, top, steps).round().astype(int))[::-1]
    x = rng.normal(size=(n, dim)) if x_T is None else x_T.copy()
    for i, t in enumerate(grid):
        ab = sched.alpha_bar[t]
        ab_prev = sched.alpha_bar[grid[i + 1]] if i + 1 < len(grid) else 1.0
        eps = predict_eps(net, x, int(t), sched.T, labels, n_classes, guidance)
        x0 = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)
        sigma = eta * np.sqrt((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev))
        dir_xt = np.sqrt(np.maximum(1 - ab_prev - sigma**2, 0.0)) * eps
        x = np.sqrt(ab_prev) * x0 + dir_xt
        if eta > 0 and i + 1 < len(grid):
            x = x + sigma * rng.normal(size=x.shape)
    return x
