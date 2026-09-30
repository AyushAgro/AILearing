"""DDPM and DDIM on 2-D points. A circle is the dataset; radius error is the probe."""
from __future__ import annotations

import numpy as np

from forge.multimodal.gen.schedules import q_sample, time_embed, toy_schedule
from forge.multimodal.nn import MLP


def circle(rng: np.random.Generator, n: int) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    return np.concatenate([np.cos(theta), np.sin(theta)], axis=1)


def radius_error(samples: np.ndarray) -> float:
    radius = np.linalg.norm(samples, axis=1)
    return float(np.mean(np.abs(radius - 1.0)))


def train_eps(rng: np.random.Generator, *, steps: int = 400, batch: int = 128,
              t_steps: int = 40, lr: float = 2e-2) -> tuple[MLP, dict, list[float]]:
    schedule = toy_schedule(t_steps)
    net = MLP(rng, [2 + 16, 64, 64, 2])
    losses: list[float] = []
    data = circle(rng, 2048)
    for step in range(steps):
        idx = rng.integers(0, len(data), size=batch)
        x0 = data[idx]
        t = rng.integers(0, t_steps, size=batch)
        noise = rng.normal(size=x0.shape)
        xt = q_sample(x0, t, schedule["alpha_bar"], noise)
        features = np.concatenate([xt, time_embed(t, 16)], axis=1)
        losses.append(net.step_loss(features, noise, lr))
    return net, schedule, losses


def _predict(net: MLP, x: np.ndarray, t_value: int, batch: int) -> np.ndarray:
    t = np.full(batch, t_value)
    features = np.concatenate([x, time_embed(t, 16)], axis=1)
    eps, _ = net.forward(features)
    return eps


def sample_ddpm(net: MLP, schedule: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    t_steps = len(schedule["alpha_bar"])
    x = rng.normal(size=(n, 2))
    alphas = schedule["alphas"]
    alpha_bar = schedule["alpha_bar"]
    betas = schedule["betas"]
    for t in range(t_steps - 1, -1, -1):
        eps = _predict(net, x, t, n)
        coef = betas[t] / np.sqrt(1.0 - alpha_bar[t])
        mean = (x - coef * eps) / np.sqrt(alphas[t])
        if t > 0:
            x = mean + np.sqrt(betas[t]) * rng.normal(size=x.shape)
        else:
            x = mean
    return x


def sample_ddim(net: MLP, schedule: dict, rng: np.random.Generator, n: int,
                steps: int) -> np.ndarray:
    """Deterministic DDIM (eta = 0) on a uniform subset of the training grid."""
    alpha_bar = schedule["alpha_bar"]
    t_steps = len(alpha_bar)
    grid = np.linspace(0, t_steps - 1, steps, dtype=int)
    x = rng.normal(size=(n, 2))
    for i, t in enumerate(grid[::-1]):
        eps = _predict(net, x, int(t), n)
        a_t = alpha_bar[int(t)]
        x0 = (x - np.sqrt(1.0 - a_t) * eps) / np.sqrt(a_t)
        prev = int(grid[::-1][i + 1]) if i + 1 < len(grid) else -1
        a_prev = 1.0 if prev < 0 else alpha_bar[prev]
        x = np.sqrt(a_prev) * x0 + np.sqrt(1.0 - a_prev) * eps
    return x
