"""Conditional flow matching on the same circle, integrated with Euler."""
from __future__ import annotations

import numpy as np

from forge.multimodal.gen.diffuse import circle, radius_error
from forge.multimodal.gen.schedules import time_embed
from forge.multimodal.nn import MLP


def train_velocity(rng: np.random.Generator, *, steps: int = 400, batch: int = 128,
                   lr: float = 2e-2) -> tuple[MLP, list[float]]:
    net = MLP(rng, [2 + 16, 64, 64, 2])
    losses: list[float] = []
    data = circle(rng, 2048)
    for _ in range(steps):
        x1 = data[rng.integers(0, len(data), size=batch)]
        x0 = rng.normal(size=x1.shape)
        t = rng.uniform(0.0, 1.0, size=(batch, 1))
        xt = (1.0 - t) * x0 + t * x1
        target = x1 - x0
        # Time embedding expects an integer index. Scale t into 0..999.
        features = np.concatenate([xt, time_embed((t[:, 0] * 999).astype(int), 16)], axis=1)
        losses.append(net.step_loss(features, target, lr))
    return net, losses


def sample_euler(net: MLP, rng: np.random.Generator, n: int, steps: int) -> np.ndarray:
    x = rng.normal(size=(n, 2))
    dt = 1.0 / steps
    for i in range(steps):
        t_value = int(min(999, round(i / steps * 999)))
        t = np.full(n, t_value)
        features = np.concatenate([x, time_embed(t, 16)], axis=1)
        velocity, _ = net.forward(features)
        x = x + dt * velocity
    return x


def cfg_combine(uncond: np.ndarray, cond: np.ndarray, scale: float) -> np.ndarray:
    """Classifier-free guidance: extrapolate from the unconditional prediction."""
    return uncond + scale * (cond - uncond)


__all__ = ["train_velocity", "sample_euler", "cfg_combine", "radius_error"]
