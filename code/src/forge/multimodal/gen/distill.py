"""One-step distillation: regress a student onto a multi-step teacher's ODE map
(the idea behind knowledge distillation for samplers; consistency models refine it)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from forge.multimodal.gen.nets import MLP


def distill_one_step(
    teacher: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    *,
    dim: int = 2,
    n_pairs: int = 20000,
    steps: int = 3000,
    batch: int = 256,
    hidden: int = 128,
) -> tuple[MLP, list[float]]:
    """teacher maps noise x_T to a sample deterministically (e.g. DDIM, eta = 0)."""
    z = rng.normal(size=(n_pairs, dim))
    x = teacher(z)  # the expensive part, done once
    student = MLP([dim, hidden, hidden, hidden, dim], rng)
    losses = []
    for _ in range(steps):
        i = rng.integers(0, n_pairs, size=batch)
        losses.append(student.mse_step(z[i], x[i]))
    return student, losses
