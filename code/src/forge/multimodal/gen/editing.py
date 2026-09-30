"""Editing with a flow model: SDEdit-style partial noising and replacement inpainting.
Flow convention: t = 0 noise, t = 1 data."""

from __future__ import annotations

import numpy as np

from forge.multimodal.gen.flowmatch import sample_euler, velocity
from forge.multimodal.gen.nets import MLP


def sdedit(
    net: MLP,
    z_src: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    steps: int,
    labels=None,
    n_classes: int = 0,
    guidance: float = 1.0,
) -> np.ndarray:
    """Move the source back to t0 = 1 - strength along the straight path, then
    integrate forward under the (new) condition. strength 0 = copy, 1 = new sample."""
    t0 = 1.0 - strength
    noise = rng.normal(size=z_src.shape)
    z_t0 = (1.0 - t0) * noise + t0 * z_src
    n_steps = max(1, int(round(steps * strength)))
    return sample_euler(net, z_t0, n_steps, labels, n_classes, guidance, t0=t0)


def inpaint(
    net: MLP,
    x_src: np.ndarray,
    known: np.ndarray,
    rng: np.random.Generator,
    steps: int,
    labels=None,
    n_classes: int = 0,
    guidance: float = 1.0,
) -> np.ndarray:
    """Replacement inpainting: after every step, overwrite known pixels with the
    source pushed along the same path. known: boolean mask, True = keep."""
    noise = rng.normal(size=x_src.shape)
    x = noise.copy()
    ts = np.linspace(0.0, 1.0, steps + 1)
    for a, b in zip(ts[:-1], ts[1:]):
        x = x + (b - a) * velocity(net, x, a, labels, n_classes, guidance)
        x_known = (1.0 - b) * noise + b * x_src  # source at time b
        x = np.where(known, x_known, x)
    return x
