"""Conditional flow matching and rectified flow, from scratch.

Convention (Lipman et al., 2022; Liu et al., 2022): t = 0 is noise, t = 1 is data,
x_t = (1 - t) x_0 + t x_1, and the regression target is the velocity x_1 - x_0.
Stable Diffusion 3 writes the same path with t reversed; the algebra is identical.
"""

from __future__ import annotations

import numpy as np

from forge.multimodal.gen.nets import EMA, MLP, features


def train_fm(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    steps: int = 3000,
    batch: int = 256,
    hidden: int = 128,
    labels: np.ndarray | None = None,
    n_classes: int = 0,
    p_uncond: float = 0.1,
    lr: float = 2e-3,
    pairs: tuple[np.ndarray, np.ndarray] | None = None,
    t_sampler: str = "uniform",
) -> tuple[MLP, list[float]]:
    """Regress v(x_t, t) onto x_1 - x_0.

    pairs=(x0, x1) trains on a fixed coupling (reflow) instead of independent pairs.
    t_sampler='logit_normal' concentrates t in the middle, as SD3 does.
    """
    dim = data.shape[1]
    net = MLP(
        [dim + 16 + (n_classes + 1 if n_classes else 0), hidden, hidden, hidden, dim],
        rng,
        lr=lr,
    )
    ema = EMA(net)
    losses = []
    for _ in range(steps):
        if pairs is None:
            idx = rng.integers(0, len(data), size=batch)
            x1 = data[idx]
            x0 = rng.normal(size=x1.shape)
        else:
            idx = rng.integers(0, len(pairs[0]), size=batch)
            x0, x1 = pairs[0][idx], pairs[1][idx]
        if t_sampler == "logit_normal":
            t = 1.0 / (1.0 + np.exp(-rng.normal(size=(batch, 1))))
        else:
            t = rng.uniform(size=(batch, 1))
        xt = (1.0 - t) * x0 + t * x1
        y = None
        if n_classes:
            y = labels[idx].copy()
            y[rng.random(batch) < p_uncond] = n_classes
        losses.append(net.mse_step(features(xt, t[:, 0], y, n_classes), x1 - x0))
        ema.update(net)
    return ema.net, losses


def velocity(
    net: MLP,
    x: np.ndarray,
    t: float,
    labels=None,
    n_classes: int = 0,
    guidance: float = 1.0,
) -> np.ndarray:
    n = len(x)
    tt = np.full(n, t)
    if labels is None or guidance == 1.0:
        return net(features(x, tt, labels, n_classes))
    null = np.full(n, n_classes)
    both = features(
        np.concatenate([x, x]),
        np.concatenate([tt, tt]),
        np.concatenate([labels, null]),
        n_classes,
    )
    v_c, v_u = np.split(net(both), 2)
    return v_u + guidance * (v_c - v_u)


def sample_euler(
    net: MLP,
    x0: np.ndarray,
    steps: int,
    labels=None,
    n_classes: int = 0,
    guidance: float = 1.0,
    t0: float = 0.0,
) -> np.ndarray:
    """Integrate dx/dt = v from t0 to 1 with `steps` equal Euler steps."""
    x = x0.copy()
    ts = np.linspace(t0, 1.0, steps + 1)
    for a, b in zip(ts[:-1], ts[1:]):
        x = x + (b - a) * velocity(net, x, a, labels, n_classes, guidance)
    return x


def sample_heun(
    net: MLP,
    x0: np.ndarray,
    steps: int,
    labels=None,
    n_classes: int = 0,
    guidance: float = 1.0,
) -> np.ndarray:
    """Second-order Heun: two net calls per step, except the last is still two."""
    x = x0.copy()
    ts = np.linspace(0.0, 1.0, steps + 1)
    for a, b in zip(ts[:-1], ts[1:]):
        v1 = velocity(net, x, a, labels, n_classes, guidance)
        x_pred = x + (b - a) * v1
        v2 = velocity(net, x_pred, b, labels, n_classes, guidance)
        x = x + (b - a) * 0.5 * (v1 + v2)
    return x


def straightness(net: MLP, x0: np.ndarray, steps: int = 100) -> float:
    """Mean squared deviation of the velocity from the chord (x_1 - x_0) along the
    trajectory. Zero means a straight path that one Euler step integrates exactly."""
    x = x0.copy()
    vs = []
    ts = np.linspace(0.0, 1.0, steps + 1)
    for a, b in zip(ts[:-1], ts[1:]):
        v = velocity(net, x, a)
        vs.append(v)
        x = x + (b - a) * v
    chord = x - x0
    return float(np.mean([np.mean((v - chord) ** 2) for v in vs]))
