"""A from-scratch MLP with SiLU, Adam and EMA: the denoiser used across Volume 20.

Everything is NumPy so the whole volume runs at Tier L. The network is a stand-in for
the U-Net or DiT of a production system: same inputs (noisy sample, time, condition),
same output (noise, velocity or clean sample), far fewer parameters.
"""

from __future__ import annotations

import copy

import numpy as np


def silu(z: np.ndarray) -> np.ndarray:
    return z / (1.0 + np.exp(-z))


def silu_grad(z: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-z))
    return s * (1.0 + z * (1.0 - s))


class MLP:
    """Fully connected net. SiLU on hidden layers, linear output."""

    def __init__(self, sizes: list[int], rng: np.random.Generator, lr: float = 2e-3):
        self.params: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.params.append(
                rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            )
            self.params.append(np.zeros(fan_out))
        self.params[-2] *= 0.1  # small last layer: start near a zero prediction
        self.lr = lr
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list]:
        cache = []
        h = x
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            w, b = self.params[2 * i], self.params[2 * i + 1]
            z = h @ w + b
            cache.append((h, z))
            h = z if i == n_layers - 1 else silu(z)
        return h, cache

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def backward(self, cache: list, grad: np.ndarray) -> tuple[list, np.ndarray]:
        """Return parameter gradients and the gradient with respect to the input."""
        grads: list[np.ndarray] = [np.empty(0)] * len(self.params)
        n_layers = len(cache)
        for i in range(n_layers - 1, -1, -1):
            h, z = cache[i]
            if i != n_layers - 1:
                grad = grad * silu_grad(z)
            grads[2 * i] = h.T @ grad
            grads[2 * i + 1] = grad.sum(axis=0)
            grad = grad @ self.params[2 * i].T
        return grads, grad

    def adam_step(
        self,
        grads: list,
        clip: float = 1.0,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        norm = np.sqrt(sum(float((g**2).sum()) for g in grads))
        scale = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / (1 - b1**self.t)
            v_hat = v / (1 - b2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def mse_step(self, x: np.ndarray, target: np.ndarray) -> float:
        """One Adam step on mean squared error. Returns the loss before the step."""
        pred, cache = self.forward(x)
        err = pred - target
        grads, _ = self.backward(cache, 2.0 * err / err.size)
        self.adam_step(grads)
        return float(np.mean(err**2))


class EMA:
    """Exponential moving average of weights. Sample from this copy, not the raw net."""

    def __init__(self, net: MLP, decay: float = 0.995):
        self.decay = decay
        self.net = copy.deepcopy(net)

    def update(self, net: MLP) -> None:
        for pe, p in zip(self.net.params, net.params):
            pe *= self.decay
            pe += (1.0 - self.decay) * p


def time_features(t: np.ndarray, dim: int = 16) -> np.ndarray:
    """Sinusoidal features of a continuous time t in [0, 1]. Shape (N, dim)."""
    half = dim // 2
    freqs = np.exp(np.linspace(0.0, np.log(200.0), half))  # 1 .. 200 rad per unit t
    ang = np.asarray(t, dtype=np.float64).reshape(-1, 1) * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def one_hot(labels: np.ndarray | None, n_classes: int, n: int) -> np.ndarray:
    """One-hot with an extra 'null' slot at index n_classes (the dropped condition)."""
    if n_classes == 0:
        return np.zeros((n, 0))
    out = np.zeros((n, n_classes + 1))
    idx = np.full(n, n_classes) if labels is None else np.asarray(labels)
    out[np.arange(n), idx] = 1.0
    return out


def features(
    x: np.ndarray,
    t: np.ndarray,
    labels: np.ndarray | None = None,
    n_classes: int = 0,
    t_dim: int = 16,
) -> np.ndarray:
    """Denoiser input: [x_t, time features, condition one-hot]."""
    n = len(x)
    return np.concatenate(
        [x, time_features(t, t_dim), one_hot(labels, n_classes, n)], axis=1
    )
