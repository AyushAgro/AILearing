"""Tiny latent diffusion on 8x8 digits: an autoencoder, a latent scale factor, a
classifier probe and a feature-space Fréchet distance (a stand-in for FID)."""

from __future__ import annotations

import numpy as np

from forge.multimodal.gen.metrics import frechet_gaussian
from forge.multimodal.gen.nets import MLP


def load_digits_split(seed: int = 0, n_test: int = 297):
    """sklearn's 1,797 8x8 digits, scaled to [-1, 1], split into train and test."""
    from sklearn.datasets import load_digits

    d = load_digits()
    x = d.data / 8.0 - 1.0  # (1797, 64) in [-1, 1]
    order = np.random.default_rng(seed).permutation(len(x))
    x, y = x[order], d.target[order]
    return (x[n_test:], y[n_test:]), (x[:n_test], y[:n_test])


class Autoencoder:
    """Deterministic MLP autoencoder with a tiny L2 penalty on the code."""

    def __init__(
        self,
        rng: np.random.Generator,
        dim: int = 64,
        latent: int = 8,
        hidden: int = 256,
        lr: float = 1e-3,
    ):
        self.enc = MLP([dim, hidden, hidden, latent], rng, lr=lr)
        self.dec = MLP([latent, hidden, hidden, dim], rng, lr=lr)
        self.scale = 1.0  # set after training, like SD's vae.config.scaling_factor

    def fit(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        steps: int = 4000,
        batch: int = 128,
        l2: float = 1e-3,
    ) -> list[float]:
        hist = []
        for _ in range(steps):
            xb = x[rng.integers(0, len(x), size=batch)]
            z, c_e = self.enc.forward(xb)
            xh, c_d = self.dec.forward(z)
            err = xh - xb
            hist.append(float(np.mean(err**2)))
            g_d, g_z = self.dec.backward(c_d, 2 * err / err.size)
            g_e, _ = self.enc.backward(c_e, g_z + 2 * l2 * z / z.size)
            self.dec.adam_step(g_d)
            self.enc.adam_step(g_e)
        self.scale = 1.0 / float(self.enc(x).std())
        return hist

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.enc(x) * self.scale  # unit-variance latents

    def decode(self, z: np.ndarray) -> np.ndarray:
        return np.clip(self.dec(z / self.scale), -1.0, 1.0)


class Probe:
    """A small digit classifier. Its accuracy on generated digits is the
    'does it match the prompt' score; its hidden layer is the feature space for FD."""

    def __init__(self, rng: np.random.Generator, dim: int = 64, hidden: int = 64):
        self.net = MLP([dim, hidden, 10], rng, lr=2e-3)

    def fit(self, x, y, rng, steps: int = 3000, batch: int = 128) -> "Probe":
        for _ in range(steps):
            i = rng.integers(0, len(x), size=batch)
            logits, cache = self.net.forward(x[i])
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            p[np.arange(batch), y[i]] -= 1.0
            grads, _ = self.net.backward(cache, p / batch)
            self.net.adam_step(grads)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.net(x).argmax(1)

    def feats(self, x: np.ndarray) -> np.ndarray:
        _, cache = self.net.forward(x)
        return cache[-1][0]  # hidden activations (N, 64)


def feature_frechet(probe: Probe, a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between Gaussians fitted to probe features of a and b."""
    fa, fb = probe.feats(a), probe.feats(b)
    return frechet_gaussian(
        fa.mean(0), np.cov(fa, rowvar=False), fb.mean(0), np.cov(fb, rowvar=False)
    )
