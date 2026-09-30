"""The other generative families on the same 2-D data: VAE, GAN and an autoregressive
histogram model. Each is the smallest honest version of its idea."""

from __future__ import annotations

import numpy as np

from forge.multimodal.gen.nets import MLP


def softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train_vae(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    zdim: int = 2,
    steps: int = 3000,
    batch: int = 256,
    sigma: float = 0.1,
    beta: float = 1.0,
) -> tuple[MLP, list[float]]:
    """Gaussian VAE (Kingma and Welling, 2013). Returns the decoder and the -ELBO."""
    enc = MLP([2, 64, 64, 2 * zdim], rng)
    dec = MLP([zdim, 64, 64, 2], rng)
    history = []
    for _ in range(steps):
        x = data[rng.integers(0, len(data), size=batch)]
        h, c_enc = enc.forward(x)
        mu, logvar = h[:, :zdim], np.clip(h[:, zdim:], -10, 10)
        e = rng.normal(size=mu.shape)
        z = mu + np.exp(0.5 * logvar) * e  # reparameterisation
        xhat, c_dec = dec.forward(z)
        recon = ((xhat - x) ** 2).sum(1) / (2 * sigma**2)
        kl = 0.5 * (np.exp(logvar) + mu**2 - 1.0 - logvar).sum(1)
        history.append(float((recon + beta * kl).mean()))
        g_dec, g_z = dec.backward(c_dec, (xhat - x) / sigma**2 / batch)
        g_mu = g_z + beta * mu / batch
        g_lv = (
            g_z * e * 0.5 * np.exp(0.5 * logvar)
            + beta * 0.5 * (np.exp(logvar) - 1) / batch
        )
        g_enc, _ = enc.backward(c_enc, np.concatenate([g_mu, g_lv], axis=1))
        dec.adam_step(g_dec)
        enc.adam_step(g_enc)
    return dec, history


def train_gan(
    data: np.ndarray, rng: np.random.Generator, *, steps: int = 3000, batch: int = 256
) -> tuple[MLP, list[tuple[float, float]]]:
    """Non-saturating GAN (Goodfellow et al., 2014). Returns the generator."""
    gen = MLP([2, 64, 64, 2], rng, lr=1e-3)
    disc = MLP([2, 64, 64, 1], rng, lr=1e-3)
    history = []
    for _ in range(steps):
        real = data[rng.integers(0, len(data), size=batch)]
        fake, c_gen = gen.forward(rng.normal(size=(batch, 2)))
        # discriminator: push real logits up, fake logits down
        both = np.concatenate([real, fake])
        logit, c_d = disc.forward(both)
        target = np.concatenate([np.ones((batch, 1)), np.zeros((batch, 1))])
        d_loss = float(np.mean(softplus(logit) - target * logit))
        g_d, _ = disc.backward(c_d, (sigmoid(logit) - target) / (2 * batch))
        disc.adam_step(g_d, b1=0.5)
        # generator: make D call the fakes real; gradient flows through D to x
        logit_f, c_df = disc.forward(fake)
        g_loss = float(np.mean(softplus(-logit_f)))
        _, g_x = disc.backward(c_df, (sigmoid(logit_f) - 1.0) / batch)
        g_g, _ = gen.backward(c_gen, g_x)
        gen.adam_step(g_g, b1=0.5)
        history.append((d_loss, g_loss))
    return gen, history


class HistogramAR:
    """p(x1) p(x2 | x1) on a 64 x 64 grid: an autoregressive model whose 'network'
    is a table of counts. Exact likelihood, two sequential 'calls' per sample."""

    def __init__(self, bins: int = 64, lo: float = -3.0, hi: float = 3.0):
        self.bins, self.lo, self.hi = bins, lo, hi
        self.width = (hi - lo) / bins

    def _idx(self, v: np.ndarray) -> np.ndarray:
        return np.clip(((v - self.lo) / self.width).astype(int), 0, self.bins - 1)

    def fit(self, data: np.ndarray, alpha: float = 0.01) -> "HistogramAR":
        i, j = self._idx(data[:, 0]), self._idx(data[:, 1])
        counts = np.full((self.bins, self.bins), alpha)
        np.add.at(counts, (i, j), 1.0)
        self.p1 = counts.sum(1) / counts.sum()
        self.p2_given_1 = counts / counts.sum(1, keepdims=True)
        return self

    def nll(self, x: np.ndarray) -> float:
        """Mean negative log density in nats per point (bin mass / bin area)."""
        i, j = self._idx(x[:, 0]), self._idx(x[:, 1])
        logp = (
            np.log(self.p1[i]) + np.log(self.p2_given_1[i, j]) - 2 * np.log(self.width)
        )
        return float(-logp.mean())

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        i = rng.choice(self.bins, size=n, p=self.p1)  # token 1
        j = np.array(
            [rng.choice(self.bins, p=self.p2_given_1[k]) for k in i]
        )  # token 2
        u = rng.uniform(size=(n, 2))
        return self.lo + (np.stack([i, j], 1) + u) * self.width
