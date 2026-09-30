"""Audio generation building blocks: mu-law, k-means, residual vector quantisation,
the MusicGen delay pattern, and an STFT with Griffin-Lim phase recovery."""

from __future__ import annotations

import numpy as np

SR = 16_000


def synth_voice(
    rng: np.random.Generator, seconds: float = 1.0, sr: int = SR
) -> np.ndarray:
    """Voiced-speech stand-in: gliding f0, 8 decaying harmonics, an envelope."""
    t = np.arange(int(seconds * sr)) / sr
    f0 = rng.uniform(100, 220) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(1, 4) * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    wave = sum(np.sin(k * phase) / k**1.2 for k in range(1, 9))
    env = 0.5 * (1 - np.cos(2 * np.pi * rng.uniform(2, 5) * t)) ** 0.5
    return 0.3 * env * wave + 0.003 * rng.normal(size=t.shape)


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(ref**2) / np.sum((ref - est) ** 2)))


def mu_law_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    y = np.sign(x) * np.log1p(mu * np.abs(np.clip(x, -1, 1))) / np.log1p(mu)
    return ((y + 1) / 2 * mu + 0.5).astype(np.int64)  # 0 .. mu


def mu_law_decode(q: np.ndarray, mu: int = 255) -> np.ndarray:
    y = 2 * q.astype(np.float64) / mu - 1
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


def kmeans(
    x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 20
) -> np.ndarray:
    cb = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(iters):
        d = (x**2).sum(1)[:, None] - 2 * x @ cb.T + (cb**2).sum(1)[None, :]
        a = d.argmin(1)
        for j in range(k):
            members = x[a == j]
            cb[j] = (
                members.mean(0) if len(members) else x[rng.integers(len(x))]
            )  # revive
    return cb


def nearest(x: np.ndarray, cb: np.ndarray) -> np.ndarray:
    d = (x**2).sum(1)[:, None] - 2 * x @ cb.T + (cb**2).sum(1)[None, :]
    return d.argmin(1)


class ResidualVQ:
    """Stage q quantises what stages 0..q-1 left behind (SoundStream, EnCodec)."""

    def __init__(self, n_q: int, k: int):
        self.n_q, self.k = n_q, k
        self.codebooks: list[np.ndarray] = []

    def fit(self, frames: np.ndarray, rng: np.random.Generator) -> "ResidualVQ":
        resid = frames.copy()
        for _ in range(self.n_q):
            cb = kmeans(resid, self.k, rng)
            self.codebooks.append(cb)
            resid = resid - cb[nearest(resid, cb)]
        return self

    def encode(self, frames: np.ndarray, n_q: int | None = None) -> np.ndarray:
        resid, codes = frames.copy(), []
        for cb in self.codebooks[: n_q or self.n_q]:
            idx = nearest(resid, cb)
            codes.append(idx)
            resid = resid - cb[idx]
        return np.stack(codes)  # (n_q, n_frames)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return sum(cb[c] for cb, c in zip(self.codebooks, codes))


def frame(x: np.ndarray, size: int) -> np.ndarray:
    n = len(x) // size
    return x[: n * size].reshape(n, size)


def delay_pattern(codes: np.ndarray, pad: int = -1) -> np.ndarray:
    """MusicGen interleaving: codebook q is shifted right by q steps, so one
    decoder step emits one token per codebook. Shape (n_q, T) -> (n_q, T + n_q - 1)."""
    n_q, T = codes.shape
    out = np.full((n_q, T + n_q - 1), pad)
    for q in range(n_q):
        out[q, q : q + T] = codes[q]
    return out


def undo_delay(delayed: np.ndarray) -> np.ndarray:
    n_q, L = delayed.shape
    T = L - n_q + 1
    return np.stack([delayed[q, q : q + T] for q in range(n_q)])


def stft(x: np.ndarray, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    win = np.hanning(n_fft)
    n = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[i * hop : i * hop + n_fft] * win for i in range(n)])
    return np.fft.rfft(frames, axis=1)  # (frames, n_fft/2 + 1)


def istft(
    spec: np.ndarray, length: int, n_fft: int = 512, hop: int = 128
) -> np.ndarray:
    win = np.hanning(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * win
    out, norm = np.zeros(length), np.zeros(length)
    for i, f in enumerate(frames):
        out[i * hop : i * hop + n_fft] += f
        norm[i * hop : i * hop + n_fft] += win**2
    return out / np.maximum(norm, 1e-8)


def griffin_lim(
    mag: np.ndarray,
    length: int,
    iters: int,
    rng: np.random.Generator,
    n_fft: int = 512,
    hop: int = 128,
) -> np.ndarray:
    """Recover a waveform from a magnitude spectrogram by alternating projections."""
    phase = np.exp(2j * np.pi * rng.uniform(size=mag.shape))
    x = istft(mag * phase, length, n_fft, hop)
    for _ in range(iters):
        spec = stft(x, n_fft, hop)
        x = istft(mag * np.exp(1j * np.angle(spec)), length, n_fft, hop)
    return x


def spectral_convergence(
    mag: np.ndarray, x: np.ndarray, n_fft: int = 512, hop: int = 128
) -> float:
    est = np.abs(stft(x, n_fft, hop))
    return float(np.linalg.norm(mag - est) / np.linalg.norm(mag))
