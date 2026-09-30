"""Log-mel energy of a sine, greedy CTC collapse, and mu-law round-trip."""
from __future__ import annotations

import numpy as np


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_freqs)
    mels = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    hz = mel_to_hz(mels)
    weights = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        left, centre, right = hz[i], hz[i + 1], hz[i + 2]
        rise = (fft_freqs - left) / max(centre - left, 1e-6)
        fall = (right - fft_freqs) / max(right - centre, 1e-6)
        weights[i] = np.maximum(0.0, np.minimum(rise, fall))
    return weights


def log_mel(wave: np.ndarray, sr: int, *, n_fft: int = 512, hop: int = 160,
            n_mels: int = 40) -> np.ndarray:
    frames = []
    window = np.hanning(n_fft)
    for start in range(0, len(wave) - n_fft + 1, hop):
        chunk = wave[start:start + n_fft] * window
        spec = np.fft.rfft(chunk)
        frames.append(np.abs(spec) ** 2)
    power = np.stack(frames, axis=0)
    mel = power @ mel_filterbank(sr, n_fft, n_mels).T
    return np.log(mel + 1e-6)


def greedy_ctc(ids: list[int], blank: int = 0) -> list[int]:
    """Collapse repeats, then drop blanks. The standard greedy CTC decode."""
    collapsed: list[int] = []
    previous = None
    for token in ids:
        if token != previous:
            collapsed.append(token)
        previous = token
    return [token for token in collapsed if token != blank]


def mulaw(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def inv_mulaw(y: np.ndarray, mu: float = 255.0) -> np.ndarray:
    return np.sign(y) * (1.0 / mu) * ((1.0 + mu) ** np.abs(y) - 1.0)


def quantise_mulaw(x: np.ndarray, bits: int = 8) -> np.ndarray:
    encoded = mulaw(np.clip(x, -1.0, 1.0))
    levels = 2 ** bits
    codes = np.rint((encoded + 1.0) * 0.5 * (levels - 1)).astype(int)
    restored = codes / (levels - 1) * 2.0 - 1.0
    return inv_mulaw(restored)


def ctc_loss(log_probs: np.ndarray, target: list[int], blank: int = 0) -> float:
    """Negative log-likelihood of `target` under CTC, by the forward algorithm.
    log_probs: (T, V) per-frame log-softmax. Works in log space throughout."""
    ext = [blank]
    for label in target:
        ext += [label, blank]                       # l' = b, l1, b, l2, ..., b
    t_len, s_len = log_probs.shape[0], len(ext)
    alpha = np.full((t_len, s_len), -np.inf)
    alpha[0, 0] = log_probs[0, blank]
    if s_len > 1:
        alpha[0, 1] = log_probs[0, ext[1]]
    for t in range(1, t_len):
        for s in range(s_len):
            paths = [alpha[t - 1, s]]
            if s >= 1:
                paths.append(alpha[t - 1, s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                paths.append(alpha[t - 1, s - 2])   # skip a blank between labels
            alpha[t, s] = np.logaddexp.reduce(paths) + log_probs[t, ext[s]]
    end = alpha[-1, -1] if s_len == 1 else np.logaddexp(alpha[-1, -1], alpha[-1, -2])
    return float(-end)


def prefix_beam_search(probs: np.ndarray, beam: int = 8, blank: int = 0
                       ) -> list[tuple[tuple[int, ...], float]]:
    """CTC prefix beam search over (T, V) probabilities. Each prefix tracks the
    probability of ending in a blank and of ending in its last label."""
    beams = {(): (1.0, 0.0)}                        # prefix -> (p_blank, p_label)
    for t in range(probs.shape[0]):
        nxt: dict[tuple[int, ...], list[float]] = {}
        for prefix, (pb, pnb) in beams.items():
            for v in range(probs.shape[1]):
                p = probs[t, v]
                if v == blank:
                    e = nxt.setdefault(prefix, [0.0, 0.0])
                    e[0] += (pb + pnb) * p
                    continue
                last = prefix[-1] if prefix else None
                ext = prefix + (v,)
                e = nxt.setdefault(ext, [0.0, 0.0])
                if v == last:
                    e[1] += pb * p                  # repeat after a blank: new label
                    same = nxt.setdefault(prefix, [0.0, 0.0])
                    same[1] += pnb * p              # repeat without blank: collapse
                else:
                    e[1] += (pb + pnb) * p
        ranked = sorted(nxt.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:beam]
        beams = {k: (v[0], v[1]) for k, v in ranked}
    return [(k, v[0] + v[1]) for k, v in beams.items()]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance divided by the number of reference words."""
    ref, hyp = reference.split(), hypothesis.split()
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def energy_vad(wave: np.ndarray, sr: int, frame_ms: int = 30,
               threshold_db: float = -35.0) -> list[tuple[float, float]]:
    """Speech regions (start_s, end_s) where frame RMS is above a dB threshold
    relative to the loudest frame. A baseline, not a trained VAD."""
    n = int(sr * frame_ms / 1000)
    frames = wave[: len(wave) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms / rms.max())
    active = db > threshold_db
    regions, start = [], None
    for i, on in enumerate(np.append(active, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            regions.append((start * n / sr, i * n / sr))
            start = None
    return regions


def chunk_spans(duration: float, window: float = 30.0, overlap: float = 5.0
                ) -> list[tuple[float, float]]:
    """Fixed windows with overlap, the usual way to feed long audio to a model with
    a fixed input length. Stitching must de-duplicate words in the overlaps."""
    spans, start = [], 0.0
    step = window - overlap
    while True:
        spans.append((start, min(start + window, duration)))
        if start + window >= duration:
            return spans
        start += step
