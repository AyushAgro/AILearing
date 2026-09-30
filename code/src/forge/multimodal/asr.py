"""A toy speech recogniser trained with CTC on synthetic 'tone speech'.

Each letter is a two-tone chord of random duration; words are separated by
silence. The point is the pipeline (waveform -> log-mel -> encoder -> CTC ->
decode -> WER), not the acoustics of real speech.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from forge.multimodal.audio import log_mel

LETTERS = "adeinost"
CHORDS = {ch: (300 + 110 * i, 1200 + 170 * ((i * 3) % 8)) for i, ch in enumerate(LETTERS)}
LEXICON = ["at", "dot", "net", "sit", "ten", "tin", "toad", "side", "dose", "note",
           "done", "sand", "nose", "tide", "seat", "dens"]
LABELS = ["<blank>", " ", *LETTERS]          # CTC blank is index 0
SR = 16_000


def synthesise(rng: np.random.Generator, text: str, noise: float = 0.02) -> np.ndarray:
    parts = [np.zeros(int(SR * rng.uniform(0.05, 0.1)))]
    for ch in text:
        if ch == " ":
            parts.append(np.zeros(int(SR * rng.uniform(0.08, 0.15))))
            continue
        n = int(SR * rng.uniform(0.05, 0.11))
        t = np.arange(n) / SR
        f1, f2 = CHORDS[ch]
        env = np.hanning(n) ** 0.3
        parts.append(env * (0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)))
    parts.append(np.zeros(int(SR * rng.uniform(0.05, 0.1))))
    wave = np.concatenate(parts)
    return wave + rng.normal(0, noise, len(wave))


def utterance(rng: np.random.Generator) -> str:
    return " ".join(rng.choice(LEXICON, size=int(rng.integers(2, 4))))


def features(wave: np.ndarray) -> np.ndarray:
    mel = log_mel(wave, SR, n_fft=400, hop=160, n_mels=40)      # (T, 40), 100 frames/s
    return ((mel - mel.mean()) / (mel.std() + 1e-6)).astype(np.float32)


class CTCModel(nn.Module):
    def __init__(self, n_mels: int = 40, hidden: int = 96):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_mels, hidden, 5, padding=2), nn.GELU())
        self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                          bidirectional=True)
        self.out = nn.Linear(2 * hidden, len(LABELS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, T, M)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.out(h).log_softmax(-1)                        # (B, T, V)


def encode(text: str) -> list[int]:
    return [LABELS.index(ch) for ch in text]


def decode(ids: list[int]) -> str:
    return "".join(LABELS[i] for i in ids)
