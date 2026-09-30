"""Video understanding utilities: synthetic clips, frame samplers, token and attention
cost for joint, divided and tubelet schemes, and moment retrieval over frame vectors."""
from __future__ import annotations

import numpy as np

from forge.multimodal.data import COLOURS, COLOUR_NAMES, SHAPES, shape_mask


def make_clip(rng: np.random.Generator, n_frames: int = 300, size: int = 16,
              event: tuple[int, int] = (140, 152)) -> tuple[np.ndarray, list[str]]:
    """A static 'camera' with sensor noise. A red square walks across the frame for
    the frames in [event[0], event[1]); every other frame is empty background."""
    frames = rng.uniform(0, 0.08, (n_frames, 3, size, size)).astype(np.float32)
    labels = ["empty"] * n_frames
    rgb = np.array(COLOURS["red"], dtype=np.float32)[:, None, None]
    m = shape_mask("square", 6)
    for k, t in enumerate(range(*event)):
        x = int(round(k * (size - 6) / max(event[1] - event[0] - 1, 1)))
        frames[t, :, 5:11, x:x + 6] = np.where(m, rgb, frames[t, :, 5:11, x:x + 6])
        labels[t] = "a red square"
    return frames, labels


def uniform_indices(n_frames: int, k: int) -> list[int]:
    if k >= n_frames:
        return list(range(n_frames))
    return [int(round(i * (n_frames - 1) / (k - 1))) for i in range(k)]


def change_indices(frames: np.ndarray, k: int, min_gap: int = 1) -> list[int]:
    """Keep frame 0, then the k-1 frames with the largest mean absolute change from
    the previous frame, at least min_gap apart. A cheap shot/event detector."""
    diff = np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2, 3))
    chosen = [0]
    for i in np.argsort(-diff):
        t = int(i) + 1
        if all(abs(t - c) >= min_gap for c in chosen):
            chosen.append(t)
        if len(chosen) == k:
            break
    return sorted(chosen)


def video_tokens(frames: int, height: int, width: int, patch: int,
                 tubelet: int = 1) -> int:
    """Tokens for a clip cut into tubelet x patch x patch cubes (ViViT-style)."""
    return (frames // tubelet) * (height // patch) * (width // patch)


def attention_pairs(frames: int, per_frame: int, scheme: str) -> int:
    """Query-key pairs one attention layer scores.
    joint: every token attends to every token, (T*N)^2.
    divided: time attention (N * T^2) then space attention (T * N^2)."""
    if scheme == "joint":
        return (frames * per_frame) ** 2
    if scheme == "divided":
        return per_frame * frames ** 2 + frames * per_frame ** 2
    raise ValueError(scheme)


def moments(scores: np.ndarray, threshold: float, fps: float) -> list[tuple[float, float]]:
    """Contiguous runs of frames whose query similarity exceeds threshold, in seconds."""
    runs, start = [], None
    for t, on in enumerate(np.append(scores > threshold, False)):
        if on and start is None:
            start = t
        elif not on and start is not None:
            runs.append((start / fps, t / fps))
            start = None
    return runs
