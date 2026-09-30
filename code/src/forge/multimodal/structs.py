"""IoU, a brute-force set matcher, character error rate, frame sampling."""
from __future__ import annotations

import itertools

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union for xyxy boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def match_boxes(preds: np.ndarray, gts: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Minimum-cost assignment. Cost is 1 - IoU. Fine for a handful of boxes."""
    n = len(preds)
    if n != len(gts):
        raise ValueError("this teaching matcher expects equal counts")
    best: tuple[list[tuple[int, int]], float] | None = None
    for perm in itertools.permutations(range(n)):
        pairs = [(i, perm[i]) for i in range(n)]
        cost = sum(1.0 - box_iou(preds[i], gts[j]) for i, j in pairs)
        if best is None or cost < best[1]:
            best = (pairs, cost)
    assert best is not None
    return best


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def sample_frame_indices(n_frames: int, k: int) -> list[int]:
    if k >= n_frames:
        return list(range(n_frames))
    if k == 1:
        return [0]
    return [int(round(i * (n_frames - 1) / (k - 1))) for i in range(k)]


def frame_delta(frames: np.ndarray) -> float:
    """Mean L2 gap between consecutive frames. Lower means more consistent."""
    gaps = frames[1:] - frames[:-1]
    return float(np.mean(np.linalg.norm(gaps, axis=1)))
