"""Confidence estimators: logprob, consistency, semantic entropy, calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidenceEstimate:
    method: str
    value: float  # higher = more confident, in [0, 1] when calibrated-ish
    detail: str = ""


def mean_token_prob(logprobs: list[float]) -> ConfidenceEstimate:
    arr = np.asarray(logprobs, dtype=float)
    p = float(np.exp(arr).mean())
    return ConfidenceEstimate("mean_token_prob", max(0.0, min(1.0, p)), f"n={len(arr)}")


def self_consistency(answers: list[str]) -> ConfidenceEstimate:
    """Fraction agreeing with the plurality answer (exact string match)."""
    if not answers:
        return ConfidenceEstimate("self_consistency", 0.0, "empty")
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    top = max(counts.values())
    return ConfidenceEstimate("self_consistency", top / len(answers), f"modes={len(counts)}")


def _bag(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in text.split() if len(w) > 2)


def semantic_entropy(answers: list[str], *, sim_threshold: float = 0.5) -> ConfidenceEstimate:
    """Cluster answers by Jaccard bag-of-words, then entropy over cluster masses.

    Lower entropy → higher confidence. We return ``1 - H/Hmax`` in [0, 1].
    """
    if not answers:
        return ConfidenceEstimate("semantic_entropy", 0.0, "empty")
    bags = [_bag(a) for a in answers]
    clusters: list[list[int]] = []
    assigned = [-1] * len(answers)

    def _sim(a: frozenset[str], b: frozenset[str]) -> float:
        union = len(a | b) or 1
        return len(a & b) / union

    for i, bi in enumerate(bags):
        if assigned[i] >= 0:
            continue
        cid = len(clusters)
        clusters.append([i])
        assigned[i] = cid
        centroid = set(bi)
        for j in range(i + 1, len(answers)):
            if assigned[j] >= 0:
                continue
            if _sim(frozenset(centroid), bags[j]) >= sim_threshold:
                clusters[cid].append(j)
                assigned[j] = cid
                centroid |= bags[j]
    masses = np.array([len(c) for c in clusters], dtype=float)
    masses /= masses.sum()
    H = float(-(masses * np.log(masses + 1e-12)).sum())
    # Normalise by log(n) so identical answers → 1 and all-distinct → 0.
    Hmax = float(np.log(len(answers))) if len(answers) > 1 else 1.0
    conf = 1.0 - (H / Hmax if Hmax > 0 else 0.0)
    return ConfidenceEstimate(
        "semantic_entropy",
        conf,
        f"clusters={len(clusters)} H={H:.3f}",
    )


def verbalised_confidence(text: str) -> ConfidenceEstimate:
    """Parse a trailing 'confidence: 0.x' style declaration."""
    low = text.lower()
    for key in ("confidence:", "conf:"):
        if key in low:
            tail = low.split(key, 1)[1].strip().split()[0].rstrip("%.,;")
            try:
                v = float(tail)
                if v > 1.0:
                    v /= 100.0
                return ConfidenceEstimate("verbalised", max(0.0, min(1.0, v)))
            except ValueError:
                break
    return ConfidenceEstimate("verbalised", 0.5, "unparsed_default")


def estimate_confidence(
    *,
    logprobs: list[float] | None = None,
    samples: list[str] | None = None,
    verbalised_text: str | None = None,
    method: str = "auto",
) -> ConfidenceEstimate:
    if method == "logprob" or (method == "auto" and logprobs is not None and samples is None):
        assert logprobs is not None
        return mean_token_prob(logprobs)
    if method in {"self_consistency", "auto"} and samples is not None:
        if method == "self_consistency" or (method == "auto" and len(set(samples)) <= 3):
            return self_consistency(samples)
        return semantic_entropy(samples)
    if method == "semantic_entropy" and samples is not None:
        return semantic_entropy(samples)
    if verbalised_text is not None:
        return verbalised_confidence(verbalised_text)
    return ConfidenceEstimate("none", 0.0, "no_signal")


@dataclass(frozen=True)
class CalibrationReport:
    ece: float
    bins: int
    n: int


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """ECE = Σ (|B_m|/n) |acc(B_m) - conf(B_m)|."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        mask = (confidences >= edges[i]) & (confidences < edges[i + 1] + (1e-9 if i == n_bins - 1 else 0))
        if not np.any(mask):
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return CalibrationReport(ece=float(ece), bins=n_bins, n=n)


def conformal_threshold(
    calibration_scores: np.ndarray,
    *,
    alpha: float = 0.1,
) -> float:
    """Split-conformal threshold: keep predictions with score ≥ q̂.

    ``calibration_scores`` should be higher-is-better nonconformity inverted
    (e.g. confidence of the true class on a calibration set).
    """
    s = np.sort(np.asarray(calibration_scores, dtype=float))
    n = len(s)
    # standard quantile level for prediction sets at coverage 1-alpha
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(1.0, q_level)
    idx = int(np.ceil(q_level * n) - 1)
    idx = max(0, min(n - 1, idx))
    return float(s[idx])
