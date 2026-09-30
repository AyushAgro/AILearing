"""Evaluation for visual question answering: answer normalisation, VQA soft accuracy,
relaxed numeric accuracy, POPE-style yes/no probes, CHAIR-style caption hallucination,
and selective answering (coverage versus accuracy)."""
from __future__ import annotations

import re

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUMBERS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalise_answer(text: str) -> str:
    text = text.lower().strip().rstrip(".")
    text = re.sub(r"[^\w\s.%-]", "", text)
    text = _ARTICLES.sub(" ", text)
    words = [_NUMBERS.get(w, w) for w in text.split()]
    return " ".join(words)


def vqa_soft_accuracy(prediction: str, human_answers: list[str]) -> float:
    """VQA-style: an answer given by at least three annotators scores 1."""
    p = normalise_answer(prediction)
    matches = sum(normalise_answer(h) == p for h in human_answers)
    return min(matches / 3.0, 1.0)


def relaxed_match(prediction: str, target: str, tolerance: float = 0.05) -> bool:
    """Numeric answers within a relative tolerance count; text needs exact match."""
    try:
        p = float(normalise_answer(prediction).rstrip("%"))
        t = float(normalise_answer(target).rstrip("%"))
    except ValueError:
        return normalise_answer(prediction) == normalise_answer(target)
    return abs(p - t) <= tolerance * abs(t) if t else p == 0.0


def pope_metrics(pred_yes: np.ndarray, truth_yes: np.ndarray) -> dict[str, float]:
    tp = float(np.sum(pred_yes & truth_yes))
    fp = float(np.sum(pred_yes & ~truth_yes))
    fn = float(np.sum(~pred_yes & truth_yes))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"acc": float(np.mean(pred_yes == truth_yes)), "precision": precision,
            "recall": recall, "f1": f1, "yes_ratio": float(np.mean(pred_yes))}


def chair(mentioned: list[set[str]], present: list[set[str]]) -> dict[str, float]:
    """CHAIR-i: hallucinated object mentions / all mentions.
    CHAIR-s: captions with at least one hallucinated object / all captions."""
    halluc = [m - p for m, p in zip(mentioned, present)]
    total = sum(len(m) for m in mentioned)
    return {"chair_i": sum(len(h) for h in halluc) / max(total, 1),
            "chair_s": float(np.mean([bool(h) for h in halluc]))}


def selective(confidence: np.ndarray, correct: np.ndarray,
              thresholds: list[float]) -> list[tuple[float, float, float]]:
    """(threshold, coverage, accuracy on answered) for each threshold."""
    rows = []
    for t in thresholds:
        answered = confidence >= t
        acc = float(correct[answered].mean()) if answered.any() else float("nan")
        rows.append((t, float(answered.mean()), acc))
    return rows
