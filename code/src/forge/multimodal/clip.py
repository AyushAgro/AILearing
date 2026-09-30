"""Batch contrastive loss, and the sigmoid variant used by SigLIP."""
from __future__ import annotations

import numpy as np


def l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cross_entropy_rows(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=-1))
    nll = log_z - shifted[np.arange(len(labels)), labels]
    return float(nll.mean())


def contrastive_loss(image: np.ndarray, text: np.ndarray, temperature: float = 0.07) -> float:
    """Symmetric InfoNCE. Diagonals are the matched pairs."""
    image = l2_normalise(image)
    text = l2_normalise(text)
    logits = image @ text.T / temperature
    labels = np.arange(len(image))
    return 0.5 * (cross_entropy_rows(logits, labels) + cross_entropy_rows(logits.T, labels))


def sigmoid_loss(image: np.ndarray, text: np.ndarray, scale: float = 10.0,
                 bias: float = -10.0) -> float:
    """SigLIP-style pairwise sigmoid loss. +1 on the diagonal, -1 elsewhere."""
    image = l2_normalise(image)
    text = l2_normalise(text)
    logits = scale * (image @ text.T) + bias
    labels = 2.0 * np.eye(len(image)) - 1.0
    return float(np.mean(np.logaddexp(0.0, -labels * logits)))


def align_step(image: np.ndarray, text: np.ndarray, weight: np.ndarray,
               lr: float, temperature: float = 0.07) -> tuple[float, np.ndarray]:
    """One SGD step on a linear map `text @ weight` toward `image`."""
    mapped = text @ weight
    loss = contrastive_loss(image, mapped, temperature)
    image_n = l2_normalise(image)
    mapped_n = l2_normalise(mapped)
    logits = image_n @ mapped_n.T / temperature
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    labels = np.eye(len(image))
    # dL/d mapped_n is approximate: treat the normaliser as constant for the demo step.
    grad_mapped = (probs - labels) @ image_n / temperature
    grad_w = text.T @ grad_mapped / len(image)
    return loss, weight - lr * grad_w
