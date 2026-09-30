"""A tiny MLP with ReLU hidden layers, used by the generation demos."""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, rng: np.random.Generator, sizes: list[int]):
        self.sizes = list(sizes)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.W.append(rng.normal(scale=scale, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        caches: list[tuple[np.ndarray, np.ndarray]] = []
        h = x
        last = len(self.W) - 1
        for i, (weight, bias) in enumerate(zip(self.W, self.b)):
            z = h @ weight + bias
            caches.append((h, z))
            h = z if i == last else np.maximum(z, 0.0)
        return h, caches

    def backward(self, caches: list[tuple[np.ndarray, np.ndarray]], grad: np.ndarray,
                 lr: float) -> np.ndarray:
        """Apply one SGD step and return the gradient with respect to the input."""
        last = len(self.W) - 1
        for i in range(last, -1, -1):
            h, z = caches[i]
            if i != last:
                grad = grad * (z > 0.0)
            grad_w = h.T @ grad / h.shape[0]
            grad_b = grad.mean(axis=0)
            np.clip(grad_w, -1.0, 1.0, out=grad_w)
            np.clip(grad_b, -1.0, 1.0, out=grad_b)
            grad = grad @ self.W[i].T
            self.W[i] -= lr * grad_w
            self.b[i] -= lr * grad_b
        return grad

    def step_loss(self, x: np.ndarray, target: np.ndarray, lr: float) -> float:
        pred, caches = self.forward(x)
        err = pred - target
        loss = float(np.mean(err ** 2))
        self.backward(caches, 2.0 * err, lr)
        return loss
