"""Patch embeddings, a one-block ViT, masking, and image-token arithmetic."""
from __future__ import annotations

import numpy as np


def image_tokens(height: int, width: int, patch: int) -> int:
    if height % patch or width % patch:
        raise ValueError("height and width must be multiples of the patch size")
    return (height // patch) * (width // patch)


def patchify(images: np.ndarray, patch: int) -> np.ndarray:
    """(B, C, H, W) -> (B, N, P*P*C) in row-major patch order."""
    batch, channels, height, width = images.shape
    gh, gw = height // patch, width // patch
    x = images.reshape(batch, channels, gh, patch, gw, patch)
    x = np.transpose(x, (0, 2, 4, 3, 5, 1))
    return x.reshape(batch, gh * gw, patch * patch * channels)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class TinyViT:
    """One pre-norm attention block over patch tokens plus a class token.

    Weights are random. The demo checks shapes, parameter arithmetic and the
    effect of position embeddings. It does not claim ImageNet accuracy.
    """

    def __init__(self, rng: np.random.Generator, *, channels: int, size: int,
                 patch: int, dim: int):
        self.patch = patch
        self.dim = dim
        self.n = image_tokens(size, size, patch)
        in_dim = patch * patch * channels
        self.proj = rng.normal(scale=0.02, size=(in_dim, dim))
        self.cls = rng.normal(scale=0.02, size=(dim,))
        self.pos = rng.normal(scale=0.02, size=(self.n + 1, dim))
        self.wq = rng.normal(scale=0.02, size=(dim, dim))
        self.wk = rng.normal(scale=0.02, size=(dim, dim))
        self.wv = rng.normal(scale=0.02, size=(dim, dim))
        self.wo = rng.normal(scale=0.02, size=(dim, dim))

    def parameter_count(self) -> dict[str, int]:
        in_dim = self.proj.shape[0]
        dim = self.dim
        return {
            "patch_embed": in_dim * dim,
            "class_token": dim,
            "positions": (self.n + 1) * dim,
            "attention": 4 * dim * dim,
            "total": in_dim * dim + dim + (self.n + 1) * dim + 4 * dim * dim,
        }

    def forward(self, images: np.ndarray, *, use_pos: bool = True) -> np.ndarray:
        tokens = patchify(images, self.patch) @ self.proj
        batch = tokens.shape[0]
        cls = np.broadcast_to(self.cls, (batch, 1, self.dim)).copy()
        x = np.concatenate([cls, tokens], axis=1)
        if use_pos:
            x = x + self.pos
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        weights = softmax(q @ np.swapaxes(k, -1, -2) / np.sqrt(self.dim))
        return x + (weights @ v) @ self.wo


def mae_visible_tokens(n_patches: int, mask_ratio: float) -> tuple[int, int]:
    masked = int(round(n_patches * mask_ratio))
    return n_patches - masked, masked


def unpatchify(tokens: np.ndarray, patch: int, channels: int, height: int,
               width: int) -> np.ndarray:
    """Inverse of patchify: (B, N, P*P*C) -> (B, C, H, W)."""
    batch = tokens.shape[0]
    gh, gw = height // patch, width // patch
    x = tokens.reshape(batch, gh, gw, patch, patch, channels)
    x = np.transpose(x, (0, 5, 1, 3, 2, 4))
    return x.reshape(batch, channels, height, width)


def sincos_2d(grid_h: int, grid_w: int, dim: int) -> np.ndarray:
    """Fixed 2-D sine-cosine position table, (grid_h * grid_w, dim). Half the width
    encodes the row, half the column, as in MAE's fixed embeddings."""
    if dim % 4:
        raise ValueError("dim must be a multiple of 4")
    quarter = dim // 4
    omega = 1.0 / 10000 ** (np.arange(quarter) / quarter)
    rows, cols = np.mgrid[0:grid_h, 0:grid_w]
    parts = []
    for coord in (rows.ravel(), cols.ravel()):
        angles = coord[:, None] * omega[None, :]
        parts += [np.sin(angles), np.cos(angles)]
    return np.concatenate(parts, axis=1)


def random_masking(rng: np.random.Generator, batch: int, n: int,
                   ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MAE-style per-sample shuffle. Returns keep indices (B, n_keep), the binary
    mask in original order (B, n; 1 = masked) and the restore permutation."""
    n_keep = n - int(round(n * ratio))
    noise = rng.random((batch, n))
    shuffle = np.argsort(noise, axis=1)
    restore = np.argsort(shuffle, axis=1)
    keep = shuffle[:, :n_keep]
    mask = np.ones((batch, n))
    mask[:, :n_keep] = 0
    mask = np.take_along_axis(mask, restore, axis=1)
    return keep, mask, restore


def plan_tiles(height: int, width: int, tile: int, max_tiles: int,
               thumbnail: bool = True) -> dict[str, object]:
    """Pick a (rows, cols) grid of square tiles whose aspect ratio is closest to the
    image's, using at most max_tiles tiles. Among equal ratios, prefer the grid whose
    pixel count is closest to the image's, so small images are not blown up."""
    aspect = width / height
    best, best_key = (1, 1), None
    for rows in range(1, max_tiles + 1):
        for cols in range(1, max_tiles // rows + 1):
            err = abs(np.log((cols / rows) / aspect))
            area = abs(np.log(rows * cols * tile * tile / (height * width)))
            key = (round(err, 6), area)
            if best_key is None or key < best_key:
                best, best_key = (rows, cols), key
    rows, cols = best
    tiles = rows * cols + (1 if thumbnail and rows * cols > 1 else 0)
    return {"grid": best, "tiles": tiles, "resized_to": (rows * tile, cols * tile)}


def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = 56 * 56, max_pixels: int = 1280 * 28 * 28
                 ) -> tuple[int, int]:
    """Round both sides to a multiple of `factor`, keeping the pixel count within
    [min_pixels, max_pixels] and the aspect ratio roughly fixed. This mirrors the
    dynamic-resolution preprocessing published with Qwen2-VL; defaults are ours."""
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = np.sqrt(height * width / max_pixels)
        h = int(np.floor(height / beta / factor) * factor)
        w = int(np.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h = int(np.ceil(height * beta / factor) * factor)
        w = int(np.ceil(width * beta / factor) * factor)
    return h, w
