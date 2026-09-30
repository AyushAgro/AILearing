"""A small Vision Transformer, a CNN baseline and a masked autoencoder in PyTorch.

Everything is sized for a laptop CPU: 16x16 images, patch 4, width 64, two blocks.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Conv2d with kernel = stride = patch is exactly a linear map on each patch."""

    def __init__(self, channels: int, patch: int, dim: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, C, H, W)
        x = self.proj(x)                                  # (B, D, H/P, W/P)
        return x.flatten(2).transpose(1, 2)               # (B, N, D)


class Block(nn.Module):
    """Pre-norm Transformer block: x + MHA(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                 nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class ViT(nn.Module):
    def __init__(self, *, size: int = 16, patch: int = 4, channels: int = 3,
                 dim: int = 64, depth: int = 2, heads: int = 4, classes: int = 4,
                 pool: str = "cls"):
        super().__init__()
        self.grid = size // patch
        self.embed = PatchEmbed(channels, patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, self.grid ** 2 + 1, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)
        self.pool = pool

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)                                   # (B, N, D)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos           # (B, N+1, D)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        t = self.tokens(x)
        return t[:, 0] if self.pool == "cls" else t[:, 1:].mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SmallCNN(nn.Module):
    def __init__(self, channels: int = 3, width: int = 32, classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * width, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_classifier(model: nn.Module, x: np.ndarray, y: np.ndarray, *, epochs: int,
                     lr: float = 3e-3, batch: int = 64, seed: int = 0) -> float:
    """Plain AdamW training loop. Returns wall-clock seconds."""
    torch.manual_seed(seed)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y).long()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            idx = order[i:i + batch]
            loss = F.cross_entropy(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return time.perf_counter() - start


@torch.no_grad()
def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    pred = model(torch.from_numpy(x)).argmax(dim=-1).numpy()
    return float((pred == y).mean())


class TinyMAE(nn.Module):
    """Masked autoencoder: the encoder sees only visible patches; a narrower decoder
    receives mask tokens and predicts the pixels of the masked patches."""

    def __init__(self, *, size: int = 16, patch: int = 4, channels: int = 3,
                 dim: int = 64, depth: int = 2, dec_dim: int = 32, ratio: float = 0.75):
        super().__init__()
        self.patch, self.ratio = patch, ratio
        n = (size // patch) ** 2
        self.embed = PatchEmbed(channels, patch, dim)
        self.pos = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, 4) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.to_dec = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.randn(1, n, dec_dim) * 0.02)
        self.decoder = Block(dec_dim, 2)
        self.to_pixels = nn.Linear(dec_dim, patch * patch * channels)

    def encode(self, x: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        t = self.embed(x) + self.pos                               # (B, N, D)
        if keep is not None:
            t = torch.gather(t, 1, keep[..., None].expand(-1, -1, t.shape[-1]))
        for block in self.blocks:
            t = block(t)
        return self.norm(t)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n = x.shape[0], self.pos.shape[1]
        n_keep = n - int(round(n * self.ratio))
        shuffle = torch.rand(b, n).argsort(dim=1)
        restore = shuffle.argsort(dim=1)
        keep = shuffle[:, :n_keep]
        z = self.to_dec(self.encode(x, keep))                     # (B, n_keep, d)
        fill = self.mask_token.expand(b, n - n_keep, -1)
        z = torch.cat([z, fill], dim=1)
        z = torch.gather(z, 1, restore[..., None].expand(-1, -1, z.shape[-1]))
        pred = self.to_pixels(self.decoder(z + self.dec_pos))      # (B, N, P*P*C)
        target = F.unfold(x, self.patch, stride=self.patch).transpose(1, 2)
        mask = torch.ones(b, n)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, restore)                     # 1 = masked
        per_patch = ((pred - target) ** 2).mean(dim=-1)
        return (per_patch * mask).sum() / mask.sum(), pred
