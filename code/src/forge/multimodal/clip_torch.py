"""CLIP-style dual encoder on synthetic shapes, trained with softmax or sigmoid loss."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.multimodal.data import COLOUR_NAMES, SHAPES, Sample, stack
from forge.multimodal.vit_torch import ViT

VOCAB = ["<pad>", "<unk>", "a", "photo", "of", *COLOUR_NAMES, *SHAPES]
WORD_ID = {w: i for i, w in enumerate(VOCAB)}


def tokenize(texts: list[str], length: int = 6) -> torch.Tensor:
    rows = []
    for text in texts:
        ids = [WORD_ID.get(w, 1) for w in text.lower().split()][:length]
        rows.append(ids + [0] * (length - len(ids)))
    return torch.tensor(rows)


class TextTower(nn.Module):
    """Embedding bag plus an MLP. Enough for three-word captions; a real CLIP text
    tower is a causal Transformer read at the end-of-text token."""

    def __init__(self, dim: int = 64, out: int = 32):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), dim, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:          # (B, L)
        mask = (ids != 0).float()[..., None]
        pooled = (self.emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.mlp(pooled)                                      # (B, out)


class DualEncoder(nn.Module):
    def __init__(self, out: int = 32):
        super().__init__()
        self.image = ViT(pool="mean", classes=out)     # the head is the projection
        self.text = TextTower(out=out)
        self.log_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(-10.0))  # used by the sigmoid loss only

    def embed_images(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image(x), dim=-1)

    def embed_texts(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text(ids), dim=-1)


def clip_loss(img: torch.Tensor, txt: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logits = scale * img @ txt.T                                     # (B, B)
    labels = torch.arange(len(img))
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def siglip_loss(img: torch.Tensor, txt: torch.Tensor, scale: torch.Tensor,
                bias: torch.Tensor) -> torch.Tensor:
    logits = scale * img @ txt.T + bias
    labels = 2 * torch.eye(len(img)) - 1                             # +1 diag, -1 off
    return -F.logsigmoid(labels * logits).sum() / len(img)


def train_dual(samples: list[Sample], *, loss: str = "clip", batch: int = 64,
               epochs: int = 20, lr: float = 2e-3, seed: int = 0
               ) -> tuple[DualEncoder, list[float], float]:
    torch.manual_seed(seed)
    model = DualEncoder()
    if loss == "sigmoid":
        with torch.no_grad():
            model.log_scale.fill_(np.log(10.0))
    x = torch.from_numpy(stack(samples))
    ids = tokenize([s.caption for s in samples])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    history, start = [], time.perf_counter()
    for _ in range(epochs):
        order, total = torch.randperm(len(x)), 0.0
        for i in range(0, len(x) - batch + 1, batch):
            idx = order[i:i + batch]
            img, txt = model.embed_images(x[idx]), model.embed_texts(ids[idx])
            scale = model.log_scale.exp().clamp(max=100)
            value = (clip_loss(img, txt, scale) if loss == "clip"
                     else siglip_loss(img, txt, scale, model.bias))
            opt.zero_grad()
            value.backward()
            opt.step()
            total += value.item()
        history.append(total / (len(x) // batch))
    return model, history, time.perf_counter() - start


@torch.no_grad()
def zero_shot(model: DualEncoder, samples: list[Sample],
              template: str = "a {colour} {shape}") -> float:
    """Classify each image into one of the 16 colour-shape captions."""
    model.eval()
    classes = [(s, c) for s in range(len(SHAPES)) for c in range(len(COLOUR_NAMES))]
    prompts = [template.format(colour=COLOUR_NAMES[c], shape=SHAPES[s]) for s, c in classes]
    txt = model.embed_texts(tokenize(prompts))                       # (16, D)
    img = model.embed_images(torch.from_numpy(stack(samples)))       # (N, D)
    pred = (img @ txt.T).argmax(dim=1).numpy()
    truth = np.array([classes.index((s.shape, s.colour)) for s in samples])
    return float((pred == truth).mean())
