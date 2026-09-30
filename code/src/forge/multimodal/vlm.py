"""A mini vision-language model: frozen image encoder, projector, tiny causal LM.

The shape is LLaVA's (projected patch tokens are prefixed to the text), shrunk to a
laptop: 16 image tokens, a 2-block decoder, a 30-word vocabulary. A perceiver-style
resampler (Flamingo / BLIP-2 idea: learned queries cross-attend to image tokens) is
provided as an alternative connector.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.multimodal.data import COLOUR_NAMES, SHAPES, Sample
from forge.multimodal.vit_torch import Block

WORDS = ["<pad>", "<bos>", "<eos>", "q:", "a:", "what", "colour", "shape", "side",
         "is", "there", "it", "on", "the", "left", "right", "yes", "no", "a",
         *COLOUR_NAMES, *SHAPES]
WID = {w: i for i, w in enumerate(WORDS)}


@dataclass
class QA:
    image: np.ndarray
    question: str
    answer: str


def make_qa(rng: np.random.Generator, samples: list[Sample]) -> list[QA]:
    """One question per image: colour, shape, side, or a balanced yes/no probe."""
    out = []
    for s in samples:
        kind = int(rng.integers(4))
        if kind == 0:
            out.append(QA(s.image, "what colour is it", COLOUR_NAMES[s.colour]))
        elif kind == 1:
            out.append(QA(s.image, "what shape is it", SHAPES[s.shape]))
        elif kind == 2:
            out.append(QA(s.image, "what side is it on", s.side))
        else:
            if rng.random() < 0.5:
                c, sh, ans = s.colour, s.shape, "yes"
            else:   # a different object: change colour, shape or both
                c, sh = s.colour, s.shape
                while (c, sh) == (s.colour, s.shape):
                    c, sh = int(rng.integers(4)), int(rng.integers(4))
                ans = "no"
            out.append(QA(s.image, f"is there a {COLOUR_NAMES[c]} {SHAPES[sh]}", ans))
    return out


def encode_text(qa: list[QA], length: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids `<bos> q: ... a: ANSWER <eos>` and the index of the answer token."""
    rows, pos = [], []
    for item in qa:
        ids = [WID["<bos>"], WID["q:"], *(WID[w] for w in item.question.split()),
               WID["a:"]]
        pos.append(len(ids))
        ids += [WID[item.answer], WID["<eos>"]]
        rows.append(ids + [0] * (length - len(ids)))
    return torch.tensor(rows), torch.tensor(pos)


class TinyLM(nn.Module):
    def __init__(self, dim: int = 64, depth: int = 2, heads: int = 4, max_len: int = 40):
        super().__init__()
        self.tok = nn.Embedding(len(WORDS), dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, len(WORDS))

    def forward(self, prefix: torch.Tensor | None, ids: torch.Tensor) -> torch.Tensor:
        """prefix: (B, T_img, D) projected image tokens or None; ids: (B, T_txt)."""
        x = self.tok(ids)
        if prefix is not None:
            x = torch.cat([prefix, x], dim=1)                     # (B, T_img+T_txt, D)
        t = x.shape[1]
        x = x + self.pos[:, :t]
        causal = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
        for block in self.blocks:
            x = block(x, causal)
        return self.out(self.norm(x))                             # (B, T, V)


class Resampler(nn.Module):
    """k learned queries cross-attend to the image tokens: T_img -> k tokens."""

    def __init__(self, dim: int, k: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, k, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.queries.expand(tokens.shape[0], -1, -1)
        z = q + self.attn(q, tokens, tokens, need_weights=False)[0]
        return z + self.mlp(z)


class MiniVLM(nn.Module):
    def __init__(self, encoder: nn.Module, connector: str = "mlp", lm_dim: int = 64,
                 enc_dim: int = 64, k: int = 4):
        super().__init__()
        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False                               # frozen, always
        if connector == "linear":
            self.connector = nn.Linear(enc_dim, lm_dim)
        elif connector == "mlp":
            self.connector = nn.Sequential(nn.Linear(enc_dim, lm_dim), nn.GELU(),
                                           nn.Linear(lm_dim, lm_dim))
        elif connector == "resampler":
            self.connector = nn.Sequential(nn.Linear(enc_dim, lm_dim),
                                           Resampler(lm_dim, k))
        self.lm = TinyLM(lm_dim)

    def image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            patches = self.encoder.tokens(images)[:, 1:]          # drop cls: (B, 16, D)
        return self.connector(patches)

    def forward(self, images: torch.Tensor | None, ids: torch.Tensor) -> torch.Tensor:
        prefix = None if images is None else self.image_tokens(images)
        return self.lm(prefix, ids)


def answer_logits(model: MiniVLM, images: torch.Tensor | None, ids: torch.Tensor,
                  pos: torch.Tensor) -> torch.Tensor:
    logits = model(images, ids)
    offset = 0 if images is None else logits.shape[1] - ids.shape[1]
    rows = torch.arange(len(ids))
    return logits[rows, offset + pos - 1]                         # predicts ids[pos]


def train_vlm(model: MiniVLM, qa: list[QA], *, train: str, epochs: int, blind: bool = False,
              lr: float = 3e-3, batch: int = 64, seed: int = 0) -> float:
    """train: 'connector' (LM frozen) or 'all' (connector + LM). Loss on answer only."""
    torch.manual_seed(seed)
    for p in model.lm.parameters():
        p.requires_grad = train == "all" or blind
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    x = torch.from_numpy(np.stack([q.image for q in qa]).astype(np.float32))
    ids, pos = encode_text(qa)
    start = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(len(qa))
        for i in range(0, len(qa), batch):
            idx = order[i:i + batch]
            logits = answer_logits(model, None if blind else x[idx], ids[idx], pos[idx])
            loss = F.cross_entropy(logits, ids[idx, pos[idx]])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return time.perf_counter() - start


@torch.no_grad()
def evaluate(model: MiniVLM, qa: list[QA], blind: bool = False) -> dict[str, float]:
    model.eval()
    x = torch.from_numpy(np.stack([q.image for q in qa]).astype(np.float32))
    ids, pos = encode_text(qa)
    pred = answer_logits(model, None if blind else x, ids, pos).argmax(-1)
    ok = (pred == ids[torch.arange(len(qa)), pos]).numpy()
    kinds = [q.question.split()[1] if q.question.startswith("what") else "yes/no"
             for q in qa]
    out = {"all": float(ok.mean())}
    for kind in sorted(set(kinds)):
        out[kind] = float(ok[[k == kind for k in kinds]].mean())
    return out
