"""DETR-style set prediction at toy scale: a CNN backbone, a Transformer decoder with
learned object queries, and a loss computed after Hungarian matching."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from forge.multimodal.data import COLOURS, COLOUR_NAMES, SHAPES, shape_mask


def make_scene(rng: np.random.Generator, size: int = 32, max_objects: int = 3):
    """Canvas with 1..max_objects shapes that do not overlap. Class = shape index."""
    img = rng.uniform(0, 0.08, (3, size, size)).astype(np.float32)
    boxes, classes, taken = [], [], np.zeros((size, size), bool)
    for _ in range(int(rng.integers(1, max_objects + 1))):
        for _attempt in range(20):
            side = int(rng.integers(6, 11))
            x, y = int(rng.integers(0, size - side)), int(rng.integers(0, size - side))
            if not taken[y:y + side, x:x + side].any():
                break
        else:
            continue
        taken[y:y + side, x:x + side] = True
        s, c = int(rng.integers(len(SHAPES))), int(rng.integers(len(COLOURS)))
        m = shape_mask(SHAPES[s], side)
        rgb = np.array(COLOURS[COLOUR_NAMES[c]], dtype=np.float32)[:, None, None]
        img[:, y:y + side, x:x + side] = np.where(m, rgb, img[:, y:y + side, x:x + side])
        boxes.append([x, y, x + side, y + side])
        classes.append(s)
    return img, np.array(boxes, np.float32) / size, np.array(classes)


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


class TinyDETR(nn.Module):
    def __init__(self, n_classes: int = 4, queries: int = 5, dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, dim, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(), nn.MaxPool2d(2))   # 4x4 map
        self.pos = nn.Parameter(torch.randn(1, 16, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, queries, dim) * 0.1)
        layer = nn.TransformerDecoderLayer(dim, 4, 2 * dim, dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 3)
        self.cls = nn.Linear(dim, n_classes + 1)            # last class = "no object"
        self.box = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem = self.backbone(x).flatten(2).transpose(1, 2) + self.pos       # (B, 16, D)
        h = self.decoder(self.queries.expand(len(x), -1, -1), mem)         # (B, Q, D)
        return self.cls(h), self.box(h).sigmoid()                          # cxcywh in [0,1]


def set_loss(logits: torch.Tensor, boxes: torch.Tensor, targets: list[tuple],
             w_cls: float = 1.0, w_l1: float = 5.0, w_giou: float = 2.0,
             eos_weight: float = 0.2) -> torch.Tensor:
    """Match queries to objects per image with the Hungarian algorithm, then apply
    cross-entropy to every query (unmatched -> no-object) and box losses to matches."""
    from torchvision.ops import generalized_box_iou
    n_cls = logits.shape[-1] - 1
    weight = torch.ones(n_cls + 1)
    weight[-1] = eos_weight
    total_cls, total_box, n_obj = 0.0, 0.0, 0
    for b, (gt_boxes, gt_cls) in enumerate(targets):
        gt_boxes, gt_cls = torch.from_numpy(gt_boxes), torch.from_numpy(gt_cls).long()
        pred_xyxy = cxcywh_to_xyxy(boxes[b])
        target_cls = torch.full((logits.shape[1],), n_cls)
        if len(gt_cls):
            with torch.no_grad():
                prob = logits[b].softmax(-1)[:, gt_cls]                    # (Q, G)
                cost = (-w_cls * prob + w_l1 * torch.cdist(pred_xyxy, gt_boxes, p=1)
                        - w_giou * generalized_box_iou(pred_xyxy, gt_boxes))
            qi, gi = linear_sum_assignment(cost.numpy())
            qi, gi = torch.from_numpy(qi), torch.from_numpy(gi)
            target_cls[qi] = gt_cls[gi]
            matched, gt = pred_xyxy[qi], gt_boxes[gi]
            total_box = total_box + w_l1 * F.l1_loss(matched, gt, reduction="sum") \
                + w_giou * (1 - torch.diag(generalized_box_iou(matched, gt))).sum()
            n_obj += len(gi)
        total_cls = total_cls + F.cross_entropy(logits[b], target_cls, weight=weight)
    return total_cls / len(targets) + total_box / max(n_obj, 1)
