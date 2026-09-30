"""Detection building blocks: vectorised IoU and GIoU, NMS, Hungarian matching,
average precision at an IoU threshold, and a YOLO-style grid assignment.
All boxes are xyxy floats."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, 4) x (M, 4) -> (N, M) IoU."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def giou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Generalised IoU: IoU minus the empty fraction of the smallest enclosing box.
    Non-zero gradient signal even when boxes do not overlap."""
    iou = iou_matrix(a, b)
    lt = np.minimum(a[:, None, :2], b[None, :, :2])
    rb = np.maximum(a[:, None, 2:], b[None, :, 2:])
    hull = (rb - lt).prod(-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    inter_wh = (np.minimum(a[:, None, 2:], b[None, :, 2:])
                - np.maximum(a[:, None, :2], b[None, :, :2])).clip(min=0)
    union = area_a[:, None] + area_b[None, :] - inter_wh.prod(-1)
    return iou - (hull - union) / hull


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    order = list(np.argsort(-scores))
    keep = []
    while order:
        best = order.pop(0)
        keep.append(int(best))
        if order:
            ious = iou_matrix(boxes[[best]], boxes[order])[0]
            order = [i for i, v in zip(order, ious) if v <= threshold]
    return keep


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))


def average_precision(preds: list[tuple[int, float, np.ndarray]],
                      gts: dict[int, np.ndarray], iou_thr: float = 0.5) -> float:
    """VOC-style AP for one class. preds: (image_id, score, box). gts: image_id -> (G, 4).
    Each ground-truth box can be claimed once; later duplicates are false positives."""
    n_gt = sum(len(g) for g in gts.values())
    claimed = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = []
    for img, _score, box in sorted(preds, key=lambda p: -p[1]):
        g = gts.get(img, np.zeros((0, 4)))
        if len(g):
            ious = iou_matrix(box[None], g)[0]
            j = int(ious.argmax())
            if ious[j] >= iou_thr and not claimed[img][j]:
                claimed[img][j] = True
                tp.append(1)
                continue
        tp.append(0)
    tp_arr = np.array(tp)
    cum_tp = np.cumsum(tp_arr)
    precision = cum_tp / np.arange(1, len(tp_arr) + 1)
    recall = cum_tp / max(n_gt, 1)
    # all-point interpolation: precision envelope, integrated over recall steps
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(steps * envelope))


def yolo_targets(boxes: np.ndarray, classes: np.ndarray, image: int, grid: int,
                 n_classes: int) -> np.ndarray:
    """One box per cell: the cell containing the box centre is responsible.
    Target per cell: [objectness, x_off, y_off, w, h, one-hot class]."""
    t = np.zeros((grid, grid, 5 + n_classes))
    cell = image / grid
    for (x1, y1, x2, y2), c in zip(boxes, classes):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        i, j = int(cy // cell), int(cx // cell)
        t[i, j, :5] = [1.0, cx / cell - j, cy / cell - i, (x2 - x1) / image, (y2 - y1) / image]
        t[i, j, 5 + int(c)] = 1.0
    return t
