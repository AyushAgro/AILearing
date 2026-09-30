"""Synthetic shapes: a laptop-sized image dataset with captions, boxes and questions.

Every chapter of Volume 19 trains or evaluates on this generator so that no image is
downloaded. Images are (3, S, S) floats in [0, 1] with one coloured shape on a noisy
dark background. The generator also returns the caption and the xyxy box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHAPES = ("square", "circle", "triangle", "cross")
COLOURS = {
    "red": (1.0, 0.1, 0.1),
    "green": (0.1, 0.9, 0.1),
    "blue": (0.2, 0.3, 1.0),
    "yellow": (1.0, 0.9, 0.1),
}
COLOUR_NAMES = tuple(COLOURS)


def shape_mask(shape: str, size: int) -> np.ndarray:
    """Boolean (size, size) mask for one shape drawn in its own box."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    if shape == "square":
        return np.ones((size, size), dtype=bool)
    if shape == "circle":
        return (yy - c) ** 2 + (xx - c) ** 2 <= (size / 2) ** 2
    if shape == "triangle":
        return np.abs(xx - c) <= (yy + 1) / 2
    if shape == "cross":
        w = max(1, size // 4)
        return (np.abs(xx - c) < w) | (np.abs(yy - c) < w)
    raise ValueError(shape)


@dataclass
class Sample:
    image: np.ndarray        # (3, S, S)
    shape: int               # index into SHAPES
    colour: int              # index into COLOUR_NAMES
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 (exclusive)

    @property
    def caption(self) -> str:
        return f"a {COLOUR_NAMES[self.colour]} {SHAPES[self.shape]}"

    @property
    def side(self) -> str:
        cx = (self.box[0] + self.box[2]) / 2
        return "left" if cx < self.image.shape[-1] / 2 else "right"


def draw(rng: np.random.Generator, shape: int, colour: int, size: int = 16,
         obj: tuple[int, int] = (6, 9), noise: float = 0.08) -> Sample:
    image = rng.uniform(0.0, noise, size=(3, size, size))
    side = int(rng.integers(obj[0], obj[1] + 1))
    x0 = int(rng.integers(0, size - side + 1))
    y0 = int(rng.integers(0, size - side + 1))
    mask = shape_mask(SHAPES[shape], side)
    rgb = np.array(COLOURS[COLOUR_NAMES[colour]])[:, None, None]
    patch = image[:, y0:y0 + side, x0:x0 + side]
    image[:, y0:y0 + side, x0:x0 + side] = np.where(mask, rgb, patch)
    return Sample(image, shape, colour, (x0, y0, x0 + side, y0 + side))


def make_dataset(rng: np.random.Generator, n: int, size: int = 16,
                 exclude: set[tuple[int, int]] | None = None) -> list[Sample]:
    """n random samples. `exclude` holds (shape, colour) pairs never generated."""
    out: list[Sample] = []
    while len(out) < n:
        s = int(rng.integers(len(SHAPES)))
        c = int(rng.integers(len(COLOUR_NAMES)))
        if exclude and (s, c) in exclude:
            continue
        out.append(draw(rng, s, c, size))
    return out


def stack(samples: list[Sample]) -> np.ndarray:
    return np.stack([s.image for s in samples]).astype(np.float32)  # (B, 3, S, S)
