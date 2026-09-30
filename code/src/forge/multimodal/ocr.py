"""A classical OCR pipeline around the Tesseract CLI, plus the pieces around it:
page rendering, degradation, word boxes, reading order and field extraction.

Tesseract must be on PATH (`brew install tesseract` / `apt install tesseract-ocr`).
Pages are rendered with Pillow so that the ground truth is known exactly.
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def have_tesseract() -> bool:
    return shutil.which("tesseract") is not None


def render_page(lines: list[tuple[int, int, str]], size: tuple[int, int] = (850, 1100),
                font_size: int = 22) -> Image.Image:
    """Draw (x, y, text) lines in black on white. Returns a greyscale page."""
    page = Image.new("L", size, 255)
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype(FONT, font_size)
    for x, y, text in lines:
        draw.text((x, y), text, fill=0, font=font)
    return page


def degrade(page: Image.Image, scale: float = 1.0, blur: float = 0.0,
            noise: float = 0.0, angle: float = 0.0, seed: int = 0) -> Image.Image:
    """Simulate a phone scan: downscale, blur, rotate, add pixel noise."""
    w, h = page.size
    img = page.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if angle:
        img = img.rotate(angle, expand=True, fillcolor=255)
    if noise:
        arr = np.asarray(img, dtype=np.float32)
        arr += np.random.default_rng(seed).normal(0, noise, arr.shape)
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img


@dataclass
class Word:
    text: str
    conf: float
    box: tuple[int, int, int, int]   # x1, y1, x2, y2
    block: int
    line: int


def tesseract(img: Image.Image, psm: int = 3) -> tuple[str, list[Word]]:
    """Run Tesseract once for plain text and once for TSV word boxes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()
    run = lambda *extra: subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", str(psm), *extra],
        input=png, capture_output=True, check=True).stdout.decode()
    text = run()
    words = []
    for row in run("tsv").splitlines()[1:]:
        f = row.split("\t")
        if len(f) == 12 and f[11].strip():
            x, y, w, h = map(int, f[6:10])
            words.append(Word(f[11], float(f[10]), (x, y, x + w, y + h), int(f[2]), int(f[4])))
    return text, words


def naive_order(words: list[Word]) -> str:
    """Sort by top edge, then left edge. Interleaves columns on multi-column pages."""
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.box[1], w.box[0])):
        if rows and abs(rows[-1][0].box[1] - w.box[1]) < 10:
            rows[-1].append(w)
        else:
            rows.append([w])
    return " ".join(w.text for r in rows for w in sorted(r, key=lambda w: w.box[0]))


def column_order(words: list[Word], gutter: int) -> str:
    """Split at a known gutter x, then read the left column before the right."""
    left = [w for w in words if w.box[2] <= gutter]
    right = [w for w in words if w.box[0] > gutter]
    return " ".join([naive_order(left), naive_order(right)]).strip()


def extract_total(text: str) -> str | None:
    m = re.search(r"total\s*due[:\s]*\$?\s*([0-9OoIl,]+\.[0-9OoIl]{2})", text, re.I)
    return m.group(1) if m else None
