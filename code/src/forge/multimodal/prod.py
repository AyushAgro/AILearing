"""Production helpers: image-token estimators, a request budget gate, EXIF-safe
loading, perceptual-hash de-duplication and region redaction.

The vendor estimators restate formulas from public documentation (see Volume 19,
Chapter 11). Vendors change them between model versions: treat them as planning
estimates and reconcile against billed usage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from forge.multimodal.vision import smart_resize


def tokens_openai_high(width: int, height: int) -> int:
    """Fit in 2048x2048, shrink so the short side is at most 768, then
    85 + 170 per 512-pixel tile (published for GPT-4o-class models, high detail)."""
    s = min(1.0, 2048 / max(width, height))
    w, h = width * s, height * s
    s = min(1.0, 768 / min(w, h))
    w, h = w * s, h * s
    return 85 + 170 * math.ceil(w / 512) * math.ceil(h / 512)


def tokens_anthropic(width: int, height: int) -> int:
    """About width*height/750, after shrinking so the long edge is at most 1568 px
    and the estimate is at most about 1600 tokens (published guidance)."""
    s = min(1.0, 1568 / max(width, height))
    s = min(s, math.sqrt(1600 * 750 / (width * height)))
    return math.ceil(width * s * height * s / 750)


def tokens_gemini(width: int, height: int) -> int:
    """258 tokens if both sides are at most 384 px, else 258 per 768x768 tile
    (published for Gemini 2.x image input)."""
    if width <= 384 and height <= 384:
        return 258
    return 258 * math.ceil(width / 768) * math.ceil(height / 768)


def tokens_qwen2vl(width: int, height: int) -> int:
    """Patch 14 with 2x2 merging = one token per 28x28 pixels after smart_resize,
    plus the two vision start/end markers."""
    h, w = smart_resize(height, width)
    return (h // 28) * (w // 28) + 2


@dataclass
class Budget:
    max_side: int = 2048
    max_pixels: int = 4_000_000
    max_images: int = 20
    max_audio_s: float = 1800.0
    max_video_frames: int = 64


def check_request(images: list[tuple[int, int]], audio_s: float, video_frames: int,
                  budget: Budget) -> list[str]:
    """Return the list of violations; empty means admit. Checked before any decode."""
    problems = []
    if len(images) > budget.max_images:
        problems.append(f"{len(images)} images > {budget.max_images}")
    for i, (w, h) in enumerate(images):
        if max(w, h) > budget.max_side or w * h > budget.max_pixels:
            problems.append(f"image {i} is {w}x{h}")
    if audio_s > budget.max_audio_s:
        problems.append(f"audio {audio_s:.0f}s > {budget.max_audio_s:.0f}s")
    if video_frames > budget.max_video_frames:
        problems.append(f"{video_frames} frames > {budget.max_video_frames}")
    return problems


def load_upright(img: Image.Image) -> Image.Image:
    """Apply the EXIF orientation tag, then drop metadata (GPS, device) by copying
    pixels into a fresh image."""
    upright = ImageOps.exif_transpose(img)
    clean = Image.new(upright.mode, upright.size)
    clean.putdata(list(upright.getdata()))
    return clean


def dhash(img: Image.Image, size: int = 8) -> int:
    """Difference hash: greyscale, resize to (size+1, size), compare neighbours."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def redact(img: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    """Fill boxes with solid black. Blur is not redaction: it can be reversed or read."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        draw.rectangle(box, fill=0)
    return out
