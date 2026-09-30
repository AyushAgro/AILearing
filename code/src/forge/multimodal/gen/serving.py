"""Serving arithmetic and provenance for generated media: cost per image, cache keys,
a signed manifest bound to content bytes, and a toy spread-spectrum watermark."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GenRequest:
    model: str
    prompt: str
    seed: int
    steps: int
    guidance: float
    width: int
    height: int
    sampler: str

    def cache_key(self) -> str:
        """Every field that changes the pixels is in the key; nothing else is."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def cost_per_image(
    steps: int,
    sec_per_nfe: float,
    gpu_usd_per_hour: float,
    *,
    cfg_batched: bool = True,
    decode_sec: float = 0.0,
    utilisation: float = 1.0,
) -> dict:
    """Latency and cost of one image. Unbatched CFG doubles the denoiser calls."""
    nfe = steps * (1 if cfg_batched else 2)
    seconds = nfe * sec_per_nfe + decode_sec
    usd = seconds / 3600 * gpu_usd_per_hour / utilisation
    return {"nfe": nfe, "seconds": round(seconds, 3), "usd": round(usd, 5)}


def sign_manifest(content: bytes, claims: dict, key: bytes) -> dict:
    """A C2PA-shaped manifest: claims bound to the content hash, then signed.
    Real C2PA signs with X.509 certificates (COSE); HMAC keeps this runnable."""
    body = {"content_sha256": hashlib.sha256(content).hexdigest(), **claims}
    payload = json.dumps(body, sort_keys=True).encode()
    return {"body": body, "sig": hmac.new(key, payload, hashlib.sha256).hexdigest()}


def verify_manifest(content: bytes, manifest: dict, key: bytes) -> str:
    payload = json.dumps(manifest["body"], sort_keys=True).encode()
    if not hmac.compare_digest(
        hmac.new(key, payload, hashlib.sha256).hexdigest(), manifest["sig"]
    ):
        return "bad signature"
    if hashlib.sha256(content).hexdigest() != manifest["body"]["content_sha256"]:
        return "content changed"
    return "ok"


def watermark_pattern(key: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(key).choice([-1.0, 1.0], size=shape)


def embed_watermark(img: np.ndarray, key: int, strength: float) -> np.ndarray:
    return img + strength * watermark_pattern(key, img.shape)


def detect_watermark(img: np.ndarray, key: int) -> float:
    """z-score of the correlation with the keyed pattern. ~N(0, 1) if unmarked."""
    p = watermark_pattern(key, img.shape).ravel()
    x = img.ravel() - img.mean()
    return float(x @ p / (np.linalg.norm(x) + 1e-12))  # x.p ~ N(0, |x|^2) if unmarked
