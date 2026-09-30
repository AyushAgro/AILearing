"""Chapter demos for Volumes 19 and 20. Printed output is pasted into the book."""
from __future__ import annotations

import numpy as np

from forge.multimodal.audio import greedy_ctc, log_mel, mel_filterbank, quantise_mulaw
from forge.multimodal.clip import contrastive_loss, sigmoid_loss
from forge.multimodal.gen.diffuse import radius_error, sample_ddim, sample_ddpm, train_eps
from forge.multimodal.gen.flow import cfg_combine, sample_euler, train_velocity
from forge.multimodal.gen.metrics import (
    cosine, frechet_gaussian, sdedit_correlation, train_autoencoder, vector_quantise,
)
from forge.multimodal.gen.schedules import toy_schedule
from forge.multimodal.structs import (
    box_iou, character_error_rate, frame_delta, match_boxes, sample_frame_indices,
)
from forge.multimodal.vision import TinyViT, image_tokens, mae_visible_tokens


def chapter(title: str) -> None:
    print(f"\n===== {title} =====")


def demo_tokens_and_vit() -> None:
    chapter("vit")
    for height, patch in ((224, 16), (448, 16), (1024, 16), (1024, 32)):
        print(f"{height}px patch {patch}: {image_tokens(height, height, patch)} tokens")
    visible, masked = mae_visible_tokens(image_tokens(224, 224, 16), 0.75)
    print(f"mae 75% of 196 patches: visible={visible} masked={masked}")
    rng = np.random.default_rng(0)
    vit = TinyViT(rng, channels=3, size=8, patch=4, dim=16)
    counts = vit.parameter_count()
    print("params", counts)
    image = rng.normal(size=(2, 3, 8, 8))
    swapped = image.copy()
    swapped[:, :, :4, :] = image[:, :, 4:, :]
    swapped[:, :, 4:, :] = image[:, :, :4, :]
    out = vit.forward(image)
    base = vit.forward(image, use_pos=False)
    moved = vit.forward(swapped, use_pos=False)
    no_pos = np.max(np.abs(base[:, 1] - moved[:, 3]))
    held = vit.forward(image)
    held_swap = vit.forward(swapped)
    with_pos = np.max(np.abs(held[:, 1] - held_swap[:, 3]))
    print(f"output {out.shape} pos_pair_delta={with_pos:.4f} nopos_pair_delta={no_pos:.4e}")


def demo_clip() -> None:
    chapter("clip")
    rng = np.random.default_rng(1)
    image = rng.normal(size=(8, 16))
    text = image + rng.normal(scale=0.05, size=image.shape)
    shuffled = text[rng.permutation(len(text))]
    print(f"matched={contrastive_loss(image, text):.4f} shuffled={contrastive_loss(image, shuffled):.4f}")
    print(f"temp 1.0 matched={contrastive_loss(image, text, temperature=1.0):.4f}")
    print(f"sigmoid matched={sigmoid_loss(image, text):.4f}")


def demo_documents_audio_video() -> None:
    chapter("documents")
    ref = "total due 42.00"
    hyp = "total due 42.OO"
    print(f"cer={character_error_rate(ref, hyp):.3f} edits={ref} -> {hyp}")
    preds = np.array([[0, 0, 2, 2], [5, 5, 8, 7]], dtype=float)
    gts = np.array([[5.2, 5.1, 8.2, 7.1], [0.1, 0.2, 1.8, 2.2]], dtype=float)
    pairs, cost = match_boxes(preds, gts)
    print("iou diagonal", round(box_iou(preds[0], gts[0]), 3), "matched", pairs, "cost", round(cost, 3))

    chapter("audio")
    sr = 16_000
    t = np.arange(sr) / sr
    wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    mel = log_mel(wave[:1600], sr)
    peak = int(mel.mean(axis=0).argmax())
    freqs = np.linspace(0, sr / 2, 512 // 2 + 1)
    centre = float(freqs[mel_filterbank(sr, 512, 40)[peak].argmax()])
    print(f"logmel {mel.shape} peak_mel_bin={peak} centre_hz={centre:.0f}")
    blank, h, e, l, o = 0, 1, 2, 3, 4
    ids = [h, h, blank, e, e, l, l, blank, l, o]
    decoded = greedy_ctc(ids, blank)
    text = "".join({1: "h", 2: "e", 3: "l", 4: "o"}[i] for i in decoded)
    print("ctc", decoded, text)
    restored = quantise_mulaw(wave[:1000])
    err = float(np.mean((restored - wave[:1000]) ** 2))
    print(f"mulaw mse={err:.6f}")

    chapter("video")
    print("frames", sample_frame_indices(32, 8))
    static = np.ones((8, 4))
    shuffled = static.copy()
    shuffled[1::2] *= -1
    print(f"delta static={frame_delta(static):.3f} flicker={frame_delta(shuffled):.3f}")


def demo_generation() -> None:
    chapter("ddpm")
    rng = np.random.default_rng(2)
    net, schedule, losses = train_eps(rng, steps=500, batch=128, t_steps=20, lr=3e-2)
    prior = rng.normal(size=(256, 2))
    ddpm = sample_ddpm(net, schedule, rng, 256)
    ddim = sample_ddim(net, schedule, rng, 256, steps=5)
    print(f"loss start={losses[0]:.4f} end={losses[-1]:.4f}")
    print(f"radius prior={radius_error(prior):.3f} ddpm={radius_error(ddpm):.3f} "
          f"ddim5={radius_error(ddim):.3f}")
    ab = schedule["alpha_bar"]
    print(f"alpha_bar t0={ab[0]:.4f} t_last={ab[-1]:.4f}")

    chapter("flow")
    rng = np.random.default_rng(3)
    flow_net, flow_losses = train_velocity(rng, steps=500, batch=128, lr=3e-2)
    flowed = sample_euler(flow_net, rng, 256, steps=8)
    print(f"loss start={flow_losses[0]:.4f} end={flow_losses[-1]:.4f} "
          f"radius={radius_error(flowed):.3f}")

    chapter("guidance")
    uncond = np.array([0.2, 0.2])
    cond = np.array([1.0, 0.0])
    for scale in (0.0, 1.0, 3.0, 7.5):
        out = cfg_combine(uncond, cond, scale)
        print(f"scale={scale:<4} cosine_to_cond={cosine(out, cond):.3f} out={out}")

    chapter("latent")
    rng = np.random.default_rng(4)
    wide, _ = train_autoencoder(rng, latent=2, steps=400)
    rng = np.random.default_rng(4)
    narrow, _ = train_autoencoder(rng, latent=1, steps=400)
    print(f"recon mse latent2={wide:.4f} latent1={narrow:.4f}")
    rng = np.random.default_rng(5)
    frames = rng.normal(size=(64, 4))
    codebook = rng.normal(size=(8, 4))
    _idx, mse = vector_quantise(frames, codebook)
    print(f"vq mse={mse:.4f} codes=8")

    chapter("edit-metrics")
    rng = np.random.default_rng(6)
    x = rng.normal(size=(64,))
    schedule = toy_schedule(20)
    for t in (0, 5, 19):
        corr = sdedit_correlation(x, float(schedule["alpha_bar"][t]), rng)
        print(f"t={t:02d} alpha_bar={schedule['alpha_bar'][t]:.3f} corr={corr:.3f}")
    mu1 = np.zeros(2)
    cov1 = np.eye(2)
    mu2 = np.array([1.0, 0.0])
    cov2 = np.eye(2) * 0.5
    print(f"frechet={frechet_gaussian(mu1, cov1, mu2, cov2):.4f}")
    print(f"frechet identical={frechet_gaussian(mu1, cov1, mu1, cov1):.4e}")
    print(f"cosine matched={cosine(cond, cond):.3f} unrelated={cosine(cond, uncond):.3f}")


def main() -> None:
    demo_tokens_and_vit()
    demo_clip()
    demo_documents_audio_video()
    demo_generation()


if __name__ == "__main__":
    main()
