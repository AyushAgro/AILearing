"""Tests for forge.multimodal.gen (Volume 20). All run at Tier L in seconds."""
from __future__ import annotations

import numpy as np
import pytest

from forge.multimodal.gen.codec import (ResidualVQ, delay_pattern, frame, istft,
                                        mu_law_decode, mu_law_encode, snr_db, stft,
                                        synth_voice, undo_delay)
from forge.multimodal.gen.ddpm import (cosine_schedule, ddim_sample, linear_schedule,
                                       q_sample, train_ddpm)
from forge.multimodal.gen.evalmedia import (clip_score, frechet_from_samples,
                                            knn_precision_recall, wer)
from forge.multimodal.gen.flowmatch import sample_euler, train_fm, velocity
from forge.multimodal.gen.nets import MLP
from forge.multimodal.gen.serving import (GenRequest, detect_watermark, embed_watermark,
                                          sign_manifest, verify_manifest)
from forge.multimodal.gen.toy2d import eight_gaussians, mode_report
from forge.multimodal.gen.video import correlated_noise, flicker


def test_mlp_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(0)
    net = MLP([3, 8, 8, 2], rng)
    x, y = rng.normal(size=(5, 3)), rng.normal(size=(5, 2))
    pred, cache = net.forward(x)
    grads, _ = net.backward(cache, 2 * (pred - y) / (pred - y).size)
    p, h = net.params[0], 1e-6
    p[1, 2] += h
    up = np.mean((net(x) - y) ** 2)
    p[1, 2] -= 2 * h
    down = np.mean((net(x) - y) ** 2)
    p[1, 2] += h
    assert abs((up - down) / (2 * h) - grads[0][1, 2]) < 1e-7


def test_schedules_reach_noise() -> None:
    assert linear_schedule(1000).alpha_bar[-1] < 1e-4
    assert cosine_schedule(1000).alpha_bar[-1] < 1e-4
    assert np.all(np.diff(linear_schedule(1000).alpha_bar) < 0)


def test_forward_process_moments() -> None:
    rng = np.random.default_rng(0)
    s = linear_schedule(1000)
    x0 = np.full((100_000, 2), 2.0)
    xt = q_sample(x0, np.full(len(x0), 500), s, rng.normal(size=x0.shape))
    assert abs(xt.mean() - 2 * np.sqrt(s.alpha_bar[500])) < 0.02
    assert abs(xt.var() - (1 - s.alpha_bar[500])) < 0.02


@pytest.fixture(scope="module")
def eps_model():
    rng = np.random.default_rng(0)
    data, _ = eight_gaussians(rng, 5000)
    s = linear_schedule(200)
    net, losses = train_ddpm(data, s, rng, steps=1500, hidden=64)
    return net, s, losses


def test_ddpm_learns_and_ddim_is_deterministic(eps_model) -> None:
    net, s, losses = eps_model
    assert np.mean(losses[-100:]) < np.mean(losses[:100])
    x_T = np.random.default_rng(3).normal(size=(500, 2))
    a = ddim_sample(net, s, np.random.default_rng(1), 500, 20, x_T=x_T)
    b = ddim_sample(net, s, np.random.default_rng(2), 500, 20, x_T=x_T)
    assert np.array_equal(a, b)
    assert mode_report(a)["precision"] > mode_report(x_T)["precision"] + 0.3


def test_flow_matching_and_cfg_endpoints() -> None:
    rng = np.random.default_rng(0)
    data, labels = eight_gaussians(rng, 5000)
    net, losses = train_fm(data, rng, steps=1500, hidden=64, labels=labels, n_classes=8)
    assert np.mean(losses[-100:]) < np.mean(losses[:100])
    z, y = rng.normal(size=(4, 2)), np.arange(4)
    v_c = velocity(net, z, 0.5, y, 8)
    v_u = velocity(net, z, 0.5, np.full(4, 8), 8)
    assert np.allclose(velocity(net, z, 0.5, y, 8, guidance=1.0), v_c)
    assert np.allclose(velocity(net, z, 0.5, y, 8, guidance=3.0), v_u + 3.0 * (v_c - v_u))
    x = sample_euler(net, rng.normal(size=(400, 2)), 16, np.zeros(400, int), 8, 2.0)
    assert mode_report(x)["precision"] > 0.5


def test_codec_pieces() -> None:
    rng = np.random.default_rng(0)
    wave = synth_voice(rng, 0.5)
    assert snr_db(wave, mu_law_decode(mu_law_encode(wave))) > 30
    frames = frame(np.concatenate([synth_voice(rng, 1.0) for _ in range(3)]), 32)
    rvq = ResidualVQ(4, 64).fit(frames, rng)
    snrs = [snr_db(frames.ravel(), rvq.decode(rvq.encode(frames, q)).ravel())
            for q in (1, 2, 4)]
    assert snrs[0] < snrs[1] < snrs[2]
    codes = np.arange(12).reshape(3, 4)
    assert np.array_equal(undo_delay(delay_pattern(codes)), codes)
    spec = stft(wave)
    assert snr_db(wave[512:-512], istft(spec, len(wave))[512:-512]) > 100


def test_metrics() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4000, 4))
    assert frechet_from_samples(a, rng.normal(size=(4000, 4))) < 0.05
    real, _ = eight_gaussians(rng, 600)
    blob = real.mean(0) + rng.normal(size=(600, 2)) * real.std(0)
    p, r = knn_precision_recall(real, blob)
    assert p < 0.5 < r
    assert np.allclose(clip_score(np.array([[1.0, 0.0]]), np.array([[-1.0, 0.0]])), 0.0)
    assert wer("a b c d", "a b c d") == 0.0 and wer("a b c d", "a x c") == 0.5


def test_video_noise_and_flicker() -> None:
    z = correlated_noise(np.random.default_rng(0), 50, 200, 4, 0.9)
    assert abs(z.std() - 1.0) < 0.05
    assert flicker(np.repeat(z[:1], 5, axis=0)) == 0.0


def test_serving_provenance_and_watermark() -> None:
    a = GenRequest("m@1", "cat", 1, 20, 5.0, 512, 512, "euler")
    b = GenRequest("m@1", "cat", 1, 20, 7.5, 512, 512, "euler")
    assert a.cache_key() != b.cache_key()
    key = b"k"
    man = sign_manifest(b"bytes", {"generator": "x"}, key)
    assert verify_manifest(b"bytes", man, key) == "ok"
    assert verify_manifest(b"bytez", man, key) == "content changed"
    img = np.random.default_rng(0).uniform(-1, 1, (64, 64))
    assert detect_watermark(embed_watermark(img, 42, 0.05), 42) > 4
    assert abs(detect_watermark(img, 42)) < 4
