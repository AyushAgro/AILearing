"""Tests for the multimodal understanding and generation demos."""
from __future__ import annotations

import numpy as np

from forge.multimodal.audio import greedy_ctc
from forge.multimodal.clip import contrastive_loss
from forge.multimodal.gen.diffuse import radius_error, train_eps, sample_ddpm
from forge.multimodal.gen.flow import cfg_combine
from forge.multimodal.gen.metrics import frechet_gaussian
from forge.multimodal.structs import box_iou, character_error_rate, sample_frame_indices
from forge.multimodal.vision import image_tokens, mae_visible_tokens


def test_image_tokens_and_mae() -> None:
    assert image_tokens(224, 224, 16) == 196
    assert image_tokens(1024, 1024, 16) == 4096
    visible, masked = mae_visible_tokens(196, 0.75)
    assert visible == 49 and masked == 147


def test_contrastive_prefers_matched_pairs() -> None:
    rng = np.random.default_rng(0)
    image = rng.normal(size=(6, 8))
    text = image + rng.normal(scale=0.01, size=image.shape)
    shuffled = np.roll(text, 1, axis=0)
    assert contrastive_loss(image, text) < contrastive_loss(image, shuffled)


def test_ctc_and_cer() -> None:
    assert greedy_ctc([1, 1, 0, 2, 2, 2]) == [1, 2]
    assert character_error_rate("ab", "ab") == 0.0
    assert character_error_rate("ab", "ac") == 0.5


def test_iou_and_frames() -> None:
    box = np.array([0.0, 0.0, 2.0, 2.0])
    assert box_iou(box, box) == 1.0
    assert sample_frame_indices(32, 8)[0] == 0
    assert sample_frame_indices(32, 8)[-1] == 31


def test_frechet_identical_is_zero() -> None:
    mu = np.zeros(2)
    cov = np.eye(2)
    assert frechet_gaussian(mu, cov, mu, cov) < 1e-8


def test_cfg_endpoints() -> None:
    uncond = np.array([0.0, 1.0])
    cond = np.array([1.0, 0.0])
    assert np.allclose(cfg_combine(uncond, cond, 0.0), uncond)
    assert np.allclose(cfg_combine(uncond, cond, 1.0), cond)


def test_ddpm_loss_drops() -> None:
    rng = np.random.default_rng(0)
    _net, _schedule, losses = train_eps(rng, steps=200, batch=64, t_steps=10, lr=3e-2)
    assert losses[-1] < losses[0]
    samples = sample_ddpm(_net, _schedule, rng, 64)
    assert radius_error(samples) < radius_error(rng.normal(size=(64, 2)))


# ---------------------------------------------------------------- Volume 19 additions


def test_patch_embed_conv_matches_patchify() -> None:
    import torch

    from forge.multimodal.vision import patchify, unpatchify
    from forge.multimodal.vit_torch import PatchEmbed

    torch.manual_seed(0)
    pe = PatchEmbed(3, 4, 8)
    x = torch.randn(2, 3, 8, 12)
    w = pe.proj.weight.detach().numpy().transpose(2, 3, 1, 0).reshape(-1, 8)
    ours = patchify(x.numpy(), 4) @ w + pe.proj.bias.detach().numpy()
    assert np.abs(ours - pe(x).detach().numpy()).max() < 1e-5
    assert np.allclose(unpatchify(patchify(x.numpy(), 4), 4, 3, 8, 12), x.numpy())


def test_tiles_and_dynamic_resize() -> None:
    from forge.multimodal.vision import plan_tiles, random_masking, smart_resize

    h, w = smart_resize(1100, 850)
    assert h % 28 == 0 and w % 28 == 0 and h * w <= 1280 * 28 * 28
    assert plan_tiles(600, 2400, 448, 12)["grid"] == (1, 4)
    keep, mask, _ = random_masking(np.random.default_rng(0), 2, 196, 0.75)
    assert keep.shape == (2, 49) and mask.sum() == 2 * 147


def test_contrastive_matches_torch() -> None:
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(8, 16)), rng.normal(size=(8, 16))
    ta, tb = F.normalize(torch.tensor(a), dim=-1), F.normalize(torch.tensor(b), dim=-1)
    lg = ta @ tb.T / 0.07
    lab = torch.arange(8)
    ref = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)).item()
    assert abs(contrastive_loss(a, b, 0.07) - ref) < 1e-9


def test_retrieval_and_maxsim() -> None:
    from forge.multimodal.retrieval import ExactIndex, maxsim, recall_at_k

    eye = np.eye(4)
    ids, scores = ExactIndex(eye).search(eye[[2]], k=2)
    assert ids[0, 0] == 2 and scores[0, 0] >= scores[0, 1]
    assert recall_at_k(ids, [[2]], 1) == 1.0
    assert maxsim(eye[[1]], eye[[0, 1]]) == 1.0


def test_detection_ops_match_torchvision() -> None:
    import torch
    import torchvision.ops as tvo

    from forge.multimodal.detect import average_precision, giou_matrix, hungarian, iou_matrix, nms

    rng = np.random.default_rng(1)
    a = rng.uniform(0, 50, (5, 2))
    a = np.concatenate([a, a + rng.uniform(5, 20, (5, 2))], 1)
    ref = tvo.box_iou(torch.tensor(a), torch.tensor(a)).numpy()
    assert np.allclose(iou_matrix(a, a), ref)
    assert np.allclose(giou_matrix(a, a), tvo.generalized_box_iou(torch.tensor(a), torch.tensor(a)).numpy())
    scores = rng.uniform(size=5)
    assert nms(a, scores, 0.5) == tvo.nms(torch.tensor(a), torch.tensor(scores), 0.5).tolist()
    assert sorted(hungarian(1 - iou_matrix(a, a[::-1]))) == [(i, 4 - i) for i in range(5)]
    gts = {0: a[:2]}
    perfect = [(0, 0.9, a[0]), (0, 0.8, a[1])]
    assert average_precision(perfect, gts) == 1.0
    assert average_precision(perfect + [(0, 0.95, a[0])], gts) < 1.0


def test_ctc_loss_matches_torch_and_beam() -> None:
    import torch
    import torch.nn.functional as F

    from forge.multimodal.audio import ctc_loss, prefix_beam_search, word_error_rate

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(12, 5))
    lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    for tgt in ([1, 2, 3], [3, 3]):
        ref = F.ctc_loss(torch.tensor(lp)[:, None], torch.tensor([tgt]), torch.tensor([12]),
                         torch.tensor([len(tgt)]), reduction="sum").item()
        assert abs(ctc_loss(lp, tgt) - ref) < 1e-6
    best, p = prefix_beam_search(np.array([[0.6, 0.4], [0.6, 0.4]]), beam=4)[0]
    assert best == (1,) and abs(p - 0.64) < 1e-9
    assert abs(word_error_rate("the cat sat on the mat", "the cat sat on mat") - 1 / 6) < 1e-9


def test_eval_and_prod_helpers() -> None:
    from forge.multimodal.mm_eval import pope_metrics, relaxed_match, vqa_soft_accuracy
    from forge.multimodal.prod import tokens_openai_high
    from forge.multimodal.video import attention_pairs, change_indices, make_clip

    always_yes = pope_metrics(np.ones(4, bool), np.array([True, False, True, False]))
    assert always_yes["recall"] == 1.0 and always_yes["yes_ratio"] == 1.0
    assert relaxed_match("41.2", "42") and not relaxed_match("39", "42")
    assert vqa_soft_accuracy("Two", ["2", "2", "two"]) == 1.0
    assert tokens_openai_high(1024, 1024) == 765
    frames, labels = make_clip(np.random.default_rng(0), n_frames=60, event=(30, 36))
    assert any(labels[i] != "empty" for i in change_indices(frames, 4, min_gap=2))
    assert attention_pairs(8, 16, "divided") < attention_pairs(8, 16, "joint")
