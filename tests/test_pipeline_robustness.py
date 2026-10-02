"""Robustness perturbations: correct strength, monotone severity, and the degraded-input
path working end to end on a (synthetic) clip cache."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.pipeline import robustness as R


def _frames(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (16, 224, 224, 3), dtype=np.uint8)
    import cv2
    return np.stack([cv2.GaussianBlur(f, (0, 0), 2) for f in base])  # natural-ish texture


def test_noise_hits_requested_snr():
    rng = np.random.default_rng(0)
    w = (0.1 * np.sin(np.linspace(0, 400 * np.pi, 64000))).astype(np.float32)
    for snr in (20, 10, 0):
        n = R._noise(w, snr, np.random.default_rng(1)) - w
        got = 10 * np.log10(np.mean(w ** 2) / np.mean(n ** 2))
        assert abs(got - snr) < 0.5


def test_video_degradations_are_monotone():
    f = _frames()
    err = lambda g: float(np.mean(np.abs(g.astype(float) - f.astype(float))))
    assert err(R._downscale(f, 2)) < err(R._downscale(f, 4))
    assert err(R._blur(f, 1)) < err(R._blur(f, 2))
    e = [err(R._h264(f, c)[0]) for c in (23, 30, 40)]
    assert e[0] <= e[1] <= e[2]


def test_telephone_band_removes_high_frequencies():
    t = np.arange(64000) / 16000
    w = (0.2 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)   # above 4 kHz Nyquist of 8 kHz
    out = R._telephone(w)
    assert out.shape == w.shape
    assert np.mean(out ** 2) < 0.05 * np.mean(w ** 2)


class _Cache:
    def __init__(self, blobs):
        self.where = {k: None for k in blobs}
        self._b = blobs

    def blob(self, cid):
        return self._b[cid]


class _Extractor:
    def video(self, x):              # (B,16,3,H,W) -> (B,4,8)
        return x.mean(dim=(2, 3, 4)).view(x.shape[0], 4, 4).repeat(1, 1, 2).half()

    def audio(self, x):              # (B,N) -> (B,5,8)
        return x.view(x.shape[0], 5, -1)[:, :, :8].half()


def test_perturbed_feature_table_and_pixels_on_synthetic_cache():
    from src.data.clipcache import encode_clip
    recs, blobs = [], {}
    for i in range(3):
        cid = f"c{i}"
        frames = np.concatenate([_frames(i), _frames(i + 9)[:8]])          # 24 frames
        audio = (np.random.default_rng(i).normal(0, 0.1, 96000) * 32767).astype(np.int16)
        blobs[cid] = encode_clip(frames, audio, has_video=True, has_audio=True, duration=6.0,
                                 span_start=0.0)
        recs.append({"clip_id": cid, "video_label": i % 2, "audio_label": 0, "clip_label": i % 2,
                     "quadrant": "FVRA" if i % 2 else "RVRA", "generator": "g", "dataset": "d",
                     "meta": {}, "video_segments": [], "audio_segments": []})
    cache = _Cache(blobs)
    P = R.perturbations()
    t, codecs = R.perturbed_feature_table(recs, cache, "blur_s2", P["blur_s2"], _Extractor())
    assert t.n == 3 and set(t.rows) == {"c0", "c1", "c2"}
    clean, _ = R.perturbed_feature_table(recs, cache, "clean", P["clean"], _Extractor())
    assert not torch.equal(clean.video, t.video)          # the degradation reached the features
    assert torch.equal(clean.audio, t.audio)               # a video perturbation leaves audio alone
    noisy, _ = R.perturbed_feature_table(recs, cache, "noise_snr0", P["noise_snr0"], _Extractor())
    assert torch.equal(clean.video, noisy.video) and not torch.equal(clean.audio, noisy.audio)
    ds = R.PerturbedPixelDataset(recs, cache, "downscale_x4", P["downscale_x4"])
    item = ds[0]
    assert item["video"].shape == (16, 3, 224, 224) and item["audio"].shape == (64000,)


def test_robustness_rows_aggregate_over_seeds():
    res = [{"name": "m", "seed": s, "per_perturbation": {"clean": {"video_auc": 0.9 + s / 1000},
                                                        "blur_s1": {"video_auc": 0.8}}}
           for s in (1, 2, 3)]
    rows = {r["Perturbation"]: r for r in R.robustness_rows(res)}
    assert rows["clean"]["Seeds"] == 3
    assert rows["clean"]["Video AUC"] == pytest.approx(0.902)
