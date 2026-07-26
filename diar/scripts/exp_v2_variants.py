#!/usr/bin/env python3
"""One-shot A/B harness for pipeline_v2 variants on the reference meeting.

Computes local embeddings once per embedding model (seg output is cached by
pipeline_v2), then scores clustering/output variants in-process.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diar_lab import pipeline_v2 as v2
from diar_lab.compare_benchmark import best_map_acc, parse_gt, raster, turn_majority
from diar_lab.fbank import compute_fbank_knf
from diar_lab.pipeline import EmbeddingONNX, Turn, ahc_labels, load_wav
from diar_lab.plda import PldaTransform, l2_norm

WAV = ROOT / "data/audio/meeting_06-29.wav"
GT = ROOT / "data/gt/meeting_06-29_transcript.md"


def extract_locals(pcm, sr, starts, win_hard, fhz, emb_path):
    emb = EmbeddingONNX(emb_path)
    locals_ = []
    for wi, s in enumerate(starts):
        chunk = pcm[s : s + int(v2.WIN_SEC * sr)]
        hard = win_hard[wi]
        n_active = hard.sum(axis=1)
        fb = None
        for k in range(hard.shape[1]):
            active = hard[:, k] > 0.5
            act_sec = float(active.sum()) / fhz
            if act_sec < v2.MIN_EMB_SEC:
                continue
            if fb is None:
                fb = compute_fbank_knf(chunk, sample_rate=sr, subtract_mean=False)
            solo = active & (n_active < 1.5)
            vv = v2._masked_embedding(emb, fb, active, solo, fhz)
            if vv is None:
                continue
            locals_.append(v2.LocalSpeaker(wi, k, act_sec, vv))
    return locals_


def aggregate(locals_, labels, starts, win_hard, fhz, sr, dur, k):
    n_frames = int(np.ceil(dur * fhz)) + 1
    agg = np.zeros((n_frames, k))
    wsum = np.zeros((n_frames, k))
    t_local = np.arange(win_hard[0].shape[0])
    tri = 1.0 - np.abs(t_local - t_local.mean()) / (t_local.mean() + 1.0)
    tri = 0.1 + 0.9 * tri
    lab_of = {(ls.win_idx, ls.local_idx): int(l) for ls, l in zip(locals_, labels)}
    for wi, s in enumerate(starts):
        f0 = int(round(s / sr * fhz))
        hard = win_hard[wi]
        t_len = min(hard.shape[0], n_frames - f0)
        for kk in range(hard.shape[1]):
            g = lab_of.get((wi, kk))
            if g is None:
                continue
            agg[f0 : f0 + t_len, g] += hard[:t_len, kk] * tri[:t_len]
            wsum[f0 : f0 + t_len, g] += tri[:t_len]
    return np.where(wsum > 0, agg / np.maximum(wsum, 1e-9), 0.0)


def turns_overlap(act, fhz, k):
    turns = []
    for g in range(k):
        on = act[:, g] >= v2.ONSET
        on = v2._fill_gaps(on, int(v2.MIN_OFF_SEC * fhz))
        on = v2._drop_short(on, int(v2.MIN_ON_SEC * fhz))
        for a, b in v2._runs(on):
            turns.append(Turn(a / fhz, b / fhz, g))
    turns.sort(key=lambda t: t.start)
    return turns


def turns_primary(act, fhz, k):
    """Single-label: per frame, the most active speaker (if any ≥ onset)."""
    lab = np.where(act.max(axis=1) >= v2.ONSET, act.argmax(axis=1), -1)
    # per-speaker fill/drop on the single-label track
    turns = []
    for g in range(k):
        on = lab == g
        on = v2._fill_gaps(on, int(v2.MIN_OFF_SEC * fhz))
        on = v2._drop_short(on, int(v2.MIN_ON_SEC * fhz))
        for a, b in v2._runs(on):
            turns.append(Turn(a / fhz, b / fhz, g))
    turns.sort(key=lambda t: t.start)
    return turns


def score(turns, name):
    tl = [{"start": t.start, "end": t.end, "speaker": t.speaker} for t in turns]
    dur = max(t["end"] for t in tl)
    gt = parse_gt(GT, "md-zh", dur)
    step = 0.1
    n = int(dur / step) + 2
    R = raster(gt, step, n)
    H = raster(tl, step, n)
    acc, mp, miss, fa, conf, der = best_map_acc(R, H)
    maj, ok, tot = turn_majority(gt, H, step)
    k = len({t.speaker for t in turns})
    print(
        f"{name:34s} acc={acc*100:5.1f}% der={der*100:5.1f}% "
        f"miss/fa/conf={miss*100:4.1f}/{fa*100:4.1f}/{conf*100:4.1f} "
        f"maj={maj*100:4.1f}% spk={k}"
    )


def main():
    pcm, sr = load_wav(WAV)
    dur = len(pcm) / sr
    starts = v2._sliding_windows(len(pcm), sr)
    win_hard, fhz = v2._segment_all(pcm, sr, v2.DEFAULT_SEG, starts)

    plda = PldaTransform.load(v2.DEFAULT_PLDA)

    for emb_name, emb_path, spaces in [
        ("vox", v2.DEFAULT_EMB, ["xvec", "raw"]),
        ("cnceleb", ROOT / "models/emb_cnceleb.onnx", ["raw"]),
    ]:
        locals_ = extract_locals(pcm, sr, starts, win_hard, fhz, emb_path)
        E = np.stack([ls.emb for ls in locals_])
        dur_arr = np.array([ls.active_sec for ls in locals_])
        for space in spaces:
            x = plda.xvec_transform(E) if space == "xvec" else l2_norm(E)
            labels = ahc_labels(x, max_speakers=8, min_cluster_size=1)
            labels = v2._absorb_by_duration(x, labels, dur_arr, v2.MIN_CLUSTER_SEC)
            k = int(labels.max()) + 1
            act = aggregate(locals_, labels, starts, win_hard, fhz, sr, dur, k)
            score(turns_overlap(act, fhz, k), f"{emb_name}/{space} overlap-turns")
            score(turns_primary(act, fhz, k), f"{emb_name}/{space} primary-label")


if __name__ == "__main__":
    main()
