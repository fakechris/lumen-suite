"""Diarization v2: local segmentation + speaker-masked embeddings.

The v1 pipeline collapses the DiariZen powerset output into a binary VAD
mask and slides fixed 1.5 s windows over it — speaker changes inside a
window contaminate the embedding, and short back-channel turns never get
a clean window, so small speakers are absorbed. v2 uses the segmentation
model as intended (pyannote-3.x style):

  wav
    → seg.onnx on overlapping 16 s windows (hop 8 s)
    → per-window, per-local-speaker binary activity (powerset argmax)
    → one embedding per (window, local speaker), fbank frames masked to
      that speaker's active frames (solo frames preferred)
    → xvec transform → AHC (cosine + 2-GMM), duration-based min cluster
    → map each (window, local speaker) → global cluster
    → overlap-add aggregation of activities into global speaker tracks
    → primary-label frame track → min-on/min-off cleanup → turns
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .fbank import compute_fbank_knf
from .pipeline import (
    DEFAULT_EMB,
    DEFAULT_PLDA,
    DEFAULT_SEG,
    DiarizationResult,
    EmbeddingONNX,
    SegmentationONNX,
    Turn,
    ahc_labels,
    load_wav,
)
from .plda import PldaTransform, l2_norm

# window geometry
WIN_SEC = 16.0
HOP_SEC = 8.0

# per-(window, speaker) embedding requirements
MIN_EMB_SEC = 0.15      # min active speech to emit an embedding
SOLO_PREF_SEC = 0.6     # if this much solo speech exists, use solo frames only

# clustering
MIN_CLUSTER_SEC = 2.0   # absorb clusters with less total speech than this

# binarization of aggregated activity
ONSET = 0.45
MIN_ON_SEC = 0.20
MIN_OFF_SEC = 0.50


@dataclass
class LocalSpeaker:
    win_idx: int
    local_idx: int
    active_sec: float
    emb: np.ndarray  # raw 256-d


def _sliding_windows(n: int, sr: int) -> List[int]:
    """Start samples of 16 s windows covering the file (hop 8 s)."""
    win = int(WIN_SEC * sr)
    hop = int(HOP_SEC * sr)
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    return starts


def _binary_multilabel(seg: SegmentationONNX, logits_probs: np.ndarray) -> np.ndarray:
    """Powerset probs [T,11] → hard multilabel [T,4] via argmax class."""
    hard = np.zeros((logits_probs.shape[0], 4), dtype=np.float64)
    top = logits_probs.argmax(axis=-1)
    for c, members in enumerate(seg.classes):
        rows = top == c
        for s in members:
            hard[rows, s] = 1.0
    return hard


def _seg_probs(seg: SegmentationONNX, chunk: np.ndarray) -> np.ndarray:
    """Run seg model on one 16 s chunk → class probabilities [T,11]."""
    x = np.asarray(chunk, dtype=np.float32).ravel()
    if x.size < seg.CHUNK_SAMPLES:
        x = np.pad(x, (0, seg.CHUNK_SAMPLES - x.size))
    out = seg.sess.run(None, {"waveforms": x[None, None, :]})[0][0]
    z = out - out.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / (p.sum(axis=-1, keepdims=True) + 1e-12)


def _masked_embedding(
    emb: EmbeddingONNX,
    fbank: np.ndarray,
    active: np.ndarray,
    solo: np.ndarray,
    fhz: float,
) -> Optional[np.ndarray]:
    """Embed fbank frames where the speaker is active (solo preferred).

    fbank: [T100, 80] at 100 fps; active/solo: seg-frame booleans at fhz.
    """
    t100 = fbank.shape[0]
    idx100 = np.minimum(
        ((np.arange(t100) + 1.25) / 100.0 * fhz).astype(int), len(active) - 1
    )
    solo100 = solo[idx100]
    act100 = active[idx100]
    use = solo100 if solo100.sum() / 100.0 >= SOLO_PREF_SEC else act100
    if use.sum() < int(MIN_EMB_SEC * 100):
        return None
    sel = fbank[use]
    sel = sel - sel.mean(axis=0, keepdims=True)  # CMN over this speaker's frames
    return np.asarray(emb(sel), dtype=np.float64)


def diarize_v2(
    wav_path: Path,
    emb_path: Path = DEFAULT_EMB,
    seg_path: Path = DEFAULT_SEG,
    plda_dir: Path = DEFAULT_PLDA,
    ahc_threshold_bias: float = 0.0,
    max_speakers: int = 8,
    cluster_space: str = "xvec",   # "xvec" (PLDA LDA 128-d) | "raw" (256-d cosine)
    prepad: float = 0.0,           # extend turn starts backward (s); trades acc → DER
) -> DiarizationResult:
    t0 = time.time()
    pcm, sr = load_wav(Path(wav_path))
    assert sr == 16000
    dur = len(pcm) / sr
    print(f"[1/5] load: {dur/60:.2f} min", flush=True)

    emb = EmbeddingONNX(emb_path)

    starts = _sliding_windows(len(pcm), sr)
    print(f"[2/5] segmentation on {len(starts)} windows (16s/8s)", flush=True)
    win_hard, fhz = _segment_all(pcm, sr, seg_path, starts)

    locals_: List[LocalSpeaker] = []
    for wi, s in enumerate(starts):
        chunk = pcm[s : s + int(WIN_SEC * sr)]
        hard = win_hard[wi]
        n_active = hard.sum(axis=1)
        fb = None
        for k in range(hard.shape[1]):
            active = hard[:, k] > 0.5
            act_sec = float(active.sum()) / fhz
            if act_sec < MIN_EMB_SEC:
                continue
            if fb is None:
                fb = compute_fbank_knf(chunk, sample_rate=sr, subtract_mean=False)
            solo = active & (n_active < 1.5)
            v = _masked_embedding(emb, fb, active, solo, fhz)
            if v is None:
                continue
            locals_.append(LocalSpeaker(wi, k, act_sec, v))

    if not locals_ or fhz is None:
        raise RuntimeError("segmentation found no speech")
    print(f"  local speakers with embeddings: {len(locals_)}", flush=True)

    print(f"[3/5] cluster ({cluster_space}) + AHC", flush=True)
    E = np.stack([ls.emb for ls in locals_])
    if cluster_space == "xvec":
        plda = PldaTransform.load(plda_dir)
        x128 = plda.xvec_transform(E)
    else:
        x128 = l2_norm(E)
    labels = ahc_labels(
        x128,
        threshold_bias=ahc_threshold_bias,
        max_speakers=max_speakers,
        min_cluster_size=1,  # size handled by duration below
        allow_single=True,   # a memo/dictation really can be one speaker
    )
    labels = _absorb_by_duration(
        x128, labels, np.array([ls.active_sec for ls in locals_]), MIN_CLUSTER_SEC
    )
    k = int(labels.max()) + 1
    sec_per = {
        g: round(sum(ls.active_sec for ls, l in zip(locals_, labels) if l == g), 1)
        for g in range(k)
    }
    print(f"  global speakers={k} speech_sec={sec_per}", flush=True)

    print("[4/5] overlap-add frame aggregation", flush=True)
    n_frames = int(np.ceil(dur * fhz)) + 1
    agg = np.zeros((n_frames, k))
    wsum = np.zeros((n_frames, k))
    # triangular weight de-emphasizes window borders where the model is weak
    t_local = np.arange(win_hard[0].shape[0])
    tri = 1.0 - np.abs(t_local - t_local.mean()) / (t_local.mean() + 1.0)
    tri = 0.1 + 0.9 * tri
    lab_of: Dict[Tuple[int, int], int] = {
        (ls.win_idx, ls.local_idx): int(l) for ls, l in zip(locals_, labels)
    }
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
    act = np.where(wsum > 0, agg / np.maximum(wsum, 1e-9), 0.0)

    print("[5/5] binarize + turns", flush=True)
    # primary-label track: per frame, the most active global speaker.
    # Beats per-speaker overlapping tracks on single-label references, and is
    # what transcript attribution needs; overlap info stays in `act`.
    lab_track = np.where(act.max(axis=1) >= ONSET, act.argmax(axis=1), -1)
    turns: List[Turn] = []
    for g in range(k):
        on = lab_track == g
        on = _fill_gaps(on, int(MIN_OFF_SEC * fhz))
        on = _drop_short(on, int(MIN_ON_SEC * fhz))
        for a, b in _runs(on):
            turns.append(Turn(a / fhz, b / fhz, g))
    turns.sort(key=lambda t: t.start)

    if prepad > 0:
        turns = _prepad_turns(turns, prepad)

    talk: Dict[int, float] = {}
    for t in turns:
        talk[t.speaker] = talk.get(t.speaker, 0.0) + (t.end - t.start)
    order = sorted(talk, key=lambda x: -talk[x])
    rm = {old: i for i, old in enumerate(order)}
    turns = [Turn(t.start, t.end, rm[t.speaker]) for t in turns]
    talk = {rm[s]: v for s, v in talk.items()}

    elapsed = time.time() - t0
    result = DiarizationResult(
        timeline=turns,
        n_speakers=len(talk),
        talk_sec={f"SPEAKER_{s}": round(v, 1) for s, v in sorted(talk.items())},
        elapsed_sec=round(elapsed, 1),
        method=(
            "diar-rs/v2: DiariZen local seg (16s/8s) + masked WeSpeaker emb + "
            f"AHC({cluster_space} cos+2GMM, dur-min) + overlap-add + primary-label"
            + (f" + prepad{prepad}" if prepad else "")
        ),
        n_xvectors=len(locals_),
        meta={"duration_sec": round(dur, 2), "n_windows": len(starts)},
    )
    print(
        f"done: speakers={result.n_speakers} turns={len(turns)} "
        f"talk={result.talk_sec} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def _segment_all(
    pcm: np.ndarray, sr: int, seg_path: Path, starts: List[int]
) -> Tuple[List[np.ndarray], float]:
    """Hard multilabel per window, with an on-disk cache for fast iteration."""
    import hashlib

    key = hashlib.sha1(
        f"{len(pcm)}:{float(pcm[::100000].sum()):.6f}:{WIN_SEC}:{HOP_SEC}:{seg_path}".encode()
    ).hexdigest()[:16]
    cache = Path("runs/.segcache") / f"{key}.npz"
    if cache.exists():
        z = np.load(cache)
        return [z[f"w{i}"] for i in range(len(starts))], float(z["fhz"])
    seg = SegmentationONNX(seg_path)
    out, fhz = [], None
    for s in starts:
        probs = _seg_probs(seg, pcm[s : s + int(WIN_SEC * sr)])
        if fhz is None:
            fhz = probs.shape[0] / WIN_SEC
        out.append(_binary_multilabel(seg, probs))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, fhz=fhz, **{f"w{i}": w for i, w in enumerate(out)})
    return out, float(fhz)


def _absorb_by_duration(
    x: np.ndarray, labels: np.ndarray, dur: np.ndarray, min_sec: float
) -> np.ndarray:
    """Absorb clusters with < min_sec total speech into the nearest centroid."""
    labels = labels.copy()
    x = l2_norm(x)
    for _ in range(10):
        ks = sorted(set(labels.tolist()))
        tot = {g: float(dur[labels == g].sum()) for g in ks}
        large = [g for g in ks if tot[g] >= min_sec]
        small = [g for g in ks if tot[g] < min_sec]
        if not small or not large:
            break
        cents = {g: l2_norm(x[labels == g].mean(0)) for g in large}
        for g in small:
            for j in np.where(labels == g)[0]:
                labels[j] = max(cents, key=lambda c: float(np.dot(x[j], cents[c])))
        used = sorted(set(labels.tolist()))
        rm = {u: i for i, u in enumerate(used)}
        labels = np.array([rm[int(v)] for v in labels], dtype=np.int32)
    return labels


def _prepad_turns(turns: List[Turn], pad: float) -> List[Turn]:
    """Extend each turn's start backward by ≤ pad (never into the previous
    turn). ASR-timed references mark a turn at its first word; the acoustic
    onset detected here fires later, so a small prepad recovers that lead-in
    (lower miss/DER) at a small frame_acc cost.
    """
    out: List[Turn] = []
    for t in turns:
        prev_end = out[-1].end if out else 0.0
        s = max(t.start - pad, prev_end, 0.0) if t.start > prev_end else t.start
        out.append(Turn(s, t.end, t.speaker))
    return out


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    out = []
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        out.append((int(a), int(b)))
    return out


def _fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = mask.copy()
    runs = _runs(~mask)
    for a, b in runs:
        if a == 0 or b == len(mask):
            continue
        if b - a <= max_gap:
            mask[a:b] = True
    return mask


def _drop_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    mask = mask.copy()
    for a, b in _runs(mask):
        if b - a < min_len:
            mask[a:b] = False
    return mask
