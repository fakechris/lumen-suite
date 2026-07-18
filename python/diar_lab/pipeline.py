"""Open-source speaker diarization pipeline.

Pipeline (open weights + BUT VBx recipe):

  wav
    → segmentation.onnx (powerset) OR energy VAD fallback
    → speech segments
    → sliding sub-windows (1.5s / 0.75s) → embedding.onnx
    → xvec transform (mean1/LDA/mean2)
    → AHC (cosine + 2-GMM threshold)
    → VBx HMM refine (official BUT)
    → merge adjacent labels
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.special import softmax

from .fbank import compute_fbank_knf
from .plda import PldaTransform, l2_norm
from .vbx import VBx, twoGMMcalib_lin

ROOT = Path(__file__).resolve().parents[2]
_MODELS = ROOT / "models"


def _env_path(name: str, default: Path) -> Path:
    """Resolve a model path from an env var, else the models/ default."""
    import os
    return Path(os.environ[name]) if name in os.environ else default


DEFAULT_EMB = _env_path("DIAR_EMB_ONNX", _MODELS / "emb.onnx")
DEFAULT_SEG = _env_path("DIAR_SEG_ONNX", _MODELS / "seg.onnx")
DEFAULT_PLDA = _env_path("DIAR_PLDA_DIR", _MODELS / "plda")

# x-vector extraction geometry (classic diarization; matches production practice)
XVEC_WIN = 1.5
XVEC_HOP = 0.75
MIN_SEG = 0.5

# VBx hyperparams (typical CALLHOME / VoxConverse recipe range)
VBX_FA = 0.3
VBX_FB = 17.0
VBX_LOOP = 0.99
VBX_INIT_SMOOTH = 5.0
AHC_THRESHOLD_BIAS = 0.0  # extra bias on 2-GMM thr


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


@dataclass
class DiarizationResult:
    timeline: List[Turn]
    n_speakers: int
    talk_sec: dict
    elapsed_sec: float
    method: str
    n_xvectors: int = 0
    meta: dict = field(default_factory=dict)

    def to_duration_lines(self) -> List[str]:
        lines = []
        for t in self.timeline:
            dur = t.end - t.start
            total = int(round(dur))
            m, s = divmod(max(0, total), 60)
            h, m = divmod(m, 60)
            ds = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
            lines.append(f"Speaker{t.speaker + 1}    {ds}")
        return lines

    def to_abs_lines(self) -> List[str]:
        lines = []
        for t in self.timeline:
            total = int(round(t.start))
            m, s = divmod(max(0, total), 60)
            h, m = divmod(m, 60)
            ds = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
            lines.append(f"Speaker{t.speaker + 1}    {ds}")
        return lines


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EmbeddingONNX:
    def __init__(self, path: Path = DEFAULT_EMB):
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]

    def __call__(self, fbank: np.ndarray) -> np.ndarray:
        x = np.asarray(fbank, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        feeds = {}
        if "fbank" in self.inputs:
            feeds["fbank"] = x
        else:
            feeds[self.inputs[0]] = x
        if "weights" in self.inputs:
            feeds["weights"] = np.ones((x.shape[0], x.shape[1], 1), dtype=np.float32)
        out = self.sess.run(None, feeds)[0]
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 3:
            out = out[:, 0, :]
        return out[0] if out.shape[0] == 1 else out


class SegmentationONNX:
    """Powerset 11-class → multi-label 4-spk activity."""

    def __init__(self, path: Path = DEFAULT_SEG):
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.classes = self._powerset(4, 2)

    @staticmethod
    def _powerset(n_spk: int, max_active: int):
        classes = [()]
        for k in range(1, max_active + 1):
            classes.extend(combinations(range(n_spk), k))
        return classes

    # DiariZen seg expects exactly 16 s (256000 samples); pad/truncate to it.
    CHUNK_SAMPLES = 16 * 16000

    def multilabel(self, pcm: np.ndarray) -> Tuple[np.ndarray, float]:
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size < self.CHUNK_SAMPLES:
            x = np.pad(x, (0, self.CHUNK_SAMPLES - x.size))
        elif x.size > self.CHUNK_SAMPLES:
            x = x[: self.CHUNK_SAMPLES]
        out = self.sess.run(None, {"waveforms": x[None, None, :]})[0][0]  # [T,11]
        # softmax over powerset classes
        z = out - out.max(axis=-1, keepdims=True)
        p = np.exp(z)
        p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
        multi = np.zeros((p.shape[0], 4), dtype=np.float64)
        for c, members in enumerate(self.classes):
            for s in members:
                multi[:, s] += p[:, c]
        frame_hz = multi.shape[0] / (self.CHUNK_SAMPLES / 16000.0)
        return multi, frame_hz


# ---------------------------------------------------------------------------
# Speech regions + x-vector windows
# ---------------------------------------------------------------------------

def speech_mask_from_seg(
    pcm: np.ndarray, sr: int, seg_model: SegmentationONNX, chunk_sec: float = 16.0
) -> np.ndarray:
    """Boolean mask per sample: any speaker active."""
    n = len(pcm)
    mask = np.zeros(n, dtype=bool)
    chunk = int(chunk_sec * sr)
    hop = chunk  # non-overlap for mask (fast)
    for s in range(0, n, hop):
        e = min(n, s + chunk)
        piece = pcm[s:e]
        if len(piece) < sr * 0.5:
            continue
        pad = piece
        if len(pad) < chunk:
            pad = np.pad(pad, (0, chunk - len(pad)))
        multi, fhz = seg_model.multilabel(pad)
        act = multi.max(axis=1) >= 0.4  # frame active
        # map frames to samples
        for ti, a in enumerate(act):
            if not a:
                continue
            a0 = s + int(ti / fhz * sr)
            a1 = s + int((ti + 1) / fhz * sr)
            mask[a0 : min(e, a1)] = True
    return mask


def energy_vad_mask(pcm: np.ndarray, sr: int, frame_ms=30, hop_ms=10, thr_db=-40) -> np.ndarray:
    """Fallback energy VAD."""
    x = pcm.astype(np.float64)
    fl = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    mask = np.zeros(len(x), dtype=bool)
    for i in range(0, len(x) - fl + 1, hop):
        rms = np.sqrt(np.mean(x[i : i + fl] ** 2) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)
        if db > thr_db:
            mask[i : i + fl] = True
    return mask


def xvector_windows(
    pcm: np.ndarray,
    sr: int,
    speech_mask: np.ndarray,
    emb: EmbeddingONNX,
    win: float = XVEC_WIN,
    hop: float = XVEC_HOP,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract x-vectors on sliding windows that fall mostly on speech.

    Returns:
      E: (N, 256)
      times: (N, 2) start/end sec
    """
    win_n = int(win * sr)
    hop_n = int(hop * sr)
    embs = []
    times = []
    n = len(pcm)
    for s in range(0, max(1, n - win_n + 1), hop_n):
        e = s + win_n
        if e > n:
            break
        # require >= 50% speech
        if speech_mask[s:e].mean() < 0.4:
            continue
        chunk = pcm[s:e]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < 0.005:
            continue
        fb = compute_fbank_knf(chunk, sample_rate=sr)
        if fb.shape[0] < 10:
            continue
        v = emb(fb)
        embs.append(np.asarray(v, dtype=np.float64))
        times.append((s / sr, e / sr))
    if not embs:
        return np.zeros((0, 256)), np.zeros((0, 2))
    return np.stack(embs), np.array(times, dtype=np.float64)


# ---------------------------------------------------------------------------
# Clustering: AHC + VBx
# ---------------------------------------------------------------------------

def cos_similarity(x: np.ndarray) -> np.ndarray:
    x = l2_norm(np.asarray(x, dtype=np.float64))
    return x @ x.T


def ahc_labels(
    x: np.ndarray,
    threshold_bias: float = AHC_THRESHOLD_BIAS,
    max_speakers: int = 6,
    min_cluster_size: int = 8,
) -> np.ndarray:
    """AHC on cosine similarity.

    Primary path: 2-GMM calibrated threshold (BUT). If that over-splits
    beyond max_speakers, fall back to best k in [2, max_speakers] by
    silhouette, then absorb tiny clusters into nearest large centroid.
    """
    import fastcluster
    from sklearn.metrics import silhouette_score

    N = x.shape[0]
    if N == 0:
        return np.zeros(0, dtype=np.int32)
    if N == 1:
        return np.zeros(1, dtype=np.int32)

    x = l2_norm(np.asarray(x, dtype=np.float64))
    scr = cos_similarity(x)
    np.fill_diagonal(scr, 0.0)

    thr, _ = twoGMMcalib_lin(scr[np.triu_indices(N, k=1)])
    thr = thr + threshold_bias

    dist = squareform(np.clip(1.0 - scr, 0, 2), checks=False)
    lin = fastcluster.linkage(dist, method="average", preserve_input=False)

    # distance-threshold cut (similarity thr → distance 1-thr)
    cut = max(1e-6, 1.0 - thr)
    labels = fcluster(lin, cut, criterion="distance") - 1
    k = int(labels.max()) + 1

    # If over-split, pick k by silhouette on cosine
    if k > max_speakers or k < 2:
        best_k, best_sil = 2, -1.0
        for kk in range(2, min(max_speakers, N) + 1):
            lab_k = fcluster(lin, kk, criterion="maxclust") - 1
            try:
                sil = float(silhouette_score(x, lab_k, metric="cosine"))
            except Exception:
                sil = -1.0
            if sil > best_sil:
                best_sil, best_k = sil, kk
        labels = fcluster(lin, best_k, criterion="maxclust") - 1
        k = best_k

    labels = _absorb_small_clusters(x, labels, min_size=min_cluster_size)
    return labels.astype(np.int32)


def _absorb_small_clusters(
    x: np.ndarray, labels: np.ndarray, min_size: int = 8
) -> np.ndarray:
    """Reassign clusters smaller than min_size to nearest large centroid."""
    labels = labels.copy()
    x = l2_norm(x)
    for _ in range(10):
        sizes = np.bincount(labels)
        large = [i for i, s in enumerate(sizes) if s >= min_size]
        small = [i for i, s in enumerate(sizes) if 0 < s < min_size]
        if not small or not large:
            break
        cents = {i: l2_norm(x[labels == i].mean(0)) for i in large}
        for si in small:
            idx = np.where(labels == si)[0]
            for j in idx:
                best, best_d = large[0], -1.0
                for li, c in cents.items():
                    d = float(np.dot(x[j], c))
                    if d > best_d:
                        best_d, best = d, li
                labels[j] = best
        # renumber
        used = sorted(set(labels.tolist()))
        rm = {u: i for i, u in enumerate(used)}
        labels = np.array([rm[int(v)] for v in labels], dtype=np.int32)
    return labels


def vbx_refine(
    x_ahc: np.ndarray,
    labels: np.ndarray,
    plda: PldaTransform,
    Fa: float = VBX_FA,
    Fb: float = VBX_FB,
    loopP: float = VBX_LOOP,
    init_smoothing: float = VBX_INIT_SMOOTH,
) -> np.ndarray:
    """AHC hard labels → soft gamma → VBx → hard labels."""
    N = x_ahc.shape[0]
    if N == 0:
        return labels
    K = int(labels.max()) + 1
    if K <= 1:
        return labels

    qinit = np.zeros((N, K), dtype=np.float64)
    qinit[np.arange(N), labels] = 1.0
    qinit = softmax(qinit * init_smoothing, axis=1)

    # PLDA space features from AHC-space x-vectors (already 128-d after xvec_tf)
    fea = plda.plda_space(x_ahc)  # (N, 128)
    # temporal order assumed = input order
    gamma, pi, Li = VBx(
        fea,
        plda.psi,
        pi=qinit.shape[1],
        gamma=qinit,
        maxIters=40,
        epsilon=1e-6,
        loopProb=loopP,
        Fa=Fa,
        Fb=Fb,
    )
    hard = np.argmax(gamma, axis=1).astype(np.int32)
    # drop empty speakers
    used = sorted(set(hard.tolist()))
    remap = {u: i for i, u in enumerate(used)}
    return np.array([remap[int(h)] for h in hard], dtype=np.int32)


def merge_adjacent_labels(
    starts: np.ndarray, ends: np.ndarray, labels: np.ndarray
) -> List[Turn]:
    """Merge adjacent/overlapping same-label segments; split overlaps mid-point."""
    if len(labels) == 0:
        return []
    order = np.argsort(starts)
    starts, ends, labels = starts[order], ends[order], labels[order]

    # first pass: compact same-label adjacent
    s0, e0, l0 = float(starts[0]), float(ends[0]), int(labels[0])
    segs = []
    for s, e, l in zip(starts[1:], ends[1:], labels[1:]):
        s, e, l = float(s), float(e), int(l)
        if l == l0 and s <= e0 + 1e-3:
            e0 = max(e0, e)
        else:
            segs.append((s0, e0, l0))
            s0, e0, l0 = s, e, l
    segs.append((s0, e0, l0))

    # resolve overlapping different labels: mid split
    fixed = []
    for i, (s, e, l) in enumerate(segs):
        if fixed and s < fixed[-1][1] and l != fixed[-1][2]:
            mid = 0.5 * (s + fixed[-1][1])
            fixed[-1] = (fixed[-1][0], mid, fixed[-1][2])
            s = mid
        if fixed and l == fixed[-1][2] and s <= fixed[-1][1] + 0.25:
            fixed[-1] = (fixed[-1][0], max(fixed[-1][1], e), l)
        else:
            fixed.append((s, e, l))

    # renumber by talk time
    talk = {}
    for s, e, l in fixed:
        talk[l] = talk.get(l, 0.0) + (e - s)
    order_spk = sorted(talk.keys(), key=lambda k: -talk[k])
    rm = {old: i for i, old in enumerate(order_spk)}
    return [Turn(s, e, rm[l]) for s, e, l in fixed if e - s >= 0.15]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_wav(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    path = Path(path)
    try:
        import soundfile as sf

        pcm, sr = sf.read(str(path), always_2d=False)
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
    except Exception:
        import wave

        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            ch = w.getnchannels()
            raw = w.readframes(n)
            sw = w.getsampwidth()
        if sw != 2:
            raise RuntimeError("need 16-bit wav or soundfile")
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1)

    if sr != target_sr:
        duration = len(pcm) / sr
        new_len = int(duration * target_sr)
        t_old = np.linspace(0, 1, num=len(pcm), endpoint=False)
        t_new = np.linspace(0, 1, num=new_len, endpoint=False)
        pcm = np.interp(t_new, t_old, pcm).astype(np.float32)
        sr = target_sr
    return pcm, sr


def diarize(
    wav_path: Path,
    emb_path: Path = DEFAULT_EMB,
    seg_path: Path = DEFAULT_SEG,
    plda_dir: Path = DEFAULT_PLDA,
    use_vbx: bool = True,
    ahc_threshold_bias: float = AHC_THRESHOLD_BIAS,
) -> DiarizationResult:
    t0 = time.time()
    wav_path = Path(wav_path)
    print(f"[1/5] load {wav_path}", flush=True)
    pcm, sr = load_wav(wav_path)
    assert sr == 16000
    dur = len(pcm) / sr
    print(f"  duration={dur/60:.2f} min", flush=True)

    print("[2/5] segmentation speech mask", flush=True)
    seg = SegmentationONNX(seg_path)
    try:
        mask = speech_mask_from_seg(pcm, sr, seg)
        speech_frac = float(mask.mean())
        print(f"  speech_frac={speech_frac:.3f}", flush=True)
        if speech_frac < 0.05:
            print("  fallback energy VAD", flush=True)
            mask = energy_vad_mask(pcm, sr)
    except Exception as e:
        print(f"  seg failed ({e}), energy VAD", flush=True)
        mask = energy_vad_mask(pcm, sr)

    print("[3/5] x-vector extraction (emb.onnx + kaldi fbank)", flush=True)
    emb = EmbeddingONNX(emb_path)
    E, times = xvector_windows(pcm, sr, mask, emb)
    print(f"  N_xvec={len(E)}", flush=True)
    if len(E) < 2:
        raise RuntimeError("too few x-vectors")

    print("[4/5] PLDA xvec_tf + AHC + VBx", flush=True)
    plda = PldaTransform.load(plda_dir)
    x128 = plda.xvec_transform(E)  # (N, 128) for AHC cosine
    labels = ahc_labels(x128, threshold_bias=ahc_threshold_bias)
    k_ahc = int(labels.max()) + 1
    print(f"  AHC speakers={k_ahc} sizes={np.bincount(labels)}", flush=True)

    if use_vbx and k_ahc >= 2:
        # sort x-vectors by time for HMM sticky transitions
        order = np.argsort(times[:, 0])
        x_ord = x128[order]
        lab_ord = labels[order]
        lab_vb_ord = vbx_refine(x_ord, lab_ord, plda, Fa=0.2, Fb=12.0, loopP=0.95)
        labels_vb = np.empty_like(lab_vb_ord)
        labels_vb[order] = lab_vb_ord
        k_vb = int(labels_vb.max()) + 1
        print(f"  VBx speakers={k_vb} sizes={np.bincount(labels_vb)}", flush=True)
        if k_vb >= 2:
            labels = labels_vb
        else:
            print("  VBx collapsed → keep AHC", flush=True)
        # final tiny-cluster absorb on embedding space
        labels = _absorb_small_clusters(x128, labels, min_size=max(5, len(E) // 40))

    print("[5/5] merge adjacent", flush=True)
    turns = merge_adjacent_labels(times[:, 0], times[:, 1], labels)
    talk = {}
    for t in turns:
        talk[t.speaker] = talk.get(t.speaker, 0.0) + (t.end - t.start)

    elapsed = time.time() - t0
    result = DiarizationResult(
        timeline=turns,
        n_speakers=len(talk),
        talk_sec={f"SPEAKER_{k}": round(v, 1) for k, v in sorted(talk.items())},
        elapsed_sec=round(elapsed, 1),
        method=(
            "diar-rs/open: kaldi-fbank + WeSpeaker emb.onnx + DiariZen seg.onnx + "
            "xvec_tf(plda_npz) + AHC(cos+2GMM) + BUT-VBx + merge"
        ),
        n_xvectors=len(E),
        meta={"duration_sec": round(dur, 2), "ahc_speakers": k_ahc},
    )
    print(
        f"done: speakers={result.n_speakers} turns={len(turns)} "
        f"talk={result.talk_sec} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def save_result(result: DiarizationResult, out_dir: Path, stem: str = "diarization"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": result.method,
        "n_speakers": result.n_speakers,
        "n_turns": len(result.timeline),
        "n_xvectors": result.n_xvectors,
        "talk_sec": result.talk_sec,
        "elapsed_sec": result.elapsed_sec,
        "meta": result.meta,
        "timeline": [
            {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": t.speaker}
            for t in result.timeline
        ],
    }
    (out_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )
    (out_dir / f"{stem}_duration.txt").write_text(
        "\n".join(result.to_duration_lines()) + "\n"
    )
    (out_dir / f"{stem}_abs.txt").write_text(
        "\n".join(result.to_abs_lines()) + "\n"
    )
    rttm = [
        f"SPEAKER meeting 1 {t.start:.3f} {t.end - t.start:.3f} "
        f"<NA> <NA> SPEAKER_{t.speaker} <NA> <NA>"
        for t in result.timeline
    ]
    (out_dir / f"{stem}.rttm").write_text("\n".join(rttm) + "\n")

    def fmt(t):
        m, s = divmod(int(t), 60)
        return f"{m}:{s:02d}"

    lines = [
        f"# {result.method}",
        "",
        f"- speakers: **{result.n_speakers}**",
        f"- turns: {len(result.timeline)}",
        f"- x-vectors: {result.n_xvectors}",
        f"- talk: {result.talk_sec}",
        f"- elapsed: {result.elapsed_sec}s",
        "",
        "## Timeline",
        "",
    ]
    for t in result.timeline:
        lines.append(
            f"- `{fmt(t.start)}`–`{fmt(t.end)}`  **SPEAKER_{t.speaker}**  "
            f"({t.end - t.start:.1f}s)"
        )
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")
