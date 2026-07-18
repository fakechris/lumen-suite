#!/usr/bin/env python3
"""Prototype: ASR token-level fusion on top of a diarization timeline.

Pipeline:
  1. SenseVoice (sherpa-onnx, CTC token timestamps) over ~28 s chunks cut at
     diarization silence.
  2. Each token -> speaker = diarization primary label at the token midpoint
     (nearest active frame within ±1 s when the timeline is silent there).
  3. Rebuild turns from consecutive same-speaker token groups (word-boundary
     snapped turn starts/ends, transcript attached).
  4. Score the rebuilt timeline against GT with the standard harness.

Usage:
  .venv/bin/python scripts/exp_asr_fusion.py \
      --diar runs/v2_seglarge/diarization.json -o runs/asr_fusion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diar_lab.pipeline import load_wav  # noqa: E402

SV_DIR = Path("/tmp/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
CHUNK_SEC = 28.0
STEP = 0.01  # raster step for the label track


def build_label_track(timeline, dur):
    n = int(dur / STEP) + 2
    lab = np.full(n, -1, dtype=np.int32)
    for t in timeline:
        a, b = int(t["start"] / STEP), int(np.ceil(t["end"] / STEP))
        lab[max(0, a): min(n, b)] = int(t["speaker"])
    return lab


def chunk_bounds(lab, dur):
    """Cut points every <=CHUNK_SEC, preferring diarization silence."""
    bounds = [0.0]
    while bounds[-1] + CHUNK_SEC < dur:
        t0 = bounds[-1]
        hard = t0 + CHUNK_SEC
        # search backward from the hard limit for a silent frame
        lo, hi = int((hard - 6.0) / STEP), int(hard / STEP)
        seg = lab[lo:hi]
        sil = np.where(seg < 0)[0]
        cut = (lo + sil[-1]) * STEP if len(sil) else hard
        bounds.append(max(cut, t0 + 5.0))
    bounds.append(dur)
    return bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=str(ROOT / "data/audio/meeting_06-29.wav"))
    ap.add_argument("--diar", required=True, help="diarization.json to fuse with")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    import sherpa_onnx

    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(SV_DIR / "model.int8.onnx"),
        tokens=str(SV_DIR / "tokens.txt"),
        language="zh",
        use_itn=True,
    )

    pcm, sr = load_wav(Path(args.wav))
    dur = len(pcm) / sr
    diar = json.loads(Path(args.diar).read_text())
    lab = build_label_track(diar["timeline"], dur)

    # 1. ASR chunks with token timestamps
    tokens = []  # (t_global, token_text)
    bounds = chunk_bounds(lab, dur)
    for a, b in zip(bounds[:-1], bounds[1:]):
        s = rec.create_stream()
        s.accept_waveform(sr, pcm[int(a * sr): int(b * sr)])
        rec.decode_stream(s)
        r = s.result
        ts = list(getattr(r, "timestamps", []) or [])
        toks = list(getattr(r, "tokens", []) or [])
        for tt, tok in zip(ts, toks):
            tokens.append((a + float(tt), tok))
    print(f"ASR: {len(tokens)} tokens over {len(bounds)-1} chunks")

    # 2. token -> speaker at token midpoint, then median-smooth the label
    #    sequence (kills single-token flips; keeps sustained backchannels)
    n = len(lab)
    tok_spk = []
    for t, tok in tokens:
        i = min(int(t / STEP), n - 1)
        spk = lab[i]
        if spk < 0:  # diarization silence: nearest active frame within ±1 s
            w = 100
            lo, hi = max(0, i - w), min(n, i + w)
            near = np.where(lab[lo:hi] >= 0)[0]
            if len(near):
                spk = lab[lo + near[np.abs(near - (i - lo)).argmin()]]
        tok_spk.append(int(spk))
    sm = list(tok_spk)
    for i in range(1, len(sm) - 1):
        if tok_spk[i - 1] == tok_spk[i + 1] != tok_spk[i]:
            sm[i] = tok_spk[i - 1]

    # 3. rebuild turns from consecutive same-speaker tokens
    turns, texts = [], []
    for (t, tok), spk in zip(tokens, sm):
        if spk < 0:
            continue
        if turns and spk == turns[-1]["speaker"] and t - turns[-1]["end"] <= 2.0:
            turns[-1]["end"] = round(t + 0.25, 3)
            texts[-1] += tok
        else:
            turns.append({"start": round(t, 3), "end": round(t + 0.25, 3), "speaker": spk})
            texts.append(tok)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diarization.json").write_text(json.dumps({
        "method": f"asr-fusion(SenseVoice tokens) over {diar.get('method','')}",
        "n_speakers": len({t['speaker'] for t in turns}),
        "timeline": turns,
    }, ensure_ascii=False, indent=1))
    lines = [
        f"[{tr['start']:7.2f}] SPEAKER_{tr['speaker']}  {tx}"
        for tr, tx in zip(turns, texts)
    ]
    (out / "transcript.txt").write_text("\n".join(lines) + "\n")
    print(f"turns={len(turns)}  wrote {out}/diarization.json, transcript.txt")


if __name__ == "__main__":
    main()
