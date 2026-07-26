#!/usr/bin/env python3
"""Run open-source diarization baselines and save timelines for compare_benchmark.

Supports:
  - pyannote community-1  (needs HF token + model access)
  - DiariZen              (pip install diarizen / git)

Usage (baselines venv):
  source .venv-diar-baselines/bin/activate
  export HF_TOKEN=...   # for gated models
  python -m diar_lab.run_baselines \\
    --wav data/audio/meeting_06-29.wav \\
    --which pyannote,diarizen \\
    -o runs/meeting_06-29_baselines
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]


def turns_to_result(turns: List[dict], method: str, elapsed: float) -> dict:
    talk = {}
    for t in turns:
        talk[t["speaker"]] = talk.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    # renumber by talk time
    order = sorted(talk.keys(), key=lambda k: -talk[k])
    rm = {old: i for i, old in enumerate(order)}
    renormed = [
        {
            "start": round(t["start"], 3),
            "end": round(t["end"], 3),
            "speaker": rm[t["speaker"]],
        }
        for t in turns
        if t["end"] - t["start"] >= 0.05
    ]
    talk2 = {}
    for t in renormed:
        talk2[t["speaker"]] = talk2.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    return {
        "method": method,
        "elapsed_sec": round(elapsed, 1),
        "n_turns": len(renormed),
        "talk_sec": {f"SPEAKER_{k}": round(v, 1) for k, v in sorted(talk2.items())},
        "timeline": renormed,
    }


def save_result(result: dict, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    # abs lines
    def fmt(t):
        total = int(round(t))
        m, s = divmod(max(0, total), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    abs_path = out_dir / f"{name}_abs.txt"
    abs_path.write_text(
        "\n".join(
            f"Speaker{t['speaker']+1}    {fmt(t['start'])}" for t in result["timeline"]
        )
        + "\n"
    )
    print(f"wrote {path} turns={result['n_turns']} talk={result['talk_sec']}", flush=True)
    return path


def run_pyannote(wav: Path, token: Optional[str], device: str = "cpu") -> dict:
    import torch
    from pyannote.audio import Pipeline

    t0 = time.time()
    print("[pyannote] loading community-1 ...", flush=True)
    kwargs = {}
    if token:
        kwargs["token"] = token
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", **kwargs
    )
    if device == "mps" and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        print("[pyannote] device=mps", flush=True)
    elif device == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("[pyannote] device=cuda", flush=True)
    else:
        print("[pyannote] device=cpu", flush=True)

    print(f"[pyannote] running on {wav} ...", flush=True)
    # try with num_speakers=2 (meeting) and also unconstrained — use unconstrained default
    output = pipeline(str(wav))
    # community-1 returns object with speaker_diarization or Annotation
    diar = getattr(output, "speaker_diarization", None) or getattr(
        output, "exclusive_speaker_diarization", None
    )
    if diar is None:
        # older API: Annotation directly
        diar = output

    turns = []
    # Annotation.itertracks
    if hasattr(diar, "itertracks"):
        for turn, _, speaker in diar.itertracks(yield_label=True):
            # speaker like "SPEAKER_00"
            m = re.search(r"(\d+)", str(speaker))
            spk = int(m.group(1)) if m else hash(str(speaker)) % 100
            turns.append(
                {"start": float(turn.start), "end": float(turn.end), "speaker": spk}
            )
    else:
        raise RuntimeError(f"unknown pyannote output type: {type(output)} / {type(diar)}")

    elapsed = time.time() - t0
    print(f"[pyannote] done {len(turns)} raw turns in {elapsed:.1f}s", flush=True)
    return turns_to_result(
        turns, "pyannote/speaker-diarization-community-1", elapsed
    )


def run_diarizen(wav: Path, model_id: str, token: Optional[str], device: str = "cpu") -> dict:
    t0 = time.time()
    print(f"[diarizen] loading {model_id} ...", flush=True)
    try:
        from diarizen.pipelines.inference import DiariZenPipeline
    except ImportError as e:
        raise RuntimeError(
            "diarizen not installed. Try:\n"
            "  pip install git+https://github.com/BUTSpeechFIT/DiariZen.git\n"
            f"Original import error: {e}"
        ) from e

    kwargs = {}
    if token:
        kwargs["token"] = token
    try:
        pipeline = DiariZenPipeline.from_pretrained(model_id, **kwargs)
    except TypeError:
        pipeline = DiariZenPipeline.from_pretrained(model_id)

    # diarizen's pipeline hardcodes cuda/cpu (ignores MPS); relocate if requested.
    if device == "mps":
        import torch
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
            print("[diarizen] device=mps", flush=True)
        else:
            print("[diarizen] MPS unavailable, using cpu", flush=True)

    print(f"[diarizen] running on {wav} ...", flush=True)
    diar = pipeline(str(wav))
    turns = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        m = re.search(r"(\d+)", str(speaker))
        spk = int(m.group(1)) if m else hash(str(speaker)) % 100
        turns.append(
            {"start": float(turn.start), "end": float(turn.end), "speaker": spk}
        )
    elapsed = time.time() - t0
    print(f"[diarizen] done {len(turns)} raw turns in {elapsed:.1f}s", flush=True)
    return turns_to_result(turns, f"DiariZen:{model_id}", elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=str(ROOT / "data/audio/meeting_06-29.wav"))
    ap.add_argument(
        "-o",
        "--out",
        default=str(ROOT / "runs/meeting_06-29_baselines"),
    )
    ap.add_argument(
        "--which",
        default="pyannote,diarizen",
        help="comma list: pyannote,diarizen",
    )
    ap.add_argument(
        "--diarizen-model",
        default="BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        help="or BUT-FIT/diarizen-meeting-base / diarizen-wavlm-base-s80-md",
    )
    ap.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--token", default=None, help="HF token (else env HF_TOKEN)")
    args = ap.parse_args()

    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    which = [w.strip().lower() for w in args.which.split(",") if w.strip()]
    out = Path(args.out)
    wav = Path(args.wav)
    assert wav.exists(), wav

    errors = []
    if "pyannote" in which:
        try:
            r = run_pyannote(wav, token=token, device=args.device)
            save_result(r, out, "pyannote_community1")
        except Exception as e:
            errors.append(f"pyannote: {e}")
            print(f"[pyannote] FAILED: {e}", flush=True)

    if "diarizen" in which:
        try:
            r = run_diarizen(wav, model_id=args.diarizen_model, token=token, device=args.device)
            save_result(r, out, "diarizen")
        except Exception as e:
            errors.append(f"diarizen: {e}")
            print(f"[diarizen] FAILED: {e}", flush=True)

    if errors:
        print("\nSome baselines failed:", file=sys.stderr)
        for e in errors:
            print(" -", e, file=sys.stderr)
        # exit 0 if at least one succeeded
        if len(errors) == len(which):
            sys.exit(2)


if __name__ == "__main__":
    main()
