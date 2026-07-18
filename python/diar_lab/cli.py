#!/usr/bin/env python3
"""CLI for the diar_lab diarization pipeline.

  python -m diar_lab.cli meeting.wav -o out_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import diarize, save_result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--no-vbx", action="store_true")
    ap.add_argument("--ahc-bias", type=float, default=0.0)
    args = ap.parse_args(argv)

    wav = Path(args.wav)
    out = Path(args.out) if args.out else Path("runs") / (wav.stem + "_diar")
    result = diarize(wav, use_vbx=not args.no_vbx, ahc_threshold_bias=args.ahc_bias)
    save_result(result, out)
    print("\n# duration format")
    print("\n".join(result.to_duration_lines()[:30]))
    if len(result.timeline) > 30:
        print(f"... ({len(result.timeline)} turns)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
