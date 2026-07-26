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
    ap.add_argument(
        "--v2", action="store_true",
        help="local-segmentation architecture (pipeline_v2)",
    )
    ap.add_argument("--emb", default=None, help="override embedding onnx path")
    ap.add_argument(
        "--cluster-space", default="xvec", choices=["xvec", "raw"],
        help="v2: AHC space (xvec = PLDA LDA 128-d, raw = 256-d cosine)",
    )
    ap.add_argument(
        "--prepad", type=float, default=0.0,
        help="v2: extend turn starts backward (s); lowers miss/DER, costs a "
             "little frame_acc (0.3 is a good transcript-attribution setting)",
    )
    args = ap.parse_args(argv)

    wav = Path(args.wav)
    out = Path(args.out) if args.out else Path("runs") / (wav.stem + "_diar")
    if args.v2:
        from .pipeline import DEFAULT_EMB
        from .pipeline_v2 import diarize_v2

        result = diarize_v2(
            wav,
            emb_path=Path(args.emb) if args.emb else DEFAULT_EMB,
            ahc_threshold_bias=args.ahc_bias,
            cluster_space=args.cluster_space,
            prepad=args.prepad,
        )
    else:
        result = diarize(wav, use_vbx=not args.no_vbx, ahc_threshold_bias=args.ahc_bias)
    save_result(result, out)
    print("\n# duration format")
    print("\n".join(result.to_duration_lines()[:30]))
    if len(result.timeline) > 30:
        print(f"... ({len(result.timeline)} turns)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
