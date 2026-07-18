#!/usr/bin/env python3
"""Inspect ONNX model IO: input/output names, shapes, dtypes, metadata.

Scratch utility for Stage 4/5 — read num_classes / IO contract from the actual
open weights rather than guessing. Usage:

  python scripts/inspect_onnx.py models/emb.onnx [models/seg.onnx ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnxruntime as ort


def inspect(path: Path) -> None:
    print(f"\n=== {path} ({path.stat().st_size} bytes) ===")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print("inputs:")
    for i in sess.get_inputs():
        print(f"  - {i.name}: shape={i.shape} dtype={i.type}")
    print("outputs:")
    for o in sess.get_outputs():
        print(f"  - {o.name}: shape={o.shape} dtype={o.type}")
    meta = sess.get_modelmeta()
    if meta:
        print("metadata:", meta.custom_metadata_map or "(none)")
        print("producer:", meta.producer_name, "| graph:", meta.graph_name)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect(Path(p))
