#!/usr/bin/env python3
"""Probe the DiariZen segmentation model: load weights, report num_classes and
forward output shape. De-risks the ONNX export. Run with the diarizen venv.
"""
from __future__ import annotations

import os
import sys
import toml
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
tok = os.environ.get("HF_TOKEN")
if not tok and (ROOT / ".hf_token").exists():
    os.environ["HF_TOKEN"] = (ROOT / ".hf_token").read_text().strip()

from diarizen.models.eend.model_wavlm_conformer import Model  # noqa: E402

hub = Path(snapshot_download("BUT-FIT/diarizen-wavlm-base-s80-md"))
print("hub:", hub)
cfg = toml.load(hub / "config.toml")
margs = cfg["model"]["args"]
print("model args:", margs)

model = Model(**margs)
print("dimension (num_classes):", model.dimension)
print("specifications powerset:", model.specifications.powerset,
      "classes:", model.specifications.classes,
      "num_powerset_classes:", model.specifications.num_powerset_classes)

sd = torch.load(hub / "pytorch_model.bin", map_location="cpu", weights_only=True)
print("state_dict keys:", len(sd), "sample:", list(sd.keys())[:3])
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))
if missing:
    print("  missing[:6]:", missing[:6])
if unexpected:
    print("  unexpected[:6]:", unexpected[:6])
model.eval()

chunk = model.chunk_size * model.sample_rate
print(f"chunk_size={model.chunk_size}s samples={chunk} num_frames={model.num_frames(chunk)}")
with torch.no_grad():
    x = torch.randn(1, 1, chunk)
    y = model(x)
print("forward output shape:", tuple(y.shape), "dtype:", y.dtype)
print("activation sample (row 0, first 5):", y[0, 0, :5].tolist())
