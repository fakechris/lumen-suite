#!/usr/bin/env python3
"""Export the DiariZen WavLM segmentation model to ONNX (models/seg.onnx).

Input contract:  waveforms [B, 1, 256000]  (16 s mono @ 16 kHz, dynamic batch)
Output contract: logits    [B, 799, 11]    (11-class powerset, ~50 Hz frames)

Run with the diarizen venv (has torch + the diarizen package):
  export HF_TOKEN=$(cat .hf_token)
  <diarizen-venv>/python scripts/export_diarizen_seg.py            # base
  <diarizen-venv>/python scripts/export_diarizen_seg.py \
      --repo BUT-FIT/diarizen-wavlm-large-s80-md --out models/seg_large.onnx

wavlm-large note: its waveform pre-norm (`F.layer_norm(x, x.shape[-1:])`)
breaks the TorchScript ONNX exporter (dynamic normalized_shape). Apply
scripts/diarizen_wavlm_large_export.patch to the DiariZen clone first —
it swaps in the mathematically identical mean/var normalization.
Env recipe that works: torch==2.5.1, numpy<2, DiariZen's vendored
pyannote-audio fork installed editable with its pkg_resources
`pyannote/__init__.py` removed (PEP-420 namespace merging).
"""
from __future__ import annotations

import os
import sys
import toml
import numpy as np
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
tok = os.environ.get("HF_TOKEN")
if not tok and (ROOT / ".hf_token").exists():
    os.environ["HF_TOKEN"] = (ROOT / ".hf_token").read_text().strip()

from diarizen.models.eend.model_wavlm_conformer import Model  # noqa: E402

CHUNK = 16 * 16000  # 256000 samples

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="BUT-FIT/diarizen-wavlm-base-s80-md",
                    help="HF checkpoint (e.g. BUT-FIT/diarizen-wavlm-large-s80-md)")
    ap.add_argument("--out", default=str(ROOT / "models" / "seg.onnx"))
    args = ap.parse_args()
    OUT = Path(args.out)

    hub = Path(snapshot_download(args.repo))
    margs = toml.load(hub / "config.toml")["model"]["args"]
    model = Model(**margs).eval()
    sd = torch.load(hub / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded weights: missing={len(missing)} unexpected={len(unexpected)}")
    assert model.dimension == 11, f"expected 11 classes, got {model.dimension}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 1, CHUNK)
    # warm forward (build caches)
    with torch.no_grad():
        _ = model(dummy)

    print(f"exporting → {OUT} ...")
    torch.onnx.export(
        model,
        (dummy,),
        OUT.as_posix(),
        input_names=["waveforms"],
        output_names=["logits"],
        dynamic_axes={"waveforms": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")

    # parity check: torch vs onnxruntime on a fresh input
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    x = np.random.randn(2, 1, CHUNK).astype(np.float32)
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x)).cpu().numpy()
    y_onnx = sess.run(None, {"waveforms": x})[0]
    print("torch out:", y_torch.shape, "onnx out:", y_onnx.shape)
    max_abs = float(np.max(np.abs(y_torch - y_onnx)))
    print(f"parity max|torch - onnx| = {max_abs:.6e}")
    print("PASS" if max_abs < 1e-3 and y_onnx.shape[0] == 2 and y_onnx.shape[2] == 11
          else "CHECK PARITY")


if __name__ == "__main__":
    main()
