# models/ — open weights

User-supplied open weights, populated by `../scripts/fetch_models.py`. **Gitignored** — not committed. Verify each model's license before redistribution.

## Stack

| File (target name) | Model | License | Source |
|---|---|---|---|
| `silero_vad.onnx` | Silero VAD (not yet wired; seg provides speech activity) | MIT | https://github.com/snakers4/silero-vad |
| `emb.onnx` | WeSpeaker ResNet34-LM (256-d, 6.63M) | toolkit Apache-2.0; weights CC-BY-4.0 (verify) | https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM |
| `seg.onnx` | DiariZen WavLM-conformer, **ONNX export** of the PyTorch checkpoint (11-class powerset) | code MIT; **weights non-commercial** ⚠️ | https://huggingface.co/BUT-FIT/diarizen-wavlm-base-s80-md (export via `scripts/export_diarizen_seg.py`) |
| `plda/` | open PLDA npz (`plda.npz` + `xvec_transform.npz`, 256→128) | Apache-2.0 | same DiariZen repo (`plda/*`) |

## ⚠️ Non-commercial: DiariZen segmentation weights

The DiariZen weights (`BUT-FIT/diarizen-wavlm-base-s80-md`) carry an academic clause that **does not allow commercial use**. For this research/exploration repo that is acceptable. `scripts/fetch_models.py --accept-nc` gates the download, and `scripts/export_diarizen_seg.py` produces `seg.onnx` from the PyTorch checkpoint (torch↔onnx parity `1.5e-5`).

**Commercial-safe upgrade path** (pick one):

1. **Train your own segmentation head** on MIT-licensed `microsoft/wavlm-base` using the DiariZen training recipe (MIT code). Same architecture, your weights, your license.
2. **Swap segmentation** for NeMo Sortformer (Apache-2.0). Because `crates/diar-rs/src/onnx_seg.rs` reads the output class dim at load time (not hardcoded), this is near drop-in.

## On-disk layout (after fetch)

```
models/
  silero_vad.onnx          # MIT (not yet wired into the pipeline; deferred)
  emb.onnx                 # WeSpeaker ResNet34-LM, 256-d
  seg.onnx                 # DiariZen WavLM, num_classes read from output dim
  plda/                    # optional; only used with --with-vbx
```

## SHA256

Record the SHA256 of each fetched file here once verified (TODO: fill after first `fetch_models.py` run).

```
silero_vad.onnx:           TODO (not yet wired in)
emb.onnx:                  7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068  (WeSpeaker)
seg.onnx:                  15287b101ec4b8777d1cb855c5e4a42d8f85d5a9570a6b03dc4e2f332ea35e32  (DiariZen ONNX export)
plda/plda.npz:             9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255
plda/xvec_transform.npz:   325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f
```

## Resolution order

Code resolves weights via env vars first, then this directory:

`DIAR_SEG_ONNX` / `DIAR_EMB_ONNX` / `DIAR_PLDA_DIR` / `DIAR_VAD_ONNX` → `models/{seg.onnx, emb.onnx, plda/, silero_vad.onnx}`.
