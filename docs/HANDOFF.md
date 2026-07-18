# Handoff — diar-rs

## What this is

`diar-rs` is an open-source speaker-diarization toolkit: a **Rust library** + a
**Python lab**, running on fully open ONNX weights and measured against
human-annotated meeting ground truth (GT). Success = `frame_acc` / DER vs the
human annotation (see `docs/PROBLEM.md`).

## Open stack

| Component | Model / library | License |
|---|---|---|
| Embedding | WeSpeaker ResNet34-LM (256-d), `models/emb.onnx` | **CC-BY-4.0** (commercial OK, attribution) |
| Segmentation | DiariZen WavLM-conformer, `models/seg.onnx` (ONNX export of the upstream checkpoint) | **CC-BY-NC-4.0** (non-commercial — see `models/README.md`) |
| Clustering | AHC (cosine + 2-GMM) + BUT VBx | Apache-2.0 |
| PLDA | open DiariZen npz (Python) / converted `.bin` (Rust) | Apache-2.0 |
| Fbank | kaldi-native-fbank (C++ FFI) | Apache-2.0 |

## Results on the reference meeting (vs human GT)

| System | frame_acc | DER | speakers |
|---|---:|---:|---:|
| DiariZen large-v2 (open ref) | 92.1% | 18.2% | 3 |
| **diar-rs Rust (own ONNX)** | **90.1%** | 14.8% | 2 |
| diar-rs Python (own ONNX) | 89.0% | 18.1% | 2 |
| pyannote community-1 (open ref) | 80.7% | 28.4% | 4 |

Rust is within ~1 pt of the Python lab; both sit at the open-reference band
(~90–92%). The brief third speaker (~23 s of short replies) is missed by every
system and is the main remaining gap.

## Status by stage

1. ✅ Rust crate — clean layout, C++ knf FFI, compiles, `cargo test` passes.
2. ✅ Docs + repo hygiene (`PROBLEM.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `models/README.md`).
3. ✅ Eval reframed GT-first (`compare_benchmark`: `--hyp` required, `--native` optional, `--gt-format`); `ModelPaths::resolve()` (env → `models/`); `fetch_models.py`.
4. ✅ Python open pipeline → 89.0% vs GT.
5. ✅ Rust open pipeline → 90.1% vs GT (within 1.1 pt of Python).
6. 🟡 Eval harness is RTTM + multi-file capable; a public-corpus run is the next step.
7. ✅ Packaging: CI, `CHANGELOG`, `CONTRIBUTING`, v0.1.0.

## Reproduce

```bash
python scripts/fetch_models.py --accept-nc      # → models/ (DiariZen seg is NC)
python scripts/export_diarizen_seg.py           # PyTorch checkpoint → models/seg.onnx
python scripts/convert_plda_npz_to_bin.py       # npz → models/plda/*.bin (for the Rust crate)

# Python lab:
python -m diar_lab.cli data/audio/meeting_06-29.wav -o runs/open
python -m diar_lab.compare_benchmark --hyp runs/open/diarization.json \
  --gt data/gt/meeting_06-29_transcript.md -o runs/open/_gt

# Rust library + CLI:
cd crates/diar-rs && cargo run --release -- diarize \
  --wav ../../data/audio/meeting_06-29.wav --out ../../runs/rust
```

## Public benchmark (multi-file DER)

The harness is corpus-ready — drop any RTTM corpus under `data/public/` and:

```bash
python -m diar_lab.compare_benchmark --gt-dir data/public/<corpus>_rttm \
  --hyp-dir runs/<corpus> --gt-format rttm -o runs/<corpus>_eval
```

Recommended open sets: **VoxConverse 2020**, **AMI**, **Aishell-4**. A single
meeting (06-29) is not conclusive; ≥5 files aggregate DER is the credibility bar.

## TODOs

- Rust VBx refine (currently AHC-only; VBx collapses on this open embedding space in the Python lab too).
- npz-native PLDA loader in Rust (currently npz → `.bin` conversion).
- Multi-file DER on a public corpus.
- **Commercial-safe segmentation**: replace the NC DiariZen weights — train an
  MIT-licensed head on `microsoft/wavlm-base` with the DiariZen recipe, or swap
  to NeMo Sortformer (Apache-2.0). `onnx_seg.rs` reads the class dim at load, so
  either is near drop-in.

## Build note (knf)

The Rust crate links C++ `kaldi-native-fbank` at build time. `build.rs` resolves
it via `KNF_LIB_DIR` / `KNF_INCLUDE_DIR` env vars, else via `$PYTHON` (a venv
with `kaldi_native_fbank` installed). On macOS, set `DYLD_LIBRARY_PATH` to the
knf `lib/` dir at runtime.
