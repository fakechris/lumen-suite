# Changelog

## [0.1.0] — 2026-07-18

First release of **diar-rs**: an open-source speaker-diarization toolkit (Rust
library + Python lab) running on fully open weights and measured against
human-annotated ground truth (not against any closed binary).

### Open stack
- **Embedding**: WeSpeaker ResNet34-LM (256-d) — `emb.onnx`.
- **Segmentation**: DiariZen WavLM-conformer, **exported to ONNX** from the
  PyTorch checkpoint (`scripts/export_diarizen_seg.py`; torch↔onnx parity 1.5e-5),
  11-class powerset — `seg.onnx`.
- **Clustering**: AHC (cosine + 2-GMM calibration) + BUT VBx (Python).
- **PLDA**: open DiariZen npz (Python loads npz; Rust reads converted `.bin`).
- **Fbank**: kaldi-native-fbank (C++ FFI).

### Pipeline
- Python `diar_lab` (`pipeline.py`): seg-as-VAD → sliding WeSpeaker x-vectors →
  PLDA LDA → AHC → VBx → merge.
- Rust `crates/diar-rs`: same flow (AHC; VBx refine TODO).
- Eval `compare_benchmark.py`: GT-first, `--native` optional, `--gt-format
  {md-zh,md-en,rttm}`, multi-file aggregation (`--gt-dir/--hyp-dir`).

### Result on the reference meeting (vs human GT)
| System | frame_acc | DER |
|---|---:|---:|
| diar-rs Rust (own ONNX) | 90.1% | 14.8% |
| diar-rs Python (own ONNX) | 89.0% | 18.1% |

Rust is within 1.1 pt of the Python lab (Stage 5 ±2 pt target met).

### Known limitations / TODO
- Single annotated meeting evaluated; a public RTTM corpus run is the next
  credibility step (harness is ready — see `docs/HANDOFF.md`).
- VBx collapses on this open embedding space (AHC kept); Rust VBx refine +
  npz-native PLDA loader pending.
- DiariZen segmentation weights are **non-commercial** (see `models/README.md`).
- The Rust crate links C++ `kaldi-native-fbank` at build time (see README).
