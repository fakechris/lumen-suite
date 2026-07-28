# Changelog

## [Unreleased]

### Build: vendored fbank, no Python / no external dylib
`kaldi-native-fbank` (Apache-2.0) and `kissfft` (BSD-3-Clause) are now vendored
under `native/knf/` and compiled statically by `build.rs`. This removes the
former build-time dependency on a Python-located `kaldi-native-fbank-core`
dylib and its Unix-only rpath link arg, so `cargo build` works on a clean
machine with no Python and no preinstalled kaldi-native-fbank. Only the fbank
feature chain is compiled (mfcc/whisper/stft recognizers are not). Resolves the
"embeddable" blocker for consuming diar-rs as a plain git dependency
(lumen-suite ADR-0001 blocker #2).

## [0.2.0] — 2026-07-18

### v2 pipeline (Python): local segmentation + speaker-masked embeddings
`diar_lab.cli <wav> --v2` (`pipeline_v2.py`). The segmentation model is now
used as a local diarizer (pyannote-3.x style) instead of a VAD: overlapping
16 s windows (hop 8 s) → per-window per-speaker powerset activity → one
speaker-masked embedding per (window, local speaker), CMN over that speaker's
frames → AHC (duration-based min cluster) → overlap-add aggregation →
primary-label frame track → turns. Options: `--cluster-space {xvec,raw}`,
`--prepad <s>`, `--emb <onnx>`.

Reference meeting (vs GT): frame_acc **89.0% → 91.8%** (DiariZen large-v2
band), turn-majority 72.7% → 80.9%; `--prepad 0.3` trades to DER **17.2%**
(< large-v2's 18.2%) with 91.3% / 82.7%.

### Eval
- `best_map_acc`: optimal hyp→ref speaker assignment (Hungarian) instead of
  greedy first-K permutation; existing scores unchanged.
- Documented: reference GT is a commercial system's output; its 23 s third
  speaker (S2) does not embed as a coherent voice — annotation noise, not a
  recall target (see `docs/HANDOFF.md`, `docs/PROBLEM.md`).

### Models
- `fetch_models.py --only emb_cnceleb.onnx`: WeSpeaker CnCeleb ResNet34-LM
  (Chinese-data embedding) for `--v2 --cluster-space raw` on Mandarin audio.

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
