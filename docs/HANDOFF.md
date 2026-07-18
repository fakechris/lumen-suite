# Handoff — diar-rs

## Where this repo came from

`diar-rs` was split off from a prior Clipto.app reverse-engineering workspace. That workspace's goal was "match the native `cbs` binary (≥97% frame_acc vs native)". **That goal is voided here.** This repo's goal is "approach human-annotated GT" (see `docs/PROBLEM.md`).

Carried over (open, reusable):

- Rust diarization crate (`crates/diar-rs/`) — math kernels (audio, fbank, powerset, PLDA, AHC/VBx, merge, io) + pipeline + optional C FFI.
- Python lab (`python/diar_lab/`) — `pipeline.py`, `vbx.py` (official BUT VBx), `fbank.py` (knf), `plda.py`, `cli.py`, `compare_benchmark.py` (GT eval), `run_baselines.py`.

Left behind (NOT in this repo): all reverse-engineering artifacts — IDA databases, Frida hooks, decompile-locked specs, native-gap sweeps, the product-weight-tuned `native_pipeline.py` / `pipeline_soft.py` / `fbank_fast.py`, and the proprietary `spk_1.0.1` weights.

## Starting numbers (recorded in the prior workspace, on meeting_06-29 vs human GT)

| System | frame_acc vs GT | DER vs GT | turns | speakers |
|---|---:|---:|---:|---:|
| Native cbs (closed) | 92.1% | 11.8% | 81 | 3 |
| DiariZen large-v2 | 92.1% | 18.2% | 226 | 3 |
| Reimpl v5 (sw_mapped, product weights) | 89.9% | 15.6% | 91 | 2 |
| pyannote community-1 (default) | 80.7% | 28.4% | 318 | 4 |

> The 89.9% "Reimpl v5" number used **proprietary product weights**. After the open-weight pivot it will change and must be **re-measured**. Treat the row above as a historical anchor, not a current result.

Every system loses the brief third speaker (~23 s); only community-1 with aggressive speaker counts gets non-trivial recall there.

### Open baseline — re-confirmed in-repo (GT-first eval)

Re-scoring the prior DiariZen-large-v2 timeline with the new `compare_benchmark` against `data/gt/meeting_06-29_transcript.md` (`runs/diarizen_prior_gt/`):

- **frame_acc 92.1%**, DER 18.2%, turn-majority (≥1 s) 81.8%, change-recall@0.5 s 56%.
- Per-GT-speaker recall: interviewer 76.5%, candidate 90.4%, brief-replies **0%** (the known hard speaker).
- → the **open stack already reaches the GT target band** (native was also 92.1%); open weights are not the bottleneck. The remaining gap is the short third speaker + DER overhead, not model openness.

### Our own ONNX pipeline — first open result (Stage 4)

`python -m diar_lab.cli data/audio/meeting_06-29.wav -o runs/open_vbx_06-29` using **only open weights** under `models/`:

- seg `seg.onnx` = DiariZen WavLM-conformer, **exported from the PyTorch checkpoint** (`scripts/export_diarizen_seg.py`; torch↔onnx parity `1.5e-5`); contract `waveforms[B,1,256000]→logits[B,799,11]` (11-class powerset, same as the prior product seg).
- emb `emb.onnx` = WeSpeaker ResNet34-LM (`feats[B,T,80]→embs[B,256]`).
- PLDA `plda/{plda.npz, xvec_transform.npz}` = open DiariZen (loads via `plda.py`).
- Pipeline: seg-as-VAD (16 s non-overlapping windows) → 1.5 s/0.75 s WeSpeaker x-vectors → PLDA LDA(256→128) → AHC(cos+2GMM) → VBx → merge.

Result vs human GT (`runs/open_vbx_06-29/_gt/`):

- **frame_acc 89.0%**, DER 18.1%, turn-majority 72.7%.
- AHC → 2 speakers (sizes 432/623); **VBx collapsed to 1 → AHC kept**. Per-GT-speaker recall: interviewer 75.6%, candidate 91.4%, brief-replies 0%.

Gap to the 92.1% references is explainable: we use the seg **only as VAD** (no dense overlap / median filter), we lose the brief 3rd speaker, and VBx isn't tuned for this open emb space. All tunable without new weights.

### Rust library — open pipeline (Stage 5)

`crates/diar-rs` now runs the same open stack end-to-end and is verified against the Python lab:

- ONNX adapters repointed to the open contracts: seg `waveforms[B,1,256000]→logits[B,799,11]` (16 s fixed chunk); emb `feats[B,T,80]→embs[B,256]` (no `weights` input).
- PLDA: the open DiariZen npz is converted to the 6 float64-LE `.bin` files the crate reads (`scripts/convert_plda_npz_to_bin.py`); npz-native loading is a TODO.
- `pipeline.rs` rewritten to the Python-lab flow: seg-as-VAD → sliding WeSpeaker x-vectors → PLDA LDA → AHC(cos+2GMM) → merge (VBx refine TODO).
- `cargo test` passes (6 unit tests incl. open PLDA load + model validation).

Result vs human GT (`runs/rust_open_06-29/_gt/`):

- **frame_acc 90.1%**, DER 14.8%, 2 speakers, 107 turns.
- vs Python lab 89.0% → **within 1.1 pt** (Stage 5 ±2 pt target ✅).

| System | frame_acc vs GT | DER |
|---|---:|---:|
| Native cbs (closed) | 92.1% | 11.8% |
| DiariZen large-v2 (full pyannote) | 92.1% | 18.2% |
| **diar-rs Rust (own ONNX)** | **90.1%** | 14.8% |
| diar-rs Python lab (own ONNX) | 89.0% | 18.1% |
| pyannote community-1 | 80.7% | 28.4% |

## Next steps (stages)

1. ✅ Stage 1 — rename to `diar-rs`, restore C++ knf FFI, scrub product literals (crate compiles).
2. ✅ Stage 2 — docs + repo hygiene (this file, `PROBLEM.md`, `requirements.txt`, `.gitignore`, `LICENSE`, `models/README.md`).
3. ✅ Stage 3 — eval reframed GT-first (`compare_benchmark`: `--hyp` required, `--native` optional, `--gt-format`, in-repo GT default); `ModelPaths::resolve()` (env → `models/`); `fetch_models.py` (HF-token aware, `--accept-nc`).
4. ✅ Stage 4 — open-weight pivot **done**. WeSpeaker `emb.onnx` (introspected: `feats→embs[256]`). DiariZen WavLM seg **exported to ONNX** from the PyTorch checkpoint (parity `1.5e-5`), 11-class powerset (matches the prior seg). Open PLDA npz loaded. **Own ONNX pipeline runs end-to-end → 89.0% frame_acc vs GT** (above); VBx collapses→AHC kept (tuning TODO). `models/` now holds all four open assets.
5. ✅ Stage 5 — Rust open-weight pivot **done**. Adapters repointed (seg 16 s/logits/11, emb feats/no-weights), PLDA via npz→bin conversion, pipeline rewritten to the Python-lab flow. **Rust scores 90.1% vs GT (within 1.1 pt of Python's 89.0%)**, `cargo test` passes. Remaining: VBx refine in Rust, npz-native PLDA loader.
6. 🟡 Stage 6 — eval harness is now **RTTM + multi-file capable** (`compare_benchmark --gt-format rttm` + `--gt-dir/--hyp-dir` aggregate DER; synthetic-tested). A real multi-corpus run is the remaining headline — drop any RTTM corpus under `data/public/` and run (see "Public benchmark" below). Rust VBx refine + npz-native PLDA loader are TODOs (low priority: VBx collapses on this open emb space; `.bin` conversion works).
7. ⬜ Stage 7 — packaging / CI / v0.1.0.

## Public benchmark (how to get a multi-file DER)

The harness is corpus-ready. Recommended sets (drop audio + RTTM under `data/public/<corpus>/`):

- **VoxConverse 2020** (short clips, fast): audio `https://www.robots.ox.ac.uk/~vgg/data/voxconverse/`, RTTM `https://github.com/joonson/voxconverse` ([HF mirror](https://huggingface.co/datasets/diarizers-community/voxconverse)).
- **AMI** (meetings) / **Aishell-4** (Chinese meetings) via the pyannote/ModelScope recipes.

Then per file: `python -m diar_lab.cli <wav> -o runs/<corpus>/<stem>` (or the Rust `diar-rs diarize`), and aggregate:

```bash
python -m diar_lab.compare_benchmark --gt-dir data/public/<corpus>_rttm \
  --hyp-dir runs/<corpus> --gt-format rttm -o runs/<corpus>_eval
```

A single meeting (06-29) is not conclusive; ≥5 files aggregate DER is the credibility bar (still TODO on this machine).

## Build note (knf)

The Rust crate links C++ `kaldi-native-fbank` at build time. `build.rs` resolves it via `KNF_LIB_DIR`/`KNF_INCLUDE_DIR` env vars, else via `$PYTHON` (a venv with `kaldi_native_fbank` installed). On macOS, set `DYLD_LIBRARY_PATH` to the knf `lib/` dir at runtime.
