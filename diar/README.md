# diar-rs

> Part of [lumen-suite](../README.md) since 2026-07-26 (merged from the standalone `diar-rs` repo with full history). The crate is a workspace member but **not** in the default build set — `build.rs` needs a Python env providing `kaldi-native-fbank`; build with `PYTHON=<venv>/bin/python3 cargo test -p diar-rs`.

Open-source **speaker diarization** toolkit: **Rust library** for shipping + **Python lab** for experimentation. Runs on fully open weights and is measured against **human-annotated meeting ground truth (GT)** — not against any closed binary.

| Layer | Path | Role |
|---|---|---|
| Rust | `crates/diar-rs/` | Production-oriented library + CLI |
| Python | `python/diar_lab/` | Research, ablations, metric scripts |
| Data | `data/gt/`, `data/audio/` | Annotated meetings + waveforms |
| Models | `models/` | Local ONNX / PLDA assets (user-supplied, gitignored) |

## Problem definition

**Primary goal:** approach **human-annotated diarization / transcript ground truth (GT)** as closely as practical on real multi-speaker meetings.

**Not a goal:** matching any specific closed-source system, or treating any one binary's output as the success criterion.

Success is measured against our labeled meetings:

| Metric | Definition |
|---|---|
| `frame_acc` | 100 ms frames, speech-union, best speaker permutation |
| DER-ish | miss + FA + confusion (relative comparison, not full NIST DER) |
| Turn quality | majority-vote on ≥1 s reference turns, change-point recall/precision |

Reference meeting (checked in as data, not weights):

- Audio: `data/audio/meeting_06-29.wav`
- GT transcript (human turns): `data/gt/meeting_06-29_transcript.md`

See **[docs/PROBLEM.md](docs/PROBLEM.md)** (metric contract) and **[docs/HANDOFF.md](docs/HANDOFF.md)** (baseline numbers + next steps).

## Stack (open components)

| Stage | Component | License |
|---|---|---|
| Embedding | WeSpeaker ResNet34-LM (256-d) | Apache-2.0 / CC-BY-4.0 |
| Segmentation | DiariZen `diarizen-wavlm-base` | MIT code; weights NC (see `models/README.md`) |
| Clustering | AHC + BUT VBx | Apache-2.0 |
| PLDA (optional) | open npz transforms | Apache-2.0 |
| Fbank | kaldi-native-fbank | Apache-2.0 |
| Inference | ONNX Runtime | MIT |

Fetch weights: `python scripts/fetch_models.py` (see `models/README.md`).

## Quick start (Python lab)

```bash
cd /Users/chris/source/diar-rs
python -m venv .venv && source .venv/bin/activate
pip install -r python/requirements.txt
export PYTHONPATH=python
python scripts/fetch_models.py        # → models/ (open weights)

# Diarize, then score vs human GT
python -m diar_lab.cli data/audio/meeting_06-29.wav -o runs/demo
python -m diar_lab.compare_benchmark \
  --gt data/gt/meeting_06-29_transcript.md \
  --hyp runs/demo/diarization.json \
  -o runs/demo_gt
```

## Quick start (Rust)

```bash
cd crates/diar-rs
# Requires a Python env with kaldi_native_fbank for build.rs
export PYTHON=/path/to/venv/bin/python   # has kaldi_native_fbank
cargo test
cargo build --release
cargo run --release -- diarize --wav ../../data/audio/meeting_06-29.wav --out ../../runs/demo
```

## Layout

```
diar-rs/
  crates/diar-rs/   # Rust crate (lib + CLI + optional C FFI)
  python/diar_lab/  # experiments + GT eval
  docs/             # PROBLEM, HANDOFF, design notes
  data/audio|gt/    # fixtures (the eval meeting + its human GT)
  models/           # gitignored open weights (populated by fetch_models.py)
  runs/             # local experiment outputs (gitignored)
```

## License

Code: MIT (see `LICENSE`). Model weights under `models/` are **user-supplied**; each carries its own license (see `models/README.md`) — verify before redistribution. The DiariZen segmentation weights are **non-commercial**.

## Status snapshot

| Track | Status |
|---|---|
| Python pipeline (open weights) | lab quality; scored vs human GT |
| Rust `sw_mapped` path | parity-oriented vs Python lab |
| Primary KPI | **↑ frame_acc / ↓ DER vs human GT** |
