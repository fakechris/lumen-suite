# Contributing to diar-rs

Thanks for your interest. diar-rs is an open-source speaker-diarization toolkit
(Rust library + Python lab) evaluated against human-annotated ground truth.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r python/requirements.txt
export PYTHONPATH=python

# open weights → models/  (DiariZen seg is non-commercial; needs --accept-nc)
python scripts/fetch_models.py --accept-nc
python scripts/export_diarizen_seg.py        # PyTorch → seg.onnx
python scripts/convert_plda_npz_to_bin.py    # npz → models/plda/*.bin (for Rust)
```

## Layout

- `crates/diar-rs/` — Rust library + CLI (`cargo test`, `cargo run --release -- diarize`).
- `python/diar_lab/` — Python lab + GT eval (`python -m diar_lab.cli`, `compare_benchmark`).
- `scripts/` — weight fetch, DiariZen→ONNX export, PLDA conversion, ONNX inspect.
- `data/{audio,gt}/` — the reference meeting + its human transcript GT.
- `docs/` — `PROBLEM.md` (metric contract), `HANDOFF.md` (results + next steps).

## The one rule that matters

**Success is measured against human-annotated GT**, not against any closed binary.
A change that improves `frame_acc / DER vs GT` is good; a change that chases
"native parity" is out of scope. See `docs/PROBLEM.md`.

## Conventions

- Small, compiling, tested increments. `cargo test` + `python -m py_compile
  python/diar_lab/*.py` must pass.
- Keep all weights open-source (MIT / Apache-2.0 / CC-BY-4.0); flag any NC
  license in `models/README.md` and gate downloads behind `--accept-nc`.
- Don't commit weights, runs, or secrets (`.gitignore` covers `models/**`,
  `runs/`, `.hf_token`).

## CI

`.github/workflows/ci.yml` runs the Python eval-harness smoke test and
`cargo test` (needs `kaldi-native-fbank` for the build-time FFI).
