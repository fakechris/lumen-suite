# Problem definition — diar-rs

## One line

Build an open-source speaker-diarization stack (Rust library + Python lab, open weights) whose output **approaches human-annotated meeting ground truth (GT)** as closely as practical.

## Goal

Approach **human-annotated GT** on real multi-speaker meetings.

Concretely, on our reference meeting (`data/audio/meeting_06-29.wav` + `data/gt/meeting_06-29_transcript.md`), the open stack should reach `frame_acc` in the band of the strongest references:

| System | frame_acc vs GT | DER vs GT |
|---|---:|---:|
| Native cbs (closed) | 92.1% | 11.8% |
| DiariZen large-v2 | 92.1% | 18.2% |
| pyannote community-1 | 80.7% | 28.4% |
| **diar-rs (open) target** | **≥ ~92%** | — |

These reference numbers come from the prior reverse-engineering workspace (recorded before this repo was split off) and are reproduced here only as **targets**, not as a success criterion we must match bit-for-bit. See `docs/HANDOFF.md`.

## Non-goals

- Reverse-engineering closed apps.
- "Native binary Detect" parity as a success metric.
- Carrying or redistributing proprietary weights.

## Metric contract (`python/diar_lab/compare_benchmark.py`)

GT is a human transcript with **turn start times only** (end = next turn's start; adjacent same-speaker turns merged). The harness rasterizes timelines at 100 ms and computes:

| Metric | Definition |
|---|---|
| `frame_acc` | agreement on speech-**union** frames under the best speaker permutation |
| `miss` / `fa` / `conf` | ref-speech-hyp-silent / hyp-speech-ref-silent / both-speech-wrong-speaker |
| `der` | (miss + fa + conf) / speech-union (relative, not full NIST DER) |
| `turn_majority` | for each ref turn ≥1 s, hyp majority label in window matches mapped id |
| `change_*` | speaker-change recall/precision at ±0.5 s |

`--native` is **optional**: when omitted, the report is a clean hypothesis-vs-GT 2-way compare. Native (or any external baseline) can be supplied purely as one more column.

## Known limitations

- **Single annotated meeting** (06-29, ~15 min, 3 speakers). A result on one file is not conclusive; `docs/HANDOFF.md` tracks adding public RTTM benchmarks (Aishell-4 / AMI / VoxConverse) for breadth.
- The short third speaker (~23 s of brief replies) is consistently missed by every system; it is the known hard case to attack.

## Open weights

See `models/README.md`. Segmentation (DiariZen) weights are **non-commercial**; the rest are MIT/Apache/CC-BY-4.0. A commercial-safe upgrade path (train an MIT segmentation head, or swap to NeMo Sortformer) is documented there.
