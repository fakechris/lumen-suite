# Problem definition — diar-rs

## One line

Build an open-source speaker-diarization stack (Rust library + Python lab, open
weights) whose output **approaches human-annotated meeting ground truth (GT)**
as closely as practical.

## Goal

Approach **human-annotated GT** on real multi-speaker meetings. Concretely, on
the reference meeting (`data/audio/meeting_06-29.wav` +
`data/gt/meeting_06-29_transcript.md`), the open stack reaches `frame_acc` in
the band of the strongest open references:

| System | frame_acc vs GT | DER vs GT |
|---|---:|---:|
| DiariZen large-v2 (open) | 92.1% | 18.2% |
| pyannote community-1 (open) | 80.7% | 28.4% |
| **diar-rs (open) — target** | **≥ ~90%** | — |
| **diar-rs Rust (measured)** | **90.1%** | 14.8% |

These are open reference systems, used only as **targets** for relative
comparison — not a bar we must match exactly.

## Non-goals

- Matching any specific closed-source system's output.
- Treating any one binary's "Detect" output as the success criterion.
- Carrying or redistributing proprietary weights.

## Metric contract (`python/diar_lab/compare_benchmark.py`)

The GT is a human transcript with **turn start times only** (end = next turn's
start; adjacent same-speaker turns merged). The harness rasterizes timelines at
100 ms and computes:

| Metric | Definition |
|---|---|
| `frame_acc` | agreement on speech-**union** frames under the best speaker permutation |
| `miss` / `fa` / `conf` | ref-speech-hyp-silent / hyp-speech-ref-silent / both-speech-wrong-speaker |
| `der` | (miss + fa + conf) / speech-union (relative, not full NIST DER) |
| `turn_majority` | for each ref turn ≥1 s, hyp majority label in window matches mapped id |
| `change_*` | speaker-change recall/precision at ±0.5 s |

`--native` is **optional**: when omitted, the report is a clean hypothesis-vs-GT
2-way compare. Any external system can be supplied as one more column.

## Known limitations

- **Single annotated meeting** (06-29, ~15 min, 3 speakers). A result on one
  file is not conclusive; `docs/HANDOFF.md` tracks adding public RTTM benchmarks
  (VoxConverse / AMI / Aishell-4) for breadth.
- The short third speaker (~23 s of brief replies) is missed by every system;
  it is the known hard case to attack.

## Open weights & licensing

See `models/README.md`. Summary:

- WeSpeaker embedding — **CC-BY-4.0** (commercial OK, attribution).
- DiariZen segmentation — **CC-BY-NC-4.0** (non-commercial). A commercial-safe
  upgrade path (train an MIT head on `microsoft/wavlm-base`, or swap to NeMo
  Sortformer) is documented in `models/README.md`.
- VBx, kaldi-native-fbank — Apache-2.0.
