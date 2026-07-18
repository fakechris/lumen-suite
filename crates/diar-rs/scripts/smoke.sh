#!/usr/bin/env bash
# Smoke: unit/fixture tests always; optional E2E if open weights + wav exist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CRATE="$ROOT/crates/diar-rs"
cd "$CRATE"

if [[ -z "${PYTHON:-}" ]]; then
  for c in \
    "$ROOT/.venv/bin/python" \
    "$ROOT/.venv-diar-baselines/bin/python" \
    python3; do
    if { [[ -x "$c" ]] || command -v "$c" >/dev/null 2>&1; } && "$c" -c "import kaldi_native_fbank" 2>/dev/null; then
      export PYTHON="$c"
      break
    fi
  done
fi
if [[ -z "${PYTHON:-}" ]]; then
  echo "WARN: no python with kaldi_native_fbank; build may fail" >&2
fi

# Runtime dylib for knf on macOS
if [[ -n "${PYTHON:-}" ]]; then
  KNF_LIB="$("$PYTHON" -c "import kaldi_native_fbank as k, pathlib; print(pathlib.Path(k.__file__).parent/'lib')" 2>/dev/null || true)"
  if [[ -n "$KNF_LIB" && -d "$KNF_LIB" ]]; then
    export DYLD_LIBRARY_PATH="$KNF_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$KNF_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

echo "[smoke] cargo test"
cargo test --release

WAV="$ROOT/data/audio/meeting_06-29.wav"
MODELS="$ROOT/models"
# TODO(stage5): confirm open-weight filenames once fetch_models.py lands.
if [[ -f "$WAV" && -f "$MODELS/seg.onnx" && -f "$MODELS/emb.onnx" ]]; then
  echo "[smoke] E2E diarize"
  cargo build --release
  OUT="$ROOT/runs/smoke_e2e"
  ./target/release/diar-rs diarize --wav "$WAV" --out "$OUT" --threads 2
  python3 - <<PY
import json
from pathlib import Path
p = Path("$OUT/diarization.json")
d = json.loads(p.read_text())
assert d["n_turns"] >= 2, d
assert d["n_xvec"] >= 10, d
print("OK smoke e2e turns=", d["n_turns"], "xvec=", d["n_xvec"])
PY
else
  echo "[smoke] skip E2E (missing wav or open weights under models/)"
fi

echo "[smoke] done"
