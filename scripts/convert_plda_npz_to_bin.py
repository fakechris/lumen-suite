#!/usr/bin/env python3
"""Convert the open DiariZen PLDA npz → the 6 float64-LE .bin files the Rust
crate's Plda::load expects (models/plda/*.bin). Python keeps using the npz.

  plda.npz          {mu, tr, psi}          → plda_mu.bin, plda_tr_final.bin, plda_psi_final.bin
  xvec_transform.npz {mean1, mean2, lda}    → xvec_transform_mean1.bin, _mean2.bin, _lda.bin
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLDA = ROOT / "models" / "plda"


def dump(name: str, arr: np.ndarray) -> None:
    a = np.ascontiguousarray(arr, dtype="<f8")
    (PLDA / f"{name}.bin").write_bytes(a.tobytes())
    print(f"  {name}.bin  shape={a.shape}  {a.nbytes} bytes")


def main() -> None:
    assert PLDA.is_dir(), f"{PLDA} missing — run fetch first"
    plda = np.load(PLDA / "plda.npz")
    xvec = np.load(PLDA / "xvec_transform.npz")
    dump("plda_mu", plda["mu"].reshape(-1))
    dump("plda_tr_final", plda["tr"])
    dump("plda_psi_final", plda["psi"].reshape(-1))
    dump("xvec_transform_mean1", xvec["mean1"].reshape(-1))
    dump("xvec_transform_mean2", xvec["mean2"].reshape(-1))
    lda = xvec["lda"]
    if lda.shape == (xvec["mean2"].shape[0], xvec["mean1"].shape[0]):
        lda = lda.T  # → (256, 128) row-major
    assert lda.shape == (xvec["mean1"].shape[0], xvec["mean2"].shape[0]), lda.shape
    dump("xvec_transform_lda", lda)
    print("done → models/plda/*.bin (for the Rust crate)")


if __name__ == "__main__":
    main()
