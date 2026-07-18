"""PLDA + x-vector transform (float64 LE; open DiariZen npz or legacy bin).

Matches BUT VBx recipe transform:
  x = l2_norm( lda.T @ l2_norm(x - mean1) - mean2 )
  fea = (x - plda_mu) @ plda_tr.T
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _load_f64(path: Path) -> np.ndarray:
    data = path.read_bytes()
    assert len(data) % 8 == 0, path
    return np.frombuffer(data, dtype="<f8").copy()


def l2_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


@dataclass
class PldaTransform:
    mean1: np.ndarray  # (256,)
    lda: np.ndarray  # (256, 128)  — apply as (x-mean1) @ lda
    mean2: np.ndarray  # (128,)
    mu: np.ndarray  # (128,)
    tr: np.ndarray  # (128, 128)
    psi: np.ndarray  # (128,)

    @classmethod
    def load(cls, plda_dir: Path) -> "PldaTransform":
        """Load from DiariZen npz dir (plda.npz + xvec_transform.npz) or legacy bin dir."""
        plda_dir = Path(plda_dir)
        npz_plda = plda_dir / "plda.npz"
        npz_xvec = plda_dir / "xvec_transform.npz"
        if npz_plda.is_file() and npz_xvec.is_file():
            return cls.load_npz(npz_plda, npz_xvec)

        mean1 = _load_f64(plda_dir / "xvec_transform_mean1.bin")
        lda_flat = _load_f64(plda_dir / "xvec_transform_lda.bin")
        mean2 = _load_f64(plda_dir / "xvec_transform_mean2.bin")
        mu = _load_f64(plda_dir / "plda_mu.bin")
        tr_flat = _load_f64(plda_dir / "plda_tr_final.bin")
        psi = _load_f64(plda_dir / "plda_psi_final.bin")

        d_in, d_out = mean1.shape[0], mean2.shape[0]
        assert lda_flat.size == d_in * d_out
        lda = lda_flat.reshape(d_in, d_out)
        assert tr_flat.size == d_out * d_out
        tr = tr_flat.reshape(d_out, d_out)
        assert mu.shape == (d_out,) and psi.shape == (d_out,)
        # psi must be positive for VBx
        psi = np.maximum(psi, 1e-6)
        return cls(mean1=mean1, lda=lda, mean2=mean2, mu=mu, tr=tr, psi=psi)

    @classmethod
    def load_npz(cls, plda_npz: Path, xvec_npz: Path) -> "PldaTransform":
        plda = np.load(plda_npz)
        xvec = np.load(xvec_npz)
        mean1 = np.asarray(xvec["mean1"], dtype=np.float64).reshape(-1)
        mean2 = np.asarray(xvec["mean2"], dtype=np.float64).reshape(-1)
        lda = np.asarray(xvec["lda"], dtype=np.float64)
        if lda.shape == (mean2.shape[0], mean1.shape[0]):
            lda = lda.T  # (256, 128) expected
        mu = np.asarray(plda["mu"], dtype=np.float64).reshape(-1)
        tr = np.asarray(plda["tr"], dtype=np.float64)
        psi = np.maximum(np.asarray(plda["psi"], dtype=np.float64).reshape(-1), 1e-6)
        assert lda.shape == (mean1.shape[0], mean2.shape[0])
        assert tr.shape == (mean2.shape[0], mean2.shape[0])
        return cls(mean1=mean1, lda=lda, mean2=mean2, mu=mu, tr=tr, psi=psi)

    def xvec_transform(self, emb: np.ndarray) -> np.ndarray:
        """256-d embeddings → 128-d LDA space (BUT recipe)."""
        x = np.asarray(emb, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        x = l2_norm(x - self.mean1)
        x = x @ self.lda  # (N, 128)
        x = l2_norm(x - self.mean2)
        return x[0] if single else x

    def plda_space(self, x128: np.ndarray, lda_dim: int | None = None) -> np.ndarray:
        """128-d → PLDA projected features for VBx."""
        x = np.asarray(x128, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        fea = (x - self.mu) @ self.tr.T
        if lda_dim is not None:
            fea = fea[:, :lda_dim]
        return fea[0] if single else fea

    def transform_for_vbx(self, emb256: np.ndarray, lda_dim: int | None = None) -> np.ndarray:
        return self.plda_space(self.xvec_transform(emb256), lda_dim=lda_dim)

    def transform_for_ahc(self, emb256: np.ndarray) -> np.ndarray:
        """Features for cosine AHC (after xvec transform, before PLDA tr)."""
        return self.xvec_transform(emb256)
