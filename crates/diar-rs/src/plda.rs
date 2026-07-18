//! PLDA + x-vector transform (float64 LE; open npz or legacy bin).
//!
//! Mirrors the Python lab `plda.py` / BUT VBx recipe:
//!   x = l2_norm( (l2_norm(x - mean1) @ lda) - mean2 )
//!   fea = (x - mu) @ tr.T

use std::fs;
use std::path::Path;

use byteorder::{ByteOrder, LittleEndian};

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct Plda {
    pub mean1: Vec<f64>, // 256
    pub lda: Vec<f64>,   // 256 * 128 row-major (x @ lda)
    pub mean2: Vec<f64>, // 128
    pub mu: Vec<f64>,    // 128
    pub tr: Vec<f64>,    // 128 * 128 row-major
    pub psi: Vec<f64>,   // 128
    pub d_in: usize,
    pub d_out: usize,
}

fn load_f64_le(path: &Path) -> Result<Vec<f64>> {
    let bytes = fs::read(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Error::MissingModel(path.to_path_buf())
        } else {
            Error::Io(e)
        }
    })?;
    if bytes.len() % 8 != 0 {
        return Err(Error::Plda(format!(
            "{}: size {} not multiple of 8",
            path.display(),
            bytes.len()
        )));
    }
    let n = bytes.len() / 8;
    let mut out = vec![0.0f64; n];
    LittleEndian::read_f64_into(&bytes, &mut out);
    Ok(out)
}

fn l2_norm_row(x: &mut [f64]) {
    let mut s = 0.0;
    for v in x.iter() {
        s += *v * *v;
    }
    let n = s.sqrt().max(1e-12);
    for v in x.iter_mut() {
        *v /= n;
    }
}

impl Plda {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mean1 = load_f64_le(&dir.join("xvec_transform_mean1.bin"))?;
        let lda = load_f64_le(&dir.join("xvec_transform_lda.bin"))?;
        let mean2 = load_f64_le(&dir.join("xvec_transform_mean2.bin"))?;
        let mu = load_f64_le(&dir.join("plda_mu.bin"))?;
        let tr = load_f64_le(&dir.join("plda_tr_final.bin"))?;
        let mut psi = load_f64_le(&dir.join("plda_psi_final.bin"))?;

        let d_in = mean1.len();
        let d_out = mean2.len();
        if lda.len() != d_in * d_out {
            return Err(Error::Plda(format!(
                "lda len {} != {}x{}",
                lda.len(),
                d_in,
                d_out
            )));
        }
        if tr.len() != d_out * d_out || mu.len() != d_out || psi.len() != d_out {
            return Err(Error::Plda("mu/tr/psi shape mismatch".into()));
        }
        for p in psi.iter_mut() {
            *p = p.max(1e-6);
        }
        Ok(Self {
            mean1,
            lda,
            mean2,
            mu,
            tr,
            psi,
            d_in,
            d_out,
        })
    }

    /// Single 256-d embedding → 128-d AHC space.
    pub fn xvec_transform(&self, emb: &[f64]) -> Result<Vec<f64>> {
        if emb.len() != self.d_in {
            return Err(Error::Plda(format!(
                "emb dim {} != {}",
                emb.len(),
                self.d_in
            )));
        }
        let mut x: Vec<f64> = emb
            .iter()
            .zip(self.mean1.iter())
            .map(|(a, b)| a - b)
            .collect();
        l2_norm_row(&mut x);
        // x @ lda  → 128
        let mut y = vec![0.0f64; self.d_out];
        for j in 0..self.d_out {
            let mut s = 0.0;
            for i in 0..self.d_in {
                s += x[i] * self.lda[i * self.d_out + j];
            }
            y[j] = s;
        }
        for (v, m) in y.iter_mut().zip(self.mean2.iter()) {
            *v -= m;
        }
        l2_norm_row(&mut y);
        Ok(y)
    }

    /// 128-d → PLDA projected features for VBx: (x - mu) @ tr.T
    pub fn plda_space(&self, x128: &[f64]) -> Result<Vec<f64>> {
        if x128.len() != self.d_out {
            return Err(Error::Plda(format!(
                "x128 dim {} != {}",
                x128.len(),
                self.d_out
            )));
        }
        let mut centered = vec![0.0f64; self.d_out];
        for i in 0..self.d_out {
            centered[i] = x128[i] - self.mu[i];
        }
        // fea_j = sum_i centered_i * tr[j, i]  == (x-mu) @ tr.T
        let mut fea = vec![0.0f64; self.d_out];
        for j in 0..self.d_out {
            let mut s = 0.0;
            for i in 0..self.d_out {
                s += centered[i] * self.tr[j * self.d_out + i];
            }
            fea[j] = s;
        }
        Ok(fea)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Open PLDA (.bin converted from the DiariZen npz) under models/plda.
    fn open_plda() -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/plda");
        if p.join("plda_mu.bin").is_file() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn load_open_plda_shapes() {
        let Some(dir) = open_plda() else {
            eprintln!("skip: open plda not under models/plda");
            return;
        };
        let p = Plda::load(&dir).expect("load");
        assert_eq!(p.d_in, 256);
        assert_eq!(p.d_out, 128);
        assert_eq!(p.psi.len(), 128);
        assert!(p.psi.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn xvec_roundtrip_dims() {
        let Some(dir) = open_plda() else {
            return;
        };
        let p = Plda::load(&dir).unwrap();
        let emb = vec![0.01f64; 256];
        let x = p.xvec_transform(&emb).unwrap();
        assert_eq!(x.len(), 128);
        let f = p.plda_space(&x).unwrap();
        assert_eq!(f.len(), 128);
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "xvec should be l2-normalized");
    }
}
