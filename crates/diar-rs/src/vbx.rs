//! Bayesian HMM clustering (VBx) — BUT VBx core, Landini / Burget / Diez.
//!
//! Port of the Python lab `vbx.py` (from BUTSpeechFIT/VBx, Apache-2.0).

use crate::error::{Error, Result};

const LN_2PI: f64 = 1.837_877_066_409_345_3; // ln(2π)

pub struct VbxParams {
    pub fa: f64,
    pub fb: f64,
    pub loop_prob: f64,
    pub max_iters: usize,
    pub epsilon: f64,
}

impl Default for VbxParams {
    fn default() -> Self {
        Self {
            fa: 0.3,
            fb: 17.0,
            loop_prob: 0.99,
            max_iters: 20,
            epsilon: 1e-4,
        }
    }
}

/// logsumexp over a slice.
fn logsumexp(xs: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in xs {
        if x > m {
            m = x;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0;
    for &x in xs {
        s += (x - m).exp();
    }
    m + s.ln()
}

/// Forward-backward. Returns (gamma [T*K], log_pX, logA [T*K], logB [T*K]).
pub fn forward_backward(lls: &[f64], t: usize, k: usize, tr: &[f64], ip: &[f64]) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>) {
    let eps = 1e-8;
    let mut ltr = vec![0.0f64; k * k];
    for i in 0..k * k {
        ltr[i] = (tr[i] + eps).ln();
    }
    let mut lfw = vec![f64::NEG_INFINITY; t * k];
    let mut lbw = vec![f64::NEG_INFINITY; t * k];
    for j in 0..k {
        lfw[j] = lls[j] + (ip[j] + eps).ln();
    }
    for j in 0..k {
        lbw[(t - 1) * k + j] = 0.0;
    }
    let mut tmp = vec![0.0f64; k];
    for ii in 1..t {
        for j in 0..k {
            // logsumexp_i lfw[ii-1,i] + ltr[i,j]  (ltr.T means ltr[i,j] when iterating i for fixed j from ltr.T)
            // Python: logsumexp(lfw[ii-1] + ltr.T, axis=1)[j]
            // ltr.T[j,i] = ltr[i,j]
            for i in 0..k {
                tmp[i] = lfw[(ii - 1) * k + i] + ltr[i * k + j];
            }
            lfw[ii * k + j] = lls[ii * k + j] + logsumexp(&tmp);
        }
    }
    for ii in (0..t - 1).rev() {
        for i in 0..k {
            // logsumexp_j ltr[i,j] + lls[ii+1,j] + lbw[ii+1,j]
            for j in 0..k {
                tmp[j] = ltr[i * k + j] + lls[(ii + 1) * k + j] + lbw[(ii + 1) * k + j];
            }
            lbw[ii * k + i] = logsumexp(&tmp);
        }
    }
    let tll = logsumexp(&lfw[(t - 1) * k..t * k]);
    let mut gamma = vec![0.0f64; t * k];
    for i in 0..t * k {
        gamma[i] = (lfw[i] + lbw[i] - tll).exp();
    }
    (gamma, tll, lfw, lbw)
}

/// VBx clustering.
///
/// - `x`: features (T, D) row-major after PLDA transform
/// - `phi`: (D,) across-class covariance diagonal (plda psi)
/// - `gamma`: initial (T, K) responsibilities (row-stochastic)
///
/// Returns (gamma, pi, elbo_history).
pub fn vbx(
    x: &[f64],
    t: usize,
    d: usize,
    phi: &[f64],
    gamma: &mut [f64],
    k: usize,
    params: &VbxParams,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if x.len() != t * d || phi.len() != d || gamma.len() != t * k {
        return Err(Error::Pipeline("vbx shape mismatch".into()));
    }
    let fa = params.fa;
    let fb = params.fb;
    let loop_prob = params.loop_prob;

    // G = -0.5 * (sum X^2 + D ln 2π)
    let mut g = vec![0.0f64; t];
    for i in 0..t {
        let mut s2 = 0.0;
        for j in 0..d {
            let v = x[i * d + j];
            s2 += v * v;
        }
        g[i] = -0.5 * (s2 + d as f64 * LN_2PI);
    }
    // V = sqrt(Phi); rho = X * V
    let mut rho = vec![0.0f64; t * d];
    for j in 0..d {
        let v = phi[j].sqrt();
        for i in 0..t {
            rho[i * d + j] = x[i * d + j] * v;
        }
    }

    let mut pi = vec![1.0 / k as f64; k];
    let mut alpha = vec![0.0f64; k * d];
    let mut inv_l = vec![0.0f64; k * d];
    let mut elbos: Vec<f64> = Vec::new();
    let mut log_p = vec![0.0f64; t * k];

    for iter in 0..params.max_iters {
        // invL = 1 / (1 + Fa/Fb * gamma.sum(0).T * Phi)
        // gamma.sum(axis=0) → (K,)
        let mut gsum = vec![0.0f64; k];
        for i in 0..t {
            for c in 0..k {
                gsum[c] += gamma[i * k + c];
            }
        }
        for c in 0..k {
            for j in 0..d {
                inv_l[c * d + j] = 1.0 / (1.0 + fa / fb * gsum[c] * phi[j]);
            }
        }
        // alpha = Fa/Fb * invL * gamma.T @ rho   → (K, D)
        // gamma.T @ rho = sum_t gamma[t,c] * rho[t,j]
        for c in 0..k {
            for j in 0..d {
                let mut s = 0.0;
                for i in 0..t {
                    s += gamma[i * k + c] * rho[i * d + j];
                }
                alpha[c * d + j] = fa / fb * inv_l[c * d + j] * s;
            }
        }
        // log_p[t,c] = Fa * (rho[t]·alpha[c] - 0.5 * (invL[c]+alpha[c]^2)·Phi + G[t])
        for i in 0..t {
            for c in 0..k {
                let mut dot = 0.0;
                let mut quad = 0.0;
                for j in 0..d {
                    dot += rho[i * d + j] * alpha[c * d + j];
                    let a = alpha[c * d + j];
                    quad += (inv_l[c * d + j] + a * a) * phi[j];
                }
                log_p[i * k + c] = fa * (dot - 0.5 * quad + g[i]);
            }
        }
        // tr = I * loop + (1-loop) * pi  (rows sum? Python: eye*loop + (1-loop)*pi
        // pi broadcasts: tr[i,j] = loop*δ_ij + (1-loop)*pi[j]
        let mut tr = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                tr[i * k + j] = (1.0 - loop_prob) * pi[j];
            }
            tr[i * k + i] += loop_prob;
        }
        let (gamma_new, log_px, log_a, log_b) = forward_backward(&log_p, t, k, &tr, &pi);
        gamma.copy_from_slice(&gamma_new);

        // ELBO = log_pX + Fb * 0.5 * sum(log(invL) - invL - alpha^2 + 1)
        let mut elbo_extra = 0.0;
        for c in 0..k {
            for j in 0..d {
                let il = inv_l[c * d + j];
                let a = alpha[c * d + j];
                elbo_extra += il.ln() - il - a * a + 1.0;
            }
        }
        let elbo = log_px + fb * 0.5 * elbo_extra;
        elbos.push(elbo);

        // pi update
        // pi = gamma[0] + (1-loop) * pi * sum_t exp(logsumexp(logA[:-1],1) + log_p[1:] + logB[1:] - log_pX)
        let mut new_pi = vec![0.0f64; k];
        for c in 0..k {
            new_pi[c] = gamma[c]; // gamma[0]
        }
        if t > 1 {
            let mut lsum_a = vec![0.0f64; t - 1];
            for i in 0..t - 1 {
                lsum_a[i] = logsumexp(&log_a[i * k..(i + 1) * k]);
            }
            for c in 0..k {
                let mut acc = 0.0;
                for i in 0..t - 1 {
                    // row i of the sum over time for class c uses log_p[i+1,c] + logB[i+1,c]
                    let val = lsum_a[i] + log_p[(i + 1) * k + c] + log_b[(i + 1) * k + c] - log_px;
                    acc += val.exp();
                }
                new_pi[c] += (1.0 - loop_prob) * pi[c] * acc;
            }
        }
        let s: f64 = new_pi.iter().sum();
        for c in 0..k {
            pi[c] = new_pi[c] / s;
        }

        if iter > 0 && elbo - elbos[iter - 1] < params.epsilon {
            break;
        }
        let _ = log_b; // used
    }

    Ok((pi, elbos))
}

/// Argmax labels from gamma (T, K).
pub fn labels_from_gamma(gamma: &[f64], t: usize, k: usize) -> Vec<u32> {
    let mut out = vec![0u32; t];
    for i in 0..t {
        let mut best = 0usize;
        let mut bv = f64::NEG_INFINITY;
        for c in 0..k {
            let v = gamma[i * k + c];
            if v > bv {
                bv = v;
                best = c;
            }
        }
        out[i] = best as u32;
    }
    out
}

/// Refine labels; returns None if collapsed to 1 speaker.
pub fn refine(
    fea: &[f64],
    n: usize,
    dim: usize,
    labels: &[u32],
    psi: &[f64],
    params: &VbxParams,
) -> Result<Option<Vec<u32>>> {
    let k = labels.iter().copied().max().unwrap_or(0) as usize + 1;
    if n < 2 || k < 2 {
        return Ok(Some(labels.to_vec()));
    }
    let mut gamma = vec![0.0f64; n * k];
    let smooth = 5.0;
    for i in 0..n {
        for c in 0..k {
            gamma[i * k + c] = if labels[i] as usize == c { smooth } else { 0.0 };
        }
        // softmax
        let m = gamma[i * k..(i + 1) * k]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut s = 0.0;
        for c in 0..k {
            let e = (gamma[i * k + c] - m).exp();
            gamma[i * k + c] = e;
            s += e;
        }
        for c in 0..k {
            gamma[i * k + c] /= s;
        }
    }
    let (_pi, _elbo) = vbx(fea, n, dim, psi, &mut gamma, k, params)?;
    let lab = labels_from_gamma(&gamma, n, k);
    let used: std::collections::BTreeSet<u32> = lab.iter().copied().collect();
    if used.len() < 2 {
        return Ok(None);
    }
    // remap contiguous
    let map: std::collections::BTreeMap<u32, u32> = used
        .iter()
        .enumerate()
        .map(|(i, &u)| (u, i as u32))
        .collect();
    Ok(Some(lab.iter().map(|x| map[x]).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logsumexp_basic() {
        let xs = [1.0, 2.0, 3.0];
        let v = logsumexp(&xs);
        let expected = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln();
        assert!((v - expected).abs() < 1e-12);
    }
}
