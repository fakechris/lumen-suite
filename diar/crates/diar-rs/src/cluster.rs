//! Agglomerative clustering matching Python `native_pipeline.ahc_cosine`
//! (fastcluster average linkage + twoGMM cut + silhouette + absorb).

use kodama::{linkage, Method, Step};

use crate::error::Result;

pub fn l2_normalize_rows(x: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        let row = &mut x[i * dim..(i + 1) * dim];
        let mut s = 0.0;
        for v in row.iter() {
            s += *v * *v;
        }
        let nrm = s.sqrt().max(1e-12);
        for v in row.iter_mut() {
            *v /= nrm;
        }
    }
}

/// Two-Gaussian score calibration → AHC threshold (BUT diarization_lib).
pub fn two_gmm_calib_lin(s: &[f64], niters: usize) -> f64 {
    let n = s.len();
    if n == 0 {
        return f64::INFINITY;
    }
    let mean: f64 = s.iter().sum::<f64>() / n as f64;
    let var0: f64 = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let std = var0.sqrt();
    let mut weights = [0.5f64, 0.5];
    let mut means = [mean - std, mean + std];
    let mut var = var0.max(1e-12);
    let mut threshold = f64::INFINITY;
    for _ in 0..niters {
        let mut cnts = [0.0f64; 2];
        let mut sum_s = [0.0f64; 2];
        let mut sum_s2 = [0.0f64; 2];
        for &si in s {
            let mut lls = [0.0f64; 2];
            for k in 0..2 {
                lls[k] = weights[k].ln()
                    - 0.5 * (var + 1e-12).ln()
                    - 0.5 * (si - means[k]) * (si - means[k]) / (var + 1e-12);
            }
            let m = lls[0].max(lls[1]);
            let e0 = (lls[0] - m).exp();
            let e1 = (lls[1] - m).exp();
            let z = e0 + e1;
            let g0 = e0 / z;
            let g1 = e1 / z;
            cnts[0] += g0;
            cnts[1] += g1;
            sum_s[0] += si * g0;
            sum_s[1] += si * g1;
            sum_s2[0] += si * si * g0;
            sum_s2[1] += si * si * g1;
        }
        cnts[0] += 1e-12;
        cnts[1] += 1e-12;
        let tot = cnts[0] + cnts[1];
        weights[0] = cnts[0] / tot;
        weights[1] = cnts[1] / tot;
        means[0] = sum_s[0] / cnts[0];
        means[1] = sum_s[1] / cnts[1];
        let v0 = sum_s2[0] / cnts[0] - means[0] * means[0];
        let v1 = sum_s2[1] / cnts[1] - means[1] * means[1];
        var = (v0 * weights[0] + v1 * weights[1]).max(1e-12);
        let t0 = (weights[0] * weights[0] / var).ln() - means[0] * means[0] / var;
        let t1 = (weights[1] * weights[1] / var).ln() - means[1] * means[1] / var;
        let num = -0.5 * (t0 - t1);
        let den = (means[0] / var - means[1] / var) + 1e-12;
        threshold = num / den;
    }
    threshold
}

/// Absorb clusters smaller than min_size into nearest large centroid (cosine).
pub fn absorb_small(x: &[f64], n: usize, dim: usize, labels: &[u32], min_size: usize) -> Vec<u32> {
    let mut labels = labels.to_vec();
    let mut x_n = x.to_vec();
    l2_normalize_rows(&mut x_n, n, dim);
    for _ in 0..8 {
        let max_lab = labels.iter().copied().max().unwrap_or(0) as usize;
        let mut sizes = vec![0usize; max_lab + 1];
        for &l in &labels {
            sizes[l as usize] += 1;
        }
        let large: Vec<usize> = sizes
            .iter()
            .enumerate()
            .filter(|(_, &s)| s >= min_size)
            .map(|(i, _)| i)
            .collect();
        let small: Vec<usize> = sizes
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > 0 && s < min_size)
            .map(|(i, _)| i)
            .collect();
        if small.is_empty() || large.is_empty() {
            break;
        }
        let mut cents = vec![vec![0.0f64; dim]; large.len()];
        let mut cnt = vec![0.0f64; large.len()];
        for i in 0..n {
            let lab = labels[i] as usize;
            if let Some(pos) = large.iter().position(|&l| l == lab) {
                for j in 0..dim {
                    cents[pos][j] += x_n[i * dim + j];
                }
                cnt[pos] += 1.0;
            }
        }
        for (c, &cc) in cents.iter_mut().zip(cnt.iter()) {
            for v in c.iter_mut() {
                *v /= cc.max(1e-12);
            }
            let mut s = 0.0;
            for v in c.iter() {
                s += *v * *v;
            }
            let nrm = s.sqrt().max(1e-12);
            for v in c.iter_mut() {
                *v /= nrm;
            }
        }
        for &si in &small {
            for i in 0..n {
                if labels[i] as usize != si {
                    continue;
                }
                let mut best = large[0];
                let mut bd = -1.0f64;
                for (pos, &li) in large.iter().enumerate() {
                    let mut d = 0.0;
                    for j in 0..dim {
                        d += x_n[i * dim + j] * cents[pos][j];
                    }
                    if d > bd {
                        bd = d;
                        best = li;
                    }
                }
                labels[i] = best as u32;
            }
        }
        let mut used: Vec<u32> = labels.clone();
        used.sort_unstable();
        used.dedup();
        let map: std::collections::BTreeMap<u32, u32> = used
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i as u32))
            .collect();
        for l in labels.iter_mut() {
            *l = map[l];
        }
    }
    labels
}

fn condensed_cosine_dist(xn: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n - 1 {
        for j in i + 1..n {
            let mut d = 0.0;
            for k in 0..dim {
                d += xn[i * dim + k] * xn[j * dim + k];
            }
            condensed.push((1.0 - d).clamp(0.0, 2.0));
        }
    }
    condensed
}

/// Labels from dendrogram steps: take first `n - k` merges (maxclust=k).
fn labels_maxclust(steps: &[Step<f64>], n: usize, k: usize) -> Vec<u32> {
    let k = k.clamp(1, n);
    let merges = n - k;
    let total_nodes = n + merges;
    let mut parent: Vec<usize> = (0..total_nodes).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for (si, step) in steps.iter().enumerate().take(merges) {
        let new_id = n + si;
        let ra = find(&mut parent, step.cluster1);
        let rb = find(&mut parent, step.cluster2);
        parent[ra] = new_id;
        parent[rb] = new_id;
        parent[new_id] = new_id;
    }
    let mut root_ids: Vec<usize> = Vec::new();
    let mut labels = vec![0u32; n];
    for i in 0..n {
        let r = find(&mut parent, i);
        if let Some(pos) = root_ids.iter().position(|&x| x == r) {
            labels[i] = pos as u32;
        } else {
            labels[i] = root_ids.len() as u32;
            root_ids.push(r);
        }
    }
    labels
}

/// Labels by distance cut: merge while dissimilarity <= cut.
fn labels_distance_cut(steps: &[Step<f64>], n: usize, cut: f64) -> Vec<u32> {
    let mut merges = 0usize;
    for step in steps {
        if step.dissimilarity > cut {
            break;
        }
        merges += 1;
    }
    let k = n - merges;
    labels_maxclust(steps, n, k.max(1))
}

/// Mean silhouette (cosine) — simplified O(n²k) for n~1k.
fn silhouette_cosine(xn: &[f64], n: usize, dim: usize, labels: &[u32]) -> f64 {
    let k = labels.iter().copied().max().unwrap_or(0) as usize + 1;
    if k < 2 || n < 2 {
        return -1.0;
    }
    let mut sizes = vec![0usize; k];
    for &l in labels {
        sizes[l as usize] += 1;
    }
    if sizes.iter().any(|&s| s == 0) {
        return -1.0;
    }
    let mut sil_sum = 0.0f64;
    for i in 0..n {
        let li = labels[i] as usize;
        // mean dist to each cluster
        let mut sum_d = vec![0.0f64; k];
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut d = 0.0;
            for t in 0..dim {
                d += xn[i * dim + t] * xn[j * dim + t];
            }
            let dist = (1.0 - d).clamp(0.0, 2.0);
            sum_d[labels[j] as usize] += dist;
        }
        let a = if sizes[li] > 1 {
            sum_d[li] / (sizes[li] - 1) as f64
        } else {
            0.0
        };
        let mut b = f64::INFINITY;
        for c in 0..k {
            if c == li {
                continue;
            }
            let mean = sum_d[c] / sizes[c] as f64;
            if mean < b {
                b = mean;
            }
        }
        let s = if a == 0.0 && b == 0.0 {
            0.0
        } else {
            (b - a) / a.max(b)
        };
        sil_sum += s;
    }
    sil_sum / n as f64
}

/// Iteratively merge closest pairs of clusters (by centroid cosine similarity) until k <= target_k.
pub fn merge_closest_clusters(
    x_n: &[f64],
    n: usize,
    dim: usize,
    labels: &mut [u32],
    target_k: usize,
) {
    loop {
        let mut used: Vec<u32> = labels.to_vec();
        used.sort_unstable();
        used.dedup();
        let k = used.len();
        if k <= target_k {
            break;
        }
        let map: std::collections::BTreeMap<u32, usize> = used
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect();
        let mut cents = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0.0f64; k];
        for i in 0..n {
            let l = map[&labels[i]];
            for d in 0..dim {
                cents[l][d] += x_n[i * dim + d];
            }
            counts[l] += 1.0;
        }
        for (c, cnt) in cents.iter_mut().zip(&counts) {
            let inv = 1.0 / cnt.max(1e-12);
            for v in c.iter_mut() {
                *v *= inv;
            }
            let norm = c.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            for v in c.iter_mut() {
                *v /= norm;
            }
        }
        let mut best_sim = -2.0f64;
        let mut best_pair = (0, 1);
        for i in 0..k - 1 {
            for j in i + 1..k {
                let mut sim = 0.0f64;
                for d in 0..dim {
                    sim += cents[i][d] * cents[j][d];
                }
                if sim > best_sim {
                    best_sim = sim;
                    best_pair = (i, j);
                }
            }
        }
        let (merge_into, to_remove) = (used[best_pair.0], used[best_pair.1]);
        for l in labels.iter_mut() {
            if *l == to_remove {
                *l = merge_into;
            }
        }
        // renumber
        let mut u2 = labels.to_vec();
        u2.sort_unstable();
        u2.dedup();
        let m2: std::collections::BTreeMap<u32, u32> = u2
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i as u32))
            .collect();
        for l in labels.iter_mut() {
            *l = m2[l];
        }
    }
}

/// Cosine AHC matching Python `ahc_cosine`.
///
/// - `force_k`: if Some, maxclust cut only (+ absorb)
/// - else: twoGMM distance cut; if k∉[2, max_speakers], silhouette pick maxclust
pub fn ahc_cosine(
    x: &[f64],
    n: usize,
    dim: usize,
    force_k: Option<usize>,
    max_speakers: usize,
) -> Result<Vec<u32>> {
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![0]);
    }
    let mut xn = x.to_vec();
    l2_normalize_rows(&mut xn, n, dim);

    let mut condensed = condensed_cosine_dist(&xn, n, dim);
    let dend = linkage(&mut condensed, n, Method::Average);
    let steps = dend.steps();
    let min_size = 3.max(8.min(n / 20));

    let labels = if let Some(k) = force_k {
        let raw = labels_maxclust(steps, n, k.min(n));
        absorb_small(&xn, n, dim, &raw, min_size)
    } else {
        // pairwise cosine scores for twoGMM (not distances)
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n - 1 {
            for j in i + 1..n {
                let mut d = 0.0;
                for k in 0..dim {
                    d += xn[i * dim + k] * xn[j * dim + k];
                }
                scores.push(d);
            }
        }
        let thr = two_gmm_calib_lin(&scores, 20);
        let cut = (1.0 - thr).max(1e-6);
        let raw_labels = labels_distance_cut(steps, n, cut);
        let mut labels = absorb_small(&xn, n, dim, &raw_labels, min_size);
        let k = labels.iter().copied().max().unwrap_or(0) as usize + 1;
        eprintln!("  two_gmm thr={:.4} cut={:.4} post_absorb_k={}", thr, cut, k);

        if k > max_speakers {
            eprintln!("  merging {} clusters down to max_speakers={}", k, max_speakers);
            merge_closest_clusters(&xn, n, dim, &mut labels, max_speakers);
        } else if k < 2 {
            let mut best_k = 2usize;
            let mut best_sil = -1.0f64;
            let mut best_labels = labels_maxclust(steps, n, 2);
            let kmax = max_speakers.min(n);
            for kk in 2..=kmax {
                let lab_k = labels_maxclust(steps, n, kk);
                let lab_absorbed = absorb_small(&xn, n, dim, &lab_k, min_size);
                let k_abs = lab_absorbed.iter().copied().max().unwrap_or(0) as usize + 1;
                if k_abs < 2 {
                    continue;
                }
                let sil = silhouette_cosine(&xn, n, dim, &lab_absorbed);
                if sil > best_sil {
                    best_sil = sil;
                    best_k = kk;
                    best_labels = lab_absorbed;
                }
            }
            if best_sil > -1.0 {
                eprintln!("  AHC silhouette best_k={} sil={:.4}", best_k, best_sil);
                labels = best_labels;
            } else {
                eprintln!("  AHC single speaker confirmed");
                labels = vec![0; n];
            }
        }
        labels
    };

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_gmm_runs() {
        let s: Vec<f64> = (0..50).map(|i| if i < 25 { 0.1 } else { 0.9 }).collect();
        let thr = two_gmm_calib_lin(&s, 20);
        assert!(thr.is_finite());
    }

    #[test]
    fn merge_closest_clusters_reduces_k() {
        // 3 clusters in 2D: cluster 0 and 1 close, cluster 2 far
        let n = 6;
        let dim = 2;
        let x = vec![
            1.0, 0.0,
            1.0, 0.1,
            0.9, 0.1,
            0.0, 1.0,
            0.0, 0.9,
            -1.0, 0.0,
        ];
        let mut labels = vec![0, 0, 1, 2, 2, 3];
        merge_closest_clusters(&x, n, dim, &mut labels, 2);
        let k = labels.iter().copied().max().unwrap() + 1;
        assert_eq!(k, 2);
    }
}

