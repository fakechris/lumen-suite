//! Powerset multi-class → multi-label (max 4 speakers, max 2 concurrent → 11 classes).

/// Generate powerset class membership lists (same order as Python `powerset_classes`).
pub fn powerset_classes(n: usize, max_active: usize) -> Vec<Vec<usize>> {
    let mut classes: Vec<Vec<usize>> = vec![vec![]];
    for k in 1..=max_active {
        let mut combos = Vec::new();
        combinations(n, k, &mut combos);
        classes.extend(combos);
    }
    classes
}

fn combinations(n: usize, k: usize, out: &mut Vec<Vec<usize>>) {
    let mut cur = Vec::with_capacity(k);
    fn rec(start: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..n {
            cur.push(i);
            rec(i + 1, n, k, cur, out);
            cur.pop();
        }
    }
    rec(0, n, k, &mut cur, out);
}

/// log-probs [T, 11] → multi-label [T, 4] soft scores.
pub fn logprobs_to_multilabel(logp: &[f64], t: usize, n_class: usize, n_spk: usize) -> Vec<f64> {
    assert_eq!(logp.len(), t * n_class);
    let classes = powerset_classes(n_spk, 2);
    assert_eq!(classes.len(), n_class);
    let mut multi = vec![0.0f64; t * n_spk];
    for ti in 0..t {
        let row = &logp[ti * n_class..(ti + 1) * n_class];
        let maxv = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut p = vec![0.0f64; n_class];
        let mut s = 0.0;
        for c in 0..n_class {
            p[c] = (row[c] - maxv).exp();
            s += p[c];
        }
        for c in 0..n_class {
            p[c] /= s + 1e-12;
            for &spk in &classes[c] {
                multi[ti * n_spk + spk] += p[c];
            }
        }
    }
    multi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eleven_classes() {
        let c = powerset_classes(4, 2);
        assert_eq!(c.len(), 11);
        assert!(c[0].is_empty());
        assert_eq!(c[1], vec![0]);
        assert_eq!(c[2], vec![1]);
        // last should be pair (2,3)
        assert_eq!(c[10], vec![2, 3]);
    }
}
