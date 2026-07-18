//! Exclusive argmax + median smooth + gap merge (Python `native_pipeline`).

use std::collections::HashMap;

use crate::pipeline::Turn;

/// Mode of labels; ties broken by first-seen (Python `Counter.most_common`).
fn mode_first_seen(vals: &[i32]) -> Option<i32> {
    if vals.is_empty() {
        return None;
    }
    let mut counts: HashMap<i32, usize> = HashMap::new();
    let mut order: Vec<i32> = Vec::new();
    for &v in vals {
        let e = counts.entry(v).or_insert(0);
        if *e == 0 {
            order.push(v);
        }
        *e += 1;
    }
    let mut best = order[0];
    let mut best_c = counts[&best];
    for &v in &order[1..] {
        let c = counts[&v];
        if c > best_c {
            best = v;
            best_c = c;
        }
    }
    Some(best)
}

fn median_filter_labels(votes: &[i32], active: &[bool], radius: usize) -> Vec<i32> {
    if radius == 0 {
        return votes.to_vec();
    }
    let n = votes.len();
    let mut out = votes.to_vec();
    for i in 0..n {
        if !active[i] {
            continue;
        }
        let a = i.saturating_sub(radius);
        let b = (i + radius + 1).min(n);
        let mut window = Vec::new();
        for j in a..b {
            if active[j] {
                window.push(votes[j]);
            }
        }
        if let Some(best) = mode_first_seen(&window) {
            out[i] = best;
        }
    }
    out
}

/// Per-frame argmax if max ≥ onset.
pub fn exclusive_turns(hard: &[f64], t: usize, k: usize, frame_hz: f64, onset: f64, median_sec: f64) -> Vec<Turn> {
    if t == 0 || k == 0 {
        return vec![];
    }
    let mut votes = vec![0i32; t];
    let mut active = vec![false; t];
    for i in 0..t {
        let mut best = 0usize;
        let mut mx = f64::NEG_INFINITY;
        for c in 0..k {
            let v = hard[i * k + c];
            if v > mx {
                mx = v;
                best = c;
            }
        }
        votes[i] = best as i32;
        active[i] = mx >= onset;
    }
    let radius = ((median_sec * frame_hz / 2.0).round() as i64).max(0) as usize;
    if radius > 0 {
        votes = median_filter_labels(&votes, &active, radius);
    }

    let mut turns: Vec<Turn> = Vec::new();
    let mut cur: Option<Turn> = None;
    for i in 0..t {
        if !active[i] {
            if let Some(t0) = cur.take() {
                turns.push(t0);
            }
            continue;
        }
        let sp = votes[i] as u32;
        let t0 = i as f64 / frame_hz;
        let t1 = (i + 1) as f64 / frame_hz;
        match cur.as_mut() {
            Some(c) if c.speaker == sp => c.end = t1,
            _ => {
                if let Some(prev) = cur.take() {
                    turns.push(prev);
                }
                cur = Some(Turn {
                    start: t0,
                    end: t1,
                    speaker: sp,
                });
            }
        }
    }
    if let Some(c) = cur {
        turns.push(c);
    }
    turns
}

/// Python `merge_postprocess`.
pub fn merge_postprocess(turns: &[Turn], same_spk_gap: f64, min_seg_sec: f64) -> Vec<Turn> {
    if turns.is_empty() {
        return vec![];
    }
    let mut turns: Vec<Turn> = turns.to_vec();
    turns.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap()
            .then(a.end.partial_cmp(&b.end).unwrap())
    });

    // 1) tight adjacent
    let mut merged: Vec<Turn> = Vec::new();
    for t in turns {
        if let Some(last) = merged.last_mut() {
            if last.speaker == t.speaker && t.start <= last.end + 0.05 {
                last.end = last.end.max(t.end);
                continue;
            }
        }
        merged.push(t);
    }

    // 2) midpoint resolve overlaps
    let mut fixed: Vec<Turn> = Vec::new();
    for mut t in merged {
        if let Some(last) = fixed.last_mut() {
            if t.start < last.end {
                if t.speaker == last.speaker {
                    last.end = last.end.max(t.end);
                    continue;
                }
                let mid = 0.5 * (t.start + last.end);
                last.end = mid;
                t.start = mid;
                if t.end - t.start < min_seg_sec {
                    continue;
                }
            }
        }
        fixed.push(t);
    }

    // 3) absorb short + same-spk gap fill
    let mut out: Vec<Turn> = Vec::new();
    for t in fixed {
        if t.end - t.start < min_seg_sec {
            if let Some(last) = out.last_mut() {
                last.end = last.end.max(t.end);
            }
            continue;
        }
        if let Some(last) = out.last_mut() {
            if last.speaker == t.speaker && t.start <= last.end + same_spk_gap {
                last.end = last.end.max(t.end);
                continue;
            }
        }
        out.push(t);
    }

    // 4) final midpoint + tight same-spk
    let mut final_t: Vec<Turn> = Vec::new();
    for mut t in out {
        if let Some(last) = final_t.last_mut() {
            if t.start < last.end && t.speaker != last.speaker {
                let mid = 0.5 * (t.start + last.end);
                last.end = mid;
                t.start = mid;
            }
            if last.speaker == t.speaker && t.start <= last.end + same_spk_gap {
                last.end = last.end.max(t.end);
                continue;
            }
        }
        if t.end - t.start >= 0.15 {
            final_t.push(t);
        }
    }

    // renumber by talk time
    let mut talk: HashMap<u32, f64> = HashMap::new();
    for t in &final_t {
        *talk.entry(t.speaker).or_insert(0.0) += t.end - t.start;
    }
    let mut order: Vec<(u32, f64)> = talk.into_iter().collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let map: HashMap<u32, u32> = order
        .iter()
        .enumerate()
        .map(|(i, (old, _))| (*old, i as u32))
        .collect();
    final_t
        .into_iter()
        .map(|t| Turn {
            start: t.start,
            end: t.end,
            speaker: map[&t.speaker],
        })
        .collect()
}
