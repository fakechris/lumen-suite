//! Open diarization pipeline — mirrors `python/diar_lab/pipeline.py`:
//!   pcm → seg-as-VAD mask → sliding WeSpeaker x-vectors → PLDA LDA →
//!   AHC(cos+2GMM) → merge. (VBx refine is TODO; it collapsed in the Python
//!   lab on this open embedding space, so AHC is the current result.)

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::audio::load_wav_mono16k;
use crate::cluster::{absorb_small, ahc_cosine, l2_normalize_rows};
use crate::config::{DiarizeConfig, ModelPaths};
use crate::error::{Error, Result};
use crate::fbank::{compute_fbank, FbankOptions};
use crate::merge::merge_postprocess;
use crate::onnx_emb::EmbModel;
use crate::onnx_seg::SegModel;
use crate::plda::Plda;
use crate::powerset::logprobs_to_multilabel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizeResult {
    pub method: String,
    pub n_turns: usize,
    pub n_chunks: usize,
    pub n_xvec: usize,
    pub talk_sec: BTreeMap<u32, f64>,
    pub timeline: Vec<Turn>,
    pub elapsed_sec: f64,
    pub frame_hz: f64,
}

#[derive(Debug, Default)]
pub struct Trace {
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DumpOpts {
    pub dir: Option<std::path::PathBuf>,
}

/// Validate open weight files exist and the PLDA loads.
pub fn validate_models(models: &ModelPaths) -> Result<()> {
    for p in [&models.segmentation, &models.embedding] {
        if !p.is_file() {
            return Err(Error::MissingModel(p.clone()));
        }
    }
    let mu = models.plda_dir.join("plda_mu.bin");
    if !mu.is_file() {
        return Err(Error::MissingModel(mu));
    }
    let _ = Plda::load(&models.plda_dir)?;
    Ok(())
}

/// Segmentation-as-VAD: chunk the audio into 16 s windows, run the seg model,
/// decode powerset → per-speaker activity, mark samples where any speaker is
/// active (>= onset). Returns a per-sample speech mask.
fn seg_speech_mask(
    pcm: &[f32],
    sr: u32,
    seg: &mut SegModel,
    cfg: &DiarizeConfig,
) -> Result<(Vec<bool>, usize)> {
    let chunk = 16 * sr as usize; // DiariZen seg expects 16 s
    let n = pcm.len();
    let mut mask = vec![false; n];
    let mut frame_hz = 0usize;
    let mut s = 0usize;
    while s < n {
        let e = (s + chunk).min(n);
        if e - s < sr as usize / 2 {
            break;
        }
        let mut piece = pcm[s..e].to_vec();
        if piece.len() < chunk {
            piece.resize(chunk, 0.0);
        }
        let (logp, t, nc) = seg.forward(&piece)?;
        if t == 0 {
            s += chunk;
            continue;
        }
        frame_hz = (t as f64 / 16.0).round() as usize;
        let logp64: Vec<f64> = logp.iter().map(|&v| v as f64).collect();
        let multi = logprobs_to_multilabel(&logp64, t, nc, cfg.num_speakers);
        let real_frames = (((e - s) as f64 / sr as f64) * frame_hz as f64).round() as usize;
        let real_frames = real_frames.min(t);
        for i in 0..real_frames {
            let mut mx = 0.0f64;
            for l in 0..cfg.num_speakers {
                mx = mx.max(multi[i * cfg.num_speakers + l]);
            }
            if mx >= cfg.onset {
                let a = s + ((i as f64 / frame_hz as f64) * sr as f64) as usize;
                let b = s + ((((i + 1) as f64) / frame_hz as f64) * sr as f64) as usize;
                for j in a.min(n)..b.min(n) {
                    mask[j] = true;
                }
            }
        }
        s += chunk;
    }
    Ok((mask, frame_hz.max(1)))
}

/// Sliding WeSpeaker x-vectors over speech regions.
/// Returns (embeddings flat [N*256], times [(start,end)]).
fn sliding_xvectors(
    pcm: &[f32],
    sr: u32,
    mask: &[bool],
    emb: &mut EmbModel,
    cfg: &DiarizeConfig,
) -> Result<(Vec<f64>, Vec<(f64, f64)>)> {
    let win_n = (cfg.xvec_win_sec * sr as f64).round() as usize;
    let hop_n = (cfg.xvec_hop_sec * sr as f64).round() as usize;
    let n = pcm.len();
    let mut fb_opts = FbankOptions::default();
    fb_opts.sample_rate = sr;
    fb_opts.subtract_mean = true;
    let mut embs = Vec::new();
    let mut times = Vec::new();
    let mut s = 0usize;
    while s + win_n <= n {
        let e = s + win_n;
        let speech = mask[s..e].iter().filter(|&&m| m).count() as f64 / win_n as f64;
        if speech >= 0.4 {
            let chunk = &pcm[s..e];
            let rms = {
                let mut ss = 0.0f64;
                for &v in chunk {
                    ss += (v as f64) * (v as f64);
                }
                (ss / win_n as f64).sqrt()
            };
            if rms >= 0.005 {
                let (fb, t_fb) = compute_fbank(chunk, &fb_opts)?;
                if t_fb >= 10 {
                    let v = emb.embed_fbank(&fb, t_fb)?;
                    embs.extend_from_slice(&v);
                    times.push((s as f64 / sr as f64, e as f64 / sr as f64));
                }
            }
        }
        s += hop_n;
    }
    Ok((embs, times))
}

/// End-to-end open diarization.
pub fn diarize(wav: &Path, models: &ModelPaths, cfg: &DiarizeConfig) -> Result<DiarizeResult> {
    diarize_ex(wav, models, cfg, &DumpOpts::default())
}

pub fn diarize_ex(
    wav: &Path,
    models: &ModelPaths,
    cfg: &DiarizeConfig,
    _dump: &DumpOpts,
) -> Result<DiarizeResult> {
    validate_models(models)?;
    let t0 = Instant::now();
    eprintln!("[1/5] load {}", wav.display());
    let (pcm, sr) = load_wav_mono16k(wav)?;
    let dur = pcm.len() as f64 / sr as f64;
    eprintln!("  {:.2} min @ {}", dur / 60.0, sr);

    eprintln!("[2/5] segmentation speech mask");
    let mut seg = SegModel::load(&models.segmentation, cfg.threads)?;
    let (mask, frame_hz) = seg_speech_mask(&pcm, sr, &mut seg, cfg)?;
    let speech_frac = mask.iter().filter(|&&m| m).count() as f64 / mask.len().max(1) as f64;
    eprintln!("  speech_frac={speech_frac:.3} frame_hz={frame_hz}");

    eprintln!("[3/5] x-vectors (WeSpeaker emb.onnx + kaldi fbank)");
    let mut emb = EmbModel::load(&models.embedding, cfg.threads)?;
    let (e_sw, times) = sliding_xvectors(&pcm, sr, &mask, &mut emb, cfg)?;
    let n_xvec = times.len();
    eprintln!("  N_xvec={n_xvec}");
    if n_xvec < 2 {
        return Err(Error::Pipeline("too few x-vectors".into()));
    }

    eprintln!("[4/5] PLDA xvec_tf + AHC");
    let plda = Plda::load(&models.plda_dir)?;
    let dim_in = plda.d_in;
    let dim_out = plda.d_out;
    let mut x128 = Vec::with_capacity(n_xvec * dim_out);
    for i in 0..n_xvec {
        let emb = &e_sw[i * dim_in..(i + 1) * dim_in];
        x128.extend_from_slice(&plda.xvec_transform(emb)?);
    }
    let labels = ahc_cosine(&x128, n_xvec, dim_out, None, cfg.ahc_max_speakers)?;
    let k = labels.iter().copied().max().unwrap_or(0) as usize + 1;
    eprintln!("  AHC speakers={k} sizes={:?}", bincount(&labels));

    eprintln!("[5/5] build turns + merge");
    let win_turns: Vec<Turn> = times
        .iter()
        .zip(labels.iter())
        .map(|((s, e), &l)| Turn {
            start: *s,
            end: *e,
            speaker: l,
        })
        .collect();
    let turns = merge_postprocess(&win_turns, cfg.merge_gap_sec, cfg.min_seg_sec);

    let mut talk: BTreeMap<u32, f64> = BTreeMap::new();
    for t in &turns {
        *talk.entry(t.speaker).or_insert(0.0) += t.end - t.start;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "done turns={} talk={:?} {:.1}s",
        turns.len(),
        talk,
        elapsed
    );

    Ok(DiarizeResult {
        method: "diar-rs/open: seg-VAD + WeSpeaker xvec + PLDA AHC + merge".into(),
        n_turns: turns.len(),
        n_chunks: (dur / 16.0).ceil() as usize,
        n_xvec,
        talk_sec: talk,
        timeline: turns,
        elapsed_sec: (elapsed * 10.0).round() / 10.0,
        frame_hz: frame_hz as f64,
    })
}

pub fn diarize_with_trace(
    wav: &Path,
    models: &ModelPaths,
    cfg: &DiarizeConfig,
) -> Result<(DiarizeResult, Trace)> {
    let r = diarize(wav, models, cfg)?;
    Ok((
        r,
        Trace {
            notes: vec!["open: seg-VAD + xvec + PLDA AHC".into()],
        },
    ))
}

fn bincount(labels: &[u32]) -> Vec<usize> {
    let m = labels.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut c = vec![0usize; m];
    for &l in labels {
        c[l as usize] += 1;
    }
    c
}

// silence unused-import warnings for helpers retained for the VBx follow-up
#[allow(dead_code)]
fn _silence(_: &crate::vbx::VbxParams, _: &[f64], _: &[f64]) {
    let _ = l2_normalize_rows;
    let _ = absorb_small;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn validate_open_models_if_present() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let models = ModelPaths::resolve(&root);
        if !models.segmentation.is_file() {
            eprintln!("skip: open weights not under models/");
            return;
        }
        validate_models(&models).expect("open models should load plda");
    }
}
