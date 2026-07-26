//! WAV load → mono f32 @ 16 kHz.

use std::path::Path;

use hound::{SampleFormat, WavReader};

use crate::error::{Error, Result};

/// Load WAV as mono f32. Resamples with linear interpolation if needed.
pub fn load_wav_mono16k(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        WavReader::open(path).map_err(|e| Error::Pipeline(format!("wav open: {e}")))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let ch = spec.channels as usize;

    let mut mono: Vec<f32> = Vec::new();
    match spec.sample_format {
        SampleFormat::Float => {
            let samples: std::result::Result<Vec<f32>, _> = reader.samples::<f32>().collect();
            let samples = samples.map_err(|e| Error::Pipeline(format!("wav f32: {e}")))?;
            if ch == 1 {
                mono = samples;
            } else {
                for frame in samples.chunks(ch) {
                    let s: f32 = frame.iter().sum::<f32>() / ch as f32;
                    mono.push(s);
                }
            }
        }
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let samples: std::result::Result<Vec<i32>, _> = reader.samples::<i32>().collect();
            let samples = samples.map_err(|e| Error::Pipeline(format!("wav i32: {e}")))?;
            let scale = (1i64 << (bits.saturating_sub(1))) as f32;
            if ch == 1 {
                mono = samples.into_iter().map(|s| s as f32 / scale).collect();
            } else {
                for frame in samples.chunks(ch) {
                    let s: f32 = frame.iter().map(|&v| v as f32 / scale).sum::<f32>() / ch as f32;
                    mono.push(s);
                }
            }
        }
    }

    if sr == 16_000 {
        return Ok((mono, 16_000));
    }
    // linear resample
    let new_len = (mono.len() as u64 * 16_000 / sr as u64) as usize;
    if new_len == 0 {
        return Ok((vec![], 16_000));
    }
    let mut out = vec![0.0f32; new_len];
    let old_n = mono.len();
    for i in 0..new_len {
        let u = if new_len == 1 {
            0.0
        } else {
            i as f64 / (new_len - 1) as f64
        };
        let src = u * (old_n.saturating_sub(1)) as f64;
        let j = src.floor() as usize;
        let f = (src - j as f64) as f32;
        let a = mono[j.min(old_n - 1)];
        let b = mono[(j + 1).min(old_n - 1)];
        out[i] = a * (1.0 - f) + b * f;
    }
    Ok((out, 16_000))
}
