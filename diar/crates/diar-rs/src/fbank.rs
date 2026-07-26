//! Fbank via C++ `kaldi-native-fbank` (same library as Python `kaldi_native_fbank`).
//!
//! Linked at build time through `native/knf_c_api.cpp` + `libkaldi-native-fbank-core`.

use std::os::raw::{c_char, c_float, c_int};
use std::ptr;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct FbankOptions {
    pub sample_rate: u32,
    pub num_mel_bins: usize,
    pub frame_length_ms: f64,
    pub frame_shift_ms: f64,
    pub dither: f64,
    pub preemph: f64,
    pub remove_dc: bool,
    pub snip_edges: bool,
    pub low_freq: f64,
    /// Kaldi: negative → nyquist + high_freq (Python uses -400).
    pub high_freq: f64,
    pub subtract_mean: bool,
    pub window_type: String,
}

impl Default for FbankOptions {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            num_mel_bins: 80,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            dither: 0.0,
            preemph: 0.97,
            remove_dc: true,
            snip_edges: true,
            low_freq: 20.0,
            high_freq: -400.0,
            subtract_mean: true,
            window_type: "hamming".into(),
        }
    }
}

#[repr(C)]
struct KnfOnlineFbank {
    _private: [u8; 0],
}

extern "C" {
    fn knf_online_fbank_new(
        samp_freq: c_float,
        frame_length_ms: c_float,
        frame_shift_ms: c_float,
        dither: c_float,
        preemph: c_float,
        remove_dc: c_int,
        window_type: *const c_char,
        snip_edges: c_int,
        num_bins: c_int,
        low_freq: c_float,
        high_freq: c_float,
        use_energy: c_int,
        use_log_fbank: c_int,
        use_power: c_int,
    ) -> *mut KnfOnlineFbank;

    fn knf_online_fbank_accept(
        f: *mut KnfOnlineFbank,
        sampling_rate: c_float,
        waveform: *const c_float,
        n: c_int,
    );
    fn knf_online_fbank_input_finished(f: *mut KnfOnlineFbank);
    fn knf_online_fbank_num_frames(f: *const KnfOnlineFbank) -> c_int;
    fn knf_online_fbank_dim(f: *const KnfOnlineFbank) -> c_int;
    fn knf_online_fbank_get_frame(f: *const KnfOnlineFbank, i: c_int, out: *mut c_float) -> c_int;
    fn knf_online_fbank_free(f: *mut KnfOnlineFbank);
}

/// Compute log-mel fbank: row-major [T * num_mel_bins] f32, and T.
pub fn compute_fbank(pcm: &[f32], opts: &FbankOptions) -> Result<(Vec<f32>, usize)> {
    if pcm.is_empty() {
        return Ok((vec![], 0));
    }
    let win = std::ffi::CString::new(opts.window_type.as_str())
        .map_err(|_| Error::Pipeline("window_type contains NUL".into()))?;

    let f = unsafe {
        knf_online_fbank_new(
            opts.sample_rate as f32,
            opts.frame_length_ms as f32,
            opts.frame_shift_ms as f32,
            opts.dither as f32,
            opts.preemph as f32,
            if opts.remove_dc { 1 } else { 0 },
            win.as_ptr(),
            if opts.snip_edges { 1 } else { 0 },
            opts.num_mel_bins as c_int,
            opts.low_freq as f32,
            opts.high_freq as f32,
            0, // use_energy
            1, // use_log_fbank
            1, // use_power
        )
    };
    if f.is_null() {
        return Err(Error::Pipeline("knf_online_fbank_new failed".into()));
    }

    let result = (|| {
        unsafe {
            knf_online_fbank_accept(
                f,
                opts.sample_rate as f32,
                pcm.as_ptr(),
                pcm.len() as c_int,
            );
            knf_online_fbank_input_finished(f);
        }
        let t = unsafe { knf_online_fbank_num_frames(f) } as usize;
        let dim = unsafe { knf_online_fbank_dim(f) } as usize;
        if t == 0 {
            return Ok((vec![], 0));
        }
        let bins = opts.num_mel_bins.min(dim);
        let mut feats = vec![0.0f32; t * bins];
        let mut frame = vec![0.0f32; dim];
        for i in 0..t {
            let rc = unsafe { knf_online_fbank_get_frame(f, i as c_int, frame.as_mut_ptr()) };
            if rc != 0 {
                return Err(Error::Pipeline(format!("get_frame({i}) failed")));
            }
            feats[i * bins..(i + 1) * bins].copy_from_slice(&frame[..bins]);
        }
        if opts.subtract_mean && t > 0 {
            for j in 0..bins {
                let mut m = 0.0f64;
                for i in 0..t {
                    m += feats[i * bins + j] as f64;
                }
                m /= t as f64;
                for i in 0..t {
                    feats[i * bins + j] -= m as f32;
                }
            }
        }
        Ok((feats, t))
    })();

    unsafe { knf_online_fbank_free(f) };
    let _ = ptr::null::<()>(); // silence unused if any
    result
}

pub fn compute_fbank_default(pcm: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
    let mut opts = FbankOptions::default();
    opts.sample_rate = sample_rate;
    compute_fbank(pcm, &opts)
}
