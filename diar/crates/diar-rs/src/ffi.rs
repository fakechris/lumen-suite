//! Minimal C ABI for embedding (feature `ffi`).
//!
//! ```c
//! // Returns 0 on success. Writes JSON diarization to out_json_path.
//! int diar_diarize(
//!     const char *wav_path,
//!     const char *seg_onnx,
//!     const char *emb_onnx,
//!     const char *plda_dir,
//!     const char *out_json_path,
//!     int threads
//! );
//! ```

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;

use crate::config::{DiarizeConfig, ModelPaths};
use crate::io::write_diarization_json;
use crate::pipeline::diarize;

/// Diarize wav → write `diarization.json` to `out_json_path`.
///
/// Returns 0 on success, negative on error.
#[no_mangle]
pub unsafe extern "C" fn diar_diarize(
    wav_path: *const c_char,
    seg_onnx: *const c_char,
    emb_onnx: *const c_char,
    plda_dir: *const c_char,
    out_json_path: *const c_char,
    threads: c_int,
) -> c_int {
    if wav_path.is_null()
        || seg_onnx.is_null()
        || emb_onnx.is_null()
        || plda_dir.is_null()
        || out_json_path.is_null()
    {
        return -1;
    }
    let wav = match cstr_path(wav_path) {
        Ok(p) => p,
        Err(_) => return -2,
    };
    let models = ModelPaths {
        segmentation: match cstr_path(seg_onnx) {
            Ok(p) => p,
            Err(_) => return -2,
        },
        embedding: match cstr_path(emb_onnx) {
            Ok(p) => p,
            Err(_) => return -2,
        },
        plda_dir: match cstr_path(plda_dir) {
            Ok(p) => p,
            Err(_) => return -2,
        },
    };
    let out = match cstr_path(out_json_path) {
        Ok(p) => p,
        Err(_) => return -2,
    };
    let mut cfg = DiarizeConfig::default();
    if threads > 0 {
        cfg.threads = threads as usize;
    }
    match diarize(&wav, &models, &cfg) {
        Ok(result) => {
            if let Some(parent) = out.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match write_diarization_json(&result, &out) {
                Ok(()) => 0,
                Err(_) => -4,
            }
        }
        Err(_) => -3,
    }
}

unsafe fn cstr_path(p: *const c_char) -> Result<PathBuf, ()> {
    let s = CStr::from_ptr(p).to_str().map_err(|_| ())?;
    Ok(PathBuf::from(s))
}
