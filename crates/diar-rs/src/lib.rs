//! # diar-rs
//!
//! Open-source speaker diarization library. Runs on open ONNX weights
//! (WeSpeaker ResNet34-LM embedding + DiariZen WavLM segmentation + BUT VBx
//! clustering) and is evaluated against human-annotated meeting ground truth.
//!
//! ## Goals
//! - **Weights**: open-source only (see `models/README.md`).
//! - **Pipeline**: segmentation → embedding → AHC/VBx clustering → merge.
//! - **Success metric**: frame_acc / DER vs human GT (not binary parity).
//!
//! See `docs/PROBLEM.md` for the problem definition.

pub mod audio;
pub mod cluster;
pub mod config;
pub mod error;
pub mod fbank;
pub mod io;
pub mod merge;
pub mod onnx_emb;
pub mod onnx_seg;
pub mod pipeline;
pub mod plda;
pub mod powerset;
pub mod vbx;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use config::{DiarizeConfig, ModelPaths};
pub use error::{Error, Result};
pub use pipeline::{
    diarize, diarize_ex, diarize_with_trace, validate_models, DiarizeResult, DumpOpts, Trace, Turn,
};
pub use plda::Plda;
