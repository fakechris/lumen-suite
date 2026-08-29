//! Runtime diagnostics attached to [`crate::AsrResult`].
//!
//! Ported verbatim from `lumen-core` (lumen-asr repo) so this crate has zero
//! coupling to product crates. lumen-asr / lumen-navi should re-export these
//! from here once they migrate.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AsrRuntimeDiagnostics {
    /// `Some(false)` means the request paid worker/model cold-start cost.
    /// Engines without a persistent worker leave this unknown; sherpa-onnx
    /// engines report whether the recognizer was already warm.
    pub worker_reused: Option<bool>,
    /// Stable model name without exposing the absolute local filesystem path.
    pub model: Option<String>,
    /// Immutable model revision when the runtime path exposes one.
    pub model_revision: Option<String>,
}
