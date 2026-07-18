//! Default diarization parameters (open-source profile; tune via DiarizeConfig).

use serde::{Deserialize, Serialize};

/// Open-source diarization profile. Defaults below carry over from the
/// initial port; `num_classes` / `rf_shift` are reconciled to the open
/// segmentation model after ONNX introspection (Stage 5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizeConfig {
    pub sample_rate: u32,
    pub num_speakers: usize,
    pub num_classes: usize,
    pub rf_shift: u32,

    pub seg_chunk_sec: f64,
    pub seg_hop_sec: f64,

    pub onset: f64,
    pub offset: f64,
    pub min_seg_sec: f64,
    pub min_active_sec: f64,

    pub xvec_win_sec: f64,
    pub xvec_hop_sec: f64,

    pub ahc_max_speakers: usize,
    pub ahc_min_cluster: usize,

    /// VBx grid: (Fa, Fb, loop_prob). First non-collapse wins in Python reimpl.
    pub vbx_grid: Vec<(f64, f64, f64)>,
    pub vbx_max_iters: usize,

    pub excl_median_sec: f64,
    pub merge_gap_sec: f64,

    /// ORT / BLAS style thread hint (Stage 2+).
    pub threads: usize,
}

impl Default for DiarizeConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            num_speakers: 4,
            num_classes: 11,
            rf_shift: 320,
            seg_chunk_sec: 10.0,
            seg_hop_sec: 8.0,
            onset: 0.5,
            offset: 0.5,
            min_seg_sec: 3200.0 / 16_000.0,
            min_active_sec: 0.20,
            xvec_win_sec: 1.5,
            xvec_hop_sec: 0.75,
            ahc_max_speakers: 6,
            ahc_min_cluster: 8,
            vbx_grid: vec![
                (0.3, 17.0, 0.99),
                (0.2, 12.0, 0.95),
                (0.4, 10.0, 0.90),
                (0.15, 8.0, 0.99),
                (0.5, 20.0, 0.99),
            ],
            vbx_max_iters: 20,
            excl_median_sec: 0.12,
            merge_gap_sec: 3.0,
            threads: 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub segmentation: std::path::PathBuf,
    pub embedding: std::path::PathBuf,
    pub plda_dir: std::path::PathBuf,
}

impl ModelPaths {
    /// Resolve open-weight paths: env vars first, then `<root>/models/`.
    ///
    /// Env: `DIAR_SEG_ONNX`, `DIAR_EMB_ONNX`, `DIAR_PLDA_DIR`.
    /// Defaults: `<root>/models/{seg.onnx, emb.onnx, plda/}`.
    pub fn resolve(root: impl AsRef<std::path::Path>) -> Self {
        let models = root.as_ref().join("models");
        Self {
            segmentation: std::env::var("DIAR_SEG_ONNX")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| models.join("seg.onnx")),
            embedding: std::env::var("DIAR_EMB_ONNX")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| models.join("emb.onnx")),
            plda_dir: std::env::var("DIAR_PLDA_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| models.join("plda")),
        }
    }
}
