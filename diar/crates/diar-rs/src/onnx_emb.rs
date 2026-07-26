//! embedding.onnx via ONNX Runtime.
//!
//! WeSpeaker ResNet34-LM contract: `feats [1, T, 80] → embs [1, 256]`.

use std::path::{Path, PathBuf};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{Error, Result};

pub struct EmbModel {
    session: Session,
    path: PathBuf,
}

impl EmbModel {
    pub fn load(path: impl AsRef<Path>, threads: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.is_file() {
            return Err(Error::MissingModel(path));
        }
        let threads = threads.max(1);
        let session = Session::builder()
            .map_err(|e| Error::Pipeline(format!("ort builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Pipeline(format!("ort opt: {e}")))?
            .with_intra_threads(threads)
            .map_err(|e| Error::Pipeline(format!("ort intra: {e}")))?
            .with_inter_threads(1)
            .map_err(|e| Error::Pipeline(format!("ort inter: {e}")))?
            .commit_from_file(&path)
            .map_err(|e| Error::Pipeline(format!("ort load emb: {e}")))?;
        Ok(Self { session, path })
    }

    /// fbank row-major [T*80] → 256-d embedding (f64).
    pub fn embed_fbank(&mut self, fbank: &[f32], t: usize) -> Result<Vec<f64>> {
        if t == 0 || fbank.len() != t * 80 {
            return Err(Error::Pipeline(format!(
                "emb fbank len {} != T*80 (T={t})",
                fbank.len()
            )));
        }
        let fbank_t = Tensor::from_array(([1usize, t, 80], fbank.to_vec()))
            .map_err(|e| Error::Pipeline(format!("emb fbank tensor: {e}")))?;
        let outputs = self
            .session
            .run(ort::inputs!["feats" => fbank_t])
            .map_err(|e| Error::Pipeline(format!("emb run: {e}")))?;
        let out = if let Some(v) = outputs.get("embs") {
            v
        } else if outputs.len() > 0 {
            &outputs[0]
        } else {
            return Err(Error::Pipeline("emb: no outputs".into()));
        };
        let (shape, data) = out
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Pipeline(format!("emb extract: {e}")))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if data.len() < 256 {
            return Err(Error::Pipeline(format!(
                "emb unexpected shape {dims:?} len={}",
                data.len()
            )));
        }
        let emb: Vec<f64> = data.iter().take(256).map(|&v| v as f64).collect();
        let _ = &self.path;
        Ok(emb)
    }
}
