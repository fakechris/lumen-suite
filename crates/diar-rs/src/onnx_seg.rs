//! segmentation.onnx via ONNX Runtime (Stage 2).

use std::path::{Path, PathBuf};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{Error, Result};

pub struct SegModel {
    session: Session,
    path: PathBuf,
}

impl SegModel {
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
            .map_err(|e| Error::Pipeline(format!("ort load seg: {e}")))?;
        Ok(Self { session, path })
    }

    /// waveforms mono f32 @ 16k → powerset logits row-major [T * num_classes];
    /// returns (flat_logits, T, num_classes). The DiariZen seg expects exactly
    /// 16 s (256000 samples); we pad/truncate to that.
    pub fn forward(&mut self, pcm: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        const CHUNK: usize = 16 * 16000; // 256000
        if pcm.is_empty() {
            return Ok((vec![], 0, 0));
        }
        let mut x = vec![0.0f32; CHUNK];
        let n = pcm.len().min(CHUNK);
        x[..n].copy_from_slice(&pcm[..n]);
        // input [1, 1, CHUNK]
        let input = Tensor::from_array(([1usize, 1, CHUNK], x))
            .map_err(|e| Error::Pipeline(format!("seg tensor: {e}")))?;
        let outputs = self
            .session
            .run(ort::inputs!["waveforms" => input])
            .map_err(|e| Error::Pipeline(format!("seg run: {e}")))?;
        // output "logits" [1, time, num_classes] (fall back to first output)
        let out = if let Some(v) = outputs.get("logits") {
            v
        } else if outputs.len() > 0 {
            &outputs[0]
        } else {
            return Err(Error::Pipeline("seg: no outputs".into()));
        };
        let (shape, data) = out
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Pipeline(format!("seg extract: {e}")))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() != 3 || dims[0] != 1 {
            return Err(Error::Pipeline(format!("seg unexpected shape {dims:?}")));
        }
        let t = dims[1] as usize;
        let nc = dims[2] as usize;
        let take = t * nc;
        let mut flat = vec![0.0f32; take];
        flat.copy_from_slice(&data[..take]);
        let _ = &self.path;
        Ok((flat, t, nc))
    }
}
