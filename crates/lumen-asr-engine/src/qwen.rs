//! Qwen3-ASR via sherpa-onnx (offline recognizer, in-process — the former
//! MLX Python worker is gone).
//!
//! The recognizer wraps sherpa's `OfflineQwen3ASRModelConfig`
//! (conv_frontend + encoder + decoder + tokenizer dir) and is created lazily
//! on the first request, then reused — mirroring [`crate::SenseVoiceSherpaAsr`].

use crate::model_paths::QwenModelPaths;
use crate::{AsrEngine, AsrEngineId, AsrError, AsrRequest, AsrResult};
use async_trait::async_trait;
#[cfg(feature = "sherpa")]
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "sherpa")]
use sherpa_onnx::{OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig};

#[derive(Debug, Clone)]
pub struct QwenAsrConfig {
    /// sherpa-onnx Qwen3-ASR model directory (resolved by the caller /
    /// lumen-models). Layout: `conv_frontend.onnx`, `encoder.int8.onnx`,
    /// `decoder.int8.onnx`, `tokenizer/{vocab.json,merges.txt,…}`.
    pub model_dir: PathBuf,
    /// Reported back as [`AsrResult::language`]. sherpa Qwen3-ASR has no
    /// language setting (the model auto-detects); this is informational.
    pub language: Option<String>,
    /// Bounds one decode call. On expiry the request errors out while the
    /// in-flight sherpa decode runs to completion in the background (ONNX
    /// Runtime calls are not cancellable).
    pub timeout: Duration,
}

impl QwenAsrConfig {
    pub fn product(
        model_dir: impl Into<PathBuf>,
        language: Option<String>,
        timeout: Duration,
    ) -> Self {
        Self {
            model_dir: model_dir.into(),
            language,
            timeout,
        }
    }
}

struct QwenInner {
    config: QwenAsrConfig,
    #[cfg(feature = "sherpa")]
    recognizer: Mutex<Option<OfflineRecognizer>>,
}

pub struct QwenAsr {
    inner: Arc<QwenInner>,
}

impl Clone for QwenAsr {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl QwenAsr {
    pub fn new(config: QwenAsrConfig) -> Self {
        Self {
            inner: Arc::new(QwenInner {
                config,
                #[cfg(feature = "sherpa")]
                recognizer: Mutex::new(None),
            }),
        }
    }

    pub fn model_dir(&self) -> &std::path::Path {
        &self.inner.config.model_dir
    }

    pub fn is_ready(&self) -> bool {
        qwen_model_paths(&self.inner.config.model_dir).is_some()
    }

    /// Release the loaded model when the user switches to another ASR engine.
    ///
    /// If a request is in flight it finishes normally; the next request
    /// reloads the recognizer lazily. Returns false when no model was loaded.
    pub fn unload(&self) -> bool {
        #[cfg(feature = "sherpa")]
        {
            self.inner.recognizer.lock().take().is_some()
        }
        #[cfg(not(feature = "sherpa"))]
        {
            false
        }
    }
}

fn qwen_model_paths(model_dir: &std::path::Path) -> Option<QwenModelPaths> {
    QwenModelPaths::discover(model_dir)
}

#[cfg(feature = "sherpa")]
impl QwenInner {
    /// Ensure the recognizer exists; returns true when it was already warm.
    fn ensure_recognizer(&self) -> Result<bool, AsrError> {
        let mut guard = self.recognizer.lock();
        if guard.is_some() {
            return Ok(true);
        }
        let dir = &self.config.model_dir;
        let paths = qwen_model_paths(dir).ok_or_else(|| {
            AsrError::NotConfigured(format!(
                "Qwen3-ASR sherpa model not found under {} (need conv_frontend.onnx, \
                 encoder.int8.onnx, decoder.int8.onnx, tokenizer/)",
                dir.display()
            ))
        })?;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.qwen3_asr = OfflineQwen3ASRModelConfig {
            conv_frontend: Some(paths.conv_frontend.display().to_string()),
            encoder: Some(paths.encoder.display().to_string()),
            decoder: Some(paths.decoder.display().to_string()),
            tokenizer: Some(paths.tokenizer_dir.display().to_string()),
            ..OfflineQwen3ASRModelConfig::default()
        };
        config.model_config.num_threads = 2;
        config.model_config.provider = Some("cpu".into());

        tracing::info!(model = %dir.display(), "creating Qwen3-ASR OfflineRecognizer");
        let rec = OfflineRecognizer::create(&config).ok_or_else(|| {
            AsrError::Inference(format!(
                "failed to create Qwen3-ASR recognizer (check model files under {})",
                dir.display()
            ))
        })?;
        *guard = Some(rec);
        Ok(false)
    }

    fn decode_sync(&self, samples: &[f32], sample_rate: u32) -> Result<(String, bool), AsrError> {
        let warm = self.ensure_recognizer()?;
        let guard = self.recognizer.lock();
        let recognizer = guard
            .as_ref()
            .ok_or_else(|| AsrError::NotConfigured("recognizer missing".into()))?;

        let stream = recognizer.create_stream();
        stream.accept_waveform(sample_rate as i32, samples);
        recognizer.decode(&stream);
        let text = stream
            .get_result()
            .map(|r| r.text)
            .unwrap_or_default()
            .trim()
            .to_string();
        Ok((text, warm))
    }
}

#[async_trait]
impl AsrEngine for QwenAsr {
    fn id(&self) -> AsrEngineId {
        AsrEngineId::Qwen3Asr
    }

    fn is_supported(&self) -> bool {
        cfg!(feature = "sherpa") && self.is_ready()
    }

    async fn transcribe(&self, req: AsrRequest) -> Result<AsrResult, AsrError> {
        if req.samples.is_empty() {
            return Err(AsrError::EmptyAudio);
        }

        #[cfg(not(feature = "sherpa"))]
        {
            let _ = req;
            Err(AsrError::Unsupported(
                "build with feature `sherpa` for Qwen3-ASR".into(),
            ))
        }

        #[cfg(feature = "sherpa")]
        {
            let inner = Arc::clone(&self.inner);
            let samples = req.samples;
            let sr = req.sample_rate;
            let decode = tokio::task::spawn_blocking(move || inner.decode_sync(&samples, sr));
            let (text, warm) = match tokio::time::timeout(self.inner.config.timeout, decode).await {
                Ok(joined) => joined.map_err(|e| AsrError::Inference(e.to_string()))??,
                Err(_) => {
                    return Err(AsrError::Inference(format!(
                        "Qwen3-ASR decode timed out after {}s",
                        self.inner.config.timeout.as_secs()
                    )))
                }
            };
            let (model, model_revision) =
                crate::model_identity_from_path(&self.inner.config.model_dir);

            let mut result = AsrResult::new(text, AsrEngineId::Qwen3Asr);
            result.language = self.inner.config.language.clone();
            result.diagnostics.worker_reused = Some(warm);
            result.diagnostics.model = model;
            result.diagnostics.model_revision = model_revision;
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(model_dir: impl Into<PathBuf>) -> QwenAsrConfig {
        QwenAsrConfig::product(model_dir, Some("zh".into()), Duration::from_secs(60))
    }

    #[test]
    fn not_ready_when_model_files_missing() {
        let engine = QwenAsr::new(config("/nonexistent/qwen3-sherpa"));
        assert!(!engine.is_ready());
        assert!(!engine.is_supported());
        assert_eq!(
            engine.model_dir(),
            std::path::Path::new("/nonexistent/qwen3-sherpa")
        );
    }

    #[tokio::test]
    async fn empty_audio_is_rejected_before_touching_the_model() {
        let engine = QwenAsr::new(config("/nonexistent/qwen3-sherpa"));
        let err = engine
            .transcribe(AsrRequest::new(Vec::new(), 16_000))
            .await
            .unwrap_err();
        assert!(matches!(err, AsrError::EmptyAudio));
    }

    #[cfg(feature = "sherpa")]
    #[tokio::test]
    async fn missing_model_dir_is_a_config_error() {
        let engine = QwenAsr::new(config("/nonexistent/qwen3-sherpa"));
        let err = engine
            .transcribe(AsrRequest::new(vec![0.0f32; 1_600], 16_000))
            .await
            .unwrap_err();
        assert!(matches!(err, AsrError::NotConfigured(_)), "{err}");
    }

    #[cfg(feature = "sherpa")]
    #[test]
    fn unload_without_loaded_model_returns_false() {
        let engine = QwenAsr::new(config("/nonexistent/qwen3-sherpa"));
        assert!(!engine.unload());
    }
}
