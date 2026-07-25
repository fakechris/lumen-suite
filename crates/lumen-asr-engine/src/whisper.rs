//! Whisper offline ASR via sherpa-onnx.
//!
//! Merged from lumen-asr and lumen-navi (identical decode path; navi added
//! `max_audio_bytes`, lumen-asr added runtime diagnostics).

use crate::model_paths::WhisperModelPaths;
use crate::{AsrEngine, AsrEngineId, AsrError, AsrRequest, AsrResult};
use async_trait::async_trait;
#[cfg(feature = "sherpa")]
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "sherpa")]
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineWhisperModelConfig};

#[derive(Debug, Clone)]
enum ModelSource {
    Dir(PathBuf),
    Files(WhisperModelPaths),
}

impl ModelSource {
    fn resolve(&self) -> Option<WhisperModelPaths> {
        match self {
            Self::Dir(dir) => WhisperModelPaths::discover(dir),
            Self::Files(paths) => paths.is_ready().then(|| paths.clone()),
        }
    }

    fn display_dir(&self) -> PathBuf {
        match self {
            Self::Dir(dir) => dir.clone(),
            Self::Files(paths) => paths
                .encoder
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| paths.encoder.clone()),
        }
    }
}

struct WhisperInner {
    source: ModelSource,
    language: String,
    max_audio_bytes: usize,
    #[cfg(feature = "sherpa")]
    recognizer: Mutex<Option<OfflineRecognizer>>,
}

pub struct WhisperAsr {
    inner: Arc<WhisperInner>,
}

impl Clone for WhisperAsr {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl WhisperAsr {
    /// Directory-based construction with lazy file discovery
    /// (drop-in for both products' `WhisperAsr::new(model_dir)`).
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self::with_source(
            ModelSource::Dir(model_dir.into()),
            "en".into(),
            8 * 1024 * 1024,
        )
    }

    /// Explicit file paths chosen by the caller.
    pub fn from_paths(paths: WhisperModelPaths) -> Self {
        Self::with_source(ModelSource::Files(paths), "en".into(), 8 * 1024 * 1024)
    }

    fn with_source(source: ModelSource, language: String, max_audio_bytes: usize) -> Self {
        Self {
            inner: Arc::new(WhisperInner {
                source,
                language,
                max_audio_bytes,
                #[cfg(feature = "sherpa")]
                recognizer: Mutex::new(None),
            }),
        }
    }

    pub fn with_language(self, language: impl Into<String>) -> Self {
        Self::with_source(
            self.inner.source.clone(),
            language.into(),
            self.inner.max_audio_bytes,
        )
    }

    pub fn with_max_audio_bytes(self, max_audio_bytes: usize) -> Self {
        Self::with_source(
            self.inner.source.clone(),
            self.inner.language.clone(),
            max_audio_bytes,
        )
    }

    pub fn model_dir(&self) -> PathBuf {
        self.inner.source.display_dir()
    }

    pub fn is_ready(&self) -> bool {
        self.inner.source.resolve().is_some()
    }
}

#[cfg(feature = "sherpa")]
impl WhisperInner {
    fn ensure_recognizer(&self) -> Result<(), AsrError> {
        let mut guard = self.recognizer.lock();
        if guard.is_some() {
            return Ok(());
        }
        let dir = self.source.display_dir();
        let paths = self.source.resolve().ok_or_else(|| {
            AsrError::NotConfigured(format!(
                "Whisper encoder/decoder/tokens not found under {}",
                dir.display()
            ))
        })?;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.whisper = OfflineWhisperModelConfig {
            encoder: Some(paths.encoder.display().to_string()),
            decoder: Some(paths.decoder.display().to_string()),
            language: Some(self.language.clone()),
            task: Some("transcribe".into()),
            tail_paddings: 0,
            enable_token_timestamps: false,
            enable_segment_timestamps: false,
        };
        config.model_config.tokens = Some(paths.tokens.display().to_string());
        config.model_config.num_threads = 2;
        config.model_config.provider = Some("cpu".into());

        tracing::info!(encoder = %paths.encoder.display(), "creating Whisper OfflineRecognizer");
        let rec = OfflineRecognizer::create(&config).ok_or_else(|| {
            AsrError::Inference(format!(
                "failed to create Whisper recognizer (check model paths under {})",
                dir.display()
            ))
        })?;
        *guard = Some(rec);
        Ok(())
    }

    fn decode_sync(&self, samples: &[f32], sample_rate: u32) -> Result<String, AsrError> {
        self.ensure_recognizer()?;
        let guard = self.recognizer.lock();
        let recognizer = guard
            .as_ref()
            .ok_or_else(|| AsrError::NotConfigured("whisper recognizer missing".into()))?;

        let stream = recognizer.create_stream();
        stream.accept_waveform(sample_rate as i32, samples);
        recognizer.decode(&stream);
        let text = stream
            .get_result()
            .map(|r| r.text)
            .unwrap_or_default()
            .trim()
            .to_string();
        Ok(text)
    }
}

#[async_trait]
impl AsrEngine for WhisperAsr {
    fn id(&self) -> AsrEngineId {
        AsrEngineId::Whisper
    }

    fn is_supported(&self) -> bool {
        cfg!(feature = "sherpa") && self.is_ready()
    }

    fn max_audio_bytes(&self) -> Option<usize> {
        Some(self.inner.max_audio_bytes)
    }

    async fn transcribe(&self, req: AsrRequest) -> Result<AsrResult, AsrError> {
        if req.samples.is_empty() {
            return Err(AsrError::EmptyAudio);
        }

        #[cfg(not(feature = "sherpa"))]
        {
            let _ = req;
            Err(AsrError::Unsupported(
                "build with feature `sherpa` for Whisper".into(),
            ))
        }

        #[cfg(feature = "sherpa")]
        {
            let inner = Arc::clone(&self.inner);
            let samples = req.samples;
            let sr = req.sample_rate;
            let text = tokio::task::spawn_blocking(move || inner.decode_sync(&samples, sr))
                .await
                .map_err(|e| AsrError::Inference(e.to_string()))??;
            let (model, model_revision) = crate::model_identity_from_path(&self.model_dir());

            let mut result = AsrResult::new(text, AsrEngineId::Whisper);
            result.engine_label = "whisper".into();
            result.language = Some(self.inner.language.clone());
            result.diagnostics.model = model;
            result.diagnostics.model_revision = model_revision;
            Ok(result)
        }
    }
}
