//! SenseVoice offline ASR via sherpa-onnx.
//!
//! Merged from lumen-asr and lumen-navi (identical decode path; navi added
//! `max_audio_bytes`, lumen-asr added runtime diagnostics).

use crate::model_paths::SenseVoiceModelPaths;
use crate::{AsrEngine, AsrEngineId, AsrError, AsrRequest, AsrResult};
use async_trait::async_trait;
#[cfg(feature = "sherpa")]
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "sherpa")]
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig};

#[derive(Debug, Clone)]
enum ModelSource {
    /// Lazy discovery inside a directory (legacy `new(model_dir)` behavior).
    Dir(PathBuf),
    /// Explicit files chosen by the caller (e.g. lumen-models).
    Files(SenseVoiceModelPaths),
}

impl ModelSource {
    fn resolve(&self) -> Option<SenseVoiceModelPaths> {
        match self {
            Self::Dir(dir) => SenseVoiceModelPaths::discover(dir),
            Self::Files(paths) => paths.is_ready().then(|| paths.clone()),
        }
    }

    fn display_dir(&self) -> PathBuf {
        match self {
            Self::Dir(dir) => dir.clone(),
            Self::Files(paths) => paths
                .model
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| paths.model.clone()),
        }
    }
}

struct SenseVoiceInner {
    source: ModelSource,
    language: String,
    max_audio_bytes: usize,
    #[cfg(feature = "sherpa")]
    recognizer: Mutex<Option<OfflineRecognizer>>,
}

pub struct SenseVoiceSherpaAsr {
    inner: Arc<SenseVoiceInner>,
}

impl Clone for SenseVoiceSherpaAsr {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl SenseVoiceSherpaAsr {
    /// Directory-based construction with lazy file discovery
    /// (drop-in for both products' `SenseVoiceSherpaAsr::new(model_dir)`).
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self::with_source(
            ModelSource::Dir(model_dir.into()),
            "auto".into(),
            8 * 1024 * 1024,
        )
    }

    /// Explicit file paths chosen by the caller.
    pub fn from_paths(paths: SenseVoiceModelPaths) -> Self {
        Self::with_source(ModelSource::Files(paths), "auto".into(), 8 * 1024 * 1024)
    }

    fn with_source(source: ModelSource, language: String, max_audio_bytes: usize) -> Self {
        Self {
            inner: Arc::new(SenseVoiceInner {
                source,
                language,
                max_audio_bytes,
                #[cfg(feature = "sherpa")]
                recognizer: Mutex::new(None),
            }),
        }
    }

    pub fn with_language(self, language: impl Into<String>) -> Self {
        // Rebuild instead of mutating so the old Arc keeps its warm recognizer.
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

    /// The configured model directory (parent of the model file for explicit paths).
    pub fn model_dir(&self) -> PathBuf {
        self.inner.source.display_dir()
    }

    pub fn is_ready(&self) -> bool {
        self.inner.source.resolve().is_some()
    }
}

#[cfg(feature = "sherpa")]
impl SenseVoiceInner {
    fn ensure_recognizer(&self) -> Result<(), AsrError> {
        let mut guard = self.recognizer.lock();
        if guard.is_some() {
            return Ok(());
        }
        let dir = self.source.display_dir();
        let paths = self.source.resolve().ok_or_else(|| {
            AsrError::NotConfigured(format!(
                "SenseVoice model/tokens not found under {}",
                dir.display()
            ))
        })?;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
            model: Some(paths.model.display().to_string()),
            language: Some(self.language.clone()),
            use_itn: true,
        };
        config.model_config.tokens = Some(paths.tokens.display().to_string());
        config.model_config.num_threads = 2;
        config.model_config.provider = Some("cpu".into());

        tracing::info!(model = %paths.model.display(), "creating SenseVoice OfflineRecognizer");
        let rec = OfflineRecognizer::create(&config).ok_or_else(|| {
            AsrError::Inference(format!(
                "failed to create SenseVoice recognizer (check model paths under {})",
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
        Ok(cleanup_sensevoice_text(&text))
    }
}

#[async_trait]
impl AsrEngine for SenseVoiceSherpaAsr {
    fn id(&self) -> AsrEngineId {
        AsrEngineId::SenseVoiceSherpa
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
                "build with feature `sherpa` for SenseVoice".into(),
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

            let mut result = AsrResult::new(text, AsrEngineId::SenseVoiceSherpa);
            result.engine_label = "sensevoice".into();
            result.language = Some(self.inner.language.clone());
            result.diagnostics.model = model;
            result.diagnostics.model_revision = model_revision;
            Ok(result)
        }
    }
}

// Only reachable from the sherpa decode path at runtime, but kept unconditional
// so the unit test covers every feature combination.
#[cfg_attr(not(feature = "sherpa"), allow(dead_code))]
fn cleanup_sensevoice_text(text: &str) -> String {
    let mut s = text.to_string();
    for tag in [
        "<|zh|>",
        "<|en|>",
        "<|yue|>",
        "<|ja|>",
        "<|ko|>",
        "<|nospeech|>",
        "<|EMO_UNKNOWN|>",
        "<|Event_UNK|>",
        "<|woitn|>",
        "<|withitn|>",
    ] {
        s = s.replace(tag, "");
    }
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    // From both products.
    #[test]
    fn cleanup_tags() {
        assert_eq!(cleanup_sensevoice_text("<|zh|>你好"), "你好");
    }

    #[test]
    fn explicit_paths_not_ready_when_files_missing() {
        let eng = SenseVoiceSherpaAsr::from_paths(SenseVoiceModelPaths {
            model: "/nonexistent/model.onnx".into(),
            tokens: "/nonexistent/tokens.txt".into(),
        });
        assert!(!eng.is_ready());
        assert_eq!(eng.model_dir(), PathBuf::from("/nonexistent"));
    }
}
