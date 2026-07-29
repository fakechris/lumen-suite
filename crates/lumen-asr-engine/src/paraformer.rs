//! Offline Paraformer ASR via sherpa-onnx.
//!
//! Paraformer is the meeting workhorse (see `docs/MEETING.md` Stage M6): it
//! emits **word/token-level timestamps** natively and supports **hotword /
//! contextual biasing**. This engine wraps sherpa-onnx's
//! `OfflineParaformerModelConfig` (a single `model.onnx` + shared
//! `tokens.txt`) and maps the decoded token timestamps into
//! [`WordTiming`]s on the [`AsrResult`].
//!
//! ## Hotwords
//! sherpa-onnx contextual biasing only takes effect under
//! `decoding_method = "modified_beam_search"`, so the recognizer is built with
//! that method (a deliberate default for the meeting use-case; slightly slower
//! than greedy but enables per-request hotwords). Per-request
//! [`AsrRequest::hotwords`] are injected with
//! `OfflineRecognizer::create_stream_with_hotwords`; when a request has no
//! hotwords a plain stream is used. Global `hotwords_score` and `modeling_unit`
//! are configurable via the builders.

use crate::model_paths::ParaformerOfflineModelPaths;
use crate::{AsrEngine, AsrEngineId, AsrError, AsrRequest, AsrResult, WordTiming};
use async_trait::async_trait;
#[cfg(feature = "sherpa")]
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "sherpa")]
use sherpa_onnx::{OfflineParaformerModelConfig, OfflineRecognizer, OfflineRecognizerConfig};

/// Default contextual-biasing boost applied to hotword tokens.
const DEFAULT_HOTWORDS_SCORE: f32 = 2.0;
/// Chinese Paraformer models are character-modelled; hotwords bias per CJK char.
const DEFAULT_MODELING_UNIT: &str = "cjkchar";

#[derive(Debug, Clone)]
enum ModelSource {
    /// Lazy discovery inside a directory.
    Dir(PathBuf),
    /// Explicit files chosen by the caller (e.g. lumen-models).
    Files(ParaformerOfflineModelPaths),
}

impl ModelSource {
    fn resolve(&self) -> Option<ParaformerOfflineModelPaths> {
        match self {
            Self::Dir(dir) => ParaformerOfflineModelPaths::discover(dir),
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

struct ParaformerInner {
    source: ModelSource,
    /// Result label only — Paraformer offline has no language config field.
    language: String,
    hotwords_score: f32,
    modeling_unit: String,
    max_audio_bytes: usize,
    #[cfg(feature = "sherpa")]
    recognizer: Mutex<Option<OfflineRecognizer>>,
}

/// Offline Paraformer engine (word timestamps + hotwords).
pub struct ParaformerAsr {
    inner: Arc<ParaformerInner>,
}

impl Clone for ParaformerAsr {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl ParaformerAsr {
    /// Directory-based construction with lazy file discovery.
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self::with_source(ModelSource::Dir(model_dir.into()))
    }

    /// Explicit file paths chosen by the caller.
    pub fn from_paths(paths: ParaformerOfflineModelPaths) -> Self {
        Self::with_source(ModelSource::Files(paths))
    }

    fn with_source(source: ModelSource) -> Self {
        Self {
            inner: Arc::new(ParaformerInner {
                source,
                language: "zh".into(),
                hotwords_score: DEFAULT_HOTWORDS_SCORE,
                modeling_unit: DEFAULT_MODELING_UNIT.into(),
                max_audio_bytes: 8 * 1024 * 1024,
                #[cfg(feature = "sherpa")]
                recognizer: Mutex::new(None),
            }),
        }
    }

    // Rebuild instead of mutating so an existing Arc keeps its warm recognizer.
    fn rebuilt(&self, f: impl FnOnce(&mut ParaformerInnerSettings)) -> Self {
        let mut s = ParaformerInnerSettings {
            language: self.inner.language.clone(),
            hotwords_score: self.inner.hotwords_score,
            modeling_unit: self.inner.modeling_unit.clone(),
            max_audio_bytes: self.inner.max_audio_bytes,
        };
        f(&mut s);
        Self {
            inner: Arc::new(ParaformerInner {
                source: self.inner.source.clone(),
                language: s.language,
                hotwords_score: s.hotwords_score,
                modeling_unit: s.modeling_unit,
                max_audio_bytes: s.max_audio_bytes,
                #[cfg(feature = "sherpa")]
                recognizer: Mutex::new(None),
            }),
        }
    }

    /// Result-language label (Paraformer offline has no language config knob).
    pub fn with_language(self, language: impl Into<String>) -> Self {
        let language = language.into();
        self.rebuilt(|s| s.language = language)
    }

    /// Contextual-biasing boost for hotword tokens (default `2.0`).
    pub fn with_hotwords_score(self, score: f32) -> Self {
        self.rebuilt(|s| s.hotwords_score = score)
    }

    /// Hotword modeling unit (default `cjkchar`; use `bpe`/`cjkchar+bpe` for
    /// BPE Paraformer variants).
    pub fn with_modeling_unit(self, unit: impl Into<String>) -> Self {
        let unit = unit.into();
        self.rebuilt(|s| s.modeling_unit = unit)
    }

    pub fn with_max_audio_bytes(self, max_audio_bytes: usize) -> Self {
        self.rebuilt(|s| s.max_audio_bytes = max_audio_bytes)
    }

    /// The configured model directory (parent of the model file for explicit paths).
    pub fn model_dir(&self) -> PathBuf {
        self.inner.source.display_dir()
    }

    pub fn is_ready(&self) -> bool {
        self.inner.source.resolve().is_some()
    }
}

struct ParaformerInnerSettings {
    language: String,
    hotwords_score: f32,
    modeling_unit: String,
    max_audio_bytes: usize,
}

/// Map sherpa token timestamps into [`WordTiming`]s.
///
/// `timestamps[i]` is the start of `tokens[i]` (seconds). The end is
/// `timestamps[i] + durations[i]` when durations are present, else the next
/// token's start, else the token's own start (last token without duration).
/// Empty/blank tokens are dropped. Returns empty when timestamps are absent or
/// their length disagrees with the token count.
#[cfg_attr(not(feature = "sherpa"), allow(dead_code))]
fn map_words(
    tokens: &[String],
    timestamps: Option<&[f32]>,
    durations: Option<&[f32]>,
) -> Vec<WordTiming> {
    let Some(ts) = timestamps else {
        return Vec::new();
    };
    if ts.len() != tokens.len() || tokens.is_empty() {
        return Vec::new();
    }
    let durations = durations.filter(|d| d.len() == tokens.len());

    let mut words = Vec::with_capacity(tokens.len());
    for (i, tok) in tokens.iter().enumerate() {
        let word = tok.trim();
        if word.is_empty() {
            continue;
        }
        let start = ts[i] as f64;
        let end = if let Some(d) = durations {
            (ts[i] + d[i].max(0.0)) as f64
        } else if i + 1 < ts.len() {
            (ts[i + 1] as f64).max(start)
        } else {
            start
        };
        words.push(WordTiming {
            word: word.to_string(),
            start,
            end,
        });
    }
    words
}

#[cfg(feature = "sherpa")]
impl ParaformerInner {
    fn ensure_recognizer(&self) -> Result<(), AsrError> {
        let mut guard = self.recognizer.lock();
        if guard.is_some() {
            return Ok(());
        }
        let dir = self.source.display_dir();
        let paths = self.source.resolve().ok_or_else(|| {
            AsrError::NotConfigured(format!(
                "Paraformer model/tokens not found under {}",
                dir.display()
            ))
        })?;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.paraformer = OfflineParaformerModelConfig {
            model: Some(paths.model.display().to_string()),
        };
        config.model_config.tokens = Some(paths.tokens.display().to_string());
        config.model_config.num_threads = 2;
        config.model_config.provider = Some("cpu".into());
        // Contextual biasing (hotwords) requires modified_beam_search.
        config.model_config.modeling_unit = Some(self.modeling_unit.clone());
        config.decoding_method = Some("modified_beam_search".into());
        config.hotwords_score = self.hotwords_score;

        tracing::info!(model = %paths.model.display(), "creating Paraformer OfflineRecognizer");
        let rec = OfflineRecognizer::create(&config).ok_or_else(|| {
            AsrError::Inference(format!(
                "failed to create Paraformer recognizer (check model paths under {})",
                dir.display()
            ))
        })?;
        *guard = Some(rec);
        Ok(())
    }

    fn decode_sync(
        &self,
        samples: &[f32],
        sample_rate: u32,
        hotwords: &[String],
    ) -> Result<(String, Vec<WordTiming>), AsrError> {
        self.ensure_recognizer()?;
        let guard = self.recognizer.lock();
        let recognizer = guard
            .as_ref()
            .ok_or_else(|| AsrError::NotConfigured("paraformer recognizer missing".into()))?;

        // Per-request hotwords: one phrase per line (sherpa contract).
        let joined: String = hotwords
            .iter()
            .map(|h| h.trim())
            .filter(|h| !h.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        let stream = if joined.is_empty() {
            recognizer.create_stream()
        } else {
            recognizer.create_stream_with_hotwords(&joined)
        };

        stream.accept_waveform(sample_rate as i32, samples);
        recognizer.decode(&stream);

        let result = stream.get_result();
        let (text, words) = match result {
            Some(r) => {
                let words = map_words(&r.tokens, r.timestamps.as_deref(), r.durations.as_deref());
                (r.text.trim().to_string(), words)
            }
            None => (String::new(), Vec::new()),
        };
        Ok((text, words))
    }
}

#[async_trait]
impl AsrEngine for ParaformerAsr {
    fn id(&self) -> AsrEngineId {
        AsrEngineId::Paraformer
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
                "build with feature `sherpa` for Paraformer".into(),
            ))
        }

        #[cfg(feature = "sherpa")]
        {
            let inner = Arc::clone(&self.inner);
            let samples = req.samples;
            let sr = req.sample_rate;
            let hotwords = req.hotwords;
            let (text, words) =
                tokio::task::spawn_blocking(move || inner.decode_sync(&samples, sr, &hotwords))
                    .await
                    .map_err(|e| AsrError::Inference(e.to_string()))??;
            let (model, model_revision) = crate::model_identity_from_path(&self.model_dir());

            let mut result = AsrResult::new(text, AsrEngineId::Paraformer);
            result.engine_label = "paraformer".into();
            result.language = Some(self.inner.language.clone());
            result.words = words;
            result.diagnostics.model = model;
            result.diagnostics.model_revision = model_revision;
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn map_words_uses_next_start_as_end_without_durations() {
        let tokens = vec!["你".to_string(), "好".to_string(), "吗".to_string()];
        let ts = [0.0f32, 0.5, 1.2];
        let words = map_words(&tokens, Some(&ts), None);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].word, "你");
        assert!(close(words[0].start, 0.0));
        assert!(close(words[0].end, 0.5));
        assert!(close(words[1].end, 1.2));
        // Last token without duration: end falls back to its own start.
        assert!(close(words[2].start, 1.2));
        assert!(close(words[2].end, 1.2));
    }

    #[test]
    fn map_words_prefers_durations_when_present() {
        let tokens = vec!["a".to_string(), "b".to_string()];
        let ts = [0.0f32, 1.0];
        let dur = [0.4f32, 0.3];
        let words = map_words(&tokens, Some(&ts), Some(&dur));
        assert!(close(words[0].end, 0.4));
        assert!(close(words[1].start, 1.0));
        assert!(close(words[1].end, 1.3));
    }

    #[test]
    fn map_words_empty_when_no_timestamps_or_length_mismatch() {
        let tokens = vec!["a".to_string(), "b".to_string()];
        assert!(map_words(&tokens, None, None).is_empty());
        // length mismatch → drop rather than mis-align.
        assert!(map_words(&tokens, Some(&[0.0]), None).is_empty());
    }

    #[test]
    fn map_words_drops_blank_tokens() {
        let tokens = vec!["a".to_string(), " ".to_string(), "b".to_string()];
        let ts = [0.0f32, 0.5, 1.0];
        let words = map_words(&tokens, Some(&ts), None);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "a");
        assert_eq!(words[1].word, "b");
    }

    #[test]
    fn explicit_paths_not_ready_when_files_missing() {
        let eng = ParaformerAsr::from_paths(ParaformerOfflineModelPaths {
            model: "/nonexistent/model.onnx".into(),
            tokens: "/nonexistent/tokens.txt".into(),
        });
        assert!(!eng.is_ready());
        assert_eq!(eng.model_dir(), PathBuf::from("/nonexistent"));
    }

    #[test]
    fn builders_preserve_settings() {
        let eng = ParaformerAsr::new("/models/paraformer/offline")
            .with_language("en")
            .with_hotwords_score(3.5)
            .with_modeling_unit("bpe")
            .with_max_audio_bytes(1234);
        assert_eq!(eng.inner.language, "en");
        assert_eq!(eng.inner.hotwords_score, 3.5);
        assert_eq!(eng.inner.modeling_unit, "bpe");
        assert_eq!(eng.inner.max_audio_bytes, 1234);
    }
}
