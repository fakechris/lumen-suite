//! Shared ASR engine layer for Lumen products (lumen-asr, lumen-navi, and
//! future lumen-cut).
//!
//! One [`AsrEngine`] trait over:
//!
//! | Engine | Backend | Feature |
//! |--------|---------|---------|
//! | [`SenseVoiceSherpaAsr`] | Local sherpa-onnx SenseVoice | `sherpa` (default) |
//! | [`WhisperAsr`] | Local sherpa-onnx Whisper | `sherpa` (default) |
//! | [`QwenAsr`] | Local Qwen3-ASR via persistent MLX Python worker | always |
//! | `OpenAiAudioAsr` | OpenAI-compatible HTTP `/audio/transcriptions` | `cloud` |
//! | macOS Speech.framework | Consumer-provided (lumen-navi platform layer) | — |
//!
//! Path policy: engines receive model directories/files from the caller
//! ([`SenseVoiceModelPaths`], [`QwenAsrConfig::model_dir`], ...). Default-path
//! resolution and model downloads live in the consumer / `lumen-models`,
//! never here.

pub mod audio;
#[cfg(feature = "cloud")]
mod cloud_openai;
mod diagnostics;
mod model_paths;
mod paraformer;
mod qwen;
mod sensevoice;
mod streaming;
mod whisper;

pub use audio::{
    decode_wav_pcm_s16le, prepare_for_asr, prepare_for_offline_asr, resample_linear,
    samples_to_wav_mono_i16, write_wav_mono_i16, DecodedPcm, ASR_TARGET_SAMPLE_RATE,
};
#[cfg(feature = "cloud")]
pub use cloud_openai::{OpenAiAudioAsr, OpenAiAudioConfig};
pub use diagnostics::{
    AsrRuntimeDiagnostics, AsrTokenEvidence, QwenDecodeMode, QwenRuntimeMetrics,
    QwenShadowCandidate, QwenShadowDiagnostics, QwenShadowScore, QwenShadowSpan, QwenShadowStatus,
};
pub use model_paths::{
    paraformer_offline_ready, paraformer_streaming_ready, qwen_ready, sensevoice_model_path,
    sensevoice_ready, sensevoice_tokens_path, whisper_decoder_path, whisper_encoder_path,
    whisper_ready, whisper_tokens_path, ParaformerOfflineModelPaths, ParaformerStreamingModelPaths,
    SenseVoiceModelPaths, WhisperModelPaths,
};
pub use paraformer::ParaformerAsr;
pub use qwen::{QwenAsr, QwenAsrConfig, QwenShadowRequest, QwenShadowTerm};
pub use sensevoice::SenseVoiceSherpaAsr;
pub use streaming::{StreamingAsrEngine, StreamingParaformerAsr, StreamingResult};
pub use whisper::WhisperAsr;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AsrError {
    #[error("asr engine not configured: {0}")]
    NotConfigured(String),
    #[error("empty audio")]
    EmptyAudio,
    #[error("audio too large: {actual} bytes (max {max})")]
    AudioTooLarge { actual: usize, max: usize },
    #[error("invalid audio: {0}")]
    InvalidAudio(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Runtime identity of the engine that produced a result
/// (was `lumen_core::AsrEngineId`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrEngineId {
    SenseVoiceSherpa,
    Whisper,
    Qwen3Asr,
    /// Local sherpa-onnx Paraformer (offline, with word timestamps + hotwords).
    Paraformer,
    /// OpenAI-compatible HTTP engine (was serialized as `other` in lumen-asr).
    OpenAiAudio,
    /// macOS Speech.framework (engine implemented by the consumer).
    Speech,
    #[serde(other)]
    Other,
}

impl AsrEngineId {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SenseVoiceSherpa => "sensevoice_sherpa",
            Self::Whisper => "whisper",
            Self::Qwen3Asr => "qwen3_asr",
            Self::Paraformer => "paraformer",
            Self::OpenAiAudio => "openai_audio",
            Self::Speech => "speech",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AsrRequest {
    /// PCM f32 mono samples in [-1, 1].
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub hotwords: Vec<String>,
    /// Per-request ISO-639-1 language hint. Honored by HTTP engines; local
    /// engines keep their construction-time language (recognizer is warm).
    pub language_hint: Option<String>,
}

impl AsrRequest {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            hotwords: Vec::new(),
            language_hint: None,
        }
    }

    pub fn with_hotwords(mut self, hotwords: Vec<String>) -> Self {
        self.hotwords = hotwords;
        self
    }

    pub fn with_language_hint(mut self, hint: impl Into<String>) -> Self {
        self.language_hint = Some(hint.into());
        self
    }
}

/// Word- (or token-) level timing produced by engines that expose alignment
/// (offline Paraformer today). `start`/`end` are seconds from the start of the
/// decoded audio. Consumed by lumen-transcript `Word` and meeting playback.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordTiming {
    /// The token or word surface form (a CJK character for Chinese Paraformer,
    /// a sub-word/word for BPE models).
    pub word: String,
    /// Start offset in seconds.
    pub start: f64,
    /// End offset in seconds. Falls back to `start` when the model only
    /// reports a start timestamp for the last token.
    pub end: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrResult {
    pub text: String,
    pub engine: AsrEngineId,
    /// Free-form label for transcript storage (lumen-navi transcript.v1,
    /// e.g. `qwen_asr` for a DashScope-hosted model). Defaults to
    /// `engine.as_str()`.
    pub engine_label: String,
    pub language: Option<String>,
    /// 0.0 = unknown (no engine currently reports calibrated confidence).
    #[serde(default)]
    pub confidence: f32,
    /// Word/token-level timestamps when the engine emits them (offline
    /// Paraformer). Empty for engines without alignment; skipped in JSON so
    /// existing transcript payloads are byte-for-byte unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<WordTiming>,
    #[serde(default)]
    pub diagnostics: AsrRuntimeDiagnostics,
}

impl AsrResult {
    pub fn new(text: impl Into<String>, engine: AsrEngineId) -> Self {
        let engine_label = engine.as_str().to_string();
        Self {
            text: text.into(),
            engine,
            engine_label,
            language: None,
            confidence: 0.0,
            words: Vec::new(),
            diagnostics: AsrRuntimeDiagnostics::default(),
        }
    }
}

/// Unified ASR port. Semantics are the superset of the previous
/// `lumen_asr::AsrEngine` (PCM in) and `lumen_platform::AsrEngine`
/// (WAV blob + locale in).
#[async_trait]
pub trait AsrEngine: Send + Sync {
    fn id(&self) -> AsrEngineId;

    /// False when the engine cannot run in this build/host (e.g. missing
    /// `sherpa` feature or model files). Mirrors lumen-navi's probe.
    fn is_supported(&self) -> bool {
        true
    }

    /// Byte budget enforced by [`AsrEngine::transcribe_wav`]. `None` = unlimited.
    fn max_audio_bytes(&self) -> Option<usize> {
        None
    }

    /// Core path: PCM f32 mono samples (lumen-asr style).
    async fn transcribe(&self, req: AsrRequest) -> Result<AsrResult, AsrError>;

    /// Convenience path: RIFF/WAVE PCM s16le blob + locale (lumen-navi style).
    /// Decodes, resamples to 16 kHz mono, and forwards to
    /// [`AsrEngine::transcribe`] with a language hint derived from `locale`.
    async fn transcribe_wav(&self, audio: &[u8], locale: &str) -> Result<AsrResult, AsrError> {
        if audio.is_empty() {
            return Err(AsrError::EmptyAudio);
        }
        if let Some(max) = self.max_audio_bytes() {
            if audio.len() > max {
                return Err(AsrError::AudioTooLarge {
                    actual: audio.len(),
                    max,
                });
            }
        }
        let pcm = prepare_for_offline_asr(audio)?;
        if pcm.samples.is_empty() {
            return Err(AsrError::InvalidAudio("empty pcm after decode".into()));
        }
        let mut req = AsrRequest::new(pcm.samples, pcm.sample_rate);
        req.language_hint = locale_to_lang_hint(locale);
        self.transcribe(req).await
    }
}

/// Deterministic stub for tests.
pub struct StubAsr {
    canned: String,
}

impl StubAsr {
    pub fn new(canned: impl Into<String>) -> Self {
        Self {
            canned: canned.into(),
        }
    }
}

#[async_trait]
impl AsrEngine for StubAsr {
    fn id(&self) -> AsrEngineId {
        AsrEngineId::Other
    }

    async fn transcribe(&self, req: AsrRequest) -> Result<AsrResult, AsrError> {
        if req.samples.is_empty() {
            return Err(AsrError::EmptyAudio);
        }
        let mut result = AsrResult::new(self.canned.clone(), self.id());
        result.language = Some("zh".into());
        Ok(result)
    }
}

/// Derive a publish-safe identity from the model directory that actually ran.
pub fn model_identity_from_path(path: &Path) -> (Option<String>, Option<String>) {
    let leaf = path
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    let is_hugging_face_snapshot = path
        .parent()
        .and_then(|value| value.file_name())
        .and_then(|value| value.to_str())
        == Some("snapshots");
    if !is_hugging_face_snapshot {
        return (leaf, None);
    }

    let model = path
        .parent()
        .and_then(|value| value.parent())
        .and_then(|value| value.file_name())
        .and_then(|value| value.to_str())
        .map(|value| value.trim_start_matches("models--").replace("--", "/"));
    (model, leaf)
}

/// Engine selector (config / UI). Superset of both products:
///
/// - [`EngineKind::Qwen`] is the **local** Qwen3-ASR MLX worker (lumen-asr
///   semantics). Cloud-hosted Qwen (DashScope etc.) is
///   [`EngineKind::OpenAiAudio`] with a `qwen_asr` engine label.
/// - [`EngineKind::Speech`] is macOS Speech.framework; the engine itself is
///   built by the consumer (lumen-navi platform layer), this enum only
///   represents it for config/status purposes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    #[default]
    SenseVoice,
    Whisper,
    /// Local Qwen3-ASR via MLX Python worker.
    Qwen,
    /// Local sherpa-onnx offline Paraformer (word timestamps + hotwords).
    /// The **streaming** Paraformer engine is a different port
    /// ([`crate::StreamingParaformerAsr`]) and is not represented here, since
    /// it implements [`crate::StreamingAsrEngine`], not [`AsrEngine`].
    Paraformer,
    /// macOS Speech.framework (consumer-built engine).
    Speech,
    /// OpenAI-compatible HTTP ASR (Whisper API, DashScope Qwen ASR, ...).
    OpenAiAudio,
}

impl EngineKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SenseVoice => "sensevoice",
            Self::Whisper => "whisper",
            Self::Qwen => "qwen",
            Self::Paraformer => "paraformer",
            Self::Speech => "speech",
            Self::OpenAiAudio => "openai_audio",
        }
    }

    /// Accepts every alias either product used.
    ///
    /// Behavior note: lumen-navi used to map `qwen` / `qwen3-asr` to its HTTP
    /// path because it had no local worker; with the local worker available
    /// here those names now mean [`EngineKind::Qwen`]. Explicit cloud-model
    /// names (`qwen_asr`, `qwen-asr`, `qwen_asr_0.8b`) keep mapping to
    /// [`EngineKind::OpenAiAudio`].
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sensevoice" | "sensevoice_sherpa" | "sherpa" | "local_sensevoice" => {
                Some(Self::SenseVoice)
            }
            "whisper" | "local_whisper" => Some(Self::Whisper),
            "qwen" | "qwen3_asr" | "qwen3-asr" | "local_qwen" => Some(Self::Qwen),
            "paraformer" | "paraformer_offline" | "paraformer-offline" | "funasr" => {
                Some(Self::Paraformer)
            }
            "speech" | "macos_speech" | "apple" => Some(Self::Speech),
            "openai_audio" | "openai" | "http" | "cloud" | "qwen_asr" | "qwen-asr"
            | "qwen_asr_0.8b" => Some(Self::OpenAiAudio),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    pub kind: EngineKind,
    pub ready: bool,
    pub model_dir: String,
    pub detail: String,
}

/// Status probe for settings UI / logs (does not load models).
///
/// Unlike lumen-navi's `engine_status`, this never falls back to default
/// directories: pass the directory your path-resolution layer chose.
/// `model_dir = None` for a local engine reports not-ready.
pub fn probe_status(kind: EngineKind, model_dir: Option<&Path>) -> EngineStatus {
    let dir_display = |d: Option<&Path>| d.map(|p| p.display().to_string()).unwrap_or_default();
    match kind {
        EngineKind::SenseVoice => {
            let ready = model_dir.map(sensevoice_ready).unwrap_or(false);
            EngineStatus {
                kind,
                ready,
                model_dir: dir_display(model_dir),
                detail: if ready {
                    "SenseVoice model ready".into()
                } else {
                    "missing model*.onnx + tokens.txt (or no model_dir provided)".into()
                },
            }
        }
        EngineKind::Whisper => {
            let ready = model_dir.map(whisper_ready).unwrap_or(false);
            EngineStatus {
                kind,
                ready,
                model_dir: dir_display(model_dir),
                detail: if ready {
                    "Whisper model ready".into()
                } else {
                    "missing encoder/decoder/tokens onnx layout (or no model_dir provided)".into()
                },
            }
        }
        EngineKind::Qwen => {
            let ready = model_dir.map(qwen_ready).unwrap_or(false);
            EngineStatus {
                kind,
                ready,
                model_dir: dir_display(model_dir),
                detail: if ready {
                    "Qwen3-ASR MLX snapshot ready".into()
                } else {
                    "missing config/weights/tokenizer files (or no model_dir provided)".into()
                },
            }
        }
        EngineKind::Paraformer => {
            let ready = model_dir.map(paraformer_offline_ready).unwrap_or(false);
            EngineStatus {
                kind,
                ready,
                model_dir: dir_display(model_dir),
                detail: if ready {
                    "Paraformer offline model ready".into()
                } else {
                    "missing model*.onnx + tokens.txt (or no model_dir provided)".into()
                },
            }
        }
        EngineKind::Speech => EngineStatus {
            kind,
            ready: cfg!(target_os = "macos"),
            model_dir: String::new(),
            detail: "macOS Speech.framework (consumer-built engine)".into(),
        },
        EngineKind::OpenAiAudio => EngineStatus {
            kind,
            ready: cfg!(feature = "cloud"), // network; validated at first request
            model_dir: String::new(),
            detail: if cfg!(feature = "cloud") {
                "OpenAI-compatible HTTP (Qwen ASR / Whisper API)".into()
            } else {
                "build with feature `cloud`".into()
            },
        },
    }
}

/// Build inputs for engines owned by this crate (not Speech).
///
/// Replaces lumen-navi's `EngineBuildConfig`, minus the `models_root`
/// auto-resolution: `model_dir` is now mandatory for local engines.
#[derive(Debug, Clone)]
pub struct EngineBuildConfig {
    pub kind: EngineKind,
    /// Model directory for local engines. Required (no fallback resolution).
    pub model_dir: PathBuf,
    pub locale: String,
    pub max_audio_bytes: usize,
    pub http_base_url: String,
    pub http_api_key: String,
    pub http_model: String,
    pub http_timeout_ms: u64,
    /// Label stored in transcript.v1 for HTTP engines.
    pub http_engine_label: String,
    /// Python interpreter for [`EngineKind::Qwen`] (venv with mlx installed).
    pub qwen_python: PathBuf,
    pub qwen_timeout_ms: u64,
}

impl Default for EngineBuildConfig {
    fn default() -> Self {
        Self {
            kind: EngineKind::SenseVoice,
            model_dir: PathBuf::new(),
            locale: "zh-CN".into(),
            max_audio_bytes: 8 * 1024 * 1024,
            http_base_url: String::new(),
            http_api_key: String::new(),
            http_model: "whisper-1".into(),
            http_timeout_ms: 120_000,
            http_engine_label: String::new(),
            qwen_python: PathBuf::new(),
            qwen_timeout_ms: 120_000,
        }
    }
}

/// Build a local/HTTP engine. Returns `Ok(None)` for [`EngineKind::Speech`]
/// (caller supplies its own platform engine).
pub fn build_engine(cfg: &EngineBuildConfig) -> Result<Option<Arc<dyn AsrEngine>>, String> {
    match cfg.kind {
        EngineKind::Speech => Ok(None),
        EngineKind::SenseVoice => {
            if cfg.model_dir.as_os_str().is_empty() {
                return Err(
                    "sensevoice requires model_dir (path resolution is consumer-side)".into(),
                );
            }
            let eng = SenseVoiceSherpaAsr::new(cfg.model_dir.clone())
                .with_language(sensevoice_language_from_locale(&cfg.locale))
                .with_max_audio_bytes(cfg.max_audio_bytes);
            if !eng.is_ready() {
                return Err(format!(
                    "SenseVoice model not ready under {}",
                    cfg.model_dir.display()
                ));
            }
            tracing::info!(dir = %cfg.model_dir.display(), "ASR engine: sensevoice");
            Ok(Some(Arc::new(eng)))
        }
        EngineKind::Whisper => {
            if cfg.model_dir.as_os_str().is_empty() {
                return Err("whisper requires model_dir (path resolution is consumer-side)".into());
            }
            let eng = WhisperAsr::new(cfg.model_dir.clone())
                .with_language(whisper_language_from_locale(&cfg.locale))
                .with_max_audio_bytes(cfg.max_audio_bytes);
            if !eng.is_ready() {
                return Err(format!(
                    "Whisper model not ready under {}",
                    cfg.model_dir.display()
                ));
            }
            tracing::info!(dir = %cfg.model_dir.display(), "ASR engine: whisper");
            Ok(Some(Arc::new(eng)))
        }
        EngineKind::Paraformer => {
            if cfg.model_dir.as_os_str().is_empty() {
                return Err(
                    "paraformer requires model_dir (path resolution is consumer-side)".into(),
                );
            }
            let eng = ParaformerAsr::new(cfg.model_dir.clone())
                .with_language(locale_to_lang_hint(&cfg.locale).unwrap_or_else(|| "zh".into()))
                .with_max_audio_bytes(cfg.max_audio_bytes);
            if !eng.is_ready() {
                return Err(format!(
                    "Paraformer offline model not ready under {}",
                    cfg.model_dir.display()
                ));
            }
            tracing::info!(dir = %cfg.model_dir.display(), "ASR engine: paraformer (offline)");
            Ok(Some(Arc::new(eng)))
        }
        EngineKind::Qwen => {
            if cfg.model_dir.as_os_str().is_empty() {
                return Err("qwen requires model_dir (MLX snapshot)".into());
            }
            if cfg.qwen_python.as_os_str().is_empty() {
                return Err("qwen requires qwen_python (interpreter with mlx installed)".into());
            }
            let eng = QwenAsr::new(QwenAsrConfig::product(
                cfg.qwen_python.clone(),
                cfg.model_dir.clone(),
                locale_to_lang_hint(&cfg.locale),
                Duration::from_millis(cfg.qwen_timeout_ms.max(5_000)),
            ));
            tracing::info!(dir = %cfg.model_dir.display(), "ASR engine: qwen (local MLX worker)");
            Ok(Some(Arc::new(eng)))
        }
        EngineKind::OpenAiAudio => {
            #[cfg(not(feature = "cloud"))]
            {
                Err("openai_audio requires building lumen-asr-engine with feature `cloud`".into())
            }
            #[cfg(feature = "cloud")]
            {
                let base = cfg.http_base_url.trim();
                if base.is_empty() {
                    return Err(
                        "openai_audio/qwen_asr requires http_base_url (OpenAI-compatible endpoint)"
                            .into(),
                    );
                }
                let model = if cfg.http_model.trim().is_empty() {
                    "whisper-1".to_string()
                } else {
                    cfg.http_model.clone()
                };
                let label = if cfg.http_engine_label.trim().is_empty() {
                    guess_http_label(base, &model)
                } else {
                    cfg.http_engine_label.clone()
                };
                let http = OpenAiAudioConfig {
                    base_url: base.to_string(),
                    api_key: cfg.http_api_key.clone(),
                    model: model.clone(),
                    timeout: Duration::from_millis(cfg.http_timeout_ms.max(5_000)),
                    language: locale_to_lang_hint(&cfg.locale),
                    max_audio_bytes: cfg.max_audio_bytes,
                    engine_label: label,
                };
                let eng = OpenAiAudioAsr::new(http).map_err(|e| e.to_string())?;
                tracing::info!(base = %base, model = %model, "ASR engine: openai_audio");
                Ok(Some(Arc::new(eng)))
            }
        }
    }
}

pub fn sensevoice_language_from_locale(locale: &str) -> String {
    let primary = locale
        .split(['-', '_'])
        .next()
        .unwrap_or("auto")
        .to_ascii_lowercase();
    match primary.as_str() {
        "zh" | "yue" | "ja" | "ko" | "en" => primary,
        _ => "auto".into(),
    }
}

pub fn whisper_language_from_locale(locale: &str) -> String {
    locale
        .split(['-', '_'])
        .next()
        .unwrap_or("en")
        .to_ascii_lowercase()
}

pub fn locale_to_lang_hint(locale: &str) -> Option<String> {
    let primary = locale.split(['-', '_']).next()?.to_ascii_lowercase();
    if primary.is_empty() {
        None
    } else {
        Some(primary)
    }
}

/// Heuristic transcript label for HTTP engines (from lumen-navi).
pub fn guess_http_label(base: &str, model: &str) -> String {
    let b = base.to_ascii_lowercase();
    let m = model.to_ascii_lowercase();
    if b.contains("dashscope") || m.contains("qwen") {
        "qwen_asr".into()
    } else {
        "openai_audio".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // From lumen-asr lib.rs.
    #[tokio::test]
    async fn stub_transcribes() {
        let eng = StubAsr::new("hello");
        let r = eng
            .transcribe(AsrRequest::new(vec![0.1, 0.2], 16000))
            .await
            .unwrap();
        assert_eq!(r.text, "hello");
        assert_eq!(r.engine_label, "other");
    }

    #[tokio::test]
    async fn stub_transcribes_wav_blob() {
        let eng = StubAsr::new("hi");
        let wav = samples_to_wav_mono_i16(&[0.1f32; 1600], 16_000);
        let r = eng.transcribe_wav(&wav, "zh-CN").await.unwrap();
        assert_eq!(r.text, "hi");
    }

    // Union of both products' parse tests.
    #[test]
    fn parse_engines() {
        assert_eq!(
            EngineKind::parse("sensevoice"),
            Some(EngineKind::SenseVoice)
        );
        assert_eq!(EngineKind::parse("sherpa"), Some(EngineKind::SenseVoice));
        assert_eq!(
            EngineKind::parse("local_sensevoice"),
            Some(EngineKind::SenseVoice)
        );
        assert_eq!(EngineKind::parse("whisper"), Some(EngineKind::Whisper));
        assert_eq!(
            EngineKind::parse("local_whisper"),
            Some(EngineKind::Whisper)
        );
        assert_eq!(EngineKind::parse("speech"), Some(EngineKind::Speech));
        assert_eq!(EngineKind::parse("apple"), Some(EngineKind::Speech));
        assert_eq!(EngineKind::parse("openai"), Some(EngineKind::OpenAiAudio));
        assert_eq!(EngineKind::parse("http"), Some(EngineKind::OpenAiAudio));
        assert_eq!(EngineKind::parse("nope"), None);
    }

    // From lumen-asr: qwen aliases mean the local worker.
    #[test]
    fn qwen_engine_kind_accepts_product_provider_names() {
        assert_eq!(EngineKind::parse("qwen"), Some(EngineKind::Qwen));
        assert_eq!(EngineKind::parse("qwen3_asr"), Some(EngineKind::Qwen));
        assert_eq!(EngineKind::parse("local_qwen"), Some(EngineKind::Qwen));
        assert_eq!(EngineKind::Qwen.as_str(), "qwen");
    }

    // Adapted from lumen-navi: cloud-hosted Qwen names stay on the HTTP path.
    // (Bare "qwen" now means the local worker — documented behavior change.)
    #[test]
    fn cloud_qwen_names_map_to_openai_audio() {
        assert_eq!(EngineKind::parse("qwen_asr"), Some(EngineKind::OpenAiAudio));
        assert_eq!(
            EngineKind::parse("qwen_asr_0.8b"),
            Some(EngineKind::OpenAiAudio)
        );
    }

    // From lumen-navi lib.rs.
    #[test]
    fn speech_build_is_none() {
        let cfg = EngineBuildConfig {
            kind: EngineKind::Speech,
            ..EngineBuildConfig::default()
        };
        assert!(build_engine(&cfg).unwrap().is_none());
    }

    // From lumen-navi lib.rs (also passes without the `cloud` feature, where
    // the error is "build with feature cloud").
    #[test]
    fn openai_requires_url() {
        let cfg = EngineBuildConfig {
            kind: EngineKind::OpenAiAudio,
            http_base_url: String::new(),
            ..EngineBuildConfig::default()
        };
        assert!(build_engine(&cfg).is_err());
    }

    #[test]
    fn local_engines_require_model_dir() {
        for kind in [
            EngineKind::SenseVoice,
            EngineKind::Whisper,
            EngineKind::Qwen,
        ] {
            let cfg = EngineBuildConfig {
                kind,
                ..EngineBuildConfig::default()
            };
            assert!(
                build_engine(&cfg).is_err(),
                "{kind:?} should require model_dir"
            );
        }
    }

    // From lumen-asr lib.rs.
    #[test]
    fn hugging_face_snapshot_identity_omits_local_path() {
        let path =
            Path::new("/tmp/cache/models--mlx-community--Qwen3-ASR-0.6B-8bit/snapshots/abcdef123");
        let (model, revision) = model_identity_from_path(path);

        assert_eq!(model.as_deref(), Some("mlx-community/Qwen3-ASR-0.6B-8bit"));
        assert_eq!(revision.as_deref(), Some("abcdef123"));
        assert!(!model.unwrap().contains("/tmp/cache"));
    }

    // From lumen-asr lib.rs.
    #[test]
    fn direct_model_identity_uses_only_directory_name() {
        let (model, revision) =
            model_identity_from_path(Path::new("/private/models/sensevoice-int8"));

        assert_eq!(model.as_deref(), Some("sensevoice-int8"));
        assert_eq!(revision, None);
    }

    #[test]
    fn probe_status_without_dir_is_not_ready() {
        assert!(!probe_status(EngineKind::SenseVoice, None).ready);
        assert!(!probe_status(EngineKind::Whisper, None).ready);
        assert!(!probe_status(EngineKind::Qwen, None).ready);
        assert!(!probe_status(EngineKind::Paraformer, None).ready);
    }

    #[test]
    fn parse_paraformer_aliases() {
        assert_eq!(
            EngineKind::parse("paraformer"),
            Some(EngineKind::Paraformer)
        );
        assert_eq!(
            EngineKind::parse("paraformer_offline"),
            Some(EngineKind::Paraformer)
        );
        assert_eq!(
            EngineKind::parse("Paraformer-Offline"),
            Some(EngineKind::Paraformer)
        );
        assert_eq!(EngineKind::parse("funasr"), Some(EngineKind::Paraformer));
        assert_eq!(EngineKind::Paraformer.as_str(), "paraformer");
        assert_eq!(AsrEngineId::Paraformer.as_str(), "paraformer");
    }

    #[test]
    fn paraformer_build_requires_model_dir() {
        let cfg = EngineBuildConfig {
            kind: EngineKind::Paraformer,
            ..EngineBuildConfig::default()
        };
        assert!(build_engine(&cfg).is_err());
    }

    #[test]
    fn asr_result_words_round_trip_and_skip_when_empty() {
        // Empty words are omitted entirely (backward-compatible payload).
        let mut r = AsrResult::new("hi", AsrEngineId::SenseVoiceSherpa);
        let json = serde_json::to_string(&r).unwrap();
        assert!(
            !json.contains("words"),
            "empty words must be skipped: {json}"
        );

        // Populated words serialize and round-trip.
        r.words = vec![
            WordTiming {
                word: "你".into(),
                start: 0.0,
                end: 0.5,
            },
            WordTiming {
                word: "好".into(),
                start: 0.5,
                end: 1.0,
            },
        ];
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"words\""));
        let back: AsrResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.words, r.words);

        // Legacy payloads without a `words` key still deserialize.
        let legacy = r#"{"text":"x","engine":"whisper","engine_label":"whisper","language":null}"#;
        let parsed: AsrResult = serde_json::from_str(legacy).unwrap();
        assert!(parsed.words.is_empty());
    }

    #[test]
    fn locale_hint() {
        assert_eq!(locale_to_lang_hint("zh-CN").as_deref(), Some("zh"));
        assert_eq!(locale_to_lang_hint("en_US").as_deref(), Some("en"));
        assert_eq!(sensevoice_language_from_locale("fr-FR"), "auto");
        assert_eq!(whisper_language_from_locale("en-US"), "en");
    }
}
