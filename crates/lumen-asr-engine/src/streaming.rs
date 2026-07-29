//! Streaming (real-time) Paraformer ASR via sherpa-onnx `OnlineRecognizer`.
//!
//! The streaming contract is fundamentally different from the offline
//! [`crate::AsrEngine`] ("whole clip in → one result out"): the caller feeds
//! audio incrementally (VAD segments or small chunks), pulls a rolling
//! *partial* hypothesis, and finalizes on an endpoint. That state machine is
//! captured by [`StreamingAsrEngine`]; [`StreamingParaformerAsr`] implements it
//! on top of sherpa-onnx's online Paraformer (`encoder.onnx` + `decoder.onnx`
//! + shared `tokens.txt`).
//!
//! Meeting usage (see `docs/MEETING.md` Stage M6 / P3): while recording, a
//! background task pushes each VAD speech segment through
//! [`StreamingAsrEngine::accept_waveform`] + [`StreamingAsrEngine::decode`],
//! reads [`StreamingAsrEngine::partial_text`] to drive live captions, and on
//! [`StreamingAsrEngine::is_endpoint`] snapshots the segment then
//! [`StreamingAsrEngine::reset`]s for the next utterance.

use crate::model_paths::ParaformerStreamingModelPaths;
use crate::AsrError;
use std::path::{Path, PathBuf};

#[cfg(feature = "sherpa")]
use sherpa_onnx::{
    OnlineParaformerModelConfig, OnlineRecognizer, OnlineRecognizerConfig, OnlineStream,
};

/// A rolling streaming hypothesis.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamingResult {
    /// The current best transcript for the in-progress utterance.
    pub text: String,
    /// True once the recognizer has committed this segment (endpoint reached
    /// and flushed). Streaming Paraformer sets this on finalized segments.
    pub is_final: bool,
}

/// Incremental ASR port for the real-time meeting layer.
///
/// Object-safe (usable as `Box<dyn StreamingAsrEngine>`). Construction is
/// engine-specific (e.g. [`StreamingParaformerAsr::new`]) because a `new`
/// returning `Self` is not object-safe; the trait covers only the streaming
/// state machine.
pub trait StreamingAsrEngine: Send {
    /// Feed one chunk of mono f32 samples. sherpa resamples internally to the
    /// model rate, so any `sample_rate` is accepted.
    fn accept_waveform(&mut self, samples: &[f32], sample_rate: u32);

    /// Run the recognizer over all audio buffered so far. Cheap to call after
    /// every [`accept_waveform`](Self::accept_waveform).
    fn decode(&mut self);

    /// The current rolling result (`text` + `is_final`).
    fn result(&self) -> StreamingResult;

    /// Convenience: just the rolling partial text.
    fn partial_text(&self) -> String {
        self.result().text
    }

    /// True when endpoint rules say the current utterance has ended. The caller
    /// should snapshot the result and then [`reset`](Self::reset).
    fn is_endpoint(&self) -> bool;

    /// Clear utterance state to start the next segment (keeps the model warm).
    fn reset(&mut self);

    /// Signal end of input so the recognizer can flush trailing context
    /// (call once when the whole capture is done).
    fn input_finished(&mut self);
}

/// Streaming Paraformer engine (real-time partials + endpointing).
pub struct StreamingParaformerAsr {
    /// The model directory (diagnostics/label only).
    model_dir: PathBuf,
    #[cfg(feature = "sherpa")]
    recognizer: OnlineRecognizer,
    #[cfg(feature = "sherpa")]
    stream: OnlineStream,
}

/// Endpoint-detection tuning (sherpa `rule*` knobs). Defaults mirror sherpa's
/// streaming examples.
#[derive(Debug, Clone, Copy)]
pub struct StreamingEndpointConfig {
    /// Trailing silence (s) after which a decoded utterance is an endpoint.
    pub rule1_min_trailing_silence: f32,
    /// Trailing silence (s) after any decoded text.
    pub rule2_min_trailing_silence: f32,
    /// Max utterance length (s) before forcing an endpoint.
    pub rule3_min_utterance_length: f32,
    pub num_threads: i32,
}

impl Default for StreamingEndpointConfig {
    fn default() -> Self {
        Self {
            rule1_min_trailing_silence: 2.4,
            rule2_min_trailing_silence: 1.2,
            rule3_min_utterance_length: 20.0,
            num_threads: 2,
        }
    }
}

impl StreamingParaformerAsr {
    /// Build from explicit encoder/decoder/tokens paths with default
    /// endpointing.
    pub fn new(paths: ParaformerStreamingModelPaths) -> Result<Self, AsrError> {
        Self::with_config(paths, StreamingEndpointConfig::default())
    }

    /// Directory convenience: discover `<dir>/{encoder,decoder}.onnx` +
    /// `tokens.txt`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        let dir = dir.as_ref();
        let paths = ParaformerStreamingModelPaths::discover(dir).ok_or_else(|| {
            AsrError::NotConfigured(format!(
                "streaming Paraformer encoder/decoder/tokens not found under {}",
                dir.display()
            ))
        })?;
        Self::new(paths)
    }

    #[cfg(feature = "sherpa")]
    pub fn with_config(
        paths: ParaformerStreamingModelPaths,
        cfg: StreamingEndpointConfig,
    ) -> Result<Self, AsrError> {
        if !paths.is_ready() {
            return Err(AsrError::NotConfigured(
                "streaming Paraformer model files missing".into(),
            ));
        }
        let model_dir = paths
            .encoder
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| paths.encoder.clone());

        let mut config = OnlineRecognizerConfig::default();
        config.model_config.paraformer = OnlineParaformerModelConfig {
            encoder: Some(paths.encoder.display().to_string()),
            decoder: Some(paths.decoder.display().to_string()),
        };
        config.model_config.tokens = Some(paths.tokens.display().to_string());
        config.model_config.num_threads = cfg.num_threads;
        config.model_config.provider = Some("cpu".into());
        config.decoding_method = Some("greedy_search".into());
        config.enable_endpoint = true;
        config.rule1_min_trailing_silence = cfg.rule1_min_trailing_silence;
        config.rule2_min_trailing_silence = cfg.rule2_min_trailing_silence;
        config.rule3_min_utterance_length = cfg.rule3_min_utterance_length;

        tracing::info!(encoder = %paths.encoder.display(), "creating streaming Paraformer OnlineRecognizer");
        let recognizer = OnlineRecognizer::create(&config).ok_or_else(|| {
            AsrError::Inference(format!(
                "failed to create streaming Paraformer recognizer under {}",
                model_dir.display()
            ))
        })?;
        let stream = recognizer.create_stream();

        Ok(Self {
            model_dir,
            recognizer,
            stream,
        })
    }

    #[cfg(not(feature = "sherpa"))]
    pub fn with_config(
        _paths: ParaformerStreamingModelPaths,
        _cfg: StreamingEndpointConfig,
    ) -> Result<Self, AsrError> {
        Err(AsrError::Unsupported(
            "build with feature `sherpa` for streaming Paraformer".into(),
        ))
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
}

impl StreamingAsrEngine for StreamingParaformerAsr {
    fn accept_waveform(&mut self, samples: &[f32], sample_rate: u32) {
        #[cfg(feature = "sherpa")]
        self.stream.accept_waveform(sample_rate as i32, samples);
        #[cfg(not(feature = "sherpa"))]
        {
            let _ = (samples, sample_rate);
        }
    }

    fn decode(&mut self) {
        #[cfg(feature = "sherpa")]
        while self.recognizer.is_ready(&self.stream) {
            self.recognizer.decode(&self.stream);
        }
    }

    fn result(&self) -> StreamingResult {
        #[cfg(feature = "sherpa")]
        {
            match self.recognizer.get_result(&self.stream) {
                Some(r) => StreamingResult {
                    text: r.text,
                    is_final: r.is_final,
                },
                None => StreamingResult::default(),
            }
        }
        #[cfg(not(feature = "sherpa"))]
        StreamingResult::default()
    }

    fn is_endpoint(&self) -> bool {
        #[cfg(feature = "sherpa")]
        {
            self.recognizer.is_endpoint(&self.stream)
        }
        #[cfg(not(feature = "sherpa"))]
        false
    }

    fn reset(&mut self) {
        #[cfg(feature = "sherpa")]
        self.recognizer.reset(&self.stream);
    }

    fn input_finished(&mut self) {
        #[cfg(feature = "sherpa")]
        self.stream.input_finished();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_result_default_is_empty_nonfinal() {
        let r = StreamingResult::default();
        assert!(r.text.is_empty());
        assert!(!r.is_final);
    }

    #[test]
    fn endpoint_config_defaults() {
        let c = StreamingEndpointConfig::default();
        assert_eq!(c.rule1_min_trailing_silence, 2.4);
        assert_eq!(c.num_threads, 2);
    }

    #[test]
    fn from_dir_errors_when_models_missing() {
        let err = StreamingParaformerAsr::from_dir("/nonexistent/paraformer/streaming");
        assert!(err.is_err());
    }

    // Object-safety: the trait must be usable behind a pointer.
    #[test]
    fn trait_is_object_safe() {
        fn _takes(_e: &mut dyn StreamingAsrEngine) {}
    }
}
