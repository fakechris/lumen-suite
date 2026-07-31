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
//!
//! # Multi-stream (one model, many tracks)
//!
//! sherpa-onnx supports creating **many `OnlineStream`s from one
//! `OnlineRecognizer`** — the ~1 GB streaming Paraformer weights are loaded
//! once and shared by every stream. That split is exposed here as
//! [`StreamingRecognizer`] (holds the model) + [`StreamingStream`] (cheap,
//! per-track decoding state):
//!
//! ```no_run
//! use lumen_asr_engine::StreamingRecognizer;
//!
//! let recognizer = StreamingRecognizer::from_dir("/path/to/streaming-paraformer")?;
//! let mut mic = recognizer.new_stream();
//! let mut system = recognizer.new_stream();
//!
//! // Poll both tracks from the same dedicated thread:
//! mic.accept_waveform(&[0.0f32; 1600], 16_000);
//! system.accept_waveform(&[0.0f32; 1600], 16_000);
//! recognizer.decode_batch(&mut [&mut mic, &mut system]); // or mic.decode(); system.decode();
//! let _caption = mic.result();
//! # Ok::<(), lumen_asr_engine::AsrError>(())
//! ```
//!
//! ## Threading contract
//!
//! The `sherpa-onnx` crate marks `OnlineRecognizer`/`OnlineStream` as
//! `Send + Sync` (via `unsafe impl`, "thread-safe for single-object usage"),
//! so these wrappers are `Send` without any `unsafe` on our side and the
//! existing pattern of moving the engine onto a dedicated worker thread keeps
//! working. However the underlying C objects are **not** internally
//! synchronized for concurrent calls, so do not decode the same recognizer or
//! stream from two threads at once. The supported multi-track pattern is
//! **multi-stream = same-thread polling**: one dedicated thread owns the
//! [`StreamingRecognizer`] and *all* of its [`StreamingStream`]s, and round
//! robins accept/decode/result across them.
//!
//! Lifetime is handled internally: every [`StreamingStream`] keeps the shared
//! recognizer alive (`Arc`), so streams may outlive the
//! [`StreamingRecognizer`] handle that created them.

use crate::model_paths::ParaformerStreamingModelPaths;
use crate::AsrError;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

/// Shared recognizer state: the loaded model plus its directory label.
/// Held behind an `Arc` by [`StreamingRecognizer`] *and* every
/// [`StreamingStream`], guaranteeing the sherpa recognizer outlives all of
/// its streams.
struct RecognizerInner {
    /// The model directory (diagnostics/label only).
    model_dir: PathBuf,
    #[cfg(feature = "sherpa")]
    recognizer: OnlineRecognizer,
}

/// Streaming Paraformer model, loaded once and shared by any number of
/// [`StreamingStream`]s (see the [module docs](self) for the threading
/// contract: drive a recognizer and all of its streams from one thread).
///
/// Cloning is cheap (`Arc` handle to the same loaded model).
#[derive(Clone)]
pub struct StreamingRecognizer {
    inner: Arc<RecognizerInner>,
}

impl StreamingRecognizer {
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

        Ok(Self {
            inner: Arc::new(RecognizerInner {
                model_dir,
                recognizer,
            }),
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
        &self.inner.model_dir
    }

    /// Create a new lightweight decoding stream backed by this recognizer's
    /// (already loaded) model. Call once per track (e.g. mic + system audio).
    pub fn new_stream(&self) -> StreamingStream {
        StreamingStream {
            #[cfg(feature = "sherpa")]
            stream: self.inner.recognizer.create_stream(),
            inner: Arc::clone(&self.inner),
        }
    }

    /// Decode all given streams until none has buffered audio left, batching
    /// ready streams into single sherpa `DecodeMultipleOnlineStreams` calls.
    ///
    /// Equivalent to calling [`StreamingStream::decode`] on each stream, but
    /// lets sherpa batch the model forward pass across tracks.
    ///
    /// # Panics
    ///
    /// Panics if any stream was created by a *different* recognizer.
    pub fn decode_batch(&self, streams: &mut [&mut StreamingStream]) {
        for s in streams.iter() {
            assert!(
                Arc::ptr_eq(&self.inner, &s.inner),
                "StreamingRecognizer::decode_batch: stream was created by a different recognizer"
            );
        }
        #[cfg(feature = "sherpa")]
        loop {
            let ready: Vec<&OnlineStream> = streams
                .iter()
                .filter(|s| self.inner.recognizer.is_ready(&s.stream))
                .map(|s| &s.stream)
                .collect();
            if ready.is_empty() {
                break;
            }
            self.inner.recognizer.decode_multiple_streams(&ready);
        }
    }
}

/// One track's decoding state on a shared [`StreamingRecognizer`].
///
/// Cheap to create ([`StreamingRecognizer::new_stream`]); holds no model
/// weights of its own. Keeps the shared recognizer alive via `Arc`, so it may
/// outlive the `StreamingRecognizer` handle. Implements [`StreamingAsrEngine`],
/// so per-track consumers can stay generic over `Box<dyn StreamingAsrEngine>`.
///
/// Threading: use on the same thread as its sibling streams — see the
/// [module docs](self).
pub struct StreamingStream {
    inner: Arc<RecognizerInner>,
    #[cfg(feature = "sherpa")]
    stream: OnlineStream,
}

impl StreamingStream {
    /// Feed one chunk of mono f32 samples. sherpa resamples internally to the
    /// model rate, so any `sample_rate` is accepted.
    pub fn accept_waveform(&mut self, samples: &[f32], sample_rate: u32) {
        #[cfg(feature = "sherpa")]
        self.stream.accept_waveform(sample_rate as i32, samples);
        #[cfg(not(feature = "sherpa"))]
        {
            let _ = (samples, sample_rate);
        }
    }

    /// Run the recognizer over all audio buffered so far on this stream.
    pub fn decode(&mut self) {
        #[cfg(feature = "sherpa")]
        while self.inner.recognizer.is_ready(&self.stream) {
            self.inner.recognizer.decode(&self.stream);
        }
    }

    /// The current rolling result (`text` + `is_final`).
    pub fn result(&self) -> StreamingResult {
        #[cfg(feature = "sherpa")]
        {
            match self.inner.recognizer.get_result(&self.stream) {
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

    /// Convenience: just the rolling partial text.
    pub fn partial_text(&self) -> String {
        self.result().text
    }

    /// True when endpoint rules say the current utterance has ended.
    pub fn is_endpoint(&self) -> bool {
        #[cfg(feature = "sherpa")]
        {
            self.inner.recognizer.is_endpoint(&self.stream)
        }
        #[cfg(not(feature = "sherpa"))]
        false
    }

    /// Clear utterance state to start the next segment (keeps the model warm).
    pub fn reset(&mut self) {
        #[cfg(feature = "sherpa")]
        self.inner.recognizer.reset(&self.stream);
    }

    /// Signal end of input so the recognizer can flush trailing context.
    pub fn input_finished(&mut self) {
        #[cfg(feature = "sherpa")]
        self.stream.input_finished();
    }

    /// The shared model directory (diagnostics/label only).
    pub fn model_dir(&self) -> &Path {
        &self.inner.model_dir
    }
}

impl StreamingAsrEngine for StreamingStream {
    fn accept_waveform(&mut self, samples: &[f32], sample_rate: u32) {
        StreamingStream::accept_waveform(self, samples, sample_rate);
    }

    fn decode(&mut self) {
        StreamingStream::decode(self);
    }

    fn result(&self) -> StreamingResult {
        StreamingStream::result(self)
    }

    fn is_endpoint(&self) -> bool {
        StreamingStream::is_endpoint(self)
    }

    fn reset(&mut self) {
        StreamingStream::reset(self);
    }

    fn input_finished(&mut self) {
        StreamingStream::input_finished(self);
    }
}

/// Streaming Paraformer engine (real-time partials + endpointing).
///
/// Backward-compatible single-stream façade: a [`StreamingRecognizer`] plus
/// one [`StreamingStream`], behaving exactly like the historical 1:1
/// recognizer/stream pairing. New multi-track code (mic + system audio)
/// should use [`StreamingRecognizer`] directly and create one
/// [`StreamingStream`] per track so the ~1 GB model is loaded only once.
pub struct StreamingParaformerAsr {
    stream: StreamingStream,
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
        let recognizer = StreamingRecognizer::from_dir(dir)?;
        Ok(Self {
            stream: recognizer.new_stream(),
        })
    }

    pub fn with_config(
        paths: ParaformerStreamingModelPaths,
        cfg: StreamingEndpointConfig,
    ) -> Result<Self, AsrError> {
        let recognizer = StreamingRecognizer::with_config(paths, cfg)?;
        Ok(Self {
            stream: recognizer.new_stream(),
        })
    }

    pub fn model_dir(&self) -> &Path {
        self.stream.model_dir()
    }
}

impl StreamingAsrEngine for StreamingParaformerAsr {
    fn accept_waveform(&mut self, samples: &[f32], sample_rate: u32) {
        self.stream.accept_waveform(samples, sample_rate);
    }

    fn decode(&mut self) {
        self.stream.decode();
    }

    fn result(&self) -> StreamingResult {
        self.stream.result()
    }

    fn is_endpoint(&self) -> bool {
        self.stream.is_endpoint()
    }

    fn reset(&mut self) {
        self.stream.reset();
    }

    fn input_finished(&mut self) {
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

    #[test]
    fn recognizer_from_dir_errors_when_models_missing() {
        let err = StreamingRecognizer::from_dir("/nonexistent/paraformer/streaming");
        assert!(err.is_err());
    }

    // Object-safety: the trait must be usable behind a pointer.
    #[test]
    fn trait_is_object_safe() {
        fn _takes(_e: &mut dyn StreamingAsrEngine) {}
    }

    // Compile-time API-shape guarantees for the multi-stream split (no model
    // needed): one recognizer hands out many streams, each usable as a
    // `Box<dyn StreamingAsrEngine>`, and everything is `Send` so the
    // dedicated-thread pattern keeps working.
    #[test]
    fn multi_stream_api_shape() {
        fn _assert_send<T: Send>() {}
        _assert_send::<StreamingRecognizer>();
        _assert_send::<StreamingStream>();
        _assert_send::<StreamingParaformerAsr>();

        fn _two_tracks(recognizer: &StreamingRecognizer) {
            let mut mic = recognizer.new_stream();
            let mut system = recognizer.new_stream();
            mic.accept_waveform(&[0.0; 160], 16_000);
            system.accept_waveform(&[0.0; 160], 16_000);
            recognizer.decode_batch(&mut [&mut mic, &mut system]);
            let _ = (mic.result(), system.result());
            let _boxed: Box<dyn StreamingAsrEngine> = Box::new(mic);
        }
    }
}
