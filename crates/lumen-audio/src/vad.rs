//! Voice-activity helpers for dictation: RMS-based silence auto-stop and
//! trailing-silence trimming, plus a silero VAD backend (feature `silero`).
//!
//! The RMS gate reuses the `0.005` (≈ −46 dBFS) baseline that
//! `lumen-meeting`'s preflight shares with diar-rs. The config `mode` key
//! selects the implementation; unknown modes fall back to RMS.

use std::time::{Duration, Instant};

#[cfg(feature = "silero")]
use std::path::Path;
#[cfg(feature = "silero")]
use std::sync::atomic::{AtomicU64, Ordering};

/// Decision returned by [`SilenceAutoStop::update`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadAction {
    Continue,
    /// Sustained silence after speech — end the current dictation.
    AutoStop,
}

/// RMS state machine: wait for first speech, then stop after a sustained
/// silent stretch. Hysteresis (separate start/end thresholds) avoids flapping
/// on borderline input.
#[derive(Debug, Clone)]
pub struct SilenceAutoStop {
    /// RMS that marks the onset of speech.
    pub start_threshold: f32,
    /// RMS below which input counts as silence once speaking.
    pub end_threshold: f32,
    /// Sustained silence needed to trigger [`VadAction::AutoStop`].
    pub silence_timeout: Duration,
    speaking: bool,
    silence_since: Option<Instant>,
}

impl SilenceAutoStop {
    pub fn new(start_threshold: f32, end_threshold: f32, silence_timeout: Duration) -> Self {
        Self {
            start_threshold,
            end_threshold,
            silence_timeout,
            speaking: false,
            silence_since: None,
        }
    }

    /// Feed the latest capture-window RMS. `now` is passed in so tests control
    /// the clock.
    pub fn update(&mut self, rms: f32, now: Instant) -> VadAction {
        if !self.speaking {
            if rms >= self.start_threshold {
                self.speaking = true;
                self.silence_since = None;
            }
            return VadAction::Continue;
        }
        if rms < self.end_threshold {
            let since = *self.silence_since.get_or_insert(now);
            if now.duration_since(since) >= self.silence_timeout {
                return VadAction::AutoStop;
            }
        } else {
            self.silence_since = None;
        }
        VadAction::Continue
    }
}

/// Auto-stop decision driven by a backend-reported "last speech" timestamp
/// (used by the silero path). Mirrors [`SilenceAutoStop`] policy — never stop
/// before the first detected speech, then stop after `silence_timeout` without
/// new speech — so watchers can swap backends without changing semantics.
#[derive(Debug, Clone)]
pub struct TimestampAutoStop {
    /// Sustained silence needed to trigger [`VadAction::AutoStop`].
    pub silence_timeout: Duration,
}

impl TimestampAutoStop {
    pub fn new(silence_timeout: Duration) -> Self {
        Self { silence_timeout }
    }

    /// `last_speech_at_ms`: end-of-chunk timestamp (ms of audio fed) of the
    /// most recent detected speech, `None` when no speech has been detected
    /// yet. `elapsed_ms`: ms of audio fed to the backend so far. Both share
    /// the backend's own clock (samples fed), not wall time.
    pub fn update(&self, last_speech_at_ms: Option<u64>, elapsed_ms: u64) -> VadAction {
        let Some(last_speech_at_ms) = last_speech_at_ms else {
            // No speech yet — same as the RMS path, never auto-stop.
            return VadAction::Continue;
        };
        let silent_ms = elapsed_ms.saturating_sub(last_speech_at_ms);
        if silent_ms >= self.silence_timeout.as_millis() as u64 {
            VadAction::AutoStop
        } else {
            VadAction::Continue
        }
    }
}

/// Errors opening the silero VAD model. Fail-open by contract: callers fall
/// back to the RMS path, so a VAD problem can never break recording.
#[cfg(feature = "silero")]
#[derive(Debug, thiserror::Error)]
pub enum SileroVadError {
    #[error("silero vad model file missing: {0}")]
    ModelMissing(std::path::PathBuf),
    #[error("silero vad init failed (unsupported or corrupt model file?)")]
    InitFailed,
}

/// Silero VAD (via sherpa-onnx) fed with 16 kHz mono chunks from the capture
/// callback. Turns `detected()` into a "last speech" timestamp (ms of 16 kHz
/// audio fed) that [`TimestampAutoStop`] consumes — the same "stop only after
/// a sustained silent stretch" policy as the RMS path.
///
/// Construction loads the ONNX model (tens of ms): never construct on the
/// realtime audio thread — build it on a control thread and share the `Arc`
/// into the callback. Feeding is microseconds per 32 ms window.
#[cfg(feature = "silero")]
pub struct SileroVad {
    detector: sherpa_onnx::VoiceActivityDetector,
    /// 16 kHz samples fed since the last [`SileroVad::reset`].
    fed_samples: AtomicU64,
    /// End-of-chunk timestamp (ms) of the latest detected speech; 0 = none yet.
    last_speech_at_ms: AtomicU64,
}

#[cfg(feature = "silero")]
impl SileroVad {
    /// silero only accepts 16 kHz mono input.
    pub const SAMPLE_RATE: u32 = 16_000;

    /// Load the model at `model_path`. `threshold` is the silero speech
    /// probability cutoff (0.5 is the upstream default).
    pub fn new(model_path: &Path, threshold: f32) -> Result<Self, SileroVadError> {
        if !model_path.is_file() {
            return Err(SileroVadError::ModelMissing(model_path.to_path_buf()));
        }
        let config = sherpa_onnx::VadModelConfig {
            silero_vad: sherpa_onnx::SileroVadModelConfig {
                model: Some(model_path.to_string_lossy().into_owned()),
                threshold,
                min_silence_duration: 0.5,
                min_speech_duration: 0.25,
                window_size: 512,
                // No forced segment split during a dictation session.
                max_speech_duration: 3600.0,
            },
            sample_rate: Self::SAMPLE_RATE as i32,
            num_threads: 1,
            ..Default::default()
        };
        let detector = sherpa_onnx::VoiceActivityDetector::create(&config, 30.0)
            .ok_or(SileroVadError::InitFailed)?;
        Ok(Self {
            detector,
            fed_samples: AtomicU64::new(0),
            last_speech_at_ms: AtomicU64::new(0),
        })
    }

    /// Feed a 16 kHz mono chunk. Called from the capture callback: no
    /// allocation beyond sherpa's internal window buffer, no I/O.
    pub fn accept_waveform_16k(&self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        self.detector.accept_waveform(samples);
        let fed = self
            .fed_samples
            .fetch_add(samples.len() as u64, Ordering::SeqCst)
            + samples.len() as u64;
        if self.detector.detected() {
            // .max(1): 0 is the "no speech yet" sentinel.
            let at_ms = (fed * 1000 / u64::from(Self::SAMPLE_RATE)).max(1);
            self.last_speech_at_ms.store(at_ms, Ordering::SeqCst);
        }
        // Speech segments are not consumed by this wrapper; drop them so the
        // internal queue cannot grow without bound over a long recording.
        while !self.detector.is_empty() {
            self.detector.pop();
        }
    }

    /// ms since the last `reset` of the most recent detected speech; `None`
    /// when no speech has been detected yet.
    pub fn last_speech_at_ms(&self) -> Option<u64> {
        match self.last_speech_at_ms.load(Ordering::SeqCst) {
            0 => None,
            value => Some(value),
        }
    }

    /// ms of 16 kHz audio fed since the last `reset`.
    pub fn elapsed_ms(&self) -> u64 {
        self.fed_samples.load(Ordering::SeqCst) * 1000 / u64::from(Self::SAMPLE_RATE)
    }

    /// Start a fresh session: resets detector state and both timestamps.
    pub fn reset(&self) {
        self.detector.reset();
        self.fed_samples.store(0, Ordering::SeqCst);
        self.last_speech_at_ms.store(0, Ordering::SeqCst);
    }
}

/// Length (in samples) with trailing silence removed, keeping `padding` of the
/// quiet tail so the last syllable is never clipped. Returns the original
/// length when nothing trimmable is found.
pub fn trim_trailing_silence(
    samples: &[f32],
    sample_rate: u32,
    threshold: f32,
    window: Duration,
    padding: Duration,
) -> usize {
    let window_len = ((sample_rate as u64 * window.as_millis() as u64) / 1000).max(1) as usize;
    let padding_len = (sample_rate as u64 * padding.as_millis() as u64 / 1000) as usize;
    if samples.len() < window_len * 2 {
        return samples.len();
    }
    // Find the last voiced window scanning from the end.
    let mut cut = samples.len();
    let mut windows: Vec<usize> = (0..samples.len()).step_by(window_len).collect();
    if windows.last().copied() != Some(samples.len()) {
        windows.push(samples.len());
    }
    for pair in windows.windows(2).rev() {
        let (start, end) = (pair[0], pair[1]);
        let slice = &samples[start..end];
        let sum_sq: f64 = slice.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let rms = (sum_sq / slice.len() as f64).sqrt() as f32;
        if rms >= threshold {
            cut = end.saturating_add(padding_len).min(samples.len());
            break;
        }
        cut = start;
    }
    if cut >= samples.len() {
        return samples.len();
    }
    // All windows silent → keep the first window so ASR still sees something.
    cut.max(window_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(ms: u64) -> Instant {
        Instant::now() + Duration::from_millis(ms)
    }

    #[test]
    fn never_stops_before_first_speech() {
        let mut vad = SilenceAutoStop::new(0.02, 0.005, Duration::from_millis(1500));
        for i in 0..100 {
            assert_eq!(
                vad.update(0.001, t(i * 100)),
                VadAction::Continue,
                "silence before any speech must not auto-stop"
            );
        }
    }

    #[test]
    fn stops_after_sustained_silence_following_speech() {
        let mut vad = SilenceAutoStop::new(0.02, 0.005, Duration::from_millis(1500));
        assert_eq!(vad.update(0.05, t(0)), VadAction::Continue); // speech onset
        assert_eq!(vad.update(0.001, t(100)), VadAction::Continue);
        assert_eq!(vad.update(0.001, t(1500)), VadAction::Continue);
        assert_eq!(vad.update(0.001, t(1601)), VadAction::AutoStop);
    }

    #[test]
    fn speech_burst_resets_the_silence_clock() {
        let mut vad = SilenceAutoStop::new(0.02, 0.005, Duration::from_millis(1500));
        assert_eq!(vad.update(0.05, t(0)), VadAction::Continue);
        assert_eq!(vad.update(0.001, t(1000)), VadAction::Continue);
        assert_eq!(vad.update(0.04, t(1200)), VadAction::Continue); // reset
        assert_eq!(vad.update(0.001, t(1300)), VadAction::Continue); // new silence start
        assert_eq!(vad.update(0.001, t(2700)), VadAction::Continue); // 1400ms < timeout
        assert_eq!(vad.update(0.001, t(2801)), VadAction::AutoStop);
    }

    #[test]
    fn hysteresis_holds_during_borderline_input() {
        let mut vad = SilenceAutoStop::new(0.02, 0.005, Duration::from_millis(500));
        assert_eq!(vad.update(0.05, t(0)), VadAction::Continue);
        // Between start/end thresholds: counts as speech, silence never starts.
        for i in 1..20 {
            assert_eq!(vad.update(0.01, t(i * 100)), VadAction::Continue);
        }
    }

    #[test]
    fn trims_trailing_silence_with_padding() {
        let rate = 16_000u32;
        let mut samples = vec![0.5f32; rate as usize * 2]; // 2s voiced
        samples.extend(std::iter::repeat(0.0).take(rate as usize * 3)); // 3s silence
        let cut = trim_trailing_silence(
            &samples,
            rate,
            0.005,
            Duration::from_millis(100),
            Duration::from_millis(300),
        );
        let kept_ms = cut as u64 * 1000 / rate as u64;
        assert!(
            (2000..=2400).contains(&kept_ms),
            "expected ~2s + ≤300ms padding, kept {kept_ms}ms"
        );
    }

    #[test]
    fn keeps_first_window_when_all_silent() {
        let rate = 16_000u32;
        let samples = vec![0.0f32; rate as usize];
        let cut = trim_trailing_silence(
            &samples,
            rate,
            0.005,
            Duration::from_millis(100),
            Duration::from_millis(300),
        );
        assert_eq!(cut, rate as usize / 10);
    }

    #[test]
    fn short_input_is_never_trimmed() {
        let samples = vec![0.3f32; 1000];
        let cut = trim_trailing_silence(
            &samples,
            16_000,
            0.005,
            Duration::from_millis(100),
            Duration::from_millis(300),
        );
        assert_eq!(cut, samples.len());
    }

    // --- TimestampAutoStop (silero-path decision logic, fake signals) ------

    #[test]
    fn timestamp_never_stops_before_first_speech() {
        let watcher = TimestampAutoStop::new(Duration::from_millis(1500));
        for elapsed_s in 0..60 {
            assert_eq!(
                watcher.update(None, elapsed_s * 1000),
                VadAction::Continue,
                "no speech detected yet — must not auto-stop"
            );
        }
    }

    #[test]
    fn timestamp_stops_after_timeout_past_last_speech() {
        let watcher = TimestampAutoStop::new(Duration::from_millis(1500));
        assert_eq!(watcher.update(Some(1000), 1000), VadAction::Continue);
        assert_eq!(watcher.update(Some(1000), 2499), VadAction::Continue);
        assert_eq!(watcher.update(Some(1000), 2500), VadAction::AutoStop);
    }

    #[test]
    fn timestamp_speech_refresh_resets_the_clock() {
        let watcher = TimestampAutoStop::new(Duration::from_millis(1500));
        assert_eq!(watcher.update(Some(1000), 2000), VadAction::Continue);
        // New speech at t=2000 moves the reference point forward.
        assert_eq!(watcher.update(Some(2000), 3400), VadAction::Continue);
        assert_eq!(watcher.update(Some(2000), 3500), VadAction::AutoStop);
    }

    #[test]
    fn timestamp_clock_skew_saturates_instead_of_stopping() {
        let watcher = TimestampAutoStop::new(Duration::from_millis(1500));
        // elapsed < last_speech (session restarted between reads) — treat as
        // zero silence, never as an instant auto-stop.
        assert_eq!(watcher.update(Some(5000), 1000), VadAction::Continue);
    }

    // --- SileroVad (feature silero) -----------------------------------------

    #[cfg(feature = "silero")]
    mod silero {
        use super::*;
        use std::path::PathBuf;

        /// Real model for the inference smoke test: env override first, then
        /// the shared lumen-models install dir. Missing → skip (CI-friendly).
        fn model_path() -> Option<PathBuf> {
            if let Some(path) = std::env::var_os("LUMEN_SILERO_VAD_MODEL") {
                let path = PathBuf::from(path);
                if path.is_file() {
                    return Some(path);
                }
            }
            lumen_models::silero_vad_model_path(&lumen_models::default_silero_vad_dir())
        }

        #[test]
        fn missing_model_file_errors_instead_of_panicking() {
            let result = SileroVad::new(Path::new("/nonexistent/silero_vad.onnx"), 0.5);
            assert!(matches!(result, Err(SileroVadError::ModelMissing(_))));
        }

        #[test]
        fn silence_produces_no_speech_timestamp() {
            let Some(model) = model_path() else {
                eprintln!("skip: silero_vad.onnx not installed");
                return;
            };
            let vad = SileroVad::new(&model, 0.5).expect("load silero model");
            vad.accept_waveform_16k(&vec![0.0; 16_000]);
            assert_eq!(vad.last_speech_at_ms(), None);
            assert_eq!(vad.elapsed_ms(), 1000);
            // Reset clears both timestamps for the next session.
            vad.accept_waveform_16k(&vec![0.0; 1_600]);
            vad.reset();
            assert_eq!(vad.last_speech_at_ms(), None);
            assert_eq!(vad.elapsed_ms(), 0);
        }

        #[test]
        fn loud_speech_like_signal_is_detected_and_stamped() {
            let Some(model) = model_path() else {
                eprintln!("skip: silero_vad.onnx not installed");
                return;
            };
            let vad = SileroVad::new(&model, 0.5).expect("load silero model");
            // 1s of zeros, then 1s of amplitude-modulated harmonic stack
            // (formant-ish: 120 Hz fundamental + harmonics, 4 Hz syllable
            // envelope) — silero flags this reliably where a pure sine fails.
            vad.accept_waveform_16k(&vec![0.0; 16_000]);
            assert_eq!(vad.last_speech_at_ms(), None);
            let mut chunk = Vec::with_capacity(16_000);
            for i in 0..16_000u32 {
                let t = i as f32 / 16_000.0;
                let envelope = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 4.0 * t).sin();
                let mut s = 0.0f32;
                for h in 1..=8u32 {
                    s += (2.0 * std::f32::consts::PI * 120.0 * h as f32 * t).sin() / h as f32;
                }
                chunk.push(0.3 * envelope * s);
            }
            vad.accept_waveform_16k(&chunk);
            assert_eq!(vad.elapsed_ms(), 2000);
            assert!(
                vad.last_speech_at_ms().is_some_and(|at| at > 1000),
                "speech burst should be stamped within the second second, got {:?}",
                vad.last_speech_at_ms()
            );
        }
    }
}
