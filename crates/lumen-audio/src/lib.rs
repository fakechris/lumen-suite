//! Audio capture and recording machinery shared across the Lumen products.
//!
//! Mic capture (cpal) with the macOS zombie-callback epoch guard, RMS and
//! silero (feature `silero`) VAD backends, VAD-driven silence trimming, PCM16
//! WAV range editing, and the dual-track meeting recorder with live-tap
//! forwarding and WAV-header repair. Promoted from lumen-asr's product-local
//! `lumen-asr` crate.

pub mod audio;
pub mod meeting_recorder;
pub mod opus_sink;
pub mod vad;
pub mod wav_edit;

pub use audio::{AudioCapture, AudioDeviceInfo, AudioError, CaptureResult};
pub use meeting_recorder::{
    live_tap_channel, repair_wav_header, LiveAudioPacket, LiveTapSender, MeetingAudioFormat,
    MeetingRecorder, MeetingRecorderError, RecordingSummary, RepairedWav, SampleSink,
    SystemTrackRecorder, SystemTrackSender, WavSink, LIVE_TAP_CAPACITY,
};
pub use opus_sink::{decode_opus_to_pcm, pcm_to_wav_bytes, OpusSink, OPUS_SAMPLE_RATE};
pub use vad::{trim_trailing_silence, SilenceAutoStop, TimestampAutoStop, VadAction};
#[cfg(feature = "silero")]
pub use vad::{SileroVad, SileroVadError};
pub use wav_edit::{copy_pcm16_wav_range, WavRangeError, WavRangeSummary};
