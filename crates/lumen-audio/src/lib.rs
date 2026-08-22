//! Audio capture and recording machinery shared across the Lumen products.
//!
//! Mic capture (cpal) with the macOS zombie-callback epoch guard, VAD-driven
//! silence trimming, PCM16 WAV range editing, and the dual-track meeting
//! recorder with live-tap forwarding and WAV-header repair. Promoted from
//! lumen-asr's product-local `lumen-asr` crate.

pub mod audio;
pub mod meeting_recorder;
pub mod vad;
pub mod wav_edit;

pub use audio::{AudioCapture, AudioDeviceInfo, AudioError, CaptureResult};
pub use meeting_recorder::{
    live_tap_channel, repair_wav_header, LiveAudioPacket, LiveTapSender, MeetingRecorder,
    MeetingRecorderError, RecordingSummary, RepairedWav, SampleSink, SystemTrackRecorder,
    SystemTrackSender, WavSink, LIVE_TAP_CAPACITY,
};
pub use vad::{trim_trailing_silence, SilenceAutoStop, VadAction};
pub use wav_edit::{copy_pcm16_wav_range, WavRangeError, WavRangeSummary};
