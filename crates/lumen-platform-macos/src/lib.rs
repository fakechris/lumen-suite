//! macOS platform ports — multi-display capture, frontmost, lock, permissions, OCR, ASR.
//!
//! Observe capture and process enrichment — does **not** use cua-driver.
//!
//! The navi-native ports are macOS-only (gated below); `lumen-platform-host`
//! only pulls this crate into the graph for macOS targets for those. The
//! capability modules promoted from lumen-asr (system audio process tap,
//! hotkey CGEvent tap, AUVoiceIO, power monitor/assertion) follow asr's
//! "compile everywhere, gate at runtime" contract: they expose their full API
//! surface on every platform and degrade to `Unsupported` where the capability
//! is missing, so Windows product builds link against real types.

#[cfg(target_os = "macos")]
mod asr;
#[cfg(target_os = "macos")]
pub mod ax;
#[cfg(target_os = "macos")]
pub mod ax_tree;
#[cfg(target_os = "macos")]
mod capture;
#[cfg(target_os = "macos")]
mod clipboard;
#[cfg(target_os = "macos")]
mod frontmost;
#[cfg(target_os = "macos")]
mod idle;
#[cfg(target_os = "macos")]
pub mod inject;
#[cfg(target_os = "macos")]
mod input_counter;
#[cfg(target_os = "macos")]
mod lock;
#[cfg(target_os = "macos")]
mod ocr;
#[cfg(target_os = "macos")]
mod permissions;
#[cfg(target_os = "macos")]
mod power;
#[cfg(target_os = "macos")]
mod selection;

#[cfg(target_os = "macos")]
pub use asr::MacSpeechAsr;
#[cfg(target_os = "macos")]
pub use ax_tree::MacAxTreeWalker;
#[cfg(target_os = "macos")]
pub use input_counter::{
    start_input_counter, tap_reenable_count, tap_should_reenable, InputCounterState, InputCounts,
};
/// Convenience wrappers matching the daemon's call sites.
#[cfg(target_os = "macos")]
pub fn input_snapshot(state: &InputCounterState) -> InputCounts {
    input_counter::snapshot(state)
}
#[cfg(target_os = "macos")]
pub fn input_reset(state: &InputCounterState) {
    input_counter::reset(state);
}
#[cfg(target_os = "macos")]
pub fn input_drain_hid(state: &InputCounterState) -> Vec<lumen_platform::ObserveHidEvent> {
    input_counter::drain_hid(state)
}
#[cfg(target_os = "macos")]
pub use capture::{MacDisplays, MacScreenCapturer};
#[cfg(target_os = "macos")]
pub use clipboard::clipboard_grab_selection;
#[cfg(target_os = "macos")]
pub use frontmost::MacFrontmost;
#[cfg(target_os = "macos")]
pub use idle::MacIdle;
#[cfg(target_os = "macos")]
pub use lock::{is_screen_locked, MacScreenLock};
pub use lumen_platform::normalize_selection;
#[cfg(target_os = "macos")]
pub use ocr::{default_ocr_languages, MacVisionOcr};
#[cfg(target_os = "macos")]
pub use permissions::{
    accessibility_permission_state, microphone_permission_state, request_microphone_access,
    request_screen_recording, screen_recording_access_granted, MacPermissions,
};
#[cfg(target_os = "macos")]
pub use power::MacPower;
#[cfg(target_os = "macos")]
pub use selection::{
    accessibility_trusted, focused_element_pid, focused_selection, maybe_selection, mouse_location,
    start_mouse_up_monitor, MouseUp, SelectionInfo,
};

// Modules promoted from lumen-asr's (former) product-local lumen-platform-macos.
// They are self-contained capability implementations; the asr product keeps its
// own trait-shaped adapters and re-exports these.
mod hotkey_tap;
mod power_assertion;
mod power_monitor;
mod system_audio;
mod voice_processing;

pub use hotkey_tap::{
    physical_fn_down, start_monitor, start_multi_monitor, stop_monitor, HotkeyBinding, HotkeyEdge,
    HotkeyMode, HotkeySpec,
};
pub use power_assertion::MeetingPowerGuard;
pub use power_monitor::{battery_status, install_will_sleep_observer, BatteryStatus};
pub use system_audio::{
    capability_available as system_audio_capability_available, SystemAudioCapture,
    SystemAudioError, SystemAudioSink, SystemAudioTarget,
};
pub use voice_processing::{
    voice_processing_supported, VoiceInputSink, VoiceProcessingError, VoiceProcessingInput,
};

mod permissions_tcc;
pub use permissions_tcc::{
    ensure_accessibility_onboarding, is_accessibility_trusted, prompt_accessibility,
    MacTccPermissions,
};

/// Open a System Settings / arbitrary x-apple URL via the `open` command.
pub fn open_url(url: &str) -> Result<(), lumen_platform::PlatformError> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(url)
            .spawn()
            .map_err(|e| lumen_platform::PlatformError::Message(e.to_string()))?;
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = url;
        Err(lumen_platform::PlatformError::Message("not macOS".into()))
    }
}
