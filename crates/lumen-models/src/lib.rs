//! Shared ASR model management for the Lumen product cluster.
//!
//! First crate of the shared platform layer in `lumen-suite`. Unifies the
//! model path resolution, readiness checks, cross-process install locking,
//! and SenseVoice download logic previously duplicated in
//! `lumen-asr/crates/lumen-asr` and `lumen-navi/crates/lumen-asr-engine`.
//!
//! Behavior is governed by the cluster contract in
//! `contracts/SHARED_MODELS_CONTRACT.md` (canonical here since v1.1;
//! previously in `lumen-asr/docs/`). The contract-hash test below pins the
//! contract body byte-for-byte to cluster v1.
//!
//! Feature `download` (default on — uses system `curl` + `tar`, plus `sha2`
//! for raw-file integrity pins) gates the model package installers.

mod install_lock;
mod paths;

#[cfg(feature = "download")]
mod download;

pub use install_lock::{ModelInstallLock, SENSEVOICE_INSTALL_LOCK_NAME};
#[allow(deprecated)]
pub use paths::app_models_dir;
pub use paths::{
    default_paraformer_offline_dir, default_paraformer_offline_dir_with_root,
    default_paraformer_streaming_dir, default_paraformer_streaming_dir_with_root, default_qwen_dir,
    default_sensevoice_dir, default_sensevoice_dir_with_root, default_silero_vad_dir,
    default_whisper_dir, default_whisper_dir_with_root, legacy_model_roots, lumen_models_dir,
    lumen_models_dir_with_override, paraformer_decoder_path, paraformer_encoder_path,
    paraformer_offline_model_path, paraformer_offline_ready, paraformer_streaming_ready,
    paraformer_tokens_path, qwen_ready, resolve_qwen_asr_dir, resolve_sensevoice_dir,
    scan_model_candidates, scan_model_candidates_with_root, sensevoice_model_path,
    sensevoice_ready, sensevoice_tokens_path, shared_paraformer_offline_dir,
    shared_paraformer_streaming_dir, shared_sensevoice_dir, shared_silero_vad_dir,
    shared_whisper_dir, silero_vad_model_path, silero_vad_ready, user_home_dir,
    whisper_decoder_path, whisper_encoder_path, whisper_ready, whisper_tokens_path, ModelCandidate,
    ENV_LUMEN_MODELS_DIR, ENV_LUMEN_NAVI_SENSEVOICE_DIR, ENV_LUMEN_NAVI_WHISPER_DIR,
    ENV_LUMEN_SENSEVOICE_DIR, ENV_LUMEN_WHISPER_DIR,
};

#[cfg(feature = "download")]
pub use download::{
    default_models_root, download_paraformer_offline_package,
    download_paraformer_streaming_package, download_sensevoice_package,
    download_silero_vad_package, DownloadError, DownloadProgress, PARAFORMER_OFFLINE_ARCHIVE_NAME,
    PARAFORMER_OFFLINE_ARCHIVE_URL, PARAFORMER_STREAMING_ARCHIVE_NAME,
    PARAFORMER_STREAMING_ARCHIVE_URL, SENSEVOICE_ARCHIVE_NAME, SENSEVOICE_ARCHIVE_URL,
    SILERO_VAD_BYTES, SILERO_VAD_FILE_NAME, SILERO_VAD_SHA256, SILERO_VAD_URL,
};

#[cfg(test)]
pub(crate) mod test_support {
    use std::ffi::OsStr;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Serializes tests that mutate process environment variables.
    pub static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Unique, not-yet-created temp path for a test.
    pub fn temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("lumen-models-{name}-{nonce}"))
    }

    /// Sets/unsets an env var and restores the previous value on drop.
    /// Callers must hold [`ENV_LOCK`] for the guard's whole lifetime.
    pub struct EnvGuard {
        key: String,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        pub fn set(key: &str, value: impl AsRef<OsStr>) -> Self {
            let previous = std::env::var_os(key);
            std::env::set_var(key, value);
            Self {
                key: key.to_string(),
                previous,
            }
        }

        pub fn unset(key: &str) -> Self {
            let previous = std::env::var_os(key);
            std::env::remove_var(key);
            Self {
                key: key.to_string(),
                previous,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => std::env::set_var(&self.key, value),
                None => std::env::remove_var(&self.key),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// The cluster contract body must stay byte-identical to v1
    /// (`lumen-asr` / `lumen-navi` pin the same hash over their doc copies).
    /// The lumen-suite copy prepends one provenance line before the first
    /// heading; everything from `# ` onward is the unmodified contract.
    #[test]
    fn shared_model_contract_matches_cluster_v1() {
        let bytes = include_bytes!("../../../contracts/SHARED_MODELS_CONTRACT.md");
        let body_start = bytes
            .windows(2)
            .position(|window| window == b"# ")
            .expect("contract has a title heading");
        assert!(
            body_start > 0,
            "provenance note missing above the contract body"
        );
        assert_eq!(fnv1a64(&bytes[body_start..]), 0xc877_89f4_de20_5e71);
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }
}
