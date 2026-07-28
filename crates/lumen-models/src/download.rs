//! Download + install the SenseVoice sherpa package (onboarding).
//!
//! Uses an in-process HTTP client and bzip2/tar decoder and follows the
//! cross-process install protocol from contract §6:
//! exclusive file lock → re-check → pid-unique scratch paths → validate →
//! atomic publish → cleanup before releasing the lock.

use crate::install_lock::ModelInstallLock;
use crate::paths::sensevoice_ready;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Official int8 SenseVoice package (zh/en/ja/ko/yue).
pub const SENSEVOICE_ARCHIVE_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2";
pub const SENSEVOICE_ARCHIVE_NAME: &str =
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2";

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// `"waiting"`, `"downloading"`, `"extracting"` or `"done"`.
    pub phase: String,
    pub message: String,
    pub bytes: u64,
    pub total: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("download cancelled")]
    Cancelled,
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Failed(String),
}

/// Default install root: shared `…/Lumen/models` (cluster-wide).
pub fn default_models_root() -> PathBuf {
    crate::paths::lumen_models_dir()
}

/// Install SenseVoice under `models_root/sensevoice`.
///
/// Default `models_root` is the **shared Lumen cluster** path
/// ([`default_models_root`]) so asr / navi / future apps share one download.
/// `cancel` may be flipped from another thread (e.g. a UI cancel button).
pub fn download_sensevoice_package(
    models_root: &Path,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(DownloadProgress),
) -> Result<PathBuf, DownloadError> {
    std::fs::create_dir_all(models_root)?;
    let final_dir = models_root.join("sensevoice");

    if sensevoice_ready(&final_dir) {
        on_progress(progress("done", "SenseVoice already installed", 0, None));
        return Ok(final_dir);
    }

    let mut announced_wait = false;
    let _install_lock = loop {
        if cancel.load(Ordering::SeqCst) {
            return Err(DownloadError::Cancelled);
        }
        match ModelInstallLock::try_acquire(models_root)? {
            Some(lock) => break lock,
            None => {
                if !announced_wait {
                    on_progress(progress(
                        "waiting",
                        "Another Lumen app is installing SenseVoice…",
                        0,
                        None,
                    ));
                    announced_wait = true;
                }
                thread::sleep(Duration::from_millis(250));
            }
        }
    };
    if sensevoice_ready(&final_dir) {
        on_progress(progress(
            "done",
            "SenseVoice installed by another Lumen app",
            0,
            None,
        ));
        return Ok(final_dir);
    }

    if cancel.load(Ordering::SeqCst) {
        return Err(DownloadError::Cancelled);
    }

    let process_id = std::process::id();
    let archive_path = models_root.join(format!(".{SENSEVOICE_ARCHIVE_NAME}.{process_id}.part"));
    let extract_tmp = models_root.join(format!(".sensevoice-extract-{process_id}"));
    let _scratch = DownloadScratch::new(archive_path.clone(), extract_tmp.clone());

    on_progress(progress(
        "downloading",
        "Downloading SenseVoice model…",
        0,
        None,
    ));

    let client = reqwest::blocking::Client::builder()
        .user_agent("Lumen-ASR/model-installer")
        .build()
        .map_err(|error| DownloadError::Failed(format!("create HTTP client: {error}")))?;
    let mut response = client
        .get(SENSEVOICE_ARCHIVE_URL)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| DownloadError::Failed(format!("download SenseVoice: {error}")))?;
    let total = response.content_length();
    let mut output = File::create(&archive_path)?;
    let mut bytes = 0u64;
    let mut buffer = vec![0u8; 128 * 1024];
    loop {
        if cancel.load(Ordering::SeqCst) {
            return Err(DownloadError::Cancelled);
        }
        let read = response
            .read(&mut buffer)
            .map_err(|error| DownloadError::Failed(format!("read model download: {error}")))?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        bytes += read as u64;
        on_progress(progress(
            "downloading",
            "Downloading SenseVoice model…",
            bytes,
            total,
        ));
    }
    output.flush()?;
    on_progress(progress(
        "extracting",
        "Extracting archive…",
        bytes,
        total.or(Some(bytes)),
    ));

    std::fs::create_dir_all(&extract_tmp)?;
    let archive_file = File::open(&archive_path)?;
    let decoder = bzip2::read::BzDecoder::new(archive_file);
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(&extract_tmp)
        .map_err(|error| DownloadError::Failed(format!("extract SenseVoice archive: {error}")))?;

    let found = find_sensevoice_dir(&extract_tmp).ok_or_else(|| {
        DownloadError::Failed(
            "extracted archive but could not find model*.onnx + tokens.txt".into(),
        )
    })?;

    if final_dir.exists() {
        let _ = std::fs::remove_dir_all(&final_dir);
    }
    std::fs::rename(&found, &final_dir)
        .map_err(|error| DownloadError::Failed(format!("publish model atomically: {error}")))?;

    if !sensevoice_ready(&final_dir) {
        return Err(DownloadError::Failed(
            "model installed but validation failed".into(),
        ));
    }

    on_progress(progress("done", "SenseVoice ready", bytes, Some(bytes)));
    Ok(final_dir)
}

fn progress(phase: &str, message: &str, bytes: u64, total: Option<u64>) -> DownloadProgress {
    DownloadProgress {
        phase: phase.into(),
        message: message.into(),
        bytes,
        total,
    }
}

/// Removes pid-unique scratch paths on scope exit (including error paths).
struct DownloadScratch {
    archive: PathBuf,
    extract_dir: PathBuf,
}

impl DownloadScratch {
    fn new(archive: PathBuf, extract_dir: PathBuf) -> Self {
        let _ = std::fs::remove_file(&archive);
        let _ = std::fs::remove_dir_all(&extract_dir);
        Self {
            archive,
            extract_dir,
        }
    }
}

impl Drop for DownloadScratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.archive);
        let _ = std::fs::remove_dir_all(&self.extract_dir);
    }
}

fn find_sensevoice_dir(root: &Path) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if sensevoice_ready(&dir) {
            return Some(dir);
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    stack.push(entry.path());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_dir;

    #[test]
    fn archive_url_is_https() {
        assert!(SENSEVOICE_ARCHIVE_URL.starts_with("https://"));
        assert!(SENSEVOICE_ARCHIVE_NAME.ends_with(".tar.bz2"));
    }

    #[test]
    fn already_installed_short_circuits_without_network() {
        let root = temp_dir("download-ready");
        let shared = root.join("sensevoice");
        std::fs::create_dir_all(&shared).unwrap();
        std::fs::write(shared.join("model.int8.onnx"), b"model").unwrap();
        std::fs::write(shared.join("tokens.txt"), b"tokens").unwrap();

        let cancel = AtomicBool::new(false);
        let mut phases = Vec::new();
        let installed =
            download_sensevoice_package(&root, &cancel, |p| phases.push(p.phase)).unwrap();

        assert_eq!(installed, shared);
        assert_eq!(phases, vec!["done".to_string()]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cancelled_before_start_returns_cancelled() {
        let root = temp_dir("download-cancelled");
        let cancel = AtomicBool::new(true);
        let error = download_sensevoice_package(&root, &cancel, |_| {}).unwrap_err();
        assert!(matches!(error, DownloadError::Cancelled));
        let _ = std::fs::remove_dir_all(root);
    }
}
