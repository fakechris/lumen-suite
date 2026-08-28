//! Download + install sherpa model packages (onboarding).
//!
//! Uses system `curl` + `tar` (macOS-friendly, zero extra Rust dependencies)
//! and follows the cross-process install protocol from contract §6:
//! exclusive file lock → re-check → pid-unique scratch paths → validate →
//! atomic publish → cleanup before releasing the lock.
//!
//! One generic engine ([`download_model_package`]) backs every model; each
//! model is described by a [`ModelPackage`]. SenseVoice, offline Paraformer,
//! and streaming Paraformer ship as `.tar.bz2` archives (extracted with
//! `tar -xjf`); Silero VAD ships as a single raw `.onnx` file that is verified
//! against a pinned SHA256 + size before publishing.

use crate::install_lock::ModelInstallLock;
use crate::paths::{
    paraformer_offline_ready, paraformer_streaming_ready, sensevoice_ready, silero_vad_ready,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Official int8 SenseVoice package (zh/en/ja/ko/yue).
pub const SENSEVOICE_ARCHIVE_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2";
pub const SENSEVOICE_ARCHIVE_NAME: &str =
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2";

/// Offline (non-streaming) Chinese Paraformer package.
///
/// Extracts to `sherpa-onnx-paraformer-zh-2023-09-14/` containing
/// `model.int8.onnx`, `model.onnx` and `tokens.txt`.
pub const PARAFORMER_OFFLINE_ARCHIVE_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2";
pub const PARAFORMER_OFFLINE_ARCHIVE_NAME: &str = "sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2";

/// Streaming bilingual (zh/en) Paraformer package.
///
/// Extracts to `sherpa-onnx-streaming-paraformer-bilingual-zh-en/` containing
/// `encoder.{int8.,}onnx`, `decoder.{int8.,}onnx` and `tokens.txt`.
pub const PARAFORMER_STREAMING_ARCHIVE_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2";
pub const PARAFORMER_STREAMING_ARCHIVE_NAME: &str =
    "sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2";

/// Official Silero VAD model (single ONNX file, ~2 MB, not an archive).
pub const SILERO_VAD_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";
pub const SILERO_VAD_FILE_NAME: &str = "silero_vad.onnx";
/// Pinned integrity for [`SILERO_VAD_URL`], verified after every download
/// (fail closed: a corrupted or tampered download is deleted, not installed).
pub const SILERO_VAD_SHA256: &str =
    "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6";
pub const SILERO_VAD_BYTES: u64 = 643_854;

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

/// How a package's extracted files are moved into their final directory.
enum Publish {
    /// Rename the whole discovered directory into place (SenseVoice legacy —
    /// keeps every extracted file byte-for-byte).
    WholeDir,
    /// Copy only the selected files into a fresh target dir. Each entry is a
    /// list of candidate file names in priority order; the first that exists in
    /// the extracted directory is copied (keeping its original name).
    SelectFiles(&'static [&'static [&'static str]]),
}

/// What the downloaded bytes are and how they become the installed package.
enum Payload {
    /// `.tar.bz2` archive, extracted with `tar -xjf` then published.
    TarBz2(Publish),
    /// Single raw file downloaded as-is. Verified against the pinned
    /// SHA256 + size, then renamed into `final_dir/file_name`.
    RawFile {
        file_name: &'static str,
        sha256: &'static str,
        bytes: u64,
    },
}

/// A downloadable sherpa model package.
struct ModelPackage {
    url: &'static str,
    archive_name: &'static str,
    /// Final directory relative to `models_root`, `/`-separated
    /// (e.g. `"sensevoice"`, `"paraformer/offline"`).
    rel_dir: &'static str,
    /// Slug for pid-unique scratch dirs (no path separators).
    slug: &'static str,
    /// Human name used in progress messages.
    display: &'static str,
    /// Readiness predicate for the final directory.
    ready: fn(&Path) -> bool,
    payload: Payload,
}

const SENSEVOICE_PACKAGE: ModelPackage = ModelPackage {
    url: SENSEVOICE_ARCHIVE_URL,
    archive_name: SENSEVOICE_ARCHIVE_NAME,
    rel_dir: "sensevoice",
    slug: "sensevoice",
    display: "SenseVoice",
    ready: sensevoice_ready,
    payload: Payload::TarBz2(Publish::WholeDir),
};

const PARAFORMER_OFFLINE_PACKAGE: ModelPackage = ModelPackage {
    url: PARAFORMER_OFFLINE_ARCHIVE_URL,
    archive_name: PARAFORMER_OFFLINE_ARCHIVE_NAME,
    rel_dir: "paraformer/offline",
    slug: "paraformer-offline",
    display: "Paraformer (offline)",
    ready: paraformer_offline_ready,
    payload: Payload::TarBz2(Publish::SelectFiles(&[
        &["model.int8.onnx", "model.onnx"],
        &["tokens.txt"],
    ])),
};

const PARAFORMER_STREAMING_PACKAGE: ModelPackage = ModelPackage {
    url: PARAFORMER_STREAMING_ARCHIVE_URL,
    archive_name: PARAFORMER_STREAMING_ARCHIVE_NAME,
    rel_dir: "paraformer/streaming",
    slug: "paraformer-streaming",
    display: "Paraformer (streaming)",
    ready: paraformer_streaming_ready,
    payload: Payload::TarBz2(Publish::SelectFiles(&[
        &["encoder.int8.onnx", "encoder.onnx"],
        &["decoder.int8.onnx", "decoder.onnx"],
        &["tokens.txt"],
    ])),
};

const SILERO_VAD_PACKAGE: ModelPackage = ModelPackage {
    url: SILERO_VAD_URL,
    archive_name: SILERO_VAD_FILE_NAME,
    rel_dir: "silero-vad",
    slug: "silero-vad",
    display: "Silero VAD",
    ready: silero_vad_ready,
    payload: Payload::RawFile {
        file_name: SILERO_VAD_FILE_NAME,
        sha256: SILERO_VAD_SHA256,
        bytes: SILERO_VAD_BYTES,
    },
};

/// Install SenseVoice under `models_root/sensevoice`.
///
/// Default `models_root` is the **shared Lumen cluster** path
/// ([`default_models_root`]) so asr / navi / future apps share one download.
/// `cancel` may be flipped from another thread (e.g. a UI cancel button).
pub fn download_sensevoice_package(
    models_root: &Path,
    cancel: &AtomicBool,
    on_progress: impl FnMut(DownloadProgress),
) -> Result<PathBuf, DownloadError> {
    download_model_package(models_root, &SENSEVOICE_PACKAGE, cancel, on_progress)
}

/// Install the offline Paraformer model under `models_root/paraformer/offline`.
///
/// Shares the cluster install lock, progress protocol, and curl+tar mechanism
/// with [`download_sensevoice_package`]; short-circuits when already installed.
pub fn download_paraformer_offline_package(
    models_root: &Path,
    cancel: &AtomicBool,
    on_progress: impl FnMut(DownloadProgress),
) -> Result<PathBuf, DownloadError> {
    download_model_package(
        models_root,
        &PARAFORMER_OFFLINE_PACKAGE,
        cancel,
        on_progress,
    )
}

/// Install the streaming Paraformer model under
/// `models_root/paraformer/streaming`.
pub fn download_paraformer_streaming_package(
    models_root: &Path,
    cancel: &AtomicBool,
    on_progress: impl FnMut(DownloadProgress),
) -> Result<PathBuf, DownloadError> {
    download_model_package(
        models_root,
        &PARAFORMER_STREAMING_PACKAGE,
        cancel,
        on_progress,
    )
}

/// Install the Silero VAD model under `models_root/silero-vad`.
///
/// Single raw ONNX file (no archive): the download is verified against the
/// pinned [`SILERO_VAD_SHA256`] + [`SILERO_VAD_BYTES`] before publishing —
/// fail closed on mismatch. Shares the cluster install lock and progress
/// protocol with the other packages; short-circuits when already installed.
pub fn download_silero_vad_package(
    models_root: &Path,
    cancel: &AtomicBool,
    on_progress: impl FnMut(DownloadProgress),
) -> Result<PathBuf, DownloadError> {
    download_model_package(models_root, &SILERO_VAD_PACKAGE, cancel, on_progress)
}

/// The generic install engine backing every [`ModelPackage`].
fn download_model_package(
    models_root: &Path,
    package: &ModelPackage,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(DownloadProgress),
) -> Result<PathBuf, DownloadError> {
    std::fs::create_dir_all(models_root)?;
    let final_dir = package_final_dir(models_root, package);

    if (package.ready)(&final_dir) {
        on_progress(progress(
            "done",
            &format!("{} already installed", package.display),
            0,
            None,
        ));
        return Ok(final_dir);
    }

    // Cluster install lock is shared across all engines (one per models_root),
    // so SenseVoice and Paraformer installs never run concurrently.
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
                        &format!("Another Lumen app is installing {}…", package.display),
                        0,
                        None,
                    ));
                    announced_wait = true;
                }
                thread::sleep(Duration::from_millis(250));
            }
        }
    };
    if (package.ready)(&final_dir) {
        on_progress(progress(
            "done",
            &format!("{} installed by another Lumen app", package.display),
            0,
            None,
        ));
        return Ok(final_dir);
    }

    if cancel.load(Ordering::SeqCst) {
        return Err(DownloadError::Cancelled);
    }

    let process_id = std::process::id();
    let archive_path = models_root.join(format!(".{}.{process_id}.part", package.archive_name));
    let extract_tmp = models_root.join(format!(".{}-extract-{process_id}", package.slug));
    let _scratch = DownloadScratch::new(archive_path.clone(), extract_tmp.clone());

    on_progress(progress(
        "downloading",
        &format!("Downloading {} model…", package.display),
        0,
        None,
    ));

    let archive_str = archive_path
        .to_str()
        .ok_or_else(|| DownloadError::Failed("bad archive path".into()))?;
    let mut child = Command::new("curl")
        .args(["-fL", "--progress-bar", "-o", archive_str, package.url])
        .spawn()
        .map_err(|error| DownloadError::Failed(format!("curl failed to start: {error}")))?;
    let status = loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(DownloadError::Cancelled);
        }
        match child.try_wait()? {
            Some(status) => break status,
            None => thread::sleep(Duration::from_millis(100)),
        }
    };
    if !status.success() {
        return Err(DownloadError::Failed(format!(
            "download failed (curl exit {:?}). Check network or place model under {}",
            status.code(),
            final_dir.display()
        )));
    }

    let bytes = std::fs::metadata(&archive_path)
        .map(|meta| meta.len())
        .unwrap_or(0);

    let publish = match &package.payload {
        Payload::RawFile {
            file_name,
            sha256,
            bytes: expected_bytes,
        } => {
            on_progress(progress(
                "extracting",
                "Verifying download integrity…",
                bytes,
                Some(bytes),
            ));
            // Fail closed: a corrupted or tampered download is removed with
            // the scratch cleanup, never published.
            verify_pinned_file(&archive_path, sha256, *expected_bytes)?;
            std::fs::create_dir_all(&final_dir)?;
            let target = final_dir.join(file_name);
            // Same filesystem as the scratch archive, so the rename is atomic.
            std::fs::rename(&archive_path, &target).map_err(|error| {
                DownloadError::Failed(format!("publish model atomically: {error}"))
            })?;
            if !(package.ready)(&final_dir) {
                return Err(DownloadError::Failed(
                    "model installed but validation failed".into(),
                ));
            }
            on_progress(progress(
                "done",
                &format!("{} ready", package.display),
                bytes,
                Some(bytes),
            ));
            return Ok(final_dir);
        }
        Payload::TarBz2(publish) => publish,
    };

    on_progress(progress(
        "extracting",
        "Extracting archive…",
        bytes,
        Some(bytes),
    ));

    std::fs::create_dir_all(&extract_tmp)?;

    let extract_str = extract_tmp
        .to_str()
        .ok_or_else(|| DownloadError::Failed("bad extract path".into()))?;
    let tar_status = Command::new("tar")
        .args(["-xjf", archive_str, "-C", extract_str])
        .status()
        .map_err(|error| DownloadError::Failed(format!("tar failed: {error}")))?;
    if !tar_status.success() {
        return Err(DownloadError::Failed(
            "failed to extract model archive".into(),
        ));
    }

    // sherpa archives extract into a single prefixed directory; find the one
    // that satisfies readiness rather than assuming a fixed name.
    let found = find_ready_dir(&extract_tmp, package.ready).ok_or_else(|| {
        DownloadError::Failed(format!(
            "extracted archive but could not find {} model files",
            package.display
        ))
    })?;

    let staging = extract_tmp.join("_staging");
    publish_package(&found, &final_dir, publish, &staging)?;

    if !(package.ready)(&final_dir) {
        return Err(DownloadError::Failed(
            "model installed but validation failed".into(),
        ));
    }

    on_progress(progress(
        "done",
        &format!("{} ready", package.display),
        bytes,
        Some(bytes),
    ));
    Ok(final_dir)
}

/// Resolve `models_root` + a `/`-separated `rel_dir` into a platform path.
fn package_final_dir(models_root: &Path, package: &ModelPackage) -> PathBuf {
    let mut dir = models_root.to_path_buf();
    for part in package.rel_dir.split('/') {
        dir.push(part);
    }
    dir
}

/// Move a package's files from the discovered `found` dir into `final_dir`.
///
/// `WholeDir` renames `found` into place. `SelectFiles` copies only the
/// requested files into `staging`, then renames `staging` into place — so the
/// published directory contains exactly the needed files and the final swap is
/// atomic. `staging` must live on the same filesystem as `final_dir`.
fn publish_package(
    found: &Path,
    final_dir: &Path,
    publish: &Publish,
    staging: &Path,
) -> Result<(), DownloadError> {
    if let Some(parent) = final_dir.parent() {
        std::fs::create_dir_all(parent)?;
    }
    match publish {
        Publish::WholeDir => {
            if final_dir.exists() {
                let _ = std::fs::remove_dir_all(final_dir);
            }
            std::fs::rename(found, final_dir).map_err(|error| {
                DownloadError::Failed(format!("publish model atomically: {error}"))
            })?;
        }
        Publish::SelectFiles(selectors) => {
            let _ = std::fs::remove_dir_all(staging);
            std::fs::create_dir_all(staging)?;
            for candidates in *selectors {
                let src = candidates
                    .iter()
                    .map(|name| found.join(name))
                    .find(|path| path.is_file())
                    .ok_or_else(|| {
                        DownloadError::Failed(format!(
                            "extracted archive missing a required file (one of {candidates:?})"
                        ))
                    })?;
                let file_name = src
                    .file_name()
                    .ok_or_else(|| DownloadError::Failed("bad extracted file name".into()))?;
                std::fs::copy(&src, staging.join(file_name))?;
            }
            if final_dir.exists() {
                let _ = std::fs::remove_dir_all(final_dir);
            }
            std::fs::rename(staging, final_dir).map_err(|error| {
                DownloadError::Failed(format!("publish model atomically: {error}"))
            })?;
        }
    }
    Ok(())
}

fn progress(phase: &str, message: &str, bytes: u64, total: Option<u64>) -> DownloadProgress {
    DownloadProgress {
        phase: phase.into(),
        message: message.into(),
        bytes,
        total,
    }
}

/// Verify a downloaded raw file against its pinned size + SHA256.
fn verify_pinned_file(path: &Path, sha256: &str, bytes: u64) -> Result<(), DownloadError> {
    let actual_bytes = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    if actual_bytes != bytes {
        return Err(DownloadError::Failed(format!(
            "downloaded file size mismatch: expected {bytes} bytes, got {actual_bytes}"
        )));
    }
    let digest = sha256_file(path)?;
    if !digest.eq_ignore_ascii_case(sha256) {
        return Err(DownloadError::Failed(
            "downloaded file failed integrity check (sha256 mismatch)".into(),
        ));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, DownloadError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
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

/// Depth-first search for the first directory (root or descendant) satisfying
/// `ready` — handles the single prefixed directory sherpa archives extract to.
fn find_ready_dir(root: &Path, ready: fn(&Path) -> bool) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if ready(&dir) {
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
    fn archive_urls_are_https_bz2() {
        for (url, name) in [
            (SENSEVOICE_ARCHIVE_URL, SENSEVOICE_ARCHIVE_NAME),
            (
                PARAFORMER_OFFLINE_ARCHIVE_URL,
                PARAFORMER_OFFLINE_ARCHIVE_NAME,
            ),
            (
                PARAFORMER_STREAMING_ARCHIVE_URL,
                PARAFORMER_STREAMING_ARCHIVE_NAME,
            ),
        ] {
            assert!(url.starts_with("https://"), "{url}");
            assert!(
                url.ends_with(name),
                "url should end with archive name: {url}"
            );
            assert!(name.ends_with(".tar.bz2"), "{name}");
        }
    }

    #[test]
    fn silero_vad_url_is_https_raw_onnx() {
        assert!(SILERO_VAD_URL.starts_with("https://"));
        assert!(SILERO_VAD_URL.ends_with(SILERO_VAD_FILE_NAME));
        assert!(SILERO_VAD_FILE_NAME.ends_with(".onnx"));
        assert_eq!(SILERO_VAD_SHA256.len(), 64);
        assert!(SILERO_VAD_BYTES > 0);
    }

    /// Extract the tar publish mode of a package (panics for raw-file ones).
    fn publish_of(package: &ModelPackage) -> &Publish {
        match &package.payload {
            Payload::TarBz2(publish) => publish,
            Payload::RawFile { .. } => panic!("not a tar package"),
        }
    }

    #[test]
    fn package_final_dir_splits_rel_dir() {
        let root = Path::new("/models");
        assert_eq!(
            package_final_dir(root, &PARAFORMER_OFFLINE_PACKAGE),
            root.join("paraformer").join("offline")
        );
        assert_eq!(
            package_final_dir(root, &SENSEVOICE_PACKAGE),
            root.join("sensevoice")
        );
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
    fn paraformer_offline_already_installed_short_circuits() {
        let root = temp_dir("download-pf-offline-ready");
        let dir = root.join("paraformer").join("offline");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.int8.onnx"), b"model").unwrap();
        std::fs::write(dir.join("tokens.txt"), b"tokens").unwrap();

        let cancel = AtomicBool::new(false);
        let mut phases = Vec::new();
        let installed =
            download_paraformer_offline_package(&root, &cancel, |p| phases.push(p.phase)).unwrap();

        assert_eq!(installed, dir);
        assert_eq!(phases, vec!["done".to_string()]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn paraformer_streaming_already_installed_short_circuits() {
        let root = temp_dir("download-pf-streaming-ready");
        let dir = root.join("paraformer").join("streaming");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("encoder.int8.onnx"), b"e").unwrap();
        std::fs::write(dir.join("decoder.int8.onnx"), b"d").unwrap();
        std::fs::write(dir.join("tokens.txt"), b"t").unwrap();

        let cancel = AtomicBool::new(false);
        let mut phases = Vec::new();
        let installed =
            download_paraformer_streaming_package(&root, &cancel, |p| phases.push(p.phase))
                .unwrap();

        assert_eq!(installed, dir);
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

    #[test]
    fn find_ready_dir_descends_into_archive_prefix() {
        let root = temp_dir("find-ready");
        let prefix = root.join("sherpa-onnx-paraformer-zh-2023-09-14");
        std::fs::create_dir_all(&prefix).unwrap();
        std::fs::write(prefix.join("model.int8.onnx"), b"m").unwrap();
        std::fs::write(prefix.join("tokens.txt"), b"t").unwrap();

        let found = find_ready_dir(&root, paraformer_offline_ready).unwrap();
        assert_eq!(found, prefix);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn select_files_publishes_only_needed_offline_files() {
        // Simulate the extracted prefix dir with int8 + fp32 + junk files.
        let root = temp_dir("publish-offline");
        let found = root.join("sherpa-onnx-paraformer-zh-2023-09-14");
        std::fs::create_dir_all(found.join("test_wavs")).unwrap();
        std::fs::write(found.join("model.int8.onnx"), b"int8").unwrap();
        std::fs::write(found.join("model.onnx"), b"fp32").unwrap();
        std::fs::write(found.join("tokens.txt"), b"tokens").unwrap();
        std::fs::write(found.join("README.md"), b"readme").unwrap();
        std::fs::write(found.join("test_wavs/0.wav"), b"wav").unwrap();

        let final_dir = root.join("paraformer").join("offline");
        let staging = root.join("_staging");
        publish_package(
            &found,
            &final_dir,
            publish_of(&PARAFORMER_OFFLINE_PACKAGE),
            &staging,
        )
        .unwrap();

        assert!(paraformer_offline_ready(&final_dir));
        // int8 preferred, fp32 + junk dropped.
        assert!(final_dir.join("model.int8.onnx").is_file());
        assert!(final_dir.join("tokens.txt").is_file());
        assert!(!final_dir.join("model.onnx").exists());
        assert!(!final_dir.join("README.md").exists());
        assert!(!final_dir.join("test_wavs").exists());
        assert!(!staging.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn select_files_publishes_streaming_encoder_decoder_tokens() {
        let root = temp_dir("publish-streaming");
        let found = root.join("sherpa-onnx-streaming-paraformer-bilingual-zh-en");
        std::fs::create_dir_all(&found).unwrap();
        std::fs::write(found.join("encoder.int8.onnx"), b"e8").unwrap();
        std::fs::write(found.join("encoder.onnx"), b"e").unwrap();
        std::fs::write(found.join("decoder.int8.onnx"), b"d8").unwrap();
        std::fs::write(found.join("decoder.onnx"), b"d").unwrap();
        std::fs::write(found.join("tokens.txt"), b"t").unwrap();

        let final_dir = root.join("paraformer").join("streaming");
        let staging = root.join("_staging");
        publish_package(
            &found,
            &final_dir,
            publish_of(&PARAFORMER_STREAMING_PACKAGE),
            &staging,
        )
        .unwrap();

        assert!(paraformer_streaming_ready(&final_dir));
        assert!(final_dir.join("encoder.int8.onnx").is_file());
        assert!(final_dir.join("decoder.int8.onnx").is_file());
        assert!(final_dir.join("tokens.txt").is_file());
        assert!(!final_dir.join("encoder.onnx").exists());
        assert!(!final_dir.join("decoder.onnx").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn whole_dir_publish_keeps_all_files_for_sensevoice() {
        // Proves the SenseVoice publish path is byte-for-byte the legacy
        // rename-the-whole-directory behavior.
        let root = temp_dir("publish-sensevoice");
        let found = root.join("sherpa-onnx-sense-voice");
        std::fs::create_dir_all(&found).unwrap();
        std::fs::write(found.join("model.int8.onnx"), b"m").unwrap();
        std::fs::write(found.join("tokens.txt"), b"t").unwrap();
        std::fs::write(found.join("README.md"), b"r").unwrap();

        let final_dir = root.join("sensevoice");
        let staging = root.join("_staging");
        publish_package(
            &found,
            &final_dir,
            publish_of(&SENSEVOICE_PACKAGE),
            &staging,
        )
        .unwrap();

        assert!(sensevoice_ready(&final_dir));
        assert!(final_dir.join("model.int8.onnx").is_file());
        assert!(final_dir.join("tokens.txt").is_file());
        // WholeDir keeps everything (no file selection).
        assert!(final_dir.join("README.md").is_file());
        assert!(!found.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn silero_vad_already_installed_short_circuits() {
        let root = temp_dir("download-silero-ready");
        let dir = root.join("silero-vad");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("silero_vad.onnx"), b"model").unwrap();

        let cancel = AtomicBool::new(false);
        let mut phases = Vec::new();
        let installed =
            download_silero_vad_package(&root, &cancel, |p| phases.push(p.phase)).unwrap();

        assert_eq!(installed, dir);
        assert_eq!(phases, vec!["done".to_string()]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn verify_pinned_file_accepts_matching_content() {
        let root = temp_dir("verify-ok");
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("silero_vad.onnx");
        std::fs::write(&file, b"hello").unwrap();
        let digest = sha256_file(&file).unwrap();
        verify_pinned_file(&file, &digest, 5).unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn verify_pinned_file_rejects_size_and_hash_mismatch() {
        let root = temp_dir("verify-bad");
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("silero_vad.onnx");
        std::fs::write(&file, b"hello").unwrap();

        let err = verify_pinned_file(&file, SILERO_VAD_SHA256, 6).unwrap_err();
        assert!(err.to_string().contains("size mismatch"), "{err}");

        let err = verify_pinned_file(&file, SILERO_VAD_SHA256, 5).unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
        let _ = std::fs::remove_dir_all(root);
    }

    /// Real end-to-end download (~2 MB) against the pinned URL. Ignored by
    /// default; run explicitly to validate the download path:
    /// `cargo test -p lumen-models -- --ignored silero_vad_real_download`.
    #[test]
    #[ignore]
    fn silero_vad_real_download_verifies_and_installs() {
        let root = temp_dir("silero-real-download");
        let cancel = AtomicBool::new(false);
        let installed = download_silero_vad_package(&root, &cancel, |_| {}).unwrap();
        assert!(silero_vad_ready(&installed));
        let model = installed.join(SILERO_VAD_FILE_NAME);
        let size = std::fs::metadata(&model)
            .map(|meta| meta.len())
            .unwrap_or(0);
        assert_eq!(size, SILERO_VAD_BYTES);
        assert_eq!(sha256_file(&model).unwrap(), SILERO_VAD_SHA256);
        // Idempotent: a second run short-circuits without network.
        let mut phases = Vec::new();
        download_silero_vad_package(&root, &cancel, |p| phases.push(p.phase)).unwrap();
        assert_eq!(phases, vec!["done".to_string()]);
        let _ = std::fs::remove_dir_all(root);
    }
}
